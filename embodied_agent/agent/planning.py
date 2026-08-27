from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodied_agent.core import Capability

from .tools import RobotToolRouter, ToolCallResult


@dataclass(frozen=True, slots=True)
class TaskStep:
    skill: str
    capability: Capability
    arguments: dict[str, Any] = field(default_factory=dict)
    preferred_robot: str | None = None
    label: str = ""


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    steps: tuple[TaskStep, ...]


@dataclass(frozen=True, slots=True)
class PlanStep:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True, slots=True)
class Plan:
    task_name: str
    steps: tuple[PlanStep, ...]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    task_name: str
    ok: bool
    calls: tuple[ToolCallResult, ...]


class CapabilityPlanner:
    """Deterministic baseline planner that binds task steps to available robot tools.

    This is intentionally not an LLM planner. It gives us a stable orchestration
    baseline and eval target before a language model is allowed to produce plans.
    """

    def __init__(self, tools: RobotToolRouter) -> None:
        self.tools = tools

    def plan(self, task: Task) -> Plan:
        bound: list[PlanStep] = []
        for step in task.steps:
            candidates = [
                tool
                for tool in self.tools.tools_for_skill(step.skill)
                if tool["capability"] == step.capability.value
            ]
            if step.preferred_robot is not None:
                preferred = [tool for tool in candidates if tool["robot"] == step.preferred_robot]
                if preferred:
                    candidates = preferred
            if not candidates:
                raise ValueError(
                    f"no available tool for skill={step.skill} capability={step.capability.value}"
                )
            selected = sorted(candidates, key=lambda item: item["name"])[0]
            bound.append(
                PlanStep(
                    tool=selected["name"],
                    arguments=dict(step.arguments),
                    label=step.label,
                )
            )
        return Plan(task_name=task.name, steps=tuple(bound))


class PlanExecutor:
    def __init__(self, tools: RobotToolRouter, *, stop_on_failure: bool = True) -> None:
        self.tools = tools
        self.stop_on_failure = stop_on_failure

    async def execute(self, plan: Plan) -> ExecutionResult:
        calls: list[ToolCallResult] = []
        for step in plan.steps:
            result = await self.tools.call(step.tool, step.arguments)
            calls.append(result)
            if self.stop_on_failure and not result.ok:
                break
        return ExecutionResult(
            task_name=plan.task_name,
            ok=len(calls) == len(plan.steps) and all(call.ok for call in calls),
            calls=tuple(calls),
        )
