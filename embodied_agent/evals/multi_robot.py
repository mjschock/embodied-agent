from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from embodied_agent.agent import CapabilityPlanner, PlanExecutor, RobotToolRouter, Task, TaskStep
from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    RobotRegistry,
    SkillRequest,
    SkillResult,
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    name: str
    instruction: str
    task: Task
    expected_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    name: str
    expected_tools: tuple[str, ...]
    planned_tools: tuple[str, ...]
    selection_accuracy: float
    plan_exact_match: bool
    execution_ok: bool
    calls_attempted: int
    calls_succeeded: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_tools": list(self.expected_tools),
            "planned_tools": list(self.planned_tools),
            "selection_accuracy": self.selection_accuracy,
            "plan_exact_match": self.plan_exact_match,
            "execution_ok": self.execution_ok,
            "calls_attempted": self.calls_attempted,
            "calls_succeeded": self.calls_succeeded,
        }


@dataclass(frozen=True, slots=True)
class EvalSuiteResult:
    cases: tuple[EvalCaseResult, ...]
    robot_selection_accuracy: float
    plan_exact_match_rate: float
    task_completion_rate: float
    tool_call_success_rate: float
    executed_step_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": len(self.cases),
            "robot_selection_accuracy": self.robot_selection_accuracy,
            "plan_exact_match_rate": self.plan_exact_match_rate,
            "task_completion_rate": self.task_completion_rate,
            "tool_call_success_rate": self.tool_call_success_rate,
            "executed_step_coverage": self.executed_step_coverage,
            "cases": [case.to_dict() for case in self.cases],
        }


def default_multi_robot_cases() -> tuple[EvalCase, ...]:
    waypoints = (
        ("workbench-a", 1.5, 0.5),
        ("workbench-b", -1.0, 1.25),
        ("workbench-c", 0.75, -1.5),
    )
    cases: list[EvalCase] = []
    expected = (
        "crazyflie.takeoff",
        "crazyflie.goto",
        "xlerobot.navigate_to",
        "humanoid.stand",
        "crazyflie.land",
    )
    for label, x_m, y_m in waypoints:
        task = Task(
            name=f"scout-approach-{label}",
            steps=(
                TaskStep("takeoff", Capability.FLY, {"altitude_m": 1.0}, label="launch scout"),
                TaskStep(
                    "goto",
                    Capability.FLY,
                    {"x_m": x_m, "y_m": y_m, "z_m": 1.0},
                    label="scout waypoint",
                ),
                TaskStep(
                    "navigate_to",
                    Capability.NAVIGATE,
                    {"x_m": x_m, "y_m": y_m},
                    label="move ground robot",
                ),
                TaskStep("stand", Capability.STAND, {}, label="ready humanoid"),
                TaskStep("land", Capability.FLY, {}, label="recover scout"),
            ),
        )
        cases.append(
            EvalCase(
                name=task.name,
                instruction=(
                    f"Scout {label} at ({x_m:.2f}, {y_m:.2f}) with the aerial robot, "
                    "move the ground robot to the same waypoint, have the humanoid stand ready, "
                    "then land the aerial robot."
                ),
                task=task,
                expected_tools=expected,
            )
        )
    return tuple(cases)


async def evaluate_cases(
    planner: CapabilityPlanner,
    executor: PlanExecutor,
    cases: tuple[EvalCase, ...],
) -> EvalSuiteResult:
    results: list[EvalCaseResult] = []
    total_expected_steps = 0
    total_correct_selection = 0
    total_calls = 0
    total_successful_calls = 0

    for case in cases:
        total_expected_steps += len(case.expected_tools)
        try:
            plan = planner.plan(case.task)
            planned_tools = tuple(step.tool for step in plan.steps)
            matches = sum(
                1 for expected, actual in zip(case.expected_tools, planned_tools) if expected == actual
            )
            total_correct_selection += matches
            selection_accuracy = matches / len(case.expected_tools) if case.expected_tools else 1.0
            plan_exact_match = planned_tools == case.expected_tools
            execution = await executor.execute(plan)
            calls_attempted = len(execution.calls)
            calls_succeeded = sum(1 for call in execution.calls if call.ok)
            execution_ok = execution.ok
        except (KeyError, RuntimeError, ValueError):
            planned_tools = ()
            selection_accuracy = 0.0
            plan_exact_match = False
            calls_attempted = 0
            calls_succeeded = 0
            execution_ok = False

        total_calls += calls_attempted
        total_successful_calls += calls_succeeded
        results.append(
            EvalCaseResult(
                name=case.name,
                expected_tools=case.expected_tools,
                planned_tools=planned_tools,
                selection_accuracy=selection_accuracy,
                plan_exact_match=plan_exact_match,
                execution_ok=execution_ok,
                calls_attempted=calls_attempted,
                calls_succeeded=calls_succeeded,
            )
        )

    case_count = len(results)
    return EvalSuiteResult(
        cases=tuple(results),
        robot_selection_accuracy=(
            total_correct_selection / total_expected_steps if total_expected_steps else 1.0
        ),
        plan_exact_match_rate=(
            sum(1 for result in results if result.plan_exact_match) / case_count if case_count else 1.0
        ),
        task_completion_rate=(
            sum(1 for result in results if result.execution_ok) / case_count if case_count else 1.0
        ),
        tool_call_success_rate=(
            total_successful_calls / total_calls if total_calls else 1.0
        ),
        executed_step_coverage=(total_calls / total_expected_steps if total_expected_steps else 1.0),
    )


class _ScriptedRobot(Embodiment):
    def __init__(
        self,
        name: str,
        capabilities: set[Capability],
        *,
        fail_skill: str | None = None,
    ) -> None:
        self.name = name
        self.backend = "scripted-eval"
        self._capabilities = frozenset(capabilities)
        self.fail_skill = fail_skill
        self.connected = False
        self.state: dict[str, Any] = {}

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def observe(self) -> Observation:
        return Observation(self.name, dict(self.state))

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if request.name == self.fail_skill:
            return SkillResult(self.name, request.name, False, "scripted failure")

        if request.name == "takeoff":
            self.state["altitude_m"] = request.params["altitude_m"]
        elif request.name == "goto":
            self.state["position"] = request.params["position"]
        elif request.name == "land":
            self.state["altitude_m"] = request.params["height_m"]
        elif request.name == "navigate_to":
            self.state["pose"] = (
                request.params["x_m"],
                request.params["y_m"],
                request.params["yaw_rad"],
            )
        elif request.name == "stand":
            self.state["posture"] = "standing"
        return SkillResult(self.name, request.name, True, "scripted success", dict(self.state))


def _scripted_stack(
    failures: dict[str, str] | None = None,
) -> tuple[RobotRegistry, RobotToolRouter]:
    failures = failures or {}
    registry = RobotRegistry()
    registry.register(
        _ScriptedRobot(
            "xlerobot",
            {Capability.OBSERVE, Capability.NAVIGATE},
            fail_skill=failures.get("xlerobot"),
        )
    )
    registry.register(
        _ScriptedRobot(
            "crazyflie",
            {Capability.OBSERVE, Capability.FLY},
            fail_skill=failures.get("crazyflie"),
        )
    )
    registry.register(
        _ScriptedRobot(
            "humanoid",
            {Capability.OBSERVE, Capability.STAND},
            fail_skill=failures.get("humanoid"),
        )
    )
    router = RobotToolRouter(
        registry,
        {
            "xlerobot": ["navigate_to"],
            "crazyflie": ["takeoff", "goto", "land"],
            "humanoid": ["stand"],
        },
    )
    return registry, router


async def run_scripted_baseline(
    *,
    failures: dict[str, str] | None = None,
) -> EvalSuiteResult:
    registry, router = _scripted_stack(failures)
    for robot in registry:
        await robot.connect()
    try:
        planner = CapabilityPlanner(router)
        executor = PlanExecutor(router)
        return await evaluate_cases(planner, executor, default_multi_robot_cases())
    finally:
        for robot in registry:
            await robot.disconnect()


def main() -> None:
    result = asyncio.run(run_scripted_baseline())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
