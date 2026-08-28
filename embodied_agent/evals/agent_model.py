from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mcp import Client

from embodied_agent.mcp import AgentDecision, AgentModel, MCPAgentRunner, build_mcp_server
from embodied_agent.world import WorldState

from .multi_robot import EvalCase, _scripted_stack, default_multi_robot_cases


@dataclass(frozen=True, slots=True)
class ExpectedAgentAction:
    tool: str
    arguments: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class AgentModelCaseResult:
    name: str
    instruction: str
    expected_actions: tuple[ExpectedAgentAction, ...]
    actual_tools: tuple[str, ...]
    tool_selection_accuracy: float
    argument_accuracy: float
    sequence_exact_match: bool
    arguments_exact_match: bool
    tool_execution_success_rate: float
    runner_finished: bool
    runner_ok: bool
    strict_task_success: bool
    action_count: int
    expected_action_count: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "expected_actions": [action.to_dict() for action in self.expected_actions],
            "actual_tools": list(self.actual_tools),
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "argument_accuracy": self.argument_accuracy,
            "sequence_exact_match": self.sequence_exact_match,
            "arguments_exact_match": self.arguments_exact_match,
            "tool_execution_success_rate": self.tool_execution_success_rate,
            "runner_finished": self.runner_finished,
            "runner_ok": self.runner_ok,
            "strict_task_success": self.strict_task_success,
            "action_count": self.action_count,
            "expected_action_count": self.expected_action_count,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class AgentModelEvalResult:
    cases: tuple[AgentModelCaseResult, ...]
    tool_selection_accuracy: float
    argument_accuracy: float
    sequence_exact_match_rate: float
    arguments_exact_match_rate: float
    tool_execution_success_rate: float
    runner_finish_rate: float
    runner_ok_rate: float
    strict_task_success_rate: float
    mean_action_efficiency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": len(self.cases),
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "argument_accuracy": self.argument_accuracy,
            "sequence_exact_match_rate": self.sequence_exact_match_rate,
            "arguments_exact_match_rate": self.arguments_exact_match_rate,
            "tool_execution_success_rate": self.tool_execution_success_rate,
            "runner_finish_rate": self.runner_finish_rate,
            "runner_ok_rate": self.runner_ok_rate,
            "strict_task_success_rate": self.strict_task_success_rate,
            "mean_action_efficiency": self.mean_action_efficiency,
            "cases": [case.to_dict() for case in self.cases],
        }


AgentModelFactory = Callable[[EvalCase], AgentModel]


def _expected_actions(case: EvalCase) -> tuple[ExpectedAgentAction, ...]:
    if len(case.task.steps) != len(case.expected_tools):
        raise ValueError(f"eval case {case.name} has mismatched task/tool lengths")

    world = WorldState()
    for entity in case.entities:
        world.upsert_entity(entity)

    return tuple(
        ExpectedAgentAction(
            tool=tool,
            arguments=world.resolve_arguments(step.arguments),
        )
        for step, tool in zip(case.task.steps, case.expected_tools, strict=True)
    )


def _value_matches(expected: Any, actual: Any, *, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _value_matches(value, actual[key], tolerance=tolerance)
            for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return False
        return all(
            _value_matches(left, right, tolerance=tolerance)
            for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def _arguments_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    tolerance: float,
) -> bool:
    """Match expected semantic arguments while allowing safe optional defaults/extras."""
    return all(
        key in actual and _value_matches(value, actual[key], tolerance=tolerance)
        for key, value in expected.items()
    )


def _selection_matches(expected_tools: tuple[str, ...], actual_tools: tuple[str, ...]) -> int:
    return sum(
        1
        for index, expected in enumerate(expected_tools)
        if index < len(actual_tools) and actual_tools[index] == expected
    )


async def evaluate_agent_model(
    model_factory: AgentModelFactory,
    *,
    cases: tuple[EvalCase, ...] | None = None,
    max_steps: int = 8,
    argument_tolerance: float = 1e-6,
) -> AgentModelEvalResult:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if argument_tolerance < 0:
        raise ValueError("argument_tolerance must be >= 0")

    selected_cases = cases or default_multi_robot_cases()
    results: list[AgentModelCaseResult] = []

    total_expected_actions = 0
    total_selection_matches = 0
    total_argument_matches = 0
    total_executed_actions = 0
    total_successful_actions = 0
    efficiency_sum = 0.0

    for case in selected_cases:
        expected = _expected_actions(case)
        expected_tools = tuple(action.tool for action in expected)
        world = WorldState()
        for entity in case.entities:
            world.upsert_entity(entity)

        registry, router = _scripted_stack()
        server = build_mcp_server(router, registry=registry, world=world)
        model = model_factory(case)

        async with Client(server) as client:
            run = await MCPAgentRunner(
                client,
                model,
                max_steps=max_steps,
            ).run(case.instruction)

        actual_tools = tuple(action.tool for action in run.actions)
        selection_matches = _selection_matches(expected_tools, actual_tools)
        total_expected_actions += len(expected)
        total_selection_matches += selection_matches

        argument_matches = 0
        for index, expected_action in enumerate(expected):
            if index >= len(run.actions):
                continue
            actual_action = run.actions[index]
            if actual_action.tool != expected_action.tool:
                continue
            if _arguments_match(
                expected_action.arguments,
                actual_action.arguments,
                tolerance=argument_tolerance,
            ):
                argument_matches += 1
        total_argument_matches += argument_matches

        sequence_exact = actual_tools == expected_tools
        arguments_exact = (
            len(run.actions) == len(expected)
            and argument_matches == len(expected)
        )
        successful_actions = sum(1 for action in run.actions if action.ok)
        total_executed_actions += len(run.actions)
        total_successful_actions += successful_actions
        strict_success = (
            run.ok
            and run.finished
            and sequence_exact
            and arguments_exact
            and successful_actions == len(expected)
        )
        efficiency = (
            min(1.0, len(expected) / len(run.actions))
            if run.actions
            else (1.0 if not expected else 0.0)
        )
        efficiency_sum += efficiency

        results.append(
            AgentModelCaseResult(
                name=case.name,
                instruction=case.instruction,
                expected_actions=expected,
                actual_tools=actual_tools,
                tool_selection_accuracy=(
                    selection_matches / len(expected) if expected else 1.0
                ),
                argument_accuracy=(argument_matches / len(expected) if expected else 1.0),
                sequence_exact_match=sequence_exact,
                arguments_exact_match=arguments_exact,
                tool_execution_success_rate=(
                    successful_actions / len(run.actions) if run.actions else 1.0
                ),
                runner_finished=run.finished,
                runner_ok=run.ok,
                strict_task_success=strict_success,
                action_count=len(run.actions),
                expected_action_count=len(expected),
                error=run.error,
            )
        )

    case_count = len(results)
    return AgentModelEvalResult(
        cases=tuple(results),
        tool_selection_accuracy=(
            total_selection_matches / total_expected_actions if total_expected_actions else 1.0
        ),
        argument_accuracy=(
            total_argument_matches / total_expected_actions if total_expected_actions else 1.0
        ),
        sequence_exact_match_rate=(
            sum(result.sequence_exact_match for result in results) / case_count
            if case_count
            else 1.0
        ),
        arguments_exact_match_rate=(
            sum(result.arguments_exact_match for result in results) / case_count
            if case_count
            else 1.0
        ),
        tool_execution_success_rate=(
            total_successful_actions / total_executed_actions if total_executed_actions else 1.0
        ),
        runner_finish_rate=(
            sum(result.runner_finished for result in results) / case_count if case_count else 1.0
        ),
        runner_ok_rate=(
            sum(result.runner_ok for result in results) / case_count if case_count else 1.0
        ),
        strict_task_success_rate=(
            sum(result.strict_task_success for result in results) / case_count
            if case_count
            else 1.0
        ),
        mean_action_efficiency=(efficiency_sum / case_count if case_count else 1.0),
    )


class ExpectedActionModel:
    """Oracle model used only to validate the AgentModel evaluation harness."""

    def __init__(self, case: EvalCase) -> None:
        self.expected = _expected_actions(case)

    async def decide(self, context: Any) -> AgentDecision:
        index = len(context.history)
        if index >= len(self.expected):
            return AgentDecision.finish("reference sequence complete")
        action = self.expected[index]
        return AgentDecision.call(action.tool, action.arguments)


async def run_expected_action_baseline() -> AgentModelEvalResult:
    return await evaluate_agent_model(lambda case: ExpectedActionModel(case))


def main() -> None:
    result = asyncio.run(run_expected_action_baseline())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
