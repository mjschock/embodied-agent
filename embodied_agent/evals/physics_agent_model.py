from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .agent_model import (
    AgentModelEvalResult,
    AgentModelFactory,
    ExpectedActionModel,
    evaluate_agent_model,
)
from .multi_robot import EvalSuiteResult, default_multi_robot_cases
from .physics_multi_robot import build_physics_stack, run_physics_baseline


@dataclass(frozen=True, slots=True)
class PhysicsAgentComparisonResult:
    deterministic: EvalSuiteResult
    agent: AgentModelEvalResult

    @property
    def strict_task_success_gap(self) -> float:
        return self.agent.strict_task_success_rate - self.deterministic.task_completion_rate

    @property
    def tool_execution_success_gap(self) -> float:
        return self.agent.tool_execution_success_rate - self.deterministic.tool_call_success_rate

    def to_dict(self) -> dict:
        return {
            "deterministic": self.deterministic.to_dict(),
            "agent": self.agent.to_dict(),
            "comparison": {
                "strict_task_success_gap": self.strict_task_success_gap,
                "tool_execution_success_gap": self.tool_execution_success_gap,
            },
        }


async def run_physics_agent_eval(
    model_factory: AgentModelFactory,
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
    max_steps: int = 8,
    argument_tolerance: float = 1e-6,
) -> AgentModelEvalResult:
    """Run an AgentModel over the standard A→B→C real-simulator sequence.

    The same connected XLeRobot, Crazyflie, and humanoid simulator instances are
    reused across all three cases so embodiment state carries between waypoints in
    exactly the same way as the deterministic physics baseline.
    """
    registry, router = build_physics_stack(
        xlerobot_runtime_root=xlerobot_runtime_root,
        humanoid_runtime_root=humanoid_runtime_root,
    )
    connected = []
    try:
        for robot in registry:
            await robot.connect()
            connected.append(robot)

        return await evaluate_agent_model(
            model_factory,
            cases=default_multi_robot_cases(),
            max_steps=max_steps,
            argument_tolerance=argument_tolerance,
            stack=(registry, router),
        )
    finally:
        for robot in reversed(connected):
            await robot.disconnect()


async def compare_physics_agent_to_deterministic(
    model_factory: AgentModelFactory,
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
    max_steps: int = 8,
    argument_tolerance: float = 1e-6,
) -> PhysicsAgentComparisonResult:
    """Compare deterministic orchestration with an AgentModel on fresh physics stacks."""
    deterministic = await run_physics_baseline(
        xlerobot_runtime_root=xlerobot_runtime_root,
        humanoid_runtime_root=humanoid_runtime_root,
    )
    agent = await run_physics_agent_eval(
        model_factory,
        xlerobot_runtime_root=xlerobot_runtime_root,
        humanoid_runtime_root=humanoid_runtime_root,
        max_steps=max_steps,
        argument_tolerance=argument_tolerance,
    )
    return PhysicsAgentComparisonResult(deterministic=deterministic, agent=agent)


async def run_expected_action_physics_comparison(
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
) -> PhysicsAgentComparisonResult:
    """Oracle comparison used to integration-test the physics AgentModel path."""
    return await compare_physics_agent_to_deterministic(
        lambda case: ExpectedActionModel(case),
        xlerobot_runtime_root=xlerobot_runtime_root,
        humanoid_runtime_root=humanoid_runtime_root,
    )


def main() -> None:
    xlerobot_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
    humanoid_root = os.environ.get("LEROBOT_HUMANOID_RUNTIME_ROOT")
    if not xlerobot_root or not humanoid_root:
        raise SystemExit(
            "Set XLEROBOT_UPSTREAM_ROOT and LEROBOT_HUMANOID_RUNTIME_ROOT to "
            "the pinned upstream checkouts before running the physics AgentModel benchmark."
        )
    result = asyncio.run(
        run_expected_action_physics_comparison(
            xlerobot_runtime_root=xlerobot_root,
            humanoid_runtime_root=humanoid_root,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
