from __future__ import annotations

import asyncio
import json
from typing import Any

from embodied_agent.agent import RobotToolRouter, Task, TaskStep
from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    RobotRegistry,
    SkillRequest,
    SkillResult,
)

from .agent_model import AgentModelEvalResult, ExpectedActionModel, evaluate_agent_model
from .multi_robot import EvalCase


def default_microduck_skill_cases() -> tuple[EvalCase, ...]:
    """Reference cases for semantic selection over Microduck learned behaviors."""
    return (
        EvalCase(
            name="microduck-stand-ready",
            instruction="Have Microduck stand ready.",
            task=Task(
                "microduck-stand-ready",
                (TaskStep("stand", Capability.STAND, {}, label="stand ready"),),
            ),
            expected_tools=("microduck.stand",),
        ),
        EvalCase(
            name="microduck-walk-forward",
            instruction=(
                "Have Microduck walk forward at 0.10 meters per second for 1.0 second, "
                "without turning."
            ),
            task=Task(
                "microduck-walk-forward",
                (
                    TaskStep(
                        "walk_velocity",
                        Capability.WALK,
                        {
                            "lin_x_mps": 0.10,
                            "lin_y_mps": 0.0,
                            "yaw_rate_rps": 0.0,
                            "duration_s": 1.0,
                        },
                        label="walk forward",
                    ),
                ),
            ),
            expected_tools=("microduck.walk_velocity",),
        ),
        EvalCase(
            name="microduck-kick-left",
            instruction="Have Microduck kick with its left foot.",
            task=Task(
                "microduck-kick-left",
                (TaskStep("kick", Capability.KICK, {"foot": "left"}, label="left kick"),),
            ),
            expected_tools=("microduck.kick",),
        ),
        EvalCase(
            name="microduck-kick-right",
            instruction="Have Microduck kick with its right foot.",
            task=Task(
                "microduck-kick-right",
                (TaskStep("kick", Capability.KICK, {"foot": "right"}, label="right kick"),),
            ),
            expected_tools=("microduck.kick",),
        ),
        EvalCase(
            name="microduck-roll",
            instruction="Have Microduck perform its learned roll trick.",
            task=Task(
                "microduck-roll",
                (TaskStep("roll", Capability.ROLL, {}, label="roll trick"),),
            ),
            expected_tools=("microduck.roll",),
        ),
    )


class _ScriptedMicroduck(Embodiment):
    """Dependency-free execution double for isolating agent/eval behavior."""

    name = "microduck"
    backend = "scripted-microduck-eval"

    def __init__(self) -> None:
        self.connected = False
        self.state: dict[str, Any] = {
            "position_m": (0.0, 0.0, 0.12),
            "projected_gravity": (0.0, 0.0, -1.0),
            "joint_position_rad": tuple(0.0 for _ in range(14)),
            "joint_velocity_rps": tuple(0.0 for _ in range(14)),
            "policy": "standing",
            "behavior": None,
            "control_hz": 50.0,
        }

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.OBSERVE,
                Capability.STAND,
                Capability.WALK,
                Capability.KICK,
                Capability.ROLL,
            }
        )

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def observe(self) -> Observation:
        if not self.connected:
            raise RuntimeError("microduck is not connected")
        return Observation(self.name, dict(self.state))

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if not self.connected:
            raise RuntimeError("microduck is not connected")
        if request.name not in {"stand", "walk_velocity", "kick", "roll", "reset"}:
            return SkillResult(self.name, request.name, False, "unsupported scripted skill")
        data = dict(self.state)
        data["params"] = dict(request.params)
        return SkillResult(self.name, request.name, True, "scripted Microduck skill executed", data)


def scripted_microduck_stack() -> tuple[RobotRegistry, RobotToolRouter]:
    registry = RobotRegistry()
    registry.register(_ScriptedMicroduck())
    router = RobotToolRouter(
        registry,
        {
            "microduck": [
                "observe",
                "reset",
                "stand",
                "walk_velocity",
                "kick",
                "roll",
            ]
        },
    )
    return registry, router


async def evaluate_expected_microduck_agent(
    *,
    stack: tuple[RobotRegistry, RobotToolRouter] | None = None,
    max_steps: int = 3,
) -> AgentModelEvalResult:
    cases = default_microduck_skill_cases()
    return await evaluate_agent_model(
        lambda case: ExpectedActionModel(case),
        cases=cases,
        max_steps=max_steps,
        stack=stack or scripted_microduck_stack(),
    )


def main() -> None:
    result = asyncio.run(evaluate_expected_microduck_agent())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
