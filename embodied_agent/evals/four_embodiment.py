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


def default_four_embodiment_cases() -> tuple[EvalCase, ...]:
    """Selection cases spanning all current simulated embodiments.

    The humanoid and Microduck deliberately overlap on STAND/WALK. Those cases
    ensure evaluation rewards choosing the requested embodiment, not merely any
    robot capable of executing a semantically compatible skill.
    """
    return (
        EvalCase(
            name="four-aerial-scout",
            instruction=(
                "Use the aerial robot to take off to 1 meter, fly to "
                "(1.5, -0.5, 1.0), then land."
            ),
            task=Task(
                "four-aerial-scout",
                (
                    TaskStep("takeoff", Capability.FLY, {"altitude_m": 1.0}),
                    TaskStep(
                        "goto",
                        Capability.FLY,
                        {"x_m": 1.5, "y_m": -0.5, "z_m": 1.0},
                    ),
                    TaskStep("land", Capability.FLY, {}),
                ),
            ),
            expected_tools=(
                "crazyflie.takeoff",
                "crazyflie.goto",
                "crazyflie.land",
            ),
        ),
        EvalCase(
            name="four-ground-approach",
            instruction="Move XLeRobot to (1.25, 0.75) facing forward.",
            task=Task(
                "four-ground-approach",
                (
                    TaskStep(
                        "navigate_to",
                        Capability.NAVIGATE,
                        {"x_m": 1.25, "y_m": 0.75, "yaw_rad": 0.0},
                    ),
                ),
            ),
            expected_tools=("xlerobot.navigate_to",),
        ),
        EvalCase(
            name="four-humanoid-stand",
            instruction="Have the humanoid stand ready.",
            task=Task(
                "four-humanoid-stand",
                (TaskStep("stand", Capability.STAND, {}),),
            ),
            expected_tools=("humanoid.stand",),
        ),
        EvalCase(
            name="four-humanoid-walk",
            instruction=(
                "Have the humanoid walk forward at 0.10 meters per second for "
                "1.0 second without turning."
            ),
            task=Task(
                "four-humanoid-walk",
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
                    ),
                ),
            ),
            expected_tools=("humanoid.walk_velocity",),
        ),
        EvalCase(
            name="four-microduck-kick-right",
            instruction="Have Microduck kick with its right foot.",
            task=Task(
                "four-microduck-kick-right",
                (TaskStep("kick", Capability.KICK, {"foot": "right"}),),
            ),
            expected_tools=("microduck.kick",),
        ),
        EvalCase(
            name="four-microduck-roll",
            instruction="Have Microduck perform its learned roll trick.",
            task=Task(
                "four-microduck-roll",
                (TaskStep("roll", Capability.ROLL, {}),),
            ),
            expected_tools=("microduck.roll",),
        ),
        EvalCase(
            name="four-all-embodiments-mission",
            instruction=(
                "Coordinate every embodiment: launch Crazyflie to 1 meter and fly it "
                "to (1.0, 0.5, 1.0); move XLeRobot to (1.0, 0.5); have the humanoid "
                "stand ready; have Microduck perform its learned roll; then land Crazyflie."
            ),
            task=Task(
                "four-all-embodiments-mission",
                (
                    TaskStep("takeoff", Capability.FLY, {"altitude_m": 1.0}),
                    TaskStep(
                        "goto",
                        Capability.FLY,
                        {"x_m": 1.0, "y_m": 0.5, "z_m": 1.0},
                    ),
                    TaskStep(
                        "navigate_to",
                        Capability.NAVIGATE,
                        {"x_m": 1.0, "y_m": 0.5, "yaw_rad": 0.0},
                    ),
                    TaskStep("stand", Capability.STAND, {}),
                    TaskStep("roll", Capability.ROLL, {}),
                    TaskStep("land", Capability.FLY, {}),
                ),
            ),
            expected_tools=(
                "crazyflie.takeoff",
                "crazyflie.goto",
                "xlerobot.navigate_to",
                "humanoid.stand",
                "microduck.roll",
                "crazyflie.land",
            ),
        ),
    )


class _ScriptedEmbodiment(Embodiment):
    def __init__(
        self,
        name: str,
        capabilities: set[Capability],
        skills: set[str],
    ) -> None:
        self.name = name
        self.backend = "scripted-four-embodiment-eval"
        self._capabilities = frozenset(capabilities)
        self._skills = frozenset(skills)
        self.connected = False
        self.calls: list[SkillRequest] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def observe(self) -> Observation:
        if not self.connected:
            raise RuntimeError(f"{self.name} is not connected")
        return Observation(
            self.name,
            {"backend": self.backend, "connected": self.connected},
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if not self.connected:
            raise RuntimeError(f"{self.name} is not connected")
        self.calls.append(request)
        if request.name not in self._skills:
            return SkillResult(
                self.name,
                request.name,
                False,
                f"unsupported scripted skill: {request.name}",
            )
        return SkillResult(
            self.name,
            request.name,
            True,
            "scripted four-embodiment skill executed",
            {"params": dict(request.params)},
        )


def scripted_four_embodiment_stack() -> tuple[RobotRegistry, RobotToolRouter]:
    registry = RobotRegistry()
    robots = (
        _ScriptedEmbodiment(
            "crazyflie",
            {Capability.OBSERVE, Capability.FLY},
            {"takeoff", "goto", "land"},
        ),
        _ScriptedEmbodiment(
            "xlerobot",
            {Capability.OBSERVE, Capability.NAVIGATE},
            {"reset", "drive_velocity", "navigate_to"},
        ),
        _ScriptedEmbodiment(
            "humanoid",
            {Capability.OBSERVE, Capability.STAND, Capability.WALK},
            {"reset", "stand", "walk_velocity"},
        ),
        _ScriptedEmbodiment(
            "microduck",
            {
                Capability.OBSERVE,
                Capability.STAND,
                Capability.WALK,
                Capability.KICK,
                Capability.ROLL,
            },
            {"reset", "stand", "walk_velocity", "kick", "roll"},
        ),
    )
    for robot in robots:
        registry.register(robot)

    router = RobotToolRouter(
        registry,
        {
            "crazyflie": ["observe", "takeoff", "goto", "land"],
            "xlerobot": ["observe", "reset", "drive_velocity", "navigate_to"],
            "humanoid": ["observe", "reset", "stand", "walk_velocity"],
            "microduck": ["observe", "reset", "stand", "walk_velocity", "kick", "roll"],
        },
    )
    return registry, router


async def evaluate_expected_four_embodiment_agent(
    *,
    max_steps: int = 10,
) -> AgentModelEvalResult:
    cases = default_four_embodiment_cases()
    return await evaluate_agent_model(
        lambda case: ExpectedActionModel(case),
        cases=cases,
        max_steps=max_steps,
        stack=scripted_four_embodiment_stack(),
    )


def main() -> None:
    result = asyncio.run(evaluate_expected_four_embodiment_agent())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
