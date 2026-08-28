from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from embodied_agent.agent import RobotToolRouter
from embodied_agent.core import RobotRegistry
from embodied_agent.embodiments import (
    CrazyfliePyBullet,
    HumanoidMuJoCo,
    MicroduckMuJoCo,
    XLeRobotMuJoCo,
)

from .agent_model import AgentModelEvalResult, ExpectedActionModel, evaluate_agent_model
from .four_embodiment import default_four_embodiment_cases


def four_embodiment_physics_case():
    return next(
        case
        for case in default_four_embodiment_cases()
        if case.name == "four-all-embodiments-mission"
    )


def build_four_embodiment_physics_stack(
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
    microduck_runtime_root: str | Path,
    microduck_policy_dir: str | Path,
) -> tuple[RobotRegistry, RobotToolRouter]:
    policy_dir = Path(microduck_policy_dir)
    registry = RobotRegistry()
    registry.register(
        XLeRobotMuJoCo(
            name="xlerobot",
            runtime_root=xlerobot_runtime_root,
            position_tolerance_m=0.05,
            yaw_tolerance_rad=0.06,
        )
    )
    registry.register(
        CrazyfliePyBullet(
            name="crazyflie",
            gui=False,
            seed=0,
            ctrl_freq_hz=60,
            initial_position=(0.0, 0.0, 0.1),
            position_tolerance_m=0.06,
            slow_radius_m=0.25,
        )
    )
    registry.register(
        HumanoidMuJoCo(
            name="humanoid",
            runtime_root=humanoid_runtime_root,
            control_hz=100.0,
            fixed_base=True,
        )
    )
    registry.register(
        MicroduckMuJoCo(
            name="microduck",
            runtime_root=microduck_runtime_root,
            walking_policy_path=policy_dir / "BEST_alpha_walking.onnx",
            standing_policy_path=policy_dir / "BEST_alpha_stand.onnx",
            roll_policy_path=policy_dir / "roulade.onnx",
        )
    )

    router = RobotToolRouter(
        registry,
        {
            "xlerobot": ["navigate_to"],
            "crazyflie": ["takeoff", "goto", "land"],
            "humanoid": ["stand"],
            "microduck": ["stand", "walk_velocity", "roll"],
        },
    )
    return registry, router


async def run_expected_four_embodiment_physics(
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
    microduck_runtime_root: str | Path,
    microduck_policy_dir: str | Path,
    max_steps: int = 8,
) -> AgentModelEvalResult:
    """Run the all-four mission against four simultaneously connected real simulators."""
    registry, router = build_four_embodiment_physics_stack(
        xlerobot_runtime_root=xlerobot_runtime_root,
        humanoid_runtime_root=humanoid_runtime_root,
        microduck_runtime_root=microduck_runtime_root,
        microduck_policy_dir=microduck_policy_dir,
    )
    case = four_embodiment_physics_case()
    connected = []
    try:
        for robot in registry:
            await robot.connect()
            connected.append(robot)

        return await evaluate_agent_model(
            lambda selected_case: ExpectedActionModel(selected_case),
            cases=(case,),
            max_steps=max_steps,
            stack=(registry, router),
        )
    finally:
        for robot in reversed(connected):
            await robot.disconnect()


def main() -> None:
    xlerobot_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT", "")
    humanoid_root = os.environ.get("LEROBOT_HUMANOID_RUNTIME_ROOT", "")
    microduck_root = os.environ.get("MICRODUCK_RL_ROOT", "")
    microduck_policy_dir = os.environ.get("MICRODUCK_POLICY_DIR", "")
    missing = [
        name
        for name, value in (
            ("XLEROBOT_UPSTREAM_ROOT", xlerobot_root),
            ("LEROBOT_HUMANOID_RUNTIME_ROOT", humanoid_root),
            ("MICRODUCK_RL_ROOT", microduck_root),
            ("MICRODUCK_POLICY_DIR", microduck_policy_dir),
        )
        if not value
    ]
    if missing:
        raise SystemExit("Set required runtime inputs: " + ", ".join(missing))

    result = asyncio.run(
        run_expected_four_embodiment_physics(
            xlerobot_runtime_root=xlerobot_root,
            humanoid_runtime_root=humanoid_root,
            microduck_runtime_root=microduck_root,
            microduck_policy_dir=microduck_policy_dir,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
