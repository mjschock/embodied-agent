from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from embodied_agent.agent import CapabilityPlanner, PlanExecutor, RobotToolRouter
from embodied_agent.core import RobotRegistry
from embodied_agent.embodiments import CrazyfliePyBullet, HumanoidMuJoCo, XLeRobotMuJoCo
from embodied_agent.world import WorldState

from .multi_robot import EvalSuiteResult, default_multi_robot_cases, evaluate_cases


def build_physics_stack(
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
) -> tuple[RobotRegistry, RobotToolRouter]:
    """Build the shared three-robot stack used by physics-backed evals."""
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

    router = RobotToolRouter(
        registry,
        {
            "xlerobot": ["navigate_to"],
            "crazyflie": ["takeoff", "goto", "land"],
            "humanoid": ["stand"],
        },
    )
    return registry, router


async def run_physics_baseline(
    *,
    xlerobot_runtime_root: str | Path,
    humanoid_runtime_root: str | Path,
) -> EvalSuiteResult:
    """Run the standard three-robot benchmark against real simulator adapters.

    This uses the same task cases and deterministic capability planner as the
    dependency-free baseline, but swaps the scripted embodiments for upstream
    PyBullet/MuJoCo runtimes.
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

        planner = CapabilityPlanner(router)
        executor = PlanExecutor(router, world=WorldState())
        return await evaluate_cases(planner, executor, default_multi_robot_cases())
    finally:
        for robot in reversed(connected):
            await robot.disconnect()


def main() -> None:
    xlerobot_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
    humanoid_root = os.environ.get("LEROBOT_HUMANOID_RUNTIME_ROOT")
    if not xlerobot_root or not humanoid_root:
        raise SystemExit(
            "Set XLEROBOT_UPSTREAM_ROOT and LEROBOT_HUMANOID_RUNTIME_ROOT to "
            "the pinned upstream checkouts before running the physics benchmark."
        )
    result = asyncio.run(
        run_physics_baseline(
            xlerobot_runtime_root=xlerobot_root,
            humanoid_runtime_root=humanoid_root,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
