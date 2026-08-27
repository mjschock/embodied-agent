from __future__ import annotations

import asyncio
import unittest

from embodied_agent.core import Capability
from embodied_agent.embodiments.humanoid_mujoco import HumanoidMuJoCo
from test_humanoid_mujoco import FakeSimBipedalRobotController


class FakeRLAgent:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.commands: list[tuple[float, float, float]] = []

    def set_command_twist(self, lin_x: float, lin_y: float, yaw_rate: float) -> None:
        self.commands.append((lin_x, lin_y, yaw_rate))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def fake_agent_factory(*, controller, policy_dir):
    del controller, policy_dir
    return FakeRLAgent()


class HumanoidWalkPolicyTests(unittest.TestCase):
    def test_walk_capability_only_when_policy_configured(self) -> None:
        bare = HumanoidMuJoCo(controller_factory=FakeSimBipedalRobotController)
        policy = HumanoidMuJoCo(
            controller_factory=FakeSimBipedalRobotController,
            agent_factory=fake_agent_factory,
        )
        self.assertNotIn(Capability.WALK, bare.capabilities)
        self.assertIn(Capability.WALK, policy.capabilities)

    def test_bounded_walk_velocity_zeros_command_after_duration(self) -> None:
        async def scenario() -> None:
            robot = HumanoidMuJoCo(
                controller_factory=FakeSimBipedalRobotController,
                agent_factory=fake_agent_factory,
            )
            await robot.connect()
            agent = robot._agent
            self.assertTrue(agent.started)
            self.assertEqual(agent.commands[-1], (0.0, 0.0, 0.0))

            result = await robot.execute(
                "walk_velocity",
                lin_x_mps=0.2,
                lin_y_mps=-0.1,
                yaw_rate_rps=0.15,
                duration_s=0.001,
            )
            self.assertTrue(result.ok)
            self.assertIn((0.2, -0.1, 0.15), agent.commands)
            self.assertEqual(agent.commands[-1], (0.0, 0.0, 0.0))

            await robot.disconnect()
            self.assertTrue(agent.stopped)

        asyncio.run(scenario())

    def test_walk_velocity_rejects_out_of_bounds_command(self) -> None:
        async def scenario() -> None:
            robot = HumanoidMuJoCo(
                controller_factory=FakeSimBipedalRobotController,
                agent_factory=fake_agent_factory,
            )
            await robot.connect()
            with self.assertRaises(ValueError):
                await robot.execute("walk_velocity", lin_x_mps=1.0, duration_s=0.1)
            await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
