from __future__ import annotations

import asyncio
import unittest

from embodied_agent.core import Capability
from embodied_agent.embodiments.humanoid_mujoco import HumanoidMuJoCo


class FakeSimBipedalRobotController:
    def __init__(self, *, control_hz: float, fixed_base: bool) -> None:
        self.control_hz = control_hz
        self.fixed_base = fixed_base
        self.started = False
        self.stopped = False
        self.mode = "state_only"
        self.step_count = 0
        self.reset_count = 0
        self.last_action = None

    def start(self, *, mode: str, auto_enable: bool) -> None:
        self.mode = mode
        self.started = auto_enable

    def stop(self, *, disable_motors: bool) -> None:
        self.stopped = disable_motors

    def request_state_once(self) -> list[int]:
        self.step_count += 1
        return []

    def reset(self) -> None:
        self.reset_count += 1
        self.step_count = 0

    def set_action(self, *, left, right):
        self.last_action = {"left": left, "right": right}
        return {motor_id: 0.0 for motor_id in range(1, 13)}

    def get_combined_state_snapshot(self, *, include_joint_state: bool):
        assert include_joint_state
        return {
            "mode": self.mode,
            "sim_step_count": self.step_count,
            "sim_reset_count": self.reset_count,
            "fixed_base": self.fixed_base,
            "joint_state_rad": [0.01 * i for i in range(12)],
            "joint_velocity_rad_s": [0.1] * 12,
            "joint_torque_nm": [0.2] * 12,
            "orientation_quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
            "imu": {"gyro_rad_s": [0.0, 0.0, 0.0]},
        }


class HumanoidMuJoCoTests(unittest.TestCase):
    def test_lifecycle_observation_and_stand(self) -> None:
        async def scenario() -> None:
            robot = HumanoidMuJoCo(
                controller_factory=FakeSimBipedalRobotController,
                fixed_base=True,
            )
            self.assertIn(Capability.STAND, robot.capabilities)
            self.assertNotIn(Capability.WALK, robot.capabilities)

            await robot.connect()
            controller = robot._controller
            self.assertTrue(controller.started)
            self.assertEqual(controller.mode, "control")

            observation = await robot.observe()
            self.assertEqual(observation.state["backend"], "lerobot-humanoid-mujoco")
            self.assertTrue(observation.state["fixed_base"])
            self.assertEqual(len(observation.state["joint_position_rad"]), 12)

            result = await robot.execute("stand")
            self.assertTrue(result.ok)
            self.assertEqual(set(controller.last_action["left"]), {
                "hipz", "hipx", "hipy", "knee", "ankle_pitch", "ankle_roll"
            })
            self.assertTrue(all(v == 0.0 for v in controller.last_action["left"].values()))
            self.assertTrue(all(v == 0.0 for v in controller.last_action["right"].values()))

            await robot.disconnect()
            self.assertTrue(controller.stopped)

        asyncio.run(scenario())

    def test_reset_returns_runtime_state(self) -> None:
        async def scenario() -> None:
            robot = HumanoidMuJoCo(controller_factory=FakeSimBipedalRobotController)
            await robot.connect()
            result = await robot.execute("reset")
            self.assertTrue(result.ok)
            self.assertEqual(result.data["sim_reset_count"], 1)
            self.assertEqual(len(result.data["joint_position_rad"]), 12)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_requires_connection(self) -> None:
        async def scenario() -> None:
            robot = HumanoidMuJoCo(controller_factory=FakeSimBipedalRobotController)
            with self.assertRaises(RuntimeError):
                await robot.observe()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
