from __future__ import annotations

import asyncio
import math
import unittest

from embodied_agent.core import Capability
from embodied_agent.embodiments.xlerobot_mujoco import XLeRobotMuJoCo


class FakeXLeRobotRuntime:
    def __init__(self, scene_path) -> None:
        self.scene_path = scene_path
        self.timestep_s = 0.01
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx_body = 0.0
        self.vy_body = 0.0
        self.yaw_rate = 0.0
        self.time_s = 0.0

    def reset(self) -> None:
        self.x = self.y = self.yaw = self.time_s = 0.0
        self.zero_base_velocity()

    def set_base_velocity(self, lin_x_mps, lin_y_mps, yaw_rate_rps) -> None:
        self.vx_body = float(lin_x_mps)
        self.vy_body = float(lin_y_mps)
        self.yaw_rate = float(yaw_rate_rps)

    def zero_base_velocity(self) -> None:
        self.vx_body = self.vy_body = self.yaw_rate = 0.0

    def step(self) -> None:
        world_x = self.vx_body * math.cos(self.yaw) - self.vy_body * math.sin(self.yaw)
        world_y = self.vx_body * math.sin(self.yaw) + self.vy_body * math.cos(self.yaw)
        self.x += world_x * self.timestep_s
        self.y += world_y * self.timestep_s
        self.yaw += self.yaw_rate * self.timestep_s
        self.time_s += self.timestep_s

    def get_state(self):
        world_x = self.vx_body * math.cos(self.yaw) - self.vy_body * math.sin(self.yaw)
        world_y = self.vx_body * math.sin(self.yaw) + self.vy_body * math.cos(self.yaw)
        return {
            "sim_time_s": self.time_s,
            "x_m": self.x,
            "y_m": self.y,
            "yaw_rad": self.yaw,
            "linear_velocity_world_mps": [world_x, world_y],
            "yaw_rate_rps": self.yaw_rate,
            "arm_joint_position_rad": {},
        }

    def close(self) -> None:
        pass


class XLeRobotMuJoCoTests(unittest.TestCase):
    def _robot(self, **kwargs) -> XLeRobotMuJoCo:
        return XLeRobotMuJoCo(
            scene_path="fake-scene.xml",
            sim_factory=FakeXLeRobotRuntime,
            **kwargs,
        )

    def test_capability_boundary_and_drive_velocity(self) -> None:
        async def scenario() -> None:
            robot = self._robot()
            self.assertIn(Capability.NAVIGATE, robot.capabilities)
            self.assertNotIn(Capability.MANIPULATE, robot.capabilities)
            await robot.connect()
            result = await robot.execute(
                "drive_velocity",
                lin_x_mps=0.2,
                lin_y_mps=0.1,
                yaw_rate_rps=0.0,
                duration_s=0.1,
            )
            self.assertTrue(result.ok)
            self.assertAlmostEqual(result.data["x_m"], 0.02, places=6)
            self.assertAlmostEqual(result.data["y_m"], 0.01, places=6)
            self.assertEqual(robot._sim.vx_body, 0.0)
            self.assertEqual(robot._sim.vy_body, 0.0)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_closed_loop_navigate_to(self) -> None:
        async def scenario() -> None:
            robot = self._robot()
            await robot.connect()
            result = await robot.execute(
                "navigate_to",
                x_m=0.20,
                y_m=-0.10,
                yaw_rad=0.25,
                max_duration_s=3.0,
            )
            self.assertTrue(result.ok, result)
            self.assertEqual(result.data["timeout_source"], "explicit")
            self.assertEqual(result.data["max_duration_s"], 3.0)
            self.assertLessEqual(result.data["position_error_m"], robot.position_tolerance_m)
            self.assertLessEqual(abs(result.data["yaw_error_rad"]), robot.yaw_tolerance_rad)
            self.assertEqual(robot._sim.vx_body, 0.0)
            self.assertEqual(robot._sim.vy_body, 0.0)
            self.assertEqual(robot._sim.yaw_rate, 0.0)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_default_navigation_timeout_expands_for_long_transit(self) -> None:
        async def scenario() -> None:
            robot = self._robot(max_linear_x_mps=0.1, max_linear_y_mps=0.1)
            await robot.connect()
            result = await robot.execute(
                "navigate_to",
                x_m=0.6,
                y_m=0.0,
                yaw_rad=0.0,
            )
            self.assertTrue(result.ok, result)
            self.assertEqual(result.data["timeout_source"], "distance-aware")
            self.assertGreater(result.data["max_duration_s"], 10.0)
            self.assertLessEqual(result.data["max_duration_s"], 30.0)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_explicit_navigation_timeout_is_bounded(self) -> None:
        async def scenario() -> None:
            robot = self._robot()
            await robot.connect()
            with self.assertRaises(ValueError):
                await robot.execute(
                    "navigate_to",
                    x_m=1.0,
                    y_m=0.0,
                    max_duration_s=31.0,
                )
            await robot.disconnect()

        asyncio.run(scenario())

    def test_reset_and_velocity_bounds(self) -> None:
        async def scenario() -> None:
            robot = self._robot()
            await robot.connect()
            await robot.execute("drive_velocity", lin_x_mps=0.2, duration_s=0.1)
            reset = await robot.execute("reset")
            self.assertTrue(reset.ok)
            self.assertEqual(reset.data["x_m"], 0.0)
            with self.assertRaises(ValueError):
                await robot.execute("drive_velocity", lin_x_mps=1.0, duration_s=0.1)
            await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
