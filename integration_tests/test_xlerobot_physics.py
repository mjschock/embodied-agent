from __future__ import annotations

import asyncio
import os
import unittest

from embodied_agent.core import Capability
from embodied_agent.embodiments import XLeRobotMuJoCo


class XLeRobotPhysicsIntegrationTests(unittest.TestCase):
    def test_drive_reset_and_navigate_in_real_upstream_mujoco(self) -> None:
        async def scenario() -> None:
            runtime_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
            self.assertTrue(runtime_root, "XLEROBOT_UPSTREAM_ROOT must point to the pinned upstream checkout")

            robot = XLeRobotMuJoCo(
                runtime_root=runtime_root,
                position_tolerance_m=0.04,
                yaw_tolerance_rad=0.06,
            )
            self.assertIn(Capability.NAVIGATE, robot.capabilities)
            self.assertNotIn(Capability.MANIPULATE, robot.capabilities)

            await robot.connect()
            try:
                initial = await robot.observe()
                self.assertEqual(initial.state["backend"], "xlerobot-mujoco")
                self.assertEqual(len(initial.state["arm_joint_position_rad"]), 12)

                drive = await robot.execute(
                    "drive_velocity",
                    lin_x_mps=0.12,
                    lin_y_mps=0.04,
                    yaw_rate_rps=0.15,
                    duration_s=0.35,
                )
                self.assertTrue(drive.ok, drive.detail)
                moved = await robot.observe()
                displacement = (
                    moved.state["base"]["x_m"] ** 2
                    + moved.state["base"]["y_m"] ** 2
                ) ** 0.5
                self.assertGreater(displacement, 0.005)
                self.assertGreater(abs(moved.state["base"]["yaw_rad"]), 0.005)

                reset = await robot.execute("reset")
                self.assertTrue(reset.ok, reset.detail)
                self.assertAlmostEqual(reset.data["x_m"], 0.0, delta=1e-6)
                self.assertAlmostEqual(reset.data["y_m"], 0.0, delta=1e-6)
                self.assertAlmostEqual(reset.data["yaw_rad"], 0.0, delta=1e-6)

                navigate = await robot.execute(
                    "navigate_to",
                    x_m=0.10,
                    y_m=-0.06,
                    yaw_rad=0.12,
                    max_duration_s=4.0,
                )
                self.assertTrue(navigate.ok, f"{navigate.detail} data={navigate.data}")
                self.assertEqual(navigate.data["timeout_source"], "explicit")
                self.assertLessEqual(
                    navigate.data["position_error_m"], robot.position_tolerance_m
                )
                self.assertLessEqual(
                    abs(navigate.data["yaw_error_rad"]), robot.yaw_tolerance_rad
                )
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_sequential_eval_waypoints_in_real_upstream_mujoco(self) -> None:
        async def scenario() -> None:
            runtime_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
            self.assertTrue(runtime_root, "XLEROBOT_UPSTREAM_ROOT must point to the pinned upstream checkout")

            robot = XLeRobotMuJoCo(
                runtime_root=runtime_root,
                position_tolerance_m=0.05,
                yaw_tolerance_rad=0.06,
            )
            await robot.connect()
            try:
                resolved_budgets: list[float] = []
                for label, x_m, y_m in (
                    ("workbench-a", 1.5, 0.5),
                    ("workbench-b", -1.0, 1.25),
                    ("workbench-c", 0.75, -1.5),
                ):
                    result = await robot.execute(
                        "navigate_to",
                        x_m=x_m,
                        y_m=y_m,
                        yaw_rad=0.0,
                    )
                    self.assertTrue(
                        result.ok,
                        f"{label}: {result.detail} data={result.data}",
                    )
                    self.assertEqual(result.data["timeout_source"], "distance-aware")
                    resolved_budgets.append(result.data["max_duration_s"])
                    self.assertLessEqual(
                        result.data["position_error_m"], robot.position_tolerance_m
                    )
                    self.assertLessEqual(
                        abs(result.data["yaw_error_rad"]), robot.yaw_tolerance_rad
                    )
                self.assertGreater(resolved_budgets[-1], 10.0)
                self.assertLessEqual(resolved_budgets[-1], 30.0)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
