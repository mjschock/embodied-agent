from __future__ import annotations

import asyncio
import json
import os
import unittest

from embodied_agent.core import Capability
from embodied_agent.embodiments import XLeRobotMuJoCo
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills


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

    def test_navigation_reliability_from_identical_reset_state(self) -> None:
        async def scenario() -> None:
            runtime_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
            self.assertTrue(runtime_root, "XLEROBOT_UPSTREAM_ROOT must point to the pinned upstream checkout")

            robot = XLeRobotMuJoCo(
                runtime_root=runtime_root,
                position_tolerance_m=0.04,
                yaw_tolerance_rad=0.06,
            )

            async def reset_to_origin(robot, probe, attempt) -> None:
                reset = await robot.execute("reset")
                if not reset.ok:
                    raise RuntimeError(reset.detail)
                if (
                    abs(float(reset.data["x_m"])) > 1e-6
                    or abs(float(reset.data["y_m"])) > 1e-6
                    or abs(float(reset.data["yaw_rad"])) > 1e-6
                ):
                    raise RuntimeError(
                        f"XLeRobot reset did not restore origin before attempt {attempt}"
                    )

            result = await benchmark_robot_skills(
                robot,
                (
                    SkillProbe(
                        "navigate_to",
                        {
                            "x_m": 0.10,
                            "y_m": -0.06,
                            "yaw_rad": 0.12,
                            "max_duration_s": 4.0,
                        },
                        attempts=3,
                        label="navigate-short-pose",
                    ),
                ),
                before_attempt=reset_to_origin,
            )

            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

            self.assertEqual(result.robot, "xlerobot")
            self.assertEqual(result.backend, "xlerobot-mujoco")
            self.assertEqual(result.attempt_count, 3)
            self.assertEqual(result.success_count, 3)
            self.assertEqual(result.success_rate, 1.0)

            metric = result.metrics[0]
            self.assertEqual(metric.label, "navigate-short-pose")
            self.assertEqual(metric.attempts, 3)
            self.assertEqual(metric.successes, 3)
            self.assertEqual(metric.success_rate, 1.0)
            self.assertGreater(metric.mean_latency_ms, 0.0)
            self.assertGreaterEqual(metric.p95_latency_ms, metric.p50_latency_ms)
            self.assertGreaterEqual(metric.max_latency_ms, metric.p95_latency_ms)
            self.assertIsNotNone(metric.successful_mean_latency_ms)
            self.assertTrue(all(sample.ok for sample in metric.samples))
            self.assertTrue(all(sample.error == "" for sample in metric.samples))

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
