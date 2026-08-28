from __future__ import annotations

import asyncio
import json
import unittest

from embodied_agent.embodiments import CrazyfliePyBullet
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills


class CrazyfliePhysicsIntegrationTests(unittest.TestCase):
    def test_takeoff_translate_observe_and_land_in_real_pybullet(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                gui=False,
                seed=0,
                ctrl_freq_hz=60,
                initial_position=(0.0, 0.0, 0.1),
                position_tolerance_m=0.06,
                slow_radius_m=0.25,
            )
            await robot.connect()
            try:
                initial = await robot.observe()
                self.assertEqual(initial.state["backend"], "gym-pybullet-drones")
                self.assertEqual(len(initial.state["position_m"]), 3)
                self.assertEqual(len(initial.state["motor_rpm"]), 4)

                takeoff = await robot.execute(
                    "takeoff",
                    altitude_m=0.4,
                    timeout_s=6.0,
                )
                self.assertTrue(takeoff.ok, takeoff.detail)
                self.assertLessEqual(takeoff.data["position_error_m"], 0.06)

                move = await robot.execute(
                    "goto",
                    position=(0.15, 0.10, 0.4),
                    timeout_s=6.0,
                )
                self.assertTrue(move.ok, move.detail)
                self.assertLessEqual(move.data["position_error_m"], 0.06)

                moved = await robot.observe()
                self.assertAlmostEqual(moved.state["position_m"][0], 0.15, delta=0.07)
                self.assertAlmostEqual(moved.state["position_m"][1], 0.10, delta=0.07)
                self.assertAlmostEqual(moved.state["position_m"][2], 0.4, delta=0.07)

                land = await robot.execute(
                    "land",
                    height_m=0.08,
                    timeout_s=7.0,
                )
                self.assertTrue(land.ok, land.detail)
                self.assertLessEqual(land.data["position_error_m"], 0.06)

                landed = await robot.observe()
                self.assertAlmostEqual(landed.state["position_m"][2], 0.08, delta=0.07)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_skill_reliability_from_seeded_comparable_states(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                gui=False,
                seed=0,
                ctrl_freq_hz=60,
                initial_position=(0.0, 0.0, 0.1),
                position_tolerance_m=0.06,
                slow_radius_m=0.25,
            )

            async def prepare_attempt(robot, probe, attempt) -> None:
                # Reconstruct and re-seed VelocityAviary before every sample. This is
                # intentionally outside the benchmark clock so every timed skill sees
                # the same simulator start condition without counting environment setup.
                await robot.disconnect()
                await robot.connect()
                initial = await robot.observe()
                self.assertEqual(initial.state["backend"], "gym-pybullet-drones")
                self.assertAlmostEqual(initial.state["position_m"][0], 0.0, delta=1e-6)
                self.assertAlmostEqual(initial.state["position_m"][1], 0.0, delta=1e-6)
                self.assertAlmostEqual(initial.state["position_m"][2], 0.1, delta=1e-6)

                # Horizontal translation and landing both require a stable airborne
                # precondition. Establish it before timing the skill under test.
                if probe.skill in {"goto", "land"}:
                    takeoff = await robot.execute(
                        "takeoff",
                        altitude_m=0.4,
                        timeout_s=6.0,
                    )
                    if not takeoff.ok:
                        raise RuntimeError(
                            f"Crazyflie airborne precondition failed: {takeoff.detail}"
                        )
                    airborne = await robot.observe()
                    self.assertAlmostEqual(
                        airborne.state["position_m"][2], 0.4, delta=0.07
                    )

            try:
                result = await benchmark_robot_skills(
                    robot,
                    [
                        SkillProbe(
                            "takeoff",
                            {"altitude_m": 0.4, "timeout_s": 6.0},
                            attempts=3,
                        ),
                        SkillProbe(
                            "goto",
                            {
                                "position": (0.15, 0.10, 0.4),
                                "timeout_s": 6.0,
                            },
                            attempts=3,
                            label="goto-short-translation",
                        ),
                        SkillProbe(
                            "land",
                            {"height_m": 0.08, "timeout_s": 7.0},
                            attempts=3,
                        ),
                    ],
                    manage_connection=False,
                    before_attempt=prepare_attempt,
                )
            finally:
                await robot.disconnect()

            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            self.assertEqual(result.backend, "gym-pybullet-drones")
            self.assertEqual(result.attempt_count, 9)
            self.assertEqual(result.success_count, 9)
            self.assertEqual(result.success_rate, 1.0)
            for metric in result.metrics:
                self.assertEqual(metric.success_rate, 1.0, metric.to_dict())
                self.assertGreaterEqual(metric.p50_latency_ms, 0.0)
                self.assertGreaterEqual(metric.p95_latency_ms, metric.p50_latency_ms)
                self.assertGreaterEqual(metric.max_latency_ms, metric.p95_latency_ms)
                self.assertIsNotNone(metric.successful_mean_latency_ms)
                self.assertTrue(all(not sample.error for sample in metric.samples))

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
