from __future__ import annotations

import asyncio
import json
import unittest

from embodied_agent.embodiments import CrazyfliePyBullet
from embodied_agent.evals.reproducibility import benchmark_reproducibility
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills


class CrazyfliePhysicsIntegrationTests(unittest.TestCase):
    def test_takeoff_translate_observe_land_and_reset_in_real_pybullet(self) -> None:
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

                reset = await robot.execute("reset", seed=11)
                self.assertTrue(reset.ok, reset.detail)
                self.assertEqual(reset.data["seed"], 11)
                self.assertAlmostEqual(reset.data["position_m"][0], 0.0, delta=1e-6)
                self.assertAlmostEqual(reset.data["position_m"][1], 0.0, delta=1e-6)
                self.assertAlmostEqual(reset.data["position_m"][2], 0.1, delta=1e-6)

                repeat_reset = await robot.execute("reset")
                self.assertTrue(repeat_reset.ok, repeat_reset.detail)
                self.assertEqual(repeat_reset.data["seed"], 11)
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
            await robot.connect()

            async def prepare_attempt(robot, probe, attempt) -> None:
                reset = await robot.execute("reset", seed=0)
                if not reset.ok:
                    raise RuntimeError(f"Crazyflie reset failed: {reset.detail}")
                self.assertAlmostEqual(reset.data["position_m"][0], 0.0, delta=1e-6)
                self.assertAlmostEqual(reset.data["position_m"][1], 0.0, delta=1e-6)
                self.assertAlmostEqual(reset.data["position_m"][2], 0.1, delta=1e-6)

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

    def test_seeded_takeoff_and_translation_are_reproducible(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                gui=False,
                seed=123,
                ctrl_freq_hz=60,
                initial_position=(0.0, 0.0, 0.1),
                position_tolerance_m=0.06,
                slow_radius_m=0.25,
            )
            await robot.connect()

            async def run_episode(attempt: int):
                reset = await robot.execute("reset", seed=123)
                if not reset.ok:
                    raise RuntimeError(reset.detail)
                takeoff = await robot.execute(
                    "takeoff",
                    altitude_m=0.4,
                    timeout_s=6.0,
                )
                move = await robot.execute(
                    "goto",
                    position=(0.15, 0.10, 0.4),
                    timeout_s=6.0,
                )
                if not takeoff.ok or not move.ok:
                    raise RuntimeError(
                        f"episode {attempt} failed: takeoff={takeoff.detail}; goto={move.detail}"
                    )
                return {
                    "takeoff": {
                        "ok": takeoff.ok,
                        "steps": int(takeoff.data["steps"]),
                        "final_position_m": takeoff.data["final_position_m"],
                        "position_error_m": float(takeoff.data["position_error_m"]),
                    },
                    "goto": {
                        "ok": move.ok,
                        "steps": int(move.data["steps"]),
                        "final_position_m": move.data["final_position_m"],
                        "position_error_m": float(move.data["position_error_m"]),
                    },
                }

            try:
                result = await benchmark_reproducibility(
                    run_episode,
                    attempts=3,
                    atol=1e-7,
                    label="crazyflie-seeded-takeoff-goto",
                )
            finally:
                await robot.disconnect()

            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            self.assertEqual(result.comparison_count, 2)
            self.assertEqual(result.matching_comparisons, 2)
            self.assertEqual(result.reproducibility_rate, 1.0)
            self.assertTrue(all(sample.matches_baseline for sample in result.samples))

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
