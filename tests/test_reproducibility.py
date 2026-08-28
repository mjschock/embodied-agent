from __future__ import annotations

import asyncio
import unittest

from embodied_agent.evals.reproducibility import benchmark_reproducibility


class ReproducibilityMetricTests(unittest.TestCase):
    def test_matching_nested_payloads_respect_float_tolerance(self) -> None:
        payloads = [
            {
                "ok": True,
                "steps": 42,
                "pose": [1.0, -2.0, 0.5],
                "state": {"yaw": 0.25, "mode": "done"},
            },
            {
                "ok": True,
                "steps": 42,
                "pose": [1.0 + 1e-7, -2.0, 0.5],
                "state": {"yaw": 0.25 - 2e-7, "mode": "done"},
            },
            {
                "ok": True,
                "steps": 42,
                "pose": [1.0, -2.0 + 3e-7, 0.5],
                "state": {"yaw": 0.25, "mode": "done"},
            },
        ]

        async def scenario():
            async def run_episode(attempt: int):
                return payloads[attempt - 1]

            return await benchmark_reproducibility(
                run_episode,
                attempts=3,
                atol=1e-6,
                label="nested",
            )

        result = asyncio.run(scenario())
        self.assertEqual(result.comparison_count, 2)
        self.assertEqual(result.matching_comparisons, 2)
        self.assertEqual(result.reproducibility_rate, 1.0)
        self.assertTrue(all(sample.matches_baseline for sample in result.samples))
        self.assertGreater(result.samples[1].max_abs_error, 0.0)

    def test_discrete_and_numeric_divergence_report_paths(self) -> None:
        payloads = [
            {"ok": True, "steps": 10, "pose": [1.0, 2.0]},
            {"ok": True, "steps": 11, "pose": [1.0, 2.01]},
            {"ok": False, "steps": 10, "pose": [1.0, 2.0]},
        ]

        async def scenario():
            async def run_episode(attempt: int):
                return payloads[attempt - 1]

            return await benchmark_reproducibility(
                run_episode,
                attempts=3,
                atol=1e-3,
            )

        result = asyncio.run(scenario())
        self.assertEqual(result.matching_comparisons, 0)
        self.assertEqual(result.reproducibility_rate, 0.0)
        self.assertIn("steps", result.samples[1].mismatch_paths)
        self.assertIn("pose[1]", result.samples[1].mismatch_paths)
        self.assertAlmostEqual(result.samples[1].max_abs_error, 0.01)
        self.assertIn("ok", result.samples[2].mismatch_paths)

    def test_structure_mismatches_are_visible(self) -> None:
        payloads = [
            {"pose": [1.0, 2.0], "meta": {"seed": 0}},
            {"pose": [1.0], "extra": True},
        ]

        async def scenario():
            async def run_episode(attempt: int):
                return payloads[attempt - 1]

            return await benchmark_reproducibility(run_episode, attempts=2)

        result = asyncio.run(scenario())
        paths = set(result.samples[1].mismatch_paths)
        self.assertIn("meta", paths)
        self.assertIn("extra", paths)
        self.assertIn("pose.length", paths)

    def test_validation_requires_repeatable_finite_configuration(self) -> None:
        async def run_episode(attempt: int):
            return {"attempt": attempt}

        with self.assertRaises(ValueError):
            asyncio.run(benchmark_reproducibility(run_episode, attempts=1))
        with self.assertRaises(ValueError):
            asyncio.run(benchmark_reproducibility(run_episode, atol=-1.0))
        with self.assertRaises(ValueError):
            asyncio.run(benchmark_reproducibility(run_episode, atol=float("nan")))
        with self.assertRaises(ValueError):
            asyncio.run(benchmark_reproducibility(run_episode, label="   "))

        async def bad_episode(attempt: int):
            return [attempt]

        with self.assertRaises(TypeError):
            asyncio.run(benchmark_reproducibility(bad_episode))


if __name__ == "__main__":
    unittest.main()
