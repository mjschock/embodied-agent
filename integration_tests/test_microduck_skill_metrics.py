from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path

from embodied_agent.embodiments import MicroduckMuJoCo
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills


class MicroduckSkillMetricIntegrationTests(unittest.TestCase):
    def test_pinned_policy_skills_report_reliability_and_latency(self) -> None:
        runtime_root = Path(os.environ["MICRODUCK_RL_ROOT"])
        policy_dir = Path(os.environ["MICRODUCK_POLICY_DIR"])
        robot = MicroduckMuJoCo(
            runtime_root=runtime_root,
            walking_policy_path=policy_dir / "BEST_alpha_walking.onnx",
            standing_policy_path=policy_dir / "BEST_alpha_stand.onnx",
            kick_left_policy_path=policy_dir / "ball_kick_left.onnx",
            kick_right_policy_path=policy_dir / "ball_kick_right.onnx",
            roll_policy_path=policy_dir / "roulade.onnx",
        )

        result = asyncio.run(
            benchmark_robot_skills(
                robot,
                (
                    SkillProbe("stand", attempts=3),
                    SkillProbe("kick", {"foot": "left"}, attempts=3, label="kick-left"),
                    SkillProbe("kick", {"foot": "right"}, attempts=3, label="kick-right"),
                    SkillProbe("roll", attempts=3),
                ),
            )
        )

        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

        self.assertEqual(result.robot, "microduck")
        self.assertEqual(result.backend, "microduck-rl-mujoco")
        self.assertEqual(result.attempt_count, 12)
        self.assertEqual(result.success_count, 12)
        self.assertEqual(result.success_rate, 1.0)
        self.assertGreater(result.mean_latency_ms, 0.0)

        self.assertEqual(
            [metric.label for metric in result.metrics],
            ["stand", "kick-left", "kick-right", "roll"],
        )
        for metric in result.metrics:
            self.assertEqual(metric.attempts, 3)
            self.assertEqual(metric.successes, 3)
            self.assertEqual(metric.success_rate, 1.0)
            self.assertGreater(metric.mean_latency_ms, 0.0)
            self.assertGreaterEqual(metric.p50_latency_ms, 0.0)
            self.assertGreaterEqual(metric.p95_latency_ms, metric.p50_latency_ms)
            self.assertGreaterEqual(metric.max_latency_ms, metric.p95_latency_ms)
            self.assertIsNotNone(metric.successful_mean_latency_ms)
            self.assertEqual(len(metric.samples), 3)
            self.assertTrue(all(sample.ok for sample in metric.samples))
            self.assertTrue(all(sample.error == "" for sample in metric.samples))


if __name__ == "__main__":
    unittest.main()
