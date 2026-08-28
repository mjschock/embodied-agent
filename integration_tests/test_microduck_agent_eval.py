from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path

from embodied_agent.agent import RobotToolRouter
from embodied_agent.core import RobotRegistry
from embodied_agent.embodiments import MicroduckMuJoCo
from embodied_agent.evals.microduck_skills import evaluate_expected_microduck_agent


class MicroduckAgentPhysicsEvalTests(unittest.TestCase):
    def test_oracle_agent_scores_perfectly_on_real_microduck_policies(self) -> None:
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
        registry = RobotRegistry()
        registry.register(robot)
        router = RobotToolRouter(
            registry,
            {
                "microduck": [
                    "observe",
                    "reset",
                    "stand",
                    "walk_velocity",
                    "kick",
                    "roll",
                ]
            },
        )

        result = asyncio.run(
            evaluate_expected_microduck_agent(
                stack=(registry, router),
                max_steps=3,
            )
        )

        self.assertEqual(len(result.cases), 5)
        self.assertEqual(result.tool_selection_accuracy, 1.0)
        self.assertEqual(result.argument_accuracy, 1.0)
        self.assertEqual(result.sequence_exact_match_rate, 1.0)
        self.assertEqual(result.arguments_exact_match_rate, 1.0)
        self.assertEqual(result.tool_execution_success_rate, 1.0)
        self.assertEqual(result.runner_finish_rate, 1.0)
        self.assertEqual(result.runner_ok_rate, 1.0)
        self.assertEqual(result.strict_task_success_rate, 1.0)
        self.assertEqual(result.mean_action_efficiency, 1.0)


if __name__ == "__main__":
    unittest.main()
