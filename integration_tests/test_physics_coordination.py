from __future__ import annotations

import asyncio
import json
import os
import unittest

from embodied_agent.evals.physics_agent_model import run_expected_action_physics_comparison


class PhysicsCoordinationIntegrationTests(unittest.TestCase):
    def test_deterministic_and_agent_paths_are_perfect_on_real_simulators(self) -> None:
        async def scenario() -> None:
            xlerobot_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
            humanoid_root = os.environ.get("LEROBOT_HUMANOID_RUNTIME_ROOT")
            self.assertTrue(xlerobot_root, "XLEROBOT_UPSTREAM_ROOT is required")
            self.assertTrue(
                humanoid_root, "LEROBOT_HUMANOID_RUNTIME_ROOT is required"
            )

            result = await run_expected_action_physics_comparison(
                xlerobot_runtime_root=xlerobot_root,
                humanoid_runtime_root=humanoid_root,
            )
            diagnostics = result.to_dict()
            print(json.dumps(diagnostics, indent=2, sort_keys=True))

            deterministic = result.deterministic
            self.assertEqual(len(deterministic.cases), 3, diagnostics)
            self.assertEqual(deterministic.robot_selection_accuracy, 1.0, diagnostics)
            self.assertEqual(deterministic.plan_exact_match_rate, 1.0, diagnostics)
            self.assertEqual(deterministic.task_completion_rate, 1.0, diagnostics)
            self.assertEqual(deterministic.tool_call_success_rate, 1.0, diagnostics)
            self.assertEqual(deterministic.executed_step_coverage, 1.0, diagnostics)
            self.assertTrue(
                all(case.execution_ok for case in deterministic.cases), diagnostics
            )

            agent = result.agent
            self.assertEqual(len(agent.cases), 3, diagnostics)
            self.assertEqual(agent.tool_selection_accuracy, 1.0, diagnostics)
            self.assertEqual(agent.argument_accuracy, 1.0, diagnostics)
            self.assertEqual(agent.sequence_exact_match_rate, 1.0, diagnostics)
            self.assertEqual(agent.arguments_exact_match_rate, 1.0, diagnostics)
            self.assertEqual(agent.tool_execution_success_rate, 1.0, diagnostics)
            self.assertEqual(agent.runner_finish_rate, 1.0, diagnostics)
            self.assertEqual(agent.runner_ok_rate, 1.0, diagnostics)
            self.assertEqual(agent.strict_task_success_rate, 1.0, diagnostics)
            self.assertEqual(agent.mean_action_efficiency, 1.0, diagnostics)

            self.assertEqual(result.strict_task_success_gap, 0.0, diagnostics)
            self.assertEqual(result.tool_execution_success_gap, 0.0, diagnostics)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
