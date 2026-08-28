from __future__ import annotations

import asyncio
import json
import os
import unittest

from embodied_agent.evals.physics_multi_robot import run_physics_baseline


class PhysicsCoordinationIntegrationTests(unittest.TestCase):
    def test_standard_three_robot_suite_is_perfect_on_real_simulators(self) -> None:
        async def scenario() -> None:
            xlerobot_root = os.environ.get("XLEROBOT_UPSTREAM_ROOT")
            humanoid_root = os.environ.get("LEROBOT_HUMANOID_RUNTIME_ROOT")
            self.assertTrue(xlerobot_root, "XLEROBOT_UPSTREAM_ROOT is required")
            self.assertTrue(
                humanoid_root, "LEROBOT_HUMANOID_RUNTIME_ROOT is required"
            )

            result = await run_physics_baseline(
                xlerobot_runtime_root=xlerobot_root,
                humanoid_runtime_root=humanoid_root,
            )
            diagnostics = result.to_dict()
            print(json.dumps(diagnostics, indent=2, sort_keys=True))

            self.assertEqual(len(result.cases), 3, diagnostics)
            self.assertEqual(result.robot_selection_accuracy, 1.0, diagnostics)
            self.assertEqual(result.plan_exact_match_rate, 1.0, diagnostics)
            self.assertEqual(result.task_completion_rate, 1.0, diagnostics)
            self.assertEqual(result.tool_call_success_rate, 1.0, diagnostics)
            self.assertEqual(result.executed_step_coverage, 1.0, diagnostics)
            self.assertTrue(
                all(case.execution_ok for case in result.cases), diagnostics
            )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
