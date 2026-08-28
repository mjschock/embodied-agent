from __future__ import annotations

import asyncio
import os
import unittest

from embodied_agent.evals.physics_four_embodiment import (
    four_embodiment_physics_case,
    run_expected_four_embodiment_physics,
)


class FourEmbodimentPhysicsIntegrationTests(unittest.TestCase):
    def test_oracle_coordinates_all_four_real_simulator_adapters(self) -> None:
        expected = four_embodiment_physics_case()
        result = asyncio.run(
            run_expected_four_embodiment_physics(
                xlerobot_runtime_root=os.environ["XLEROBOT_UPSTREAM_ROOT"],
                humanoid_runtime_root=os.environ["LEROBOT_HUMANOID_RUNTIME_ROOT"],
                microduck_runtime_root=os.environ["MICRODUCK_RL_ROOT"],
                microduck_policy_dir=os.environ["MICRODUCK_POLICY_DIR"],
                max_steps=8,
            )
        )

        self.assertEqual(len(result.cases), 1)
        self.assertEqual(result.tool_selection_accuracy, 1.0, result.to_dict())
        self.assertEqual(result.argument_accuracy, 1.0, result.to_dict())
        self.assertEqual(result.sequence_exact_match_rate, 1.0, result.to_dict())
        self.assertEqual(result.arguments_exact_match_rate, 1.0, result.to_dict())
        self.assertEqual(result.tool_execution_success_rate, 1.0, result.to_dict())
        self.assertEqual(result.runner_finish_rate, 1.0, result.to_dict())
        self.assertEqual(result.runner_ok_rate, 1.0, result.to_dict())
        self.assertEqual(result.strict_task_success_rate, 1.0, result.to_dict())
        self.assertEqual(result.mean_action_efficiency, 1.0, result.to_dict())

        case = result.cases[0]
        self.assertEqual(case.actual_tools, expected.expected_tools)
        self.assertEqual(
            {tool.split(".", 1)[0] for tool in case.actual_tools},
            {"crazyflie", "xlerobot", "humanoid", "microduck"},
        )


if __name__ == "__main__":
    unittest.main()
