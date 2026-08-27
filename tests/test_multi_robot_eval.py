from __future__ import annotations

import asyncio
import unittest

from embodied_agent.evals import default_multi_robot_cases, run_scripted_baseline


class MultiRobotEvalTests(unittest.TestCase):
    def test_scripted_baseline_is_perfect(self) -> None:
        result = asyncio.run(run_scripted_baseline())
        self.assertEqual(len(result.cases), 3)
        self.assertEqual(result.robot_selection_accuracy, 1.0)
        self.assertEqual(result.plan_exact_match_rate, 1.0)
        self.assertEqual(result.task_completion_rate, 1.0)
        self.assertEqual(result.tool_call_success_rate, 1.0)
        self.assertEqual(result.executed_step_coverage, 1.0)
        for case in result.cases:
            self.assertEqual(
                case.planned_tools,
                (
                    "crazyflie.takeoff",
                    "crazyflie.goto",
                    "xlerobot.navigate_to",
                    "humanoid.stand",
                    "crazyflie.land",
                ),
            )

    def test_execution_failure_does_not_look_like_planning_failure(self) -> None:
        result = asyncio.run(
            run_scripted_baseline(failures={"xlerobot": "navigate_to"})
        )
        self.assertEqual(result.robot_selection_accuracy, 1.0)
        self.assertEqual(result.plan_exact_match_rate, 1.0)
        self.assertEqual(result.task_completion_rate, 0.0)
        self.assertAlmostEqual(result.tool_call_success_rate, 2.0 / 3.0)
        self.assertAlmostEqual(result.executed_step_coverage, 3.0 / 5.0)
        for case in result.cases:
            self.assertFalse(case.execution_ok)
            self.assertEqual(case.calls_attempted, 3)
            self.assertEqual(case.calls_succeeded, 2)

    def test_eval_cases_retain_natural_language_instruction_for_future_planners(self) -> None:
        cases = default_multi_robot_cases()
        self.assertTrue(all(case.instruction for case in cases))
        self.assertIn("aerial robot", cases[0].instruction)
        self.assertIn("ground robot", cases[0].instruction)
        self.assertIn("humanoid", cases[0].instruction)


if __name__ == "__main__":
    unittest.main()
