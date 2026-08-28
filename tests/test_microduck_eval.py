from __future__ import annotations

import asyncio
import unittest

from embodied_agent.evals.agent_model import ExpectedActionModel, evaluate_agent_model
from embodied_agent.evals.microduck_skills import (
    default_microduck_skill_cases,
    evaluate_expected_microduck_agent,
    scripted_microduck_stack,
)
from embodied_agent.mcp import AgentDecision


class WrongKickFootModel(ExpectedActionModel):
    async def decide(self, context):
        decision = await super().decide(context)
        if decision.kind == "tool" and decision.tool == "microduck.kick":
            return AgentDecision.call("microduck.kick", {"foot": "left"})
        return decision


class MicroduckSkillEvalTests(unittest.TestCase):
    def test_oracle_baseline_is_perfect(self) -> None:
        result = asyncio.run(evaluate_expected_microduck_agent())
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

    def test_wrong_kick_foot_loses_arguments_without_losing_tool_selection(self) -> None:
        cases = default_microduck_skill_cases()
        result = asyncio.run(
            evaluate_agent_model(
                lambda case: WrongKickFootModel(case),
                cases=cases,
                max_steps=3,
                stack=scripted_microduck_stack(),
            )
        )
        self.assertEqual(result.tool_selection_accuracy, 1.0)
        self.assertLess(result.argument_accuracy, 1.0)
        self.assertLess(result.arguments_exact_match_rate, 1.0)
        self.assertLess(result.strict_task_success_rate, 1.0)

        right_kick = next(case for case in result.cases if case.name == "microduck-kick-right")
        self.assertEqual(right_kick.tool_selection_accuracy, 1.0)
        self.assertEqual(right_kick.argument_accuracy, 0.0)
        self.assertFalse(right_kick.strict_task_success)


if __name__ == "__main__":
    unittest.main()
