from __future__ import annotations

import asyncio
import unittest

from embodied_agent.evals.agent_model import (
    ExpectedActionModel,
    evaluate_agent_model,
    run_expected_action_baseline,
)
from embodied_agent.mcp import AgentDecision


class ImmediateFinishModel:
    async def decide(self, context):
        return AgentDecision.finish("done")


class WrongCoordinateModel(ExpectedActionModel):
    async def decide(self, context):
        index = len(context.history)
        if index >= len(self.expected):
            return AgentDecision.finish("done")
        action = self.expected[index]
        arguments = dict(action.arguments)
        if action.tool == "xlerobot.navigate_to":
            arguments["x_m"] = float(arguments["x_m"]) + 0.25
        return AgentDecision.call(action.tool, arguments)


class ExtraActionModel(ExpectedActionModel):
    async def decide(self, context):
        index = len(context.history)
        if index < len(self.expected):
            action = self.expected[index]
            return AgentDecision.call(action.tool, action.arguments)
        if index == len(self.expected):
            return AgentDecision.call("humanoid.stand", {})
        return AgentDecision.finish("done")


class AgentModelEvalTests(unittest.TestCase):
    def test_expected_action_baseline_scores_perfectly(self) -> None:
        result = asyncio.run(run_expected_action_baseline())
        self.assertEqual(len(result.cases), 3)
        self.assertEqual(result.tool_selection_accuracy, 1.0)
        self.assertEqual(result.argument_accuracy, 1.0)
        self.assertEqual(result.sequence_exact_match_rate, 1.0)
        self.assertEqual(result.arguments_exact_match_rate, 1.0)
        self.assertEqual(result.tool_execution_success_rate, 1.0)
        self.assertEqual(result.runner_finish_rate, 1.0)
        self.assertEqual(result.runner_ok_rate, 1.0)
        self.assertEqual(result.strict_task_success_rate, 1.0)
        self.assertEqual(result.mean_action_efficiency, 1.0)

    def test_immediate_finish_does_not_count_as_task_success(self) -> None:
        result = asyncio.run(evaluate_agent_model(lambda case: ImmediateFinishModel()))
        self.assertEqual(result.runner_finish_rate, 1.0)
        self.assertEqual(result.runner_ok_rate, 1.0)
        self.assertEqual(result.strict_task_success_rate, 0.0)
        self.assertEqual(result.tool_selection_accuracy, 0.0)
        self.assertEqual(result.argument_accuracy, 0.0)
        self.assertEqual(result.sequence_exact_match_rate, 0.0)

    def test_wrong_grounded_coordinate_separates_tool_and_argument_accuracy(self) -> None:
        result = asyncio.run(evaluate_agent_model(lambda case: WrongCoordinateModel(case)))
        self.assertEqual(result.tool_selection_accuracy, 1.0)
        self.assertEqual(result.sequence_exact_match_rate, 1.0)
        self.assertEqual(result.tool_execution_success_rate, 1.0)
        self.assertEqual(result.runner_finish_rate, 1.0)
        self.assertEqual(result.runner_ok_rate, 1.0)
        self.assertAlmostEqual(result.argument_accuracy, 4.0 / 5.0)
        self.assertEqual(result.arguments_exact_match_rate, 0.0)
        self.assertEqual(result.strict_task_success_rate, 0.0)

    def test_extra_action_is_penalized_even_when_all_tools_execute(self) -> None:
        result = asyncio.run(evaluate_agent_model(lambda case: ExtraActionModel(case), max_steps=8))
        self.assertEqual(result.tool_execution_success_rate, 1.0)
        self.assertEqual(result.runner_finish_rate, 1.0)
        self.assertEqual(result.runner_ok_rate, 1.0)
        self.assertEqual(result.sequence_exact_match_rate, 0.0)
        self.assertEqual(result.strict_task_success_rate, 0.0)
        self.assertAlmostEqual(result.mean_action_efficiency, 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
