from __future__ import annotations

import asyncio
import unittest

from embodied_agent.evals.agent_model import ExpectedActionModel, evaluate_agent_model
from embodied_agent.evals.four_embodiment import (
    default_four_embodiment_cases,
    evaluate_expected_four_embodiment_agent,
    scripted_four_embodiment_stack,
)
from embodied_agent.mcp import AgentDecision


class SharedCapabilityConfusionModel(ExpectedActionModel):
    """Routes humanoid STAND/WALK to Microduck even though both tools are valid."""

    async def decide(self, context):
        index = len(context.history)
        if index >= len(self.expected):
            return AgentDecision.finish("reference sequence complete")
        action = self.expected[index]
        if action.tool == "humanoid.stand":
            return AgentDecision.call("microduck.stand", action.arguments)
        if action.tool == "humanoid.walk_velocity":
            return AgentDecision.call("microduck.walk_velocity", action.arguments)
        return AgentDecision.call(action.tool, action.arguments)


class FourEmbodimentEvalTests(unittest.TestCase):
    def test_oracle_baseline_is_perfect(self) -> None:
        result = asyncio.run(evaluate_expected_four_embodiment_agent())
        self.assertEqual(len(result.cases), 7)
        self.assertEqual(result.tool_selection_accuracy, 1.0)
        self.assertEqual(result.argument_accuracy, 1.0)
        self.assertEqual(result.sequence_exact_match_rate, 1.0)
        self.assertEqual(result.arguments_exact_match_rate, 1.0)
        self.assertEqual(result.tool_execution_success_rate, 1.0)
        self.assertEqual(result.runner_finish_rate, 1.0)
        self.assertEqual(result.runner_ok_rate, 1.0)
        self.assertEqual(result.strict_task_success_rate, 1.0)
        self.assertEqual(result.mean_action_efficiency, 1.0)

    def test_shared_capability_misrouting_executes_but_loses_selection_credit(self) -> None:
        cases = default_four_embodiment_cases()
        result = asyncio.run(
            evaluate_agent_model(
                lambda case: SharedCapabilityConfusionModel(case),
                cases=cases,
                max_steps=10,
                stack=scripted_four_embodiment_stack(),
            )
        )

        # Both Microduck substitutions are schema-valid and execute successfully,
        # so execution-only scoring would miss the embodiment-selection error.
        self.assertEqual(result.tool_execution_success_rate, 1.0)
        self.assertLess(result.tool_selection_accuracy, 1.0)
        self.assertLess(result.sequence_exact_match_rate, 1.0)
        self.assertLess(result.strict_task_success_rate, 1.0)

        humanoid_stand = next(case for case in result.cases if case.name == "four-humanoid-stand")
        humanoid_walk = next(case for case in result.cases if case.name == "four-humanoid-walk")
        combined = next(
            case for case in result.cases if case.name == "four-all-embodiments-mission"
        )
        self.assertEqual(humanoid_stand.tool_selection_accuracy, 0.0)
        self.assertEqual(humanoid_walk.tool_selection_accuracy, 0.0)
        self.assertFalse(humanoid_stand.strict_task_success)
        self.assertFalse(humanoid_walk.strict_task_success)
        self.assertFalse(combined.strict_task_success)

    def test_tool_surface_contains_overlap_and_microduck_unique_behaviors(self) -> None:
        _, router = scripted_four_embodiment_stack()
        names = {tool["name"] for tool in router.list_tools()}

        self.assertIn("humanoid.stand", names)
        self.assertIn("microduck.stand", names)
        self.assertIn("humanoid.walk_velocity", names)
        self.assertIn("microduck.walk_velocity", names)
        self.assertIn("microduck.kick", names)
        self.assertIn("microduck.roll", names)
        self.assertNotIn("humanoid.kick", names)
        self.assertNotIn("humanoid.roll", names)

    def test_combined_mission_reaches_all_four_embodiments(self) -> None:
        combined = next(
            case
            for case in default_four_embodiment_cases()
            if case.name == "four-all-embodiments-mission"
        )
        robots = {tool.split(".", 1)[0] for tool in combined.expected_tools}
        self.assertEqual(robots, {"crazyflie", "xlerobot", "humanoid", "microduck"})


if __name__ == "__main__":
    unittest.main()
