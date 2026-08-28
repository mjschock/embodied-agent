from __future__ import annotations

import asyncio
import unittest

from embodied_agent.evals.agent_model import ExpectedActionModel
from embodied_agent.evals.openai_four_embodiment import (
    BENCHMARK_NAME,
    build_openai_four_embodiment_record,
    evaluate_four_embodiment_model,
)


class OpenAIFourEmbodimentEvalTests(unittest.TestCase):
    def test_provider_agnostic_wiring_scores_oracle_perfectly(self) -> None:
        payload = asyncio.run(
            evaluate_four_embodiment_model(
                lambda case: ExpectedActionModel(case),
                max_steps=10,
            )
        )
        self.assertEqual(payload["case_count"], 7)
        self.assertEqual(payload["tool_selection_accuracy"], 1.0)
        self.assertEqual(payload["argument_accuracy"], 1.0)
        self.assertEqual(payload["tool_execution_success_rate"], 1.0)
        self.assertEqual(payload["strict_task_success_rate"], 1.0)
        self.assertEqual(payload["mean_action_efficiency"], 1.0)

    def test_result_record_is_self_describing(self) -> None:
        payload = {
            "case_count": 7,
            "strict_task_success_rate": 0.75,
            "tool_selection_accuracy": 0.9,
        }
        record = build_openai_four_embodiment_record(
            model_name="example-model",
            max_steps=10,
            payload=payload,
        )
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["benchmark"], BENCHMARK_NAME)
        self.assertEqual(record["provider"], "openai")
        self.assertEqual(record["model"], "example-model")
        self.assertEqual(record["max_steps"], 10)
        self.assertEqual(record["result"], payload)
        self.assertEqual(
            record["environment"]["robot_backend"],
            "scripted-four-embodiment-eval",
        )
        self.assertTrue(record["environment"]["python_version"])


if __name__ == "__main__":
    unittest.main()
