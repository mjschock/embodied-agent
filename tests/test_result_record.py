from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embodied_agent.evals.result_record import (
    RESULT_SCHEMA_VERSION,
    build_result_record,
    default_result_path,
    slugify,
    write_result_record,
)


class ResultRecordTests(unittest.TestCase):
    def test_build_result_record_is_self_describing(self) -> None:
        record = build_result_record(
            benchmark="three-robot-physics-comparison",
            provider="openai",
            model="example/model",
            max_steps=8,
            created_at="2026-08-28T03:30:00Z",
            environment={"repository_revision": "abc123"},
            result={"agent": {"strict_task_success_rate": 2.0 / 3.0}},
        )
        self.assertEqual(record["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(record["benchmark"], "three-robot-physics-comparison")
        self.assertEqual(record["provider"], "openai")
        self.assertEqual(record["model"], "example/model")
        self.assertEqual(record["max_steps"], 8)
        self.assertEqual(record["created_at"], "2026-08-28T03:30:00Z")
        self.assertEqual(record["environment"]["repository_revision"], "abc123")
        self.assertAlmostEqual(record["result"]["agent"]["strict_task_success_rate"], 2.0 / 3.0)

    def test_validation_rejects_missing_identity_and_invalid_step_budget(self) -> None:
        with self.assertRaises(ValueError):
            build_result_record(
                benchmark="",
                provider="openai",
                model="model",
                max_steps=8,
                result={},
            )
        with self.assertRaises(ValueError):
            build_result_record(
                benchmark="benchmark",
                provider="openai",
                model="model",
                max_steps=0,
                result={},
            )

    def test_default_result_path_is_stable_and_model_safe(self) -> None:
        path = default_result_path(
            root="eval_results",
            benchmark="three robot physics",
            model="provider/model:preview",
            created_at="2026-08-28T03:30:00Z",
        )
        self.assertEqual(
            path,
            Path("eval_results")
            / "three-robot-physics"
            / "provider-model-preview"
            / "20260828T033000Z.json",
        )
        self.assertEqual(slugify(" /// "), "unknown")

    def test_write_result_record_creates_valid_json(self) -> None:
        record = build_result_record(
            benchmark="benchmark",
            provider="openai",
            model="model",
            max_steps=8,
            created_at="2026-08-28T03:30:00Z",
            result={"value": 1},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "nested" / "result.json"
            written = write_result_record(record, destination)
            self.assertEqual(written, destination)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), record)
            self.assertFalse(destination.with_name("result.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
