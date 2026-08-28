from __future__ import annotations

import asyncio
import unittest
from collections import defaultdict, deque

from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    SkillRequest,
    SkillResult,
)
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FakeMetricRobot(Embodiment):
    def __init__(self, outcomes: dict[str, list[bool | Exception]]) -> None:
        self.name = "metricbot"
        self.backend = "fake-metric-runtime"
        self._capabilities = frozenset({Capability.OBSERVE, Capability.STAND, Capability.WALK})
        self.outcomes = {
            skill: deque(items)
            for skill, items in outcomes.items()
        }
        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.calls: dict[str, int] = defaultdict(int)

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def connect(self) -> None:
        self.connected = True
        self.connect_count += 1

    async def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    async def observe(self) -> Observation:
        return Observation(self.name, {"connected": self.connected})

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if not self.connected:
            raise RuntimeError("robot is not connected")
        self.calls[request.name] += 1
        outcome = self.outcomes[request.name].popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return SkillResult(
            self.name,
            request.name,
            outcome,
            "ok" if outcome else "controller rejected command",
            {"params": dict(request.params)},
        )


class SkillMetricTests(unittest.TestCase):
    def test_mixed_outcomes_report_success_and_latency_independently(self) -> None:
        robot = FakeMetricRobot(
            {
                "stand": [True, False, RuntimeError("policy timeout"), True],
            }
        )
        clock = FakeClock(
            [
                0.00, 0.01,
                0.02, 0.04,
                0.05, 0.08,
                0.09, 0.13,
            ]
        )

        result = asyncio.run(
            benchmark_robot_skills(
                robot,
                [SkillProbe("stand", attempts=4)],
                clock=clock,
            )
        )

        self.assertEqual(robot.connect_count, 1)
        self.assertEqual(robot.disconnect_count, 1)
        self.assertFalse(robot.connected)
        self.assertEqual(result.attempt_count, 4)
        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.success_rate, 0.5)
        self.assertAlmostEqual(result.mean_latency_ms, 25.0)

        metric = result.metrics[0]
        self.assertEqual(metric.label, "stand")
        self.assertEqual(metric.successes, 2)
        self.assertEqual(metric.success_rate, 0.5)
        self.assertAlmostEqual(metric.mean_latency_ms, 25.0)
        self.assertAlmostEqual(metric.p50_latency_ms, 25.0)
        self.assertAlmostEqual(metric.p95_latency_ms, 38.5)
        self.assertAlmostEqual(metric.max_latency_ms, 40.0)
        self.assertAlmostEqual(metric.successful_mean_latency_ms or 0.0, 25.0)
        self.assertFalse(metric.samples[1].ok)
        self.assertEqual(metric.samples[1].detail, "controller rejected command")
        self.assertFalse(metric.samples[2].ok)
        self.assertIn("RuntimeError: policy timeout", metric.samples[2].error)

    def test_multiple_skills_use_weighted_aggregate_attempt_metrics(self) -> None:
        robot = FakeMetricRobot(
            {
                "stand": [True, True],
                "walk_velocity": [False],
            }
        )
        clock = FakeClock(
            [
                0.00, 0.01,
                0.02, 0.04,
                0.05, 0.08,
            ]
        )

        result = asyncio.run(
            benchmark_robot_skills(
                robot,
                [
                    SkillProbe("stand", attempts=2, label="stable stand"),
                    SkillProbe(
                        "walk_velocity",
                        {"lin_x_mps": 0.1, "duration_s": 1.0},
                        attempts=1,
                    ),
                ],
                clock=clock,
            )
        )

        self.assertEqual(result.attempt_count, 3)
        self.assertEqual(result.success_count, 2)
        self.assertAlmostEqual(result.success_rate, 2 / 3)
        self.assertAlmostEqual(result.mean_latency_ms, 20.0)
        self.assertEqual(result.metrics[0].label, "stable stand")
        self.assertEqual(result.metrics[1].arguments["duration_s"], 1.0)
        self.assertIsNone(result.metrics[1].successful_mean_latency_ms)
        self.assertEqual(robot.calls["stand"], 2)
        self.assertEqual(robot.calls["walk_velocity"], 1)

        payload = result.to_dict()
        self.assertEqual(payload["robot"], "metricbot")
        self.assertEqual(payload["backend"], "fake-metric-runtime")
        self.assertEqual(payload["probe_count"], 2)
        self.assertEqual(len(payload["metrics"][0]["samples"]), 2)

    def test_state_hooks_run_outside_timing_and_cleanup_after_skill_exception(self) -> None:
        robot = FakeMetricRobot(
            {"stand": [RuntimeError("controller timeout"), True]}
        )
        events: list[tuple[str, int, str]] = []

        async def before_attempt(robot, probe, attempt) -> None:
            events.append(("before", attempt, probe.skill))

        async def after_attempt(robot, probe, attempt) -> None:
            events.append(("after", attempt, probe.skill))

        result = asyncio.run(
            benchmark_robot_skills(
                robot,
                [SkillProbe("stand", attempts=2)],
                clock=FakeClock([0.00, 0.01, 0.02, 0.04]),
                before_attempt=before_attempt,
                after_attempt=after_attempt,
            )
        )

        self.assertEqual(
            events,
            [
                ("before", 1, "stand"),
                ("after", 1, "stand"),
                ("before", 2, "stand"),
                ("after", 2, "stand"),
            ],
        )
        self.assertEqual(robot.calls["stand"], 2)
        self.assertEqual(result.success_rate, 0.5)
        self.assertAlmostEqual(result.mean_latency_ms, 15.0)
        self.assertIn("controller timeout", result.metrics[0].samples[0].error)

    def test_setup_hook_failure_aborts_invalid_benchmark_and_disconnects(self) -> None:
        robot = FakeMetricRobot({"stand": [True]})

        async def fail_setup(robot, probe, attempt) -> None:
            raise RuntimeError("could not establish start state")

        with self.assertRaisesRegex(RuntimeError, "start state"):
            asyncio.run(
                benchmark_robot_skills(
                    robot,
                    [SkillProbe("stand")],
                    before_attempt=fail_setup,
                )
            )
        self.assertFalse(robot.connected)
        self.assertEqual(robot.disconnect_count, 1)
        self.assertEqual(robot.calls["stand"], 0)

    def test_existing_connection_can_be_reused_without_lifecycle_mutation(self) -> None:
        robot = FakeMetricRobot({"stand": [True]})
        robot.connected = True
        result = asyncio.run(
            benchmark_robot_skills(
                robot,
                [SkillProbe("stand")],
                manage_connection=False,
                clock=FakeClock([1.0, 1.005]),
            )
        )
        self.assertTrue(robot.connected)
        self.assertEqual(robot.connect_count, 0)
        self.assertEqual(robot.disconnect_count, 0)
        self.assertEqual(result.success_rate, 1.0)
        self.assertAlmostEqual(result.mean_latency_ms, 5.0)

    def test_validation_rejects_empty_probes_and_bad_attempt_count(self) -> None:
        with self.assertRaises(ValueError):
            SkillProbe("stand", attempts=0)
        with self.assertRaises(ValueError):
            SkillProbe("   ")

        robot = FakeMetricRobot({"stand": [True]})
        with self.assertRaises(ValueError):
            asyncio.run(benchmark_robot_skills(robot, []))

    def test_backwards_clock_is_rejected(self) -> None:
        robot = FakeMetricRobot({"stand": [True]})
        with self.assertRaises(ValueError):
            asyncio.run(
                benchmark_robot_skills(
                    robot,
                    [SkillProbe("stand")],
                    clock=FakeClock([2.0, 1.0]),
                )
            )
        self.assertFalse(robot.connected)
        self.assertEqual(robot.disconnect_count, 1)


if __name__ == "__main__":
    unittest.main()
