from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from embodied_agent.core import Embodiment

Clock = Callable[[], float]
AttemptHook = Callable[[Embodiment, "SkillProbe", int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SkillProbe:
    """One semantic skill invocation shape to measure repeatedly."""

    skill: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if not self.skill.strip():
            raise ValueError("skill must be non-empty")
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")

    @property
    def display_name(self) -> str:
        return self.label.strip() or self.skill


@dataclass(frozen=True, slots=True)
class SkillAttemptResult:
    attempt: int
    ok: bool
    latency_ms: float
    detail: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class SkillMetricResult:
    label: str
    skill: str
    arguments: Mapping[str, Any]
    attempts: int
    successes: int
    success_rate: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    successful_mean_latency_ms: float | None
    samples: tuple[SkillAttemptResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "skill": self.skill,
            "arguments": dict(self.arguments),
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "successful_mean_latency_ms": self.successful_mean_latency_ms,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class SkillBenchmarkResult:
    robot: str
    backend: str
    metrics: tuple[SkillMetricResult, ...]
    attempt_count: int
    success_count: int
    success_rate: float
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "backend": self.backend,
            "probe_count": len(self.metrics),
            "attempt_count": self.attempt_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linearly interpolated percentile on an already measured sample set."""
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


async def _measure_probe(
    robot: Embodiment,
    probe: SkillProbe,
    *,
    clock: Clock,
    before_attempt: AttemptHook | None,
    after_attempt: AttemptHook | None,
) -> SkillMetricResult:
    samples: list[SkillAttemptResult] = []

    for attempt in range(1, probe.attempts + 1):
        if before_attempt is not None:
            await before_attempt(robot, probe, attempt)

        started = clock()
        result = None
        error = ""
        try:
            result = await robot.execute(probe.skill, **dict(probe.arguments))
        except Exception as exc:  # measurement boundary: exceptions are failed attempts
            error = f"{type(exc).__name__}: {exc}"
        finally:
            finished = clock()
            if after_attempt is not None:
                await after_attempt(robot, probe, attempt)

        latency_ms = (finished - started) * 1000.0
        if latency_ms < 0:
            raise ValueError("clock moved backwards while measuring skill latency")

        samples.append(
            SkillAttemptResult(
                attempt=attempt,
                ok=bool(result.ok) if result is not None else False,
                latency_ms=latency_ms,
                detail=result.detail if result is not None else "",
                error=error,
            )
        )

    latencies = [sample.latency_ms for sample in samples]
    successful_latencies = [sample.latency_ms for sample in samples if sample.ok]
    successes = len(successful_latencies)
    return SkillMetricResult(
        label=probe.display_name,
        skill=probe.skill,
        arguments=dict(probe.arguments),
        attempts=len(samples),
        successes=successes,
        success_rate=successes / len(samples),
        mean_latency_ms=fmean(latencies),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        max_latency_ms=max(latencies),
        successful_mean_latency_ms=(
            fmean(successful_latencies) if successful_latencies else None
        ),
        samples=tuple(samples),
    )


async def benchmark_robot_skills(
    robot: Embodiment,
    probes: Sequence[SkillProbe],
    *,
    manage_connection: bool = True,
    clock: Clock = time.perf_counter,
    before_attempt: AttemptHook | None = None,
    after_attempt: AttemptHook | None = None,
) -> SkillBenchmarkResult:
    """Measure semantic-skill reliability and latency for one embodiment.

    This layer intentionally bypasses AgentModel/MCP planning so the resulting
    metrics describe the robot/backend skill boundary itself. Failed
    ``SkillResult`` values and raised exceptions both count as failed attempts;
    latency includes all attempts so timeouts remain visible instead of being
    silently removed from timing statistics.

    ``before_attempt`` and ``after_attempt`` are optional state-conditioning
    hooks for skills such as navigation or flight. They run outside the measured
    interval, so reset/setup/cleanup time does not contaminate semantic-skill
    latency. Hook failures abort the benchmark because the requested test
    precondition or cleanup was not established; they are not scored as robot
    skill failures.
    """
    selected = tuple(probes)
    if not selected:
        raise ValueError("at least one skill probe is required")

    metrics: list[SkillMetricResult] = []
    connected_here = False
    try:
        if manage_connection:
            await robot.connect()
            connected_here = True
        for probe in selected:
            metrics.append(
                await _measure_probe(
                    robot,
                    probe,
                    clock=clock,
                    before_attempt=before_attempt,
                    after_attempt=after_attempt,
                )
            )
    finally:
        if connected_here:
            await robot.disconnect()

    attempt_count = sum(metric.attempts for metric in metrics)
    success_count = sum(metric.successes for metric in metrics)
    all_latencies = [
        sample.latency_ms
        for metric in metrics
        for sample in metric.samples
    ]
    return SkillBenchmarkResult(
        robot=robot.name,
        backend=robot.backend,
        metrics=tuple(metrics),
        attempt_count=attempt_count,
        success_count=success_count,
        success_rate=success_count / attempt_count,
        mean_latency_ms=fmean(all_latencies),
    )
