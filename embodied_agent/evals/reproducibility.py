from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EpisodeRunner = Callable[[int], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class ReproducibilitySample:
    attempt: int
    is_baseline: bool
    matches_baseline: bool
    max_abs_error: float
    mismatch_paths: tuple[str, ...]
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "is_baseline": self.is_baseline,
            "matches_baseline": self.matches_baseline,
            "max_abs_error": self.max_abs_error,
            "mismatch_paths": list(self.mismatch_paths),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class ReproducibilityResult:
    label: str
    attempts: int
    comparison_count: int
    matching_comparisons: int
    reproducibility_rate: float
    atol: float
    baseline: Mapping[str, Any]
    samples: tuple[ReproducibilitySample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "attempts": self.attempts,
            "comparison_count": self.comparison_count,
            "matching_comparisons": self.matching_comparisons,
            "reproducibility_rate": self.reproducibility_rate,
            "atol": self.atol,
            "baseline": dict(self.baseline),
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class _Comparison:
    mismatch_paths: tuple[str, ...]
    max_abs_error: float

    @property
    def matches(self) -> bool:
        return not self.mismatch_paths


def _child_path(path: str, child: str) -> str:
    return child if not path else f"{path}.{child}"


def _compare_values(
    baseline: Any,
    candidate: Any,
    *,
    atol: float,
    path: str = "",
) -> _Comparison:
    mismatch_paths: list[str] = []
    max_abs_error = 0.0

    if isinstance(baseline, bool) or isinstance(candidate, bool):
        if type(baseline) is not type(candidate) or baseline != candidate:
            mismatch_paths.append(path or "$root")
        return _Comparison(tuple(mismatch_paths), max_abs_error)

    if isinstance(baseline, int) or isinstance(candidate, int):
        if type(baseline) is not type(candidate) or baseline != candidate:
            mismatch_paths.append(path or "$root")
        return _Comparison(tuple(mismatch_paths), max_abs_error)

    if isinstance(baseline, float) or isinstance(candidate, float):
        if not isinstance(baseline, (int, float)) or not isinstance(candidate, (int, float)):
            mismatch_paths.append(path or "$root")
            return _Comparison(tuple(mismatch_paths), max_abs_error)
        baseline_float = float(baseline)
        candidate_float = float(candidate)
        if not math.isfinite(baseline_float) or not math.isfinite(candidate_float):
            if baseline_float != candidate_float:
                mismatch_paths.append(path or "$root")
            return _Comparison(tuple(mismatch_paths), max_abs_error)
        max_abs_error = abs(candidate_float - baseline_float)
        if max_abs_error > atol:
            mismatch_paths.append(path or "$root")
        return _Comparison(tuple(mismatch_paths), max_abs_error)

    if isinstance(baseline, Mapping) or isinstance(candidate, Mapping):
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            return _Comparison((path or "$root",), max_abs_error)
        baseline_keys = set(baseline)
        candidate_keys = set(candidate)
        for key in sorted(baseline_keys ^ candidate_keys, key=str):
            mismatch_paths.append(_child_path(path, str(key)))
        for key in sorted(baseline_keys & candidate_keys, key=str):
            comparison = _compare_values(
                baseline[key],
                candidate[key],
                atol=atol,
                path=_child_path(path, str(key)),
            )
            mismatch_paths.extend(comparison.mismatch_paths)
            max_abs_error = max(max_abs_error, comparison.max_abs_error)
        return _Comparison(tuple(mismatch_paths), max_abs_error)

    if isinstance(baseline, Sequence) or isinstance(candidate, Sequence):
        sequence_types = (str, bytes, bytearray)
        if isinstance(baseline, sequence_types) or isinstance(candidate, sequence_types):
            if type(baseline) is not type(candidate) or baseline != candidate:
                mismatch_paths.append(path or "$root")
            return _Comparison(tuple(mismatch_paths), max_abs_error)
        if not isinstance(baseline, Sequence) or not isinstance(candidate, Sequence):
            return _Comparison((path or "$root",), max_abs_error)
        if len(baseline) != len(candidate):
            mismatch_paths.append(f"{path or '$root'}.length")
        for index, (baseline_item, candidate_item) in enumerate(zip(baseline, candidate)):
            comparison = _compare_values(
                baseline_item,
                candidate_item,
                atol=atol,
                path=f"{path}[{index}]" if path else f"[{index}]",
            )
            mismatch_paths.extend(comparison.mismatch_paths)
            max_abs_error = max(max_abs_error, comparison.max_abs_error)
        return _Comparison(tuple(mismatch_paths), max_abs_error)

    if type(baseline) is not type(candidate) or baseline != candidate:
        mismatch_paths.append(path or "$root")
    return _Comparison(tuple(mismatch_paths), max_abs_error)


async def benchmark_reproducibility(
    run_episode: EpisodeRunner,
    *,
    attempts: int = 3,
    atol: float = 1e-9,
    label: str = "sim-reproducibility",
) -> ReproducibilityResult:
    """Compare repeated reset-conditioned episode payloads to attempt 1.

    ``run_episode`` owns reset/preconditions and returns only stable, task-relevant
    fields. Wall-clock timing is intentionally absent: scheduler/runtime latency is
    not a simulator determinism criterion.
    """
    if attempts < 2:
        raise ValueError("attempts must be >= 2")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a finite non-negative number")
    if not label.strip():
        raise ValueError("label must be non-empty")

    payloads: list[Mapping[str, Any]] = []
    for attempt in range(1, attempts + 1):
        payload = await run_episode(attempt)
        if not isinstance(payload, Mapping):
            raise TypeError("run_episode must return a mapping")
        payloads.append(dict(payload))

    baseline = payloads[0]
    samples: list[ReproducibilitySample] = [
        ReproducibilitySample(
            attempt=1,
            is_baseline=True,
            matches_baseline=True,
            max_abs_error=0.0,
            mismatch_paths=(),
            payload=baseline,
        )
    ]
    matching_comparisons = 0
    for attempt, payload in enumerate(payloads[1:], start=2):
        comparison = _compare_values(baseline, payload, atol=atol)
        if comparison.matches:
            matching_comparisons += 1
        samples.append(
            ReproducibilitySample(
                attempt=attempt,
                is_baseline=False,
                matches_baseline=comparison.matches,
                max_abs_error=comparison.max_abs_error,
                mismatch_paths=comparison.mismatch_paths,
                payload=payload,
            )
        )

    comparison_count = attempts - 1
    return ReproducibilityResult(
        label=label.strip(),
        attempts=attempts,
        comparison_count=comparison_count,
        matching_comparisons=matching_comparisons,
        reproducibility_rate=matching_comparisons / comparison_count,
        atol=atol,
        baseline=baseline,
        samples=tuple(samples),
    )
