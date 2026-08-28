# Single-robot skill reliability metrics

`embodied_agent.evals.skill_metrics` measures the semantic skill boundary of one `Embodiment` without involving an AgentModel, MCP planning, or multi-robot task scoring.

This keeps two questions separate:

1. **Did the high-level agent choose the right action?** — answered by the AgentModel benchmarks.
2. **How reliably and quickly did the selected robot skill execute?** — answered by this benchmark.

## Probe model

A `SkillProbe` describes one semantic skill invocation plus the number of attempts to measure. For example:

```python
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills

result = await benchmark_robot_skills(
    robot,
    [
        SkillProbe("stand", attempts=10),
        SkillProbe(
            "walk_velocity",
            {"lin_x_mps": 0.1, "duration_s": 1.0},
            attempts=10,
        ),
    ],
)
```

The benchmark can own `connect()` / `disconnect()` or reuse a caller-owned connection with `manage_connection=False`.

## Per-skill metrics

Each probe reports:

- attempt count;
- success count and success rate;
- mean latency across **all** attempts;
- p50 latency;
- p95 latency;
- maximum latency;
- mean latency across successful attempts only;
- individual samples with `ok`, latency, detail, and exception text where applicable.

Failed `SkillResult` values and raised exceptions both count as failed attempts. Their latency remains in the all-attempt timing distribution so timeouts or slow failures cannot disappear from the benchmark.

`successful_mean_latency_ms` is separate because it answers a different question: how fast the skill is when it actually succeeds.

## Aggregate metrics

`SkillBenchmarkResult` reports robot/backend identity, total attempts, total successes, aggregate success rate, and attempt-weighted mean latency across all probes.

## Percentiles

p50 and p95 use linear interpolation over the measured sample latencies. Unit tests inject a deterministic clock, so the metric contract is not dependent on scheduler timing or real sleeps.

## Applying this to physics and hardware

The metric layer intentionally does not define universal pass/fail latency thresholds. Different skills have different physical durations: `microduck.roll`, a Crazyflie transit, and `humanoid.stand` should not be compared as if they were equivalent operations.

Physics- and hardware-specific evals should therefore choose explicit probes, attempt counts, reset/precondition behavior, and acceptance thresholds appropriate to each skill while reusing this common result format.
