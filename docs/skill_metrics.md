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

## State-conditioned probes

Stateful skills such as navigation and flight need comparable start conditions before every measured attempt. Otherwise attempt 1 can do real work while later attempts begin already at the destination.

`benchmark_robot_skills` therefore accepts optional async `before_attempt` and `after_attempt` hooks. Each receives `(robot, probe, attempt_number)` and runs **outside** the measured interval. This makes it possible to reset or reposition an embodiment without contaminating the semantic skill latency:

```python
async def reset_before_attempt(robot, probe, attempt):
    result = await robot.execute("reset")
    if not result.ok:
        raise RuntimeError(result.detail)

result = await benchmark_robot_skills(
    robot,
    [SkillProbe("navigate_to", {"x_m": 1.0, "y_m": 0.0}, attempts=5)],
    before_attempt=reset_before_attempt,
)
```

A setup or cleanup hook failure aborts the benchmark because the requested test condition was not established. It is **not** scored as a failure of the measured robot skill. By contrast, an exception raised by the measured skill itself is captured as a failed attempt, and `after_attempt` is still run.

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

## Pinned Microduck sample

The first physics-backed application runs inside the pinned `microduck-physics` workflow against the real Microduck MuJoCo/ONNX adapter. It measures three attempts each of `stand`, left kick, right kick, and learned `roll`.

On the successful PR #27 validation run, all **12/12 semantic attempts succeeded**. The wall-clock measurements on that GitHub Actions CPU runner were:

| Probe | Success | Mean ms | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| stand | 3/3 | 15.94 | 15.91 | 16.29 | 16.33 |
| kick-left | 3/3 | 17.25 | 17.76 | 17.96 | 17.98 |
| kick-right | 3/3 | 16.06 | 15.59 | 17.05 | 17.21 |
| roll | 3/3 | 49.24 | 48.03 | 57.58 | 58.64 |

Aggregate success rate was **1.0** and aggregate mean wall-clock latency was **24.62 ms** across 12 attempts.

These latency numbers are **accelerated-simulation compute time, not physical action duration**. For example, the kick policy represents roughly 0.5 seconds of simulated behavior and roll may cover up to 3 seconds of simulated behavior, yet MuJoCo/ONNX executes those simulated control steps faster than real time on the CI CPU. The values are useful for tracking simulator/runtime regressions on comparable runners, not for claiming how long a physical Microduck would take to move.

## Pinned LeRobot Humanoid sample

The pinned `humanoid-physics` workflow does not configure a locomotion policy and explicitly verifies that `Capability.WALK` is unavailable. Its reliability sample therefore measures only the semantic operations genuinely supported by that environment: five `stand` calls and three safe simulator `reset` calls.

The controller thread is allowed to warm up before timing, and the benchmark reuses one connection so startup/checkout cost is excluded. On the successful PR #28 validation run, all **8/8 semantic attempts succeeded**:

| Probe | Success | Mean ms | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| stand | 5/5 | 0.233 | 0.208 | 0.334 | 0.362 |
| reset | 3/3 | 0.318 | 0.303 | 0.358 | 0.365 |

Aggregate success rate was **1.0** and aggregate mean semantic-call latency was **0.265 ms** across 8 attempts.

These numbers primarily measure the adapter/controller request-and-snapshot path in fixed-base MuJoCo. They do **not** represent human-scale standing or reset movement duration, and they do not establish learned locomotion performance because no locomotion policy is loaded in this workflow.

## Pinned XLeRobot navigation sample

XLeRobot navigation is stateful, so the pinned benchmark uses the state-conditioning hook to execute an unmeasured `reset` before every timed attempt. The hook also verifies that reset returned the base to `(x, y, yaw) = (0, 0, 0)` before the benchmark clock can start.

The measured task is the same short pose already exercised by the upstream-backed smoke test:

```text
navigate_to(x=0.10 m, y=-0.06 m, yaw=0.12 rad, max_duration=4.0 s)
```

On the successful PR #30 validation run, all **3/3 reset-conditioned navigations succeeded**:

| Probe | Success | Mean ms | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| navigate-short-pose | 3/3 | 13.51 | 13.37 | 13.79 | 13.84 |

The three raw measured samples were approximately 13.84 ms, 13.37 ms, and 13.31 ms. Reset/setup latency is deliberately excluded.

These values are accelerated MuJoCo/controller compute latency for a fixed simulation task, not the elapsed duration a physical mobile base would require to drive the requested pose. The explicit 4-second value is a simulation-time safety budget, not the measured wall-clock execution time.

## Pinned Crazyflie flight sample

Crazyflie flight is also stateful. The pinned benchmark reconstructs and re-seeds `VelocityAviary` before **every** measured attempt, then verifies the initial position is `(0, 0, 0.1)` m. For `goto` and `land`, it additionally performs a validated, unmeasured takeoff to 0.4 m before the benchmark clock starts. This ensures the three samples for each skill begin from comparable simulator states.

On the successful PR #31 validation run, all **9/9 semantic attempts succeeded**:

| Probe | Success | Mean ms | p50 ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| takeoff | 3/3 | 47.44 | 46.85 | 48.59 | 48.79 |
| goto-short-translation | 3/3 | 34.43 | 34.39 | 34.61 | 34.64 |
| land | 3/3 | 42.21 | 42.28 | 42.33 | 42.33 |

Aggregate success rate was **1.0** and aggregate mean measured latency was **41.36 ms** across 9 attempts. Raw takeoff samples were approximately 48.79, 46.68, and 46.85 ms; short-translation samples were 34.64, 34.39, and 34.27 ms; landing samples were 42.04, 42.33, and 42.28 ms.

Environment construction, deterministic seeding, precondition takeoff, and validation observations are all excluded from the measured interval. As with the other simulator samples, these values describe accelerated PyBullet/controller compute time for the pinned task—not physical flight duration.

## Timing interpretation

The CI gates deliberately enforce reliability and structural timing invariants rather than absolute millisecond ceilings, because hosted-runner performance can vary. Physics compute latency, simulated physical duration, and eventual real-hardware wall-clock duration are separate quantities and should not be conflated.

## Applying this to physics and hardware

The metric layer intentionally does not define universal pass/fail latency thresholds. Different skills have different physical durations: `microduck.roll`, a Crazyflie transit, and `humanoid.stand` should not be compared as if they were equivalent operations.

Physics- and hardware-specific evals should therefore choose explicit probes, attempt counts, reset/precondition behavior, and acceptance thresholds appropriate to each skill while reusing this common result format.
