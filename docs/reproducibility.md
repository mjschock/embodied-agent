# Simulator reproducibility metrics

`embodied_agent.evals.reproducibility` measures whether repeated reset-conditioned simulator episodes produce the same task-relevant outputs.

## What is compared

Attempt 1 is the baseline. Later attempts are compared recursively:

- booleans, integers, strings, mapping keys, and sequence lengths must match exactly;
- floating-point values may differ only within an explicit absolute tolerance (`atol`);
- mismatches are reported by path;
- each comparison reports the maximum absolute numeric error.

Wall-clock latency is intentionally excluded because hosted-runner scheduling and compute performance are not simulator determinism properties.

## XLeRobot pinned result

The pinned XLeRobot MuJoCo probe repeats this episode three times:

```text
reset
navigate_to(x=0.10 m, y=-0.06 m, yaw=0.12 rad, max_duration_s=4.0)
```

The payload includes semantic success, final `(x, y, yaw)`, position/yaw errors, and timeout-source metadata. At `atol=1e-10`, both comparisons matched the baseline with **max absolute error 0.0**. The selected payload was bit-for-bit identical across all three episodes.

## Crazyflie pinned result and discovered reset bug

The pinned Crazyflie PyBullet probe repeats:

```text
reset(seed=123)
takeoff(0.4 m)
goto(0.15 m, 0.10 m, 0.4 m)
```

It compares semantic success, exact controller step counts, final positions, and position errors.

The first run revealed a real episode-reset defect. Although every episode succeeded, repeated identically seeded runs drifted by millimeters and the `goto` controller terminated at **71, 69, and 67 steps**. The reproducibility rate was 0.0.

Investigation of the pinned upstream runtime showed that `VelocityAviary` owns persistent `DSLPIDControl` instances. `BaseAviary.reset()` rebuilds the PyBullet world but does not clear controller integral/previous-error state. `DSLPIDControl` exposes a public `reset()` specifically to clear that state.

`CrazyfliePyBullet.reset` now resets both the environment and each available integrated controller. With that fix, the same probe at the unchanged `atol=1e-7` achieved **reproducibility rate 1.0**, and both comparisons had **max absolute error 0.0**. Takeoff stayed at exactly **96 steps** and `goto` at exactly **71 steps** in all three episodes.

This is why reproducibility is evaluated separately from reliability: the original Crazyflie reliability suite was 9/9 successful even while hidden controller state was leaking across resets.

## Current scope

Pinned reproducibility probes currently cover XLeRobot and Crazyflie. Microduck and LeRobot Humanoid are tracked as the next extension; their stable payloads need to exclude scheduler-dependent fields such as asynchronous simulation step counters while still detecting meaningful policy/controller state leakage.
