# Simulator reset and seed boundary

All current physics-backed simulator adapters expose a semantic `reset` operation, but randomness is deliberately kept separate from high-level agent planning.

## Agent-facing contract

`reset` is an allowlisted safe semantic tool with no agent-visible parameters. A high-level AgentModel can request a simulator reset when that tool is enabled for an embodiment, but it cannot choose random seeds.

The default all-simulation config exposes reset for XLeRobot, Crazyflie, and LeRobot Humanoid. Microduck also implements semantic reset in its simulator-specific configuration.

## Backend behavior

- **XLeRobot MuJoCo** resets the upstream MuJoCo model to its deterministic origin/reference state.
- **LeRobot Humanoid MuJoCo** delegates to the official controller reset and requests a fresh state snapshot.
- **Microduck MuJoCo** resets to the upstream `STAND` keyframe and restores standing-policy state.
- **Crazyflie PyBullet** calls `VelocityAviary.reset(seed=...)` in place, resets the environment's integrated flight controllers, and refreshes the current observation.

Crazyflie is currently the backend with an explicit RNG seed in the adapter contract. Its configured seed is used by parameterless reset. Eval/test code may call `robot.execute("reset", seed=N)` with a 32-bit unsigned integer; the supplied seed becomes the seed used by subsequent parameterless resets.

### Crazyflie controller-state reset

The pinned `VelocityAviary` creates persistent `DSLPIDControl` objects. Upstream `BaseAviary.reset()` rebuilds the PyBullet world but does not reset those controller objects. Because the controllers retain previous and integral position/attitude errors, a world-only reset is not a complete episode boundary.

The adapter therefore invokes each available integrated controller's public `reset()` immediately after environment reset. This was discovered by the reproducibility eval: before the fix, three identically seeded takeoff+translation episodes all succeeded but drifted by millimeters and terminated the translation at 71, 69, and 67 control steps. After controller reset was included, all selected episode fields and step counts were identical across all three attempts.

## Why seeds are not agent tools

Seed selection changes an experiment, not the semantic objective of a robot. Exposing it to the LLM would let planning decisions manipulate evaluation conditions and would create a tool parameter with no physical-hardware analogue. Seed control therefore remains programmatic infrastructure for tests, benchmarks, and reproducibility runs.

## Reproducibility

Reset support establishes the precondition for reproducibility metrics, but does not by itself prove deterministic outcomes. Reproducibility evals compare repeated reset-conditioned episodes using task-relevant outputs such as success/failure, controller step counts, final poses, and other stable state fields. Wall-clock latency is deliberately excluded from determinism scoring.
