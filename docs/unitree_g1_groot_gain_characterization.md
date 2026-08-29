# Unitree G1 GR00T knee-gain characterization

This note records the reset-conditioned actuator-gain A/B added after the untethered GR00T locomotion characterization established that the normalized command reaches the walk policy, Unitree `lowcmd`, and MuJoCo lower-body joints without producing meaningful body translation.

## Executable evidence

Executable head: `31b7ec33cb32dc8e694471804e8f27778ccdb9da`

All 13 project workflows passed on that exact evidence-bearing head. The `unitree-g1-groot-locomotion` run also uploaded the exact A/B JSON as an Actions artifact:

```text
groot-gain-ab_stock-fwd-0.001455_ref-fwd-0.002248_stock-drift-0.007941_ref-drift-0.011849_stock-ratio-0.241377_ref-ratio-0.212696
```

Artifact digest:

```text
sha256:6fcba81b451d1ec4cd10480b8d730c0d5cfab28c7555d8f9cfbeffc790d0a1dc
```

## Hypothesis

Pinned LeRobot v0.6.1 publishes the following proportional gains for the 15 GR00T-controlled lower-body joints:

```text
[150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 250, 250, 250]
```

A newer upstream G1/GR00T gain profile uses `200` rather than `300` at the left/right knee indices while keeping the tested lower-body damping profile unchanged. The experiment asked whether this isolated knee-stiffness difference explains the previously measured no-motion compatibility gap.

The test changes gains only inside the characterization process. It verifies on every steady-state sample that the requested `kp`/`kd` values actually reached Unitree `lowcmd`, then restores the native gains. No production adapter/controller constants are changed.

## Method

The official EnvHub simulation remains untethered (`ENABLE_ELASTIC_BAND=false`). Each episode performs:

```text
semantic reset
-> semantic stand
-> 1.0 s pre-command world-pose sampling
-> internal normalized remote.ly = 0.50
-> 2.0 s world-pose / lower-body sampling
-> semantic stand
-> 0.5 s post-command sampling
```

To reduce the chance that asynchronous reset/physics phase is mistaken for a gain effect, four episodes run in balanced crossover order:

```text
LeRobot 300 -> reference 200 -> reference 200 -> LeRobot 300
```

## Results

### Aggregate

| gain profile | knee Kp | mean forward displacement (m) | max absolute forward displacement (m) | mean pre-command drift (m) | mean motion / pre-drift ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| LeRobot v0.6.1 | 300 | +0.001455 | 0.003607 | 0.007941 | 0.241377 |
| newer upstream reference | 200 | +0.002248 | 0.004685 | 0.011849 | 0.212696 |

Lowering knee Kp from `300` to `200` increased the raw mean forward displacement by only about `0.79 mm`, while mean pre-command drift increased by about `3.91 mm`. The normalized motion-to-drift ratio therefore became slightly worse rather than better.

### Per episode

| order | profile | forward displacement (m) | pre-drift (m) | motion / pre-drift | max tilt (rad) | lower-body dq RMS (rad/s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | LeRobot 300 | -0.000698 | 0.006080 | 0.114746 | 0.085444 | 2.515649 |
| 2 | reference 200 | -0.000189 | 0.012272 | 0.015375 | 0.100738 | 1.864215 |
| 3 | reference 200 | +0.004685 | 0.011426 | 0.410017 | 0.100560 | 2.166202 |
| 4 | LeRobot 300 | +0.003607 | 0.009802 | 0.368008 | 0.064941 | 1.924284 |

Both profiles produced one tiny negative and one tiny positive forward episode. In every episode, absolute commanded forward displacement remained below that episode's reset-conditioned pre-command drift. Lower-body joint activity remained material in both profiles.

## Decision

The knee-gain hypothesis is **falsified as a locomotion fix** for this pinned stack. Knee Kp `200` does not produce repeatable body motion above the simulator's own drift and is therefore not adopted as a production override.

This result narrows the unresolved boundary further: normalized input, GR00T policy selection/inference, controller targets, `lowcmd`, and lower-body joint actuation have all been demonstrated, while useful base translation has not. The next investigation should focus on simulator/model/contact compatibility rather than exposing normalized axes or tuning gains speculatively.

G1 `WALK` remains unadvertised. No SI locomotion calibration and no physical-G1 locomotion claim are made.
