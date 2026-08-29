# Unitree G1 native LeRobot reference

`UnitreeG1LeRobot` wraps LeRobot's first-party Unitree G1 robot contract while keeping the high-level agent away from raw 29-DoF joint commands.

## Current LeRobot contract

The adapter is pinned in CI against LeRobot v0.6.1 commit `7e241bd630a3719a56157a497ce5d08f244784f1`.

That release exposes:

- `UnitreeG1Config(is_simulation=True)` for the Hub-hosted MuJoCo environment;
- 29 joint position/velocity/estimated-torque observations;
- IMU observations;
- native `reset()`;
- optional lower-body controllers including `GrootLocomotionController` and `HolosomaLocomotionController`;
- normalized remote-controller axes (`remote.lx`, `remote.ly`, `remote.rx`, `remote.ry`) when a locomotion controller owns the lower body.

## Semantic boundary

The first adapter advertises:

- `OBSERVE` always;
- `STAND` only when a lower-body controller is configured.

`reset` remains available through the common safe `reset` tool because the robot advertises `OBSERVE`.

The adapter intentionally does **not** advertise `WALK`. LeRobot's current G1 lower-body API consumes normalized joystick/controller inputs, not calibrated linear or angular velocities in SI units. Mapping those axes directly to `walk_velocity(lin_x_mps, lin_y_mps, yaw_rate_rps)` would falsely imply a physical calibration that has not been measured.

### Stand behavior

For a controller-backed G1, `stand` sends only:

```text
remote.lx = 0.0
remote.ly = 0.0
remote.rx = 0.0
remote.ry = 0.0
```

With the pinned GR00T controller, near-zero command magnitude selects the balance policy. The adapter never constructs a dictionary of 29 joint targets.

## Observation normalization

Native LeRobot observations are grouped into explicit fields:

- `joint_position_rad`
- `joint_velocity_rad_s`
- `joint_torque_est_nm`
- `imu`
- `wireless_remote` (bytes converted to an integer list for serialization)
- non-state observation features such as configured camera frames are placed in `Observation.images`.

A non-empty joint-position observation must contain all 29 joints.

## Configuration

`configs/unitree_g1.sim.example.json` exposes only the semantic tools proven by the adapter:

```json
{
  "robots": {
    "unitree_g1": {
      "adapter": "lerobot_unitree_g1",
      "params": {
        "is_simulation": true,
        "controller": "GrootLocomotionController",
        "gravity_compensation": false,
        "simulation_dds_interface": null
      },
      "tools": ["observe", "reset", "stand"]
    }
  }
}
```

For simulation, `simulation_dds_interface: null` means Unitree SDK2 interface auto-detection. This is intentional for the pinned Python 3.12 runtime; see the DDS compatibility evidence below. Physical G1 initialization is not modified by this setting.

## Pinned EnvHub physics evidence

The dedicated `unitree-g1-physics` workflow launches and steps the official Hub-hosted MuJoCo environment headlessly. It pins:

- EnvHub `lerobot/unitree-g1-mujoco` at `a38dc8617f0fca51b38e9354dc58ee35ad850fb5`;
- official `unitreerobotics/unitree_sdk2_python` source at `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`;
- CycloneDDS `0.10.2` on Python 3.10;
- MuJoCo software OpenGL through OSMesa on the CPU-only hosted runner.

The smoke performs a real reset with seed 123 and five zero-action physics steps. Every returned observation must be finite, reward remains zero, and the environment must remain non-terminated/non-truncated.

### Runtime issues discovered by the physics probe

The diagnostic bring-up found two setup issues before physics could run:

1. the published `unitree-sdk2==1.0.1` package failed while importing `unitree_sdk2py.b2`, whereas the current official source tree contains that module; the workflow therefore pins and installs the official source checkout instead;
2. EGL could import on the hosted CPU runner but had no usable EGL device, so the non-rendering smoke uses MuJoCo's OSMesa software backend instead.

Both fixes are pinned in the workflow rather than hidden in local setup instructions.

### Known upstream observation-space mismatch

The pinned EnvHub runtime exposes a 29-D action space and declares a 97-D observation space (`29 * 3 + 10`). The actual reset/step observation is **100-D**.

The physics gate records the raw component sizes explicitly:

```text
body_q              29
body_dq             29
body_tau_est        29
floating_base_pose   7
floating_base_vel    6
floating_base_acc    6
```

The environment concatenates 29 position + 29 velocity + 29 torque values, the first 4 pose values, the first 3 velocity values, and all 6 acceleration values: `87 + 4 + 3 + 6 = 100`. Its declared observation space budgets only 3 acceleration values. The test intentionally asserts both the declared `(97,)` space and the actual `(100,)` runtime result so an upstream correction becomes visible instead of silently changing our evidence.

## Full LeRobot lifecycle evidence

The separate `unitree-g1-lerobot-runtime` workflow exercises the adapter's **real default LeRobot factory**, rather than constructing the EnvHub environment directly. Its original executable evidence head is `d34eefcb3698535a3a6a04770da76e6d8ec7f014`.

The workflow pins:

- LeRobot v0.6.1 at `7e241bd630a3719a56157a497ce5d08f244784f1`;
- Unitree SDK2 Python at `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`;
- CycloneDDS C 0.10.2 at `9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd`;
- CycloneDDS Python 0.10.2 at `9cec1189a3d5a1407851dfe1f40899dd4a67f52d`;
- the official G1 EnvHub simulator at `a38dc8617f0fca51b38e9354dc58ee35ad850fb5`;
- SciPy `>=1.10,<2`, matching the independent EnvHub physics gate because the simulator's Unitree bridge imports `scipy.spatial.transform` directly.

On Python 3.12, it passes:

```text
Unitree auto-interface DDS domain
-> Unitree lowcmd/lowstate topic construction
-> LeRobot UnitreeG1 construction
-> LeRobot EnvHub loader
-> UnitreeG1LeRobot.connect()
-> observe 29 q/dq/tau joints + finite IMU state
-> native reset()
-> observe again
-> disconnect()
```

This establishes the full LeRobot/DDS/EnvHub lifecycle independently from GR00T policy loading.

### Same-process simulation reconnect lifecycle

Executable evidence head `a6fd7a7643d587834da3e7366051da424d56b6af` strengthens the full-runtime gate to execute **two consecutive complete native lifecycles in one Python process** on the same adapter instance:

```text
connect -> observe -> reset -> observe -> disconnect
connect -> observe -> reset -> observe -> disconnect
```

All 12 required project workflows passed on that exact code-bearing head.

The earlier strict GR00T diagnostic had exposed a simulator-thread/quaternion exception after disconnecting and recreating the full simulated G1 stack. Inspection of pinned LeRobot v0.6.1 showed that `UnitreeG1.disconnect()` stops its own subscriber/controller threads and closes EnvHub but does not close the Unitree SDK2 low-state subscriber or low-command publisher. The pinned SDK2 objects expose explicit `Close()` methods.

`UnitreeG1LeRobot.disconnect()` now lets native LeRobot perform its teardown first, then explicitly closes those two SDK2 endpoints **for simulation only** before releasing the native robot reference. The strengthened runtime gate proves that an immediate fresh default-factory G1 can then reconnect, observe all 29 finite q/dq/tau joint fields plus finite IMU state, execute native reset, observe again, and disconnect successfully.

Physical G1 teardown is intentionally unchanged because it has not been validated on hardware. This lifecycle compatibility fix also changes no semantic capability or MCP schema; `WALK` remains unadvertised.

## Pinned GR00T semantic stand evidence

The dedicated `unitree-g1-groot-stand` workflow proves the controller-backed semantic `STAND` path through the real LeRobot/Unitree/DDS/EnvHub stack. Its initial validated executable head is `610e6c1101148b6b066e10a6b9fcc25b29a02ab5`.

In addition to the runtime pins above, it pins `nepyope/GR00T-WholeBodyControl_g1` at `921bc56492959fa3ed0fb03cecdee14ab768eefc` and verifies SHA-256 for both the balance and walk ONNX assets before LeRobot loads them. The test invokes `UnitreeG1LeRobot.execute("stand")`, confirms all four normalized remote axes remain exactly zero, and samples 100 simulator observations over two seconds.

The validated run measured:

```text
samples                    100
max_abs_roll_rad           0.0
max_abs_pitch_rad          0.0
max_tilt_rad               0.0
mean_tilt_rad              0.0
final_roll_rad             0.0
final_pitch_rad            0.0
final_tilt_rad             0.0
controller_output_joints   15
```

All sampled joint states and controller outputs were finite. CI deliberately does **not** require bit-for-bit zero tilt: the behavioral gate allows `< 0.05 rad` (about 2.9 degrees) for both maximum and final roll/pitch tilt. This leaves a small physical tolerance while still failing a materially unstable standing episode.

### Stand reliability and reset-conditioned reproducibility

Executable evidence head `a8fc9014014d139a61e331d07b55c144adf3ef74` extends the same pinned GR00T/EnvHub gate with repeated semantic-skill measurement. All 12 required project workflows passed on that exact code-bearing head.

Five reset-conditioned `stand` attempts all succeeded, and every postcondition sample remained at `0.0 rad` maximum and final roll/pitch tilt. The semantic call latency measured:

```text
success_rate              1.0 (5/5)
mean_latency_ms           1.5291664
p50_latency_ms            0.7275590
p95_latency_ms            3.1140846
max_latency_ms            3.2118230
```

Reset and the approximately 0.5-second behavioral postcondition window are deliberately outside the timed interval, so these numbers measure the semantic `stand` request itself rather than simulator settling time.

The same connected native stack then ran three `reset -> stand` episodes through the shared reproducibility metric at `atol=1e-9`. The reproducibility contract is intentionally behavioral and semantic:

- all four normalized remote axes remain exactly zero;
- the GR00T controller command remains zero;
- all 15 lower-body controller outputs are present and finite;
- all 29 observed joint position/velocity fields are present and finite;
- maximum and final roll/pitch tilt remain within the standing envelope.

The result was `reproducibility_rate = 1.0`, `max_abs_error = 0.0`, with no mismatch paths across the two baseline comparisons.

A deliberately stricter diagnostic first compared the instantaneous values of all 29 joint positions/velocities and the 15 controller outputs across reset episodes. That hypothesis was falsified: wall-clock snapshots diverged by as much as about `9.65`, despite exact zero command and `0.0 rad` tilt in every episode. LeRobot controller inference and EnvHub physics advance on asynchronous background threads, so those instantaneous state values are phase-sensitive. The project did **not** relax the numerical tolerance to make that probe pass; instead, those fields remain checked for shape and finiteness while reproducibility is claimed only for the semantic/behavioral boundary that the simulator demonstrates.

That strict diagnostic also exposed the same-process reconnect teardown defect described above. It is now covered separately by the two-lifecycle native runtime gate rather than being hidden inside the reproducibility metric.

This evidence is simulation-only. It proves GR00T balance/standing through the semantic G1 boundary, not physical G1 execution and not calibrated locomotion.

### DDS compatibility defect and simulation-only workaround

The first full-runtime attempt exposed a native `*** buffer overflow detected ***` abort inside `dds_create_domain` when pinned LeRobot v0.6.1 called Unitree SDK2 as:

```text
ChannelFactoryInitialize(0, "lo")
```

A dedicated `unitree-g1-dds-contract` workflow reduced the issue to three process-level probes on the same Python 3.12 / CycloneDDS 0.10.2 stack:

- raw CycloneDDS `Domain(0)`: **passes**;
- Unitree `ChannelFactoryInitialize(0)` with interface auto-detection: **passes**;
- Unitree `ChannelFactoryInitialize(0, "lo")` using its explicit loopback XML: **native aborts**.

Therefore the project does not replace CycloneDDS or weaken the G1 lifecycle test. Instead, `UnitreeG1LeRobot` explicitly overrides only the **simulated** LeRobot instance's hardcoded `"lo"` initialization so `simulation_dds_interface=None` calls the working one-argument Unitree initializer. The pinned EnvHub config is likewise set to interface auto-detection, keeping both ends on DDS domain 0. Physical G1 initialization remains untouched.

The negative `"lo"` probe stays in CI as an expected incompatibility. If a future Unitree/CycloneDDS update fixes that path, the contract will change visibly rather than leaving an undocumented workaround behind.

## CI evidence

The `unitree-g1-lerobot-contract` workflow statically verifies the pinned LeRobot v0.6.1 source assumptions:

- 29-joint model;
- `connect`, `disconnect`, `get_observation`, `send_action`, and `reset` surface;
- `lerobot/unitree-g1-mujoco` environment identifier;
- four normalized remote axes;
- controller reset on robot reset;
- GR00T balance-policy selection for near-zero commands.

Normal unit tests use a fake native G1 and verify that `stand` sends only zero remote axes and never joint targets. The independent physics workflow proves the pinned official EnvHub MuJoCo runtime can initialize, reset, and advance physics headlessly. The full-runtime workflow separately proves the native LeRobot G1 lifecycle through the actual adapter, including immediate same-process reconnect after simulated SDK2 endpoint cleanup. The GR00T stand workflow then proves that the agent-facing semantic `stand` call reaches the pinned balance policy, remains upright within the bounded roll/pitch envelope, and meets the recorded reliability/reproducibility contract.

## Not yet proven

- locomotion-axis calibration to physical m/s and rad/s;
- G1 `WALK` capability through the common semantic API;
- physical G1 execution;
- upper-body manipulation through a semantic policy boundary.
