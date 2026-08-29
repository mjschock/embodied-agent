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
        "gravity_compensation": false
      },
      "tools": ["observe", "reset", "stand"]
    }
  }
}
```

## Pinned EnvHub physics evidence

The dedicated `unitree-g1-physics` workflow now launches and steps the official Hub-hosted MuJoCo environment headlessly. It pins:

- EnvHub `lerobot/unitree-g1-mujoco` at `a38dc8617f0fca51b38e9354dc58ee35ad850fb5`;
- official `unitreerobotics/unitree_sdk2_python` source at `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`;
- CycloneDDS `0.10.2` on Python 3.10;
- MuJoCo software OpenGL through OSMesa on the CPU-only hosted runner.

The smoke performs a real reset with seed 123 and five zero-action physics steps. Every returned observation must be finite, reward remains zero, and the environment must remain non-terminated/non-truncated.

### Runtime issues discovered by the probe

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

## CI evidence

The `unitree-g1-lerobot-contract` workflow statically verifies the pinned LeRobot v0.6.1 source assumptions:

- 29-joint model;
- `connect`, `disconnect`, `get_observation`, `send_action`, and `reset` surface;
- `lerobot/unitree-g1-mujoco` environment identifier;
- four normalized remote axes;
- controller reset on robot reset;
- GR00T balance-policy selection for near-zero commands.

Normal unit tests use a fake native G1 and verify that `stand` sends only zero remote axes and never joint targets. The independent physics workflow proves the pinned official EnvHub MuJoCo runtime can initialize, reset, and advance physics headlessly without implying that the full LeRobot/GR00T controller stack has been executed.

## Not yet proven

- full LeRobot v0.6.1 `UnitreeG1.connect()` against the pinned EnvHub simulator;
- GR00T balance/standing policy execution through `UnitreeG1LeRobot`;
- locomotion-axis calibration to physical m/s and rad/s;
- G1 `WALK` capability through the common semantic API;
- physical G1 execution;
- upper-body manipulation through a semantic policy boundary.
