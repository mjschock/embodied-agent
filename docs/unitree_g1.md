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

## Runtime installation caveat

The project provides an optional `unitree-g1` dependency for LeRobot's G1 extra, but the upstream G1 setup also requires Unitree SDK2 Python/CycloneDDS outside that extra. Current LeRobot documentation uses Python 3.12, installs `unitree_sdk2_python`, then installs LeRobot's `unitree_g1` extra. The MuJoCo environment additionally needs its simulation dependencies.

Because those dependencies are substantially heavier than the dependency-free contract test, runtime MuJoCo validation is tracked separately rather than implied by this adapter PR.

## CI evidence

The dedicated `unitree-g1-lerobot-contract` workflow statically verifies the pinned LeRobot v0.6.1 source assumptions:

- 29-joint model;
- `connect`, `disconnect`, `get_observation`, `send_action`, and `reset` surface;
- `lerobot/unitree-g1-mujoco` environment identifier;
- four normalized remote axes;
- controller reset on robot reset;
- GR00T balance-policy selection for near-zero commands.

Normal unit tests use a fake native G1 and verify that `stand` sends only zero remote axes and never joint targets.

## Not yet proven

- successful launch of the official Hub-hosted G1 MuJoCo environment from `embodied-agent` CI;
- locomotion-axis calibration to physical m/s and rad/s;
- G1 `WALK` capability through the common semantic API;
- physical G1 execution;
- upper-body manipulation through a semantic policy boundary.
