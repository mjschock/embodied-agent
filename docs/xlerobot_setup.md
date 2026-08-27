# XLeRobot MuJoCo setup

`XLeRobotMuJoCo` loads the MuJoCo scene shipped in the upstream
`Vector-Wangel/XLeRobot` repository. We wrap the MJCF directly instead of importing
its keyboard/viewer demo, keeping UI and manual input out of the agent boundary.

## 1. Clone XLeRobot

```bash
git clone https://github.com/Vector-Wangel/XLeRobot.git
```

Install this project's optional simulator dependencies:

```bash
pip install -e ".[xlerobot-sim]"
```

## 2. Connect the adapter

```python
from embodied_agent.embodiments import XLeRobotMuJoCo

robot = XLeRobotMuJoCo(runtime_root="/path/to/XLeRobot")
await robot.connect()
```

The adapter resolves `simulation/mujoco/scene.xml`, which includes the upstream
XLeRobot model and assets.

## Semantic skills

### Bounded base velocity

```python
await robot.execute(
    "drive_velocity",
    lin_x_mps=0.2,
    lin_y_mps=0.0,
    yaw_rate_rps=0.0,
    duration_s=1.0,
)
```

The command is expressed in the robot/body frame and is always zeroed afterward.
Each command is limited to at most 5 seconds.

### Closed-loop pose navigation

```python
await robot.execute(
    "navigate_to",
    x_m=1.0,
    y_m=0.5,
    yaw_rad=1.57,
    max_duration_s=10.0,
)
```

This is a simple MuJoCo pose controller, not obstacle-aware path planning. A later
world/navigation layer can turn named locations and paths into these local pose
goals.

## Manipulation boundary

The upstream MJCF exposes 12 arm/gripper position actuators, and the adapter reports
their joint positions in observations. It intentionally does **not** expose
`Capability.MANIPULATE` or raw joint commands to the high-level agent yet.

Manipulation will be enabled only after a LeRobot/VLA or other bounded arm controller
owns those actuators. This preserves the project rule that the high-level agent uses
semantic skills rather than servo commands.
