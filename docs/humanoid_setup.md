# LeRobot Humanoid MuJoCo setup

`HumanoidMuJoCo` wraps the official LeRobot Humanoid runtime's
`SimBipedalRobotController`. The runtime currently lives in its own repository and
requires Python 3.13.

## 1. Clone the official runtime

```bash
git clone https://github.com/huggingface/lerobot-humanoid-runtime.git
cd lerobot-humanoid-runtime
git submodule update --init --recursive
uv sync --extra sim
```

The initialized model submodule is important: the simulator resolves its MJCF scene
from the runtime checkout.

## 2. Point embodied-agent at the checkout

```python
from embodied_agent.embodiments import HumanoidMuJoCo

robot = HumanoidMuJoCo(
    runtime_root="/path/to/lerobot-humanoid-runtime",
    fixed_base=False,
)
```

Then use the normal embodiment lifecycle:

```python
await robot.connect()
await robot.execute("stand")
observation = await robot.observe()
await robot.disconnect()
```

## Current capability boundary

The MuJoCo controller itself is a 12-DOF joint controller. It exposes simulation
state, IMU state, reset, and joint commands. It does **not** by itself implement
semantic walking or navigation.

The official runtime's locomotion path is an `RLAgent` running a trained ONNX/Torch
policy. Configure a policy directory to enable `Capability.WALK` and the bounded
`walk_velocity` skill:

```python
robot = HumanoidMuJoCo(
    runtime_root="/path/to/lerobot-humanoid-runtime",
    policy_dir="control/policy/<policy-directory>",
)

await robot.connect()
await robot.execute(
    "walk_velocity",
    lin_x_mps=0.2,
    lin_y_mps=0.0,
    yaw_rate_rps=0.0,
    duration_s=1.0,
)
```

Each velocity command is self-stopping after at most 5 seconds. Default command
limits mirror the official runtime's gamepad example: 0.75 m/s forward/back,
0.50 m/s lateral, and 0.80 rad/s yaw. `walk_to` remains intentionally absent until
we add closed-loop position/navigation feedback.
