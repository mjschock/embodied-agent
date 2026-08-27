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
policy. `embodied-agent` therefore does not claim the `WALK` capability yet. The next
increment will connect a locomotion policy runner and expose a bounded semantic
walking skill above it.
