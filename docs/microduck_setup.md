# Microduck MuJoCo setup

`MicroduckMuJoCo` adds Pollen Robotics Microduck as a policy-backed simulated embodiment without changing the existing XLeRobot, Crazyflie, or LeRobot Humanoid stacks.

The adapter intentionally uses the native CPU deployment rehearsal from `pollen-robotics/microduck_rl`, not the browser/WASM Hugging Face Space as a remote control surface. The pinned upstream `scripts/infer_policy.py` remains authoritative for the shared observation layout, ONNX inference, policy switching, and 14-action joint target mapping.

## Runtime contract

The current upstream policy contract is:

- MuJoCo timestep: 0.005 s;
- policy decimation: 4 physics steps;
- policy frequency: 50 Hz;
- actor observation: 61 values;
- action: 14 servo joint targets;
- walking command: forward/backward velocity plus yaw rate; lateral walking is not exposed;
- episodic kick policies: left or right foot;
- stand policy: also used as the get-up/fall-recovery policy.

`embodied-agent` does not expose the 14 joint targets to the high-level agent. The agent sees only semantic skills.

## Install

```bash
pip install -e ".[microduck-sim]"
```

The optional dependency contains only the CPU inference/runtime layer (`mujoco`, `numpy`, and `onnxruntime`). It does not install the CUDA/MuJoCo-Warp/PPO training stack.

Clone the pinned/current `microduck_rl` source separately:

```bash
git clone https://github.com/pollen-robotics/microduck_rl
```

The CI integration test pins the exact upstream revision rather than following a moving branch.

## Policies

Provide paths to exported ONNX policies. A typical configuration uses the policies demonstrated in the official Microduck simulator Space:

- `BEST_alpha_walking.onnx`
- `BEST_alpha_stand.onnx`
- `ball_kick_left.onnx`
- `ball_kick_right.onnx`

See `configs/microduck_sim.example.json` for the config shape.

Both kick policies are required before the adapter advertises the `KICK` capability. This prevents an agent from seeing a generic `kick(foot=...)` tool when only one side can actually execute it.

## Semantic skills

### `microduck.stand`

Runs the standing policy and verifies the robot remains upright.

### `microduck.walk_velocity`

Runs the walking policy for a bounded duration. The native adapter mirrors the current upstream/browser command envelope:

- forward: up to +0.25 m/s;
- backward: down to -0.20 m/s;
- lateral: unsupported and rejected;
- yaw: up to ±1.0 rad/s;
- duration: at most 5 seconds per semantic call.

After each bounded walking call, the adapter returns the policy to standing with a zero velocity command.

### `microduck.kick`

Accepts only `foot="left"` or `foot="right"`. The selected ONNX behavior runs as a bounded one-shot and then hands control back to standing.

### `microduck.recover`

Recovery mirrors the current simulator behavior rather than silently resetting the robot:

1. allow the fallen body to settle briefly;
2. switch to the standing/get-up policy with zero commands;
3. require a continuous upright interval before reporting success;
4. return a failed `SkillResult` if the bounded recovery window expires.

A failed recovery therefore remains observable to the agent/eval layer.

## Example config

```bash
cp configs/microduck_sim.example.json /tmp/microduck.json
# Edit the runtime and policy paths.
embodied-agent-mcp --config /tmp/microduck.json
```

The existing `configs/all_sim.json` intentionally remains the three-robot baseline. Microduck is opt-in until its own physics gate is established and we decide which multi-embodiment benchmark should include it.

## Sim2real boundary

This adapter is simulation-only. A future physical Microduck backend should preserve the same agent-facing semantic skills while swapping the local implementation to Pollen's onboard runtime. No raw joint policy action should become an MCP/LLM tool during that transition.
