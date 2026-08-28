# Microduck MuJoCo setup

`MicroduckMuJoCo` adds Pollen Robotics Microduck as a policy-backed simulated embodiment without changing the existing XLeRobot, Crazyflie, or LeRobot Humanoid stacks.

The adapter intentionally uses the native CPU deployment rehearsal from `pollen-robotics/microduck_rl`, not the browser/WASM Hugging Face Space as a remote control surface. The pinned upstream `scripts/infer_policy.py` remains authoritative for the shared observation layout, ONNX inference, policy switching, and 14-action joint target mapping.

## Runtime contract

The pinned upstream policy contract used by CI is:

- MuJoCo timestep: 0.005 s;
- policy decimation: 4 physics steps;
- policy frequency: 50 Hz;
- actor observation: 61 values;
- action: 14 servo joint targets;
- walking command: forward/backward velocity plus yaw rate; lateral walking is not exposed;
- episodic kick policies: left or right foot;
- episodic `roulade` policy: intentionally rolls the robot and returns upright;
- standing policy: holds/returns to standing from its supported posture, but is not exposed as generic arbitrary-fall recovery.

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

Provide paths to exported ONNX policies. The pinned CI configuration uses policies from the official Microduck simulator Space:

- `BEST_alpha_walking.onnx`
- `BEST_alpha_stand.onnx`
- `ball_kick_left.onnx`
- `ball_kick_right.onnx`
- `roulade.onnx`

See `configs/microduck_sim.example.json` for the config shape.

Both kick policies are required before the adapter advertises the `KICK` capability. This prevents an agent from seeing a generic `kick(foot=...)` tool when only one side can actually execute it. The `ROLL` capability is exposed only when `roulade.onnx` is configured.

## Semantic skills

### `microduck.stand`

Runs the standing policy and verifies the robot remains upright.

### `microduck.walk_velocity`

Runs the walking policy for a bounded duration. The native adapter mirrors the pinned upstream/browser command envelope:

- forward: up to +0.25 m/s;
- backward: down to -0.20 m/s;
- lateral: unsupported and rejected;
- yaw: up to ±1.0 rad/s;
- duration: at most 5 seconds per semantic call.

After each bounded walking call, the adapter returns the policy to standing with a zero velocity command.

### `microduck.kick`

Accepts only `foot="left"` or `foot="right"`. The selected ONNX behavior runs as a bounded one-shot and then hands control back to standing.

### `microduck.roll`

Runs the pinned official `roulade.onnx` behavior with the same physical completion semantics used by the corresponding Space snapshot. Roll is not declared successful merely because a fixed timer expired. At 50 Hz the adapter:

1. records that the trunk actually tipped once projected-gravity Z rises above `-0.3`;
2. requires it to return upright below `-0.85`;
3. requires at least 40 control steps before declaring completion;
4. gives the behavior a hard 150-step / 3-second window.

A successful roll therefore means the policy produced a real tip-and-return cycle. If the hard window expires while the robot is still not upright, the simulator is reset to a safe `STAND` state, but the semantic skill still returns failure (`completed=false`) so the agent/eval layer cannot confuse a safety reset with task success.

This is an intentional learned trick, not a generic fall-recovery primitive.

## AgentModel eval

`embodied_agent.evals.microduck_skills` defines five semantic cases over the same MCP/AgentModel scoring machinery used by the multi-robot benchmarks:

- stand;
- bounded forward walk;
- left kick;
- right kick;
- roll.

The dependency-free scripted baseline tests tool selection and argument grounding, including an adversarial wrong-kick-foot case. The `microduck-physics` workflow then runs the oracle `ExpectedActionModel` through the real MCP server and the pinned native MuJoCo/ONNX backend. This isolates the full `AgentModel -> MCP -> semantic adapter -> learned policy -> MuJoCo` seam without requiring a live external LLM call.

## Falls and reset

The pinned official browser simulator does not expose arbitrary-fall get-up as a semantic capability for the walking policy. A genuinely fallen/dead pose is handled as a reset condition. `embodied-agent` therefore does **not** advertise `microduck.recover`; doing so would overstate the available policy capability.

`microduck.reset` remains a safe, explicit reset to the model's real `STAND` keyframe.

## Example config

```bash
cp configs/microduck_sim.example.json /tmp/microduck.json
# Edit the runtime and policy paths.
embodied-agent-mcp --config /tmp/microduck.json
```

The existing `configs/all_sim.json` intentionally remains the three-robot baseline. Microduck is opt-in while four-embodiment task-selection benchmarks are developed separately.

## Sim2real boundary

This adapter is simulation-only. A future physical Microduck backend should preserve the same agent-facing semantic skills while swapping the local implementation to Pollen's onboard runtime. No raw joint policy action should become an MCP/LLM tool during that transition.
