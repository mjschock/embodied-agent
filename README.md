# Embodied Agent

A simulation-first, multi-embodiment robotics platform for controlling:

- **XLeRobot** — mobile manipulation
- **Crazyflie** — aerial perception and navigation
- **Microduck** — learned locomotion and low-cost sim2real RL
- **LeRobot Humanoid** — legged / humanoid mobility

The design goal is **one high-level agent with stable semantic skills**, while each robot keeps its own low-level controller, simulator, learned policies, and eventual physical hardware backend.

## Core principle

The high-level agent should never emit raw motor commands. It sees an explicit, allowlisted semantic tool surface such as:

```text
crazyflie.takeoff
crazyflie.goto
crazyflie.land
xlerobot.navigate_to
microduck.walk_velocity
microduck.kick
microduck.roll
humanoid.stand
humanoid.walk_velocity   # only when a locomotion policy is configured
```

Each tool is schema-validated before it can reach an embodiment, and capabilities are checked at runtime. Configuration can remove tools, but it cannot grant a capability the underlying robot does not advertise.

```text
Natural-language task
          |
          v
High-level AgentModel
          |
  read world / choose one action
          |
          v
      MCP transport
          |
          v
Schema-validated tool router
          |
          +----------> shared WorldState
          |               entities / poses / task state
          v
Semantic skill API
          |
      +-----------+-----------+-----------+-----------+
      |           |           |           |
      v           v           v           v
   XLeRobot   Crazyflie   Microduck    Humanoid
      |           |           |           |
   sim/real     sim/real     sim/real     sim/real
```

## Why this abstraction

The simulator or hardware implementation can change without rewriting the high-level agent. XLeRobot can move from MuJoCo to physical mobile manipulation, Crazyflie from PyBullet to `cflib`, Microduck from native MuJoCo/ONNX to its onboard policy runtime, and the humanoid from the current biped simulator to a future whole-body LeRobot backend while preserving semantic agent skills.

LeRobot fits primarily at the **robot-learning/data/policy layer**, not as a universal physics simulator.

## Current status

The repository now contains:

- a common asynchronous `Embodiment` interface and capability registry;
- a real Crazyflie `gym-pybullet-drones` `VelocityAviary` adapter;
- an upstream XLeRobot MuJoCo base-navigation adapter;
- a policy-backed Microduck native MuJoCo/ONNX adapter with `stand`, `walk_velocity`, `kick`, and `roll`;
- an official LeRobot Humanoid MuJoCo controller adapter;
- optional policy-backed humanoid `walk_velocity` control;
- a config-driven robot stack;
- an allowlisted, schema-validated high-level agent tool router;
- an MCP v2 server exposing the safe router over stdio;
- live read-only MCP resources for shared world state;
- a shared `WorldState` for named entities, world-frame poses, task progress, and latest robot results;
- late-bound entity references resolved immediately before robot actions;
- a deterministic capability planner and sequential executor;
- an iterative provider-agnostic `MCPAgentRunner` that refreshes world state before every model decision;
- an optional OpenAI Responses API `AgentModel` using Structured Outputs for one decision at a time;
- dependency-free and real-physics three-robot coordination benchmarks with planning/execution metrics;
- deterministic simulator/agent contract tests and pinned real-simulator CI gates.

XLeRobot arm actuation is intentionally not agent-accessible yet. `MANIPULATE` will only be enabled after a LeRobot/VLA policy owns that lower-level control path. Microduck likewise exposes learned semantic behaviors rather than its 14 raw policy joint targets. The current Microduck walking/standing stack does **not** advertise arbitrary-fall recovery; `roll` is the separate learned `roulade` behavior and true dead/fallen states remain explicit reset conditions.

## Quick start

Requires Python 3.11+.

```bash
python -m embodied_agent.demo
python -m embodied_agent.evals.multi_robot
python -m unittest discover -s tests -v
```

The dependency-free demo and baseline eval do not require physics packages. Real simulator adapters are optional:

```bash
pip install -e ".[crazyflie-sim]"
pip install -e ".[xlerobot-sim]"
pip install -e ".[microduck-sim]"
pip install -e ".[humanoid-sim]"
```

For MCP:

```bash
pip install -e ".[mcp]"
embodied-agent-mcp --config configs/all_sim.json
```

For the OpenAI high-level agent:

```bash
pip install -e ".[openai-agent]"
export OPENAI_API_KEY="..."
export EMBODIED_AGENT_OPENAI_MODEL="YOUR_MODEL_ID"

embodied-agent-run \
  --config configs/all_sim.json \
  --max-steps 12 \
  "Take off with the drone to 1 meter, then land."
```

The project deliberately does not hard-code an OpenAI model. See `docs/openai_agent.md` for the provider contract and safety layers.

XLeRobot, Microduck, and humanoid MuJoCo adapters also require their upstream runtime/model or policy assets; see `docs/xlerobot_setup.md`, `docs/microduck_setup.md`, and `docs/humanoid_setup.md`.

## Agent layer

`configs/all_sim.json` declares both the embodiment adapter and the skills that may be exposed to an agent. The router filters that allowlist against the robot's actual capabilities. Microduck is currently opt-in via `configs/microduck_sim.example.json` so the existing three-robot baseline remains unchanged.

```python
from embodied_agent.agent import build_stack, load_config

config = load_config("configs/all_sim.json")
robots, tools = build_stack(config)

for tool in tools.list_tools():
    print(tool["name"], tool["input_schema"])
```

The same router and world model can be exposed over MCP without changing embodiment code:

```python
from embodied_agent.mcp import build_mcp_server
from embodied_agent.world import WorldState

world = WorldState()
server = build_mcp_server(tools, registry=robots, world=world)
```

MCP resources expose `world://snapshot` and live per-entity reads. They are read-only: trusted perception/localization code owns entity updates while MCP tools own safe robot actions.

The current `CapabilityPlanner` remains a deterministic orchestration baseline. `MCPAgentRunner` provides the model-driven path and asks its `AgentModel` for exactly one next action at a time, refreshing world state before every decision.

## Shared world state

Plans can reference named world entities instead of copying coordinates into every step:

```python
from embodied_agent.world import Pose3D, WorldEntity, WorldState, entity_pose_refs

world = WorldState()
world.upsert_entity(
    WorldEntity("inspection_target", "waypoint", Pose3D(1.5, 0.5, 1.0))
)

drone_args = entity_pose_refs("inspection_target", "x_m", "y_m", "z_m")
ground_args = entity_pose_refs("inspection_target", "x_m", "y_m")
```

`PlanExecutor(..., world=world)` resolves those references immediately before each tool call. If perception updates `inspection_target` after the plan was created, later robot steps use the corrected pose rather than stale coordinates. Task progress and latest robot results are recorded back into the same world state. See `docs/world_state.md`.

## High-level LLM agent

The provider-agnostic runtime is deliberately iterative:

```text
read current world
      ↓
choose one safe MCP tool
      ↓
execute tool
      ↓
record result in world
      ↓
read current world again
      ↓
choose next action or finish
```

The OpenAI implementation uses Structured Outputs for the decision envelope, but MCP and `RobotToolRouter` remain authoritative for dynamic tool schemas, capability checks, parameter bounds, and allowlists. The runner also has an explicit maximum-step limit and stops on a failed robot tool by default. See `docs/agent_loop.md` and `docs/openai_agent.md`.

## Evals

The established benchmark coordinates XLeRobot, Crazyflie, and the LeRobot Humanoid across three waypoint variants. It separately reports robot-selection accuracy, exact plan match, task completion, tool-call success, and execution coverage.

```bash
python -m embodied_agent.evals.multi_robot
```

The same benchmark is CI-verified against real PyBullet/MuJoCo adapters and through the oracle `AgentModel` → MCP path. Microduck currently has its own pinned learned-policy physics smoke; dedicated Microduck skill and multi-embodiment eval cases are the next eval layer. See `docs/evals.md`.

## Simulator backends

### XLeRobot
`XLeRobotMuJoCo` loads the upstream XLeRobot MuJoCo model directly and exposes bounded `drive_velocity`, closed-loop `navigate_to`, reset, and normalized state. Arm state is observable, but arm commands are not agent-accessible yet.

### Crazyflie
`CrazyfliePyBullet` uses `gym-pybullet-drones` `VelocityAviary`. The agent sees position-level `takeoff`, `goto`, and `land` skills while velocity/PID/RPM control remains beneath the adapter.

### Microduck
`MicroduckMuJoCo` loads Pollen's native Microduck MuJoCo model and delegates the 61-D observation → ONNX inference → 14-action mapping to the upstream CPU inference reference. The agent sees bounded `stand`, longitudinal/yaw `walk_velocity`, enum-validated left/right `kick`, and the learned `roll`/`roulade` one-shot. The adapter returns to standing after bounded behaviors and never exposes raw learned-policy joint targets. See `docs/microduck_setup.md`.

### LeRobot Humanoid
`HumanoidMuJoCo` wraps the official LeRobot Humanoid `SimBipedalRobotController` for reset, stand, and normalized observation state. When configured with an official `RLAgent` policy directory it also exposes a bounded `walk_velocity` skill; `walk_to` remains deferred until closed-loop navigation is implemented.

## Repository layout

```text
embodied-agent/
├── embodied_agent/
│   ├── agent/
│   ├── core/
│   ├── embodiments/
│   ├── evals/
│   ├── mcp/
│   ├── world/
│   └── demo.py
├── configs/
├── docs/
├── integration_tests/
├── tests/
└── pyproject.toml
```

## Next milestones

1. Record the first credentialed live-LLM result against the deterministic physics baseline.
2. Add Microduck learned-policy skill and multi-embodiment eval cases.
3. Add broader perception-update / stale-plan evals.
4. Add bounded retry/failure-recovery policies.
5. Preserve the Microduck semantic interface when moving from simulation to the physical robot.
