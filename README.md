# Embodied Agent

A simulation-first, multi-embodiment robotics platform for controlling:

- **XLeRobot** — mobile manipulation
- **Crazyflie** — aerial perception and navigation
- **LeRobot Humanoid** — legged / humanoid mobility

The design goal is **one high-level agent with stable semantic skills**, while each robot keeps its own low-level controller, simulator, and eventual physical hardware backend.

## Core principle

The high-level agent should never emit raw motor commands. It sees an explicit, allowlisted semantic tool surface such as:

```text
crazyflie.takeoff
crazyflie.goto
crazyflie.land
xlerobot.navigate_to
humanoid.stand
humanoid.walk_velocity   # only when a locomotion policy is configured
```

Each tool is schema-validated before it can reach an embodiment, and capabilities are checked at runtime. Configuration can remove tools, but it cannot grant a capability the underlying robot does not advertise.

```text
High-level agent / MCP host
          |
          v
      MCP transport
          |
          v
Schema-validated tool router
          |
          v
Semantic skill API
          |
      +---+----------------+----------------+
      |                    |                |
      v                    v                v
   XLeRobot            Crazyflie        Humanoid
      |                    |                |
   sim/real              sim/real          sim/real
```

## Why this abstraction

This lets the project evolve through four stages without rewriting the agent:

1. XLeRobot sim + Crazyflie sim + Humanoid sim
2. XLeRobot real + Crazyflie sim + Humanoid sim
3. XLeRobot real + Crazyflie real + Humanoid sim
4. XLeRobot real + Crazyflie real + Humanoid real

LeRobot fits primarily at the **robot-learning/data/policy layer**, not as a universal physics simulator.

## Current status

The repository now contains:

- a common asynchronous `Embodiment` interface and capability registry;
- a real Crazyflie `gym-pybullet-drones` `VelocityAviary` adapter;
- an upstream XLeRobot MuJoCo base-navigation adapter;
- an official LeRobot Humanoid MuJoCo controller adapter;
- optional policy-backed humanoid `walk_velocity` control;
- a config-driven robot stack;
- an allowlisted, schema-validated high-level agent tool router;
- an MCP v2 server exposing that same safe router over stdio;
- a deterministic capability planner and sequential executor;
- a dependency-free three-robot coordination benchmark with planning and execution metrics;
- deterministic simulation stubs and contract tests.

XLeRobot arm actuation is intentionally not agent-accessible yet. `MANIPULATE` will only be enabled after a LeRobot/VLA policy owns that lower-level control path.

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
pip install -e ".[humanoid-sim]"
```

For MCP:

```bash
pip install -e ".[mcp]"
embodied-agent-mcp --config configs/all_sim.json
```

The MCP integration targets the official Python SDK v2 line and publishes the router's JSON Schemas directly through the SDK's low-level server API. See `docs/mcp.md`.

XLeRobot and humanoid MuJoCo adapters also require their upstream runtime/model checkouts; see `docs/xlerobot_setup.md` and `docs/humanoid_setup.md`.

## Agent layer

`configs/all_sim.json` declares both the embodiment adapter and the skills that may be exposed to an agent. The router filters that allowlist against the robot's actual capabilities.

```python
from embodied_agent.agent import build_stack, load_config

config = load_config("configs/all_sim.json")
robots, tools = build_stack(config)

for tool in tools.list_tools():
    print(tool["name"], tool["input_schema"])
```

The same router can be exposed over MCP without changing any embodiment code:

```python
from embodied_agent.mcp import build_mcp_server

server = build_mcp_server(tools, registry=robots)
```

The current `CapabilityPlanner` is intentionally deterministic. It binds task steps to available robot tools and gives us a stable baseline before an LLM planner is introduced.

## Evals

The first benchmark coordinates all three embodiments across three waypoint variants. It separately reports robot-selection accuracy, exact plan match, task completion, tool-call success, and execution coverage.

```bash
python -m embodied_agent.evals.multi_robot
```

The benchmark currently runs against scripted embodiments so orchestration regressions are isolated from physics. It also includes a failure-injection test proving that correct planning can be distinguished from downstream execution failure. See `docs/evals.md`.

## Simulator backends

### XLeRobot
`XLeRobotMuJoCo` loads the upstream XLeRobot MuJoCo model directly and exposes bounded `drive_velocity`, closed-loop `navigate_to`, reset, and normalized state. Arm state is observable, but arm commands are not agent-accessible yet.

### Crazyflie
`CrazyfliePyBullet` uses `gym-pybullet-drones` `VelocityAviary`. The agent sees position-level `takeoff`, `goto`, and `land` skills while velocity/PID/RPM control remains beneath the adapter.

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
│   └── demo.py
├── configs/
├── docs/
├── tests/
└── pyproject.toml
```

## Next milestones

1. Add shared world/task state so multiple embodiments reason over the same entities and coordinates.
2. Run the coordination suite against physics-backed simulators.
3. Add bounded retry/failure-recovery policies.
4. Evaluate an LLM planner over MCP against the deterministic capability-planner baseline.
