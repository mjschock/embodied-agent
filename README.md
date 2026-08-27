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
High-level agent / planner
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
- a deterministic capability planner and sequential executor for baseline evals;
- deterministic simulation stubs and contract tests.

XLeRobot arm actuation is intentionally not agent-accessible yet. `MANIPULATE` will only be enabled after a LeRobot/VLA policy owns that lower-level control path.

## Quick start

Requires Python 3.11+.

```bash
python -m embodied_agent.demo
python -m unittest discover -s tests -v
```

The dependency-free demo still uses deterministic stubs. Real simulator adapters are optional:

```bash
pip install -e ".[crazyflie-sim]"
pip install -e ".[xlerobot-sim]"
pip install -e ".[humanoid-sim]"
```

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

The current `CapabilityPlanner` is intentionally deterministic. It binds task steps to available robot tools and gives us a stable baseline before an LLM or MCP planner is introduced.

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
│   │   ├── config.py
│   │   ├── planning.py
│   │   └── tools.py
│   ├── core/
│   ├── embodiments/
│   └── demo.py
├── configs/
├── docs/
├── tests/
└── pyproject.toml
```

## Next milestones

1. Multi-robot coordination evals for robot selection and task completion.
2. Shared world/task state.
3. MCP exposure of the same safe tool router.
4. An LLM planner evaluated against the deterministic capability-planner baseline.
