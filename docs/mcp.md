# MCP server

`embodied-agent` exposes the same safe semantic `RobotToolRouter` through the official Model Context Protocol Python SDK and exposes shared world state as read-only MCP resources.

## Why the low-level MCP server

The project already owns the authoritative JSON Schemas for robot tools. The MCP Python SDK's low-level `Server` lets us publish those schemas directly instead of deriving a second contract from Python function signatures. It also lets the server expose the current `WorldState` through the protocol's resource handlers without creating a second world model.

The flow is:

```text
MCP host / high-level agent
          |
          +---------- read ----------> MCP resources
          |                           world://snapshot
          |                           world://entities/<id>
          |                                  |
          |                                  v
          |                              WorldState
          |
          +---------- act -----------> MCP tools
                                             |
                                             v
                                      RobotToolRouter
                            allowlist + validation + capability checks
                                             |
                                             v
                                      Embodiment adapter
                                             |
                                             v
                                    simulator / real robot
```

MCP does not get a privileged path around the router, and MCP resources do not provide a mutation path into world state.

## Install

The MCP integration targets the current stable v2 SDK line:

```bash
pip install -e ".[mcp]"
```

For a full simulator environment, combine extras as needed, for example:

```bash
pip install -e ".[mcp,crazyflie-sim,xlerobot-sim,humanoid-sim]"
```

XLeRobot and LeRobot Humanoid still require their upstream runtime/model checkouts and configured paths. See `docs/xlerobot_setup.md` and `docs/humanoid_setup.md`.

## Run over stdio

MCP desktop and local agent hosts commonly launch a server as a child process and communicate over stdin/stdout.

```bash
embodied-agent-mcp --config configs/all_sim.json
```

Equivalent module form:

```bash
python -m embodied_agent.mcp --config configs/all_sim.json
```

The MCP server lifecycle connects enabled robots at startup and disconnects them in reverse order during teardown.

## Tool exposure

Tool names remain namespaced by embodiment:

```text
crazyflie.takeoff
crazyflie.goto
crazyflie.land
xlerobot.navigate_to
humanoid.stand
humanoid.walk_velocity
```

Only tools that pass both checks are listed:

1. the skill is explicitly allowlisted in configuration;
2. the embodiment currently advertises the capability required by that skill.

For example, `humanoid.walk_velocity` is not exposed unless the humanoid backend actually advertises `WALK`.

Each MCP `inputSchema` is the same JSON Schema returned by `RobotToolRouter.list_tools()`. The MCP SDK performs protocol-level schema validation, and the router validates again before dispatch. Raw motor/joint commands have no safe tool schema and are not exposed.

## Read-only world resources

The server exposes a live JSON view of shared world state using standard MCP resources.

### `world://snapshot`

Returns the latest complete world snapshot:

```json
{
  "version": 7,
  "entities": {
    "inspection_target": {
      "entity_id": "inspection_target",
      "kind": "waypoint",
      "pose": {
        "x_m": 1.5,
        "y_m": 0.5,
        "z_m": 1.0,
        "yaw_rad": 0.0,
        "frame": "world"
      },
      "attributes": {},
      "source": "perception",
      "confidence": 0.94
    }
  },
  "robots": {},
  "tasks": {}
}
```

### `world://entities/<id>`

Returns one current entity. Entity IDs are URI-encoded in the resource URI, so an entity named `inspection target` is exposed as:

```text
world://entities/inspection%20target
```

Both resource forms are resolved at read time. If trusted perception or localization code updates an entity after a plan is created, the next MCP resource read sees the new pose rather than a startup-time snapshot.

World resources are intentionally read-only. There is no generic `world.upsert` MCP tool. Perception/localization adapters will own entity updates so an LLM cannot invent or silently overwrite grounded world coordinates.

## Action results update world state

When an MCP robot tool reaches the router and returns a result, the MCP adapter records that result in the shared world's `robots` section. This allows a host to issue an action and then read `world://snapshot` to inspect the latest outcome.

Protocol/schema failures that never reach a robot are not recorded as robot outcomes.

## Result shape

Successful and failed robot tool calls use the same structured result shape:

```json
{
  "tool": "crazyflie.takeoff",
  "ok": true,
  "detail": "...",
  "data": {}
}
```

A downstream controller failure sets the MCP result's error flag while preserving structured details. This keeps planning/tool-selection failures distinguishable from robot execution failures in evals.

## Programmatic / in-process use

The server can be embedded directly in a Python process and given the same `WorldState` used by a planner/executor or perception subsystem:

```python
from embodied_agent.agent import build_stack, load_config
from embodied_agent.mcp import build_mcp_server
from embodied_agent.world import WorldState

config = load_config("configs/all_sim.json")
registry, router = build_stack(config)
world = WorldState()

server = build_mcp_server(
    router,
    registry=registry,
    world=world,
)
```

The official MCP v2 `Client` can connect to the low-level server in memory. Contract tests validate tool listing, exact schemas, lifecycle behavior, action calls, live world resource listing/reads, URI encoding, and action-result reflection without spawning a subprocess.
