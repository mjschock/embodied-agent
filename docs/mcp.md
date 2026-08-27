# MCP server

`embodied-agent` exposes the same safe semantic `RobotToolRouter` through the official Model Context Protocol Python SDK.

## Why the low-level MCP server

The project already owns the authoritative JSON Schemas for robot tools. The MCP Python SDK's low-level `Server` lets us publish those schemas directly instead of deriving a second contract from Python function signatures.

The flow is:

```text
MCP host / high-level agent
          |
          v
MCP low-level Server
          |
          v
RobotToolRouter
  allowlist + schema validation + capability checks
          |
          v
Embodiment adapter
          |
          v
simulator / real robot
```

MCP does not get a privileged path around the router.

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

The server can also be embedded directly in a Python process:

```python
from embodied_agent.agent import build_stack, load_config
from embodied_agent.mcp import build_mcp_server

config = load_config("configs/all_sim.json")
registry, router = build_stack(config)
server = build_mcp_server(router, registry=registry)
```

The official MCP v2 `Client` can connect to the low-level server in memory, which is how the contract tests validate tool listing, exact schemas, lifecycle behavior, and calls without spawning a subprocess.
