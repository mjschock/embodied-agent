# High-level MCP agent loop

`MCPAgentRunner` is the high-level orchestration boundary for a model-driven agent.

It deliberately does **not** give a model direct access to robot adapters, controller objects, joint commands, or simulator APIs. The model sees only:

- the original natural-language instruction;
- the current `world://snapshot` resource;
- the currently exposed safe MCP tools and their input schemas;
- prior action outcomes from the same run.

The model returns exactly one next decision: call one exposed semantic MCP tool, or finish.

```text
natural-language instruction
          |
          v
     MCPAgentRunner
          |
          +---- read world://snapshot
          |
          +---- list safe MCP tools
          |
          v
       AgentModel
          |
     one decision
          |
     +----+----+
     |         |
   tool      finish
     |
     v
MCP server -> RobotToolRouter -> embodiment
     |
     v
world state updated with result
     |
     +---- next decision re-reads world
```

## Why decisions are iterative

The runner does not ask the model to generate a fixed multi-step robot program up front. Before every decision it re-reads `world://snapshot`.

That means a perception correction, localization update, task-state change, or robot execution result can influence the next action immediately. This preserves the late-bound world-state design used by the deterministic planner instead of copying stale coordinates through a long LLM-generated plan.

## Safety boundaries

`MCPAgentRunner` adds orchestration-level bounds on top of the MCP server and router:

- a model-selected tool must be present in the MCP tool list;
- hidden/raw control methods cannot be selected through the runner;
- MCP/schema/router validation still applies to every call;
- the run has a hard `max_steps` budget;
- by default, a failed robot tool stops the run;
- the model only receives structured world/tool/action context, never a raw controller handle.

The default stop-on-error behavior is intentional. Recovery policies will be added as an explicit bounded feature rather than allowing an unconstrained model to retry indefinitely.

## Provider-agnostic model interface

A model backend implements one asynchronous method:

```python
from embodied_agent.mcp import AgentDecision, AgentModel

class MyModel:
    async def decide(self, context):
        # context.instruction
        # context.world_snapshot
        # context.tools
        # context.history
        return AgentDecision.call(
            "crazyflie.takeoff",
            {"altitude_m": 1.0},
        )
```

When the task is complete:

```python
return AgentDecision.finish("inspection complete")
```

This keeps provider-specific prompt/API code outside the robot and orchestration layers. A real LLM provider can be added without changing any embodiment adapter or MCP tool.

## Running in process

```python
from mcp import Client

from embodied_agent.mcp import MCPAgentRunner, build_mcp_server

server = build_mcp_server(router, registry=registry, world=world)

async with Client(server) as client:
    result = await MCPAgentRunner(client, model, max_steps=12).run(
        "Scout the workbench and move the ground robot there."
    )
```

## Current validation

Contract tests use an in-memory three-robot MCP server and scripted model decisions to verify:

- one agent can sequence Crazyflie, XLeRobot, and humanoid tools;
- world state is refreshed before every model decision;
- previous robot outcomes are visible in the next model context;
- an unexposed raw motor command is rejected before dispatch;
- controller failure stops the run by default;
- the maximum-step budget stops runaway decision loops.

The next layer is a real LLM implementation of `AgentModel`, followed by evals against the existing deterministic `CapabilityPlanner` baseline.
