# OpenAI high-level agent provider

`OpenAIAgentModel` is the first real LLM implementation of the provider-agnostic `AgentModel` interface used by `MCPAgentRunner`.

The provider uses the OpenAI Responses API with Structured Outputs for one decision envelope at a time. It does not bypass MCP and it does not execute robot calls inside the OpenAI SDK.

```text
user instruction
      |
      v
MCPAgentRunner
      |
      +--> read world://snapshot
      +--> list safe MCP tools
      |
      v
OpenAIAgentModel
      |
Structured Output: one decision
      |
      +--> finish
      |
      +--> semantic tool + JSON arguments
                    |
                    v
             MCP client/server
                    |
                    v
             RobotToolRouter
                    |
                    v
               embodiment
```

## Install

OpenAI support is optional:

```bash
pip install -e ".[openai-agent]"
```

The extra currently targets the OpenAI Python SDK v3 line and includes the MCP v2 dependency:

```text
openai>=3,<4
mcp>=2,<3
```

The OpenAI SDK reads `OPENAI_API_KEY` using its normal environment configuration.

## Choose the model explicitly

The project intentionally does not hard-code an API model. Supply one at runtime:

```bash
export EMBODIED_AGENT_OPENAI_MODEL="YOUR_MODEL_ID"
```

or pass `--model` to the CLI.

This keeps model selection independent from robot control code and makes it possible to evaluate different models against the same embodied-agent benchmark.

## Run a task

With the robot/simulator configuration prepared:

```bash
embodied-agent-run \
  --config configs/all_sim.json \
  --model "$EMBODIED_AGENT_OPENAI_MODEL" \
  --max-steps 12 \
  "Take off with the drone to 1 meter, then land."
```

The command prints a structured JSON run result and exits non-zero if the bounded agent run fails.

The selected simulator adapters still need their own setup. In particular, XLeRobot and LeRobot Humanoid require their upstream model/runtime checkouts as documented elsewhere in this repository.

## Decision contract

The model is asked for exactly one Structured Output object containing:

```text
kind            "tool" or "finish"
tool            one MCP tool name or null
arguments_json  JSON object encoded as a string
summary         completion summary or empty string
```

Why is `arguments_json` a string instead of one generic nested Structured Output object?

The set of available robot tools and their input schemas is dynamic: it depends on which embodiments, policies, and capabilities are currently configured. The OpenAI provider constrains the outer decision shape, then parses `arguments_json` locally. The MCP server and `RobotToolRouter` remain the authoritative validators for the chosen tool's actual schema and bounds.

That preserves one source of truth for safety validation rather than copying every changing robot schema into provider-specific Python models.

## Context sent to the model

Every decision receives a compact JSON context containing:

- the original user instruction;
- the current decision index;
- the latest `world://snapshot`;
- every currently exposed MCP tool with its description and input schema;
- the action history and structured outcomes from this run.

The runner refreshes the world before every call to `OpenAIAgentModel.decide()`, so robot outcomes and perception/localization changes can affect the next action.

## Safety layers

The OpenAI provider is deliberately not a trusted control boundary. Several checks remain below it:

1. Structured Outputs constrain the provider's decision envelope.
2. `OpenAIAgentModel` rejects a tool name not present in the current MCP context.
3. `MCPAgentRunner` independently rejects tools not exposed by MCP.
4. The MCP SDK validates arguments against the published tool schema.
5. `RobotToolRouter` validates allowlists, capabilities, bounds, and argument shape again.
6. High-frequency motor/joint control stays inside robot-specific controllers or learned policies.
7. `MCPAgentRunner.max_steps` bounds the number of actions in one run.

No raw motor/joint command is added for the OpenAI provider.

## Programmatic use

```python
from mcp import Client

from embodied_agent.mcp import MCPAgentRunner, OpenAIAgentModel

model = OpenAIAgentModel(model="YOUR_MODEL_ID")

async with Client(server) as client:
    result = await MCPAgentRunner(
        client,
        model,
        max_steps=12,
    ).run("Scout the workbench and move the ground robot there.")
```

A custom async OpenAI client can be injected through `client=` for testing or specialized configuration.

## CI behavior

CI does not call the OpenAI API. Tests use a fake async Responses client to verify decision parsing, hidden-tool rejection, context construction, and argument parsing. A separate compatibility assertion instantiates the installed OpenAI SDK with a dummy key and verifies the asynchronous `responses.parse` surface exists without sending a request.
