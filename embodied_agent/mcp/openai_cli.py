from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from mcp import Client

from embodied_agent.world import WorldState

from .agent import AgentRunResult, MCPAgentRunner
from .openai_model import OpenAIAgentModel
from .server import build_mcp_server_from_config


def _result_dict(result: AgentRunResult) -> dict[str, Any]:
    return {
        "instruction": result.instruction,
        "ok": result.ok,
        "finished": result.finished,
        "summary": result.summary,
        "error": result.error,
        "actions": [
            {
                "step_index": action.step_index,
                "tool": action.tool,
                "arguments": dict(action.arguments),
                "ok": action.ok,
                "detail": action.detail,
                "structured_content": dict(action.structured_content),
            }
            for action in result.actions
        ],
    }


async def run_agent(
    *,
    config: str,
    model_name: str,
    instruction: str,
    max_steps: int,
) -> AgentRunResult:
    world = WorldState()
    server, _, _ = build_mcp_server_from_config(config, world=world)
    model = OpenAIAgentModel(model=model_name)

    async with Client(server) as client:
        return await MCPAgentRunner(
            client,
            model,
            max_steps=max_steps,
        ).run(instruction)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded high-level embodied-agent task through OpenAI + MCP."
    )
    parser.add_argument(
        "instruction",
        help="Natural-language task for the high-level embodied agent.",
    )
    parser.add_argument(
        "--config",
        default="configs/all_sim.json",
        help="Robot/MCP configuration JSON path (default: configs/all_sim.json).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBODIED_AGENT_OPENAI_MODEL", ""),
        help=(
            "OpenAI model ID. Can also be set with EMBODIED_AGENT_OPENAI_MODEL. "
            "No model is hard-coded by the project."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum number of robot actions before the run is stopped (default: 12).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.model.strip():
        raise SystemExit(
            "An OpenAI model is required. Pass --model or set EMBODIED_AGENT_OPENAI_MODEL."
        )
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")

    result = asyncio.run(
        run_agent(
            config=args.config,
            model_name=args.model,
            instruction=args.instruction,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(_result_dict(result), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
