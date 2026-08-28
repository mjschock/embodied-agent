from __future__ import annotations

import argparse
import asyncio
import json
import os

from embodied_agent.mcp import OpenAIAgentModel

from .agent_model import evaluate_agent_model


async def run_openai_eval(
    *,
    model_name: str,
    max_steps: int = 8,
) -> dict:
    model = OpenAIAgentModel(model=model_name)
    result = await evaluate_agent_model(
        lambda _case: model,
        max_steps=max_steps,
    )
    return result.to_dict()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenAI AgentModel on the scripted three-robot benchmark. "
            "This command does not connect to physical robots or physics simulators."
        )
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBODIED_AGENT_OPENAI_MODEL", ""),
        help="OpenAI model ID, or set EMBODIED_AGENT_OPENAI_MODEL.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum semantic robot actions per case (default: 8).",
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

    payload = asyncio.run(
        run_openai_eval(
            model_name=args.model,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["strict_task_success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
