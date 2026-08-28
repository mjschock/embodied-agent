from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
from pathlib import Path

from embodied_agent.mcp import OpenAIAgentModel

from .agent_model import AgentModelFactory, evaluate_agent_model
from .four_embodiment import default_four_embodiment_cases, scripted_four_embodiment_stack
from .result_record import build_result_record, git_revision, write_result_record

BENCHMARK_NAME = "four-embodiment-selection"


async def evaluate_four_embodiment_model(
    model_factory: AgentModelFactory,
    *,
    max_steps: int = 10,
) -> dict:
    result = await evaluate_agent_model(
        model_factory,
        cases=default_four_embodiment_cases(),
        max_steps=max_steps,
        stack=scripted_four_embodiment_stack(),
    )
    return result.to_dict()


async def run_openai_four_embodiment_eval(
    *,
    model_name: str,
    max_steps: int = 10,
) -> dict:
    model = OpenAIAgentModel(model=model_name)
    return await evaluate_four_embodiment_model(
        lambda _case: model,
        max_steps=max_steps,
    )


def build_openai_four_embodiment_record(
    *,
    model_name: str,
    max_steps: int,
    payload: dict,
) -> dict:
    return build_result_record(
        benchmark=BENCHMARK_NAME,
        provider="openai",
        model=model_name,
        max_steps=max_steps,
        result=payload,
        environment={
            "repository_revision": os.getenv("GITHUB_SHA") or git_revision(Path.cwd()),
            "python_version": platform.python_version(),
            "robot_backend": "scripted-four-embodiment-eval",
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenAI AgentModel on the scripted four-embodiment selection "
            "benchmark spanning Crazyflie, XLeRobot, LeRobot Humanoid, and Microduck."
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
        default=10,
        help="Maximum semantic robot actions per case (default: 10).",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EMBODIED_AGENT_EVAL_OUTPUT", ""),
        help="Optional schema-versioned JSON result-record path.",
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
        run_openai_four_embodiment_eval(
            model_name=args.model,
            max_steps=args.max_steps,
        )
    )
    record = build_openai_four_embodiment_record(
        model_name=args.model,
        max_steps=args.max_steps,
        payload=payload,
    )
    if args.output.strip():
        write_result_record(record, args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if payload["strict_task_success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
