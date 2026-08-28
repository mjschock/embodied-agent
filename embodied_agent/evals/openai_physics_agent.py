from __future__ import annotations

import argparse
import asyncio
import json
import os

from embodied_agent.mcp import OpenAIAgentModel

from .physics_agent_model import compare_physics_agent_to_deterministic
from .result_record import (
    build_result_record,
    physics_environment_metadata,
    write_result_record,
)

BENCHMARK_NAME = "three-robot-physics-comparison"


async def run_openai_physics_comparison(
    *,
    model_name: str,
    xlerobot_runtime_root: str,
    humanoid_runtime_root: str,
    max_steps: int = 8,
) -> dict:
    model = OpenAIAgentModel(model=model_name)
    result = await compare_physics_agent_to_deterministic(
        lambda _case: model,
        xlerobot_runtime_root=xlerobot_runtime_root,
        humanoid_runtime_root=humanoid_runtime_root,
        max_steps=max_steps,
    )
    return result.to_dict()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an OpenAI high-level AgentModel with deterministic orchestration "
            "on the same sequential XLeRobot + Crazyflie + LeRobot Humanoid physics suite."
        )
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBODIED_AGENT_OPENAI_MODEL", ""),
        help="OpenAI model ID, or set EMBODIED_AGENT_OPENAI_MODEL.",
    )
    parser.add_argument(
        "--xlerobot-runtime-root",
        default=os.getenv("XLEROBOT_UPSTREAM_ROOT", ""),
        help="Pinned XLeRobot checkout, or set XLEROBOT_UPSTREAM_ROOT.",
    )
    parser.add_argument(
        "--humanoid-runtime-root",
        default=os.getenv("LEROBOT_HUMANOID_RUNTIME_ROOT", ""),
        help="Pinned LeRobot Humanoid runtime checkout, or set LEROBOT_HUMANOID_RUNTIME_ROOT.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum semantic robot actions per case (default: 8).",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EMBODIED_AGENT_EVAL_OUTPUT", ""),
        help=(
            "Optional JSON record path. The record includes model, timestamp, repository/upstream "
            "revisions, and the full deterministic/model comparison payload."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.model.strip():
        raise SystemExit(
            "An OpenAI model is required. Pass --model or set EMBODIED_AGENT_OPENAI_MODEL."
        )
    if not args.xlerobot_runtime_root.strip():
        raise SystemExit(
            "An XLeRobot checkout is required. Pass --xlerobot-runtime-root or set "
            "XLEROBOT_UPSTREAM_ROOT."
        )
    if not args.humanoid_runtime_root.strip():
        raise SystemExit(
            "A LeRobot Humanoid runtime checkout is required. Pass --humanoid-runtime-root "
            "or set LEROBOT_HUMANOID_RUNTIME_ROOT."
        )
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")

    payload = asyncio.run(
        run_openai_physics_comparison(
            model_name=args.model,
            xlerobot_runtime_root=args.xlerobot_runtime_root,
            humanoid_runtime_root=args.humanoid_runtime_root,
            max_steps=args.max_steps,
        )
    )
    record = build_result_record(
        benchmark=BENCHMARK_NAME,
        provider="openai",
        model=args.model,
        max_steps=args.max_steps,
        result=payload,
        environment=physics_environment_metadata(
            xlerobot_runtime_root=args.xlerobot_runtime_root,
            humanoid_runtime_root=args.humanoid_runtime_root,
        ),
    )
    if args.output.strip():
        write_result_record(record, args.output)
    print(json.dumps(record, indent=2, sort_keys=True))

    deterministic_ok = payload["deterministic"]["task_completion_rate"] == 1.0
    agent_ok = payload["agent"]["strict_task_success_rate"] == 1.0
    return 0 if deterministic_ok and agent_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
