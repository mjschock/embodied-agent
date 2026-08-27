from __future__ import annotations

import argparse
import asyncio

from .server import build_mcp_server_from_config, serve_stdio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose embodied-agent semantic robot tools over MCP stdio."
    )
    parser.add_argument(
        "--config",
        default="configs/all_sim.json",
        help="Path to robot JSON config (default: configs/all_sim.json).",
    )
    parser.add_argument(
        "--name",
        default="embodied-agent",
        help="MCP server name advertised to the host.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server, _, _ = build_mcp_server_from_config(args.config, name=args.name)
    asyncio.run(serve_stdio(server))


if __name__ == "__main__":
    main()
