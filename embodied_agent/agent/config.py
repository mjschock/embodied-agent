from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from embodied_agent.core import Embodiment, RobotRegistry
from embodied_agent.embodiments import (
    CrazyfliePyBullet,
    CrazyflieSim,
    HumanoidMuJoCo,
    HumanoidSim,
    MicroduckMuJoCo,
    XLeRobotMuJoCo,
    XLeRobotSim,
)

from .tools import RobotToolRouter

AdapterFactory = Callable[..., Embodiment]

DEFAULT_ADAPTER_FACTORIES: dict[str, AdapterFactory] = {
    "gym_pybullet_drones": CrazyfliePyBullet,
    "crazyflie_stub": CrazyflieSim,
    "lerobot_humanoid_mujoco": HumanoidMuJoCo,
    "humanoid_stub": HumanoidSim,
    "microduck_mujoco": MicroduckMuJoCo,
    "xlerobot_mujoco": XLeRobotMuJoCo,
    "xlerobot_stub": XLeRobotSim,
}


def load_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("robot config must contain a JSON object")
    return data


def _robot_entries(config: Mapping[str, Any]) -> Mapping[str, Any]:
    robots = config.get("robots", config)
    if not isinstance(robots, Mapping):
        raise ValueError("config 'robots' must be an object")
    return robots


def build_registry(
    config: Mapping[str, Any],
    *,
    factories: Mapping[str, AdapterFactory] | None = None,
) -> RobotRegistry:
    available = dict(DEFAULT_ADAPTER_FACTORIES)
    if factories:
        available.update(factories)

    registry = RobotRegistry()
    for name, raw_entry in _robot_entries(config).items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"robot config for {name} must be an object")
        if raw_entry.get("enabled", True) is False:
            continue

        adapter = raw_entry.get("adapter")
        if not isinstance(adapter, str) or not adapter:
            raise ValueError(f"robot {name} is missing an adapter")
        try:
            factory = available[adapter]
        except KeyError as exc:
            raise ValueError(f"unknown robot adapter for {name}: {adapter}") from exc

        params = raw_entry.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"robot {name} params must be an object")
        registry.register(factory(name=name, **dict(params)))
    return registry


def build_stack(
    config: Mapping[str, Any],
    *,
    factories: Mapping[str, AdapterFactory] | None = None,
) -> tuple[RobotRegistry, RobotToolRouter]:
    registry = build_registry(config, factories=factories)
    allowed: dict[str, list[str]] = {}
    for name, raw_entry in _robot_entries(config).items():
        if name not in {robot.name for robot in registry}:
            continue
        tools = raw_entry.get("tools", []) if isinstance(raw_entry, Mapping) else []
        if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
            raise ValueError(f"robot {name} tools must be an array of strings")
        allowed[name] = list(tools)
    return registry, RobotToolRouter(registry, allowed)
