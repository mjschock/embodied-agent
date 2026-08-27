from __future__ import annotations

from collections.abc import Iterator

from .embodiment import Embodiment
from .models import Capability


class RobotRegistry:
    def __init__(self) -> None:
        self._robots: dict[str, Embodiment] = {}

    def register(self, robot: Embodiment) -> None:
        if robot.name in self._robots:
            raise ValueError(f"Robot already registered: {robot.name}")
        self._robots[robot.name] = robot

    def get(self, name: str) -> Embodiment:
        return self._robots[name]

    def with_capabilities(self, *capabilities: Capability) -> list[Embodiment]:
        return [
            robot
            for robot in self._robots.values()
            if robot.supports(*capabilities)
        ]

    def __iter__(self) -> Iterator[Embodiment]:
        return iter(self._robots.values())

    def __len__(self) -> int:
        return len(self._robots)
