from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import Capability, Observation, SkillRequest, SkillResult


class Embodiment(ABC):
    """Stable high-level contract shared by simulated and physical robots."""

    name: str
    backend: str

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[Capability]:
        raise NotImplementedError

    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def observe(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    async def execute_request(self, request: SkillRequest) -> SkillResult:
        raise NotImplementedError

    async def execute(self, skill: str, **params: Any) -> SkillResult:
        return await self.execute_request(SkillRequest(skill, params))

    def supports(self, *capabilities: Capability) -> bool:
        required = set(capabilities)
        return required.issubset(self.capabilities)
