from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Capability(StrEnum):
    OBSERVE = "observe"
    NAVIGATE = "navigate"
    MANIPULATE = "manipulate"
    FLY = "fly"
    WALK = "walk"
    STAND = "stand"
    KICK = "kick"
    ROLL = "roll"


@dataclass(frozen=True, slots=True)
class Observation:
    embodiment: str
    state: dict[str, Any]
    images: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillRequest:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillResult:
    embodiment: str
    skill: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)
