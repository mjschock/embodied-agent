from __future__ import annotations

from typing import Any

from embodied_agent.core import Embodiment, Observation, SkillRequest, SkillResult


class DeterministicSimStub(Embodiment):
    """Temporary adapter proving the architecture before real simulators land."""

    def __init__(self, name: str, backend: str = "stub-sim") -> None:
        self.name = name
        self.backend = backend
        self._connected = False
        self._state: dict[str, Any] = {"connected": False}

    async def connect(self) -> None:
        self._connected = True
        self._state["connected"] = True

    async def disconnect(self) -> None:
        self._connected = False
        self._state["connected"] = False

    async def observe(self) -> Observation:
        if not self._connected:
            raise RuntimeError(f"{self.name} is not connected")
        return Observation(
            embodiment=self.name,
            state={"backend": self.backend, **self._state},
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"{self.name} is not connected")

    def _result(
        self,
        request: SkillRequest,
        *,
        detail: str,
        **data: Any,
    ) -> SkillResult:
        return SkillResult(
            embodiment=self.name,
            skill=request.name,
            ok=True,
            detail=detail,
            data=data,
        )
