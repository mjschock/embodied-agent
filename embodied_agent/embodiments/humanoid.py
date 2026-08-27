from __future__ import annotations

from embodied_agent.core import Capability, SkillRequest, SkillResult

from .base_stub import DeterministicSimStub


class HumanoidSim(DeterministicSimStub):
    """Stub for the official LeRobot Humanoid MuJoCo runtime adapter."""

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {Capability.OBSERVE, Capability.WALK, Capability.STAND}
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        self._require_connected()

        if request.name == "stand":
            self._state["posture"] = "standing"
            return self._result(
                request,
                detail="Humanoid is standing in simulation.",
                posture="standing",
            )

        if request.name == "walk_to":
            target = request.params["target"]
            self._state["location"] = target
            return self._result(
                request,
                detail=f"Humanoid walked to {target} in simulation.",
                location=target,
            )

        raise ValueError(f"Unsupported humanoid skill: {request.name}")
