from __future__ import annotations

from embodied_agent.core import Capability, SkillRequest, SkillResult

from .base_stub import DeterministicSimStub


class XLeRobotSim(DeterministicSimStub):
    """Stub for the future XLeRobot ManiSkill / LeRobot simulation adapter."""

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {Capability.OBSERVE, Capability.NAVIGATE, Capability.MANIPULATE}
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        self._require_connected()

        if request.name == "navigate_to":
            target = request.params["target"]
            self._state["location"] = target
            return self._result(
                request,
                detail=f"XLeRobot navigated to {target} in simulation.",
                location=target,
            )

        if request.name == "pick":
            obj = request.params["object"]
            self._state["holding"] = obj
            return self._result(
                request,
                detail=f"XLeRobot picked {obj} in simulation.",
                holding=obj,
            )

        if request.name == "place":
            obj = self._state.pop("holding", None)
            target = request.params["target"]
            return self._result(
                request,
                detail=f"XLeRobot placed {obj} at {target} in simulation.",
                object=obj,
                target=target,
            )

        raise ValueError(f"Unsupported XLeRobot skill: {request.name}")
