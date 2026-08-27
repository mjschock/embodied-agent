from __future__ import annotations

from embodied_agent.core import Capability, SkillRequest, SkillResult

from .base_stub import DeterministicSimStub


class CrazyflieSim(DeterministicSimStub):
    """Stub for the future gym-pybullet-drones / Crazyflie adapter."""

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {Capability.OBSERVE, Capability.NAVIGATE, Capability.FLY}
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        self._require_connected()

        if request.name == "takeoff":
            altitude = float(request.params.get("altitude_m", 1.0))
            self._state.update({"airborne": True, "altitude_m": altitude})
            return self._result(
                request,
                detail=f"Crazyflie took off to {altitude:.2f} m in simulation.",
                altitude_m=altitude,
            )

        if request.name == "goto":
            position = tuple(request.params["position"])
            self._state["position"] = position
            return self._result(
                request,
                detail=f"Crazyflie moved to {position} in simulation.",
                position=position,
            )

        if request.name == "land":
            self._state.update({"airborne": False, "altitude_m": 0.0})
            return self._result(
                request,
                detail="Crazyflie landed in simulation.",
                altitude_m=0.0,
            )

        raise ValueError(f"Unsupported Crazyflie skill: {request.name}")
