from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


class HumanoidMuJoCo(Embodiment):
    """LeRobot Humanoid simulation using the official MuJoCo runtime controller.

    This adapter intentionally exposes only skills that the underlying runtime can
    execute without pretending a locomotion policy exists. Walking will be added as
    a policy-backed layer over this controller in a subsequent increment.

    The official runtime is currently a separate repository. Pass ``runtime_root``
    to its checkout, or inject ``controller_factory`` in tests/custom deployments.
    """

    _NEUTRAL_SIDE = {
        "hipz": 0.0,
        "hipx": 0.0,
        "hipy": 0.0,
        "knee": 0.0,
        "ankle_pitch": 0.0,
        "ankle_roll": 0.0,
    }

    def __init__(
        self,
        name: str = "humanoid",
        *,
        runtime_root: str | Path | None = None,
        control_hz: float = 200.0,
        fixed_base: bool = False,
        controller_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.backend = "lerobot-humanoid-mujoco"
        self.runtime_root = None if runtime_root is None else Path(runtime_root)
        self.control_hz = float(control_hz)
        self.fixed_base = bool(fixed_base)
        self._controller_factory = controller_factory
        self._controller: Any | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        # WALK is intentionally absent until an official locomotion policy runner is
        # connected. The MuJoCo controller itself is joint-level, not a navigator.
        return frozenset({Capability.OBSERVE, Capability.STAND})

    async def connect(self) -> None:
        if self._controller is not None:
            return

        factory = self._controller_factory or self._load_default_controller_factory()
        controller = factory(
            control_hz=self.control_hz,
            fixed_base=self.fixed_base,
        )
        controller.start(mode="control", auto_enable=True)
        controller.request_state_once()
        self._controller = controller

    async def disconnect(self) -> None:
        if self._controller is not None:
            self._controller.stop(disable_motors=True)
        self._controller = None

    async def observe(self) -> Observation:
        snapshot = self._snapshot()
        return Observation(
            embodiment=self.name,
            state={
                "backend": self.backend,
                "mode": snapshot.get("mode"),
                "sim_step_count": snapshot.get("sim_step_count"),
                "sim_reset_count": snapshot.get("sim_reset_count"),
                "fixed_base": snapshot.get("fixed_base"),
                "joint_position_rad": snapshot.get("joint_state_rad", []),
                "joint_velocity_radps": snapshot.get("joint_velocity_rad_s", []),
                "joint_torque_nm": snapshot.get("joint_torque_nm", []),
                "orientation_quaternion_xyzw": snapshot.get(
                    "orientation_quaternion_xyzw"
                ),
                "imu": snapshot.get("imu"),
            },
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        controller = self._require_controller()

        if request.name == "reset":
            controller.reset()
            controller.request_state_once()
            snapshot = self._snapshot()
            return SkillResult(
                embodiment=self.name,
                skill=request.name,
                ok=True,
                detail="Humanoid MuJoCo simulation reset to its reference state.",
                data={
                    "sim_reset_count": snapshot.get("sim_reset_count"),
                    "joint_position_rad": snapshot.get("joint_state_rad", []),
                },
            )

        if request.name == "stand":
            controller.set_action(
                left=dict(self._NEUTRAL_SIDE),
                right=dict(self._NEUTRAL_SIDE),
            )
            controller.request_state_once()
            snapshot = self._snapshot()
            return SkillResult(
                embodiment=self.name,
                skill=request.name,
                ok=True,
                detail="Humanoid commanded to its neutral standing posture in MuJoCo.",
                data={
                    "joint_position_rad": snapshot.get("joint_state_rad", []),
                    "sim_step_count": snapshot.get("sim_step_count"),
                },
            )

        raise ValueError(f"Unsupported humanoid MuJoCo skill: {request.name}")

    def _snapshot(self) -> dict[str, Any]:
        controller = self._require_controller()
        return dict(controller.get_combined_state_snapshot(include_joint_state=True))

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._controller

    def _load_default_controller_factory(self) -> Callable[..., Any]:
        if self.runtime_root is not None:
            runtime_root = self.runtime_root.expanduser().resolve()
            if not runtime_root.exists():
                raise RuntimeError(
                    f"LeRobot Humanoid runtime checkout does not exist: {runtime_root}"
                )
            root_text = str(runtime_root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)

        try:
            from robot.sim_robot import SimBipedalRobotController
        except ImportError as exc:
            raise RuntimeError(
                "HumanoidMuJoCo requires the official lerobot-humanoid-runtime "
                "checkout with its submodules initialized. Pass runtime_root=... "
                "or make that repository importable."
            ) from exc

        return SimBipedalRobotController
