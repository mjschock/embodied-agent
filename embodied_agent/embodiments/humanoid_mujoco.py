from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


class HumanoidMuJoCo(Embodiment):
    """LeRobot Humanoid simulation using the official MuJoCo runtime controller.

    The MuJoCo controller provides reset/stand/observation directly. When configured
    with an official RLAgent policy directory, the adapter also exposes bounded
    velocity walking without pretending that open-loop velocity is navigation.

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
        policy_dir: str | Path | None = None,
        agent_factory: Callable[..., Any] | None = None,
        max_linear_x_mps: float = 0.75,
        max_linear_y_mps: float = 0.50,
        max_yaw_rate_rps: float = 0.80,
    ) -> None:
        self.name = name
        self.backend = "lerobot-humanoid-mujoco"
        self.runtime_root = None if runtime_root is None else Path(runtime_root)
        self.control_hz = float(control_hz)
        self.fixed_base = bool(fixed_base)
        self._controller_factory = controller_factory
        self.policy_dir = None if policy_dir is None else Path(policy_dir)
        self._agent_factory = agent_factory
        self.max_linear_x_mps = float(max_linear_x_mps)
        self.max_linear_y_mps = float(max_linear_y_mps)
        self.max_yaw_rate_rps = float(max_yaw_rate_rps)
        self._controller: Any | None = None
        self._agent: Any | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        capabilities = {Capability.OBSERVE, Capability.STAND}
        if self.policy_dir is not None or self._agent_factory is not None:
            capabilities.add(Capability.WALK)
        return frozenset(capabilities)

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

        if self.policy_dir is not None or self._agent_factory is not None:
            try:
                self._agent = self._create_agent(controller)
                self._agent.set_command_twist(0.0, 0.0, 0.0)
                self._agent.start()
            except Exception:
                self._agent = None
                controller.stop(disable_motors=True)
                self._controller = None
                raise

    async def disconnect(self) -> None:
        if self._agent is not None:
            self._agent.set_command_twist(0.0, 0.0, 0.0)
            self._agent.stop()
        self._agent = None
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
            if self._agent is not None:
                self._agent.set_command_twist(0.0, 0.0, 0.0)
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
            if self._agent is not None:
                self._agent.set_command_twist(0.0, 0.0, 0.0)
            else:
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
                detail="Humanoid commanded to stand in MuJoCo.",
                data={
                    "joint_position_rad": snapshot.get("joint_state_rad", []),
                    "sim_step_count": snapshot.get("sim_step_count"),
                    "policy_active": self._agent is not None,
                },
            )

        if request.name == "walk_velocity":
            agent = self._require_agent()
            lin_x = float(request.params.get("lin_x_mps", 0.0))
            lin_y = float(request.params.get("lin_y_mps", 0.0))
            yaw_rate = float(request.params.get("yaw_rate_rps", 0.0))
            duration_s = float(request.params.get("duration_s", 1.0))
            self._validate_walk_command(lin_x, lin_y, yaw_rate, duration_s)

            agent.set_command_twist(lin_x, lin_y, yaw_rate)
            try:
                await asyncio.sleep(duration_s)
            finally:
                agent.set_command_twist(0.0, 0.0, 0.0)

            controller.request_state_once()
            snapshot = self._snapshot()
            return SkillResult(
                embodiment=self.name,
                skill=request.name,
                ok=True,
                detail=(
                    "Humanoid executed a bounded locomotion-policy velocity command "
                    f"for {duration_s:.2f} s in MuJoCo."
                ),
                data={
                    "command": {
                        "lin_x_mps": lin_x,
                        "lin_y_mps": lin_y,
                        "yaw_rate_rps": yaw_rate,
                        "duration_s": duration_s,
                    },
                    "sim_step_count": snapshot.get("sim_step_count"),
                    "orientation_quaternion_xyzw": snapshot.get(
                        "orientation_quaternion_xyzw"
                    ),
                },
            )

        raise ValueError(f"Unsupported humanoid MuJoCo skill: {request.name}")

    def _require_agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError(
                "walk_velocity requires a locomotion policy. Configure policy_dir=... "
                "or provide agent_factory=..."
            )
        return self._agent

    def _validate_walk_command(
        self,
        lin_x: float,
        lin_y: float,
        yaw_rate: float,
        duration_s: float,
    ) -> None:
        if abs(lin_x) > self.max_linear_x_mps:
            raise ValueError(
                f"lin_x_mps exceeds configured limit {self.max_linear_x_mps:.3f}"
            )
        if abs(lin_y) > self.max_linear_y_mps:
            raise ValueError(
                f"lin_y_mps exceeds configured limit {self.max_linear_y_mps:.3f}"
            )
        if abs(yaw_rate) > self.max_yaw_rate_rps:
            raise ValueError(
                f"yaw_rate_rps exceeds configured limit {self.max_yaw_rate_rps:.3f}"
            )
        if duration_s <= 0.0 or duration_s > 5.0:
            raise ValueError("duration_s must be > 0 and <= 5.0")

    def _create_agent(self, controller: Any) -> Any:
        if self._agent_factory is not None:
            return self._agent_factory(controller=controller, policy_dir=self.policy_dir)

        if self.policy_dir is None:
            raise RuntimeError("policy_dir is required when agent_factory is not provided")

        policy_dir = self.policy_dir.expanduser()
        if not policy_dir.is_absolute() and self.runtime_root is not None:
            policy_dir = self.runtime_root.expanduser().resolve() / policy_dir
        policy_dir = policy_dir.resolve()
        if not policy_dir.exists():
            raise RuntimeError(f"Humanoid policy directory does not exist: {policy_dir}")

        policy_path = policy_dir / "policy.onnx"
        config_candidates = (
            policy_dir / "config.yaml",
            policy_dir / "config.yml",
            policy_dir / "model_25000_env.yaml",
            policy_dir / "config.json",
        )
        config_path = next((path for path in config_candidates if path.exists()), None)
        if not policy_path.exists() or config_path is None:
            raise RuntimeError(
                "Humanoid policy_dir must contain policy.onnx and a supported config "
                "file (config.yaml, config.yml, model_25000_env.yaml, or config.json)."
            )

        try:
            from control.rl_agent import RLAgent
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the official LeRobot Humanoid RLAgent from runtime_root."
            ) from exc

        return RLAgent.from_files(
            controller,
            config_path=str(config_path),
            policy_path=str(policy_path),
        )

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
