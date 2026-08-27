from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


class XLeRobotMuJoCo(Embodiment):
    """XLeRobot simulation backed directly by the upstream MuJoCo model.

    The upstream keyboard demo mixes UI, input handling, and control. This adapter
    loads the same MJCF but exposes only stable semantic mobile-base skills. Arm
    manipulation stays disabled until a manipulation policy owns the arm actuators.
    """

    def __init__(
        self,
        name: str = "xlerobot",
        *,
        runtime_root: str | Path | None = None,
        scene_path: str | Path | None = None,
        max_linear_x_mps: float = 0.5,
        max_linear_y_mps: float = 0.5,
        max_yaw_rate_rps: float = 0.8,
        position_tolerance_m: float = 0.03,
        yaw_tolerance_rad: float = 0.05,
        navigation_kp_linear: float = 2.0,
        navigation_kp_yaw: float = 3.0,
        sim_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.name = name
        self.backend = "xlerobot-mujoco"
        self.runtime_root = None if runtime_root is None else Path(runtime_root)
        self.scene_path = None if scene_path is None else Path(scene_path)
        self.max_linear_x_mps = float(max_linear_x_mps)
        self.max_linear_y_mps = float(max_linear_y_mps)
        self.max_yaw_rate_rps = float(max_yaw_rate_rps)
        self.position_tolerance_m = float(position_tolerance_m)
        self.yaw_tolerance_rad = float(yaw_tolerance_rad)
        self.navigation_kp_linear = float(navigation_kp_linear)
        self.navigation_kp_yaw = float(navigation_kp_yaw)
        self._sim_factory = sim_factory
        self._sim: Any | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.OBSERVE, Capability.NAVIGATE})

    async def connect(self) -> None:
        if self._sim is not None:
            return
        factory = self._sim_factory or _MuJoCoXLeRobotRuntime
        self._sim = factory(self._resolve_scene_path())

    async def disconnect(self) -> None:
        if self._sim is not None:
            self._sim.zero_base_velocity()
            close = getattr(self._sim, "close", None)
            if callable(close):
                close()
        self._sim = None

    async def observe(self) -> Observation:
        state = self._state()
        return Observation(
            embodiment=self.name,
            state={
                "backend": self.backend,
                "sim_time_s": state["sim_time_s"],
                "base": {
                    "x_m": state["x_m"],
                    "y_m": state["y_m"],
                    "yaw_rad": state["yaw_rad"],
                    "linear_velocity_world_mps": state["linear_velocity_world_mps"],
                    "yaw_rate_rps": state["yaw_rate_rps"],
                },
                "arm_joint_position_rad": state.get("arm_joint_position_rad", {}),
            },
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        sim = self._require_sim()

        if request.name == "reset":
            sim.reset()
            return SkillResult(
                embodiment=self.name,
                skill=request.name,
                ok=True,
                detail="XLeRobot MuJoCo simulation reset.",
                data=self._pose_data(self._state()),
            )

        if request.name == "drive_velocity":
            lin_x = float(request.params.get("lin_x_mps", 0.0))
            lin_y = float(request.params.get("lin_y_mps", 0.0))
            yaw_rate = float(request.params.get("yaw_rate_rps", 0.0))
            duration_s = float(request.params.get("duration_s", 1.0))
            self._validate_velocity_command(lin_x, lin_y, yaw_rate, duration_s)
            await self._drive_for(lin_x, lin_y, yaw_rate, duration_s)
            state = self._state()
            return SkillResult(
                embodiment=self.name,
                skill=request.name,
                ok=True,
                detail=f"XLeRobot drove for {duration_s:.2f} s in MuJoCo.",
                data={
                    **self._pose_data(state),
                    "command": {
                        "lin_x_mps": lin_x,
                        "lin_y_mps": lin_y,
                        "yaw_rate_rps": yaw_rate,
                        "duration_s": duration_s,
                    },
                },
            )

        if request.name == "navigate_to":
            x_m = float(request.params["x_m"])
            y_m = float(request.params["y_m"])
            yaw_rad = float(request.params.get("yaw_rad", 0.0))
            max_duration_s = float(request.params.get("max_duration_s", 10.0))
            if max_duration_s <= 0.0 or max_duration_s > 30.0:
                raise ValueError("max_duration_s must be > 0 and <= 30.0")
            return await self._navigate_to(x_m, y_m, yaw_rad, max_duration_s)

        raise ValueError(f"Unsupported XLeRobot MuJoCo skill: {request.name}")

    async def _drive_for(
        self, lin_x: float, lin_y: float, yaw_rate: float, duration_s: float
    ) -> None:
        sim = self._require_sim()
        steps = max(1, math.ceil(duration_s / sim.timestep_s))
        sim.set_base_velocity(lin_x, lin_y, yaw_rate)
        try:
            for i in range(steps):
                sim.step()
                if i % 100 == 0:
                    await asyncio.sleep(0)
        finally:
            sim.zero_base_velocity()

    async def _navigate_to(
        self, target_x: float, target_y: float, target_yaw: float, max_duration_s: float
    ) -> SkillResult:
        sim = self._require_sim()
        max_steps = max(1, math.ceil(max_duration_s / sim.timestep_s))
        try:
            for i in range(max_steps):
                state = self._state()
                dx = target_x - state["x_m"]
                dy = target_y - state["y_m"]
                yaw_error = _wrap_angle(target_yaw - state["yaw_rad"])
                position_error = math.hypot(dx, dy)
                if (
                    position_error <= self.position_tolerance_m
                    and abs(yaw_error) <= self.yaw_tolerance_rad
                ):
                    return SkillResult(
                        embodiment=self.name,
                        skill="navigate_to",
                        ok=True,
                        detail="XLeRobot reached the requested MuJoCo pose.",
                        data={
                            **self._pose_data(state),
                            "position_error_m": position_error,
                            "yaw_error_rad": yaw_error,
                        },
                    )

                yaw = state["yaw_rad"]
                body_dx = math.cos(yaw) * dx + math.sin(yaw) * dy
                body_dy = -math.sin(yaw) * dx + math.cos(yaw) * dy
                lin_x = _clip(
                    self.navigation_kp_linear * body_dx,
                    -self.max_linear_x_mps,
                    self.max_linear_x_mps,
                )
                lin_y = _clip(
                    self.navigation_kp_linear * body_dy,
                    -self.max_linear_y_mps,
                    self.max_linear_y_mps,
                )
                yaw_rate = _clip(
                    self.navigation_kp_yaw * yaw_error,
                    -self.max_yaw_rate_rps,
                    self.max_yaw_rate_rps,
                )
                sim.set_base_velocity(lin_x, lin_y, yaw_rate)
                sim.step()
                if i % 100 == 0:
                    await asyncio.sleep(0)
        finally:
            sim.zero_base_velocity()

        state = self._state()
        dx = target_x - state["x_m"]
        dy = target_y - state["y_m"]
        yaw_error = _wrap_angle(target_yaw - state["yaw_rad"])
        return SkillResult(
            embodiment=self.name,
            skill="navigate_to",
            ok=False,
            detail="XLeRobot did not reach the requested pose before the simulation timeout.",
            data={
                **self._pose_data(state),
                "position_error_m": math.hypot(dx, dy),
                "yaw_error_rad": yaw_error,
            },
        )

    def _validate_velocity_command(
        self, lin_x: float, lin_y: float, yaw_rate: float, duration_s: float
    ) -> None:
        if abs(lin_x) > self.max_linear_x_mps:
            raise ValueError(f"lin_x_mps exceeds configured limit {self.max_linear_x_mps:.3f}")
        if abs(lin_y) > self.max_linear_y_mps:
            raise ValueError(f"lin_y_mps exceeds configured limit {self.max_linear_y_mps:.3f}")
        if abs(yaw_rate) > self.max_yaw_rate_rps:
            raise ValueError(
                f"yaw_rate_rps exceeds configured limit {self.max_yaw_rate_rps:.3f}"
            )
        if duration_s <= 0.0 or duration_s > 5.0:
            raise ValueError("duration_s must be > 0 and <= 5.0")

    def _state(self) -> dict[str, Any]:
        return dict(self._require_sim().get_state())

    @staticmethod
    def _pose_data(state: dict[str, Any]) -> dict[str, float]:
        return {
            "x_m": float(state["x_m"]),
            "y_m": float(state["y_m"]),
            "yaw_rad": float(state["yaw_rad"]),
        }

    def _require_sim(self) -> Any:
        if self._sim is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._sim

    def _resolve_scene_path(self) -> Path:
        if self.scene_path is not None:
            path = self.scene_path.expanduser().resolve()
        elif self.runtime_root is not None:
            path = (
                self.runtime_root.expanduser().resolve()
                / "simulation"
                / "mujoco"
                / "scene.xml"
            )
        else:
            raise RuntimeError(
                "XLeRobotMuJoCo requires runtime_root pointing to a Vector-Wangel/XLeRobot "
                "checkout, or an explicit scene_path."
            )
        if self._sim_factory is None and not path.exists():
            raise RuntimeError(f"XLeRobot MuJoCo scene does not exist: {path}")
        return path


class _MuJoCoXLeRobotRuntime:
    _ARM_JOINTS = (
        "Rotation_R",
        "Pitch_R",
        "Elbow_R",
        "Wrist_Pitch_R",
        "Wrist_Roll_R",
        "Jaw_R",
        "Rotation_L",
        "Pitch_L",
        "Elbow_L",
        "Wrist_Pitch_L",
        "Wrist_Roll_L",
        "Jaw_L",
    )

    def __init__(self, scene_path: Path) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("XLeRobotMuJoCo requires the `mujoco` package.") from exc

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        joint_names = ("slide_joint_x", "slide_joint_y", "hinge_joint_z", *self._ARM_JOINTS)
        self._joint_qpos_adr = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"XLeRobot MJCF is missing expected joint: {name}")
            self._joint_qpos_adr[name] = int(self.model.jnt_qposadr[joint_id])
        self._joint_dof_adr = {
            name: int(
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in ("slide_joint_x", "slide_joint_y", "hinge_joint_z")
        }
        self._base_actuators = {
            key: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for key, name in {
                "x": "slider_actuator_x",
                "y": "slider_actuator_y",
                "yaw": "hinge_actuator_z",
            }.items()
        }
        self._wheel_actuators = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("wheel1", "wheel2", "wheel3")
        ]
        if any(idx < 0 for idx in (*self._base_actuators.values(), *self._wheel_actuators)):
            raise RuntimeError("XLeRobot MJCF is missing expected mobile-base actuators")

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

    def set_base_velocity(self, lin_x: float, lin_y: float, yaw_rate: float) -> None:
        yaw = float(self.data.qpos[self._joint_qpos_adr["hinge_joint_z"]])
        self.data.ctrl[self._base_actuators["x"]] = lin_x * math.cos(yaw) - lin_y * math.sin(yaw)
        self.data.ctrl[self._base_actuators["y"]] = lin_x * math.sin(yaw) + lin_y * math.cos(yaw)
        self.data.ctrl[self._base_actuators["yaw"]] = yaw_rate

        radius = 0.1
        wheel_scale = 20.0
        wheel_commands = (
            lin_y - radius * yaw_rate,
            -math.sqrt(3.0) * 0.5 * lin_x - 0.5 * lin_y - radius * yaw_rate,
            math.sqrt(3.0) * 0.5 * lin_x - 0.5 * lin_y - radius * yaw_rate,
        )
        for actuator_id, command in zip(self._wheel_actuators, wheel_commands):
            self.data.ctrl[actuator_id] = wheel_scale * command

    def zero_base_velocity(self) -> None:
        for actuator_id in (*self._base_actuators.values(), *self._wheel_actuators):
            self.data.ctrl[actuator_id] = 0.0

    def step(self) -> None:
        self._mujoco.mj_step(self.model, self.data)

    def get_state(self) -> dict[str, Any]:
        x_dof = self._joint_dof_adr["slide_joint_x"]
        y_dof = self._joint_dof_adr["slide_joint_y"]
        yaw_dof = self._joint_dof_adr["hinge_joint_z"]
        return {
            "sim_time_s": float(self.data.time),
            "x_m": float(self.data.qpos[self._joint_qpos_adr["slide_joint_x"]]),
            "y_m": float(self.data.qpos[self._joint_qpos_adr["slide_joint_y"]]),
            "yaw_rad": float(self.data.qpos[self._joint_qpos_adr["hinge_joint_z"]]),
            "linear_velocity_world_mps": [
                float(self.data.qvel[x_dof]),
                float(self.data.qvel[y_dof]),
            ],
            "yaw_rate_rps": float(self.data.qvel[yaw_dof]),
            "arm_joint_position_rad": {
                name: float(self.data.qpos[self._joint_qpos_adr[name]])
                for name in self._ARM_JOINTS
            },
        }


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
