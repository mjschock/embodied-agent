from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Callable

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


class _UpstreamMicroduckRuntime:
    """Thin headless wrapper around microduck_rl's CPU MuJoCo inference reference.

    The pinned upstream `scripts/infer_policy.py` owns the 61-D observation builder,
    ONNX session switching, and 14-action -> joint-target mapping. This wrapper owns
    lifecycle plus bounded semantic behaviors for embodied-agent.
    """

    CONTROL_DECIMATION = 4
    CONTROL_DT_S = 0.02  # upstream model timestep 0.005 * decimation 4 = 50 Hz
    MAX_FORWARD_MPS = 0.25
    MAX_BACKWARD_MPS = 0.20
    MAX_YAW_RATE_RPS = 1.0
    KICK_POLICY_S = 0.5
    POST_KICK_SETTLE_S = 0.4
    RECOVERY_SETTLE_STEPS = 15
    RECOVERY_UPRIGHT_STEPS = 50
    RECOVERY_MAX_STEPS = 300
    UPRIGHT_GRAVITY_Z = -0.85

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        walking_policy_path: str | Path,
        standing_policy_path: str | Path,
        kick_left_policy_path: str | Path | None = None,
        kick_right_policy_path: str | Path | None = None,
        scene_path: str | Path | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.walking_policy_path = Path(walking_policy_path)
        self.standing_policy_path = Path(standing_policy_path)
        self.kick_left_policy_path = Path(kick_left_policy_path) if kick_left_policy_path else None
        self.kick_right_policy_path = Path(kick_right_policy_path) if kick_right_policy_path else None
        self.scene_path = Path(scene_path) if scene_path else (
            self.runtime_root / "src/mjlab_microduck/robot/microduck/scene.xml"
        )
        self.model: Any | None = None
        self.data: Any | None = None
        self.policy: Any | None = None
        self._mujoco: Any | None = None

    def start(self) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by real setup, not unit tests
            raise RuntimeError(
                "Microduck MuJoCo support requires the 'microduck-sim' optional dependency."
            ) from exc

        inference_path = self.runtime_root / "scripts/infer_policy.py"
        if not inference_path.exists():
            raise FileNotFoundError(f"Microduck inference reference not found: {inference_path}")
        for path in (self.scene_path, self.walking_policy_path, self.standing_policy_path):
            if not path.exists():
                raise FileNotFoundError(f"Microduck runtime asset not found: {path}")

        spec = importlib.util.spec_from_file_location("embodied_agent_microduck_infer", inference_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import Microduck inference reference: {inference_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.policy = module.PolicyInference(
            self.model,
            self.data,
            walking_onnx_path=str(self.walking_policy_path),
            standing_onnx_path=str(self.standing_policy_path),
            kick_left_onnx_path=str(self.kick_left_policy_path) if self.kick_left_policy_path else None,
            kick_right_onnx_path=str(self.kick_right_policy_path) if self.kick_right_policy_path else None,
            kick_duration=self.KICK_POLICY_S,
            new_cmd_obs=True,
        )
        self.reset()

    def stop(self) -> None:
        self.policy = None
        self.data = None
        self.model = None
        self._mujoco = None

    def _require_started(self) -> tuple[Any, Any, Any, Any]:
        if self._mujoco is None or self.model is None or self.data is None or self.policy is None:
            raise RuntimeError("Microduck runtime is not connected")
        return self._mujoco, self.model, self.data, self.policy

    def _set_policy(self, name: str) -> None:
        _, _, _, policy = self._require_started()
        if name == "walking":
            session = policy.walking_session
        elif name == "standing":
            session = policy.standing_session
        else:
            raise ValueError(f"unsupported Microduck policy mode: {name}")
        if session is None:
            raise RuntimeError(f"Microduck {name} policy is not configured")
        policy.behavior_mode = None
        policy.current_policy = name
        policy.ort_session = session
        policy.vel_cmd[:] = 0.0
        policy._update_command()

    def _step_policy_once(self) -> None:
        mujoco, model, data, policy = self._require_started()
        policy.update_ground_pick_phase(self.CONTROL_DT_S)
        policy.update_behavior(self.CONTROL_DT_S)
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(self.CONTROL_DECIMATION):
            mujoco.mj_step(model, data)

    def _step_physics_only(self) -> None:
        mujoco, model, data, _ = self._require_started()
        for _ in range(self.CONTROL_DECIMATION):
            mujoco.mj_step(model, data)

    def _run_steps(self, steps: int) -> None:
        for _ in range(max(0, steps)):
            self._step_policy_once()

    def reset(self) -> dict[str, Any]:
        mujoco, model, data, policy = self._require_started()
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if key_id < 0:
            raise RuntimeError("Microduck scene does not contain the STAND keyframe")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        policy.last_action[:] = 0.0
        policy.behavior_mode = None
        policy.sit_mode = False
        self._set_policy("standing")
        return self.observe()

    def observe(self) -> dict[str, Any]:
        _, _, data, policy = self._require_started()
        projected_gravity = policy.get_projected_gravity()
        joint_pos = data.qpos[policy.joint_qpos_indices].copy()
        joint_vel = data.qvel[policy.joint_qvel_indices].copy()
        return {
            "position_m": tuple(float(v) for v in data.qpos[:3]),
            "orientation_wxyz": tuple(float(v) for v in data.qpos[3:7]),
            "projected_gravity": tuple(float(v) for v in projected_gravity),
            "joint_position_rad": tuple(float(v) for v in joint_pos),
            "joint_velocity_rps": tuple(float(v) for v in joint_vel),
            "policy": str(policy.current_policy),
            "behavior": policy.behavior_mode,
            "control_hz": 50.0,
        }

    def stand(self, duration_s: float = 1.0) -> dict[str, Any]:
        if not (0.02 <= duration_s <= 5.0):
            raise ValueError("stand duration_s must be between 0.02 and 5.0")
        self._set_policy("standing")
        self._run_steps(math.ceil(duration_s / self.CONTROL_DT_S))
        state = self.observe()
        state["upright"] = state["projected_gravity"][2] < self.UPRIGHT_GRAVITY_Z
        return state

    def walk_velocity(
        self,
        *,
        lin_x_mps: float = 0.0,
        lin_y_mps: float = 0.0,
        yaw_rate_rps: float = 0.0,
        duration_s: float = 1.0,
    ) -> dict[str, Any]:
        if not (-self.MAX_BACKWARD_MPS <= lin_x_mps <= self.MAX_FORWARD_MPS):
            raise ValueError(
                f"Microduck lin_x_mps must be between {-self.MAX_BACKWARD_MPS} and {self.MAX_FORWARD_MPS}"
            )
        if abs(lin_y_mps) > 1e-6:
            raise ValueError("Microduck walking policy does not support lateral velocity commands")
        if not (-self.MAX_YAW_RATE_RPS <= yaw_rate_rps <= self.MAX_YAW_RATE_RPS):
            raise ValueError(
                f"Microduck yaw_rate_rps must be between {-self.MAX_YAW_RATE_RPS} and {self.MAX_YAW_RATE_RPS}"
            )
        if not (0.02 <= duration_s <= 5.0):
            raise ValueError("Microduck walk duration_s must be between 0.02 and 5.0")

        _, _, _, policy = self._require_started()
        self._set_policy("walking")
        policy.vel_cmd[:] = (lin_x_mps, 0.0, yaw_rate_rps)
        policy._update_command()
        self._run_steps(math.ceil(duration_s / self.CONTROL_DT_S))
        policy.vel_cmd[:] = 0.0
        self._set_policy("standing")
        return self.observe()

    def kick(self, *, foot: str) -> dict[str, Any]:
        if foot not in {"left", "right"}:
            raise ValueError("Microduck kick foot must be 'left' or 'right'")
        _, _, _, policy = self._require_started()
        behavior = f"kick_{foot}"
        if behavior not in policy.behavior_sessions:
            raise RuntimeError(f"Microduck {foot} kick policy is not configured")
        self._set_policy("walking")
        policy.trigger_behavior(behavior)
        while policy.behavior_mode is not None:
            self._step_policy_once()
        self._set_policy("standing")
        self._run_steps(math.ceil(self.POST_KICK_SETTLE_S / self.CONTROL_DT_S))
        state = self.observe()
        state["foot"] = foot
        return state

    def recover(self) -> dict[str, Any]:
        _, _, _, policy = self._require_started()

        # Match the current Microduck browser/runtime recovery envelope: a short
        # settle, then the stand/get-up policy with all commands zeroed, requiring
        # one continuous second upright before reporting success and giving up
        # after six seconds of policy attempts.
        for _ in range(self.RECOVERY_SETTLE_STEPS):
            self._step_physics_only()
        policy.last_action[:] = 0.0
        self._set_policy("standing")

        upright_steps = 0
        attempted = 0
        for attempted in range(1, self.RECOVERY_MAX_STEPS + 1):
            self._step_policy_once()
            gravity_z = float(policy.get_projected_gravity()[2])
            upright_steps = upright_steps + 1 if gravity_z < self.UPRIGHT_GRAVITY_Z else 0
            if upright_steps >= self.RECOVERY_UPRIGHT_STEPS:
                state = self.observe()
                state.update({"recovered": True, "recovery_steps": attempted})
                return state

        state = self.observe()
        state.update({"recovered": False, "recovery_steps": attempted})
        return state


class MicroduckMuJoCo(Embodiment):
    """Policy-backed Microduck simulation using upstream MuJoCo + ONNX behavior policies."""

    backend = "microduck_mujoco"

    def __init__(
        self,
        *,
        name: str = "microduck",
        runtime_root: str | Path,
        walking_policy_path: str | Path,
        standing_policy_path: str | Path,
        kick_left_policy_path: str | Path | None = None,
        kick_right_policy_path: str | Path | None = None,
        scene_path: str | Path | None = None,
        runtime_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.name = name
        self.runtime_root = str(runtime_root)
        self.walking_policy_path = str(walking_policy_path)
        self.standing_policy_path = str(standing_policy_path)
        self.kick_left_policy_path = str(kick_left_policy_path) if kick_left_policy_path else None
        self.kick_right_policy_path = str(kick_right_policy_path) if kick_right_policy_path else None
        self.scene_path = str(scene_path) if scene_path else None
        self._runtime_factory = runtime_factory
        self._runtime: Any | None = None
        self._connected = False

    @property
    def capabilities(self) -> frozenset[Capability]:
        caps = {Capability.OBSERVE, Capability.STAND, Capability.WALK, Capability.RECOVER}
        if self.kick_left_policy_path and self.kick_right_policy_path:
            caps.add(Capability.KICK)
        return frozenset(caps)

    def _build_runtime(self) -> Any:
        if self._runtime_factory is not None:
            return self._runtime_factory()
        return _UpstreamMicroduckRuntime(
            runtime_root=self.runtime_root,
            walking_policy_path=self.walking_policy_path,
            standing_policy_path=self.standing_policy_path,
            kick_left_policy_path=self.kick_left_policy_path,
            kick_right_policy_path=self.kick_right_policy_path,
            scene_path=self.scene_path,
        )

    def _require_runtime(self) -> Any:
        if not self._connected or self._runtime is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._runtime

    async def connect(self) -> None:
        if self._connected:
            return
        runtime = self._build_runtime()
        runtime.start()
        self._runtime = runtime
        self._connected = True

    async def disconnect(self) -> None:
        if self._runtime is not None:
            self._runtime.stop()
        self._runtime = None
        self._connected = False

    async def observe(self) -> Observation:
        state = dict(self._require_runtime().observe())
        return Observation(self.name, state)

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        runtime = self._require_runtime()
        skill = request.name
        params = dict(request.params)

        if skill == "reset":
            state = runtime.reset()
            return SkillResult(self.name, skill, True, "Microduck reset to STAND keyframe.", dict(state))
        if skill == "stand":
            state = runtime.stand()
            ok = bool(state.get("upright", True))
            return SkillResult(self.name, skill, ok, "Microduck standing policy executed.", dict(state))
        if skill == "walk_velocity":
            state = runtime.walk_velocity(**params)
            return SkillResult(self.name, skill, True, "Microduck walking policy executed.", dict(state))
        if skill == "kick":
            state = runtime.kick(**params)
            return SkillResult(self.name, skill, True, "Microduck kick policy executed.", dict(state))
        if skill == "recover":
            state = runtime.recover()
            ok = bool(state.get("recovered", False))
            detail = "Microduck recovered to an upright pose." if ok else "Microduck recovery policy timed out."
            return SkillResult(self.name, skill, ok, detail, dict(state))

        return SkillResult(self.name, skill, False, f"Unsupported Microduck skill: {skill}")
