from __future__ import annotations

from collections.abc import Callable
from typing import Any

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


class CrazyfliePyBullet(Embodiment):
    """Crazyflie 2.x simulation backed by gym-pybullet-drones VelocityAviary.

    The agent-facing skills are position goals. Internally, the adapter closes the
    loop with VelocityAviary's velocity controller, so high-level callers never
    emit motor RPM commands.
    """

    def __init__(
        self,
        name: str = "crazyflie",
        *,
        gui: bool = False,
        seed: int = 0,
        ctrl_freq_hz: int = 60,
        initial_position: tuple[float, float, float] = (0.0, 0.0, 0.1),
        position_tolerance_m: float = 0.03,
        slow_radius_m: float = 0.30,
        env_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.backend = "gym-pybullet-drones"
        self.gui = gui
        self.seed = seed
        self.ctrl_freq_hz = ctrl_freq_hz
        self.initial_position = initial_position
        self.position_tolerance_m = position_tolerance_m
        self.slow_radius_m = slow_radius_m
        self._env_factory = env_factory
        self._env: Any | None = None
        self._obs: Any | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {Capability.OBSERVE, Capability.NAVIGATE, Capability.FLY}
        )

    async def connect(self) -> None:
        if self._env is not None:
            return

        factory = self._env_factory or self._load_default_env_factory()
        self._env = factory(
            gui=self.gui,
            ctrl_freq_hz=self.ctrl_freq_hz,
            initial_position=self.initial_position,
        )
        self._obs, _ = self._env.reset(seed=self.seed)

    async def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._obs = None

    async def observe(self) -> Observation:
        state = self._state_vector()
        return Observation(
            embodiment=self.name,
            state={
                "backend": self.backend,
                "position_m": state[0:3].tolist(),
                "quaternion_xyzw": state[3:7].tolist(),
                "rpy_rad": state[7:10].tolist(),
                "linear_velocity_mps": state[10:13].tolist(),
                "angular_velocity_radps": state[13:16].tolist(),
                "motor_rpm": state[16:20].tolist(),
            },
        )

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if request.name == "takeoff":
            altitude_m = float(request.params.get("altitude_m", 1.0))
            if altitude_m <= 0:
                raise ValueError("altitude_m must be positive")
            current = self._state_vector()[0:3]
            return self._move_to(
                request,
                (float(current[0]), float(current[1]), altitude_m),
            )

        if request.name == "goto":
            position = self._coerce_position(request.params["position"])
            return self._move_to(request, position)

        if request.name == "land":
            landing_height_m = float(request.params.get("height_m", 0.05))
            if landing_height_m < 0:
                raise ValueError("height_m must be non-negative")
            current = self._state_vector()[0:3]
            return self._move_to(
                request,
                (float(current[0]), float(current[1]), landing_height_m),
            )

        raise ValueError(f"Unsupported Crazyflie skill: {request.name}")

    def _move_to(
        self,
        request: SkillRequest,
        target: tuple[float, float, float],
    ) -> SkillResult:
        np = self._numpy()
        env = self._require_env()
        target_vec = np.asarray(target, dtype=float)
        max_duration_s = float(request.params.get("timeout_s", 10.0))
        if max_duration_s <= 0:
            raise ValueError("timeout_s must be positive")

        max_steps = max(1, int(max_duration_s * self.ctrl_freq_hz))
        final_error = float("inf")

        for step_index in range(max_steps):
            state = self._state_vector()
            error = target_vec - state[0:3]
            final_error = float(np.linalg.norm(error))
            if final_error <= self.position_tolerance_m:
                self._step(np.zeros((1, 4), dtype=np.float32))
                return SkillResult(
                    embodiment=self.name,
                    skill=request.name,
                    ok=True,
                    detail=(
                        f"Crazyflie reached {tuple(float(v) for v in target_vec)} "
                        "in PyBullet simulation."
                    ),
                    data={
                        "target_position_m": target_vec.tolist(),
                        "final_position_m": self._state_vector()[0:3].tolist(),
                        "position_error_m": final_error,
                        "steps": step_index,
                    },
                )

            speed_fraction = min(1.0, max(0.08, final_error / self.slow_radius_m))
            action = np.zeros((1, 4), dtype=np.float32)
            action[0, 0:3] = error
            action[0, 3] = speed_fraction
            self._step(action)

        return SkillResult(
            embodiment=self.name,
            skill=request.name,
            ok=False,
            detail=(
                f"Crazyflie did not reach {tuple(float(v) for v in target_vec)} "
                f"within {max_duration_s:.2f} s."
            ),
            data={
                "target_position_m": target_vec.tolist(),
                "final_position_m": self._state_vector()[0:3].tolist(),
                "position_error_m": final_error,
                "steps": max_steps,
            },
        )

    def _step(self, action: Any) -> None:
        env = self._require_env()
        self._obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Crazyflie simulation episode ended unexpectedly")

    def _state_vector(self) -> Any:
        if self._obs is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._obs[0]

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._env

    @staticmethod
    def _coerce_position(value: Any) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError("position must contain exactly x, y, z")
        return tuple(float(v) for v in value)  # type: ignore[return-value]

    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - only possible in minimal installs
            raise RuntimeError(
                "CrazyfliePyBullet requires the 'crazyflie-sim' optional dependency"
            ) from exc
        return np

    @classmethod
    def _load_default_env_factory(cls) -> Callable[..., Any]:
        try:
            import numpy as np
            from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
            from gym_pybullet_drones.utils.enums import DroneModel, Physics
        except ImportError as exc:
            raise RuntimeError(
                "CrazyfliePyBullet requires gym-pybullet-drones. "
                "Install embodied-agent[crazyflie-sim]."
            ) from exc

        def factory(
            *,
            gui: bool,
            ctrl_freq_hz: int,
            initial_position: tuple[float, float, float],
        ) -> Any:
            return VelocityAviary(
                drone_model=DroneModel.CF2X,
                num_drones=1,
                initial_xyzs=np.asarray([initial_position], dtype=float),
                physics=Physics.PYB,
                pyb_freq=240,
                ctrl_freq=ctrl_freq_hz,
                gui=gui,
                record=False,
                obstacles=False,
                user_debug_gui=gui,
            )

        return factory
