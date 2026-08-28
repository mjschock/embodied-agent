from __future__ import annotations

import math
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
        default_timeout_s: float = 10.0,
        max_timeout_s: float = 30.0,
        timeout_safety_factor: float = 1.5,
        timeout_settle_s: float = 1.0,
        env_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.backend = "gym-pybullet-drones"
        self.gui = gui
        self.seed = self._coerce_seed(seed)
        self.ctrl_freq_hz = ctrl_freq_hz
        self.initial_position = initial_position
        self.position_tolerance_m = position_tolerance_m
        self.slow_radius_m = slow_radius_m
        self.default_timeout_s = float(default_timeout_s)
        self.max_timeout_s = float(max_timeout_s)
        self.timeout_safety_factor = float(timeout_safety_factor)
        self.timeout_settle_s = float(timeout_settle_s)
        if self.default_timeout_s <= 0.0:
            raise ValueError("default_timeout_s must be positive")
        if self.max_timeout_s < self.default_timeout_s:
            raise ValueError("max_timeout_s must be >= default_timeout_s")
        if self.timeout_safety_factor < 1.0:
            raise ValueError("timeout_safety_factor must be >= 1.0")
        if self.timeout_settle_s < 0.0:
            raise ValueError("timeout_settle_s must be non-negative")
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
        if request.name == "reset":
            supplied_seed = request.params.get("seed")
            effective_seed = (
                self.seed if supplied_seed is None else self._coerce_seed(supplied_seed)
            )
            self.seed = effective_seed
            self._obs, _ = self._require_env().reset(seed=effective_seed)
            state = self._state_vector()
            return SkillResult(
                embodiment=self.name,
                skill=request.name,
                ok=True,
                detail=f"Crazyflie PyBullet simulation reset with seed {effective_seed}.",
                data={
                    "seed": effective_seed,
                    "position_m": state[0:3].tolist(),
                },
            )

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
        target_vec = np.asarray(target, dtype=float)
        max_duration_s, timeout_source = self._resolve_timeout_s(request, target_vec)

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
                        "timeout_s": max_duration_s,
                        "timeout_source": timeout_source,
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
                "timeout_s": max_duration_s,
                "timeout_source": timeout_source,
            },
        )

    def _resolve_timeout_s(self, request: SkillRequest, target_vec: Any) -> tuple[float, str]:
        explicit = request.params.get("timeout_s")
        if explicit is not None:
            timeout_s = float(explicit)
            if timeout_s <= 0.0 or timeout_s > self.max_timeout_s:
                raise ValueError(
                    f"timeout_s must be > 0 and <= {self.max_timeout_s:.2f}"
                )
            return timeout_s, "explicit"

        np = self._numpy()
        current = self._state_vector()[0:3]
        distance_m = float(np.linalg.norm(target_vec - current))
        env = self._require_env()
        speed_limit_mps = float(getattr(env, "SPEED_LIMIT", 0.25))
        if not math.isfinite(speed_limit_mps) or speed_limit_mps <= 0.0:
            speed_limit_mps = 0.25
        ideal_travel_s = distance_m / speed_limit_mps
        timeout_s = max(
            self.default_timeout_s,
            self.timeout_safety_factor * ideal_travel_s + self.timeout_settle_s,
        )
        return min(timeout_s, self.max_timeout_s), "distance-aware"

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
    def _coerce_seed(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        if value < 0 or value > 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")
        return value

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
