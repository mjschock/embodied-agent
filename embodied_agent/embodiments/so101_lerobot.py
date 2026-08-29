from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


class SO101ManipulationExecutor(Protocol):
    """Semantic policy/skill boundary for SO-101 manipulation.

    Implementations own all conversion from a named high-level skill to native
    LeRobot actions/policies. The SO101LeRobot adapter itself never forwards raw
    joint targets from an agent to ``Robot.send_action()``.
    """

    def execute(
        self,
        *,
        robot: Any,
        skill: str,
        target: str | None,
        max_duration_s: float | None,
    ) -> SkillResult | Awaitable[SkillResult]: ...


NativeRobotFactory = Callable[..., Any]


class SO101LeRobot(Embodiment):
    """Native LeRobot SO-101 follower boundary.

    Observation is available whenever the native LeRobot follower is connected.
    ``MANIPULATE`` is deliberately absent unless a named semantic manipulation
    executor is configured. Raw LeRobot ``send_action`` is never exposed through
    this embodiment or the agent tool router.
    """

    ARM_JOINTS = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    GRIPPER = "gripper"

    def __init__(
        self,
        name: str = "so101",
        *,
        port: str | None = None,
        robot_id: str | None = None,
        calibration_dir: str | Path | None = None,
        calibrate: bool = True,
        disable_torque_on_disconnect: bool = True,
        max_relative_target: float | Mapping[str, float] | None = None,
        use_degrees: bool = True,
        robot_factory: NativeRobotFactory | None = None,
        manipulation_executor: SO101ManipulationExecutor | None = None,
        manipulation_skills: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.backend = "lerobot-so101"
        self.port = port
        self.robot_id = robot_id
        self.calibration_dir = None if calibration_dir is None else Path(calibration_dir)
        self.calibrate_on_connect = bool(calibrate)
        self.disable_torque_on_disconnect = bool(disable_torque_on_disconnect)
        self.max_relative_target = max_relative_target
        self.use_degrees = bool(use_degrees)
        self._robot_factory = robot_factory
        self._robot: Any | None = None
        self._manipulation_executor = manipulation_executor
        normalized_skills = tuple(dict.fromkeys(skill.strip() for skill in manipulation_skills if skill.strip()))
        if manipulation_executor is None and normalized_skills:
            raise ValueError("manipulation_skills require manipulation_executor")
        if manipulation_executor is not None and not normalized_skills:
            raise ValueError("manipulation_executor requires at least one named manipulation skill")
        self._manipulation_skills = frozenset(normalized_skills)

    @property
    def capabilities(self) -> frozenset[Capability]:
        capabilities = {Capability.OBSERVE}
        if self._manipulation_executor is not None and self._manipulation_skills:
            capabilities.add(Capability.MANIPULATE)
        return frozenset(capabilities)

    @property
    def manipulation_skills(self) -> frozenset[str]:
        return self._manipulation_skills

    async def connect(self) -> None:
        if self._robot is not None:
            return
        factory = self._robot_factory or self._load_default_robot_factory()
        robot = factory(
            port=self.port,
            robot_id=self.robot_id,
            calibration_dir=self.calibration_dir,
            disable_torque_on_disconnect=self.disable_torque_on_disconnect,
            max_relative_target=self.max_relative_target,
            use_degrees=self.use_degrees,
        )
        robot.connect(calibrate=self.calibrate_on_connect)
        self._robot = robot

    async def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
        self._robot = None

    async def observe(self) -> Observation:
        native = dict(self._require_robot().get_observation())
        positions = {
            key[:-4]: float(value)
            for key, value in native.items()
            if key.endswith(".pos")
        }
        missing = [name for name in (*self.ARM_JOINTS, self.GRIPPER) if name not in positions]
        if missing:
            raise RuntimeError(
                "SO-101 observation is missing expected motor positions: " + ", ".join(missing)
            )

        if self.use_degrees:
            state: dict[str, Any] = {
                "backend": self.backend,
                "joint_position_deg": {name: positions[name] for name in self.ARM_JOINTS},
                "gripper_position_percent": positions[self.GRIPPER],
                "position_mode": "degrees",
            }
        else:
            state = {
                "backend": self.backend,
                "joint_position_normalized": {name: positions[name] for name in self.ARM_JOINTS},
                "gripper_position_percent": positions[self.GRIPPER],
                "position_mode": "normalized_-100_100",
            }

        images = {
            key: value
            for key, value in native.items()
            if not key.endswith(".pos")
        }
        return Observation(embodiment=self.name, state=state, images=images)

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if request.name != "manipulate":
            raise ValueError(f"Unsupported SO-101 skill: {request.name}")
        if self._manipulation_executor is None:
            raise RuntimeError(
                "SO-101 manipulation is disabled until a semantic policy/skill executor is configured"
            )

        skill = request.params.get("skill")
        if not isinstance(skill, str) or not skill.strip():
            raise ValueError("manipulate.skill must be a non-empty string")
        skill = skill.strip()
        if skill not in self._manipulation_skills:
            raise ValueError(
                f"SO-101 manipulation skill is not configured: {skill}. "
                f"Allowed: {', '.join(sorted(self._manipulation_skills))}"
            )

        target = request.params.get("target")
        if target is not None:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("manipulate.target must be a non-empty string when supplied")
            target = target.strip()

        max_duration = request.params.get("max_duration_s")
        if max_duration is not None:
            max_duration = float(max_duration)
            if not (0.1 <= max_duration <= 30.0):
                raise ValueError("manipulate.max_duration_s must be between 0.1 and 30.0")

        outcome = self._manipulation_executor.execute(
            robot=self._require_robot(),
            skill=skill,
            target=target,
            max_duration_s=max_duration,
        )
        if inspect.isawaitable(outcome):
            outcome = await outcome
        if not isinstance(outcome, SkillResult):
            raise TypeError("SO-101 manipulation executor must return SkillResult")
        if outcome.embodiment != self.name or outcome.skill != request.name:
            raise ValueError(
                "SO-101 manipulation executor must return a SkillResult for this embodiment and 'manipulate'"
            )
        return outcome

    def _require_robot(self) -> Any:
        if self._robot is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._robot

    @staticmethod
    def _load_default_robot_factory() -> NativeRobotFactory:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:  # pragma: no cover - exercised on real setup
            raise RuntimeError(
                "SO101LeRobot requires LeRobot with the Feetech extra. "
                "Install embodied-agent[so101]."
            ) from exc

        def factory(
            *,
            port: str | None,
            robot_id: str | None,
            calibration_dir: Path | None,
            disable_torque_on_disconnect: bool,
            max_relative_target: float | Mapping[str, float] | None,
            use_degrees: bool,
        ) -> Any:
            if not port:
                raise RuntimeError("SO-101 hardware connection requires a serial port")
            config = SO101FollowerConfig(
                port=port,
                id=robot_id,
                calibration_dir=calibration_dir,
                disable_torque_on_disconnect=disable_torque_on_disconnect,
                max_relative_target=max_relative_target,
                use_degrees=use_degrees,
            )
            return SO101Follower(config)

        return factory
