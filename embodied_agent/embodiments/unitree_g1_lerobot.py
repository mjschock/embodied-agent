from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Mapping
from typing import Any

from embodied_agent.core import Capability, Embodiment, Observation, SkillRequest, SkillResult


NativeRobotFactory = Callable[..., Any]
_SIMULATION_ENV_FACTORY_LOCK = threading.Lock()


class UnitreeG1LeRobot(Embodiment):
    """Safe semantic boundary around LeRobot's Unitree G1 robot.

    The adapter deliberately does not expose direct 29-DoF joint targets. Observation
    and reset delegate to LeRobot's native robot contract. ``STAND`` is advertised
    only when a lower-body locomotion controller is configured; the implementation
    sends zero normalized remote axes so the controller selects its balance/standing
    behavior.

    LeRobot v0.6.1 hardcodes the ``lo`` network interface when it initializes the
    Unitree SDK2 DDS transport in simulation. CycloneDDS 0.10.2 on Python 3.12 can
    initialize the same domain successfully with interface auto-detection, while the
    explicit Unitree ``lo`` XML path aborts natively. ``simulation_dds_interface``
    therefore defaults to ``None`` (auto-detect) for simulated G1 instances only.
    Physical G1 initialization is left untouched.

    The pinned LeRobot G1 path also loads EnvHub through the string-based Hub API,
    which does not forward the EnvHub ``publish_images`` option. When
    ``simulation_publish_images`` is explicitly set, the adapter temporarily wraps
    LeRobot's Hub-module call during native ``connect()`` so the pinned G1 EnvHub
    receives that option. ``None`` preserves upstream behavior exactly.
    """

    REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")

    def __init__(
        self,
        name: str = "unitree_g1",
        *,
        is_simulation: bool = True,
        controller: str | None = None,
        robot_ip: str = "192.168.123.164",
        gravity_compensation: bool = False,
        default_positions: tuple[float, ...] | list[float] | None = None,
        simulation_dds_interface: str | None = None,
        simulation_publish_images: bool | None = None,
        robot_factory: NativeRobotFactory | None = None,
    ) -> None:
        self.name = name
        self.backend = "lerobot-unitree-g1"
        self.is_simulation = bool(is_simulation)
        self.controller = controller
        self.robot_ip = robot_ip
        self.gravity_compensation = bool(gravity_compensation)
        self.default_positions = (
            None if default_positions is None else tuple(float(v) for v in default_positions)
        )
        if self.default_positions is not None and len(self.default_positions) != 29:
            raise ValueError("Unitree G1 default_positions must contain exactly 29 values")
        if simulation_dds_interface is not None:
            simulation_dds_interface = simulation_dds_interface.strip()
            if not simulation_dds_interface:
                raise ValueError("simulation_dds_interface must be a non-empty string or null")
        if (
            simulation_publish_images is not None
            and not isinstance(simulation_publish_images, bool)
        ):
            raise ValueError("simulation_publish_images must be true, false, or null")
        if not self.is_simulation and simulation_publish_images is not None:
            raise ValueError(
                "simulation_publish_images is only valid for simulated Unitree G1 instances"
            )
        self.simulation_dds_interface = simulation_dds_interface
        self.simulation_publish_images = simulation_publish_images
        self._robot_factory = robot_factory
        self._robot: Any | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        capabilities = {Capability.OBSERVE}
        if self.controller is not None:
            capabilities.add(Capability.STAND)
        return frozenset(capabilities)

    async def connect(self) -> None:
        if self._robot is not None:
            return
        if self._robot_factory is not None and self.simulation_publish_images is not None:
            raise RuntimeError(
                "simulation_publish_images requires the default LeRobot Unitree G1 factory"
            )
        use_default_factory = self._robot_factory is None
        factory = self._robot_factory or self._load_default_robot_factory()
        robot = factory(
            is_simulation=self.is_simulation,
            controller=self.controller,
            robot_ip=self.robot_ip,
            gravity_compensation=self.gravity_compensation,
            default_positions=self.default_positions,
            simulation_dds_interface=self.simulation_dds_interface,
        )
        if (
            use_default_factory
            and self.is_simulation
            and self.simulation_publish_images is not None
        ):
            self._connect_with_simulation_image_option(
                robot,
                publish_images=self.simulation_publish_images,
            )
        else:
            robot.connect()
        self._robot = robot

    async def disconnect(self) -> None:
        robot = self._robot
        if robot is None:
            return
        try:
            robot.disconnect()
        finally:
            if self.is_simulation:
                self._close_simulation_dds_endpoints(robot)
            self._robot = None

    async def observe(self) -> Observation:
        native = dict(self._require_robot().get_observation())
        joint_position: dict[str, float] = {}
        joint_velocity: dict[str, float] = {}
        joint_torque: dict[str, float] = {}
        imu: dict[str, float] = {}
        images: dict[str, Any] = {}
        extra_state: dict[str, Any] = {}

        for key, value in native.items():
            if key.endswith(".q"):
                joint_position[key[:-2]] = float(value)
            elif key.endswith(".dq"):
                joint_velocity[key[:-3]] = float(value)
            elif key.endswith(".tau"):
                joint_torque[key[:-4]] = float(value)
            elif key.startswith("imu."):
                imu[key[4:]] = float(value)
            elif key == "wireless_remote":
                extra_state[key] = list(value) if isinstance(value, (bytes, bytearray)) else value
            else:
                images[key] = value

        if joint_position and len(joint_position) != 29:
            raise RuntimeError(
                f"Unitree G1 observation expected 29 joint positions, got {len(joint_position)}"
            )

        state: dict[str, Any] = {
            "backend": self.backend,
            "is_simulation": self.is_simulation,
            "controller": self.controller,
            "simulation_dds_interface": self._simulation_dds_label(),
            "simulation_publish_images": (
                self.simulation_publish_images if self.is_simulation else None
            ),
            "joint_position_rad": joint_position,
            "joint_velocity_rad_s": joint_velocity,
            "joint_torque_est_nm": joint_torque,
            "imu": imu,
        }
        state.update(extra_state)
        return Observation(embodiment=self.name, state=state, images=images)

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        robot = self._require_robot()
        if request.name == "reset":
            if request.params:
                raise ValueError("Unitree G1 reset does not accept agent parameters")
            robot.reset()
            return SkillResult(
                embodiment=self.name,
                skill="reset",
                ok=True,
                detail="Reset Unitree G1 through the native LeRobot reset boundary.",
                data={
                    "controller": self.controller,
                    "is_simulation": self.is_simulation,
                    "simulation_dds_interface": self._simulation_dds_label(),
                    "simulation_publish_images": (
                        self.simulation_publish_images if self.is_simulation else None
                    ),
                },
            )

        if request.name == "stand":
            if request.params:
                raise ValueError("Unitree G1 stand does not accept agent parameters")
            if self.controller is None:
                raise RuntimeError("Unitree G1 stand requires a configured locomotion controller")
            action = dict.fromkeys(self.REMOTE_AXES, 0.0)
            robot.send_action(action)
            return SkillResult(
                embodiment=self.name,
                skill="stand",
                ok=True,
                detail="Requested Unitree G1 balance/standing behavior with zero controller input.",
                data={"controller": self.controller, "remote_axes": action},
            )

        raise ValueError(f"Unsupported Unitree G1 skill: {request.name}")

    def _require_robot(self) -> Any:
        if self._robot is None:
            raise RuntimeError(f"{self.name} is not connected")
        return self._robot

    def _simulation_dds_label(self) -> str | None:
        if not self.is_simulation:
            return None
        return self.simulation_dds_interface or "auto"

    @staticmethod
    def _configure_simulation_dds(robot: Any, interface: str | None) -> None:
        """Override LeRobot's hardcoded simulated ``lo`` argument explicitly.

        UnitreeG1.connect() calls ``self._ChannelFactoryInitialize(0, 'lo')`` in
        LeRobot v0.6.1. Replacing only that instance attribute keeps physical mode
        untouched and lets the simulation choose the known-working one-argument
        Unitree initializer (or an explicitly requested interface).
        """

        original = robot._ChannelFactoryInitialize

        def initialize(domain_id: int, _upstream_interface: str | None = None) -> Any:
            if interface is None:
                return original(domain_id)
            return original(domain_id, interface)

        robot._ChannelFactoryInitialize = initialize

    @staticmethod
    def _connect_with_simulation_image_option(robot: Any, *, publish_images: bool) -> None:
        """Forward EnvHub's image-publishing option through pinned LeRobot v0.6.1.

        LeRobot's string-based Hub environment API calls a remote module's
        ``make_env`` without forwarding arbitrary keyword arguments. The pinned G1
        EnvHub module exposes ``publish_images`` and defaults it to true. Temporarily
        replacing LeRobot's internal Hub-module call lets this adapter request a
        control-only headless environment without editing the downloaded Hub source.

        The override is serialized and restored before ``connect`` returns. It fails
        loudly if no Hub call was intercepted or if the remote contract no longer
        exposes ``publish_images``.
        """

        try:
            import lerobot.envs.factory as env_factory
        except ImportError as exc:  # pragma: no cover - exercised on real setup
            raise RuntimeError(
                "simulation_publish_images requires LeRobot's environment factory"
            ) from exc

        with _SIMULATION_ENV_FACTORY_LOCK:
            original_call_make_env = env_factory._call_make_env
            intercepted_calls = 0

            def call_make_env(module: Any, n_envs: int, use_async_envs: bool, cfg: Any) -> Any:
                nonlocal intercepted_calls
                if cfg is not None:
                    return original_call_make_env(module, n_envs, use_async_envs, cfg)

                entry_fn = getattr(module, "make_env", None)
                if not callable(entry_fn):
                    return original_call_make_env(module, n_envs, use_async_envs, cfg)
                parameters = inspect.signature(entry_fn).parameters
                if "publish_images" not in parameters:
                    raise RuntimeError(
                        "Pinned Unitree G1 EnvHub make_env no longer exposes publish_images"
                    )
                intercepted_calls += 1
                return entry_fn(
                    n_envs=n_envs,
                    use_async_envs=use_async_envs,
                    publish_images=publish_images,
                )

            env_factory._call_make_env = call_make_env
            try:
                try:
                    robot.connect()
                except BaseException:
                    UnitreeG1LeRobot._cleanup_failed_simulation_connect(robot)
                    raise
            finally:
                env_factory._call_make_env = original_call_make_env

            if intercepted_calls != 1:
                UnitreeG1LeRobot._cleanup_failed_simulation_connect(robot)
                raise RuntimeError(
                    "Expected exactly one Unitree G1 EnvHub make_env call while applying "
                    f"simulation_publish_images, intercepted {intercepted_calls}"
                )

    @staticmethod
    def _cleanup_failed_simulation_connect(robot: Any) -> None:
        try:
            robot.disconnect()
        finally:
            UnitreeG1LeRobot._close_simulation_dds_endpoints(robot)

    @staticmethod
    def _close_simulation_dds_endpoints(robot: Any) -> None:
        """Close SDK2 endpoints LeRobot v0.6.1 leaves open in simulation.

        UnitreeG1.disconnect() stops LeRobot's subscribe/controller threads and closes
        EnvHub, but it does not close its SDK2 low-state reader or low-command writer.
        The pinned SDK2 objects expose ``Close()`` for those endpoints. Releasing them
        after LeRobot has joined its own threads prevents stale DDS readers/writers
        from leaking into a later same-process simulated G1 lifecycle.

        This compatibility cleanup is simulation-only. Physical G1 teardown remains
        exactly LeRobot-owned until it is validated on hardware.
        """

        for attribute in ("lowstate_subscriber", "lowcmd_publisher"):
            endpoint = getattr(robot, attribute, None)
            close = getattr(endpoint, "Close", None)
            if callable(close):
                close()

    @staticmethod
    def _load_default_robot_factory() -> NativeRobotFactory:
        try:
            from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config
        except ImportError as exc:  # pragma: no cover - exercised on real setup
            raise RuntimeError(
                "UnitreeG1LeRobot requires LeRobot's Unitree G1 dependencies and unitree-sdk2py. "
                "See docs/unitree_g1.md for the pinned setup requirements."
            ) from exc

        def factory(
            *,
            is_simulation: bool,
            controller: str | None,
            robot_ip: str,
            gravity_compensation: bool,
            default_positions: tuple[float, ...] | None,
            simulation_dds_interface: str | None,
        ) -> Any:
            kwargs: dict[str, Any] = {
                "is_simulation": is_simulation,
                "controller": controller,
                "robot_ip": robot_ip,
                "gravity_compensation": gravity_compensation,
            }
            if default_positions is not None:
                kwargs["default_positions"] = list(default_positions)
            robot = UnitreeG1(UnitreeG1Config(**kwargs))
            if is_simulation:
                UnitreeG1LeRobot._configure_simulation_dds(robot, simulation_dds_interface)
            return robot

        return factory
