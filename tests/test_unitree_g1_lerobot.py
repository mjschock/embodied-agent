from __future__ import annotations

import asyncio
import unittest
from typing import Any

from embodied_agent.agent.config import DEFAULT_ADAPTER_FACTORIES, build_registry
from embodied_agent.agent.tools import RobotToolRouter
from embodied_agent.core import Capability, RobotRegistry
from embodied_agent.embodiments import UnitreeG1LeRobot


class FakeNativeG1:
    def __init__(self) -> None:
        self.connect_count = 0
        self.disconnect_count = 0
        self.reset_count = 0
        self.actions: list[dict[str, Any]] = []
        self.camera_frame = [["pixel"]]

    def connect(self) -> None:
        self.connect_count += 1

    def disconnect(self) -> None:
        self.disconnect_count += 1

    def reset(self) -> None:
        self.reset_count += 1

    def get_observation(self) -> dict[str, Any]:
        observation: dict[str, Any] = {}
        for index in range(29):
            name = f"joint_{index}"
            observation[f"{name}.q"] = float(index) / 10.0
            observation[f"{name}.dq"] = float(index) / 100.0
            observation[f"{name}.tau"] = float(index) / 1000.0
        observation.update(
            {
                "imu.gyro.x": 0.1,
                "imu.gyro.y": 0.2,
                "imu.gyro.z": 0.3,
                "imu.rpy.roll": 0.01,
                "wireless_remote": b"\x01\x02",
                "head_camera": self.camera_frame,
            }
        )
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.actions.append(dict(action))
        return dict(action)


class FakeDDSNative:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._ChannelFactoryInitialize = self._initialize

    def _initialize(self, *args: Any) -> None:
        self.calls.append(tuple(args))


class UnitreeG1LeRobotTests(unittest.TestCase):
    @staticmethod
    def _factory(native: FakeNativeG1):
        def factory(**_: Any) -> FakeNativeG1:
            return native

        return factory

    def test_observe_normalizes_joint_imu_remote_and_camera_features(self) -> None:
        async def scenario() -> None:
            native = FakeNativeG1()
            robot = UnitreeG1LeRobot(robot_factory=self._factory(native))
            self.assertEqual(robot.capabilities, frozenset({Capability.OBSERVE}))
            await robot.connect()
            try:
                observation = await robot.observe()
                self.assertEqual(observation.state["backend"], "lerobot-unitree-g1")
                self.assertTrue(observation.state["is_simulation"])
                self.assertIsNone(observation.state["controller"])
                self.assertEqual(observation.state["simulation_dds_interface"], "auto")
                self.assertEqual(len(observation.state["joint_position_rad"]), 29)
                self.assertEqual(len(observation.state["joint_velocity_rad_s"]), 29)
                self.assertEqual(len(observation.state["joint_torque_est_nm"]), 29)
                self.assertEqual(observation.state["joint_position_rad"]["joint_7"], 0.7)
                self.assertEqual(observation.state["imu"]["gyro.x"], 0.1)
                self.assertEqual(observation.state["wireless_remote"], [1, 2])
                self.assertIs(observation.images["head_camera"], native.camera_frame)
            finally:
                await robot.disconnect()
            self.assertEqual(native.connect_count, 1)
            self.assertEqual(native.disconnect_count, 1)

        asyncio.run(scenario())

    def test_reset_delegates_to_native_reset_without_joint_action(self) -> None:
        async def scenario() -> None:
            native = FakeNativeG1()
            robot = UnitreeG1LeRobot(robot_factory=self._factory(native))
            await robot.connect()
            try:
                result = await robot.execute("reset")
                self.assertTrue(result.ok)
                self.assertEqual(result.data["simulation_dds_interface"], "auto")
                self.assertEqual(native.reset_count, 1)
                self.assertEqual(native.actions, [])
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_stand_requires_controller_and_only_sends_zero_remote_axes(self) -> None:
        async def scenario() -> None:
            disabled_native = FakeNativeG1()
            disabled = UnitreeG1LeRobot(robot_factory=self._factory(disabled_native))
            self.assertFalse(disabled.supports(Capability.STAND))
            await disabled.connect()
            try:
                with self.assertRaisesRegex(RuntimeError, "requires a configured locomotion controller"):
                    await disabled.execute("stand")
                self.assertEqual(disabled_native.actions, [])
            finally:
                await disabled.disconnect()

            native = FakeNativeG1()
            robot = UnitreeG1LeRobot(
                controller="GrootLocomotionController",
                robot_factory=self._factory(native),
            )
            self.assertTrue(robot.supports(Capability.STAND))
            await robot.connect()
            try:
                result = await robot.execute("stand")
                self.assertTrue(result.ok)
                self.assertEqual(
                    native.actions,
                    [{
                        "remote.lx": 0.0,
                        "remote.ly": 0.0,
                        "remote.rx": 0.0,
                        "remote.ry": 0.0,
                    }],
                )
                self.assertTrue(all(not key.endswith(".q") for key in native.actions[0]))
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_simulation_dds_compatibility_overrides_upstream_hardcoded_interface(self) -> None:
        auto = FakeDDSNative()
        UnitreeG1LeRobot._configure_simulation_dds(auto, None)
        auto._ChannelFactoryInitialize(0, "lo")
        self.assertEqual(auto.calls, [(0,)])

        explicit = FakeDDSNative()
        UnitreeG1LeRobot._configure_simulation_dds(explicit, "eth0")
        explicit._ChannelFactoryInitialize(0, "lo")
        self.assertEqual(explicit.calls, [(0, "eth0")])

    def test_router_hides_stand_without_controller_and_exposes_it_with_controller(self) -> None:
        native = FakeNativeG1()
        observation_only = UnitreeG1LeRobot(name="g1", robot_factory=self._factory(native))
        registry = RobotRegistry()
        registry.register(observation_only)
        router = RobotToolRouter(registry, {"g1": ["observe", "reset", "stand", "walk_velocity"]})
        self.assertEqual(
            [tool["name"] for tool in router.list_tools()],
            ["g1.observe", "g1.reset"],
        )

        capable = UnitreeG1LeRobot(
            name="g1",
            controller="GrootLocomotionController",
            robot_factory=self._factory(native),
        )
        registry = RobotRegistry()
        registry.register(capable)
        router = RobotToolRouter(registry, {"g1": ["observe", "reset", "stand", "walk_velocity"]})
        self.assertEqual(
            [tool["name"] for tool in router.list_tools()],
            ["g1.observe", "g1.reset", "g1.stand"],
        )
        self.assertNotIn("g1.walk_velocity", [tool["name"] for tool in router.list_tools()])

    def test_config_factory_registers_simulation_without_importing_lerobot(self) -> None:
        self.assertIs(DEFAULT_ADAPTER_FACTORIES["lerobot_unitree_g1"], UnitreeG1LeRobot)
        registry = build_registry(
            {
                "robots": {
                    "g1": {
                        "adapter": "lerobot_unitree_g1",
                        "params": {
                            "is_simulation": True,
                            "controller": "GrootLocomotionController",
                            "simulation_dds_interface": None,
                        },
                    }
                }
            }
        )
        robot = registry.get("g1")
        self.assertIsInstance(robot, UnitreeG1LeRobot)
        self.assertTrue(robot.is_simulation)
        self.assertEqual(robot.controller, "GrootLocomotionController")
        self.assertIsNone(robot.simulation_dds_interface)
        self.assertEqual(robot.capabilities, frozenset({Capability.OBSERVE, Capability.STAND}))

    def test_default_positions_and_dds_interface_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 29"):
            UnitreeG1LeRobot(default_positions=[0.0] * 28)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            UnitreeG1LeRobot(simulation_dds_interface="   ")


if __name__ == "__main__":
    unittest.main()
