from __future__ import annotations

import asyncio
import unittest
from typing import Any

from embodied_agent.agent.config import DEFAULT_ADAPTER_FACTORIES, build_registry
from embodied_agent.agent.tools import RobotToolRouter, ToolValidationError
from embodied_agent.core import Capability, RobotRegistry, SkillResult
from embodied_agent.embodiments import SO101LeRobot


class FakeNativeSO101:
    def __init__(self) -> None:
        self.connect_calibrate: list[bool] = []
        self.disconnect_count = 0
        self.send_action_calls: list[dict[str, Any]] = []
        self.camera_frame = [["pixel"]]

    def connect(self, *, calibrate: bool = True) -> None:
        self.connect_calibrate.append(calibrate)

    def disconnect(self) -> None:
        self.disconnect_count += 1

    def get_observation(self) -> dict[str, Any]:
        return {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": 42.0,
            "wrist_camera": self.camera_frame,
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.send_action_calls.append(dict(action))
        return dict(action)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        robot: Any,
        skill: str,
        target: str | None,
        max_duration_s: float | None,
    ) -> SkillResult:
        self.calls.append(
            {
                "robot": robot,
                "skill": skill,
                "target": target,
                "max_duration_s": max_duration_s,
            }
        )
        return SkillResult(
            embodiment="so101",
            skill="manipulate",
            ok=True,
            detail=f"Executed named SO-101 skill {skill}.",
            data={"skill": skill, "target": target},
        )


class SO101LeRobotTests(unittest.TestCase):
    @staticmethod
    def _factory(native: FakeNativeSO101):
        def factory(**_: Any) -> FakeNativeSO101:
            return native

        return factory

    def test_observation_wraps_native_lerobot_motor_and_camera_features(self) -> None:
        async def scenario() -> None:
            native = FakeNativeSO101()
            robot = SO101LeRobot(
                robot_factory=self._factory(native),
                calibrate=False,
            )
            self.assertEqual(robot.capabilities, frozenset({Capability.OBSERVE}))

            await robot.connect()
            try:
                self.assertEqual(native.connect_calibrate, [False])
                observation = await robot.observe()
                self.assertEqual(observation.state["backend"], "lerobot-so101")
                self.assertEqual(
                    observation.state["joint_position_deg"],
                    {
                        "shoulder_pan": 1.0,
                        "shoulder_lift": 2.0,
                        "elbow_flex": 3.0,
                        "wrist_flex": 4.0,
                        "wrist_roll": 5.0,
                    },
                )
                self.assertEqual(observation.state["gripper_position_percent"], 42.0)
                self.assertEqual(observation.state["position_mode"], "degrees")
                self.assertIs(observation.images["wrist_camera"], native.camera_frame)
            finally:
                await robot.disconnect()

            self.assertEqual(native.disconnect_count, 1)

        asyncio.run(scenario())

    def test_raw_native_action_is_never_an_agent_skill(self) -> None:
        async def scenario() -> None:
            native = FakeNativeSO101()
            robot = SO101LeRobot(robot_factory=self._factory(native))
            await robot.connect()
            try:
                with self.assertRaisesRegex(ValueError, "Unsupported SO-101 skill"):
                    await robot.execute(
                        "send_action",
                        **{"shoulder_pan.pos": 90.0},
                    )
                self.assertEqual(native.send_action_calls, [])
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_manipulation_capability_requires_named_semantic_executor(self) -> None:
        with self.assertRaises(ValueError):
            SO101LeRobot(
                robot_factory=self._factory(FakeNativeSO101()),
                manipulation_skills=("pick",),
            )
        with self.assertRaises(ValueError):
            SO101LeRobot(
                robot_factory=self._factory(FakeNativeSO101()),
                manipulation_executor=FakeExecutor(),
            )

        async def scenario() -> None:
            native = FakeNativeSO101()
            executor = FakeExecutor()
            robot = SO101LeRobot(
                robot_factory=self._factory(native),
                manipulation_executor=executor,
                manipulation_skills=("pick", "place"),
            )
            self.assertTrue(robot.supports(Capability.MANIPULATE))
            await robot.connect()
            try:
                result = await robot.execute(
                    "manipulate",
                    behavior="pick",
                    target="red_block",
                    max_duration_s=4.0,
                )
                self.assertTrue(result.ok)
                self.assertEqual(result.data["skill"], "pick")
                self.assertEqual(len(executor.calls), 1)
                self.assertIs(executor.calls[0]["robot"], native)
                self.assertEqual(executor.calls[0]["target"], "red_block")
                self.assertEqual(native.send_action_calls, [])

                with self.assertRaisesRegex(ValueError, "not configured"):
                    await robot.execute("manipulate", behavior="wave")
                self.assertEqual(len(executor.calls), 1)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_router_hides_manipulate_without_executor_and_validates_schema(self) -> None:
        async def scenario() -> None:
            native = FakeNativeSO101()
            observation_only = SO101LeRobot(
                name="so101",
                robot_factory=self._factory(native),
            )
            registry = RobotRegistry()
            registry.register(observation_only)
            router = RobotToolRouter(registry, {"so101": ["observe", "manipulate"]})
            self.assertEqual(
                [tool["name"] for tool in router.list_tools()],
                ["so101.observe"],
            )

            executor = FakeExecutor()
            capable = SO101LeRobot(
                name="so101",
                robot_factory=self._factory(native),
                manipulation_executor=executor,
                manipulation_skills=("pick",),
            )
            registry = RobotRegistry()
            registry.register(capable)
            router = RobotToolRouter(registry, {"so101": ["observe", "manipulate"]})
            self.assertEqual(
                [tool["name"] for tool in router.list_tools()],
                ["so101.manipulate", "so101.observe"],
            )
            manipulate_schema = next(
                tool["input_schema"]
                for tool in router.list_tools()
                if tool["name"] == "so101.manipulate"
            )
            self.assertEqual(manipulate_schema["required"], ["behavior"])
            self.assertFalse(manipulate_schema["additionalProperties"])

            await capable.connect()
            try:
                call = await router.call(
                    "so101.manipulate",
                    {"behavior": "pick", "target": "cube", "max_duration_s": 3},
                )
                self.assertTrue(call.ok)
                self.assertEqual(call.data["target"], "cube")
                with self.assertRaises(ToolValidationError):
                    await router.call(
                        "so101.manipulate",
                        {"behavior": "pick", "joint_target": 90},
                    )
            finally:
                await capable.disconnect()

        asyncio.run(scenario())

    def test_config_factory_registers_observation_only_so101_without_importing_lerobot(self) -> None:
        self.assertIs(DEFAULT_ADAPTER_FACTORIES["lerobot_so101"], SO101LeRobot)
        registry = build_registry(
            {
                "robots": {
                    "so101": {
                        "adapter": "lerobot_so101",
                        "params": {
                            "port": "/dev/tty.fake",
                            "robot_id": "reference-arm",
                            "max_relative_target": 10.0,
                        },
                    }
                }
            }
        )
        robot = registry.get("so101")
        self.assertIsInstance(robot, SO101LeRobot)
        self.assertEqual(robot.port, "/dev/tty.fake")
        self.assertEqual(robot.robot_id, "reference-arm")
        self.assertEqual(robot.capabilities, frozenset({Capability.OBSERVE}))


if __name__ == "__main__":
    unittest.main()
