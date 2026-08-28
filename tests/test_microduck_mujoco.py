from __future__ import annotations

import asyncio
import unittest

from embodied_agent.agent import RobotToolRouter, ToolValidationError
from embodied_agent.core import Capability, RobotRegistry
from embodied_agent.embodiments import MicroduckMuJoCo


class FakeMicroduckRuntime:
    def __init__(self) -> None:
        self.started = False
        self.calls: list[tuple[str, dict]] = []
        self.state = {
            "position_m": (0.0, 0.0, 0.12),
            "projected_gravity": (0.0, 0.0, -1.0),
            "joint_position_rad": tuple(0.0 for _ in range(14)),
            "joint_velocity_rps": tuple(0.0 for _ in range(14)),
            "policy": "standing",
            "behavior": None,
            "control_hz": 50.0,
        }

    def start(self) -> None:
        self.started = True
        self.calls.append(("start", {}))

    def stop(self) -> None:
        self.started = False
        self.calls.append(("stop", {}))

    def observe(self) -> dict:
        self.calls.append(("observe", {}))
        return dict(self.state)

    def reset(self) -> dict:
        self.calls.append(("reset", {}))
        return dict(self.state)

    def stand(self) -> dict:
        self.calls.append(("stand", {}))
        return {**self.state, "upright": True}

    def walk_velocity(self, **params) -> dict:
        self.calls.append(("walk_velocity", dict(params)))
        return {**self.state, "policy": "standing"}

    def kick(self, **params) -> dict:
        self.calls.append(("kick", dict(params)))
        return {**self.state, "foot": params["foot"]}

    def roll(self) -> dict:
        self.calls.append(("roll", {}))
        return {**self.state, "upright": True}


def build_robot(
    fake: FakeMicroduckRuntime,
    *,
    both_kicks: bool = True,
    with_roll: bool = True,
) -> MicroduckMuJoCo:
    return MicroduckMuJoCo(
        name="microduck",
        runtime_root="/unused/microduck_rl",
        walking_policy_path="/unused/walk.onnx",
        standing_policy_path="/unused/stand.onnx",
        kick_left_policy_path="/unused/kick-left.onnx",
        kick_right_policy_path="/unused/kick-right.onnx" if both_kicks else None,
        roll_policy_path="/unused/roulade.onnx" if with_roll else None,
        runtime_factory=lambda: fake,
    )


class MicroduckAdapterTests(unittest.TestCase):
    def test_capabilities_reflect_configured_policies(self) -> None:
        full = build_robot(FakeMicroduckRuntime())
        self.assertTrue(full.supports(Capability.OBSERVE, Capability.STAND, Capability.WALK))
        self.assertTrue(full.supports(Capability.KICK, Capability.ROLL))

        partial = build_robot(FakeMicroduckRuntime(), both_kicks=False, with_roll=False)
        self.assertFalse(partial.supports(Capability.KICK))
        self.assertFalse(partial.supports(Capability.ROLL))

    def test_lifecycle_and_semantic_skills(self) -> None:
        fake = FakeMicroduckRuntime()
        robot = build_robot(fake)

        async def scenario() -> None:
            await robot.connect()
            self.assertTrue(fake.started)

            observation = await robot.observe()
            self.assertEqual(len(observation.state["joint_position_rad"]), 14)
            self.assertEqual(observation.state["control_hz"], 50.0)

            stand = await robot.execute("stand")
            self.assertTrue(stand.ok)

            walk = await robot.execute(
                "walk_velocity",
                lin_x_mps=0.15,
                lin_y_mps=0.0,
                yaw_rate_rps=0.2,
                duration_s=1.25,
            )
            self.assertTrue(walk.ok)
            self.assertEqual(
                fake.calls[-1],
                (
                    "walk_velocity",
                    {
                        "lin_x_mps": 0.15,
                        "lin_y_mps": 0.0,
                        "yaw_rate_rps": 0.2,
                        "duration_s": 1.25,
                    },
                ),
            )

            kick = await robot.execute("kick", foot="left")
            self.assertTrue(kick.ok)
            self.assertEqual(kick.data["foot"], "left")

            roll = await robot.execute("roll")
            self.assertTrue(roll.ok)
            self.assertTrue(roll.data["upright"])

            reset = await robot.execute("reset")
            self.assertTrue(reset.ok)

            await robot.disconnect()
            self.assertFalse(fake.started)

        asyncio.run(scenario())

    def test_router_exposes_enum_validated_kick_and_roll(self) -> None:
        fake = FakeMicroduckRuntime()
        robot = build_robot(fake)
        registry = RobotRegistry()
        registry.register(robot)
        router = RobotToolRouter(
            registry,
            {"microduck": ["observe", "stand", "walk_velocity", "kick", "roll"]},
        )

        tools = {item["name"]: item for item in router.list_tools()}
        self.assertIn("microduck.kick", tools)
        self.assertIn("microduck.roll", tools)
        self.assertEqual(
            tools["microduck.kick"]["input_schema"]["properties"]["foot"]["enum"],
            ["left", "right"],
        )

        async def scenario() -> None:
            await robot.connect()
            result = await router.call("microduck.kick", {"foot": "right"})
            self.assertTrue(result.ok)
            self.assertEqual(fake.calls[-1], ("kick", {"foot": "right"}))
            with self.assertRaises(ToolValidationError):
                await router.call("microduck.kick", {"foot": "middle"})
            roll = await router.call("microduck.roll", {})
            self.assertTrue(roll.ok)
            self.assertEqual(fake.calls[-1], ("roll", {}))
            await robot.disconnect()

        asyncio.run(scenario())

    def test_router_hides_skills_when_policy_assets_are_incomplete(self) -> None:
        robot = build_robot(FakeMicroduckRuntime(), both_kicks=False, with_roll=False)
        registry = RobotRegistry()
        registry.register(robot)
        router = RobotToolRouter(registry, {"microduck": ["kick", "roll"]})
        names = [item["name"] for item in router.list_tools()]
        self.assertNotIn("microduck.kick", names)
        self.assertNotIn("microduck.roll", names)


if __name__ == "__main__":
    unittest.main()
