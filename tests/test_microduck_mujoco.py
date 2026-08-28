from __future__ import annotations

import asyncio
import unittest

from embodied_agent.agent import RobotToolRouter, ToolValidationError
from embodied_agent.core import Capability, RobotRegistry
from embodied_agent.embodiments import MicroduckMuJoCo


class FakeMicroduckRuntime:
    def __init__(self, *, recover_ok: bool = True) -> None:
        self.started = False
        self.recover_ok = recover_ok
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

    def recover(self) -> dict:
        self.calls.append(("recover", {}))
        return {**self.state, "recovered": self.recover_ok, "recovery_steps": 50}


def build_robot(fake: FakeMicroduckRuntime, *, both_kicks: bool = True) -> MicroduckMuJoCo:
    return MicroduckMuJoCo(
        name="microduck",
        runtime_root="/unused/microduck_rl",
        walking_policy_path="/unused/walk.onnx",
        standing_policy_path="/unused/stand.onnx",
        kick_left_policy_path="/unused/kick-left.onnx",
        kick_right_policy_path="/unused/kick-right.onnx" if both_kicks else None,
        runtime_factory=lambda: fake,
    )


class MicroduckAdapterTests(unittest.TestCase):
    def test_capabilities_reflect_configured_policies(self) -> None:
        full = build_robot(FakeMicroduckRuntime())
        self.assertTrue(full.supports(Capability.OBSERVE, Capability.STAND, Capability.WALK))
        self.assertTrue(full.supports(Capability.KICK, Capability.RECOVER))

        no_right_kick = build_robot(FakeMicroduckRuntime(), both_kicks=False)
        self.assertFalse(no_right_kick.supports(Capability.KICK))
        self.assertTrue(no_right_kick.supports(Capability.RECOVER))

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

            recovery = await robot.execute("recover")
            self.assertTrue(recovery.ok)
            self.assertTrue(recovery.data["recovered"])

            reset = await robot.execute("reset")
            self.assertTrue(reset.ok)

            await robot.disconnect()
            self.assertFalse(fake.started)

        asyncio.run(scenario())

    def test_recovery_timeout_is_reported_as_skill_failure(self) -> None:
        robot = build_robot(FakeMicroduckRuntime(recover_ok=False))

        async def scenario() -> None:
            await robot.connect()
            result = await robot.execute("recover")
            self.assertFalse(result.ok)
            self.assertIn("timed out", result.detail)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_router_exposes_enum_validated_kick_and_recovery(self) -> None:
        fake = FakeMicroduckRuntime()
        robot = build_robot(fake)
        registry = RobotRegistry()
        registry.register(robot)
        router = RobotToolRouter(
            registry,
            {"microduck": ["observe", "stand", "walk_velocity", "kick", "recover"]},
        )

        tools = {item["name"]: item for item in router.list_tools()}
        self.assertIn("microduck.kick", tools)
        self.assertIn("microduck.recover", tools)
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
            recover = await router.call("microduck.recover", {})
            self.assertTrue(recover.ok)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_router_hides_kick_when_policy_pair_is_incomplete(self) -> None:
        robot = build_robot(FakeMicroduckRuntime(), both_kicks=False)
        registry = RobotRegistry()
        registry.register(robot)
        router = RobotToolRouter(registry, {"microduck": ["kick", "recover"]})
        names = [item["name"] for item in router.list_tools()]
        self.assertNotIn("microduck.kick", names)
        self.assertIn("microduck.recover", names)


if __name__ == "__main__":
    unittest.main()
