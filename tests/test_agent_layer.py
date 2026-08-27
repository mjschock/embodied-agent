from __future__ import annotations

import asyncio
import unittest

from embodied_agent.agent import (
    CapabilityPlanner,
    PlanExecutor,
    RobotToolRouter,
    Task,
    TaskStep,
    ToolPermissionError,
    ToolValidationError,
    build_stack,
)
from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    RobotRegistry,
    SkillRequest,
    SkillResult,
)


class FakeRobot(Embodiment):
    def __init__(
        self,
        name: str,
        capabilities: set[Capability],
        *,
        fail_skill: str | None = None,
        marker: str = "",
    ) -> None:
        self.name = name
        self.backend = "fake"
        self._capabilities = frozenset(capabilities)
        self.fail_skill = fail_skill
        self.marker = marker
        self.calls: list[SkillRequest] = []
        self.connected = False

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def observe(self) -> Observation:
        return Observation(self.name, {"marker": self.marker})

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        self.calls.append(request)
        ok = request.name != self.fail_skill
        return SkillResult(
            embodiment=self.name,
            skill=request.name,
            ok=ok,
            detail="ok" if ok else "forced failure",
            data={"params": dict(request.params)},
        )


class AgentToolTests(unittest.TestCase):
    def _router(self) -> tuple[RobotToolRouter, FakeRobot, FakeRobot, FakeRobot]:
        registry = RobotRegistry()
        xlerobot = FakeRobot("xlerobot", {Capability.OBSERVE, Capability.NAVIGATE})
        crazyflie = FakeRobot("crazyflie", {Capability.OBSERVE, Capability.FLY})
        humanoid = FakeRobot("humanoid", {Capability.OBSERVE, Capability.STAND})
        for robot in (xlerobot, crazyflie, humanoid):
            registry.register(robot)
        router = RobotToolRouter(
            registry,
            {
                "xlerobot": ["observe", "reset", "drive_velocity", "navigate_to"],
                "crazyflie": ["observe", "takeoff", "goto", "land"],
                "humanoid": ["observe", "stand", "walk_velocity"],
            },
        )
        return router, xlerobot, crazyflie, humanoid

    def test_capability_filters_allowlisted_tools(self) -> None:
        router, _, _, _ = self._router()
        names = [tool["name"] for tool in router.list_tools()]
        self.assertIn("xlerobot.navigate_to", names)
        self.assertIn("crazyflie.goto", names)
        self.assertIn("humanoid.stand", names)
        self.assertNotIn("humanoid.walk_velocity", names)

    def test_validation_blocks_unknown_and_out_of_bounds_arguments(self) -> None:
        router, _, _, _ = self._router()

        async def scenario() -> None:
            with self.assertRaises(ToolValidationError):
                await router.call(
                    "xlerobot.drive_velocity",
                    {"lin_x_mps": 99.0, "duration_s": 1.0},
                )
            with self.assertRaises(ToolValidationError):
                await router.call("crazyflie.takeoff", {"altitude_m": 1.0, "rpm": 9000})
            with self.assertRaises(ToolPermissionError):
                await router.call("xlerobot.raw_joint_command", {})

        asyncio.run(scenario())

    def test_crazyflie_goto_translates_agent_coordinates_to_robot_position(self) -> None:
        router, _, crazyflie, _ = self._router()

        async def scenario() -> None:
            result = await router.call(
                "crazyflie.goto",
                {"x_m": 1.0, "y_m": -2.0, "z_m": 1.25, "timeout_s": 4.0},
            )
            self.assertTrue(result.ok)
            request = crazyflie.calls[-1]
            self.assertEqual(request.params["position"], (1.0, -2.0, 1.25))
            self.assertEqual(request.params["timeout_s"], 4.0)

        asyncio.run(scenario())

    def test_capability_planner_binds_multi_robot_task(self) -> None:
        router, _, _, _ = self._router()
        planner = CapabilityPlanner(router)
        task = Task(
            "scout-and-approach",
            (
                TaskStep("takeoff", Capability.FLY, {"altitude_m": 1.0}),
                TaskStep("goto", Capability.FLY, {"x_m": 2.0, "y_m": 1.0, "z_m": 1.0}),
                TaskStep("navigate_to", Capability.NAVIGATE, {"x_m": 2.0, "y_m": 1.0}),
                TaskStep("land", Capability.FLY, {}),
            ),
        )
        plan = planner.plan(task)
        self.assertEqual(
            [step.tool for step in plan.steps],
            [
                "crazyflie.takeoff",
                "crazyflie.goto",
                "xlerobot.navigate_to",
                "crazyflie.land",
            ],
        )

    def test_executor_stops_after_failed_tool(self) -> None:
        registry = RobotRegistry()
        drone = FakeRobot(
            "crazyflie",
            {Capability.OBSERVE, Capability.FLY},
            fail_skill="goto",
        )
        registry.register(drone)
        router = RobotToolRouter(registry, {"crazyflie": ["takeoff", "goto", "land"]})
        planner = CapabilityPlanner(router)
        plan = planner.plan(
            Task(
                "failure-test",
                (
                    TaskStep("takeoff", Capability.FLY, {}),
                    TaskStep("goto", Capability.FLY, {"x_m": 1.0, "y_m": 0.0, "z_m": 1.0}),
                    TaskStep("land", Capability.FLY, {}),
                ),
            )
        )

        async def scenario() -> None:
            result = await PlanExecutor(router).execute(plan)
            self.assertFalse(result.ok)
            self.assertEqual([call.tool for call in result.calls], ["crazyflie.takeoff", "crazyflie.goto"])
            self.assertEqual([call.skill for call in drone.calls], ["takeoff", "goto"])

        asyncio.run(scenario())

    def test_config_driven_stack_with_injected_factories(self) -> None:
        def nav_factory(name: str, marker: str = "") -> Embodiment:
            return FakeRobot(name, {Capability.OBSERVE, Capability.NAVIGATE}, marker=marker)

        def fly_factory(name: str, marker: str = "") -> Embodiment:
            return FakeRobot(name, {Capability.OBSERVE, Capability.FLY}, marker=marker)

        config = {
            "robots": {
                "ground": {
                    "adapter": "fake_nav",
                    "params": {"marker": "ground-config"},
                    "tools": ["observe", "navigate_to"],
                },
                "drone": {
                    "adapter": "fake_fly",
                    "params": {"marker": "drone-config"},
                    "tools": ["observe", "takeoff", "goto", "land"],
                },
                "disabled": {
                    "adapter": "fake_nav",
                    "enabled": False,
                    "tools": ["observe"],
                },
            }
        }
        registry, router = build_stack(
            config,
            factories={"fake_nav": nav_factory, "fake_fly": fly_factory},
        )
        self.assertEqual(len(registry), 2)
        self.assertEqual(registry.get("ground").marker, "ground-config")
        self.assertEqual(
            [tool["name"] for tool in router.tools_for_skill("takeoff")],
            ["drone.takeoff"],
        )


if __name__ == "__main__":
    unittest.main()
