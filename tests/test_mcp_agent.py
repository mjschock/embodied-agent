from __future__ import annotations

import asyncio
import unittest

from mcp import Client

from embodied_agent.agent import RobotToolRouter
from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    RobotRegistry,
    SkillRequest,
    SkillResult,
)
from embodied_agent.mcp import build_mcp_server
from embodied_agent.mcp.agent import AgentContext, AgentDecision, MCPAgentRunner
from embodied_agent.world import Pose3D, WorldEntity, WorldState


class FakeRobot(Embodiment):
    def __init__(
        self,
        name: str,
        capabilities: set[Capability],
        *,
        fail_skill: str | None = None,
    ) -> None:
        self.name = name
        self.backend = "fake"
        self._capabilities = frozenset(capabilities)
        self.fail_skill = fail_skill
        self.connected = False
        self.calls: list[SkillRequest] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def observe(self) -> Observation:
        return Observation(self.name, {"connected": self.connected})

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


class RecordingModel:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = list(decisions)
        self.contexts: list[AgentContext] = []

    async def decide(self, context: AgentContext) -> AgentDecision:
        self.contexts.append(context)
        if not self.decisions:
            return AgentDecision.finish("done")
        return self.decisions.pop(0)


class RepeatingModel:
    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.contexts: list[AgentContext] = []

    async def decide(self, context: AgentContext) -> AgentDecision:
        self.contexts.append(context)
        return self.decision


def make_server(*, xlerobot_fail_skill: str | None = None):
    registry = RobotRegistry()
    xlerobot = FakeRobot(
        "xlerobot",
        {Capability.OBSERVE, Capability.NAVIGATE},
        fail_skill=xlerobot_fail_skill,
    )
    crazyflie = FakeRobot("crazyflie", {Capability.OBSERVE, Capability.FLY})
    humanoid = FakeRobot("humanoid", {Capability.OBSERVE, Capability.STAND})
    for robot in (xlerobot, crazyflie, humanoid):
        registry.register(robot)

    router = RobotToolRouter(
        registry,
        {
            "xlerobot": ["navigate_to"],
            "crazyflie": ["takeoff", "goto", "land"],
            "humanoid": ["stand"],
        },
    )
    world = WorldState()
    world.upsert_entity(
        WorldEntity(
            "inspection_target",
            "waypoint",
            Pose3D(1.5, 0.5, 1.0),
            source="test",
        )
    )
    return (
        build_mcp_server(router, registry=registry, world=world),
        xlerobot,
        crazyflie,
        humanoid,
        world,
    )


class MCPAgentRunnerTests(unittest.TestCase):
    def test_iterative_agent_coordinates_all_three_robots_and_finishes(self) -> None:
        async def scenario() -> None:
            server, xlerobot, crazyflie, humanoid, _ = make_server()
            model = RecordingModel(
                [
                    AgentDecision.call("crazyflie.takeoff", {"altitude_m": 1.0}),
                    AgentDecision.call(
                        "crazyflie.goto",
                        {"x_m": 1.5, "y_m": 0.5, "z_m": 1.0},
                    ),
                    AgentDecision.call(
                        "xlerobot.navigate_to",
                        {"x_m": 1.5, "y_m": 0.5},
                    ),
                    AgentDecision.call("humanoid.stand"),
                    AgentDecision.call("crazyflie.land"),
                    AgentDecision.finish("inspection handoff complete"),
                ]
            )

            async with Client(server) as client:
                result = await MCPAgentRunner(client, model, max_steps=8).run(
                    "Scout the inspection target, move the ground robot there, "
                    "have the humanoid stand ready, then land the drone."
                )

            self.assertTrue(result.ok)
            self.assertTrue(result.finished)
            self.assertEqual(result.summary, "inspection handoff complete")
            self.assertEqual(
                [action.tool for action in result.actions],
                [
                    "crazyflie.takeoff",
                    "crazyflie.goto",
                    "xlerobot.navigate_to",
                    "humanoid.stand",
                    "crazyflie.land",
                ],
            )
            self.assertEqual(len(crazyflie.calls), 3)
            self.assertEqual(len(xlerobot.calls), 1)
            self.assertEqual(len(humanoid.calls), 1)

        asyncio.run(scenario())

    def test_world_snapshot_is_refreshed_before_every_model_decision(self) -> None:
        async def scenario() -> None:
            server, _, _, _, _ = make_server()
            model = RecordingModel(
                [
                    AgentDecision.call("crazyflie.takeoff", {"altitude_m": 1.25}),
                    AgentDecision.finish("observed result"),
                ]
            )

            async with Client(server) as client:
                result = await MCPAgentRunner(client, model).run("Take off, then verify the result.")

            self.assertTrue(result.ok)
            self.assertEqual(len(model.contexts), 2)
            first_world = model.contexts[0].world_snapshot
            second_world = model.contexts[1].world_snapshot
            self.assertNotIn("crazyflie", first_world["robots"])
            self.assertEqual(
                second_world["robots"]["crazyflie"]["tool"],
                "crazyflie.takeoff",
            )
            self.assertEqual(
                second_world["robots"]["crazyflie"]["data"]["params"]["altitude_m"],
                1.25,
            )

        asyncio.run(scenario())

    def test_model_cannot_call_tool_that_mcp_did_not_expose(self) -> None:
        async def scenario() -> None:
            server, _, crazyflie, _, _ = make_server()
            model = RecordingModel(
                [AgentDecision.call("crazyflie.raw_motor_command", {"rpm": 5000})]
            )

            async with Client(server) as client:
                result = await MCPAgentRunner(client, model).run("Spin the motors directly.")

            self.assertFalse(result.ok)
            self.assertFalse(result.finished)
            self.assertIn("did not expose", result.error)
            self.assertEqual(crazyflie.calls, [])

        asyncio.run(scenario())

    def test_tool_failure_stops_agent_by_default(self) -> None:
        async def scenario() -> None:
            server, xlerobot, _, humanoid, _ = make_server(xlerobot_fail_skill="navigate_to")
            model = RecordingModel(
                [
                    AgentDecision.call(
                        "xlerobot.navigate_to",
                        {"x_m": 1.0, "y_m": 0.0},
                    ),
                    AgentDecision.call("humanoid.stand"),
                ]
            )

            async with Client(server) as client:
                result = await MCPAgentRunner(client, model).run("Move the ground robot, then stand.")

            self.assertFalse(result.ok)
            self.assertIn("tool failed", result.error)
            self.assertEqual(len(xlerobot.calls), 1)
            self.assertEqual(humanoid.calls, [])
            self.assertEqual(len(model.contexts), 1)

        asyncio.run(scenario())

    def test_max_steps_bounds_runaway_agent(self) -> None:
        async def scenario() -> None:
            server, _, crazyflie, _, _ = make_server()
            model = RepeatingModel(
                AgentDecision.call("crazyflie.takeoff", {"altitude_m": 1.0})
            )

            async with Client(server) as client:
                result = await MCPAgentRunner(client, model, max_steps=2).run(
                    "Keep taking off forever."
                )

            self.assertFalse(result.ok)
            self.assertFalse(result.finished)
            self.assertEqual(result.error, "maximum agent steps exceeded: 2")
            self.assertEqual(len(result.actions), 2)
            self.assertEqual(len(crazyflie.calls), 2)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
