from __future__ import annotations

import asyncio
import json
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
        if not self.connected:
            raise RuntimeError("robot is not connected")
        return Observation(self.name, {"connected": self.connected})

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        if not self.connected:
            raise RuntimeError("robot is not connected")
        self.calls.append(request)
        ok = request.name != self.fail_skill
        return SkillResult(
            embodiment=self.name,
            skill=request.name,
            ok=ok,
            detail="ok" if ok else "forced controller failure",
            data={"params": dict(request.params)},
        )


def make_server(*, fail_skill: str | None = None):
    registry = RobotRegistry()
    drone = FakeRobot(
        "crazyflie",
        {Capability.OBSERVE, Capability.FLY},
        fail_skill=fail_skill,
    )
    registry.register(drone)
    router = RobotToolRouter(
        registry,
        {"crazyflie": ["observe", "takeoff", "goto", "land", "raw_motor_command"]},
    )
    world = WorldState()
    world.upsert_entity(
        WorldEntity(
            "inspection target",
            "waypoint",
            Pose3D(1.0, 2.0, 0.5),
            source="test",
            confidence=0.9,
        )
    )
    return build_mcp_server(router, registry=registry, world=world), router, drone, world


class MCPServerTests(unittest.TestCase):
    def test_lists_only_safe_capability_supported_tools_with_exact_schema(self) -> None:
        async def scenario() -> None:
            server, router, _, _ = make_server()
            async with Client(server) as client:
                result = await client.list_tools()
                by_name = {tool.name: tool for tool in result.tools}
                self.assertEqual(
                    set(by_name),
                    {
                        "crazyflie.observe",
                        "crazyflie.takeoff",
                        "crazyflie.goto",
                        "crazyflie.land",
                    },
                )
                router_tool = next(
                    tool for tool in router.list_tools() if tool["name"] == "crazyflie.goto"
                )
                self.assertEqual(
                    by_name["crazyflie.goto"].input_schema,
                    router_tool["input_schema"],
                )
                self.assertNotIn("crazyflie.raw_motor_command", by_name)

        asyncio.run(scenario())

    def test_mcp_lifespan_connects_and_disconnects_robot(self) -> None:
        async def scenario() -> None:
            server, _, drone, _ = make_server()
            self.assertFalse(drone.connected)
            async with Client(server) as client:
                self.assertTrue(drone.connected)
                result = await client.call_tool("crazyflie.takeoff", {"altitude_m": 1.25})
                self.assertFalse(result.is_error)
                self.assertEqual(result.structured_content["tool"], "crazyflie.takeoff")
                self.assertTrue(result.structured_content["ok"])
                self.assertEqual(drone.calls[-1].params["altitude_m"], 1.25)
            self.assertFalse(drone.connected)

        asyncio.run(scenario())

    def test_sdk_schema_validation_blocks_bad_arguments_before_robot(self) -> None:
        async def scenario() -> None:
            server, _, drone, _ = make_server()
            async with Client(server) as client:
                result = await client.call_tool("crazyflie.takeoff", {"altitude_m": 99.0})
                self.assertTrue(result.is_error)
                self.assertEqual(drone.calls, [])

        asyncio.run(scenario())

    def test_robot_execution_failure_is_an_mcp_tool_error_not_planning_error(self) -> None:
        async def scenario() -> None:
            server, _, drone, _ = make_server(fail_skill="takeoff")
            async with Client(server) as client:
                result = await client.call_tool("crazyflie.takeoff", {"altitude_m": 1.0})
                self.assertTrue(result.is_error)
                self.assertFalse(result.structured_content["ok"])
                self.assertEqual(result.structured_content["detail"], "forced controller failure")
                self.assertEqual(len(drone.calls), 1)

        asyncio.run(scenario())

    def test_world_resources_list_snapshot_and_current_entities(self) -> None:
        async def scenario() -> None:
            server, _, _, _ = make_server()
            async with Client(server) as client:
                listed = await client.list_resources()
                uris = {str(resource.uri) for resource in listed.resources}
                self.assertIn("world://snapshot", uris)
                self.assertIn("world://entities/inspection%20target", uris)

                snapshot = await client.read_resource("world://snapshot")
                payload = json.loads(snapshot.contents[0].text)
                self.assertEqual(payload["entities"]["inspection target"]["pose"]["x_m"], 1.0)
                self.assertEqual(payload["entities"]["inspection target"]["source"], "test")

        asyncio.run(scenario())

    def test_world_resource_reads_are_live_after_entity_update(self) -> None:
        async def scenario() -> None:
            server, _, _, world = make_server()
            async with Client(server) as client:
                first = await client.read_resource("world://entities/inspection%20target")
                first_payload = json.loads(first.contents[0].text)
                self.assertEqual(first_payload["pose"]["x_m"], 1.0)

                world.upsert_entity(
                    WorldEntity(
                        "inspection target",
                        "waypoint",
                        Pose3D(4.5, -1.25, 0.75),
                        source="perception-correction",
                        confidence=0.95,
                    )
                )

                second = await client.read_resource("world://entities/inspection%20target")
                second_payload = json.loads(second.contents[0].text)
                self.assertEqual(second_payload["pose"]["x_m"], 4.5)
                self.assertEqual(second_payload["pose"]["y_m"], -1.25)
                self.assertEqual(second_payload["source"], "perception-correction")

        asyncio.run(scenario())

    def test_mcp_tool_results_are_reflected_in_world_snapshot(self) -> None:
        async def scenario() -> None:
            server, _, _, _ = make_server()
            async with Client(server) as client:
                result = await client.call_tool("crazyflie.takeoff", {"altitude_m": 1.5})
                self.assertFalse(result.is_error)

                snapshot = await client.read_resource("world://snapshot")
                payload = json.loads(snapshot.contents[0].text)
                robot = payload["robots"]["crazyflie"]
                self.assertEqual(robot["tool"], "crazyflie.takeoff")
                self.assertTrue(robot["ok"])
                self.assertEqual(robot["data"]["params"]["altitude_m"], 1.5)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
