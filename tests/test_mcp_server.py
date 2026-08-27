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
    return build_mcp_server(router, registry=registry), router, drone


class MCPServerTests(unittest.TestCase):
    def test_lists_only_safe_capability_supported_tools_with_exact_schema(self) -> None:
        async def scenario() -> None:
            server, router, _ = make_server()
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
            server, _, drone = make_server()
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
            server, _, drone = make_server()
            async with Client(server) as client:
                result = await client.call_tool("crazyflie.takeoff", {"altitude_m": 99.0})
                self.assertTrue(result.is_error)
                self.assertEqual(drone.calls, [])

        asyncio.run(scenario())

    def test_robot_execution_failure_is_an_mcp_tool_error_not_planning_error(self) -> None:
        async def scenario() -> None:
            server, _, drone = make_server(fail_skill="takeoff")
            async with Client(server) as client:
                result = await client.call_tool("crazyflie.takeoff", {"altitude_m": 1.0})
                self.assertTrue(result.is_error)
                self.assertFalse(result.structured_content["ok"])
                self.assertEqual(result.structured_content["detail"], "forced controller failure")
                self.assertEqual(len(drone.calls), 1)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
