from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from embodied_agent.mcp import (
    AgentActionRecord,
    AgentContext,
    AgentDecisionKind,
    AgentToolDescription,
)
from embodied_agent.mcp.openai_model import OpenAIAgentModel, _OpenAIDecisionPayload


class FakeResponses:
    def __init__(self, payloads: list[_OpenAIDecisionPayload]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self.payloads:
            raise AssertionError("no fake OpenAI response payload remaining")
        return SimpleNamespace(output_parsed=self.payloads.pop(0), output=[])


class FakeOpenAIClient:
    def __init__(self, payloads: list[_OpenAIDecisionPayload]) -> None:
        self.responses = FakeResponses(payloads)


def context() -> AgentContext:
    return AgentContext(
        instruction="Scout the workbench, then move the ground robot there.",
        step_index=1,
        world_snapshot={
            "version": 3,
            "entities": {
                "workbench": {
                    "entity_id": "workbench",
                    "kind": "waypoint",
                    "pose": {
                        "x_m": 1.5,
                        "y_m": 0.5,
                        "z_m": 1.0,
                        "yaw_rad": 0.0,
                        "frame": "world",
                    },
                }
            },
            "robots": {
                "crazyflie": {
                    "tool": "crazyflie.takeoff",
                    "ok": True,
                    "detail": "ok",
                    "data": {"altitude_m": 1.0},
                }
            },
            "tasks": {},
        },
        tools=(
            AgentToolDescription(
                name="crazyflie.goto",
                description="Fly to a bounded XYZ position.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "x_m": {"type": "number"},
                        "y_m": {"type": "number"},
                        "z_m": {"type": "number"},
                    },
                    "required": ["x_m", "y_m", "z_m"],
                    "additionalProperties": False,
                },
            ),
            AgentToolDescription(
                name="xlerobot.navigate_to",
                description="Navigate the ground robot to XY.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "x_m": {"type": "number"},
                        "y_m": {"type": "number"},
                    },
                    "required": ["x_m", "y_m"],
                    "additionalProperties": False,
                },
            ),
        ),
        history=(
            AgentActionRecord(
                step_index=0,
                tool="crazyflie.takeoff",
                arguments={"altitude_m": 1.0},
                ok=True,
                detail="ok",
                structured_content={"tool": "crazyflie.takeoff", "ok": True},
            ),
        ),
    )


class OpenAIAgentModelTests(unittest.TestCase):
    def test_tool_decision_is_converted_to_agent_decision(self) -> None:
        async def scenario() -> None:
            client = FakeOpenAIClient(
                [
                    _OpenAIDecisionPayload(
                        kind="tool",
                        tool="crazyflie.goto",
                        arguments_json='{"x_m":1.5,"y_m":0.5,"z_m":1.0}',
                        summary="",
                    )
                ]
            )
            model = OpenAIAgentModel(model="test-model", client=client)
            decision = await model.decide(context())

            self.assertEqual(decision.kind, AgentDecisionKind.TOOL)
            self.assertEqual(decision.tool, "crazyflie.goto")
            self.assertEqual(decision.arguments, {"x_m": 1.5, "y_m": 0.5, "z_m": 1.0})

        asyncio.run(scenario())

    def test_finish_decision_is_converted_to_agent_finish(self) -> None:
        async def scenario() -> None:
            client = FakeOpenAIClient(
                [
                    _OpenAIDecisionPayload(
                        kind="finish",
                        tool=None,
                        arguments_json="{}",
                        summary="Task complete.",
                    )
                ]
            )
            model = OpenAIAgentModel(model="test-model", client=client)
            decision = await model.decide(context())

            self.assertEqual(decision.kind, AgentDecisionKind.FINISH)
            self.assertEqual(decision.summary, "Task complete.")

        asyncio.run(scenario())

    def test_provider_rejects_tool_not_exposed_in_context(self) -> None:
        async def scenario() -> None:
            client = FakeOpenAIClient(
                [
                    _OpenAIDecisionPayload(
                        kind="tool",
                        tool="crazyflie.raw_motor_command",
                        arguments_json='{"rpm":9000}',
                        summary="",
                    )
                ]
            )
            model = OpenAIAgentModel(model="test-model", client=client)
            with self.assertRaisesRegex(ValueError, "not exposed by MCP"):
                await model.decide(context())

        asyncio.run(scenario())

    def test_provider_rejects_non_object_arguments_json(self) -> None:
        async def scenario() -> None:
            client = FakeOpenAIClient(
                [
                    _OpenAIDecisionPayload(
                        kind="tool",
                        tool="crazyflie.goto",
                        arguments_json="[1,2,3]",
                        summary="",
                    )
                ]
            )
            model = OpenAIAgentModel(model="test-model", client=client)
            with self.assertRaisesRegex(ValueError, "JSON object"):
                await model.decide(context())

        asyncio.run(scenario())

    def test_request_contains_live_world_tools_history_and_structured_output_type(self) -> None:
        async def scenario() -> None:
            client = FakeOpenAIClient(
                [
                    _OpenAIDecisionPayload(
                        kind="finish",
                        tool=None,
                        arguments_json="{}",
                        summary="done",
                    )
                ]
            )
            model = OpenAIAgentModel(
                model="configured-model",
                client=client,
                max_output_tokens=700,
            )
            await model.decide(context())

            call = client.responses.calls[0]
            self.assertEqual(call["model"], "configured-model")
            self.assertEqual(call["max_output_tokens"], 700)
            self.assertIs(call["text_format"], _OpenAIDecisionPayload)
            self.assertIn("Never invent a tool name", call["instructions"])

            raw_input = call["input"]
            start = raw_input.index("<agent_context_json>\n") + len("<agent_context_json>\n")
            end = raw_input.index("\n</agent_context_json>")
            payload = json.loads(raw_input[start:end])
            self.assertEqual(payload["world_snapshot"]["version"], 3)
            self.assertEqual(payload["tools"][0]["name"], "crazyflie.goto")
            self.assertEqual(payload["history"][0]["tool"], "crazyflie.takeoff")
            self.assertEqual(payload["instruction"], context().instruction)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
