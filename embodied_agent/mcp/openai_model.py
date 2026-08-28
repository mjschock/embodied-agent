from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .agent import AgentContext, AgentDecision


class _OpenAIDecisionPayload(BaseModel):
    """Strict Structured Output shape for one high-level agent decision."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool", "finish"]
    tool: str | None
    arguments_json: str
    summary: str


_SYSTEM_INSTRUCTIONS = """You are the high-level planner for a multi-robot embodied system.

Choose exactly one next decision from the supplied context:
- kind=\"tool\": choose exactly one tool from the provided tool list and provide its arguments as a JSON object encoded in arguments_json.
- kind=\"finish\": use only when the user's task is complete; set tool to null and arguments_json to \"{}\".

Rules:
1. Never invent a tool name. Use only a tool listed in the current context.
2. Never attempt raw motor, torque, PWM, joint, or controller access.
3. Respect each tool's input schema. arguments_json must decode to one JSON object.
4. Treat the supplied world snapshot as the current source of truth. Do not invent coordinates, objects, robot outcomes, or task state that are absent from it.
5. Consider prior action outcomes before choosing the next action.
6. Choose one action at a time. The runtime will refresh world state before asking again.
7. Finish when the requested task is satisfied; do not add unnecessary robot motion.
"""


class OpenAIAgentModel:
    """OpenAI Responses API implementation of the provider-agnostic AgentModel.

    The provider uses Structured Outputs for the decision envelope. Robot tool
    argument validation remains authoritative in MCP/RobotToolRouter, because the
    available tool schemas differ dynamically by embodiment and configuration.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        max_output_tokens: int = 600,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_output_tokens < 64:
            raise ValueError("max_output_tokens must be >= 64")

        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()

        self.model = model
        self.client = client
        self.max_output_tokens = int(max_output_tokens)

    async def decide(self, context: AgentContext) -> AgentDecision:
        response = await self.client.responses.parse(
            model=self.model,
            instructions=_SYSTEM_INSTRUCTIONS,
            input=self._context_input(context),
            text_format=_OpenAIDecisionPayload,
            max_output_tokens=self.max_output_tokens,
        )
        payload = self._extract_payload(response)
        allowed_tools = {tool.name for tool in context.tools}

        if payload.kind == "finish":
            if payload.tool is not None:
                raise ValueError("OpenAI finish decision included a tool")
            arguments = self._parse_arguments(payload.arguments_json)
            if arguments:
                raise ValueError("OpenAI finish decision included tool arguments")
            return AgentDecision.finish(payload.summary)

        if payload.tool is None or not payload.tool.strip():
            raise ValueError("OpenAI tool decision did not include a tool name")
        if payload.tool not in allowed_tools:
            raise ValueError(f"OpenAI selected a tool not exposed by MCP: {payload.tool}")

        arguments = self._parse_arguments(payload.arguments_json)
        return AgentDecision.call(payload.tool, arguments)

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI arguments_json was not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("OpenAI arguments_json must decode to a JSON object")
        return value

    @staticmethod
    def _extract_payload(response: Any) -> _OpenAIDecisionPayload:
        direct = getattr(response, "output_parsed", None)
        if isinstance(direct, _OpenAIDecisionPayload):
            return direct

        for output in getattr(response, "output", ()):
            if getattr(output, "type", None) != "message":
                continue
            for item in getattr(output, "content", ()):
                if getattr(item, "type", None) != "output_text":
                    continue
                parsed = getattr(item, "parsed", None)
                if isinstance(parsed, _OpenAIDecisionPayload):
                    return parsed

        raise ValueError("OpenAI response did not contain a parsed agent decision")

    @staticmethod
    def _context_input(context: AgentContext) -> str:
        payload = {
            "instruction": context.instruction,
            "step_index": context.step_index,
            "world_snapshot": context.world_snapshot,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in context.tools
            ],
            "history": [
                {
                    "step_index": action.step_index,
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "ok": action.ok,
                    "detail": action.detail,
                    "structured_content": action.structured_content,
                }
                for action in context.history
            ],
        }
        return (
            "Choose the single next decision for this embodied task. "
            "The runtime will execute at most one tool and then refresh the world before the next decision.\n"
            "<agent_context_json>\n"
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
            + "\n</agent_context_json>"
        )
