from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class AgentDecisionKind(StrEnum):
    TOOL = "tool"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class AgentToolDescription:
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AgentActionRecord:
    step_index: int
    tool: str
    arguments: Mapping[str, Any]
    ok: bool
    detail: str
    structured_content: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentContext:
    instruction: str
    step_index: int
    world_snapshot: Mapping[str, Any]
    tools: tuple[AgentToolDescription, ...]
    history: tuple[AgentActionRecord, ...]


@dataclass(frozen=True, slots=True)
class AgentDecision:
    kind: AgentDecisionKind
    tool: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""

    def __post_init__(self) -> None:
        if self.kind is AgentDecisionKind.TOOL:
            if not self.tool or not self.tool.strip():
                raise ValueError("tool decisions require a non-empty tool name")
            object.__setattr__(self, "arguments", dict(self.arguments))
            return
        if self.kind is AgentDecisionKind.FINISH:
            if self.tool is not None:
                raise ValueError("finish decisions cannot include a tool")
            if self.arguments:
                raise ValueError("finish decisions cannot include tool arguments")
            return
        raise ValueError(f"unsupported agent decision kind: {self.kind}")

    @classmethod
    def call(cls, tool: str, arguments: Mapping[str, Any] | None = None) -> AgentDecision:
        return cls(AgentDecisionKind.TOOL, tool=tool, arguments=arguments or {})

    @classmethod
    def finish(cls, summary: str = "") -> AgentDecision:
        return cls(AgentDecisionKind.FINISH, summary=summary)


class AgentModel(Protocol):
    """Provider-agnostic high-level decision model.

    Implementations receive only the current world snapshot, the safe MCP tool
    surface, the original instruction, and prior action outcomes. They return one
    next action or a finish decision. The model never receives a direct robot or
    controller object.
    """

    async def decide(self, context: AgentContext) -> AgentDecision:
        ...


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    instruction: str
    ok: bool
    finished: bool
    summary: str
    actions: tuple[AgentActionRecord, ...]
    error: str = ""


class MCPAgentRunner:
    """Iterative high-level agent over MCP resources and safe semantic tools.

    The runner re-reads `world://snapshot` before every model decision. This is
    intentionally different from generating a fixed multi-step plan up front: a
    perception correction or robot outcome can change the next decision without
    requiring stale coordinates to be copied through the entire plan.
    """

    def __init__(
        self,
        client: Any,
        model: AgentModel,
        *,
        max_steps: int = 12,
        stop_on_tool_error: bool = True,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.client = client
        self.model = model
        self.max_steps = int(max_steps)
        self.stop_on_tool_error = bool(stop_on_tool_error)

    async def run(self, instruction: str) -> AgentRunResult:
        if not instruction.strip():
            raise ValueError("instruction must be non-empty")

        tools = await self._load_tools()
        allowed_tools = {tool.name for tool in tools}
        history: list[AgentActionRecord] = []

        for step_index in range(self.max_steps):
            world_snapshot = await self._read_world_snapshot()
            context = AgentContext(
                instruction=instruction,
                step_index=step_index,
                world_snapshot=world_snapshot,
                tools=tools,
                history=tuple(history),
            )
            decision = await self.model.decide(context)

            if decision.kind is AgentDecisionKind.FINISH:
                return AgentRunResult(
                    instruction=instruction,
                    ok=True,
                    finished=True,
                    summary=decision.summary,
                    actions=tuple(history),
                )

            assert decision.tool is not None
            if decision.tool not in allowed_tools:
                return AgentRunResult(
                    instruction=instruction,
                    ok=False,
                    finished=False,
                    summary="",
                    actions=tuple(history),
                    error=f"model selected a tool that MCP did not expose: {decision.tool}",
                )

            result = await self.client.call_tool(decision.tool, dict(decision.arguments))
            structured = dict(result.structured_content or {})
            detail = self._result_detail(result, structured)
            record = AgentActionRecord(
                step_index=step_index,
                tool=decision.tool,
                arguments=dict(decision.arguments),
                ok=not bool(result.is_error),
                detail=detail,
                structured_content=structured,
            )
            history.append(record)

            if result.is_error and self.stop_on_tool_error:
                return AgentRunResult(
                    instruction=instruction,
                    ok=False,
                    finished=False,
                    summary="",
                    actions=tuple(history),
                    error=f"tool failed: {decision.tool}: {detail}",
                )

        return AgentRunResult(
            instruction=instruction,
            ok=False,
            finished=False,
            summary="",
            actions=tuple(history),
            error=f"maximum agent steps exceeded: {self.max_steps}",
        )

    async def _load_tools(self) -> tuple[AgentToolDescription, ...]:
        result = await self.client.list_tools()
        return tuple(
            AgentToolDescription(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema),
            )
            for tool in result.tools
        )

    async def _read_world_snapshot(self) -> dict[str, Any]:
        result = await self.client.read_resource("world://snapshot")
        for content in result.contents:
            text = getattr(content, "text", None)
            if text is None:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("world://snapshot must contain a JSON object")
            return payload
        raise ValueError("world://snapshot did not return text JSON content")

    @staticmethod
    def _result_detail(result: Any, structured: Mapping[str, Any]) -> str:
        detail = structured.get("detail")
        if isinstance(detail, str):
            return detail
        for content in getattr(result, "content", ()):
            text = getattr(content, "text", None)
            if isinstance(text, str):
                return text
        return ""
