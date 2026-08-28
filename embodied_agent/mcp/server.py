from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server, ServerRequestContext

from embodied_agent.agent import RobotToolRouter, build_stack, load_config
from embodied_agent.core import RobotRegistry
from embodied_agent.world import WorldState


TOOL_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "ok": {"type": "boolean"},
        "detail": {"type": "string"},
        "data": {"type": "object"},
    },
    "required": ["tool", "ok", "detail", "data"],
    "additionalProperties": False,
}

WORLD_SNAPSHOT_URI = "world://snapshot"
WORLD_ENTITY_URI_PREFIX = "world://entities/"


def _json_safe(value: Any) -> Any:
    """Convert common simulator values into MCP/JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass

    return repr(value)


def _tool_description(robot: str, skill: str) -> str:
    descriptions = {
        "observe": "Read the robot's current normalized state.",
        "reset": "Reset the robot simulation/controller to its initial state.",
        "takeoff": "Take off to a bounded target altitude.",
        "goto": "Fly to a bounded XYZ position using the robot's closed-loop controller.",
        "land": "Land to a bounded target height.",
        "drive_velocity": "Drive the mobile base at a bounded velocity for a bounded duration.",
        "navigate_to": "Navigate the mobile base to a bounded XY pose using closed-loop control.",
        "stand": "Command the humanoid to a standing posture.",
        "walk_velocity": "Walk with a bounded velocity command for a bounded duration.",
    }
    action = descriptions.get(skill, f"Execute the safe semantic skill {skill!r}.")
    return f"{action} Target embodiment: {robot}. Raw motor and joint control is not exposed."


def _entity_uri(entity_id: str) -> str:
    return f"{WORLD_ENTITY_URI_PREFIX}{quote(entity_id, safe='')}"


def _json_resource(uri: str, payload: Any) -> types.ReadResourceResult:
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(
                uri=uri,
                text=json.dumps(_json_safe(payload), indent=2, sort_keys=True),
                mime_type="application/json",
            )
        ]
    )


def build_mcp_server(
    router: RobotToolRouter,
    *,
    registry: RobotRegistry | None = None,
    world: WorldState | None = None,
    name: str = "embodied-agent",
    version: str = "0.1.0",
) -> Server[Any]:
    """Expose a RobotToolRouter and read-only WorldState through MCP v2.

    The low-level server is intentional: the router already owns the authoritative
    JSON Schemas, so MCP publishes those schemas verbatim instead of re-deriving a
    second contract from Python function signatures. World resources are read-only;
    trusted perception/localization code owns entity mutation.
    """

    registry = registry or router.registry
    world = world or WorldState()

    @asynccontextmanager
    async def lifespan(_server: Server[Any]):
        connected = []
        try:
            for robot in registry:
                await robot.connect()
                connected.append(robot)
            yield {"registry": registry, "router": router, "world": world}
        finally:
            for robot in reversed(connected):
                try:
                    await robot.disconnect()
                except Exception:
                    # Teardown should continue so one backend cannot strand the rest.
                    pass

    async def list_tools(
        _ctx: ServerRequestContext[Any],
        _params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=tool["name"],
                    description=_tool_description(tool["robot"], tool["skill"]),
                    input_schema=tool["input_schema"],
                    output_schema=TOOL_RESULT_SCHEMA,
                )
                for tool in router.list_tools()
            ]
        )

    async def call_tool(
        _ctx: ServerRequestContext[Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        try:
            result = await router.call(params.name, params.arguments or {})
            payload = {
                "tool": result.tool,
                "ok": result.ok,
                "detail": result.detail,
                "data": _json_safe(result.data),
            }
            world.record_robot_result(
                result.tool,
                ok=result.ok,
                detail=result.detail,
                data=payload["data"],
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result.detail)],
                structured_content=payload,
                is_error=not result.ok,
            )
        except (KeyError, PermissionError, ValueError) as exc:
            detail = str(exc)
            payload = {
                "tool": params.name,
                "ok": False,
                "detail": detail,
                "data": {},
            }
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=detail)],
                structured_content=payload,
                is_error=True,
            )

    async def list_resources(
        _ctx: ServerRequestContext[Any],
        _params: types.PaginatedRequestParams | None,
    ) -> types.ListResourcesResult:
        resources = [
            types.Resource(
                uri=WORLD_SNAPSHOT_URI,
                name="world-snapshot",
                title="Current world snapshot",
                description=(
                    "Read-only live snapshot of known entities, latest robot results, "
                    "task execution state, and world version."
                ),
                mime_type="application/json",
            )
        ]
        resources.extend(
            types.Resource(
                uri=_entity_uri(entity.entity_id),
                name=f"world-entity-{entity.entity_id}",
                title=f"World entity: {entity.entity_id}",
                description=f"Read-only live state for the {entity.kind} entity {entity.entity_id!r}.",
                mime_type="application/json",
            )
            for entity in world.entities()
        )
        return types.ListResourcesResult(resources=resources)

    async def read_resource(
        _ctx: ServerRequestContext[Any],
        params: types.ReadResourceRequestParams,
    ) -> types.ReadResourceResult:
        uri = str(params.uri)
        parsed = urlparse(uri)
        if parsed.scheme != "world":
            raise ValueError(f"unsupported resource URI: {uri}")

        if parsed.netloc == "snapshot" and parsed.path in ("", "/"):
            return _json_resource(uri, world.snapshot())

        if parsed.netloc == "entities":
            entity_id = unquote(parsed.path.lstrip("/"))
            if not entity_id:
                raise ValueError(f"missing entity id in resource URI: {uri}")
            entity = world.get_entity(entity_id)
            return _json_resource(uri, entity.to_dict())

        raise ValueError(f"unknown world resource: {uri}")

    return Server(
        name,
        version=version,
        instructions=(
            "Use resources to read current world state and use only the listed semantic "
            "robot tools for actions. The server enforces capability checks, parameter "
            "bounds, and allowlists before commands reach a robot."
        ),
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )


def build_mcp_server_from_config(
    config_path: str | Path,
    *,
    world: WorldState | None = None,
    name: str = "embodied-agent",
    version: str = "0.1.0",
) -> tuple[Server[Any], RobotRegistry, RobotToolRouter]:
    config = load_config(config_path)
    registry, router = build_stack(config)
    server = build_mcp_server(
        router,
        registry=registry,
        world=world,
        name=name,
        version=version,
    )
    return server, registry, router


async def serve_stdio(server: Server[Any]) -> None:
    """Serve a low-level MCP server over stdio for desktop/agent hosts."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
