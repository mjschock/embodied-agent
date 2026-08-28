from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from embodied_agent.core import Capability, RobotRegistry


class ToolPermissionError(PermissionError):
    """Raised when an agent attempts to call a tool that was not explicitly exposed."""


class ToolValidationError(ValueError):
    """Raised when agent-provided tool arguments fail schema validation."""


@dataclass(frozen=True, slots=True)
class ParamRule:
    kind: str = "number"
    required: bool = True
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None

    def normalize(self, name: str, value: Any) -> Any:
        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolValidationError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value):
                raise ToolValidationError(f"{name} must be finite")
            if self.minimum is not None and value < self.minimum:
                raise ToolValidationError(f"{name} must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ToolValidationError(f"{name} must be <= {self.maximum}")
            return value
        if self.kind == "string":
            if not isinstance(value, str) or not value.strip():
                raise ToolValidationError(f"{name} must be a non-empty string")
            return value
        raise ToolValidationError(f"unsupported parameter kind for {name}: {self.kind}")

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "number" if self.kind == "number" else "string"}
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if not self.required and self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass(frozen=True, slots=True)
class SkillToolSpec:
    skill: str
    capability: Capability
    params: Mapping[str, ParamRule] = field(default_factory=dict)

    def validate(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(supplied) - set(self.params))
        if unknown:
            raise ToolValidationError(f"unexpected parameters for {self.skill}: {', '.join(unknown)}")

        normalized: dict[str, Any] = {}
        for name, rule in self.params.items():
            if name in supplied:
                normalized[name] = rule.normalize(name, supplied[name])
            elif rule.required:
                raise ToolValidationError(f"missing required parameter: {name}")
            else:
                normalized[name] = rule.default
        return normalized

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {name: rule.json_schema() for name, rule in self.params.items()},
            "required": [name for name, rule in self.params.items() if rule.required],
            "additionalProperties": False,
        }


_NUMBER = ParamRule

SAFE_SKILL_SPECS: dict[str, SkillToolSpec] = {
    "observe": SkillToolSpec("observe", Capability.OBSERVE),
    "reset": SkillToolSpec("reset", Capability.OBSERVE),
    "takeoff": SkillToolSpec(
        "takeoff",
        Capability.FLY,
        {"altitude_m": _NUMBER(required=False, default=1.0, minimum=0.05, maximum=3.0)},
    ),
    "goto": SkillToolSpec(
        "goto",
        Capability.FLY,
        {
            "x_m": _NUMBER(minimum=-20.0, maximum=20.0),
            "y_m": _NUMBER(minimum=-20.0, maximum=20.0),
            "z_m": _NUMBER(minimum=0.03, maximum=3.0),
            # Omission is meaningful: the robot adapter derives a distance-aware
            # safe timeout from its actual simulator/runtime speed envelope.
            "timeout_s": _NUMBER(required=False, default=None, minimum=0.1, maximum=30.0),
        },
    ),
    "land": SkillToolSpec(
        "land",
        Capability.FLY,
        {"height_m": _NUMBER(required=False, default=0.05, minimum=0.0, maximum=1.0)},
    ),
    "drive_velocity": SkillToolSpec(
        "drive_velocity",
        Capability.NAVIGATE,
        {
            "lin_x_mps": _NUMBER(required=False, default=0.0, minimum=-0.5, maximum=0.5),
            "lin_y_mps": _NUMBER(required=False, default=0.0, minimum=-0.5, maximum=0.5),
            "yaw_rate_rps": _NUMBER(required=False, default=0.0, minimum=-0.8, maximum=0.8),
            "duration_s": _NUMBER(required=False, default=1.0, minimum=0.01, maximum=5.0),
        },
    ),
    "navigate_to": SkillToolSpec(
        "navigate_to",
        Capability.NAVIGATE,
        {
            "x_m": _NUMBER(minimum=-20.0, maximum=20.0),
            "y_m": _NUMBER(minimum=-20.0, maximum=20.0),
            "yaw_rad": _NUMBER(required=False, default=0.0, minimum=-math.tau, maximum=math.tau),
            # As with drone goto, omission lets the adapter derive a bounded
            # duration from the requested transit and configured speed envelope.
            "max_duration_s": _NUMBER(required=False, default=None, minimum=0.1, maximum=30.0),
        },
    ),
    "stand": SkillToolSpec("stand", Capability.STAND),
    "walk_velocity": SkillToolSpec(
        "walk_velocity",
        Capability.WALK,
        {
            "lin_x_mps": _NUMBER(required=False, default=0.0, minimum=-0.75, maximum=0.75),
            "lin_y_mps": _NUMBER(required=False, default=0.0, minimum=-0.5, maximum=0.5),
            "yaw_rate_rps": _NUMBER(required=False, default=0.0, minimum=-0.8, maximum=0.8),
            "duration_s": _NUMBER(required=False, default=1.0, minimum=0.01, maximum=5.0),
        },
    ),
}


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    tool: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


class RobotToolRouter:
    """Allowlisted, schema-validated tool boundary for a high-level agent."""

    def __init__(
        self,
        registry: RobotRegistry,
        allowed_skills: Mapping[str, list[str] | tuple[str, ...]],
    ) -> None:
        self.registry = registry
        self.allowed_skills = {name: tuple(skills) for name, skills in allowed_skills.items()}

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for robot in self.registry:
            for skill in self.allowed_skills.get(robot.name, ()):
                spec = SAFE_SKILL_SPECS.get(skill)
                if spec is None or not robot.supports(spec.capability):
                    continue
                tools.append(
                    {
                        "name": f"{robot.name}.{skill}",
                        "robot": robot.name,
                        "skill": skill,
                        "capability": spec.capability.value,
                        "input_schema": spec.json_schema(),
                    }
                )
        return sorted(tools, key=lambda item: item["name"])

    def tools_for_skill(self, skill: str) -> list[dict[str, Any]]:
        return [tool for tool in self.list_tools() if tool["skill"] == skill]

    async def call(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> ToolCallResult:
        robot_name, skill = self._parse_tool_name(tool_name)
        robot = self.registry.get(robot_name)
        if skill not in self.allowed_skills.get(robot_name, ()):
            raise ToolPermissionError(f"tool is not allowlisted: {tool_name}")

        spec = SAFE_SKILL_SPECS.get(skill)
        if spec is None:
            raise ToolPermissionError(f"tool has no safe schema: {tool_name}")
        if not robot.supports(spec.capability):
            raise ToolPermissionError(
                f"{robot_name} does not currently advertise {spec.capability.value}"
            )

        params = spec.validate(arguments or {})
        try:
            if skill == "observe":
                observation = await robot.observe()
                return ToolCallResult(
                    tool=tool_name,
                    ok=True,
                    detail=f"Observed {robot_name}.",
                    data={"state": observation.state, "images": observation.images},
                )

            robot_params = self._to_robot_params(skill, params)
            result = await robot.execute(skill, **robot_params)
            return ToolCallResult(
                tool=tool_name,
                ok=result.ok,
                detail=result.detail,
                data=dict(result.data),
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            return ToolCallResult(tool=tool_name, ok=False, detail=str(exc))

    @staticmethod
    def _parse_tool_name(tool_name: str) -> tuple[str, str]:
        if tool_name.count(".") != 1:
            raise ToolPermissionError("tool names must be '<robot>.<skill>'")
        robot_name, skill = tool_name.split(".", 1)
        if not robot_name or not skill:
            raise ToolPermissionError("tool names must be '<robot>.<skill>'")
        return robot_name, skill

    @staticmethod
    def _to_robot_params(skill: str, params: dict[str, Any]) -> dict[str, Any]:
        if skill == "goto":
            robot_params: dict[str, Any] = {
                "position": (params["x_m"], params["y_m"], params["z_m"]),
            }
            if params.get("timeout_s") is not None:
                robot_params["timeout_s"] = params["timeout_s"]
            return robot_params
        if skill == "navigate_to" and params.get("max_duration_s") is None:
            return {key: value for key, value in params.items() if key != "max_duration_s"}
        return params
