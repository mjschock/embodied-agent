from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


_POSE_FIELDS = frozenset({"x_m", "y_m", "z_m", "yaw_rad"})


@dataclass(frozen=True, slots=True)
class Pose3D:
    x_m: float
    y_m: float
    z_m: float = 0.0
    yaw_rad: float = 0.0
    frame: str = "world"

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "yaw_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not self.frame.strip():
            raise ValueError("frame must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "yaw_rad": self.yaw_rad,
            "frame": self.frame,
        }


@dataclass(frozen=True, slots=True)
class WorldEntity:
    entity_id: str
    kind: str
    pose: Pose3D
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must be non-empty")
        if not self.kind.strip():
            raise ValueError("kind must be non-empty")
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "attributes", dict(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "pose": self.pose.to_dict(),
            "attributes": dict(self.attributes),
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class EntityFieldRef:
    """Late-bound reference to one pose field of a named world entity."""

    entity_id: str
    field: str
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must be non-empty")
        if self.field not in _POSE_FIELDS:
            raise ValueError(f"unsupported entity pose field: {self.field}")
        offset = float(self.offset)
        if not math.isfinite(offset):
            raise ValueError("offset must be finite")
        object.__setattr__(self, "offset", offset)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TaskStepRecord:
    step_index: int
    tool: str
    ok: bool
    detail: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool": self.tool,
            "ok": self.ok,
            "detail": self.detail,
            "arguments": dict(self.arguments),
        }


@dataclass(slots=True)
class TaskRunState:
    task_name: str
    total_steps: int
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    history: list[TaskStepRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "total_steps": self.total_steps,
            "status": self.status.value,
            "current_step": self.current_step,
            "history": [record.to_dict() for record in self.history],
        }


class WorldState:
    """Shared symbolic state used by planners, executors, and multiple embodiments."""

    def __init__(self) -> None:
        self._entities: dict[str, WorldEntity] = {}
        self._robot_results: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, TaskRunState] = {}
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def upsert_entity(self, entity: WorldEntity) -> None:
        self._entities[entity.entity_id] = entity
        self._version += 1

    def remove_entity(self, entity_id: str) -> None:
        try:
            del self._entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"unknown world entity: {entity_id}") from exc
        self._version += 1

    def get_entity(self, entity_id: str) -> WorldEntity:
        try:
            return self._entities[entity_id]
        except KeyError as exc:
            raise KeyError(f"unknown world entity: {entity_id}") from exc

    def entities(self) -> tuple[WorldEntity, ...]:
        return tuple(self._entities[name] for name in sorted(self._entities))

    def resolve_value(self, value: Any) -> Any:
        if isinstance(value, EntityFieldRef):
            entity = self.get_entity(value.entity_id)
            if entity.pose.frame != "world":
                raise ValueError(
                    f"entity {value.entity_id} is in frame={entity.pose.frame!r}; world-frame value required"
                )
            return float(getattr(entity.pose, value.field)) + value.offset
        if isinstance(value, Mapping):
            return {str(key): self.resolve_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.resolve_value(item) for item in value)
        return value

    def resolve_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {str(name): self.resolve_value(value) for name, value in arguments.items()}

    def begin_task(self, task_name: str, total_steps: int) -> TaskRunState:
        if total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        state = TaskRunState(
            task_name=task_name,
            total_steps=total_steps,
            status=TaskStatus.RUNNING,
        )
        self._tasks[task_name] = state
        self._version += 1
        return state

    def record_task_step(
        self,
        task_name: str,
        *,
        step_index: int,
        tool: str,
        ok: bool,
        detail: str,
        arguments: Mapping[str, Any],
    ) -> None:
        state = self.get_task(task_name)
        state.current_step = step_index + 1
        state.history.append(
            TaskStepRecord(
                step_index=step_index,
                tool=tool,
                ok=ok,
                detail=detail,
                arguments=dict(arguments),
            )
        )
        if not ok:
            state.status = TaskStatus.FAILED
        self._version += 1

    def finish_task(self, task_name: str, *, ok: bool) -> None:
        state = self.get_task(task_name)
        state.status = TaskStatus.SUCCEEDED if ok else TaskStatus.FAILED
        self._version += 1

    def get_task(self, task_name: str) -> TaskRunState:
        try:
            return self._tasks[task_name]
        except KeyError as exc:
            raise KeyError(f"unknown task state: {task_name}") from exc

    def record_robot_result(
        self,
        tool_name: str,
        *,
        ok: bool,
        detail: str,
        data: Mapping[str, Any],
    ) -> None:
        robot_name = tool_name.split(".", 1)[0]
        self._robot_results[robot_name] = {
            "tool": tool_name,
            "ok": bool(ok),
            "detail": detail,
            "data": dict(data),
        }
        self._version += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "entities": {
                entity.entity_id: entity.to_dict()
                for entity in self.entities()
            },
            "robots": {name: dict(value) for name, value in sorted(self._robot_results.items())},
            "tasks": {
                name: state.to_dict()
                for name, state in sorted(self._tasks.items())
            },
        }


def entity_pose_refs(
    entity_id: str,
    *fields: str,
) -> dict[str, EntityFieldRef]:
    selected = fields or ("x_m", "y_m")
    return {field: EntityFieldRef(entity_id, field) for field in selected}
