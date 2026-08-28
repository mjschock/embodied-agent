from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .state import Pose3D, WorldEntity, WorldState


@dataclass(frozen=True, slots=True)
class EntityObservation:
    """Grounded entity observation produced by trusted perception/localization code.

    `observed_at_s` must use a shared time base across producers that may update the
    same entity. Frame transformation is intentionally outside this class: by
    default the ingestor accepts only world-frame poses.
    """

    entity_id: str
    kind: str
    pose: Pose3D
    source: str
    observed_at_s: float
    confidence: float
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must be non-empty")
        if not self.kind.strip():
            raise ValueError("kind must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")

        observed_at_s = float(self.observed_at_s)
        if not math.isfinite(observed_at_s):
            raise ValueError("observed_at_s must be finite")
        object.__setattr__(self, "observed_at_s", observed_at_s)

        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "attributes", dict(self.attributes))


class IngestStatus(StrEnum):
    ACCEPTED = "accepted"
    UNTRUSTED_SOURCE = "untrusted_source"
    LOW_CONFIDENCE = "low_confidence"
    NON_WORLD_FRAME = "non_world_frame"
    STALE = "stale"
    KIND_MISMATCH = "kind_mismatch"


@dataclass(frozen=True, slots=True)
class WorldIngestResult:
    status: IngestStatus
    entity_id: str
    source: str
    world_version: int
    detail: str
    entity: WorldEntity | None = None

    @property
    def accepted(self) -> bool:
        return self.status is IngestStatus.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "accepted": self.accepted,
            "entity_id": self.entity_id,
            "source": self.source,
            "world_version": self.world_version,
            "detail": self.detail,
            "entity": self.entity.to_dict() if self.entity is not None else None,
        }


class TrustedWorldIngestor:
    """Policy boundary for perception/localization updates into shared world state.

    This object is intended to be held by trusted sensor/perception infrastructure,
    not exposed as a generic LLM/MCP tool. The default policy requires an allowlisted
    source, a minimum confidence, world-frame coordinates, monotonic observation
    timestamps, and stable entity kind.
    """

    def __init__(
        self,
        world: WorldState,
        *,
        allowed_sources: Iterable[str],
        min_confidence: float = 0.5,
        require_world_frame: bool = True,
        reject_stale: bool = True,
        allow_kind_change: bool = False,
        merge_attributes: bool = True,
    ) -> None:
        normalized_sources = frozenset(source.strip() for source in allowed_sources if source.strip())
        if not normalized_sources:
            raise ValueError("allowed_sources must contain at least one non-empty source")

        min_confidence = float(min_confidence)
        if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        self.world = world
        self.allowed_sources = normalized_sources
        self.min_confidence = min_confidence
        self.require_world_frame = bool(require_world_frame)
        self.reject_stale = bool(reject_stale)
        self.allow_kind_change = bool(allow_kind_change)
        self.merge_attributes = bool(merge_attributes)

    def ingest(self, observation: EntityObservation) -> WorldIngestResult:
        if observation.source not in self.allowed_sources:
            return self._reject(
                observation,
                IngestStatus.UNTRUSTED_SOURCE,
                f"source is not allowlisted: {observation.source}",
            )

        if observation.confidence < self.min_confidence:
            return self._reject(
                observation,
                IngestStatus.LOW_CONFIDENCE,
                f"confidence {observation.confidence:.3f} is below {self.min_confidence:.3f}",
            )

        if self.require_world_frame and observation.pose.frame != "world":
            return self._reject(
                observation,
                IngestStatus.NON_WORLD_FRAME,
                (
                    f"pose frame {observation.pose.frame!r} is not 'world'; "
                    "transform the observation before ingestion"
                ),
            )

        existing = self._existing(observation.entity_id)
        if existing is not None:
            if not self.allow_kind_change and existing.kind != observation.kind:
                return self._reject(
                    observation,
                    IngestStatus.KIND_MISMATCH,
                    f"existing kind={existing.kind!r}, observed kind={observation.kind!r}",
                )
            if (
                self.reject_stale
                and existing.observed_at_s is not None
                and observation.observed_at_s <= existing.observed_at_s
            ):
                return self._reject(
                    observation,
                    IngestStatus.STALE,
                    (
                        f"observation timestamp {observation.observed_at_s} is not newer than "
                        f"current timestamp {existing.observed_at_s}"
                    ),
                )

        attributes: dict[str, Any] = {}
        if self.merge_attributes and existing is not None:
            attributes.update(existing.attributes)
        attributes.update(observation.attributes)

        entity = WorldEntity(
            entity_id=observation.entity_id,
            kind=observation.kind,
            pose=observation.pose,
            attributes=attributes,
            source=observation.source,
            confidence=observation.confidence,
            observed_at_s=observation.observed_at_s,
        )
        self.world.upsert_entity(entity)
        return WorldIngestResult(
            status=IngestStatus.ACCEPTED,
            entity_id=observation.entity_id,
            source=observation.source,
            world_version=self.world.version,
            detail="observation accepted",
            entity=entity,
        )

    def ingest_many(self, observations: Iterable[EntityObservation]) -> tuple[WorldIngestResult, ...]:
        return tuple(self.ingest(observation) for observation in observations)

    def _existing(self, entity_id: str) -> WorldEntity | None:
        try:
            return self.world.get_entity(entity_id)
        except KeyError:
            return None

    def _reject(
        self,
        observation: EntityObservation,
        status: IngestStatus,
        detail: str,
    ) -> WorldIngestResult:
        return WorldIngestResult(
            status=status,
            entity_id=observation.entity_id,
            source=observation.source,
            world_version=self.world.version,
            detail=detail,
            entity=self._existing(observation.entity_id),
        )
