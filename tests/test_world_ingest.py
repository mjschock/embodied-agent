from __future__ import annotations

import asyncio
import unittest

from embodied_agent.agent import Plan, PlanExecutor, PlanStep, RobotToolRouter
from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    RobotRegistry,
    SkillRequest,
    SkillResult,
)
from embodied_agent.world import (
    EntityObservation,
    IngestStatus,
    Pose3D,
    TrustedWorldIngestor,
    WorldEntity,
    WorldState,
    entity_pose_refs,
)


class RecordingRobot(Embodiment):
    def __init__(
        self,
        name: str,
        capabilities: set[Capability],
        *,
        on_execute=None,
    ) -> None:
        self.name = name
        self.backend = "test"
        self._capabilities = frozenset(capabilities)
        self.on_execute = on_execute
        self.calls: list[SkillRequest] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def observe(self) -> Observation:
        return Observation(self.name, {})

    async def execute_request(self, request: SkillRequest) -> SkillResult:
        self.calls.append(request)
        if self.on_execute is not None:
            self.on_execute(request)
        return SkillResult(self.name, request.name, True, "ok", dict(request.params))


class WorldIngestTests(unittest.TestCase):
    def _ingestor(self, world: WorldState | None = None) -> tuple[WorldState, TrustedWorldIngestor]:
        world = world or WorldState()
        ingestor = TrustedWorldIngestor(
            world,
            allowed_sources={"crazyflie-perception", "xlerobot-localization"},
            min_confidence=0.7,
        )
        return world, ingestor

    def test_accepts_fresh_trusted_world_frame_observation(self) -> None:
        world, ingestor = self._ingestor()
        result = ingestor.ingest(
            EntityObservation(
                "bottle",
                "object",
                Pose3D(1.0, 2.0, 0.8),
                source="crazyflie-perception",
                observed_at_s=100.0,
                confidence=0.92,
                attributes={"color": "blue"},
            )
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, IngestStatus.ACCEPTED)
        entity = world.get_entity("bottle")
        self.assertEqual(entity.pose.x_m, 1.0)
        self.assertEqual(entity.source, "crazyflie-perception")
        self.assertEqual(entity.confidence, 0.92)
        self.assertEqual(entity.observed_at_s, 100.0)
        self.assertEqual(entity.attributes["color"], "blue")
        self.assertEqual(world.snapshot()["entities"]["bottle"]["observed_at_s"], 100.0)

    def test_rejects_untrusted_low_confidence_non_world_and_stale_updates_without_version_change(self) -> None:
        world, ingestor = self._ingestor()
        accepted = ingestor.ingest(
            EntityObservation(
                "target",
                "waypoint",
                Pose3D(1.0, 1.0),
                source="crazyflie-perception",
                observed_at_s=20.0,
                confidence=0.9,
            )
        )
        self.assertTrue(accepted.accepted)
        version = world.version

        untrusted = ingestor.ingest(
            EntityObservation(
                "target",
                "waypoint",
                Pose3D(2.0, 2.0),
                source="llm",
                observed_at_s=21.0,
                confidence=1.0,
            )
        )
        low_confidence = ingestor.ingest(
            EntityObservation(
                "target",
                "waypoint",
                Pose3D(2.0, 2.0),
                source="crazyflie-perception",
                observed_at_s=21.0,
                confidence=0.2,
            )
        )
        non_world = ingestor.ingest(
            EntityObservation(
                "target",
                "waypoint",
                Pose3D(2.0, 2.0, frame="camera"),
                source="crazyflie-perception",
                observed_at_s=21.0,
                confidence=0.9,
            )
        )
        stale = ingestor.ingest(
            EntityObservation(
                "target",
                "waypoint",
                Pose3D(2.0, 2.0),
                source="crazyflie-perception",
                observed_at_s=19.0,
                confidence=0.9,
            )
        )

        self.assertEqual(untrusted.status, IngestStatus.UNTRUSTED_SOURCE)
        self.assertEqual(low_confidence.status, IngestStatus.LOW_CONFIDENCE)
        self.assertEqual(non_world.status, IngestStatus.NON_WORLD_FRAME)
        self.assertEqual(stale.status, IngestStatus.STALE)
        self.assertEqual(world.version, version)
        self.assertEqual(world.get_entity("target").pose.x_m, 1.0)

    def test_rejects_duplicate_timestamp_and_entity_kind_change(self) -> None:
        world, ingestor = self._ingestor()
        ingestor.ingest(
            EntityObservation(
                "item-1",
                "object",
                Pose3D(0.0, 0.0),
                source="crazyflie-perception",
                observed_at_s=5.0,
                confidence=0.9,
            )
        )

        duplicate = ingestor.ingest(
            EntityObservation(
                "item-1",
                "object",
                Pose3D(1.0, 1.0),
                source="crazyflie-perception",
                observed_at_s=5.0,
                confidence=0.95,
            )
        )
        kind_change = ingestor.ingest(
            EntityObservation(
                "item-1",
                "person",
                Pose3D(1.0, 1.0),
                source="crazyflie-perception",
                observed_at_s=6.0,
                confidence=0.95,
            )
        )

        self.assertEqual(duplicate.status, IngestStatus.STALE)
        self.assertEqual(kind_change.status, IngestStatus.KIND_MISMATCH)
        self.assertEqual(world.get_entity("item-1").kind, "object")

    def test_fresh_update_merges_existing_attributes_by_default(self) -> None:
        world, ingestor = self._ingestor()
        first = EntityObservation(
            "bottle",
            "object",
            Pose3D(1.0, 1.0, 0.8),
            source="crazyflie-perception",
            observed_at_s=10.0,
            confidence=0.8,
            attributes={"color": "blue", "container": True},
        )
        second = EntityObservation(
            "bottle",
            "object",
            Pose3D(1.2, 1.1, 0.8),
            source="crazyflie-perception",
            observed_at_s=11.0,
            confidence=0.9,
            attributes={"color": "navy"},
        )

        ingestor.ingest(first)
        result = ingestor.ingest(second)
        self.assertTrue(result.accepted)
        entity = world.get_entity("bottle")
        self.assertEqual(entity.attributes, {"color": "navy", "container": True})
        self.assertEqual(entity.observed_at_s, 11.0)

    def test_perception_correction_between_steps_changes_late_bound_ground_target(self) -> None:
        async def scenario() -> None:
            world = WorldState()
            world.upsert_entity(
                WorldEntity(
                    "inspection_target",
                    "waypoint",
                    Pose3D(1.0, 1.0, 1.0),
                    source="map-seed",
                    confidence=0.8,
                    observed_at_s=10.0,
                )
            )
            ingestor = TrustedWorldIngestor(
                world,
                allowed_sources={"crazyflie-perception"},
                min_confidence=0.7,
            )

            def perception_update(request: SkillRequest) -> None:
                self.assertEqual(request.name, "goto")
                result = ingestor.ingest(
                    EntityObservation(
                        "inspection_target",
                        "waypoint",
                        Pose3D(4.0, -2.0, 1.0),
                        source="crazyflie-perception",
                        observed_at_s=11.0,
                        confidence=0.95,
                    )
                )
                self.assertTrue(result.accepted)

            drone = RecordingRobot(
                "crazyflie",
                {Capability.FLY},
                on_execute=perception_update,
            )
            ground = RecordingRobot("xlerobot", {Capability.NAVIGATE})
            registry = RobotRegistry()
            registry.register(drone)
            registry.register(ground)
            router = RobotToolRouter(
                registry,
                {
                    "crazyflie": ["goto"],
                    "xlerobot": ["navigate_to"],
                },
            )

            plan = Plan(
                "perception-handoff",
                (
                    PlanStep(
                        "crazyflie.goto",
                        entity_pose_refs("inspection_target", "x_m", "y_m", "z_m"),
                    ),
                    PlanStep(
                        "xlerobot.navigate_to",
                        entity_pose_refs("inspection_target", "x_m", "y_m"),
                    ),
                ),
            )
            execution = await PlanExecutor(router, world=world).execute(plan)

            self.assertTrue(execution.ok)
            self.assertEqual(drone.calls[0].params["position"], (1.0, 1.0, 1.0))
            self.assertEqual(ground.calls[0].params["x_m"], 4.0)
            self.assertEqual(ground.calls[0].params["y_m"], -2.0)
            self.assertEqual(world.get_entity("inspection_target").source, "crazyflie-perception")
            self.assertEqual(world.get_entity("inspection_target").observed_at_s, 11.0)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
