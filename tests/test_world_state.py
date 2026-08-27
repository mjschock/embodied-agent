from __future__ import annotations

import asyncio
import unittest

from embodied_agent.agent import CapabilityPlanner, PlanExecutor, RobotToolRouter, Task, TaskStep
from embodied_agent.core import (
    Capability,
    Embodiment,
    Observation,
    RobotRegistry,
    SkillRequest,
    SkillResult,
)
from embodied_agent.world import (
    EntityFieldRef,
    Pose3D,
    TaskStatus,
    WorldEntity,
    WorldState,
    entity_pose_refs,
)


class FakeRobot(Embodiment):
    def __init__(self, name: str, capabilities: set[Capability]) -> None:
        self.name = name
        self.backend = "fake"
        self._capabilities = frozenset(capabilities)
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
        return SkillResult(
            embodiment=self.name,
            skill=request.name,
            ok=True,
            detail="ok",
            data={"params": dict(request.params)},
        )


class WorldStateTests(unittest.TestCase):
    def test_entity_references_are_late_bound(self) -> None:
        world = WorldState()
        world.upsert_entity(
            WorldEntity("target", "waypoint", Pose3D(1.0, 2.0, 0.5), source="fixture")
        )
        refs = entity_pose_refs("target", "x_m", "y_m", "z_m")
        self.assertEqual(world.resolve_arguments(refs), {"x_m": 1.0, "y_m": 2.0, "z_m": 0.5})

        world.upsert_entity(
            WorldEntity("target", "waypoint", Pose3D(3.0, -4.0, 1.25), source="perception")
        )
        self.assertEqual(
            world.resolve_arguments(refs),
            {"x_m": 3.0, "y_m": -4.0, "z_m": 1.25},
        )
        self.assertEqual(world.get_entity("target").source, "perception")

    def test_non_world_frame_cannot_feed_world_navigation(self) -> None:
        world = WorldState()
        world.upsert_entity(
            WorldEntity("camera-target", "object", Pose3D(0.2, 0.1, frame="camera"))
        )
        with self.assertRaisesRegex(ValueError, "world-frame value required"):
            world.resolve_value(EntityFieldRef("camera-target", "x_m"))

    def test_executor_resolves_one_entity_for_multiple_embodiments_at_call_time(self) -> None:
        registry = RobotRegistry()
        drone = FakeRobot("crazyflie", {Capability.OBSERVE, Capability.FLY})
        ground = FakeRobot("xlerobot", {Capability.OBSERVE, Capability.NAVIGATE})
        registry.register(drone)
        registry.register(ground)
        router = RobotToolRouter(
            registry,
            {
                "crazyflie": ["goto"],
                "xlerobot": ["navigate_to"],
            },
        )
        world = WorldState()
        world.upsert_entity(WorldEntity("inspection_target", "waypoint", Pose3D(1.0, 2.0, 1.0)))

        task = Task(
            "shared-target",
            (
                TaskStep(
                    "goto",
                    Capability.FLY,
                    entity_pose_refs("inspection_target", "x_m", "y_m", "z_m"),
                ),
                TaskStep(
                    "navigate_to",
                    Capability.NAVIGATE,
                    entity_pose_refs("inspection_target", "x_m", "y_m"),
                ),
            ),
        )
        plan = CapabilityPlanner(router).plan(task)

        # Simulate perception/localization correcting the target after planning.
        world.upsert_entity(
            WorldEntity(
                "inspection_target",
                "waypoint",
                Pose3D(3.5, -1.25, 1.4),
                source="crazyflie-perception",
                confidence=0.9,
            )
        )

        result = asyncio.run(PlanExecutor(router, world=world).execute(plan))
        self.assertTrue(result.ok)
        self.assertEqual(drone.calls[0].params["position"], (3.5, -1.25, 1.4))
        self.assertEqual(ground.calls[0].params["x_m"], 3.5)
        self.assertEqual(ground.calls[0].params["y_m"], -1.25)

        task_state = world.get_task("shared-target")
        self.assertEqual(task_state.status, TaskStatus.SUCCEEDED)
        self.assertEqual(len(task_state.history), 2)
        self.assertEqual(task_state.history[0].arguments["x_m"], 3.5)
        snapshot = world.snapshot()
        self.assertIn("crazyflie", snapshot["robots"])
        self.assertIn("xlerobot", snapshot["robots"])

    def test_missing_world_entity_fails_before_robot_call_and_marks_task_failed(self) -> None:
        registry = RobotRegistry()
        ground = FakeRobot("xlerobot", {Capability.OBSERVE, Capability.NAVIGATE})
        registry.register(ground)
        router = RobotToolRouter(registry, {"xlerobot": ["navigate_to"]})
        world = WorldState()
        task = Task(
            "missing-target",
            (
                TaskStep(
                    "navigate_to",
                    Capability.NAVIGATE,
                    entity_pose_refs("does-not-exist", "x_m", "y_m"),
                ),
            ),
        )
        plan = CapabilityPlanner(router).plan(task)
        result = asyncio.run(PlanExecutor(router, world=world).execute(plan))

        self.assertFalse(result.ok)
        self.assertEqual(ground.calls, [])
        self.assertIn("world resolution failed", result.calls[0].detail)
        self.assertEqual(world.get_task("missing-target").status, TaskStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
