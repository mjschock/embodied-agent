# Shared world and task state

The high-level agent needs a stable place to represent things that outlive one robot call: named objects, waypoints, corrected poses, task progress, and the latest result from each embodiment.

`WorldState` is that layer.

## Core model

```text
WorldState
├── entities
│   ├── inspection_target
│   ├── workbench
│   └── user
├── latest robot results
│   ├── crazyflie
│   ├── xlerobot
│   └── humanoid
└── task runs
    └── scout-and-approach
```

An entity has a stable ID, semantic kind, pose, optional attributes, source, confidence, and optional observation timestamp.

```python
from embodied_agent.world import Pose3D, WorldEntity, WorldState

world = WorldState()
world.upsert_entity(
    WorldEntity(
        entity_id="inspection_target",
        kind="waypoint",
        pose=Pose3D(x_m=1.5, y_m=0.5, z_m=1.0),
        source="map-seed",
        confidence=0.80,
        observed_at_s=100.0,
    )
)
```

`observed_at_s` uses a shared time base for producers that can update the same entity. It is surfaced in `WorldEntity.to_dict()`, `WorldState.snapshot()`, and therefore MCP world resources.

## Trusted perception/localization ingestion

Robot sensors should not write arbitrary world entities directly from an LLM decision. `TrustedWorldIngestor` provides a separate policy boundary for grounded perception and localization pipelines.

```python
from embodied_agent.world import (
    EntityObservation,
    Pose3D,
    TrustedWorldIngestor,
)

ingestor = TrustedWorldIngestor(
    world,
    allowed_sources={
        "crazyflie-perception",
        "xlerobot-localization",
    },
    min_confidence=0.70,
)

result = ingestor.ingest(
    EntityObservation(
        entity_id="inspection_target",
        kind="waypoint",
        pose=Pose3D(1.8, 0.6, 1.0),
        source="crazyflie-perception",
        observed_at_s=101.0,
        confidence=0.94,
    )
)

assert result.accepted
```

The default ingestion policy rejects an observation before it can mutate the world when:

- the source is not allowlisted;
- confidence is below the configured threshold;
- the pose is not in the `world` frame;
- its observation timestamp is stale or duplicates the currently accepted timestamp;
- the same stable entity ID suddenly changes semantic kind.

Rejected observations do not increment the world version. Accepted updates can merge existing entity attributes so perception corrections do not accidentally discard useful semantic metadata.

This class is a **trusted infrastructure API**, not a generic MCP tool. The MCP host can read grounded world resources, but there is still no `world.upsert` tool for an LLM to call.

## Coordinate frames

`EntityFieldRef` and the default `TrustedWorldIngestor` both require world-frame coordinates for navigation-consumable state.

A camera-frame or robot-body-frame detection must be transformed into the shared world frame by trusted localization/perception code before ingestion:

```text
camera detection
      |
      v
calibration + localization transform
      |
      v
world-frame EntityObservation
      |
      v
TrustedWorldIngestor
      |
      v
WorldState
```

This is deliberate. Silent frame mixing is more dangerous than forcing an explicit transform boundary.

## Late-bound references

Plans should not copy coordinates everywhere. A task can instead reference fields of a named entity:

```python
from embodied_agent.world import entity_pose_refs

TaskStep(
    "goto",
    Capability.FLY,
    entity_pose_refs("inspection_target", "x_m", "y_m", "z_m"),
)

TaskStep(
    "navigate_to",
    Capability.NAVIGATE,
    entity_pose_refs("inspection_target", "x_m", "y_m"),
)
```

The planner keeps those references symbolic. `PlanExecutor` resolves them immediately before each tool call.

That matters when perception or localization changes after planning:

```text
plan created
    |
    v
inspection_target = (1.0, 1.0)
    |
Crazyflie flies toward seeded target
    |
trusted perception correction
    v
inspection_target = (4.0, -2.0)
    |
    +--> XLeRobot step resolves (4.0, -2.0)
```

A stale coordinate is not embedded in the plan. The test suite now exercises this exact handoff: Crazyflie receives the original target, its perception callback ingests a newer correction, and XLeRobot's already-created downstream step resolves the corrected pose.

## Task state

When a `PlanExecutor` is given a `WorldState`, it records:

- task status (`running`, `succeeded`, or `failed`);
- current step;
- resolved arguments used for each step;
- tool success/failure and detail;
- latest result from every robot that was actually called.

```python
executor = PlanExecutor(router, world=world)
execution = await executor.execute(plan)

print(world.get_task(plan.task_name).status)
print(world.snapshot())
```

World-resolution failures are recorded as task failures before a robot is called. For example, a plan referring to an unknown entity cannot accidentally degrade into a default coordinate.

## MCP read boundary

MCP exposes the current world as read-only resources:

```text
world://snapshot
world://entities/<id>
```

Accepted perception/localization updates are immediately visible on the next resource read, including source, confidence, and observation timestamp. Robot actions remain MCP tools; grounded world mutation remains in trusted ingestion code.
