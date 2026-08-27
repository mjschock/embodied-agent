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

An entity has a stable ID, semantic kind, world-frame pose, optional attributes, source, and confidence.

```python
from embodied_agent.world import Pose3D, WorldEntity, WorldState

world = WorldState()
world.upsert_entity(
    WorldEntity(
        entity_id="inspection_target",
        kind="waypoint",
        pose=Pose3D(x_m=1.5, y_m=0.5, z_m=1.0),
        source="crazyflie-perception",
        confidence=0.92,
    )
)
```

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
inspection_target = (1.0, 2.0)
    |
Crazyflie/perception correction
    v
inspection_target = (1.3, 2.2)
    |
    +--> later Crazyflie step resolves (1.3, 2.2, z)
    +--> later XLeRobot step resolves (1.3, 2.2)
```

A stale coordinate is not embedded in the plan.

## Coordinate frames

`EntityFieldRef` currently resolves only poses in the `world` frame. A camera-frame detection must be transformed into the shared world frame before navigation can consume it.

This is deliberate. Silent frame mixing is more dangerous than forcing an explicit transform boundary.

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

## Evals

The dependency-free three-robot benchmark now uses shared world entities for its waypoints. Crazyflie and XLeRobot consume references to the same entity rather than receiving duplicate hard-coded XYZ/XY values.

This gives future perception tests a natural next step: have the scouting embodiment update an entity pose, then measure whether downstream embodiments act on the corrected state.
