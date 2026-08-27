# Embodied Agent

A simulation-first, multi-embodiment robotics platform for controlling:

- **XLeRobot** — mobile manipulation
- **Crazyflie** — aerial perception and navigation
- **LeRobot Humanoid** — legged / humanoid mobility

The design goal is **one high-level agent with stable semantic skills**, while each robot keeps its own low-level controller, simulator, and eventual physical hardware backend.

## Core principle

The high-level agent should never emit raw motor commands.

Instead, it calls semantic skills such as:

```python
await robots["crazyflie"].execute("takeoff", altitude_m=1.0)
await robots["xlerobot"].execute("navigate_to", target="workbench")
await robots["humanoid"].execute("stand")
```

Each embodiment maps those skills onto its own backend:

```text
High-level agent
      |
      v
Semantic skill API
      |
      +----------------+----------------+----------------+
      |                |                |
      v                v                v
   XLeRobot        Crazyflie        Humanoid
      |                |                |
   sim/real          sim/real          sim/real
```

## Why this abstraction

This lets the project evolve through four stages without rewriting the agent:

1. XLeRobot sim + Crazyflie sim + Humanoid sim
2. XLeRobot real + Crazyflie sim + Humanoid sim
3. XLeRobot real + Crazyflie real + Humanoid sim
4. XLeRobot real + Crazyflie real + Humanoid real

LeRobot fits primarily at the **robot-learning/data/policy layer**, not as a universal physics simulator.

## Current status

This initial repository contains:

- a common asynchronous `Embodiment` interface;
- capability discovery;
- semantic skill requests/results;
- a registry for multiple robots;
- deterministic simulation stubs for all three target embodiments;
- an end-to-end demo where one coordinator invokes all three;
- smoke tests;
- config placeholders for future sim and real backends.

The next milestone is replacing one stub at a time with a real simulator adapter.

## Quick start

Requires Python 3.11+.

```bash
python -m embodied_agent.demo
python -m unittest discover -s tests -v
```

## Planned simulator adapters

### XLeRobot
Target the current XLeRobot / LeRobot-compatible simulation path, likely via ManiSkill or the maintained XLeRobot simulation tooling.

### Crazyflie
Start with `gym-pybullet-drones` / Crazyflie-compatible Gymnasium simulation. Later add a physical adapter backed by Bitcraze `cflib`.

### LeRobot Humanoid
Use the official LeRobot Humanoid MuJoCo runtime first. Later swap to the physical runtime while preserving the same semantic interface.

## Repository layout

```text
embodied-agent/
├── embodied_agent/
│   ├── core/
│   │   ├── embodiment.py
│   │   ├── models.py
│   │   └── registry.py
│   ├── embodiments/
│   │   ├── xlerobot.py
│   │   ├── crazyflie.py
│   │   └── humanoid.py
│   └── demo.py
├── configs/
│   ├── all_sim.json
│   └── mixed_reality.example.json
├── docs/
│   ├── architecture.md
│   └── roadmap.md
├── tests/
│   └── test_smoke.py
└── pyproject.toml
```

## Milestone 1 definition of done

A single Python process can:

- connect to all three simulated embodiments;
- discover their capabilities;
- issue at least one semantic skill to each;
- receive structured observations/results;
- run the same high-level coordinator regardless of backend selection;
- pass deterministic smoke tests.

That is the seam we want to prove before introducing an LLM/MCP planner.
