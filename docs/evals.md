# Evals

The first benchmark is intentionally dependency-free. It evaluates the orchestration layer before physics or hardware noise is introduced.

Run it with:

```bash
python -m embodied_agent.evals.multi_robot
```

The default suite contains three waypoint variants of the same multi-robot task:

1. launch Crazyflie;
2. fly Crazyflie to a waypoint;
3. move XLeRobot to that waypoint;
4. have the humanoid stand ready;
5. land Crazyflie.

Each case retains both a natural-language instruction and a structured `Task`. The deterministic `CapabilityPlanner` consumes the structured task today. A future LLM planner can consume the natural-language instruction while being scored against the same expected tool sequence.

## Metrics

### `robot_selection_accuracy`

Per-step accuracy of the selected `<robot>.<skill>` tool against the expected tool. This isolates embodiment/tool selection from downstream controller success.

### `plan_exact_match_rate`

Fraction of cases where the complete planned tool sequence exactly matches the expected sequence.

### `task_completion_rate`

Fraction of cases whose plan executes fully with every tool call succeeding.

### `tool_call_success_rate`

Success fraction among tool calls that were actually attempted.

### `executed_step_coverage`

Attempted tool calls divided by expected steps. This exposes early-stop behavior: a failure halfway through a plan should not look like a full five-step execution.

## Why scripted robots first

The scripted baseline should score 1.0 on every metric. That gives us a fixed control condition. The tests also inject an XLeRobot navigation failure and verify that planning metrics stay perfect while execution metrics fall. This separation is important once we add LLM planning and physics-backed simulation: we want to know whether a regression came from planning, routing, or control.

## Next eval layers

- run the same cases against real simulator adapters;
- add single-robot skill reliability and latency metrics;
- add recovery tasks where the first tool call fails;
- add world-state-dependent tasks rather than pre-specified coordinates;
- evaluate an LLM planner against the deterministic capability-planner baseline;
- preserve the same suite as simulated embodiments are swapped for real hardware.
