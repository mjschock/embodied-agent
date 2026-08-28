# Evals

The evaluation stack separates planning/model quality from robot control quality. Scripted embodiments isolate orchestration behavior first; upstream-backed physics adapters then test whether the same semantic contracts survive real simulator dynamics.

## Deterministic capability-planner baseline

Run the structured-task baseline with:

```bash
python -m embodied_agent.evals.multi_robot
```

The default suite contains three waypoint variants of the same multi-robot task:

1. launch Crazyflie;
2. fly Crazyflie to a waypoint;
3. move XLeRobot to that waypoint;
4. have the humanoid stand ready;
5. land Crazyflie.

Each case retains both a natural-language instruction and a structured `Task`. The deterministic `CapabilityPlanner` consumes the structured task and should score 1.0 on every baseline metric.

### Deterministic metrics

`robot_selection_accuracy` measures per-step `<robot>.<skill>` selection. `plan_exact_match_rate` measures complete tool-sequence equality. `task_completion_rate` measures plans that execute fully. `tool_call_success_rate` measures successful calls among attempted calls. `executed_step_coverage` shows how much of the expected plan actually ran before an early stop.

The tests inject an XLeRobot navigation failure and verify that planning metrics stay perfect while execution metrics fall. This makes controller failure distinguishable from planning failure.

## Physics-backed deterministic baseline

The same three structured cases can also run against all three upstream-backed simulator adapters in one process:

```bash
export XLEROBOT_UPSTREAM_ROOT="/path/to/XLeRobot"
export LEROBOT_HUMANOID_RUNTIME_ROOT="/path/to/lerobot-humanoid-runtime"
python -m embodied_agent.evals.physics_multi_robot
```

This swaps only the embodiment implementations. The `CapabilityPlanner`, `PlanExecutor`, `RobotToolRouter`, `WorldState`, target entities, expected tool sequence, and metrics are the same as the scripted baseline.

The physics environment combines:

- Crazyflie through real `gym-pybullet-drones` `VelocityAviary` / PyBullet;
- XLeRobot through the upstream MuJoCo scene and the adapter's closed-loop base controller;
- LeRobot Humanoid through the official `SimBipedalRobotController`, fixed-base and policy-free because the benchmark only requires `stand`.

CI pins the upstream simulator/model revisions and requires the deterministic physics baseline to retain 1.0 robot-selection accuracy, exact-plan rate, task-completion rate, tool-call success rate, and executed-step coverage across all three waypoint variants. No physical hardware is used.

## AgentModel benchmark

The same natural-language cases can evaluate any provider implementing the `AgentModel` interface:

```bash
python -m embodied_agent.evals.agent_model
```

That command uses an `ExpectedActionModel` oracle only to validate the benchmark itself; it should score 1.0 on every metric. Actual LLM providers use the same evaluator but must infer the robot/tool sequence and grounded coordinates from the instruction, MCP tool schemas, and live world state.

The model is run through the real `MCPAgentRunner` and in-memory MCP v2 server. The default model benchmark keeps robot backends scripted, so high-level decisions can be measured without physics variance.

### AgentModel metrics

`tool_selection_accuracy` is the position-wise accuracy of semantic MCP tool choices against the expected actions.

`argument_accuracy` is the fraction of expected actions where the selected tool is correct and all task-relevant expected arguments match. Safe optional tool arguments/defaults are allowed. Numeric values use a small configurable tolerance.

`sequence_exact_match_rate` is the fraction of cases whose complete action-tool sequence exactly matches the reference sequence.

`arguments_exact_match_rate` is the fraction of cases where all expected actions have correct task-relevant arguments and there are no extra actions.

`tool_execution_success_rate` measures whether the selected semantic actions mechanically executed in the current robot environment. This can remain high even when the model navigates to the wrong in-bounds coordinate—or chooses a different compatible embodiment—which is why selection and argument accuracy are scored separately.

`runner_finish_rate` measures whether the model eventually returned a finish decision. This is not a task-success metric: a model that immediately says "done" can score 1.0 here and still score 0.0 strict task success.

`runner_ok_rate` measures runs that the bounded runtime considers mechanically successful, without judging whether the requested task semantics were satisfied.

`strict_task_success_rate` is the primary high-level benchmark metric. A case succeeds only when the agent finishes, uses the exact expected robot sequence, supplies correct grounded task arguments, and every expected tool executes successfully.

`mean_action_efficiency` penalizes unnecessary extra semantic robot actions. The score is capped at 1.0 and compares expected action count with actual action count.

## Sanity checks

The test suite includes adversarial models to verify that the metrics do not collapse into one number:

- an immediate-finish model gets runner completion but zero strict task success;
- a model choosing the right robots with one wrong target coordinate keeps perfect tool selection but loses argument accuracy and strict success;
- a model that performs the entire correct sequence plus an unnecessary extra action keeps tool execution success but loses exact-match/strict success and action efficiency.

## Four-embodiment capability selection

`four_embodiment.py` extends the scripted AgentModel benchmark to every current embodiment:

```bash
python -m embodied_agent.evals.four_embodiment
```

It covers seven cases:

1. Crazyflie takeoff → waypoint → land;
2. XLeRobot navigation;
3. LeRobot Humanoid stand;
4. LeRobot Humanoid bounded walking;
5. Microduck right-foot kick;
6. Microduck learned roll;
7. one mission coordinating all four embodiments in sequence.

This benchmark deliberately includes an ambiguity that does not exist in the original three-robot suite: **Humanoid and Microduck both advertise `STAND` and `WALK`**. A model can therefore call `microduck.stand` for a request that explicitly asks for the humanoid, receive a successful robot result, and still have chosen the wrong embodiment.

The `SharedCapabilityConfusionModel` sanity check does exactly that. It routes Humanoid stand/walk requests to the corresponding valid Microduck tools. `tool_execution_success_rate` remains 1.0, while tool-selection accuracy, exact-match rate, and strict task success fall. This proves the benchmark scores embodiment identity independently from mechanical tool validity.

The benchmark remains dependency-free and uses the same `evaluate_agent_model` scorer and MCP runtime as the existing suites. Microduck's learned-policy physics is validated separately by the pinned `microduck-physics` gate; this layer isolates high-level selection before introducing four simultaneous physics backends.

### Live four-embodiment OpenAI run

The same seven cases can be sent to the existing OpenAI `AgentModel` provider without changing the benchmark definition:

```bash
pip install -e ".[openai-agent]"
export OPENAI_API_KEY="..."

embodied-agent-eval-openai-four \
  --model "YOUR_MODEL_ID" \
  --max-steps 10 \
  --output eval_results/four-embodiment.json
```

The command emits the standard schema-versioned result-record envelope with benchmark identity `four-embodiment-selection`, provider/model identity, maximum step budget, timestamp, repository revision where detectable, Python version, and the complete seven-case metric payload.

Its exit code is 0 only when `strict_task_success_rate == 1.0`; a lower score is a legitimate model-evaluation result, not evidence that the scripted robot stack or CI is broken. CI never calls the live provider: it injects `ExpectedActionModel` into the same provider-agnostic function to validate evaluator and result-record wiring without requiring an API key or incurring usage.

This command makes the benchmark runnable against a live model. The roadmap keeps the **first recorded live four-embodiment result** as a separate milestone so availability of the harness is not confused with an unrun model-quality claim.

## Physics-backed AgentModel comparison

`physics_agent_model.py` runs the deterministic planner and an `AgentModel` on fresh, equivalently configured physics stacks and reports both result sets plus direct score gaps. Within each path the same connected XLeRobot, Crazyflie, and humanoid instances are reused across workbench A → B → C, preserving sequential embodiment state.

The CI integration gate uses `ExpectedActionModel`, an oracle rather than an LLM, to prove that the complete high-level path can achieve the deterministic reference through:

```text
AgentModel → MCPAgentRunner → MCP → RobotToolRouter → semantic skills → real simulators
```

Both deterministic and oracle AgentModel physics paths are required to score 1.0 in CI. This validates the transport/execution harness without pretending to measure live-model quality.

## Optional live OpenAI benchmarks

After installing the optional OpenAI provider and configuring an API key/model, the historical scripted three-robot benchmark can be run against a live OpenAI model:

```bash
pip install -e ".[openai-agent]"
export OPENAI_API_KEY="..."
export EMBODIED_AGENT_OPENAI_MODEL="YOUR_MODEL_ID"
embodied-agent-eval-openai
```

For the real-simulator comparison, install the simulator extras, provide the upstream checkouts, and run:

```bash
pip install -e ".[xlerobot-sim,humanoid-sim,openai-agent]"
export OPENAI_API_KEY="..."
export XLEROBOT_UPSTREAM_ROOT="/path/to/XLeRobot"
export LEROBOT_HUMANOID_RUNTIME_ROOT="/path/to/lerobot-humanoid-runtime"

embodied-agent-eval-openai-physics \
  --model "YOUR_MODEL_ID" \
  --output benchmark-results/openai-physics.json
```

The physics command first runs a fresh deterministic reference and then a fresh live-model stack. It emits a schema-versioned JSON record containing model identity, timestamp, step budget, repository/upstream revisions where detectable, complete deterministic/model metrics, and direct score deltas. The command's process exit code is 0 only when both the deterministic reference and live model achieve perfect task completion/strict success.

Live model calls may incur API usage/cost. No physical hardware is used.

## Manual GitHub Actions live-physics run

The `openai-physics-benchmark` workflow is intentionally `workflow_dispatch` only. It requires a repository secret named `OPENAI_API_KEY`, plus an explicit model ID and maximum-step budget at dispatch time. The workflow:

1. checks out the exact pinned XLeRobot and LeRobot Humanoid sources used by the physics CI stack;
2. installs the pinned Crazyflie simulator and OpenAI agent runtime;
3. runs the deterministic-vs-live-model physics comparison;
4. writes and uploads the self-describing JSON result record;
5. writes headline metrics and model-vs-deterministic gaps to the GitHub Actions job summary.

A live model scoring below 1.0 is treated as an evaluation result, not an infrastructure failure: if a valid result record is produced, the workflow still uploads it. Execution/authentication failures that prevent a record from being created do fail the workflow.

Selected records can be checked into `eval_results/` for long-term model comparisons. See `eval_results/README.md`.

## Next eval layers

- run and version the first live four-embodiment result and the first live physics result, then compare models over time;
- add broader perception-update and stale-plan tests where the target changes after an early action;
- add a combined four-embodiment real-simulator gate;
- add single-robot skill reliability and latency metrics;
- add bounded recovery tasks where the first tool call fails;
- add simulator reproducibility metrics;
- preserve the same semantic benchmark as simulated embodiments are swapped for real hardware.
