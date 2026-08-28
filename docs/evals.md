# Evals

The evaluation stack separates planning/model quality from robot control quality. The first benchmarks use scripted embodiments so orchestration can be measured before physics or hardware noise is introduced.

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

## AgentModel benchmark

The same natural-language cases can now evaluate any provider implementing the `AgentModel` interface:

```bash
python -m embodied_agent.evals.agent_model
```

That command uses an `ExpectedActionModel` oracle only to validate the benchmark itself; it should score 1.0 on every metric. Actual LLM providers use the same evaluator but must infer the robot/tool sequence and grounded coordinates from the instruction, MCP tool schemas, and live world state.

The model is run through the real `MCPAgentRunner` and in-memory MCP v2 server. The robot backends remain scripted, so a model benchmark measures high-level decisions without physics variance.

### AgentModel metrics

`tool_selection_accuracy` is the position-wise accuracy of semantic MCP tool choices against the five expected actions.

`argument_accuracy` is the fraction of expected actions where the selected tool is correct and all task-relevant expected arguments match. Safe optional tool arguments/defaults are allowed. Numeric values use a small configurable tolerance.

`sequence_exact_match_rate` is the fraction of cases whose complete action-tool sequence exactly matches the reference sequence.

`arguments_exact_match_rate` is the fraction of cases where all expected actions have correct task-relevant arguments and there are no extra actions.

`tool_execution_success_rate` measures whether the selected semantic actions mechanically executed in the scripted robot environment. This can remain high even when the model navigates to the wrong in-bounds coordinate, which is why argument accuracy is scored separately.

`runner_finish_rate` measures whether the model eventually returned a finish decision. This is not a task-success metric: a model that immediately says "done" can score 1.0 here and still score 0.0 strict task success.

`runner_ok_rate` measures runs that the bounded runtime considers mechanically successful, without judging whether the requested task semantics were satisfied.

`strict_task_success_rate` is the primary high-level benchmark metric. A case succeeds only when the agent finishes, uses the exact expected robot sequence, supplies correct grounded task arguments, and every expected tool executes successfully.

`mean_action_efficiency` penalizes unnecessary extra semantic robot actions. The score is capped at 1.0 and compares expected action count with actual action count.

## Sanity checks

The test suite includes adversarial models to verify that the metrics do not collapse into one number:

- an immediate-finish model gets runner completion but zero strict task success;
- a model choosing the right robots with one wrong target coordinate keeps perfect tool selection but loses argument accuracy and strict success;
- a model that performs the entire correct sequence plus an unnecessary extra action keeps tool execution success but loses exact-match/strict success and action efficiency.

## Optional live OpenAI benchmark

After installing the optional OpenAI provider and configuring an API key/model, the same scripted benchmark can be run against a live OpenAI model:

```bash
pip install -e ".[openai-agent]"
export OPENAI_API_KEY="..."
export EMBODIED_AGENT_OPENAI_MODEL="YOUR_MODEL_ID"
embodied-agent-eval-openai
```

This command sends model requests to the OpenAI API and may incur API usage/cost. It **does not** connect to physical robots or physics simulators; only the high-level model decisions are live. Each case remains bounded by `--max-steps` (default 8).

The command exits successfully only when `strict_task_success_rate == 1.0`, making it suitable for an explicit manual model-quality gate. It is not part of CI and CI never calls the OpenAI API.

## Next eval layers

- record live LLM results over time and compare models against the deterministic/oracle references;
- add perception-update and stale-plan tests where the target changes after an early action;
- run the same cases against real simulator adapters;
- add single-robot skill reliability and latency metrics;
- add bounded recovery tasks where the first tool call fails;
- add simulator reproducibility metrics;
- preserve the same semantic benchmark as simulated embodiments are swapped for real hardware.
