# Roadmap

## M0 — Contract
- [x] Common embodiment interface
- [x] Capability discovery
- [x] Semantic skill request/result model
- [x] Three deterministic simulation stubs
- [x] Smoke tests

## M1 — Real simulators
- [x] Add a real Crazyflie `VelocityAviary` adapter with semantic `takeoff/goto/land` skills
- [ ] Make the real Crazyflie adapter the default sim backend after integration testing
- [x] Add official LeRobot Humanoid MuJoCo controller adapter
- [x] Smoke-test Humanoid adapter against a local official runtime checkout
- [x] Add bounded policy-backed humanoid `walk_velocity` skill
- [ ] Add closed-loop humanoid navigation / `walk_to`
- [x] Add upstream XLeRobot MuJoCo base-navigation adapter
- [x] Smoke-test XLeRobot adapter against a local upstream checkout
- [ ] Attach XLeRobot arm/VLA policy before enabling `MANIPULATE`
- [x] Add policy-backed Microduck MuJoCo adapter with stand/walk/kick/roll skills
- [x] Smoke-test Microduck against pinned native runtime + official Space policies
- [ ] Normalize observations
- [ ] Add simulator reset/seed support
- [ ] Record episode traces

## M2 — Agent tools
- [x] Add allowlisted, schema-validated semantic agent tool router
- [x] Expose the safe semantic tool router through MCP v2 stdio
- [x] Add deterministic planner/executor loop
- [x] Add robot selection based on capabilities and available skills
- [x] Make tool exposure config-driven
- [x] Add shared world entities and task execution state
- [x] Add late-bound world entity references for cross-embodiment targets
- [x] Expose read-only live world state through MCP resources
- [x] Add provider-agnostic iterative high-level MCP agent loop
- [x] Re-read world state before every model decision
- [x] Bound high-level agent runs with explicit maximum-step limits
- [x] Add OpenAI Responses API `AgentModel` provider with Structured Outputs
- [x] Add bounded OpenAI-agent CLI with explicit model selection
- [x] Add trusted perception/localization-to-world ingestion policy boundary
- [x] Reject untrusted, low-confidence, non-world-frame, stale, and kind-changing observations
- [x] Surface entity observation recency through shared world/MCP resources
- [ ] Add calibrated camera/body-frame transform adapters before ingestion
- [ ] Add bounded retries and failure recovery
- [ ] Add at least one additional LLM provider for portability

## M3 — Evals
- [x] Add dependency-free three-robot deterministic baseline eval suite
- [x] Measure robot-selection accuracy and plan exact-match
- [x] Measure multi-step task completion, tool success, and execution coverage
- [x] Verify execution failures are distinguishable from planning failures
- [x] Use shared world entities instead of duplicated waypoint coordinates in baseline evals
- [x] Contract-test a scripted MCP agent coordinating all three embodiments
- [x] Offline-test the OpenAI decision adapter and current async Responses parse surface
- [x] Add provider-agnostic `AgentModel` benchmark over the same three-robot tasks
- [x] Separately score tool selection, grounded arguments, execution, finish behavior, strict task success, and action efficiency
- [x] Add adversarial metric checks for premature finish, wrong coordinates, and unnecessary actions
- [x] Add optional live OpenAI benchmark command against scripted robots
- [x] Test a trusted perception correction between robot steps with late-bound downstream navigation
- [x] Add self-describing live physics result records and manual artifact workflow
- [ ] Record/version live LLM benchmark results for model comparisons
- [ ] Add broader perception-update / stale-plan benchmark cases
- [x] Run the same suite against physics-backed simulator adapters
- [x] Add physics-backed `AgentModel` comparison harness with oracle MCP real-simulator gate
- [x] Add live OpenAI physics comparison command
- [x] Add Microduck learned-policy skill eval cases for walking/kick/roll
- [ ] Add four-embodiment capability-selection benchmark including Microduck
- [ ] Add single-robot skill success rate and latency metrics
- [ ] Add recovery-from-failure tasks
- [ ] Add sim reproducibility / deterministic seed metrics
- [ ] Compare physics-backed LLM task success against deterministic orchestration

## M4 — First physical embodiment
- [ ] Add XLeRobot real adapter
- [ ] Preserve identical high-level skill API
- [ ] Add emergency stop / safety state
- [ ] Compare sim vs real skill outcomes
- [ ] Add physical Microduck backend using Pollen's onboard policy runtime
- [ ] Compare Microduck sim vs real policy outcomes

## M5 — Mixed reality
- [ ] Add physical Crazyflie via Bitcraze `cflib`
- [ ] Run physical XLeRobot + physical Crazyflie + simulated humanoid
- [x] Establish shared world-state abstraction across embodiments
- [ ] Feed real-robot localization/perception into shared world state through the trusted ingestor

## M6 — Humanoid
- [ ] Track LeRobot Humanoid upper-body / whole-body progress
- [ ] Add full simulated humanoid when available
- [ ] Add physical humanoid backend without changing agent-facing API
