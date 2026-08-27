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
- [ ] Smoke-test Humanoid adapter against a local official runtime checkout
- [x] Add bounded policy-backed humanoid `walk_velocity` skill
- [ ] Add closed-loop humanoid navigation / `walk_to`
- [x] Add upstream XLeRobot MuJoCo base-navigation adapter
- [ ] Smoke-test XLeRobot adapter against a local upstream checkout
- [ ] Attach XLeRobot arm/VLA policy before enabling `MANIPULATE`
- [ ] Normalize observations
- [ ] Add simulator reset/seed support
- [ ] Record episode traces

## M2 — Agent tools
- [x] Add allowlisted, schema-validated semantic agent tool router
- [ ] Expose the safe semantic tool router through MCP
- [x] Add deterministic planner/executor loop
- [x] Add robot selection based on capabilities and available skills
- [x] Make tool exposure config-driven
- [ ] Add task state and world entities
- [ ] Add bounded retries and failure recovery

## M3 — Evals
- [ ] Single-robot skill success rate
- [ ] Robot-selection accuracy
- [ ] Multi-step task completion rate
- [ ] Recovery-from-failure rate
- [ ] Sim reproducibility / deterministic seeds

## M4 — First physical embodiment
- [ ] Add XLeRobot real adapter
- [ ] Preserve identical high-level skill API
- [ ] Add emergency stop / safety state
- [ ] Compare sim vs real skill outcomes

## M5 — Mixed reality
- [ ] Add physical Crazyflie via Bitcraze `cflib`
- [ ] Run physical XLeRobot + physical Crazyflie + simulated humanoid
- [ ] Share world state across embodiments

## M6 — Humanoid
- [ ] Track LeRobot Humanoid upper-body / whole-body progress
- [ ] Add full simulated humanoid when available
- [ ] Add physical humanoid backend without changing agent-facing API
