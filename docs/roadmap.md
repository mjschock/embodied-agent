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
- [ ] Replace Humanoid stub with official LeRobot Humanoid MuJoCo runtime
- [ ] Replace XLeRobot stub with current maintained simulation path
- [ ] Normalize observations
- [ ] Add simulator reset/seed support
- [ ] Record episode traces

## M2 — Agent tools
- [ ] Expose safe semantic skills as MCP tools
- [ ] Add planner/executor loop
- [ ] Add robot selection based on capabilities
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
