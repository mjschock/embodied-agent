# Four-embodiment physics gate

The four-embodiment physics gate proves that every current upstream-backed simulator adapter can coexist in one Python process and execute one semantic AgentModel mission through the shared MCP stack.

## Connected embodiments

- Crazyflie: `gym-pybullet-drones` `VelocityAviary` / PyBullet
- XLeRobot: upstream MuJoCo base-navigation model
- LeRobot Humanoid: official MuJoCo runtime, fixed-base for the `stand` action used by this mission
- Microduck: upstream MuJoCo runtime plus the pinned official ONNX standing/walking/roulade policies

The workflow pins every upstream revision and the Microduck Space policy snapshot rather than following moving branches.

## Mission

The gate reuses the existing `four-all-embodiments-mission` benchmark case instead of defining a physics-only reference sequence. The expected semantic actions are:

1. `crazyflie.takeoff`
2. `crazyflie.goto`
3. `xlerobot.navigate_to`
4. `humanoid.stand`
5. `microduck.roll`
6. `crazyflie.land`

The path under test is:

```text
ExpectedActionModel
  -> MCPAgentRunner
  -> MCP
  -> RobotToolRouter
  -> semantic embodiment adapters
  -> PyBullet / MuJoCo / ONNX runtimes
```

The integration test requires perfect tool-selection accuracy, argument accuracy, sequence exact match, argument exact match, tool-execution success, runner finish/ok status, strict task success, and action efficiency. It also verifies that all four robot prefixes occur in the executed sequence.

## CI result

The first dedicated `four-embodiment-physics` workflow passed on PR #25. On the same tested head, the standard Python test matrix, individual Crazyflie/XLeRobot/Humanoid/Microduck physics gates, and the existing three-robot physics-coordination workflow also passed.

This establishes simulator coexistence and high-level orchestration correctness. It does **not** measure live LLM quality because CI uses the deterministic `ExpectedActionModel` oracle and makes no external model API call.

## Why Humanoid uses `stand`

The combined workflow intentionally does not provision a learned humanoid locomotion policy. The mission therefore uses the supported `humanoid.stand` semantic skill rather than overstating the capabilities available in this particular combined runtime. Humanoid learned walking remains covered by its dedicated adapter/policy tests.

## Next layer

The next model-quality step is to record/version an actual live four-embodiment AgentModel result. The live scripted command already exists; a future live four-physics comparison can reuse this combined stack once we decide to incur provider usage and record the result as benchmark evidence.
