from __future__ import annotations

import asyncio
import unittest

import numpy as np

from embodied_agent.embodiments.crazyflie_pybullet import CrazyfliePyBullet


class FakeController:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class FakeVelocityAviary:
    SPEED_LIMIT = 0.05

    def __init__(
        self,
        *,
        gui: bool,
        ctrl_freq_hz: int,
        initial_position: tuple[float, float, float],
    ) -> None:
        del gui, ctrl_freq_hz
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.position = self.initial_position.copy()
        self.closed = False
        self.reset_seeds: list[int | None] = []
        self.ctrl = [FakeController()]

    def _obs(self) -> np.ndarray:
        obs = np.zeros((1, 20), dtype=np.float32)
        obs[0, 0:3] = self.position
        obs[0, 6] = 1.0
        return obs

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        self.position = self.initial_position.copy()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        direction = np.asarray(action[0, 0:3], dtype=float)
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 0:
            direction /= magnitude
            self.position += direction * self.SPEED_LIMIT * float(action[0, 3])
        return self._obs(), -1, False, False, {"answer": 42}

    def close(self) -> None:
        self.closed = True


class CrazyfliePyBulletTests(unittest.TestCase):
    def test_takeoff_goto_land_through_velocity_environment(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                env_factory=FakeVelocityAviary,
                position_tolerance_m=0.025,
            )
            await robot.connect()

            takeoff = await robot.execute("takeoff", altitude_m=0.5)
            self.assertTrue(takeoff.ok)

            goto = await robot.execute("goto", position=(0.4, -0.2, 0.5))
            self.assertTrue(goto.ok)

            observation = await robot.observe()
            self.assertAlmostEqual(observation.state["position_m"][0], 0.4, delta=0.03)
            self.assertAlmostEqual(observation.state["position_m"][1], -0.2, delta=0.03)

            land = await robot.execute("land")
            self.assertTrue(land.ok)
            self.assertAlmostEqual(land.data["final_position_m"][2], 0.05, delta=0.03)

            env = robot._env
            await robot.disconnect()
            self.assertTrue(env.closed)

        asyncio.run(scenario())

    def test_reset_restores_initial_state_seed_and_controller_state(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                env_factory=FakeVelocityAviary,
                seed=7,
                initial_position=(0.0, 0.0, 0.1),
                position_tolerance_m=0.025,
            )
            await robot.connect()
            env = robot._env
            controller = env.ctrl[0]
            self.assertEqual(env.reset_seeds, [7])
            self.assertEqual(controller.reset_count, 1)

            takeoff = await robot.execute("takeoff", altitude_m=0.5)
            self.assertTrue(takeoff.ok)

            reset = await robot.execute("reset", seed=123)
            self.assertTrue(reset.ok)
            self.assertEqual(reset.data["seed"], 123)
            self.assertEqual(robot.seed, 123)
            self.assertEqual(env.reset_seeds[-1], 123)
            self.assertEqual(controller.reset_count, 2)
            self.assertAlmostEqual(reset.data["position_m"][0], 0.0, delta=1e-6)
            self.assertAlmostEqual(reset.data["position_m"][1], 0.0, delta=1e-6)
            self.assertAlmostEqual(reset.data["position_m"][2], 0.1, delta=1e-6)

            repeat = await robot.execute("reset")
            self.assertTrue(repeat.ok)
            self.assertEqual(repeat.data["seed"], 123)
            self.assertEqual(env.reset_seeds[-1], 123)
            self.assertEqual(controller.reset_count, 3)

            await robot.disconnect()

        asyncio.run(scenario())

    def test_controller_reset_failure_propagates(self) -> None:
        class FailingController:
            def reset(self) -> None:
                raise RuntimeError("controller reset failed")

        class FailingResetEnv(FakeVelocityAviary):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self.ctrl = []

        async def scenario() -> None:
            robot = CrazyfliePyBullet(env_factory=FailingResetEnv)
            await robot.connect()
            robot._env.ctrl = [FailingController()]
            try:
                with self.assertRaisesRegex(RuntimeError, "controller reset failed"):
                    await robot.execute("reset")
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_seed_validation_rejects_non_integer_and_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            CrazyfliePyBullet(env_factory=FakeVelocityAviary, seed=True)
        with self.assertRaises(ValueError):
            CrazyfliePyBullet(env_factory=FakeVelocityAviary, seed=-1)

        async def scenario() -> None:
            robot = CrazyfliePyBullet(env_factory=FakeVelocityAviary)
            await robot.connect()
            try:
                with self.assertRaises(ValueError):
                    await robot.execute("reset", seed=1.5)
                with self.assertRaises(ValueError):
                    await robot.execute("reset", seed=2**32)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_default_timeout_expands_for_long_transit(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                env_factory=FakeVelocityAviary,
                position_tolerance_m=0.025,
                ctrl_freq_hz=1,
                slow_radius_m=0.25,
            )
            await robot.connect()
            result = await robot.execute("goto", position=(1.0, 0.0, 0.1))
            self.assertTrue(result.ok)
            self.assertEqual(result.data["timeout_source"], "distance-aware")
            self.assertGreater(result.data["timeout_s"], 10.0)
            self.assertLessEqual(result.data["timeout_s"], 30.0)
            await robot.disconnect()

        asyncio.run(scenario())

    def test_timeout_returns_structured_failure(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                env_factory=FakeVelocityAviary,
                position_tolerance_m=0.001,
                ctrl_freq_hz=1,
            )
            await robot.connect()
            result = await robot.execute(
                "goto",
                position=(10.0, 0.0, 0.1),
                timeout_s=1.0,
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.data["steps"], 1)
            self.assertEqual(result.data["timeout_source"], "explicit")
            await robot.disconnect()

        asyncio.run(scenario())

    def test_explicit_timeout_is_bounded(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(env_factory=FakeVelocityAviary)
            await robot.connect()
            with self.assertRaises(ValueError):
                await robot.execute(
                    "goto",
                    position=(1.0, 0.0, 0.1),
                    timeout_s=31.0,
                )
            await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
