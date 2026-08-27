from __future__ import annotations

import asyncio
import unittest

import numpy as np

from embodied_agent.embodiments.crazyflie_pybullet import CrazyfliePyBullet


class FakeVelocityAviary:
    def __init__(
        self,
        *,
        gui: bool,
        ctrl_freq_hz: int,
        initial_position: tuple[float, float, float],
    ) -> None:
        del gui, ctrl_freq_hz
        self.position = np.asarray(initial_position, dtype=float)
        self.closed = False

    def _obs(self) -> np.ndarray:
        obs = np.zeros((1, 20), dtype=np.float32)
        obs[0, 0:3] = self.position
        obs[0, 6] = 1.0
        return obs

    def reset(self, *, seed: int | None = None):
        del seed
        return self._obs(), {}

    def step(self, action: np.ndarray):
        direction = np.asarray(action[0, 0:3], dtype=float)
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 0:
            direction /= magnitude
            self.position += direction * 0.05 * float(action[0, 3])
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
            await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
