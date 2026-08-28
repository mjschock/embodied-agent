from __future__ import annotations

import asyncio
import unittest

from embodied_agent.embodiments import CrazyfliePyBullet


class CrazyfliePhysicsIntegrationTests(unittest.TestCase):
    def test_takeoff_translate_observe_and_land_in_real_pybullet(self) -> None:
        async def scenario() -> None:
            robot = CrazyfliePyBullet(
                gui=False,
                seed=0,
                ctrl_freq_hz=60,
                initial_position=(0.0, 0.0, 0.1),
                position_tolerance_m=0.06,
                slow_radius_m=0.25,
            )
            await robot.connect()
            try:
                initial = await robot.observe()
                self.assertEqual(initial.state["backend"], "gym-pybullet-drones")
                self.assertEqual(len(initial.state["position_m"]), 3)
                self.assertEqual(len(initial.state["motor_rpm"]), 4)

                takeoff = await robot.execute(
                    "takeoff",
                    altitude_m=0.4,
                    timeout_s=6.0,
                )
                self.assertTrue(takeoff.ok, takeoff.detail)
                self.assertLessEqual(takeoff.data["position_error_m"], 0.06)

                move = await robot.execute(
                    "goto",
                    position=(0.15, 0.10, 0.4),
                    timeout_s=6.0,
                )
                self.assertTrue(move.ok, move.detail)
                self.assertLessEqual(move.data["position_error_m"], 0.06)

                moved = await robot.observe()
                self.assertAlmostEqual(moved.state["position_m"][0], 0.15, delta=0.07)
                self.assertAlmostEqual(moved.state["position_m"][1], 0.10, delta=0.07)
                self.assertAlmostEqual(moved.state["position_m"][2], 0.4, delta=0.07)

                land = await robot.execute(
                    "land",
                    height_m=0.08,
                    timeout_s=7.0,
                )
                self.assertTrue(land.ok, land.detail)
                self.assertLessEqual(land.data["position_error_m"], 0.06)

                landed = await robot.observe()
                self.assertAlmostEqual(landed.state["position_m"][2], 0.08, delta=0.07)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
