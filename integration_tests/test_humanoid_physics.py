from __future__ import annotations

import asyncio
import math
import os
import unittest

from embodied_agent.core import Capability
from embodied_agent.embodiments import HumanoidMuJoCo


class HumanoidPhysicsIntegrationTests(unittest.TestCase):
    def test_official_controller_observe_stand_and_reset(self) -> None:
        async def scenario() -> None:
            runtime_root = os.environ.get("LEROBOT_HUMANOID_RUNTIME_ROOT")
            self.assertTrue(
                runtime_root,
                "LEROBOT_HUMANOID_RUNTIME_ROOT must point to the pinned official runtime checkout",
            )

            robot = HumanoidMuJoCo(
                runtime_root=runtime_root,
                control_hz=100.0,
                fixed_base=True,
            )
            self.assertIn(Capability.OBSERVE, robot.capabilities)
            self.assertIn(Capability.STAND, robot.capabilities)
            self.assertNotIn(Capability.WALK, robot.capabilities)

            await robot.connect()
            try:
                # Give the official controller thread time to execute real MuJoCo steps.
                await asyncio.sleep(0.05)
                observation = await robot.observe()
                self.assertEqual(
                    observation.state["backend"], "lerobot-humanoid-mujoco"
                )
                self.assertTrue(observation.state["fixed_base"])
                self.assertGreater(observation.state["sim_step_count"], 0)

                joints = observation.state["joint_position_rad"]
                velocities = observation.state["joint_velocity_radps"]
                torques = observation.state["joint_torque_nm"]
                self.assertEqual(len(joints), 12)
                self.assertEqual(len(velocities), 12)
                self.assertEqual(len(torques), 12)
                self.assertTrue(all(math.isfinite(float(value)) for value in joints))
                self.assertTrue(all(math.isfinite(float(value)) for value in velocities))
                self.assertTrue(all(math.isfinite(float(value)) for value in torques))

                stand = await robot.execute("stand")
                self.assertTrue(stand.ok, stand.detail)
                self.assertEqual(len(stand.data["joint_position_rad"]), 12)
                self.assertFalse(stand.data["policy_active"])

                reset = await robot.execute("reset")
                self.assertTrue(reset.ok, reset.detail)
                self.assertGreaterEqual(reset.data["sim_reset_count"], 1)
                self.assertEqual(len(reset.data["joint_position_rad"]), 12)

                after_reset = await robot.observe()
                self.assertTrue(after_reset.state["fixed_base"])
                self.assertEqual(len(after_reset.state["joint_position_rad"]), 12)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
