from __future__ import annotations

import asyncio
import unittest

from embodied_agent.core import Capability, RobotRegistry
from embodied_agent.embodiments import CrazyflieSim, HumanoidSim, XLeRobotSim


class SmokeTests(unittest.TestCase):
    def test_registry_and_capabilities(self) -> None:
        registry = RobotRegistry()
        registry.register(XLeRobotSim("xlerobot"))
        registry.register(CrazyflieSim("crazyflie"))
        registry.register(HumanoidSim("humanoid"))

        self.assertEqual(len(registry), 3)
        self.assertEqual(
            [r.name for r in registry.with_capabilities(Capability.FLY)],
            ["crazyflie"],
        )
        self.assertEqual(
            [r.name for r in registry.with_capabilities(Capability.MANIPULATE)],
            ["xlerobot"],
        )
        self.assertEqual(
            [r.name for r in registry.with_capabilities(Capability.WALK)],
            ["humanoid"],
        )

    def test_end_to_end_commands(self) -> None:
        async def scenario() -> None:
            xlerobot = XLeRobotSim("xlerobot")
            crazyflie = CrazyflieSim("crazyflie")
            humanoid = HumanoidSim("humanoid")

            for robot in (xlerobot, crazyflie, humanoid):
                await robot.connect()

            x_result = await xlerobot.execute("navigate_to", target="desk")
            c_result = await crazyflie.execute("takeoff", altitude_m=1.25)
            h_result = await humanoid.execute("stand")

            self.assertTrue(x_result.ok)
            self.assertEqual(x_result.data["location"], "desk")

            self.assertTrue(c_result.ok)
            self.assertEqual(c_result.data["altitude_m"], 1.25)

            self.assertTrue(h_result.ok)
            self.assertEqual(h_result.data["posture"], "standing")

            for robot in (xlerobot, crazyflie, humanoid):
                await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
