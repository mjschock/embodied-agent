from __future__ import annotations

import asyncio

from embodied_agent.core import RobotRegistry
from embodied_agent.embodiments import CrazyflieSim, HumanoidSim, XLeRobotSim


async def main() -> None:
    robots = RobotRegistry()
    robots.register(XLeRobotSim("xlerobot"))
    robots.register(CrazyflieSim("crazyflie"))
    robots.register(HumanoidSim("humanoid"))

    for robot in robots:
        await robot.connect()

    print("Connected embodiments:")
    for robot in robots:
        print(
            f"- {robot.name}: backend={robot.backend}, "
            f"capabilities={sorted(c.value for c in robot.capabilities)}"
        )

    print("\nOne command per embodiment:")
    print(await robots.get("xlerobot").execute("navigate_to", target="workbench"))
    print(await robots.get("crazyflie").execute("takeoff", altitude_m=1.0))
    print(await robots.get("humanoid").execute("stand"))

    print("\nObservations:")
    for robot in robots:
        print(await robot.observe())

    for robot in robots:
        await robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
