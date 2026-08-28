from __future__ import annotations

import unittest
from pathlib import Path

from embodied_agent.agent.config import build_registry, load_config
from embodied_agent.embodiments import CrazyfliePyBullet, CrazyflieSim


class DefaultSimulationConfigTests(unittest.TestCase):
    def test_all_sim_uses_real_crazyflie_pybullet_adapter(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "all_sim.json"
        config = load_config(config_path)
        crazyflie_config = config["robots"]["crazyflie"]

        self.assertEqual(crazyflie_config["backend"], "sim")
        self.assertEqual(crazyflie_config["adapter"], "gym_pybullet_drones")
        self.assertEqual(
            crazyflie_config["tools"],
            ["observe", "takeoff", "goto", "land"],
        )

        registry = build_registry(config)
        crazyflie = registry.get("crazyflie")
        self.assertIsInstance(crazyflie, CrazyfliePyBullet)
        self.assertNotIsInstance(crazyflie, CrazyflieSim)
        self.assertEqual(crazyflie.backend, "gym-pybullet-drones")


if __name__ == "__main__":
    unittest.main()
