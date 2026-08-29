from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path

from embodied_agent.core import Capability
from embodied_agent.embodiments import MicroduckMuJoCo
from embodied_agent.evals.reproducibility import benchmark_reproducibility


class MicroduckPhysicsIntegrationTests(unittest.TestCase):
    def _robot(self) -> MicroduckMuJoCo:
        runtime_root = Path(os.environ["MICRODUCK_RL_ROOT"])
        policy_dir = Path(os.environ["MICRODUCK_POLICY_DIR"])
        return MicroduckMuJoCo(
            runtime_root=runtime_root,
            walking_policy_path=policy_dir / "BEST_alpha_walking.onnx",
            standing_policy_path=policy_dir / "BEST_alpha_stand.onnx",
            kick_left_policy_path=policy_dir / "ball_kick_left.onnx",
            kick_right_policy_path=policy_dir / "ball_kick_right.onnx",
            roll_policy_path=policy_dir / "roulade.onnx",
        )

    def test_policy_backed_semantic_skills_in_real_mujoco(self) -> None:
        robot = self._robot()

        async def scenario() -> None:
            await robot.connect()
            try:
                self.assertTrue(
                    robot.supports(
                        Capability.OBSERVE,
                        Capability.STAND,
                        Capability.WALK,
                        Capability.KICK,
                        Capability.ROLL,
                    )
                )

                initial = await robot.observe()
                self.assertEqual(len(initial.state["joint_position_rad"]), 14)
                self.assertEqual(len(initial.state["joint_velocity_rps"]), 14)
                self.assertEqual(initial.state["control_hz"], 50.0)

                stand = await robot.execute("stand")
                self.assertTrue(stand.ok, stand.data)

                x0 = float(stand.data["position_m"][0])
                walk = await robot.execute(
                    "walk_velocity",
                    lin_x_mps=0.10,
                    lin_y_mps=0.0,
                    yaw_rate_rps=0.0,
                    duration_s=1.5,
                )
                self.assertTrue(walk.ok, walk.data)
                x1 = float(walk.data["position_m"][0])
                self.assertGreater(
                    x1 - x0,
                    0.02,
                    f"Microduck did not make meaningful forward progress: x0={x0}, x1={x1}",
                )

                kick = await robot.execute("kick", foot="left")
                self.assertTrue(kick.ok, kick.data)
                self.assertEqual(kick.data["foot"], "left")

                roll = await robot.execute("roll")
                self.assertTrue(roll.ok, roll.data)
                self.assertTrue(roll.data["completed"], roll.data)
                self.assertTrue(roll.data["tipped"], roll.data)
                self.assertTrue(roll.data["upright"], roll.data)
                self.assertLess(float(roll.data["projected_gravity"][2]), -0.85, roll.data)
                self.assertFalse(roll.data["reset_after_timeout"], roll.data)

                reset = await robot.execute("reset")
                self.assertTrue(reset.ok, reset.data)
            finally:
                await robot.disconnect()

        asyncio.run(scenario())

    def test_reset_stand_and_left_kick_are_reproducible(self) -> None:
        robot = self._robot()

        async def scenario() -> None:
            await robot.connect()
            try:
                async def run_episode(_: int):
                    reset = await robot.execute("reset")
                    self.assertTrue(reset.ok, reset.data)
                    stand = await robot.execute("stand")
                    self.assertTrue(stand.ok, stand.data)
                    kick = await robot.execute("kick", foot="left")
                    self.assertTrue(kick.ok, kick.data)
                    return {
                        "stand": {
                            "upright": bool(stand.data["upright"]),
                            "position_m": stand.data["position_m"],
                            "projected_gravity": stand.data["projected_gravity"],
                            "joint_position_rad": stand.data["joint_position_rad"],
                            "joint_velocity_rps": stand.data["joint_velocity_rps"],
                        },
                        "kick": {
                            "ok": bool(kick.ok),
                            "foot": kick.data["foot"],
                            "position_m": kick.data["position_m"],
                            "orientation_wxyz": kick.data["orientation_wxyz"],
                            "projected_gravity": kick.data["projected_gravity"],
                            "joint_position_rad": kick.data["joint_position_rad"],
                            "joint_velocity_rps": kick.data["joint_velocity_rps"],
                            "policy": kick.data["policy"],
                            "behavior": kick.data["behavior"],
                        },
                    }

                result = await benchmark_reproducibility(
                    run_episode,
                    attempts=3,
                    atol=1e-9,
                    label="microduck-reset-stand-left-kick",
                )
            finally:
                await robot.disconnect()

            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            self.assertEqual(result.reproducibility_rate, 1.0, result.to_dict())
            self.assertTrue(all(sample.matches_baseline for sample in result.samples))

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
