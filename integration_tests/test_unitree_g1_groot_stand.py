from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import yaml

from embodied_agent.embodiments import UnitreeG1LeRobot


class UnitreeG1GrootStandTests(TestCase):
    def test_semantic_stand_runs_pinned_groot_balance_policy_in_envhub(self) -> None:
        async def scenario() -> None:
            env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
            policy_root = Path(os.environ["GROOT_POLICY_ROOT"]).resolve()
            env_file = env_root / "env.py"
            config_path = env_root / "config.yaml"
            balance_path = policy_root / "GR00T-WholeBodyControl-Balance.onnx"
            walk_path = policy_root / "GR00T-WholeBodyControl-Walk.onnx"
            for path in (env_file, config_path, balance_path, walk_path):
                self.assertTrue(path.exists(), path)

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config.update(
                {
                    "ENABLE_ONSCREEN": False,
                    "ENABLE_OFFSCREEN": False,
                    "USE_JOYSTICK": 0,
                    "PRINT_SCENE_INFORMATION": False,
                    "INTERFACE": None,
                }
            )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            import huggingface_hub
            import lerobot.envs.utils as env_utils
            import lerobot.robots.unitree_g1.gr00t_locomotion as groot_locomotion

            def pinned_policy_download(*, repo_id: str, filename: str, **_: object) -> str:
                self.assertEqual(repo_id, "nepyope/GR00T-WholeBodyControl_g1")
                path = policy_root / filename
                self.assertIn(path, (balance_path, walk_path))
                return str(path)

            robot = UnitreeG1LeRobot(
                name="g1",
                is_simulation=True,
                controller="GrootLocomotionController",
                gravity_compensation=False,
                simulation_dds_interface=None,
            )

            with (
                patch.object(env_utils, "hf_hub_download", return_value=str(env_file)),
                patch.object(env_utils, "snapshot_download", return_value=str(env_root)),
                patch.object(huggingface_hub, "snapshot_download", return_value=str(env_root)),
                patch.object(groot_locomotion, "hf_hub_download", side_effect=pinned_policy_download),
            ):
                await robot.connect()
                try:
                    reset = await robot.execute("reset")
                    self.assertTrue(reset.ok)
                    await asyncio.sleep(0.25)

                    stand = await robot.execute("stand")
                    self.assertTrue(stand.ok)
                    self.assertEqual(stand.data["controller"], "GrootLocomotionController")
                    self.assertEqual(
                        stand.data["remote_axes"],
                        {
                            "remote.lx": 0.0,
                            "remote.ly": 0.0,
                            "remote.rx": 0.0,
                            "remote.ry": 0.0,
                        },
                    )

                    roll_samples: list[float] = []
                    pitch_samples: list[float] = []
                    for _ in range(100):
                        observation = await robot.observe()
                        self.assertEqual(len(observation.state["joint_position_rad"]), 29)
                        self.assertTrue(
                            all(np.isfinite(v) for v in observation.state["joint_position_rad"].values())
                        )
                        imu = observation.state["imu"]
                        roll = float(imu["rpy.roll"])
                        pitch = float(imu["rpy.pitch"])
                        self.assertTrue(math.isfinite(roll))
                        self.assertTrue(math.isfinite(pitch))
                        roll_samples.append(roll)
                        pitch_samples.append(pitch)
                        await asyncio.sleep(0.02)

                    native = robot._robot
                    self.assertIsNotNone(native)
                    controller = native.controller
                    self.assertIsNotNone(controller)
                    self.assertTrue(np.allclose(controller.cmd, np.zeros(3), atol=0.0))
                    with native._controller_action_lock:
                        controller_output = dict(native.controller_output)
                    self.assertEqual(len(controller_output), 15)
                    self.assertTrue(all(key.endswith(".q") for key in controller_output))
                    self.assertTrue(all(np.isfinite(v) for v in controller_output.values()))

                    tilt_samples = [
                        math.hypot(roll, pitch)
                        for roll, pitch in zip(roll_samples, pitch_samples, strict=True)
                    ]
                    metrics = {
                        "samples": len(tilt_samples),
                        "duration_s": 2.0,
                        "max_abs_roll_rad": max(abs(v) for v in roll_samples),
                        "max_abs_pitch_rad": max(abs(v) for v in pitch_samples),
                        "max_tilt_rad": max(tilt_samples),
                        "mean_tilt_rad": sum(tilt_samples) / len(tilt_samples),
                        "final_roll_rad": roll_samples[-1],
                        "final_pitch_rad": pitch_samples[-1],
                        "controller_output_joints": len(controller_output),
                    }
                    print("GROOT_STAND_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)

                    # First pinned behavioral gate: reject an obviously fallen body
                    # while recording the actual envelope so the threshold can be
                    # tightened from evidence rather than guessed.
                    self.assertLess(metrics["max_tilt_rad"], 1.0)
                finally:
                    await robot.disconnect()

        asyncio.run(scenario())
