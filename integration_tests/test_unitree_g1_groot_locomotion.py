from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import yaml

from embodied_agent.embodiments import UnitreeG1LeRobot


REMOTE_ZERO = {
    "remote.lx": 0.0,
    "remote.ly": 0.0,
    "remote.rx": 0.0,
    "remote.ry": 0.0,
}


def _quaternion_to_rpy(quaternion_wxyz: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class UnitreeG1GrootLocomotionCharacterizationTests(TestCase):
    def test_forward_normalized_axis_against_untethered_world_pose(self) -> None:
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
                    # The pinned EnvHub normally applies a very stiff world-frame
                    # elastic tether to the torso. Translation measured with that
                    # tether enabled would not be a truthful locomotion calibration.
                    "ENABLE_ELASTIC_BAND": False,
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
                    native = robot._robot
                    self.assertIsNotNone(native)
                    self.assertIsNotNone(native.controller)
                    self.assertIsNotNone(native.sim_env)
                    inner_env = native.sim_env.sim_env
                    self.assertFalse(inner_env.config["ENABLE_ELASTIC_BAND"])

                    async def sample_pose(duration_s: float, interval_s: float = 0.02) -> list[dict[str, float]]:
                        samples: list[dict[str, float]] = []
                        count = max(1, int(round(duration_s / interval_s)))
                        for _ in range(count):
                            raw = inner_env.prepare_obs()
                            pose = np.asarray(raw["floating_base_pose"], dtype=np.float64).copy()
                            self.assertEqual(pose.shape, (7,))
                            self.assertTrue(np.isfinite(pose).all())
                            roll, pitch, yaw = _quaternion_to_rpy(pose[3:7])
                            samples.append(
                                {
                                    "x_m": float(pose[0]),
                                    "y_m": float(pose[1]),
                                    "z_m": float(pose[2]),
                                    "roll_rad": roll,
                                    "pitch_rad": pitch,
                                    "yaw_rad": yaw,
                                    "tilt_rad": math.hypot(roll, pitch),
                                }
                            )
                            await asyncio.sleep(interval_s)
                        return samples

                    async def run_forward_episode(remote_ly: float) -> dict[str, Any]:
                        reset = await robot.execute("reset")
                        self.assertTrue(reset.ok, reset.detail)
                        stand = await robot.execute("stand")
                        self.assertTrue(stand.ok, stand.detail)

                        # Give the reset balance policy time to establish the
                        # untethered initial condition before measuring motion.
                        pre_samples = await sample_pose(1.0)
                        start = pre_samples[-1]

                        command = dict(REMOTE_ZERO)
                        command["remote.ly"] = remote_ly
                        native.send_action(command)
                        command_duration_s = 2.0
                        moving_samples = await sample_pose(command_duration_s)
                        end = moving_samples[-1]

                        # Return to the public semantic standing boundary after
                        # every internal calibration command.
                        stopped = await robot.execute("stand")
                        self.assertTrue(stopped.ok, stopped.detail)
                        post_samples = await sample_pose(0.5)

                        initial_yaw = start["yaw_rad"]
                        dx = end["x_m"] - start["x_m"]
                        dy = end["y_m"] - start["y_m"]
                        forward_m = math.cos(initial_yaw) * dx + math.sin(initial_yaw) * dy
                        lateral_m = -math.sin(initial_yaw) * dx + math.cos(initial_yaw) * dy
                        yaw_delta = _wrap_angle(end["yaw_rad"] - initial_yaw)
                        all_samples = pre_samples + moving_samples + post_samples

                        with native._controller_action_lock:
                            controller_input = dict(native.controller_input)
                        self.assertEqual(
                            {key: float(controller_input[key]) for key in REMOTE_ZERO},
                            REMOTE_ZERO,
                        )
                        self.assertTrue(np.allclose(native.controller.cmd, np.zeros(3), atol=0.0))

                        return {
                            "remote_ly": remote_ly,
                            "command_duration_s": command_duration_s,
                            "forward_displacement_m": forward_m,
                            "lateral_displacement_m": lateral_m,
                            "mean_forward_mps": forward_m / command_duration_s,
                            "mean_lateral_mps": lateral_m / command_duration_s,
                            "yaw_delta_rad": yaw_delta,
                            "mean_yaw_rate_rps": yaw_delta / command_duration_s,
                            "start_xyz_m": [start["x_m"], start["y_m"], start["z_m"]],
                            "end_xyz_m": [end["x_m"], end["y_m"], end["z_m"]],
                            "min_height_m": min(sample["z_m"] for sample in all_samples),
                            "max_tilt_rad": max(sample["tilt_rad"] for sample in all_samples),
                            "pre_drift_m": math.hypot(
                                pre_samples[-1]["x_m"] - pre_samples[0]["x_m"],
                                pre_samples[-1]["y_m"] - pre_samples[0]["y_m"],
                            ),
                            "post_drift_m": math.hypot(
                                post_samples[-1]["x_m"] - post_samples[0]["x_m"],
                                post_samples[-1]["y_m"] - post_samples[0]["y_m"],
                            ),
                        }

                    results = [
                        await run_forward_episode(value)
                        for value in (0.0, 0.10, 0.25, 0.50)
                    ]
                    print(
                        "GROOT_LOCOMOTION_CHARACTERIZATION "
                        + json.dumps(results, sort_keys=True),
                        flush=True,
                    )

                    # This first pass is an evidence-gathering probe, not an SI
                    # calibration contract. Only require a finite, non-collapsed
                    # untethered simulation and let CI establish the actual command
                    # response before setting accuracy/linearity thresholds.
                    for result in results:
                        numeric = [
                            value
                            for value in result.values()
                            if isinstance(value, (int, float))
                        ]
                        self.assertTrue(all(math.isfinite(float(value)) for value in numeric))
                        self.assertGreater(result["min_height_m"], 0.2)
                        self.assertLess(result["max_tilt_rad"], 1.0)
                finally:
                    await robot.disconnect()

        asyncio.run(scenario())


if __name__ == "__main__":
    import unittest

    unittest.main()
