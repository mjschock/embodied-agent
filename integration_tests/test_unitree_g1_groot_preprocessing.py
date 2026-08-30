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


class UnitreeG1GrootPreprocessingCharacterizationTests(TestCase):
    def test_stock_vs_policy_reference_angular_scaling(self) -> None:
        async def scenario() -> None:
            env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
            policy_root = Path(os.environ["GROOT_POLICY_ROOT"]).resolve()
            env_file = env_root / "env.py"
            config_path = env_root / "config.yaml"
            balance_path = policy_root / "GR00T-WholeBodyControl-Balance.onnx"
            walk_path = policy_root / "GR00T-WholeBodyControl-Walk.onnx"
            for path in (env_file, config_path, balance_path, walk_path):
                self.assertTrue(path.exists(), path)

            original_config_text = config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(original_config_text)
            config.update(
                {
                    "ENABLE_ONSCREEN": False,
                    "ENABLE_OFFSCREEN": False,
                    "USE_JOYSTICK": 0,
                    "PRINT_SCENE_INFORMATION": False,
                    "INTERFACE": None,
                    "ENABLE_ELASTIC_BAND": False,
                }
            )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            import huggingface_hub
            import lerobot.envs.utils as env_utils
            import lerobot.robots.unitree_g1.gr00t_locomotion as groot_locomotion

            # LeRobot v0.6.1 uses 0.25 for both values. The published GR00T
            # architecture metadata and NVIDIA 15-action reference runner specify
            # 0.5 for angular-velocity observation scale and yaw-command scale.
            # Keep this test as an A/B characterization; do not change production
            # adapter behavior unless measured simulator evidence justifies it.
            self.assertEqual(float(groot_locomotion.ANG_VEL_SCALE), 0.25)
            self.assertEqual([float(v) for v in groot_locomotion.CMD_SCALE], [2.0, 2.0, 0.25])
            original_ang_vel_scale = groot_locomotion.ANG_VEL_SCALE
            original_cmd_scale = list(groot_locomotion.CMD_SCALE)

            def pinned_policy_download(*, repo_id: str, filename: str, **_: object) -> str:
                self.assertEqual(repo_id, "nepyope/GR00T-WholeBodyControl_g1")
                path = policy_root / filename
                self.assertIn(path, (balance_path, walk_path))
                return str(path)

            async def episode(
                *,
                preprocessing: str,
                ang_vel_scale: float,
                yaw_cmd_scale: float,
                remote_ly: float,
            ) -> dict[str, Any]:
                # EnvHub runs MuJoCo on its own thread. Its native reset mutates
                # mjData without synchronizing against mj_step(), which can segfault.
                # A fresh simulator lifecycle gives each A/B episode the intended
                # initial state without racing the physics thread. PR #42 validates
                # repeated same-process G1 connect/disconnect lifecycles.
                groot_locomotion.ANG_VEL_SCALE = ang_vel_scale
                groot_locomotion.CMD_SCALE = [2.0, 2.0, yaw_cmd_scale]
                robot = UnitreeG1LeRobot(
                    name="g1",
                    is_simulation=True,
                    controller="GrootLocomotionController",
                    gravity_compensation=False,
                    simulation_dds_interface=None,
                )
                await robot.connect()
                try:
                    native = robot._robot
                    self.assertIsNotNone(native)
                    self.assertIsNotNone(native.controller)
                    self.assertIsNotNone(native.sim_env)
                    inner_env = native.sim_env.sim_env
                    self.assertFalse(inner_env.config["ENABLE_ELASTIC_BAND"])

                    async def sample(
                        duration_s: float,
                        interval_s: float = 0.02,
                    ) -> list[dict[str, Any]]:
                        samples: list[dict[str, Any]] = []
                        for _ in range(max(1, int(round(duration_s / interval_s)))):
                            raw = inner_env.prepare_obs()
                            pose = np.asarray(raw["floating_base_pose"], dtype=np.float64).copy()
                            body_q = np.asarray(raw["body_q"], dtype=np.float64)[:15].copy()
                            body_dq = np.asarray(raw["body_dq"], dtype=np.float64)[:15].copy()
                            self.assertEqual(pose.shape, (7,))
                            self.assertEqual(body_q.shape, (15,))
                            self.assertEqual(body_dq.shape, (15,))
                            self.assertTrue(np.isfinite(pose).all())
                            self.assertTrue(np.isfinite(body_q).all())
                            self.assertTrue(np.isfinite(body_dq).all())
                            roll, pitch, yaw = _quaternion_to_rpy(pose[3:7])
                            with native._controller_action_lock:
                                controller_input = dict(native.controller_input)
                                controller_output = dict(native.controller_output)
                            output = np.asarray(list(controller_output.values()), dtype=np.float64)
                            self.assertTrue(np.isfinite(output).all())
                            samples.append(
                                {
                                    "x_m": float(pose[0]),
                                    "y_m": float(pose[1]),
                                    "z_m": float(pose[2]),
                                    "yaw_rad": yaw,
                                    "tilt_rad": math.hypot(roll, pitch),
                                    "body_q": body_q,
                                    "body_dq": body_dq,
                                    "controller_output": output,
                                    "controller_input_remote_ly": float(controller_input["remote.ly"]),
                                    "controller_cmd_forward": float(native.controller.cmd[0]),
                                }
                            )
                            await asyncio.sleep(interval_s)
                        return samples

                    stand = await robot.execute("stand")
                    self.assertTrue(stand.ok, stand.detail)
                    pre = await sample(1.0)
                    start = pre[-1]

                    action = dict(REMOTE_ZERO)
                    action["remote.ly"] = remote_ly
                    native.send_action(action)
                    moving = await sample(2.0)
                    end = moving[-1]
                    steady = moving[10:]
                    self.assertTrue(steady)
                    self.assertLessEqual(
                        max(abs(s["controller_input_remote_ly"] - remote_ly) for s in steady),
                        1e-12,
                    )
                    self.assertLessEqual(
                        max(abs(s["controller_cmd_forward"] - remote_ly) for s in steady),
                        1e-6,
                    )

                    stopped = await robot.execute("stand")
                    self.assertTrue(stopped.ok, stopped.detail)
                    post = await sample(0.5)
                    self.assertTrue(np.allclose(native.controller.cmd, np.zeros(3), atol=0.0))

                    initial_yaw = start["yaw_rad"]
                    dx = end["x_m"] - start["x_m"]
                    dy = end["y_m"] - start["y_m"]
                    forward_m = math.cos(initial_yaw) * dx + math.sin(initial_yaw) * dy
                    lateral_m = -math.sin(initial_yaw) * dx + math.cos(initial_yaw) * dy
                    yaw_delta = _wrap_angle(end["yaw_rad"] - initial_yaw)
                    all_samples = pre + moving + post
                    body_q = np.asarray([s["body_q"] for s in steady], dtype=np.float64)
                    body_dq = np.asarray([s["body_dq"] for s in steady], dtype=np.float64)
                    outputs = [s["controller_output"] for s in steady if s["controller_output"].size == 15]
                    self.assertTrue(outputs)
                    controller_outputs = np.asarray(outputs, dtype=np.float64)

                    return {
                        "preprocessing": preprocessing,
                        "ang_vel_scale": ang_vel_scale,
                        "yaw_cmd_scale": yaw_cmd_scale,
                        "remote_ly": remote_ly,
                        "forward_displacement_m": forward_m,
                        "mean_forward_mps": forward_m / 2.0,
                        "lateral_displacement_m": lateral_m,
                        "yaw_delta_rad": yaw_delta,
                        "min_height_m": min(s["z_m"] for s in all_samples),
                        "max_tilt_rad": max(s["tilt_rad"] for s in all_samples),
                        "pre_drift_m": math.hypot(
                            pre[-1]["x_m"] - pre[0]["x_m"],
                            pre[-1]["y_m"] - pre[0]["y_m"],
                        ),
                        "body_q_peak_to_peak_l2_rad": float(np.linalg.norm(np.ptp(body_q, axis=0))),
                        "body_dq_rms_rad_s": float(np.sqrt(np.mean(np.square(body_dq)))),
                        "controller_target_peak_to_peak_l2_rad": float(
                            np.linalg.norm(np.ptp(controller_outputs, axis=0))
                        ),
                    }
                finally:
                    await robot.disconnect()

            try:
                with (
                    patch.object(env_utils, "hf_hub_download", return_value=str(env_file)),
                    patch.object(env_utils, "snapshot_download", return_value=str(env_root)),
                    patch.object(huggingface_hub, "snapshot_download", return_value=str(env_root)),
                    patch.object(groot_locomotion, "hf_hub_download", side_effect=pinned_policy_download),
                ):
                    remote_ly = 0.50
                    results = [
                        await episode(
                            preprocessing="lerobot_v0.6.1",
                            ang_vel_scale=0.25,
                            yaw_cmd_scale=0.25,
                            remote_ly=remote_ly,
                        ),
                        await episode(
                            preprocessing="groot_reference",
                            ang_vel_scale=0.50,
                            yaw_cmd_scale=0.50,
                            remote_ly=remote_ly,
                        ),
                    ]
            finally:
                groot_locomotion.ANG_VEL_SCALE = original_ang_vel_scale
                groot_locomotion.CMD_SCALE = original_cmd_scale
                config_path.write_text(original_config_text, encoding="utf-8")

            print("GROOT_PREPROCESSING_AB " + json.dumps(results, sort_keys=True), flush=True)
            for result in results:
                numeric = [value for value in result.values() if isinstance(value, (int, float))]
                self.assertTrue(all(math.isfinite(float(value)) for value in numeric))
                self.assertGreater(result["min_height_m"], 0.2)
                self.assertLess(result["max_tilt_rad"], 1.0)

        asyncio.run(scenario())


if __name__ == "__main__":
    import unittest

    unittest.main()
