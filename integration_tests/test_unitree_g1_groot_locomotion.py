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


class _PolicyProbe:
    """Transparent ONNX session wrapper that counts real policy inference calls."""

    def __init__(self, session: Any):
        self._session = session
        self.run_calls = 0

    def get_inputs(self) -> Any:
        return self._session.get_inputs()

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.run_calls += 1
        return self._session.run(*args, **kwargs)


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


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


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

            original_config_text = config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(original_config_text)
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
                    # This is the non-interactive equivalent of the simulator's
                    # documented "press 9 to release the strap" operation.
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

            async def run_forward_episode(remote_ly: float) -> dict[str, Any]:
                # EnvHub advances MuJoCo on a background thread. Its native reset
                # mutates mjData without synchronizing against mj_step(), which can
                # segfault or corrupt MuJoCo's stack. Give every command value a
                # fresh simulator lifecycle instead. PR #42 validates repeated
                # same-process G1 connect/disconnect lifecycles.
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

                    # Count calls while delegating to the actual pinned ONNX sessions.
                    # This lets the integration gate distinguish "axis never reached
                    # GR00T" from "walk policy ran but world motion was negligible".
                    balance_probe = _PolicyProbe(native.controller.policy_balance)
                    walk_probe = _PolicyProbe(native.controller.policy_walk)
                    native.controller.policy_balance = balance_probe
                    native.controller.policy_walk = walk_probe

                    async def sample_pose(
                        duration_s: float,
                        interval_s: float = 0.02,
                        *,
                        capture_controller: bool = False,
                    ) -> list[dict[str, Any]]:
                        samples: list[dict[str, Any]] = []
                        count = max(1, int(round(duration_s / interval_s)))
                        for _ in range(count):
                            raw = inner_env.prepare_obs()
                            pose = np.asarray(raw["floating_base_pose"], dtype=np.float64).copy()
                            self.assertEqual(pose.shape, (7,))
                            self.assertTrue(np.isfinite(pose).all())
                            roll, pitch, yaw = _quaternion_to_rpy(pose[3:7])
                            sample: dict[str, Any] = {
                                "x_m": float(pose[0]),
                                "y_m": float(pose[1]),
                                "z_m": float(pose[2]),
                                "roll_rad": roll,
                                "pitch_rad": pitch,
                                "yaw_rad": yaw,
                                "tilt_rad": math.hypot(roll, pitch),
                            }
                            if capture_controller:
                                with native._controller_action_lock:
                                    controller_input = dict(native.controller_input)
                                    controller_output = dict(native.controller_output)
                                cmd = np.asarray(native.controller.cmd, dtype=np.float64).copy()
                                groot_action = np.asarray(
                                    native.controller.groot_action,
                                    dtype=np.float64,
                                ).copy()
                                output_values = np.asarray(
                                    list(controller_output.values()),
                                    dtype=np.float64,
                                )
                                body_q = np.asarray(raw["body_q"], dtype=np.float64)[:15].copy()
                                body_dq = np.asarray(raw["body_dq"], dtype=np.float64)[:15].copy()
                                lowcmd_target_q = np.asarray(
                                    [native.msg.motor_cmd[index].q for index in range(15)],
                                    dtype=np.float64,
                                )
                                self.assertEqual(cmd.shape, (3,))
                                self.assertEqual(groot_action.shape, (15,))
                                self.assertEqual(output_values.shape, (15,))
                                self.assertEqual(body_q.shape, (15,))
                                self.assertEqual(body_dq.shape, (15,))
                                self.assertEqual(lowcmd_target_q.shape, (15,))
                                self.assertTrue(np.isfinite(cmd).all())
                                self.assertTrue(np.isfinite(groot_action).all())
                                self.assertTrue(np.isfinite(output_values).all())
                                self.assertTrue(np.isfinite(body_q).all())
                                self.assertTrue(np.isfinite(body_dq).all())
                                self.assertTrue(np.isfinite(lowcmd_target_q).all())
                                sample.update(
                                    {
                                        "controller_input_remote_ly": float(
                                            controller_input["remote.ly"]
                                        ),
                                        "controller_cmd_forward": float(cmd[0]),
                                        "controller_cmd_lateral": float(cmd[1]),
                                        "controller_cmd_yaw": float(cmd[2]),
                                        "groot_action_l2": float(np.linalg.norm(groot_action)),
                                        "controller_output_l2": float(
                                            np.linalg.norm(output_values)
                                        ),
                                        "controller_output_count": int(output_values.size),
                                        "controller_target_q": output_values.tolist(),
                                        "lowcmd_target_q": lowcmd_target_q.tolist(),
                                        "body_q": body_q.tolist(),
                                        "body_dq": body_dq.tolist(),
                                    }
                                )
                            samples.append(sample)
                            await asyncio.sleep(interval_s)
                        return samples

                    stand = await robot.execute("stand")
                    self.assertTrue(stand.ok, stand.detail)

                    # Let the balance policy establish the untethered initial
                    # condition before measuring motion.
                    pre_samples = await sample_pose(1.0)
                    start = pre_samples[-1]

                    command = dict(REMOTE_ZERO)
                    command["remote.ly"] = remote_ly
                    balance_calls_before = balance_probe.run_calls
                    walk_calls_before = walk_probe.run_calls
                    native.send_action(command)
                    command_duration_s = 2.0
                    moving_samples = await sample_pose(
                        command_duration_s,
                        capture_controller=True,
                    )
                    end = moving_samples[-1]
                    balance_policy_calls = balance_probe.run_calls - balance_calls_before
                    walk_policy_calls = walk_probe.run_calls - walk_calls_before

                    # Ignore only the first 200 ms when checking persisted command
                    # delivery so the asynchronous 50 Hz controller gets several
                    # cycles to consume the newly written controller_input.
                    steady_controller_samples = moving_samples[10:]
                    self.assertTrue(steady_controller_samples)
                    input_errors = [
                        abs(sample["controller_input_remote_ly"] - remote_ly)
                        for sample in steady_controller_samples
                    ]
                    cmd_forward_errors = [
                        abs(sample["controller_cmd_forward"] - remote_ly)
                        for sample in steady_controller_samples
                    ]
                    cmd_cross_axis = [
                        max(
                            abs(sample["controller_cmd_lateral"]),
                            abs(sample["controller_cmd_yaw"]),
                        )
                        for sample in steady_controller_samples
                    ]
                    self.assertLessEqual(max(input_errors), 1e-12)
                    self.assertLessEqual(max(cmd_forward_errors), 1e-6)
                    self.assertLessEqual(max(cmd_cross_axis), 1e-9)
                    self.assertTrue(
                        all(
                            sample["controller_output_count"] == 15
                            for sample in steady_controller_samples
                        )
                    )
                    if abs(remote_ly) < 0.05:
                        self.assertGreater(balance_policy_calls, 0)
                        self.assertEqual(walk_policy_calls, 0)
                    else:
                        self.assertGreater(walk_policy_calls, 0)

                    controller_targets = np.asarray(
                        [sample["controller_target_q"] for sample in steady_controller_samples],
                        dtype=np.float64,
                    )
                    lowcmd_targets = np.asarray(
                        [sample["lowcmd_target_q"] for sample in steady_controller_samples],
                        dtype=np.float64,
                    )
                    body_q = np.asarray(
                        [sample["body_q"] for sample in steady_controller_samples],
                        dtype=np.float64,
                    )
                    body_dq = np.asarray(
                        [sample["body_dq"] for sample in steady_controller_samples],
                        dtype=np.float64,
                    )
                    self.assertEqual(controller_targets.shape[1], 15)
                    self.assertEqual(lowcmd_targets.shape, controller_targets.shape)
                    self.assertEqual(body_q.shape, controller_targets.shape)
                    self.assertEqual(body_dq.shape, controller_targets.shape)

                    # Return to the public semantic standing boundary after every
                    # internal calibration command before ending this lifecycle.
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
                        "balance_policy_calls": balance_policy_calls,
                        "walk_policy_calls": walk_policy_calls,
                        "max_controller_input_error": max(input_errors),
                        "max_controller_cmd_forward_error": max(cmd_forward_errors),
                        "max_controller_cmd_cross_axis": max(cmd_cross_axis),
                        "mean_groot_action_l2": float(
                            np.mean(
                                [
                                    sample["groot_action_l2"]
                                    for sample in steady_controller_samples
                                ]
                            )
                        ),
                        "max_groot_action_l2": max(
                            sample["groot_action_l2"]
                            for sample in steady_controller_samples
                        ),
                        "mean_controller_output_l2": float(
                            np.mean(
                                [
                                    sample["controller_output_l2"]
                                    for sample in steady_controller_samples
                                ]
                            )
                        ),
                        "controller_target_temporal_std_l2_rad": float(
                            np.linalg.norm(np.std(controller_targets, axis=0))
                        ),
                        "controller_target_peak_to_peak_l2_rad": float(
                            np.linalg.norm(np.ptp(controller_targets, axis=0))
                        ),
                        "lowcmd_target_temporal_std_l2_rad": float(
                            np.linalg.norm(np.std(lowcmd_targets, axis=0))
                        ),
                        "lowcmd_target_peak_to_peak_l2_rad": float(
                            np.linalg.norm(np.ptp(lowcmd_targets, axis=0))
                        ),
                        "lowcmd_vs_controller_target_rms_rad": _rms(
                            lowcmd_targets - controller_targets
                        ),
                        "body_q_temporal_std_l2_rad": float(
                            np.linalg.norm(np.std(body_q, axis=0))
                        ),
                        "body_q_peak_to_peak_l2_rad": float(
                            np.linalg.norm(np.ptp(body_q, axis=0))
                        ),
                        "body_dq_rms_rad_s": _rms(body_dq),
                        "body_q_tracking_rms_rad": _rms(body_q - lowcmd_targets),
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
                    results = [
                        await run_forward_episode(value)
                        for value in (0.0, 0.10, 0.25, 0.50)
                    ]
            finally:
                config_path.write_text(original_config_text, encoding="utf-8")

            print(
                "GROOT_LOCOMOTION_CHARACTERIZATION "
                + json.dumps(results, sort_keys=True),
                flush=True,
            )

            # This remains a characterization gate, not an SI calibration.
            # Require the real command/policy pipeline plus a finite,
            # non-collapsed untethered simulation. World-pose response and
            # lower-body target/tracking activity are recorded so any future
            # locomotion claim comes from measured actuation, not assumptions.
            for result in results:
                numeric = [
                    value
                    for value in result.values()
                    if isinstance(value, (int, float))
                ]
                self.assertTrue(all(math.isfinite(float(value)) for value in numeric))
                self.assertGreater(result["min_height_m"], 0.2)
                self.assertLess(result["max_tilt_rad"], 1.0)

        asyncio.run(scenario())


if __name__ == "__main__":
    import unittest

    unittest.main()
