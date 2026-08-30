from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
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
GROOT_CONTROL_DT_S = 0.020
COMMAND_DURATION_S = 2.0
EPISODE_TIMEOUT_S = 45.0
EPISODE_PREFIX = "GROOT_CALIBRATION_EPISODE "
RESULT_PREFIX = "GROOT_LOCOMOTION_CALIBRATION "


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


def _body_delta(start: dict[str, float], end: dict[str, float]) -> tuple[float, float, float]:
    dx = end["x_m"] - start["x_m"]
    dy = end["y_m"] - start["y_m"]
    yaw = start["yaw_rad"]
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    yaw_delta = _wrap_angle(end["yaw_rad"] - yaw)
    return forward, lateral, yaw_delta


def _expected_controller_cmd(axis: str, value: float) -> np.ndarray:
    if axis == "remote.ly":
        return np.asarray([value, 0.0, 0.0], dtype=np.float64)
    if axis == "remote.lx":
        return np.asarray([0.0, -value, 0.0], dtype=np.float64)
    if axis == "remote.rx":
        return np.asarray([0.0, 0.0, -value], dtype=np.float64)
    raise ValueError(f"unsupported calibration axis: {axis}")


def _primary_rate(axis: str, forward_mps: float, lateral_mps: float, yaw_rps: float) -> float:
    if axis == "remote.ly":
        return forward_mps
    if axis == "remote.lx":
        return lateral_mps
    if axis == "remote.rx":
        return yaw_rps
    raise ValueError(axis)


async def _run_episode(episode_id: str, axis: str, value: float) -> dict[str, Any]:
    case = TestCase()
    env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
    policy_root = Path(os.environ["GROOT_POLICY_ROOT"]).resolve()
    env_file = env_root / "env.py"
    config_path = env_root / "config.yaml"
    balance_path = policy_root / "GR00T-WholeBodyControl-Balance.onnx"
    walk_path = policy_root / "GR00T-WholeBodyControl-Walk.onnx"
    for path in (env_file, config_path, balance_path, walk_path):
        case.assertTrue(path.exists(), path)

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

    try:
        import huggingface_hub
        import lerobot.envs.utils as env_utils
        import lerobot.robots.unitree_g1.gr00t_locomotion as groot_locomotion

        case.assertAlmostEqual(float(groot_locomotion.CONTROL_DT), GROOT_CONTROL_DT_S)

        def pinned_policy_download(*, repo_id: str, filename: str, **_: object) -> str:
            case.assertEqual(repo_id, "nepyope/GR00T-WholeBodyControl_g1")
            path = policy_root / filename
            case.assertIn(path, (balance_path, walk_path))
            return str(path)

        with (
            patch.object(env_utils, "hf_hub_download", return_value=str(env_file)),
            patch.object(env_utils, "snapshot_download", return_value=str(env_root)),
            patch.object(huggingface_hub, "snapshot_download", return_value=str(env_root)),
            patch.object(groot_locomotion, "hf_hub_download", side_effect=pinned_policy_download),
        ):
            robot = UnitreeG1LeRobot(
                name="g1",
                is_simulation=True,
                controller="GrootLocomotionController",
                gravity_compensation=False,
                simulation_dds_interface=None,
                simulation_publish_images=False,
            )
            await robot.connect()
            try:
                native = robot._robot
                case.assertIsNotNone(native)
                case.assertIsNotNone(native.controller)
                case.assertIsNotNone(native.sim_env)
                case.assertFalse(native.sim_env.camera_configs)
                inner_env = native.sim_env.sim_env
                case.assertFalse(inner_env.config["ENABLE_ELASTIC_BAND"])

                async def sample(duration_s: float) -> list[dict[str, float]]:
                    samples: list[dict[str, float]] = []
                    count = max(1, int(round(duration_s / GROOT_CONTROL_DT_S)))
                    for _ in range(count):
                        raw = inner_env.prepare_obs()
                        pose = np.asarray(raw["floating_base_pose"], dtype=np.float64).copy()
                        case.assertEqual(pose.shape, (7,))
                        case.assertTrue(np.isfinite(pose).all())
                        roll, pitch, yaw = _quaternion_to_rpy(pose[3:7])
                        samples.append(
                            {
                                "sim_time_s": float(raw["time"]),
                                "x_m": float(pose[0]),
                                "y_m": float(pose[1]),
                                "z_m": float(pose[2]),
                                "yaw_rad": yaw,
                                "tilt_rad": math.hypot(roll, pitch),
                            }
                        )
                        await asyncio.sleep(GROOT_CONTROL_DT_S)
                    return samples

                stand = await robot.execute("stand")
                case.assertTrue(stand.ok, stand.detail)
                pre = await sample(1.0)
                start = pre[-1]

                action = dict(REMOTE_ZERO)
                action[axis] = value
                native.send_action(action)
                moving = await sample(COMMAND_DURATION_S)
                end = moving[-1]

                # Give the asynchronous 50 Hz policy several updates, then verify
                # the requested normalized command has actually reached the GR00T
                # command vector. The agent never sees these raw axes.
                expected_cmd = _expected_controller_cmd(axis, value)
                case.assertTrue(
                    np.allclose(
                        np.asarray(native.controller.cmd, dtype=np.float64),
                        expected_cmd,
                        atol=1e-6,
                    )
                )

                stopped = await robot.execute("stand")
                case.assertTrue(stopped.ok, stopped.detail)
                post = await sample(0.5)
                case.assertTrue(np.allclose(native.controller.cmd, np.zeros(3), atol=0.0))

                forward_m, lateral_m, yaw_delta = _body_delta(start, end)
                pre_forward_m, pre_lateral_m, pre_yaw_delta = _body_delta(pre[0], pre[-1])
                moving_sim_time_s = end["sim_time_s"] - moving[0]["sim_time_s"]
                pre_sim_time_s = pre[-1]["sim_time_s"] - pre[0]["sim_time_s"]
                case.assertGreater(moving_sim_time_s, 1.5)
                case.assertGreater(pre_sim_time_s, 0.5)

                forward_mps = forward_m / moving_sim_time_s
                lateral_mps = lateral_m / moving_sim_time_s
                yaw_rps = yaw_delta / moving_sim_time_s
                pre_forward_mps = pre_forward_m / pre_sim_time_s
                pre_lateral_mps = pre_lateral_m / pre_sim_time_s
                pre_yaw_rps = pre_yaw_delta / pre_sim_time_s
                primary_rate = _primary_rate(axis, forward_mps, lateral_mps, yaw_rps)
                pre_primary_rate = _primary_rate(
                    axis,
                    pre_forward_mps,
                    pre_lateral_mps,
                    pre_yaw_rps,
                )
                all_samples = pre + moving + post
                min_height_m = min(sample["z_m"] for sample in all_samples)
                max_tilt_rad = max(sample["tilt_rad"] for sample in all_samples)

                result: dict[str, Any] = {
                    "episode_id": episode_id,
                    "axis": axis,
                    "normalized_value": value,
                    "nominal_command_duration_s": COMMAND_DURATION_S,
                    "moving_sim_time_s": moving_sim_time_s,
                    "forward_displacement_m": forward_m,
                    "lateral_displacement_m": lateral_m,
                    "yaw_delta_rad": yaw_delta,
                    "forward_mps": forward_mps,
                    "lateral_mps": lateral_mps,
                    "yaw_rate_rps": yaw_rps,
                    "pre_forward_mps": pre_forward_mps,
                    "pre_lateral_mps": pre_lateral_mps,
                    "pre_yaw_rate_rps": pre_yaw_rps,
                    "primary_rate_si": primary_rate,
                    "pre_primary_rate_si": pre_primary_rate,
                    "primary_to_pre_drift_ratio": abs(primary_rate)
                    / max(abs(pre_primary_rate), 1e-9),
                    "min_height_m": min_height_m,
                    "max_tilt_rad": max_tilt_rad,
                    "upright": min_height_m > 0.2 and max_tilt_rad < 1.0,
                }
                numeric = [
                    item for item in result.values() if isinstance(item, (int, float))
                ]
                case.assertTrue(all(math.isfinite(float(item)) for item in numeric))
                return result
            finally:
                await robot.disconnect()
    finally:
        config_path.write_text(original_config_text, encoding="utf-8")


def _run_episode_subprocess(episode_id: str, axis: str, value: float) -> dict[str, Any]:
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, str(test_file), "--episode", episode_id, axis, repr(value)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=EPISODE_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"calibration episode {episode_id} exited {completed.returncode}; "
            f"stdout={completed.stdout[-4000:]!r}; stderr={completed.stderr[-4000:]!r}"
        )
    markers = [
        line[len(EPISODE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(EPISODE_PREFIX)
    ]
    if len(markers) != 1:
        raise AssertionError(
            f"calibration episode {episode_id} expected one result marker, found "
            f"{len(markers)}; stdout={completed.stdout[-4000:]!r}"
        )
    return json.loads(markers[0])


def _group_key(result: dict[str, Any]) -> str:
    axis_name = {
        "remote.ly": "forward",
        "remote.lx": "lateral",
        "remote.rx": "yaw",
    }[result["axis"]]
    return f"{axis_name}_{float(result['normalized_value']):+.2f}"


class UnitreeG1GrootCalibrationTests(TestCase):
    def test_normalized_axes_against_body_frame_si_motion(self) -> None:
        # Every cardinal direction is repeated in a fresh interpreter. Order
        # interleaves axes/signs so environment startup drift cannot masquerade as
        # a command response. Forward 0.25 is included to expose local scaling.
        profiles = (
            ("forward_p025_a", "remote.ly", 0.25),
            ("lateral_p050_a", "remote.lx", 0.50),
            ("yaw_p050_a", "remote.rx", 0.50),
            ("forward_n050_a", "remote.ly", -0.50),
            ("forward_p050_a", "remote.ly", 0.50),
            ("lateral_n050_a", "remote.lx", -0.50),
            ("yaw_n050_a", "remote.rx", -0.50),
            ("forward_p025_b", "remote.ly", 0.25),
            ("lateral_n050_b", "remote.lx", -0.50),
            ("yaw_p050_b", "remote.rx", 0.50),
            ("forward_p050_b", "remote.ly", 0.50),
            ("lateral_p050_b", "remote.lx", 0.50),
            ("yaw_n050_b", "remote.rx", -0.50),
            ("forward_n050_b", "remote.ly", -0.50),
        )
        episodes = [
            _run_episode_subprocess(episode_id, axis, value)
            for episode_id, axis, value in profiles
        ]

        grouped: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            grouped.setdefault(_group_key(episode), []).append(episode)

        aggregates: dict[str, dict[str, float | int]] = {}
        for key, selected in grouped.items():
            rates = np.asarray([item["primary_rate_si"] for item in selected], dtype=np.float64)
            drift_rates = np.asarray(
                [item["pre_primary_rate_si"] for item in selected], dtype=np.float64
            )
            mean_rate = float(np.mean(rates))
            aggregates[key] = {
                "episodes": len(selected),
                "upright_episodes": sum(bool(item["upright"]) for item in selected),
                "mean_primary_rate_si": mean_rate,
                "std_primary_rate_si": float(np.std(rates)),
                "max_abs_pre_drift_rate_si": float(np.max(np.abs(drift_rates))),
                "mean_signal_to_drift_ratio": float(
                    np.mean([item["primary_to_pre_drift_ratio"] for item in selected])
                ),
                "mean_moving_sim_time_s": float(
                    np.mean([item["moving_sim_time_s"] for item in selected])
                ),
                "relative_std": float(np.std(rates) / max(abs(mean_rate), 1e-9)),
            }

        payload = {"episodes": episodes, "aggregates": aggregates}
        print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
        result_path = os.environ.get("GROOT_CALIBRATION_RESULT")
        if result_path:
            Path(result_path).write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        # Characterization gate only: prove the runtime/measurement contract, not
        # a hoped-for SI mapping or stability envelope. A fall is persisted as an
        # outcome and can restrict the eventual semantic capability.
        self.assertEqual(len(episodes), len(profiles))
        for episode in episodes:
            numeric = [
                item for item in episode.values() if isinstance(item, (int, float))
            ]
            self.assertTrue(all(math.isfinite(float(item)) for item in numeric))
            self.assertGreater(episode["moving_sim_time_s"], 1.5)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--episode":
        episode = asyncio.run(_run_episode(sys.argv[2], sys.argv[3], float(sys.argv[4])))
        print(EPISODE_PREFIX + json.dumps(episode, sort_keys=True), flush=True)
    else:
        import unittest

        unittest.main()
