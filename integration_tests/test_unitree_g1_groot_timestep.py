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

PINNED_ENVHUB_DT_S = 0.004
NVIDIA_REFERENCE_DT_S = 0.005
GROOT_CONTROL_DT_S = 0.020
EPISODE_TIMEOUT_S = 45.0
EPISODE_MEASURED_PREFIX = "GROOT_TIMESTEP_EPISODE_MEASURED "
EPISODE_CLEAN_PREFIX = "GROOT_TIMESTEP_EPISODE_CLEAN "
EPISODE_RESULT_PREFIX = "GROOT_TIMESTEP_EPISODE "
LEROBOT_LOWER_BODY_KP = np.asarray(
    [150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 250, 250, 250],
    dtype=np.float64,
)
LOWER_BODY_KD = np.asarray(
    [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5],
    dtype=np.float64,
)


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


async def _run_episode(label: str, sim_dt_s: float) -> dict[str, Any]:
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
    original_config = yaml.safe_load(original_config_text)
    case.assertAlmostEqual(float(original_config["SIMULATE_DT"]), PINNED_ENVHUB_DT_S)

    configured = yaml.safe_load(original_config_text)
    configured.update(
        {
            "ENABLE_ONSCREEN": False,
            "ENABLE_OFFSCREEN": False,
            "USE_JOYSTICK": 0,
            "PRINT_SCENE_INFORMATION": False,
            "INTERFACE": None,
            "ENABLE_ELASTIC_BAND": False,
            "SIMULATE_DT": sim_dt_s,
        }
    )
    config_path.write_text(yaml.safe_dump(configured, sort_keys=False), encoding="utf-8")

    try:
        import huggingface_hub
        import lerobot.envs.utils as env_utils
        import lerobot.robots.unitree_g1.gr00t_locomotion as groot_locomotion

        case.assertAlmostEqual(float(groot_locomotion.CONTROL_DT), GROOT_CONTROL_DT_S)
        case.assertAlmostEqual(float(groot_locomotion.ANG_VEL_SCALE), 0.25)
        case.assertEqual([float(v) for v in groot_locomotion.CMD_SCALE], [2.0, 2.0, 0.25])

        def pinned_policy_download(*, repo_id: str, filename: str, **_: object) -> str:
            case.assertEqual(repo_id, "nepyope/GR00T-WholeBodyControl_g1")
            path = policy_root / filename
            case.assertIn(path, (balance_path, walk_path))
            return str(path)

        # EnvHub step() performs exactly one MuJoCo step and explicitly leaves
        # timing to its caller. LeRobot's UnitreeG1 caller is pinned to 4 ms by
        # UnitreeG1Config.control_dt, so changing only SIMULATE_DT would still
        # execute about five physics calls per 20 ms GR00T update. The reference
        # profile must change the caller interval and MuJoCo dt together to model
        # NVIDIA's 5 ms simulation / 20 ms policy cadence truthfully.
        base_robot_factory = UnitreeG1LeRobot._load_default_robot_factory()

        def simulation_timing_factory(**kwargs: Any) -> Any:
            native_robot = base_robot_factory(**kwargs)
            case.assertAlmostEqual(float(native_robot.control_dt), PINNED_ENVHUB_DT_S)
            native_robot.control_dt = sim_dt_s
            native_robot.config.control_dt = sim_dt_s
            return native_robot

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
                robot_factory=simulation_timing_factory,
            )
            await robot.connect()
            try:
                native = robot._robot
                case.assertIsNotNone(native)
                case.assertIsNotNone(native.controller)
                case.assertIsNotNone(native.sim_env)
                simulator = native.sim_env.simulator
                inner_env = native.sim_env.sim_env

                case.assertFalse(inner_env.config["ENABLE_ELASTIC_BAND"])
                case.assertFalse(native.sim_env.camera_configs)
                case.assertAlmostEqual(float(native.control_dt), sim_dt_s)
                case.assertAlmostEqual(float(native.config.control_dt), sim_dt_s)
                case.assertAlmostEqual(float(native.controller.control_dt), GROOT_CONTROL_DT_S)
                case.assertAlmostEqual(float(simulator.sim_dt), sim_dt_s)
                case.assertAlmostEqual(float(inner_env.sim_dt), sim_dt_s)
                case.assertAlmostEqual(float(inner_env.mj_model.opt.timestep), sim_dt_s)
                case.assertAlmostEqual(
                    GROOT_CONTROL_DT_S / sim_dt_s,
                    round(GROOT_CONTROL_DT_S / sim_dt_s),
                    places=9,
                )
                case.assertTrue(np.array_equal(np.asarray(native.kp[:15]), LEROBOT_LOWER_BODY_KP))
                case.assertTrue(np.array_equal(np.asarray(native.kd[:15]), LOWER_BODY_KD))

                async def sample(
                    duration_s: float,
                    interval_s: float = GROOT_CONTROL_DT_S,
                ) -> list[dict[str, Any]]:
                    samples: list[dict[str, Any]] = []
                    for _ in range(max(1, int(round(duration_s / interval_s)))):
                        raw = inner_env.prepare_obs()
                        pose = np.asarray(raw["floating_base_pose"], dtype=np.float64).copy()
                        body_q = np.asarray(raw["body_q"], dtype=np.float64)[:15].copy()
                        body_dq = np.asarray(raw["body_dq"], dtype=np.float64)[:15].copy()
                        case.assertEqual(pose.shape, (7,))
                        case.assertEqual(body_q.shape, (15,))
                        case.assertEqual(body_dq.shape, (15,))
                        case.assertTrue(np.isfinite(pose).all())
                        case.assertTrue(np.isfinite(body_q).all())
                        case.assertTrue(np.isfinite(body_dq).all())
                        roll, pitch, yaw = _quaternion_to_rpy(pose[3:7])
                        with native._controller_action_lock:
                            controller_input = dict(native.controller_input)
                        samples.append(
                            {
                                "sim_time_s": float(raw["time"]),
                                "x_m": float(pose[0]),
                                "y_m": float(pose[1]),
                                "z_m": float(pose[2]),
                                "yaw_rad": yaw,
                                "tilt_rad": math.hypot(roll, pitch),
                                "body_q": body_q,
                                "body_dq": body_dq,
                                "remote_ly": float(controller_input["remote.ly"]),
                                "cmd_forward": float(native.controller.cmd[0]),
                            }
                        )
                        await asyncio.sleep(interval_s)
                    return samples

                stand = await robot.execute("stand")
                case.assertTrue(stand.ok, stand.detail)
                pre = await sample(1.0)
                start = pre[-1]

                action = dict(REMOTE_ZERO)
                action["remote.ly"] = 0.50
                native.send_action(action)
                moving = await sample(2.0)
                end = moving[-1]
                steady = moving[10:]
                case.assertTrue(steady)
                case.assertLessEqual(max(abs(s["remote_ly"] - 0.50) for s in steady), 1e-12)
                case.assertLessEqual(max(abs(s["cmd_forward"] - 0.50) for s in steady), 1e-6)

                stopped = await robot.execute("stand")
                case.assertTrue(stopped.ok, stopped.detail)
                post = await sample(0.5)
                case.assertTrue(np.allclose(native.controller.cmd, np.zeros(3), atol=0.0))

                initial_yaw = start["yaw_rad"]
                dx = end["x_m"] - start["x_m"]
                dy = end["y_m"] - start["y_m"]
                forward_m = math.cos(initial_yaw) * dx + math.sin(initial_yaw) * dy
                lateral_m = -math.sin(initial_yaw) * dx + math.cos(initial_yaw) * dy
                yaw_delta = _wrap_angle(end["yaw_rad"] - initial_yaw)
                all_samples = pre + moving + post
                body_q = np.asarray([s["body_q"] for s in steady], dtype=np.float64)
                body_dq = np.asarray([s["body_dq"] for s in steady], dtype=np.float64)
                pre_drift = math.hypot(
                    pre[-1]["x_m"] - pre[0]["x_m"],
                    pre[-1]["y_m"] - pre[0]["y_m"],
                )
                moving_sim_time_s = end["sim_time_s"] - moving[0]["sim_time_s"]

                result = {
                    "timestep_profile": label,
                    "sim_dt_s": sim_dt_s,
                    "simulation_loop_dt_s": float(native.control_dt),
                    "sim_frequency_hz": 1.0 / sim_dt_s,
                    "control_dt_s": GROOT_CONTROL_DT_S,
                    "physics_steps_per_policy_update": GROOT_CONTROL_DT_S / sim_dt_s,
                    "forward_displacement_m": forward_m,
                    "mean_forward_mps": forward_m / 2.0,
                    "lateral_displacement_m": lateral_m,
                    "yaw_delta_rad": yaw_delta,
                    "pre_drift_m": pre_drift,
                    "motion_to_pre_drift_ratio": abs(forward_m) / max(pre_drift, 1e-12),
                    "moving_sim_time_s": moving_sim_time_s,
                    "min_height_m": min(s["z_m"] for s in all_samples),
                    "max_tilt_rad": max(s["tilt_rad"] for s in all_samples),
                    "body_q_peak_to_peak_l2_rad": float(np.linalg.norm(np.ptp(body_q, axis=0))),
                    "body_dq_rms_rad_s": float(np.sqrt(np.mean(np.square(body_dq)))),
                }
                numeric = [
                    value for value in result.values() if isinstance(value, (int, float))
                ]
                case.assertTrue(all(math.isfinite(float(value)) for value in numeric))
                case.assertGreater(result["min_height_m"], 0.2)
                case.assertLess(result["max_tilt_rad"], 1.0)
                case.assertGreater(result["body_q_peak_to_peak_l2_rad"], 0.1)
                case.assertGreater(result["moving_sim_time_s"], 1.0)
                print(
                    EPISODE_MEASURED_PREFIX + json.dumps(result, sort_keys=True),
                    flush=True,
                )
                return result
            finally:
                await robot.disconnect()
                print(
                    EPISODE_CLEAN_PREFIX
                    + json.dumps(
                        {"timestep_profile": label, "sim_dt_s": sim_dt_s},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        config_path.write_text(original_config_text, encoding="utf-8")


def _run_episode_subprocess(
    label: str, sim_dt_s: float, original_config_text: str
) -> dict[str, Any]:
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )
    config_path = Path(env["UNITREE_G1_ENV_ROOT"]).resolve() / "config.yaml"

    try:
        completed = subprocess.run(
            [sys.executable, str(test_file), "--episode", label, repr(sim_dt_s)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=EPISODE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        measured_markers = [
            line
            for line in stdout.splitlines()
            if line.startswith(EPISODE_MEASURED_PREFIX)
        ]
        clean_markers = [
            line for line in stdout.splitlines() if line.startswith(EPISODE_CLEAN_PREFIX)
        ]
        raise AssertionError(
            f"timestep episode {label} ({sim_dt_s}s) exceeded {EPISODE_TIMEOUT_S:.0f}s; "
            f"measured_markers={measured_markers[-1:]!r}; "
            f"clean_markers={clean_markers[-1:]!r}; "
            f"stdout_tail={stdout[-4000:]!r}; stderr_tail={stderr[-4000:]!r}"
        ) from exc
    finally:
        # A killed child may not reach its own finally block. Restore the pinned
        # EnvHub config in the parent before launching the next crossover episode.
        config_path.write_text(original_config_text, encoding="utf-8")

    if completed.returncode != 0:
        raise AssertionError(
            f"timestep episode {label} ({sim_dt_s}s) exited {completed.returncode}; "
            f"stdout={completed.stdout[-4000:]!r}; stderr={completed.stderr[-4000:]!r}"
        )

    measured_markers = [
        line[len(EPISODE_MEASURED_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(EPISODE_MEASURED_PREFIX)
    ]
    clean_markers = [
        line[len(EPISODE_CLEAN_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(EPISODE_CLEAN_PREFIX)
    ]
    markers = [
        line[len(EPISODE_RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(EPISODE_RESULT_PREFIX)
    ]
    if len(measured_markers) != 1 or len(clean_markers) != 1 or len(markers) != 1:
        raise AssertionError(
            f"timestep episode {label} ({sim_dt_s}s) markers: "
            f"measured={len(measured_markers)}, clean={len(clean_markers)}, "
            f"result={len(markers)}; stdout={completed.stdout[-4000:]!r}"
        )
    measured = json.loads(measured_markers[0])
    result = json.loads(markers[0])
    if measured != result:
        raise AssertionError(
            f"timestep episode {label} ({sim_dt_s}s) measured/result mismatch: "
            f"measured={measured!r}; result={result!r}"
        )
    return result


class UnitreeG1GrootTimestepCharacterizationTests(TestCase):
    def test_pinned_250hz_vs_nvidia_reference_200hz_physics(self) -> None:
        env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
        config_path = env_root / "config.yaml"
        self.assertTrue(config_path.exists(), config_path)
        original_config_text = config_path.read_text(encoding="utf-8")
        original_config = yaml.safe_load(original_config_text)
        self.assertAlmostEqual(float(original_config["SIMULATE_DT"]), PINNED_ENVHUB_DT_S)

        profiles = {
            "pinned_envhub_250hz": PINNED_ENVHUB_DT_S,
            "nvidia_reference_200hz": NVIDIA_REFERENCE_DT_S,
        }
        # Balanced crossover order reduces the chance that asynchronous
        # DDS/simulator startup phase is mistaken for a timestep effect. Each
        # episode runs in a fresh interpreter so timestep-specific EnvHub module,
        # thread, or DDS state cannot leak into the next profile.
        order = (
            "pinned_envhub_250hz",
            "nvidia_reference_200hz",
            "nvidia_reference_200hz",
            "pinned_envhub_250hz",
        )

        episodes = [
            _run_episode_subprocess(label, profiles[label], original_config_text)
            for label in order
        ]

        aggregates: dict[str, dict[str, float]] = {}
        for label in profiles:
            selected = [e for e in episodes if e["timestep_profile"] == label]
            aggregates[label] = {
                "mean_forward_displacement_m": float(
                    np.mean([e["forward_displacement_m"] for e in selected])
                ),
                "max_abs_forward_displacement_m": max(
                    abs(e["forward_displacement_m"]) for e in selected
                ),
                "mean_pre_drift_m": float(np.mean([e["pre_drift_m"] for e in selected])),
                "mean_motion_to_pre_drift_ratio": float(
                    np.mean([e["motion_to_pre_drift_ratio"] for e in selected])
                ),
                "mean_moving_sim_time_s": float(
                    np.mean([e["moving_sim_time_s"] for e in selected])
                ),
            }

        print(
            "GROOT_TIMESTEP_AB "
            + json.dumps({"episodes": episodes, "aggregates": aggregates}, sort_keys=True),
            flush=True,
        )

        for result in episodes:
            numeric = [value for value in result.values() if isinstance(value, (int, float))]
            self.assertTrue(all(math.isfinite(float(value)) for value in numeric))
            self.assertGreater(result["min_height_m"], 0.2)
            self.assertLess(result["max_tilt_rad"], 1.0)
            self.assertGreater(result["body_q_peak_to_peak_l2_rad"], 0.1)
            self.assertGreater(result["moving_sim_time_s"], 1.0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--episode":
        episode = asyncio.run(_run_episode(sys.argv[2], float(sys.argv[3])))
        print(EPISODE_RESULT_PREFIX + json.dumps(episode, sort_keys=True), flush=True)
    else:
        import unittest

        unittest.main()
