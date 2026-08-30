from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import yaml

from embodied_agent.embodiments import UnitreeG1LeRobot


MEASURE_WINDOW_S = 2.0
SETTLE_WINDOW_S = 0.5
EPISODE_TIMEOUT_S = 45.0
EPISODE_RESULT_PREFIX = "G1_HEADLESS_THROUGHPUT_EPISODE "
AGGREGATE_RESULT_PREFIX = "G1_HEADLESS_THROUGHPUT_AB "


async def _run_episode(label: str, publish_images: bool) -> dict[str, Any]:
    case = TestCase()
    env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
    env_file = env_root / "env.py"
    config_path = env_root / "config.yaml"
    case.assertTrue(env_file.exists(), env_file)
    case.assertTrue(config_path.exists(), config_path)

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

        robot = UnitreeG1LeRobot(
            name="g1",
            is_simulation=True,
            controller=None,
            gravity_compensation=False,
            simulation_dds_interface=None,
            simulation_publish_images=publish_images,
        )

        with (
            patch.object(env_utils, "hf_hub_download", return_value=str(env_file)),
            patch.object(env_utils, "snapshot_download", return_value=str(env_root)),
            patch.object(huggingface_hub, "snapshot_download", return_value=str(env_root)),
        ):
            await robot.connect()
            try:
                native = robot._robot
                case.assertIsNotNone(native)
                case.assertIsNotNone(native.sim_env)
                camera_count = len(native.sim_env.camera_configs)
                if publish_images:
                    case.assertGreater(camera_count, 0)
                else:
                    case.assertEqual(camera_count, 0)

                observation = await robot.observe()
                case.assertEqual(observation.state["simulation_publish_images"], publish_images)

                inner_env = native.sim_env.sim_env
                image_publisher = getattr(inner_env, "image_publish_process", None)
                image_process = getattr(image_publisher, "process", None)
                image_process_alive = bool(
                    image_process is not None and image_process.is_alive()
                )
                if publish_images:
                    case.assertTrue(image_process_alive)
                else:
                    case.assertFalse(image_process_alive)

                await asyncio.sleep(SETTLE_WINDOW_S)
                start_sim_time_s = float(inner_env.prepare_obs()["time"])
                start_wall = time.perf_counter()
                await asyncio.sleep(MEASURE_WINDOW_S)
                wall_time_s = time.perf_counter() - start_wall
                end_sim_time_s = float(inner_env.prepare_obs()["time"])
                sim_time_s = end_sim_time_s - start_sim_time_s
                realtime_factor = sim_time_s / wall_time_s

                result = {
                    "profile": label,
                    "publish_images": publish_images,
                    "camera_count": camera_count,
                    "image_process_alive": image_process_alive,
                    "wall_time_s": wall_time_s,
                    "sim_time_s": sim_time_s,
                    "realtime_factor": realtime_factor,
                }
                numeric = [
                    value for value in result.values() if isinstance(value, (int, float))
                ]
                case.assertTrue(all(math.isfinite(float(value)) for value in numeric))
                case.assertGreater(wall_time_s, 1.8)
                case.assertGreater(sim_time_s, 0.0)
                return result
            finally:
                await robot.disconnect()
    finally:
        config_path.write_text(original_config_text, encoding="utf-8")


def _run_episode_subprocess(
    label: str,
    publish_images: bool,
    original_config_text: str,
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
            [
                sys.executable,
                str(test_file),
                "--episode",
                label,
                "true" if publish_images else "false",
            ],
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
        raise AssertionError(
            f"headless throughput episode {label} exceeded {EPISODE_TIMEOUT_S:.0f}s; "
            f"stdout_tail={stdout[-4000:]!r}; stderr_tail={stderr[-4000:]!r}"
        ) from exc
    finally:
        config_path.write_text(original_config_text, encoding="utf-8")

    if completed.returncode != 0:
        raise AssertionError(
            f"headless throughput episode {label} exited {completed.returncode}; "
            f"stdout={completed.stdout[-4000:]!r}; stderr={completed.stderr[-4000:]!r}"
        )

    matches = [
        line[len(EPISODE_RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(EPISODE_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"headless throughput episode {label} expected one result marker, "
            f"found {len(matches)}; stdout={completed.stdout[-4000:]!r}"
        )
    return json.loads(matches[0])


class UnitreeG1HeadlessThroughputTests(TestCase):
    def test_camera_publishing_throttles_control_only_simulation(self) -> None:
        env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
        config_path = env_root / "config.yaml"
        self.assertTrue(config_path.exists(), config_path)
        original_config_text = config_path.read_text(encoding="utf-8")

        profiles = {
            "camera_on": True,
            "camera_off": False,
        }
        order = ("camera_on", "camera_off", "camera_off", "camera_on")
        episodes = [
            _run_episode_subprocess(label, profiles[label], original_config_text)
            for label in order
        ]

        aggregates: dict[str, dict[str, float]] = {}
        for label in profiles:
            selected = [episode for episode in episodes if episode["profile"] == label]
            aggregates[label] = {
                "mean_sim_time_s": float(
                    np.mean([episode["sim_time_s"] for episode in selected])
                ),
                "mean_wall_time_s": float(
                    np.mean([episode["wall_time_s"] for episode in selected])
                ),
                "mean_realtime_factor": float(
                    np.mean([episode["realtime_factor"] for episode in selected])
                ),
            }

        camera_on = aggregates["camera_on"]
        camera_off = aggregates["camera_off"]
        speedup = camera_off["mean_sim_time_s"] / camera_on["mean_sim_time_s"]
        result = {
            "episodes": episodes,
            "aggregates": aggregates,
            "camera_off_vs_on_sim_time_speedup": speedup,
        }
        print(AGGREGATE_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)

        output_path = os.environ.get("G1_HEADLESS_THROUGHPUT_RESULT")
        if output_path:
            Path(output_path).write_text(
                json.dumps(result, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        for episode in episodes:
            self.assertGreater(episode["sim_time_s"], 0.0)
            if episode["publish_images"]:
                self.assertGreater(episode["camera_count"], 0)
                self.assertTrue(episode["image_process_alive"])
            else:
                self.assertEqual(episode["camera_count"], 0)
                self.assertFalse(episode["image_process_alive"])
                self.assertGreater(episode["sim_time_s"], 1.5)

        self.assertGreater(speedup, 5.0)
        self.assertGreater(camera_off["mean_realtime_factor"], 0.75)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--episode":
        publish_images = {"true": True, "false": False}[sys.argv[3].lower()]
        episode = asyncio.run(_run_episode(sys.argv[2], publish_images))
        print(EPISODE_RESULT_PREFIX + json.dumps(episode, sort_keys=True), flush=True)
    else:
        import unittest

        unittest.main()
