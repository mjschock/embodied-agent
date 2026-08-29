from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import yaml

from embodied_agent.core import Embodiment
from embodied_agent.embodiments import UnitreeG1LeRobot
from embodied_agent.evals.reproducibility import benchmark_reproducibility
from embodied_agent.evals.skill_metrics import SkillProbe, benchmark_robot_skills


MAX_STAND_TILT_RAD = 0.05
RobotScenario = Callable[[UnitreeG1LeRobot], Awaitable[None]]


class UnitreeG1GrootStandTests(TestCase):
    async def _with_pinned_robot(self, scenario: RobotScenario) -> None:
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
                await scenario(robot)
            finally:
                await robot.disconnect()

    async def _sample_stability(
        self,
        robot: UnitreeG1LeRobot,
        *,
        samples: int,
        interval_s: float = 0.02,
    ) -> dict[str, Any]:
        roll_samples: list[float] = []
        pitch_samples: list[float] = []
        final_observation = None

        for _ in range(samples):
            observation = await robot.observe()
            final_observation = observation
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
            await asyncio.sleep(interval_s)

        self.assertIsNotNone(final_observation)
        tilt_samples = [
            math.hypot(roll, pitch)
            for roll, pitch in zip(roll_samples, pitch_samples, strict=True)
        ]
        return {
            "samples": samples,
            "duration_s": samples * interval_s,
            "max_abs_roll_rad": max(abs(v) for v in roll_samples),
            "max_abs_pitch_rad": max(abs(v) for v in pitch_samples),
            "max_tilt_rad": max(tilt_samples),
            "mean_tilt_rad": sum(tilt_samples) / len(tilt_samples),
            "final_roll_rad": roll_samples[-1],
            "final_pitch_rad": pitch_samples[-1],
            "final_tilt_rad": tilt_samples[-1],
            "final_joint_position_rad": dict(final_observation.state["joint_position_rad"]),
            "final_joint_velocity_rad_s": dict(final_observation.state["joint_velocity_rad_s"]),
        }

    def test_semantic_stand_runs_pinned_groot_balance_policy_in_envhub(self) -> None:
        async def scenario(robot: UnitreeG1LeRobot) -> None:
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

            metrics = await self._sample_stability(robot, samples=100)

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

            metrics["controller_output_joints"] = len(controller_output)
            printable = {
                key: value
                for key, value in metrics.items()
                if not key.startswith("final_joint_")
            }
            print("GROOT_STAND_METRICS " + json.dumps(printable, sort_keys=True), flush=True)

            # The first pinned run measured 0.0 rad max/final roll-pitch tilt
            # across 100 samples. Keep a small nonzero behavioral envelope
            # rather than making standing depend on bit-for-bit IMU output.
            self.assertLess(metrics["max_tilt_rad"], MAX_STAND_TILT_RAD)
            self.assertLess(metrics["final_tilt_rad"], MAX_STAND_TILT_RAD)

        asyncio.run(self._with_pinned_robot(scenario))

    def test_semantic_stand_reliability_and_latency(self) -> None:
        async def scenario(robot: UnitreeG1LeRobot) -> None:
            postconditions: list[dict[str, float | int]] = []

            async def before_attempt(
                _robot: Embodiment,
                _probe: SkillProbe,
                _attempt: int,
            ) -> None:
                reset = await robot.execute("reset")
                self.assertTrue(reset.ok, reset.detail)
                await asyncio.sleep(0.25)

            async def after_attempt(
                _robot: Embodiment,
                _probe: SkillProbe,
                attempt: int,
            ) -> None:
                stability = await self._sample_stability(robot, samples=25)
                self.assertLess(stability["max_tilt_rad"], MAX_STAND_TILT_RAD)
                self.assertLess(stability["final_tilt_rad"], MAX_STAND_TILT_RAD)
                postconditions.append(
                    {
                        "attempt": attempt,
                        "max_tilt_rad": float(stability["max_tilt_rad"]),
                        "final_tilt_rad": float(stability["final_tilt_rad"]),
                    }
                )

            result = await benchmark_robot_skills(
                robot,
                (SkillProbe("stand", attempts=5, label="g1-groot-stand"),),
                manage_connection=False,
                before_attempt=before_attempt,
                after_attempt=after_attempt,
            )

            payload = result.to_dict()
            payload["behavioral_postconditions"] = postconditions
            print(
                "GROOT_STAND_SKILL_METRICS " + json.dumps(payload, sort_keys=True),
                flush=True,
            )

            self.assertEqual(result.robot, "g1")
            self.assertEqual(result.backend, "lerobot-unitree-g1")
            self.assertEqual(result.attempt_count, 5)
            self.assertEqual(result.success_count, 5)
            self.assertEqual(result.success_rate, 1.0)
            self.assertEqual(len(postconditions), 5)

            metric = result.metrics[0]
            self.assertEqual(metric.label, "g1-groot-stand")
            self.assertEqual(metric.skill, "stand")
            self.assertEqual(metric.attempts, 5)
            self.assertEqual(metric.successes, 5)
            self.assertEqual(metric.success_rate, 1.0)
            self.assertGreater(metric.mean_latency_ms, 0.0)
            self.assertGreaterEqual(metric.p95_latency_ms, metric.p50_latency_ms)
            self.assertGreaterEqual(metric.max_latency_ms, metric.p95_latency_ms)
            self.assertIsNotNone(metric.successful_mean_latency_ms)
            self.assertTrue(all(sample.ok for sample in metric.samples))
            self.assertTrue(all(sample.error == "" for sample in metric.samples))

        asyncio.run(self._with_pinned_robot(scenario))

    def test_reset_conditioned_stand_is_reproducible(self) -> None:
        async def scenario(robot: UnitreeG1LeRobot) -> None:
            async def run_episode(_: int) -> dict[str, Any]:
                reset = await robot.execute("reset")
                self.assertTrue(reset.ok, reset.detail)
                await asyncio.sleep(0.25)
                stand = await robot.execute("stand")
                self.assertTrue(stand.ok, stand.detail)
                stability = await self._sample_stability(robot, samples=25)
                self.assertLess(stability["max_tilt_rad"], MAX_STAND_TILT_RAD)
                self.assertLess(stability["final_tilt_rad"], MAX_STAND_TILT_RAD)

                native = robot._robot
                self.assertIsNotNone(native)
                controller = native.controller
                self.assertIsNotNone(controller)
                with native._controller_action_lock:
                    controller_output = {
                        key: float(value)
                        for key, value in native.controller_output.items()
                    }

                return {
                    "stand_remote_axes": dict(stand.data["remote_axes"]),
                    "controller_cmd": [float(value) for value in controller.cmd],
                    "controller_output": controller_output,
                    "max_tilt_rad": float(stability["max_tilt_rad"]),
                    "final_tilt_rad": float(stability["final_tilt_rad"]),
                    "final_joint_position_rad": stability["final_joint_position_rad"],
                    "final_joint_velocity_rad_s": stability["final_joint_velocity_rad_s"],
                }

            result = await benchmark_reproducibility(
                run_episode,
                attempts=3,
                atol=1e-8,
                label="unitree-g1-groot-reset-stand",
            )
            print(
                "GROOT_STAND_REPRODUCIBILITY " + json.dumps(result.to_dict(), sort_keys=True),
                flush=True,
            )
            self.assertEqual(result.reproducibility_rate, 1.0, result.to_dict())
            self.assertTrue(all(sample.matches_baseline for sample in result.samples))

        asyncio.run(self._with_pinned_robot(scenario))
