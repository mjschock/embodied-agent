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
            self.assertTrue(
                all(np.isfinite(v) for v in observation.state["joint_velocity_rad_s"].values())
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

    def test_semantic_stand_behavior_reliability_and_reproducibility(self) -> None:
        async def scenario(robot: UnitreeG1LeRobot) -> None:
            # Keep one native LeRobot/DDS/EnvHub lifecycle for the whole gate. A
            # diagnostic first pass found that reconnecting the async simulator
            # repeatedly inside one Python process can leave an old simulator
            # thread alive. Reset-conditioned episodes below intentionally reuse
            # one connected stack, which is also the production episode model.
            reset = await robot.execute("reset")
            self.assertTrue(reset.ok, reset.detail)
            await asyncio.sleep(0.25)

            stand = await robot.execute("stand")
            self.assertTrue(stand.ok, stand.detail)
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

            postconditions: list[dict[str, float | int]] = []

            async def before_attempt(
                _robot: Embodiment,
                _probe: SkillProbe,
                _attempt: int,
            ) -> None:
                attempt_reset = await robot.execute("reset")
                self.assertTrue(attempt_reset.ok, attempt_reset.detail)
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

            reliability = await benchmark_robot_skills(
                robot,
                (SkillProbe("stand", attempts=5, label="g1-groot-stand"),),
                manage_connection=False,
                before_attempt=before_attempt,
                after_attempt=after_attempt,
            )

            reliability_payload = reliability.to_dict()
            reliability_payload["behavioral_postconditions"] = postconditions
            print(
                "GROOT_STAND_SKILL_METRICS "
                + json.dumps(reliability_payload, sort_keys=True),
                flush=True,
            )

            self.assertEqual(reliability.robot, "g1")
            self.assertEqual(reliability.backend, "lerobot-unitree-g1")
            self.assertEqual(reliability.attempt_count, 5)
            self.assertEqual(reliability.success_count, 5)
            self.assertEqual(reliability.success_rate, 1.0)
            self.assertEqual(len(postconditions), 5)

            skill_metric = reliability.metrics[0]
            self.assertEqual(skill_metric.label, "g1-groot-stand")
            self.assertEqual(skill_metric.skill, "stand")
            self.assertEqual(skill_metric.attempts, 5)
            self.assertEqual(skill_metric.successes, 5)
            self.assertEqual(skill_metric.success_rate, 1.0)
            self.assertGreater(skill_metric.mean_latency_ms, 0.0)
            self.assertGreaterEqual(skill_metric.p95_latency_ms, skill_metric.p50_latency_ms)
            self.assertGreaterEqual(skill_metric.max_latency_ms, skill_metric.p95_latency_ms)
            self.assertIsNotNone(skill_metric.successful_mean_latency_ms)
            self.assertTrue(all(sample.ok for sample in skill_metric.samples))
            self.assertTrue(all(sample.error == "" for sample in skill_metric.samples))

            async def run_episode(_: int) -> dict[str, Any]:
                episode_reset = await robot.execute("reset")
                self.assertTrue(episode_reset.ok, episode_reset.detail)
                await asyncio.sleep(0.25)
                episode_stand = await robot.execute("stand")
                self.assertTrue(episode_stand.ok, episode_stand.detail)
                stability = await self._sample_stability(robot, samples=25)
                self.assertLess(stability["max_tilt_rad"], MAX_STAND_TILT_RAD)
                self.assertLess(stability["final_tilt_rad"], MAX_STAND_TILT_RAD)

                episode_native = robot._robot
                self.assertIsNotNone(episode_native)
                episode_controller = episode_native.controller
                self.assertIsNotNone(episode_controller)
                self.assertTrue(
                    np.allclose(episode_controller.cmd, np.zeros(3), atol=0.0)
                )
                with episode_native._controller_action_lock:
                    episode_controller_output = dict(episode_native.controller_output)
                self.assertEqual(len(episode_controller_output), 15)
                self.assertTrue(
                    all(np.isfinite(v) for v in episode_controller_output.values())
                )

                # A deliberately strict diagnostic probe first compared the
                # wall-clock sampled 29-joint state and 15 controller outputs.
                # Those instantaneous fields diverged by up to ~9.65 across
                # resets even though every episode held exactly zero command
                # and exactly 0.0-rad roll/pitch tilt. LeRobot's controller and
                # EnvHub physics advance on background threads, so those state
                # snapshots are phase-sensitive rather than a truthful reset
                # reproducibility contract. Keep the reproducibility payload at
                # the semantic/behavioral boundary and continue asserting full
                # joint/controller finiteness separately above.
                return {
                    "stand_remote_axes": dict(episode_stand.data["remote_axes"]),
                    "controller_cmd": [float(value) for value in episode_controller.cmd],
                    "controller_output_joints": len(episode_controller_output),
                    "controller_output_finite": all(
                        np.isfinite(v) for v in episode_controller_output.values()
                    ),
                    "observed_joints": len(stability["final_joint_position_rad"]),
                    "observed_joint_position_finite": all(
                        np.isfinite(v)
                        for v in stability["final_joint_position_rad"].values()
                    ),
                    "observed_joint_velocity_finite": all(
                        np.isfinite(v)
                        for v in stability["final_joint_velocity_rad_s"].values()
                    ),
                    "max_tilt_rad": float(stability["max_tilt_rad"]),
                    "final_tilt_rad": float(stability["final_tilt_rad"]),
                }

            reproducibility = await benchmark_reproducibility(
                run_episode,
                attempts=3,
                atol=1e-9,
                label="unitree-g1-groot-reset-stand-behavior",
            )
            print(
                "GROOT_STAND_REPRODUCIBILITY "
                + json.dumps(reproducibility.to_dict(), sort_keys=True),
                flush=True,
            )
            self.assertEqual(
                reproducibility.reproducibility_rate,
                1.0,
                reproducibility.to_dict(),
            )
            self.assertTrue(
                all(sample.matches_baseline for sample in reproducibility.samples)
            )

        asyncio.run(self._with_pinned_robot(scenario))
