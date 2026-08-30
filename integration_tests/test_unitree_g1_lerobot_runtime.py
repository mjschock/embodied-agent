from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import yaml

from embodied_agent.embodiments import UnitreeG1LeRobot


class UnitreeG1LeRobotRuntimeTests(TestCase):
    def test_default_factory_reconnects_cleanly_against_pinned_envhub(self) -> None:
        async def scenario() -> None:
            env_root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
            env_file = env_root / "env.py"
            config_path = env_root / "config.yaml"
            self.assertTrue(env_file.exists(), env_file)
            self.assertTrue(config_path.exists(), config_path)

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config.update(
                {
                    "ENABLE_ONSCREEN": False,
                    "ENABLE_OFFSCREEN": False,
                    "USE_JOYSTICK": 0,
                    "PRINT_SCENE_INFORMATION": False,
                    # The pinned Unitree/CycloneDDS 0.10.2 stack succeeds on
                    # Python 3.12 with interface auto-detection, but the explicit
                    # Unitree `lo` XML path aborts natively. Keep both ends of the
                    # simulated DDS bridge on the same auto-selected domain-0 path.
                    "INTERFACE": None,
                }
            )
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            # Keep both LeRobot's Hub env loader and the EnvHub module's own asset
            # lookup on the exact local checkout prepared by CI. UnitreeG1.connect()
            # still exercises LeRobot's native make_env/import/normalize path; only
            # network resolution is replaced with the pinned files.
            import huggingface_hub
            import lerobot.envs.utils as env_utils

            robot = UnitreeG1LeRobot(
                name="g1",
                is_simulation=True,
                controller=None,
                gravity_compensation=False,
                simulation_dds_interface=None,
                simulation_publish_images=False,
            )

            with (
                patch.object(env_utils, "hf_hub_download", return_value=str(env_file)),
                patch.object(env_utils, "snapshot_download", return_value=str(env_root)),
                patch.object(huggingface_hub, "snapshot_download", return_value=str(env_root)),
            ):
                # Run two complete default-factory lifecycles on the same adapter
                # instance. Headless image publishing makes the simulation state
                # thread run near its intended 250 Hz cadence, so the semantic reset
                # also exercises the adapter's step/reset serialization boundary.
                for lifecycle in range(2):
                    await robot.connect()
                    try:
                        native = robot._robot
                        self.assertIsNotNone(native)
                        self.assertIsNotNone(native.sim_env)
                        self.assertEqual(len(native.sim_env.camera_configs), 0)

                        observation = await robot.observe()
                        self.assertEqual(observation.state["backend"], "lerobot-unitree-g1")
                        self.assertTrue(observation.state["is_simulation"])
                        self.assertIsNone(observation.state["controller"])
                        self.assertEqual(observation.state["simulation_dds_interface"], "auto")
                        self.assertFalse(observation.state["simulation_publish_images"])
                        self.assertEqual(len(observation.state["joint_position_rad"]), 29)
                        self.assertEqual(len(observation.state["joint_velocity_rad_s"]), 29)
                        self.assertEqual(len(observation.state["joint_torque_est_nm"]), 29)
                        self.assertTrue(
                            all(
                                np.isfinite(value)
                                for values in (
                                    observation.state["joint_position_rad"],
                                    observation.state["joint_velocity_rad_s"],
                                    observation.state["joint_torque_est_nm"],
                                )
                                for value in values.values()
                            )
                        )
                        self.assertTrue(observation.state["imu"])
                        self.assertTrue(
                            all(np.isfinite(v) for v in observation.state["imu"].values())
                        )

                        result = await robot.execute("reset")
                        self.assertTrue(result.ok)
                        self.assertTrue(result.data["is_simulation"])
                        self.assertIsNone(result.data["controller"])
                        self.assertEqual(result.data["simulation_dds_interface"], "auto")
                        self.assertFalse(result.data["simulation_publish_images"])

                        after_reset = await robot.observe()
                        self.assertEqual(len(after_reset.state["joint_position_rad"]), 29)
                        self.assertTrue(
                            all(np.isfinite(v) for v in after_reset.state["joint_position_rad"].values())
                        )
                        print(f"G1_HEADLESS_LIFECYCLE_OK {lifecycle + 1}", flush=True)
                    finally:
                        await robot.disconnect()

        asyncio.run(scenario())
