from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml


PINNED_ENV_REVISION = "a38dc86"


class UnitreeG1EnvHubPhysicsTests(unittest.TestCase):
    def test_pinned_env_resets_and_steps_headless(self) -> None:
        root = Path(os.environ["UNITREE_G1_ENV_ROOT"]).resolve()
        self.assertTrue((root / "env.py").exists(), root)
        self.assertTrue((root / "config.yaml").exists(), root)

        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        self.assertTrue(
            head.startswith(PINNED_ENV_REVISION),
            f"expected pinned EnvHub revision {PINNED_ENV_REVISION}, got {head}",
        )

        config_path = root / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config.update(
            {
                "ENABLE_ONSCREEN": False,
                "ENABLE_OFFSCREEN": False,
                "USE_JOYSTICK": 0,
                "PRINT_SCENE_INFORMATION": False,
            }
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("pinned_unitree_g1_env", root / "env.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)

        # env.py currently calls snapshot_download without a revision at import time.
        # Redirect that call to the already pinned local checkout so the smoke test
        # never executes moving remote code.
        with patch("huggingface_hub.snapshot_download", return_value=str(root)):
            spec.loader.exec_module(module)

        env = module.make_env(
            n_envs=1,
            use_async_envs=False,
            publish_images=False,
            cameras=[],
        )
        try:
            self.assertEqual(env.action_space.shape, (29,))

            # The pinned EnvHub revision declares 29 * 3 + 10 = 97 values, but its
            # current simulator state contains 30 q/dq/tau entries and therefore
            # reset()/step() return 100 values. Keep both sides explicit so a future
            # upstream correction becomes a visible contract change rather than a
            # silent tolerance in our physics gate.
            self.assertEqual(env.observation_space.shape, (97,))

            observation, info = env.reset(seed=123)
            raw = env.sim_env.prepare_obs()
            component_lengths = {
                name: int(np.asarray(raw[name]).size)
                for name in (
                    "body_q",
                    "body_dq",
                    "body_tau_est",
                    "floating_base_pose",
                    "floating_base_vel",
                    "floating_base_acc",
                )
            }
            print(f"Pinned G1 observation component lengths: {component_lengths}")
            self.assertEqual(
                component_lengths,
                {
                    "body_q": 30,
                    "body_dq": 30,
                    "body_tau_est": 30,
                    "floating_base_pose": 7,
                    "floating_base_vel": 6,
                    "floating_base_acc": 3,
                },
            )
            self.assertEqual(observation.shape, (100,))
            self.assertTrue(np.isfinite(observation).all())
            self.assertEqual(info, {})

            zero_action = np.zeros(29, dtype=np.float32)
            for _ in range(5):
                observation, reward, terminated, truncated, info = env.step(zero_action)
                self.assertEqual(observation.shape, (100,))
                self.assertTrue(np.isfinite(observation).all())
                self.assertEqual(reward, 0.0)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info, {})
        finally:
            env.close()
            if str(root) in sys.path:
                sys.path.remove(str(root))


if __name__ == "__main__":
    unittest.main()
