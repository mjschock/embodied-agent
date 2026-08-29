from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path


class SO101LeRobotSourceContractTests(unittest.TestCase):
    def test_pinned_lerobot_061_so101_surface_matches_adapter_assumptions(self) -> None:
        root = Path(os.environ["LEROBOT_SOURCE_ROOT"])
        config_path = root / "src/lerobot/robots/so_follower/config_so_follower.py"
        follower_path = root / "src/lerobot/robots/so_follower/so_follower.py"
        self.assertTrue(config_path.exists(), config_path)
        self.assertTrue(follower_path.exists(), follower_path)

        config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
        follower_tree = ast.parse(follower_path.read_text(encoding="utf-8"))

        assigned_names = {
            target.id
            for node in ast.walk(config_tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("SO101FollowerConfig", assigned_names)

        follower_class = next(
            node
            for node in follower_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SOFollower"
        )
        method_names = {
            node.name
            for node in follower_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for method in ("connect", "disconnect", "get_observation", "send_action"):
            self.assertIn(method, method_names)

        string_constants = {
            node.value
            for node in ast.walk(follower_tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        expected_motors = {
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        }
        self.assertTrue(expected_motors.issubset(string_constants))

        source = follower_path.read_text(encoding="utf-8")
        self.assertIn('return {f"{motor}.pos": float for motor in self.bus.motors}', source)
        self.assertIn('self.bus.sync_read("Present_Position"', source)
        self.assertIn('key.endswith(".pos")', source)


if __name__ == "__main__":
    unittest.main()
