from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path


class UnitreeG1LeRobotSourceContractTests(unittest.TestCase):
    def test_pinned_lerobot_061_g1_surface_matches_adapter_assumptions(self) -> None:
        root = Path(os.environ["LEROBOT_SOURCE_ROOT"])
        config_path = root / "src/lerobot/robots/unitree_g1/config_unitree_g1.py"
        robot_path = root / "src/lerobot/robots/unitree_g1/unitree_g1.py"
        utils_path = root / "src/lerobot/robots/unitree_g1/g1_utils.py"
        groot_path = root / "src/lerobot/robots/unitree_g1/gr00t_locomotion.py"
        for path in (config_path, robot_path, utils_path, groot_path):
            self.assertTrue(path.exists(), path)

        config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
        robot_tree = ast.parse(robot_path.read_text(encoding="utf-8"))
        utils_tree = ast.parse(utils_path.read_text(encoding="utf-8"))

        config_class = next(
            node
            for node in config_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "UnitreeG1Config"
        )
        annotated_fields = {
            node.target.id
            for node in config_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertTrue(
            {"default_positions", "control_dt", "is_simulation", "robot_ip", "controller"}.issubset(
                annotated_fields
            )
        )

        robot_class = next(
            node
            for node in robot_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "UnitreeG1"
        )
        method_names = {
            node.name
            for node in robot_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for method in ("connect", "disconnect", "get_observation", "send_action", "reset"):
            self.assertIn(method, method_names)

        joint_enum = next(
            node
            for node in utils_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "G1_29_JointIndex"
        )
        joint_members = [
            node.targets[0].id
            for node in joint_enum.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ]
        self.assertEqual(len(joint_members), 29)

        utils_source = utils_path.read_text(encoding="utf-8")
        robot_source = robot_path.read_text(encoding="utf-8")
        groot_source = groot_path.read_text(encoding="utf-8")
        self.assertIn('REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")', utils_source)
        self.assertIn('make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)', robot_source)
        self.assertIn('obs[f"{name}.q"]', robot_source)
        self.assertIn('obs[f"{name}.dq"]', robot_source)
        self.assertIn('obs[f"{name}.tau"]', robot_source)
        self.assertIn("self.controller.reset()", robot_source)

        # The adapter's semantic stand contract depends on zero controller axes selecting
        # the GR00T balance policy rather than a locomotion command.
        self.assertIn("cmd_magnitude = np.linalg.norm(self.cmd)", groot_source)
        self.assertIn("self.policy_balance if cmd_magnitude < 0.05 else self.policy_walk", groot_source)
        self.assertIn("self.cmd[0] = ly", groot_source)
        self.assertIn("self.cmd[1] = -lx", groot_source)
        self.assertIn("self.cmd[2] = -rx", groot_source)


if __name__ == "__main__":
    unittest.main()
