import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSharedB5Policy(unittest.TestCase):
    def test_transition_has_one_definition(self):
        definitions = []
        for filename in ("2-risk_label.py", "3-train_evaluation.py", "b5_policy.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            count = sum(isinstance(node, ast.FunctionDef) and node.name == "b5_transition"
                        for node in ast.walk(tree))
            definitions.extend([filename] * count)
        self.assertEqual(["b5_policy.py"], definitions)

    def test_both_consumers_import_shared_transition(self):
        for filename in ("2-risk_label.py", "3-train_evaluation.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                       and node.module == "b5_policy"]
            self.assertEqual(1, len(imports), filename)
            self.assertIn("b5_transition", {alias.name for alias in imports[0].names})


if __name__ == "__main__":
    unittest.main()
