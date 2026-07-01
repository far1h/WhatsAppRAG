import ast
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LiteLLMChatConfigTest(unittest.TestCase):
    def test_project_declares_litellm_dependency(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

        self.assertIn("litellm>=1.0.0", pyproject["project"]["dependencies"])

    def test_chat_modules_use_litellm_completion_with_qwen_plus_default(self):
        for module_name in ("answer.py", "ingest.py"):
            with self.subTest(module=module_name):
                source = (ROOT / "whatsapp_rag" / module_name).read_text()
                tree = ast.parse(source)

                self.assertTrue(imports_litellm_completion(tree))
                self.assertIn('os.getenv("CHAT_MODEL", "dashscope/qwen-plus")', source)
                self.assertIn("completion(", source)


def imports_litellm_completion(tree: ast.Module) -> bool:
    """Return whether a module imports LiteLLM's completion helper."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "litellm":
            return any(alias.name == "completion" for alias in node.names)
    return False


if __name__ == "__main__":
    unittest.main()
