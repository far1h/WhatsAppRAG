from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LiteLLMGeminiConfigTest(unittest.TestCase):
    def test_project_uses_litellm_sdk_for_gemini(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        dependencies = pyproject["project"]["dependencies"]

        self.assertTrue(any(dep.startswith("litellm>=") for dep in dependencies))
        self.assertFalse(any(dep.startswith("google-genai>=") for dep in dependencies))
        self.assertFalse(any(dep.startswith("openai>=") for dep in dependencies))

    def test_chat_modules_use_litellm_without_qwen_defaults(self):
        for module_name in ("answer.py", "ingest.py"):
            with self.subTest(module=module_name):
                source = (ROOT / "whatsapp_rag" / module_name).read_text()

                self.assertIn("from litellm import completion", source)
                self.assertIn("completion(", source)
                self.assertNotIn("make_openai_compatible_client", source)
                self.assertNotIn(".chat.completions.create(", source)
                self.assertNotIn("qwen-plus", source)
                self.assertNotIn("DASHSCOPE", source)


if __name__ == "__main__":
    unittest.main()
