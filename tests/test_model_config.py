import os
import unittest
from unittest.mock import patch

from whatsapp_rag import model_config


class ModelConfigTest(unittest.TestCase):
    def test_uses_flash_for_summary_and_plus_for_answers_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_config.summary_model(), "qwen3.6-flash")
            self.assertEqual(model_config.query_model(), "qwen3.6-flash")
            self.assertEqual(model_config.answer_model(), "qwen3.7-plus")

    def test_normalizes_dashscope_provider_prefix_for_openai_client(self):
        with patch.dict(
            os.environ,
            {
                "SUMMARY_MODEL": "dashscope/qwen3.6-flash",
                "ANSWER_MODEL": "dashscope/qwen3.7-plus",
            },
            clear=True,
        ):
            self.assertEqual(model_config.summary_model(), "qwen3.6-flash")
            self.assertEqual(model_config.answer_model(), "qwen3.7-plus")

    def test_uses_dashscope_compatible_base_url_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                model_config.chat_api_base_url(),
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )


if __name__ == "__main__":
    unittest.main()
