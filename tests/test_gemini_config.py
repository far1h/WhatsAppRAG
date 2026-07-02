import os
import unittest
from unittest.mock import patch

from whatsapp_rag import model_config


class GeminiConfigTest(unittest.TestCase):
    def test_uses_small_models_for_summary_and_query_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_config.summary_model(), "gemini/gemini-2.5-flash")
            self.assertEqual(model_config.query_model(), "gemini/gemini-2.5-flash")

    def test_uses_frontier_non_flash_model_for_final_answers_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_config.answer_model(), "gemini/gemini-3.5")

    def test_allows_answer_model_override_for_newer_chat_models(self):
        with patch.dict(os.environ, {"ANSWER_MODEL": "gemini/gemini-3.5"}, clear=True):
            self.assertEqual(model_config.answer_model(), "gemini/gemini-3.5")

    def test_uses_strong_gemini_embedding_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_config.embedding_model(), "gemini/gemini-embedding-2")


if __name__ == "__main__":
    unittest.main()
