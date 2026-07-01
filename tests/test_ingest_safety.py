import unittest

from whatsapp_rag.safety import (
    is_data_inspection_error,
    make_fallback_summary,
    sanitize_for_model,
    should_retry_model_error,
)


class IngestSafetyTest(unittest.TestCase):
    def test_sanitizes_sensitive_terms_without_losing_chat_metadata(self):
        text = "Kei (2025-04-08 18:24): NSFW private detail"

        sanitized = sanitize_for_model(text)

        self.assertIn("Kei (2025-04-08 18:24):", sanitized)
        self.assertIn("[sensitive term]", sanitized)
        self.assertNotIn("nsfw", sanitized.lower())

    def test_data_inspection_errors_are_not_retried(self):
        error = RuntimeError(
            "DataInspectionFailed: Input text data may contain inappropriate content"
        )

        self.assertTrue(is_data_inspection_error(error))
        self.assertFalse(should_retry_model_error(error))

    def test_fallback_summary_preserves_retrieval_metadata(self):
        summary = make_fallback_summary(
            {
                "participants": "Kei, mo",
                "date": "2025-04-08",
                "start_time": "2025-04-08T18:24:04",
                "end_time": "2025-04-08T18:41:25",
            }
        )

        self.assertIn("2025-04-08", summary["headline"])
        self.assertIn("Kei, mo", summary["headline"])
        self.assertIn("content inspection", summary["summary"])


if __name__ == "__main__":
    unittest.main()
