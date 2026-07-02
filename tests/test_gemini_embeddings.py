import os
import unittest
from unittest.mock import call, patch

from whatsapp_rag import embeddings


class GeminiEmbeddingsTest(unittest.TestCase):
    def test_embeds_texts_with_gemini_embedding_model_and_strong_dimension(self):
        response = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True),
            patch.object(embeddings, "embedding", return_value=response) as embed,
        ):
            vectors = embeddings.embed_texts(["first", "second"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        embed.assert_called_once_with(
            model="gemini/gemini-embedding-2",
            input=["first", "second"],
        )

    def test_embeds_query_as_single_vector(self):
        with patch.object(embeddings, "embed_texts", return_value=[[0.1, 0.2]]) as embed:
            vector = embeddings.embed_query("hello")

        self.assertEqual(vector, [0.1, 0.2])
        embed.assert_called_once_with(["hello"])

    def test_splits_embedding_requests_into_gemini_sized_batches(self):
        texts = [f"text {index}" for index in range(101)]
        responses = [
            {"data": [{"embedding": [float(index)]} for index in range(100)]},
            {"data": [{"embedding": [100.0]}]},
        ]

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(embeddings, "embedding", side_effect=responses) as embed,
        ):
            vectors = embeddings.embed_texts(texts)

        self.assertEqual(vectors, [[float(index)] for index in range(101)])
        embed.assert_has_calls(
            [
                call(model="gemini/gemini-embedding-2", input=texts[:100]),
                call(model="gemini/gemini-embedding-2", input=texts[100:]),
            ]
        )


if __name__ == "__main__":
    unittest.main()
