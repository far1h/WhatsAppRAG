import unittest
from unittest.mock import Mock, patch

from whatsapp_rag import ingest


class IngestPoolingTest(unittest.TestCase):
    def test_create_chunks_processes_documents_with_worker_pool_and_tqdm(self):
        documents = [{"text": "first"}, {"text": "second"}]
        first = ingest.Result(page_content="first result", metadata={"source": "first"})
        second = ingest.Result(page_content="second result", metadata={"source": "second"})

        pool = Mock()
        pool.imap_unordered.return_value = [[first], [second]]
        pool_context = Mock()
        pool_context.__enter__ = Mock(return_value=pool)
        pool_context.__exit__ = Mock(return_value=None)

        with (
            patch.object(ingest, "Pool", return_value=pool_context) as pool_class,
            patch.object(ingest, "tqdm", side_effect=lambda results, total: results) as progress,
        ):
            chunks = ingest.create_chunks(documents)

        pool_class.assert_called_once_with(processes=ingest.WORKERS)
        pool.imap_unordered.assert_called_once_with(ingest.process_document, documents)
        progress.assert_called_once_with(pool.imap_unordered.return_value, total=len(documents))
        self.assertEqual(chunks, [first, second])


if __name__ == "__main__":
    unittest.main()
