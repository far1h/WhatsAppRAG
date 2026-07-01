import unittest

from whatsapp_rag.whatsapp import create_conversation_chunks, parse_whatsapp_export_text


class WhatsAppParsingTest(unittest.TestCase):
    def test_parses_messages_and_continuation_lines(self):
        text = "\n".join(
            [
                "[4/8/25, 18:24:04] Kei: mau lari ga",
                "bsk after class",
                "[4/8/25, 18:40:36] mo: gass",
                "[4/8/25, 18:41:25] Kei: trs gajadi?",
            ]
        )

        messages = parse_whatsapp_export_text(text)

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0].sender, "Kei")
        self.assertEqual(messages[0].content, "mau lari ga\nbsk after class")
        self.assertEqual(messages[1].sender, "mo")
        self.assertEqual(messages[0].timestamp.year, 2025)

    def test_groups_messages_by_time_gap(self):
        text = "\n".join(
            [
                "[4/8/25, 18:24:04] Kei: mau lari ga",
                "[4/8/25, 18:40:36] mo: gass",
                "[4/8/25, 20:04:01] Kei: mo discount ESN apa",
            ]
        )

        chunks = create_conversation_chunks(parse_whatsapp_export_text(text), gap_threshold_minutes=30)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].participants, ["Kei", "mo"])
        self.assertIn("Kei (2025-04-08 18:24): mau lari ga", chunks[0].conversation_text)
        self.assertEqual(chunks[0].message_count, 2)


if __name__ == "__main__":
    unittest.main()
