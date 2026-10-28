import unittest

from rag_retrieval import LocalTfidfRetriever, split_into_chunks


class RagRetrievalTests(unittest.TestCase):
    def test_long_text_is_split_into_bounded_chunks(self):
        text = "משפט ארוך עם מידע חשוב. " * 250
        chunks = split_into_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 1100 for chunk in chunks))

    def test_retriever_prefers_relevant_page(self):
        pages = [
            {
                "title": "הגשת תביעת רכב",
                "url": "https://example.test/car-claim",
                "text": "כדי להגיש תביעת רכב יש למלא טופס תביעה ולצרף מסמכים.",
            },
            {
                "title": "נזקי מים וצנרת",
                "url": "https://example.test/water-damage",
                "text": "במקרה של נזילה בצנרת ונזקי מים בבית, יש לבדוק את תנאי הכיסוי והחריגים.",
            },
        ]
        retriever = LocalTfidfRetriever()
        results = retriever.search("נזילה בצנרת ונזקי מים בבית", pages, top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0]["url"], "https://example.test/water-damage")
        self.assertTrue(results[0]["text"])


if __name__ == "__main__":
    unittest.main()
