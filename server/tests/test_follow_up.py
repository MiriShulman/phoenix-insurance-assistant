import unittest

from main import contextualize_follow_up


class FollowUpTests(unittest.TestCase):
    def test_short_follow_up_uses_previous_user_question(self):
        history = [
            {"role": "user", "content": "מה כולל ביטוח חובה לרכב?"},
            {"role": "assistant", "content": "ביטוח חובה מכסה נזקי גוף."},
        ]
        result = contextualize_follow_up("ומה לגבי זה?", history)
        self.assertIn("מה כולל ביטוח חובה לרכב?", result)
        self.assertIn("ומה לגבי זה?", result)

    def test_full_question_is_not_rewritten(self):
        question = "מה כולל ביטוח יומי לרכב?"
        result = contextualize_follow_up(
            question,
            [{"role": "user", "content": "מה כולל ביטוח חובה לרכב?"}],
        )
        self.assertEqual(result, question)


if __name__ == "__main__":
    unittest.main()
