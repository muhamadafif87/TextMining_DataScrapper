import unittest

from clean_data import attach_root_tweet_text


class AttachRootTweetTextTests(unittest.TestCase):
    def test_attaches_context_for_reply(self):
        rows_by_id = {
            "root-1": {
                "tweet_id": "root-1",
                "raw_text": "Teks tweet root",
                "is_reply": "False",
                "root_tweet_id": "",
                "root_tweet_text": "",
            },
            "reply-1": {
                "tweet_id": "reply-1",
                "raw_text": "Teks reply",
                "is_reply": "True",
                "root_tweet_id": "root-1",
                "root_tweet_text": "",
            },
        }

        attach_root_tweet_text(rows_by_id)

        self.assertEqual(rows_by_id["reply-1"]["root_tweet_text"], "Teks tweet root")
        self.assertEqual(rows_by_id["reply-1"]["raw_text"], "Teks reply")

    def test_leaves_context_empty_when_root_is_missing(self):
        rows_by_id = {
            "reply-1": {
                "tweet_id": "reply-1",
                "raw_text": "Teks reply",
                "is_reply": "True",
                "root_tweet_id": "unknown-root",
                "root_tweet_text": "",
            },
        }

        attach_root_tweet_text(rows_by_id)

        self.assertEqual(rows_by_id["reply-1"]["root_tweet_text"], "")


if __name__ == "__main__":
    unittest.main()
