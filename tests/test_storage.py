import tempfile
import unittest
from pathlib import Path

from storage import Store, TranslationPair


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "bot.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_glossary_isolated_by_server_and_direction(self):
        self.store.set_glossary_term(10, "SkyBlock", "スカイブロック", "en", "ja")
        self.store.set_glossary_term(11, "SkyBlock", "天空ブロック", "en", "ja")
        self.assertEqual(self.store.get_glossary(10, "en", "ja"), [("SkyBlock", "スカイブロック")])
        self.assertEqual(self.store.get_glossary(10, "ja", "en"), [])
        self.assertTrue(self.store.remove_glossary_term(10, "skyblock", "en", "ja"))
        self.assertEqual(self.store.get_glossary(11, "en", "ja"), [("SkyBlock", "天空ブロック")])

    def test_glossary_normalizes_and_rejects_empty_languages(self):
        self.store.set_glossary_term(10, "hello", "こんにちは", " en ", " ja ")
        self.assertEqual(self.store.get_glossary(10, "en", "ja"), [("hello", "こんにちは")])
        self.assertEqual(self.store.get_glossary(10, " en ", " ja "), [("hello", "こんにちは")])
        self.assertTrue(self.store.remove_glossary_term(10, "hello", " en ", " ja "))
        self.assertEqual(self.store.get_glossary(10, "en", "ja"), [])
        with self.assertRaises(ValueError):
            self.store.set_glossary_term(10, "hello", "こんにちは", "  ", "ja")
        with self.assertRaises(ValueError):
            self.store.set_glossary_term(10, "hello", "こんにちは", "en", "\t")

    def test_pairs_reverse_language_and_persist(self):
        pair = TranslationPair(10, 20, 21, "ja", "en")
        self.store.add_pair(pair)
        self.assertEqual(self.store.get_pair(10, 20), pair)
        self.assertEqual(self.store.get_pair(10, 21), TranslationPair(10, 21, 20, "en", "ja"))
        self.assertIsNone(self.store.get_pair(11, 20))

    def test_message_and_thread_links_are_bidirectional(self):
        self.store.save_message_link(10, 20, 101, 21, 201)
        self.assertEqual(self.store.get_linked_message(10, 20, 101), (21, 201))
        self.assertEqual(self.store.get_linked_message(10, 21, 201), (20, 101))
        self.store.save_thread_link(10, 30, 40)
        self.assertEqual(self.store.get_linked_thread(10, 30), 40)
        self.assertEqual(self.store.get_linked_thread(10, 40), 30)

    def test_rejects_invalid_pair(self):
        with self.assertRaises(ValueError):
            self.store.add_pair(TranslationPair(10, 20, 20, "ja", "en"))
        with self.assertRaises(ValueError):
            self.store.add_pair(TranslationPair(10, 20, 21, "ja", "ja"))
        with self.assertRaises(ValueError):
            self.store.add_pair(TranslationPair(10, 20, 21, " ", "en"))
        with self.assertRaises(ValueError):
            self.store.add_pair(TranslationPair(10, 20, 21, "ja ", " ja"))

    def test_pair_channels_cannot_overlap(self):
        self.store.add_pair(TranslationPair(10, 20, 21, "ja", "en"))
        with self.assertRaises(ValueError):
            self.store.add_pair(TranslationPair(10, 21, 22, "en", "fr"))
        with self.assertRaises(ValueError):
            self.store.add_pair(TranslationPair(10, 22, 20, "fr", "ja"))
        self.store.add_pair(TranslationPair(11, 21, 22, "en", "fr"))

    def test_source_only_lookup_does_not_reverse_link(self):
        self.store.save_message_link(10, 20, 101, 21, 201)
        self.assertEqual(self.store.get_forwarded_message(10, 20, 101), (21, 201))
        self.assertIsNone(self.store.get_forwarded_message(10, 21, 201))


if __name__ == "__main__":
    unittest.main()
