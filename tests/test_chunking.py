import unittest

from chunked_whisper.chunking import chunk_windows


class ChunkWindowsTests(unittest.TestCase):
    def test_windows_have_overlapping_context_and_adjacent_cores(self):
        windows = list(chunk_windows(1_000, 400, 50))

        self.assertEqual(
            windows,
            [(0, 450, 0, 400), (350, 850, 400, 800), (750, 1_000, 800, 1_000)],
        )

    def test_empty_audio_has_no_windows(self):
        self.assertEqual(list(chunk_windows(0, 400, 50)), [])

    def test_rejects_invalid_window_sizes(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            list(chunk_windows(1_000, 0, 50))
        with self.assertRaisesRegex(ValueError, "negative"):
            list(chunk_windows(1_000, 400, -1))


if __name__ == "__main__":
    unittest.main()

