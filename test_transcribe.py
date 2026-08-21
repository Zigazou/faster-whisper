import tempfile
import unittest
from pathlib import Path

from transcribe import (
    SegmentResult,
    TranscriptionResult,
    chunk_windows,
    format_timestamp,
    render_srt,
    write_outputs,
)


class ChunkWindowsTests(unittest.TestCase):
    def test_windows_have_overlapping_context_and_adjacent_cores(self):
        windows = list(chunk_windows(1_000, 400, 50))

        self.assertEqual(
            windows,
            [(0, 450, 0, 400), (350, 850, 400, 800), (750, 1_000, 800, 1_000)],
        )

    def test_empty_audio_has_no_windows(self):
        self.assertEqual(list(chunk_windows(0, 400, 50)), [])


class OutputTests(unittest.TestCase):
    def test_timestamp_rounds_across_boundaries(self):
        self.assertEqual(format_timestamp(3_661.9996), "01:01:02,000")

    def test_srt_rendering(self):
        segments = [SegmentResult(0.0, 1.25, "Hello world.")]

        self.assertEqual(
            render_srt(segments),
            "1\n00:00:00,000 --> 00:00:01,250\nHello world.\n",
        )

    def test_writes_all_formats(self):
        result = TranscriptionResult(
            source="recording.wav",
            language="en",
            language_probability=0.99,
            duration=1.25,
            model="large-v3",
            segments=[SegmentResult(0.0, 1.25, "Hello world.")],
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(
                result, Path("recording.wav"), Path(directory), "all"
            )

            self.assertEqual({path.suffix for path in paths}, {".txt", ".srt", ".json"})
            self.assertEqual((Path(directory) / "recording.txt").read_text(), "Hello world.\n")


if __name__ == "__main__":
    unittest.main()
