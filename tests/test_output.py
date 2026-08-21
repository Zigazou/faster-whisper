import json
import tempfile
import unittest
from pathlib import Path

from chunked_whisper.models import SegmentResult, TranscriptionResult
from chunked_whisper.output import format_timestamp, render_srt, write_outputs


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.result = TranscriptionResult(
            source="recording.wav",
            language="en",
            language_probability=0.99,
            duration=1.25,
            model="large-v3",
            segments=[SegmentResult(0.0, 1.25, "Hello world.")],
        )

    def test_timestamp_rounds_across_boundaries(self):
        self.assertEqual(format_timestamp(3_661.9996), "01:01:02,000")

    def test_srt_rendering(self):
        self.assertEqual(
            render_srt(self.result.segments),
            "1\n00:00:00,000 --> 00:00:01,250\nHello world.\n",
        )

    def test_writes_all_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = write_outputs(
                self.result, Path("recording.wav"), output_dir, "all"
            )

            self.assertEqual(
                {path.suffix for path in paths}, {".txt", ".srt", ".json"}
            )
            self.assertEqual(
                (output_dir / "recording.txt").read_text(encoding="utf-8"),
                "Hello world.\n",
            )
            payload = json.loads(
                (output_dir / "recording.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["text"], "Hello world.")

    def test_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.txt"
            audio_path.touch()

            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_outputs(self.result, audio_path, None, "txt")


if __name__ == "__main__":
    unittest.main()
