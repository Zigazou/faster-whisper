import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from chunked_whisper.transcription import TranscriptionConfig, transcribe_audio


class TranscriptionConfigTests(unittest.TestCase):
    def test_accepts_default_configuration(self):
        TranscriptionConfig().validate()

    def test_overlap_must_be_shorter_than_chunk(self):
        config = TranscriptionConfig(chunk_duration=5.0, overlap=5.0)

        with self.assertRaisesRegex(ValueError, "shorter"):
            config.validate()

    def test_beam_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            TranscriptionConfig(beam_size=0).validate()


class TranscribeAudioTests(unittest.TestCase):
    @patch("chunked_whisper.transcription.decode_audio")
    def test_detects_language_once_and_keeps_boundary_segment_once(
        self, decode_audio
    ):
        decode_audio.return_value = [0.0] * 32_000
        boundary_segment = SimpleNamespace(
            start=0.9,
            end=1.1,
            text=" Boundary speech ",
            words=[],
        )
        shifted_boundary_segment = SimpleNamespace(
            start=0.15,
            end=0.35,
            text=" Boundary speech ",
            words=[],
        )
        model = Mock()
        model.transcribe.side_effect = [
            (
                [boundary_segment],
                SimpleNamespace(language="en", language_probability=0.9),
            ),
            (
                [shifted_boundary_segment],
                SimpleNamespace(language="en", language_probability=0.9),
            ),
        ]

        result = transcribe_audio(
            model,
            Path("recording.wav"),
            TranscriptionConfig(chunk_duration=1.0, overlap=0.25),
        )

        self.assertEqual(result.language, "en")
        self.assertEqual(result.language_probability, 0.9)
        self.assertEqual(
            [segment.text for segment in result.segments], ["Boundary speech"]
        )
        self.assertIsNone(model.transcribe.call_args_list[0].kwargs["language"])
        self.assertEqual(model.transcribe.call_args_list[1].kwargs["language"], "en")


if __name__ == "__main__":
    unittest.main()
