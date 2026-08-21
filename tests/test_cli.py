import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chunked_whisper.cli import main
from chunked_whisper.models import SegmentResult, TranscriptionResult


class CliTests(unittest.TestCase):
    def test_diarize_runs_pipeline_and_enables_word_timestamps(self):
        transcription = TranscriptionResult(
            source="recording.wav",
            language="en",
            language_probability=0.9,
            duration=1.0,
            model="large-v3",
            segments=[SegmentResult(0.0, 1.0, "Hello")],
        )
        aligned_segments = [
            SegmentResult(0.0, 1.0, "Hello", speaker="SPEAKER_00")
        ]
        original_segments = transcription.segments
        diarization_pipeline = Mock()

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            audio_path.touch()
            with (
                patch("chunked_whisper.cli.WhisperModel") as whisper_model,
                patch(
                    "chunked_whisper.cli.load_diarization_pipeline",
                    return_value=diarization_pipeline,
                ) as load_pipeline,
                patch(
                    "chunked_whisper.cli.transcribe_audio",
                    return_value=transcription,
                ) as transcribe,
                patch(
                    "chunked_whisper.cli.diarize_audio", return_value=[]
                ) as diarize,
                patch(
                    "chunked_whisper.cli.assign_speakers",
                    return_value=aligned_segments,
                ) as assign,
                patch("chunked_whisper.cli.write_outputs", return_value=[]),
            ):
                exit_code = main(
                    [
                        str(audio_path),
                        "--diarize",
                        "--hf-token",
                        "token",
                        "--num-speakers",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        whisper_model.assert_called_once()
        load_pipeline.assert_called_once_with(
            "pyannote/speaker-diarization-community-1",
            token="token",
            device="cpu",
        )
        self.assertTrue(transcribe.call_args.args[2].word_timestamps)
        diarize.assert_called_once_with(
            diarization_pipeline,
            audio_path,
            num_speakers=2,
            min_speakers=None,
            max_speakers=None,
        )
        assign.assert_called_once_with(original_segments, [])
        self.assertEqual(transcription.segments, aligned_segments)

    def test_rejects_conflicting_speaker_count_options(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "recording.wav"
            audio_path.touch()
            exit_code = main(
                [
                    str(audio_path),
                    "--diarize",
                    "--num-speakers",
                    "2",
                    "--min-speakers",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
