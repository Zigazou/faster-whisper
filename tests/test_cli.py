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

    def test_alignment_runs_before_speaker_assignment(self):
        result = TranscriptionResult(
            "recording.wav", "fr", 1.0, 22.0, "large-v3",
            [SegmentResult(16.98, 22.0, "Le diable")],
        )
        refined = [SegmentResult(17.5, 22.0, "Le diable")]
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "recording.wav"
            audio.touch()
            with (
                patch("chunked_whisper.cli.WhisperModel"),
                patch("chunked_whisper.cli.load_diarization_pipeline"),
                patch("chunked_whisper.cli.transcribe_audio", return_value=result) as transcribe,
                patch("chunked_whisper.cli.WordAligner") as aligner,
                patch("chunked_whisper.cli.diarize_audio", return_value=[]) as diarize,
                patch("chunked_whisper.cli.assign_speakers", return_value=refined) as assign,
                patch("chunked_whisper.cli.write_outputs", return_value=[]) as write,
            ):
                original = result.segments
                aligner.return_value.align.return_value = refined
                def check_aligned(*args, **kwargs):
                    self.assertIs(result.segments, refined)
                    return []
                diarize.side_effect = check_aligned
                code = main([str(audio), "--diarize", "--align-words", "--language", "fr"])
        self.assertEqual(code, 0)
        self.assertTrue(transcribe.call_args.args[2].word_timestamps)
        aligner.assert_called_once_with(device="cpu", model_name=None)
        aligner.return_value.align.assert_called_once_with(original, audio, "fr")
        assign.assert_called_once_with(refined, [])
        write.assert_called_once()

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
