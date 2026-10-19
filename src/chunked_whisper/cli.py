"""Command-line interface for chunked transcription."""

from __future__ import annotations

from argparse import ArgumentTypeError, ArgumentParser
from logging import basicConfig, getLogger, INFO
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from faster_whisper import WhisperModel

from .alignment import WordAligner
from .diarization import (
    DEFAULT_DIARIZATION_MODEL,
    assign_speakers,
    diarize_audio,
    load_diarization_pipeline,
)

from .output import OutputFormat, write_outputs
from .transcription import TranscriptionConfig, transcribe_audio

LOGGER = getLogger(__name__)


def positive_float(value: str) -> float:
    """Validate a CLI value that must be strictly greater than zero.

    Args:
        value: String representation of the numeric argument.

    Returns:
        The parsed float value.

    Raises:
        ArgumentTypeError: If the float is less than or equal to zero.
    """
    try:
        number = float(value)
    except ValueError:
        raise ArgumentTypeError("must be a number")

    if number <= 0:
        raise ArgumentTypeError("must be greater than zero")

    return number


def non_negative_float(value: str) -> float:
    """Validate a CLI value that cannot be negative.

    Args:
        value: String representation of the numeric argument.

    Returns:
        The parsed float value.

    Raises:
        ArgumentTypeError: If the float is negative.
    """
    try:
        number = float(value)
    except ValueError:
        raise ArgumentTypeError("must be a number")

    if number < 0:
        raise ArgumentTypeError("cannot be negative")

    return number


def positive_int(value: str) -> int:
    """Validate a CLI integer that must be at least one.

    Args:
        value: String representation of the integer argument.

    Returns:
        The parsed integer value.

    Raises:
        ArgumentTypeError: If the integer is less than one.
    """
    try:
        number = int(value)
    except ValueError:
        raise ArgumentTypeError("must be an integer")

    if number < 1:
        raise ArgumentTypeError("must be at least 1")

    return number


def build_parser() -> ArgumentParser:
    """Construct the command-line parser for the transcription CLI.

    Returns:
        An argparse parser configured with all transcription and diarization
        options.
    """
    parser = ArgumentParser(
        description="Transcribe audio files using overlapping chunks."
    )

    parser.add_argument(
        "audio",
        nargs="+",
        type=Path,
        help="audio file(s) to process"
    )

    parser.add_argument(
        "--model",
        default="large-v3",
        help="model name or local model directory (default: large-v3)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CTranslate2 device (default: cuda)"
    )

    parser.add_argument(
        "--compute-type",
        default="int8_float16",
        help="CTranslate2 compute type (default: int8_float16)",
    )

    parser.add_argument(
        "--chunk-duration",
        type=positive_float,
        default=300.0,
        metavar="SECONDS",
        help="non-overlapping core chunk duration (default: 300)",
    )

    parser.add_argument(
        "--overlap",
        type=non_negative_float,
        default=5.0,
        metavar="SECONDS",
        help="audio context added to each side of a chunk (default: 5)",
    )

    parser.add_argument(
        "--language",
        help="spoken language code, such as en or fr (default: detect once)",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="beam size (default: 5)"
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="disable the Silero voice activity filter",
    )

    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="include word timing in JSON",
    )

    parser.add_argument(
        "--diarize",
        action="store_true",
        help="differentiate speakers with anonymous labels using pyannote.audio",
    )

    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help=f"pyannote pipeline (default: {DEFAULT_DIARIZATION_MODEL})",
    )

    parser.add_argument(
        "--diarization-device",
        default="cpu",
        help="PyTorch device for diarization (default: cpu)",
    )

    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for the gated diarization model",
    )

    parser.add_argument(
        "--align-words", action="store_true",
        help="refine word timings with WhisperX before assigning speakers",
    )
    parser.add_argument(
        "--alignment-device", default="cpu",
        help="PyTorch device for word alignment (default: cpu)",
    )
    parser.add_argument(
        "--alignment-model",
        help="alignment model name (default: select by transcript language)",
    )

    speaker_group = parser.add_argument_group("speaker count hints")

    speaker_group.add_argument(
        "--num-speakers",
        type=positive_int,
        help="expected number of speakers",
    )

    speaker_group.add_argument(
        "--min-speakers",
        type=positive_int,
        help="minimum number of speakers",
    )

    speaker_group.add_argument(
        "--max-speakers",
        type=positive_int,
        help="maximum number of speakers",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: beside each input)",
    )

    parser.add_argument(
        "--output-format",
        choices=("all", "txt", "srt", "json"),
        default="all",
        help="files to write (default: all)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line transcription workflow for one or more audio files.

    Args:
        argv: Optional argument list to parse instead of ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` on model or transcription failures, and ``2`` on
        invalid CLI input or missing files.
    """
    args = build_parser().parse_args(argv)
    basicConfig(level=INFO, format="%(levelname)s: %(message)s")

    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        for path in missing:
            LOGGER.error("Input file does not exist: %s", path)
        return 2

    config = TranscriptionConfig(
        model_name=args.model,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        word_timestamps=args.word_timestamps,
    )

    if args.diarize or args.align_words:
        config = replace(config, word_timestamps=True)

    try:
        config.validate()

        if args.num_speakers is not None and (
            args.min_speakers is not None or args.max_speakers is not None
        ):
            raise ValueError(
                "--num-speakers cannot be combined with speaker count bounds"
            )

        if (
            args.min_speakers is not None
            and args.max_speakers is not None
            and args.min_speakers > args.max_speakers
        ):
            raise ValueError("--min-speakers cannot exceed --max-speakers")

        if not args.diarize and any(
            value is not None
            for value in (args.num_speakers, args.min_speakers, args.max_speakers)
        ):
            raise ValueError("speaker count hints require --diarize")
    except ValueError as error:
        LOGGER.error("Invalid options: %s", error)
        return 2

    LOGGER.info(
        "Loading model %s on %s using %s",
        args.model,
        args.device,
        args.compute_type,
    )

    try:
        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute_type
        )

    except Exception as error:
        LOGGER.error("Could not load the model: %s", error)
        return 1

    diarization_pipeline = None

    if args.diarize:
        LOGGER.info(
            "Loading diarization model %s on %s",
            args.diarization_model,
            args.diarization_device,
        )

        try:
            diarization_pipeline = load_diarization_pipeline(
                args.diarization_model,
                token=args.hf_token,
                device=args.diarization_device,
            )
        except Exception as error:
            LOGGER.error("Could not load the diarization model: %s", error)
            return 1

    aligner = (
        WordAligner(device=args.alignment_device, model_name=args.alignment_model)
        if args.align_words else None
    )
    failed = False
    for audio_path in args.audio:
        try:
            result = transcribe_audio(model, audio_path, config)

            if aligner is not None:
                result.segments = aligner.align(
                    result.segments, audio_path, result.language
                )

            if diarization_pipeline is not None:
                turns = diarize_audio(
                    diarization_pipeline,
                    audio_path,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )

                result.segments = assign_speakers(result.segments, turns)

            paths = write_outputs(
                result,
                audio_path,
                args.output_dir,
                cast(OutputFormat, args.output_format),
            )

            LOGGER.info(
                "Finished %s (%s, %.1f%% confidence): %s",
                audio_path,
                result.language,
                result.language_probability * 100,
                ", ".join(str(path) for path in paths),
            )
        except Exception as error:
            failed = True
            LOGGER.exception("Could not transcribe %s: %s", audio_path, error)

    return 1 if failed else 0
