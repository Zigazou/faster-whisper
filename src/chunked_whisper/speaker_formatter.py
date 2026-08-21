"""Format diarized TXT transcripts into grouped, human-readable turns."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

SPEAKER_PATTERN = re.compile(r"^\s*\[SPEAKER_(\d+)\]\s*(.*)$")


def speaker_letter(number: int) -> str:
    """Convert a zero-based speaker number to a letter-based label.

    Args:
        number: Zero-based speaker index.

    Returns:
        A speaker label such as ``A``, ``B``, ``Z``, ``AA``, or ``AB``.

    Raises:
        ValueError: If ``number`` is negative.
    """
    if number < 0:
        raise ValueError("speaker number cannot be negative")

    letters = ""
    number += 1
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters

    return letters


def _join_fragments(fragments: Sequence[str]) -> str:
    """Join text fragments with a single space when needed.

    Args:
        fragments: Sequence of fragments that should be combined into one string.

    Returns:
        A single sentence-like string with spacing normalized.
    """
    text = ""
    for fragment in fragments:
        separator = ""

        if (
            text
            and not fragment.startswith(("'", "’"))
            and not text.endswith(("'", "’"))
        ):
            separator = " "

        text += separator + fragment

    return text


def format_speaker_transcript(content: str) -> str:
    """Translate speaker labels and merge consecutive turns by the same speaker.

    Args:
        content: Raw transcript text containing speaker-prefixed lines.

    Returns:
        A formatted transcript grouped by speaker with labels like
        ``[Locuteur A]``.
    """
    turns: list[tuple[int | None, list[str]]] = []
    current_speaker: int | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_PATTERN.match(line)
        if match:
            speaker = int(match.group(1))
            text = match.group(2).strip()
            current_speaker = speaker
        else:
            speaker = current_speaker
            text = line

        if not text:
            continue

        if turns and turns[-1][0] == speaker:
            turns[-1][1].append(text)
        else:
            turns.append((speaker, [text]))

    rendered = []
    for speaker, fragments in turns:
        text = _join_fragments(fragments)
        if speaker is None:
            rendered.append(text)
        else:
            rendered.append(f"[Locuteur {speaker_letter(speaker)}] {text}")

    return "\n\n".join(rendered) + ("\n" if rendered else "")


def output_path_for(input_path: Path, output_dir: Path | None = None) -> Path:
    """Return the default output path for an input transcript.

    Args:
        input_path: Source transcript path.
        output_dir: Optional destination directory; defaults to the input file's
            parent directory.

    Returns:
        The generated output path ending in ``.formatted.txt``.
    """
    directory = output_dir if output_dir is not None else input_path.parent

    return directory / f"{input_path.stem}.formatted.txt"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for transcript formatting.

    Returns:
        An argparse parser configured for formatting transcript files.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Group chunked-whisper TXT lines by speaker and rename anonymous "
            "speaker labels to Locuteur A, Locuteur B, and so on."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="TXT file(s) to format"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: beside each input file)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Format one or more raw TXT transcripts into grouped speaker output.

    Args:
        argv: Optional argument list to parse instead of ``sys.argv``.

    Returns:
        ``0`` on success and ``2`` when one or more input files do not exist.
    """
    args = build_parser().parse_args(argv)

    missing = [path for path in args.inputs if not path.is_file()]
    if missing:
        for path in missing:
            print(f"Input file does not exist: {path}")

        return 2

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        output_path = output_path_for(input_path, args.output_dir)
        content = input_path.read_text(encoding="utf-8")
        output_path.write_text(
            format_speaker_transcript(content),
            encoding="utf-8"
        )

        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
