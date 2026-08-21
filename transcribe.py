#!/usr/bin/env python3
"""Backward-compatible launcher for the chunked transcription CLI."""

import sys
from pathlib import Path

# Allow the historical ``python transcribe.py`` command from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chunked_whisper.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
