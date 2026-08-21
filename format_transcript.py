#!/usr/bin/env python3
"""Format chunked-whisper TXT output into grouped speaker turns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chunked_whisper.speaker_formatter import main


if __name__ == "__main__":
    raise SystemExit(main())
