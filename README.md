# Chunked Faster Whisper transcription

This package transcribes one or more audio files with Faster Whisper. It
processes the decoded audio in overlapping chunks so speech at a
chunk boundary is still given context, then keeps each segment only once using
the non-overlapping core of each chunk.

The defaults target a GPU with approximately 6 GB of VRAM:

- `large-v3` for multilingual transcription quality;
- `int8_float16` inference to reduce VRAM use;
- five-minute chunks with five seconds of context on both sides;
- beam size 5 and Silero VAD enabled.

## Install

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run

The installation provides the `chunked-whisper` command:

```bash
chunked-whisper interview.mp3 meeting.m4a
```

`python transcribe.py ...` remains available as a compatibility command when
running from the repository.

The model is downloaded on first use. Each input produces UTF-8 text, SRT, and
JSON files beside the audio. The JSON includes segment timestamps and can also
include word timestamps:

```bash
chunked-whisper recording.wav --language en --word-timestamps
```

Write a single output type to a separate directory:

```bash
chunked-whisper audio/*.mp3 --output-dir transcripts --output-format srt
```

If `large-v3` does not fit alongside other GPU processes, use a smaller model or
more aggressive quantization:

```bash
chunked-whisper recording.wav --model medium --compute-type int8_float16
chunked-whisper recording.wav --compute-type int8
```

CPU operation is also available explicitly:

```bash
chunked-whisper recording.wav --device cpu --compute-type int8
```

Run `chunked-whisper --help` for all options.

## Tests

The unit tests do not download a model or require a GPU:

```bash
python -m unittest discover -s tests -v
```

The implementation uses the conventional `src/chunked_whisper` package layout.
Chunking, transcription, output formatting, data models, and the CLI live in
separate modules so each concern can be tested and maintained independently.
