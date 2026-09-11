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

To enable speaker diarization, ensure the `ffmpeg` executable is installed and
install the optional pyannote.audio dependencies:

```bash
python -m pip install -e '.[diarization]'
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

## Speaker diarization

Accept the conditions for the
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
model, create a Hugging Face access token, and run:

```bash
HF_TOKEN=hf_... chunked-whisper interview.mp3 --diarize
```

The token can instead be passed with `--hf-token`. Diarization runs locally
after the model is downloaded. It assigns anonymous, file-local labels such as
`SPEAKER_00` and `SPEAKER_01`; it does not identify people. TXT output groups
consecutive turns with labels such as `[Locuteur A]` and `[Locuteur B]`,
separated by blank lines. SRT keeps the original labels, and JSON segments
gain a `speaker` field. Word timestamps
are enabled automatically so a Whisper segment can be split when its speaker
changes.

Speaker count hints can improve results when the number of participants is
known:

```bash
chunked-whisper meeting.wav --diarize --num-speakers 3
chunked-whisper meeting.wav --diarize --min-speakers 2 --max-speakers 5
```

Diarization defaults to CPU so it does not compete with Whisper for GPU memory.
Use `--diarization-device cuda` to accelerate it when enough VRAM is available,
or `--diarization-model /path/to/model` to use a locally stored pipeline.

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

## Format speaker turns

With `--diarize`, TXT output is formatted automatically, both for
`--output-format txt` and the default `--output-format all`. Consecutive lines
from the same speaker are combined, and unlabelled lines inherit the previous
speaker.

To apply the same formatting to older raw diarized TXT files, run:

```bash
python format_transcript.py transcript.txt
```

This writes `transcript.formatted.txt` beside the input. Consecutive lines from
the same speaker are combined, unlabelled lines inherit the previous speaker,
and labels such as `SPEAKER_00` and `SPEAKER_01` become `Locuteur A` and
`Locuteur B`. Several files can be processed together, optionally into another
directory:

```bash
python format_transcript.py transcript-1.txt transcript-2.txt \
  --output-dir formatted
```

## Tests

The unit tests do not download a model or require a GPU:

```bash
python -m unittest discover -s tests -v
```

The implementation uses the conventional `src/chunked_whisper` package layout.
Chunking, transcription, output formatting, data models, and the CLI live in
separate modules so each concern can be tested and maintained independently.
