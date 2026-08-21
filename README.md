# Chunked Faster Whisper transcription

`transcribe.py` transcribes one or more audio files with Faster Whisper on an
NVIDIA GPU. It processes the decoded audio in overlapping chunks so speech at a
chunk boundary is still given context, then keeps each segment only once using
the non-overlapping core of each chunk.

The defaults target a GPU with approximately 6 GB of VRAM:

- `large-v3` for multilingual transcription quality;
- `int8_float16` inference to reduce VRAM use;
- five-minute chunks with five seconds of context on both sides;
- beam size 5 and Silero VAD enabled.

## Run

Activate the existing virtual environment and pass one or more files:

```bash
source bin/activate
python transcribe.py interview.mp3 meeting.m4a
```

The model is downloaded on first use. Each input produces UTF-8 text, SRT, and
JSON files beside the audio. The JSON includes segment timestamps and can also
include word timestamps:

```bash
python transcribe.py recording.wav --language en --word-timestamps
```

Write a single output type to a separate directory:

```bash
python transcribe.py audio/*.mp3 --output-dir transcripts --output-format srt
```

If `large-v3` does not fit alongside other GPU processes, use a smaller model or
more aggressive quantization:

```bash
python transcribe.py recording.wav --model medium --compute-type int8_float16
python transcribe.py recording.wav --compute-type int8
```

CPU operation is also available explicitly:

```bash
python transcribe.py recording.wav --device cpu --compute-type int8
```

Run `python transcribe.py --help` for all options.

## Tests

The unit tests do not download a model or require a GPU:

```bash
python -m unittest -v
```
