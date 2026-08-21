import tempfile
import unittest
from pathlib import Path

from chunked_whisper.speaker_formatter import (
    format_speaker_transcript,
    main,
    speaker_letter,
)


class SpeakerFormatterTests(unittest.TestCase):
    def test_groups_speakers_and_assigns_unlabelled_lines_to_previous_speaker(self):
        content = """\
[SPEAKER_00] First sentence.
[SPEAKER_00] Second sentence
continues here.
[SPEAKER_01] A reply.
[SPEAKER_00] Last sentence.
"""

        self.assertEqual(
            format_speaker_transcript(content),
            "[Locuteur A] First sentence. Second sentence continues here.\n\n"
            "[Locuteur B] A reply.\n\n"
            "[Locuteur A] Last sentence.\n",
        )

    def test_supports_speaker_numbers_after_z(self):
        self.assertEqual(speaker_letter(0), "A")
        self.assertEqual(speaker_letter(25), "Z")
        self.assertEqual(speaker_letter(26), "AA")

    def test_rejoins_words_split_at_an_apostrophe(self):
        content = "[SPEAKER_00] L\n'histoire continue.\n"

        self.assertEqual(
            format_speaker_transcript(content),
            "[Locuteur A] L'histoire continue.\n",
        )

    def test_keeps_text_before_the_first_speaker_unlabelled(self):
        self.assertEqual(
            format_speaker_transcript("Introduction\n[SPEAKER_00] Hello\n"),
            "Introduction\n\n[Locuteur A] Hello\n",
        )

    def test_cli_formats_multiple_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            output_dir = root / "formatted"
            first.write_text("[SPEAKER_00] Hello\n", encoding="utf-8")
            second.write_text("[SPEAKER_01] Bonjour\n", encoding="utf-8")

            exit_code = main(
                [str(first), str(second), "--output-dir", str(output_dir)]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                (output_dir / "first.formatted.txt").read_text(encoding="utf-8"),
                "[Locuteur A] Hello\n",
            )
            self.assertEqual(
                (output_dir / "second.formatted.txt").read_text(encoding="utf-8"),
                "[Locuteur B] Bonjour\n",
            )


if __name__ == "__main__":
    unittest.main()
