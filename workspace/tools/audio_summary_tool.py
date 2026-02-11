"""
Audio Summary (TTS) Tool - Convert text to speech audio (MP3).

Uses Google Text-to-Speech (gTTS) — free, no API key required.
Perfect for converting paper abstracts, summaries, or any text to audio.
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field


class AudioSummaryToolArgs(BaseModel):
    text: str = Field(
        ...,
        description=(
            "Text to convert to speech audio. Can be a paper abstract, "
            "summary, any text content up to ~5000 characters."
        ),
    )
    language: Optional[str] = Field(
        "en",
        description="Language code: 'en' (English), 'es' (Spanish), 'fr' (French), 'de' (German), 'ja' (Japanese), etc.",
    )
    slow: Optional[bool] = Field(
        False,
        description="If True, speaks more slowly.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path for the MP3 audio.",
    )


class AudioSummaryTool:
    """
    Convert text to speech audio (MP3) using Google Text-to-Speech.

    Features:
    - Free, no API key needed
    - Supports 50+ languages
    - Adjustable speech speed
    - Outputs MP3 files
    - Great for paper abstracts, summaries, notes

    Uses gTTS (Google Text-to-Speech) library.
    """

    name = "audio_summary"
    description = (
        "Convert text to speech audio (MP3). Input any text — paper abstract, "
        "summary, notes — and get an audio file. Supports multiple languages. "
        "No API key required."
    )
    args_schema = AudioSummaryToolArgs

    def __init__(self, output_dir: str = "/output"):
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError:
            self.output_dir = "/tmp"

    def run(
        self,
        text: str,
        language: str = "en",
        slow: bool = False,
        output_path: str = None,
    ) -> str:
        """Convert text to speech audio.

        Args:
            text: Text to convert to speech
            language: Language code (default 'en')
            slow: Slower speech if True
            output_path: Output MP3 file path

        Returns:
            JSON with success status, output path, and audio info.
        """
        if not text or not text.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "No text provided for speech synthesis.",
                },
                indent=2,
            )

        # Trim to reasonable length for TTS
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        # Normalize language names to codes
        lang_map = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "japanese": "ja",
            "korean": "ko",
            "chinese": "zh-CN",
            "mandarin": "zh-CN",
            "hindi": "hi",
            "arabic": "ar",
            "dutch": "nl",
            "swedish": "sv",
            "polish": "pl",
            "turkish": "tr",
        }
        language = lang_map.get(language.lower().strip(), language.strip())

        try:
            from gtts import gTTS
        except ImportError:
            return json.dumps(
                {
                    "success": False,
                    "error": "gTTS library not available. Install with: pip install gTTS",
                },
                indent=2,
            )

        try:
            tts = gTTS(text=text, lang=language, slow=slow)

            # Save — always under /output/ so it persists on host volume
            if output_path and not output_path.startswith(self.output_dir):
                output_path = os.path.join(
                    self.output_dir, os.path.basename(output_path)
                )
            if not output_path:
                existing = [
                    f
                    for f in os.listdir(self.output_dir)
                    if f.startswith("audio_") and f.endswith(".mp3")
                ]
                idx = len(existing) + 1
                output_path = os.path.join(self.output_dir, f"audio_{idx}.mp3")

            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )

            tts.save(output_path)
            file_size = os.path.getsize(output_path)

            # Estimate duration (~150 words per minute for normal, ~100 for slow)
            word_count = len(text.split())
            wpm = 100 if slow else 150
            estimated_duration_sec = (word_count / wpm) * 60

            return json.dumps(
                {
                    "success": True,
                    "output_path": output_path,
                    "language": language,
                    "word_count": word_count,
                    "char_count": len(text),
                    "estimated_duration_seconds": round(estimated_duration_sec),
                    "slow_mode": slow,
                    "file_size_bytes": file_size,
                    "message": f"Audio generated ({word_count} words, ~{round(estimated_duration_sec)}s) and saved to {output_path}",
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "hint": "Check language code is valid. Common codes: en, es, fr, de, ja, ko, zh-CN",
                },
                indent=2,
            )
