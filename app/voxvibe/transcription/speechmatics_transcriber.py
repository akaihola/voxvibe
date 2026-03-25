"""Speechmatics batch transcriber using the speechmatics-batch SDK."""

import asyncio
import concurrent.futures
import io
import logging
from typing import List, Optional

import numpy as np
import soundfile as sf
from speechmatics.batch import (
    AsyncClient,
    FormatType,
    JobConfig,
    JobType,
    OperatingPoint,
)
from speechmatics.batch import (
    TranscriptionConfig as SmTranscriptionConfig,
)

from .base import BaseTranscriber

logger = logging.getLogger(__name__)


class SpeechmaticsTranscriber(BaseTranscriber):
    """Transcriber using Speechmatics Batch API for speech-to-text."""

    def __init__(self, config=None):
        """Initialize the Speechmatics transcriber.

        Args:
            config: Configuration object with speechmatics settings
        """
        super().__init__(config)
        self._validate_config()

    def _validate_config(self):
        """Validate that required Speechmatics configuration is present."""
        if not hasattr(self.config, "speechmatics"):
            raise ValueError("SpeechmaticsTranscriber requires speechmatics configuration")

        api_key = getattr(self.config.speechmatics, "api_key", None)
        if not api_key:
            raise ValueError("Speechmatics API key is required")

        logger.info("Speechmatics configuration validated successfully")

    def _build_job_config(self, language: str) -> JobConfig:
        """Build the job config for the Batch API.

        Args:
            language: Language code for transcription

        Returns:
            JobConfig for submit_job
        """
        op_str = getattr(self.config.speechmatics, "operating_point", "standard")
        operating_point = OperatingPoint(op_str)

        transcription_config = SmTranscriptionConfig(
            language=language,
            operating_point=operating_point,
        )
        return JobConfig(
            type=JobType.TRANSCRIPTION,
            transcription_config=transcription_config,
        )

    async def _transcribe_async(self, audio_bytes: bytes, language: str) -> Optional[str]:
        """Run the async Speechmatics batch transcription.

        Args:
            audio_bytes: WAV audio as bytes
            language: Language code for transcription

        Returns:
            Transcribed text or None
        """
        job_config = self._build_job_config(language)
        api_url = getattr(self.config.speechmatics, "api_url", None)

        async with AsyncClient(
            api_key=self.config.speechmatics.api_key,
            url=api_url,
        ) as client:
            job = await client.submit_job(
                audio_file=io.BytesIO(audio_bytes),
                config=job_config,
            )
            logger.info(f"Speechmatics job {job.id} submitted, waiting for transcript")

            transcript = await client.wait_for_completion(
                job.id,
                format_type=FormatType.TXT,
            )

        return transcript

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> Optional[str]:
        """Transcribe audio data to text using Speechmatics Batch API.

        Args:
            audio_data: Numpy array of audio data (float32, mono, 16kHz)
            language: Language code (e.g. "en", "de") or None to use config default

        Returns:
            Transcribed text or None if transcription failed
        """
        if not self.validate_audio(audio_data):
            return None

        try:
            audio_data = self.preprocess_audio(audio_data)
            audio_bytes = self._numpy_to_audio_bytes(audio_data)

            transcribe_language = language or self.config.speechmatics.language

            # Run the async transcription in a new event loop since the caller is sync
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # If there's already a running loop (e.g. inside Qt), use a thread
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    transcript = pool.submit(
                        asyncio.run,
                        self._transcribe_async(audio_bytes, transcribe_language),
                    ).result()
            else:
                transcript = asyncio.run(
                    self._transcribe_async(audio_bytes, transcribe_language),
                )

            if transcript and isinstance(transcript, str):
                transcribed_text = transcript.strip()
                if transcribed_text:
                    logger.info(f"Speechmatics transcribed: {transcribed_text}")
                    return transcribed_text

            logger.warning("Empty transcription result from Speechmatics")
            return None

        except Exception as e:
            logger.exception(f"Speechmatics transcription error: {e}")
            return None

    def _numpy_to_audio_bytes(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert numpy audio data to WAV bytes.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate for the audio

        Returns:
            Audio data as WAV bytes
        """
        with io.BytesIO() as buffer:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            return buffer.getvalue()

    def get_available_models(self) -> List[str]:
        """Get list of available operating points (Speechmatics' equivalent of models)."""
        return [
            "standard",  # Fast, cost-effective transcription
            "enhanced",  # Higher accuracy transcription
        ]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes.

        See https://docs.speechmatics.com/introduction/supported-languages
        """
        return [
            "ar",  # Arabic
            "bg",  # Bulgarian
            "ca",  # Catalan
            "cmn",  # Mandarin Chinese
            "cs",  # Czech
            "cy",  # Welsh
            "da",  # Danish
            "de",  # German
            "el",  # Greek
            "en",  # English
            "es",  # Spanish
            "et",  # Estonian
            "eu",  # Basque
            "fi",  # Finnish
            "fr",  # French
            "gl",  # Galician
            "he",  # Hebrew
            "hi",  # Hindi
            "hr",  # Croatian
            "hu",  # Hungarian
            "id",  # Indonesian
            "it",  # Italian
            "ja",  # Japanese
            "ko",  # Korean
            "lt",  # Lithuanian
            "lv",  # Latvian
            "ms",  # Malay
            "nl",  # Dutch
            "no",  # Norwegian
            "pl",  # Polish
            "pt",  # Portuguese
            "ro",  # Romanian
            "ru",  # Russian
            "sk",  # Slovak
            "sl",  # Slovenian
            "sv",  # Swedish
            "th",  # Thai
            "tr",  # Turkish
            "uk",  # Ukrainian
            "vi",  # Vietnamese
            "yue",  # Cantonese
        ]
