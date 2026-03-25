"""Transcription package for VoxVibe supporting multiple backends."""

from .base import BaseTranscriber
from .speechmatics_transcriber import SpeechmaticsTranscriber
from .voxtral_transcriber import VoxtralTranscriber
from .whisper_transcriber import WhisperTranscriber

__all__ = ["BaseTranscriber", "SpeechmaticsTranscriber", "WhisperTranscriber", "VoxtralTranscriber"]