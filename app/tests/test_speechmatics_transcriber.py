from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voxvibe.config import SpeechmaticsConfig, TranscriptionConfig
from voxvibe.transcription.speechmatics_transcriber import SpeechmaticsTranscriber

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def speechmatics_config():
    """Create a TranscriptionConfig with speechmatics settings."""
    return TranscriptionConfig(
        backend="speechmatics",
        speechmatics=SpeechmaticsConfig(
            api_key="test-api-key",
            language="en",
            operating_point="standard",
        ),
    )


@pytest.fixture
def mock_async_client(mocker: "MockerFixture"):
    """Mock the AsyncClient async context manager and its methods."""
    mock_client_instance = AsyncMock()

    # submit_job returns a JobDetails-like object with .id
    mock_job = MagicMock()
    mock_job.id = "job-123"
    mock_client_instance.submit_job.return_value = mock_job

    # wait_for_completion returns transcript text
    mock_client_instance.wait_for_completion.return_value = "Hello world"

    # Patch AsyncClient so that `async with AsyncClient(...) as client` yields our mock
    mock_client_class = mocker.patch(
        "voxvibe.transcription.speechmatics_transcriber.AsyncClient",
    )
    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

    return mock_client_instance


@pytest.fixture
def transcriber(speechmatics_config, mock_async_client):
    """Create a SpeechmaticsTranscriber with mocked client."""
    return SpeechmaticsTranscriber(speechmatics_config)


def test_init_requires_speechmatics_config():
    """Test that initialization fails without speechmatics config."""
    config = TranscriptionConfig(backend="faster-whisper")
    del config.speechmatics
    with pytest.raises(ValueError, match="requires speechmatics configuration"):
        SpeechmaticsTranscriber(config)


def test_init_requires_api_key():
    """Test that initialization fails without an API key."""
    config = TranscriptionConfig(
        backend="speechmatics",
        speechmatics=SpeechmaticsConfig(api_key=""),
    )
    with pytest.raises(ValueError, match="API key is required"):
        SpeechmaticsTranscriber(config)


def test_init_success(speechmatics_config, mock_async_client):
    """Test successful initialization with valid config."""
    transcriber = SpeechmaticsTranscriber(speechmatics_config)
    assert transcriber.config == speechmatics_config


def test_transcribe_no_audio(transcriber):
    """Test transcription with no audio data."""
    assert transcriber.transcribe(None) is None
    assert transcriber.transcribe(np.array([])) is None


def test_transcribe_audio_too_short(transcriber):
    """Test transcription with audio shorter than minimum length."""
    short_audio = np.random.random(100).astype(np.float32)
    result = transcriber.transcribe(short_audio)
    assert result is None


def test_transcribe_successful(transcriber, mock_async_client):
    """Test successful transcription."""
    audio = np.random.random(5000).astype(np.float32)

    result = transcriber.transcribe(audio)

    assert result == "Hello world"
    mock_async_client.submit_job.assert_called_once()
    mock_async_client.wait_for_completion.assert_called_once()


def test_transcribe_wait_uses_txt_format(transcriber, mock_async_client, mocker: "MockerFixture"):
    """Test that wait_for_completion requests TXT format."""
    from speechmatics.batch import FormatType

    audio = np.random.random(5000).astype(np.float32)
    transcriber.transcribe(audio)

    call_kwargs = mock_async_client.wait_for_completion.call_args
    assert call_kwargs.kwargs.get("format_type") == FormatType.TXT


def test_transcribe_uses_config_language(transcriber, mock_async_client, mocker: "MockerFixture"):
    """Test that transcription uses the configured language."""
    mock_sm_config = mocker.patch(
        "voxvibe.transcription.speechmatics_transcriber.SmTranscriptionConfig",
    )
    audio = np.random.random(5000).astype(np.float32)
    transcriber.transcribe(audio)

    mock_sm_config.assert_called_once()
    assert mock_sm_config.call_args.kwargs["language"] == "en"


def test_transcribe_uses_explicit_language(transcriber, mock_async_client, mocker: "MockerFixture"):
    """Test that an explicit language parameter overrides config."""
    mock_sm_config = mocker.patch(
        "voxvibe.transcription.speechmatics_transcriber.SmTranscriptionConfig",
    )
    audio = np.random.random(5000).astype(np.float32)
    transcriber.transcribe(audio, language="es")

    mock_sm_config.assert_called_once()
    assert mock_sm_config.call_args.kwargs["language"] == "es"


def test_transcribe_enhanced_operating_point(mock_async_client, mocker: "MockerFixture"):
    """Test that the enhanced operating point is passed through."""
    from speechmatics.batch import OperatingPoint

    mock_sm_config = mocker.patch(
        "voxvibe.transcription.speechmatics_transcriber.SmTranscriptionConfig",
    )
    config = TranscriptionConfig(
        backend="speechmatics",
        speechmatics=SpeechmaticsConfig(
            api_key="test-key",
            language="en",
            operating_point="enhanced",
        ),
    )
    transcriber = SpeechmaticsTranscriber(config)

    audio = np.random.random(5000).astype(np.float32)
    transcriber.transcribe(audio)

    mock_sm_config.assert_called_once()
    assert mock_sm_config.call_args.kwargs["operating_point"] == OperatingPoint.ENHANCED


def test_transcribe_uses_api_key(mock_async_client, mocker: "MockerFixture"):
    """Test that the API key is passed to AsyncClient."""
    mock_client_class = mocker.patch(
        "voxvibe.transcription.speechmatics_transcriber.AsyncClient",
    )
    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

    config = TranscriptionConfig(
        backend="speechmatics",
        speechmatics=SpeechmaticsConfig(
            api_key="my-secret-key",
        ),
    )
    transcriber = SpeechmaticsTranscriber(config)

    audio = np.random.random(5000).astype(np.float32)
    transcriber.transcribe(audio)

    mock_client_class.assert_called_once_with(
        api_key="my-secret-key",
        url="https://eu1.asr.api.speechmatics.com/v2",
    )


def test_transcribe_uses_custom_api_url(mock_async_client, mocker: "MockerFixture"):
    """Test that a custom API URL is passed to AsyncClient."""
    mock_client_class = mocker.patch(
        "voxvibe.transcription.speechmatics_transcriber.AsyncClient",
    )
    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

    config = TranscriptionConfig(
        backend="speechmatics",
        speechmatics=SpeechmaticsConfig(
            api_key="test-key",
            api_url="https://us1.asr.api.speechmatics.com/v2",
        ),
    )
    transcriber = SpeechmaticsTranscriber(config)

    audio = np.random.random(5000).astype(np.float32)
    transcriber.transcribe(audio)

    mock_client_class.assert_called_once_with(
        api_key="test-key",
        url="https://us1.asr.api.speechmatics.com/v2",
    )


def test_transcribe_empty_result(transcriber, mock_async_client):
    """Test transcription when API returns empty text."""
    audio = np.random.random(5000).astype(np.float32)
    mock_async_client.wait_for_completion.return_value = "   "

    result = transcriber.transcribe(audio)
    assert result is None


def test_transcribe_none_result(transcriber, mock_async_client):
    """Test transcription when API returns None."""
    audio = np.random.random(5000).astype(np.float32)
    mock_async_client.wait_for_completion.return_value = None

    result = transcriber.transcribe(audio)
    assert result is None


def test_transcribe_exception_handling(transcriber, mock_async_client):
    """Test that transcription exceptions are handled gracefully."""
    audio = np.random.random(5000).astype(np.float32)
    mock_async_client.submit_job.side_effect = Exception("API error")

    result = transcriber.transcribe(audio)
    assert result is None


def test_transcribe_strips_whitespace(transcriber, mock_async_client):
    """Test that transcription result is stripped of whitespace."""
    audio = np.random.random(5000).astype(np.float32)
    mock_async_client.wait_for_completion.return_value = "  Hello world  \n"

    result = transcriber.transcribe(audio)
    assert result == "Hello world"


def test_transcribe_audio_format_conversion(transcriber, mock_async_client):
    """Test that non-float32 audio is converted."""
    audio_int16 = np.array([1000, 2000, 3000] * 1000, dtype=np.int16)

    result = transcriber.transcribe(audio_int16)
    assert result == "Hello world"


def test_transcribe_with_running_event_loop(transcriber, mock_async_client, mocker: "MockerFixture"):
    """Test transcription when there's already a running event loop (e.g. Qt)."""
    audio = np.random.random(5000).astype(np.float32)

    # Simulate an already-running event loop
    mock_loop = mocker.MagicMock()
    mock_loop.is_running.return_value = True
    mocker.patch("asyncio.get_running_loop", return_value=mock_loop)

    result = transcriber.transcribe(audio)

    # Should still succeed – falls back to ThreadPoolExecutor + asyncio.run
    assert result == "Hello world"
    mock_async_client.submit_job.assert_called_once()


def test_get_available_models(transcriber):
    """Test getting list of available operating points."""
    models = transcriber.get_available_models()
    assert "standard" in models
    assert "enhanced" in models


def test_get_supported_languages(transcriber):
    """Test getting list of supported languages."""
    languages = transcriber.get_supported_languages()
    assert isinstance(languages, list)
    assert "en" in languages
    assert "de" in languages
    assert "fr" in languages
    assert len(languages) > 20
