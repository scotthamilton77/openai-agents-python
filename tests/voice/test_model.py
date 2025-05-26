from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.voice import TTSModel, TTSModelSettings, VoiceConfiguration
from agents.voice import VoiceModelProvider


# --- Mock TTSModel ---
# Using MagicMock(spec=TTSModel) should be sufficient.
# If TTSModel has abstract methods that MagicMock doesn't handle well for spec,
# a simple stub might be needed, but let's try with MagicMock first.

# --- Mock ModelProvider ---
class MockModelProvider(VoiceModelProvider):
    def __init__(self):
        # Track calls and set return values for get_tts_model
        self.get_tts_model_mock = MagicMock()

    def get_tts_model(self, model_name: str | None) -> TTSModel:
        return self.get_tts_model_mock(model_name=model_name)

    # Implement other abstract methods if VoiceModelProvider has any, otherwise pass
    def get_stt_model(self, model_name: str | None = None):
        raise NotImplementedError("STT model provider not implemented for these tests.")


# --- Tests for VoiceConfiguration.get_effective_model ---

def test_get_effective_model_name_provided_model_none():
    """
    Scenario 1: `tts_model_name` provided, `tts_model` is `None`.
    Provider should be called, model should be cached.
    """
    mock_provider = MockModelProvider()
    mock_returned_tts_model = MagicMock(spec=TTSModel)
    mock_provider.get_tts_model_mock.return_value = mock_returned_tts_model

    # Specific settings instance for the voice_config
    config_settings = TTSModelSettings(voice="test_voice_scenario1")
    voice_config = VoiceConfiguration(
        tts_model_name="test_model_name",
        tts_model=None,
        tts_settings=config_settings
    )

    # First call
    returned_model = voice_config.get_effective_model(mock_provider)

    mock_provider.get_tts_model_mock.assert_called_once_with(
        model_name="test_model_name"
    )
    assert voice_config.tts_model is mock_returned_tts_model
    assert returned_model is mock_returned_tts_model

    # Second call
    mock_provider.get_tts_model_mock.reset_mock() # Reset mock before second call
    returned_model_again = voice_config.get_effective_model(mock_provider)

    mock_provider.get_tts_model_mock.assert_not_called() # Should not be called again
    assert returned_model_again is mock_returned_tts_model # Should return cached model


def test_get_effective_model_instance_provided():
    """
    Scenario 2: `tts_model` (instance) provided.
    Provider should NOT be called, instance should be returned.
    """
    mock_provider = MockModelProvider()
    # Make provider's method raise an error if called, to ensure it's not
    mock_provider.get_tts_model_mock.side_effect = AssertionError("Provider should not be called")

    expected_tts_model_instance = MagicMock(spec=TTSModel)
    config_settings = TTSModelSettings(voice="test_voice_scenario2")
    voice_config = VoiceConfiguration(
        tts_model=expected_tts_model_instance,
        tts_model_name="some_name_that_should_be_ignored", # Name is ignored if instance is present
        tts_settings=config_settings
    )

    returned_model = voice_config.get_effective_model(mock_provider)

    mock_provider.get_tts_model_mock.assert_not_called()
    assert returned_model is expected_tts_model_instance
    assert voice_config.tts_model is expected_tts_model_instance # Should remain the same


def test_get_effective_model_neither_name_nor_instance_provided():
    """
    Scenario 3: Neither `tts_model_name` nor `tts_model` provided.
    Provider should be called to get a default model.
    """
    mock_provider = MockModelProvider()
    mock_default_tts_model = MagicMock(spec=TTSModel)
    mock_provider.get_tts_model_mock.return_value = mock_default_tts_model

    config_settings = TTSModelSettings(voice="test_voice_scenario3_default")
    voice_config = VoiceConfiguration(
        tts_model_name=None,
        tts_model=None,
        tts_settings=config_settings
    )

    # First call
    returned_model = voice_config.get_effective_model(mock_provider)

    mock_provider.get_tts_model_mock.assert_called_once_with(
        model_name=None # Expecting None or a default to be passed for model_name
    )
    assert voice_config.tts_model is mock_default_tts_model
    assert returned_model is mock_default_tts_model

    # Second call (should return cached model)
    mock_provider.get_tts_model_mock.reset_mock()
    returned_model_again = voice_config.get_effective_model(mock_provider)
    mock_provider.get_tts_model_mock.assert_not_called()
    assert returned_model_again is mock_default_tts_model

# Optional: Add a test to ensure settings are passed if tts_model instance is already present
# The current implementation of get_effective_model does not re-apply settings if model is cached.
# This is probably fine, as settings are usually bound at creation or resolution time.
# The prompt implies settings are passed to provider.get_tts_model, which is tested.

# Example of how to run tests with pytest locally (optional, will be removed)
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
