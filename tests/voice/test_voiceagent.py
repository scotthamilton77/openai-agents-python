from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest

from agents import Agent, Model, ModelSettings, Tool
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from agents.voice import (
    TTSModel,
    TTSModelSettings,
    VoiceAgentMixin,
    VoiceConfiguration,
)
from agents.voice.voiceagent import with_voice_config, with_voice_configuration



# --- Mock Model for Agent Instantiation ---
class MockModel(Model):
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,  # type: ignore
        *,
        previous_response_id: str | None,
    ) -> ModelResponse:
        raise NotImplementedError("MockModel does not implement get_response")

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,  # type: ignore
        *,
        previous_response_id: str | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("MockModel does not implement stream_response")
        if False: # Ensure it's a generator
            yield


# --- Tests for VoiceAgentMixin ---

class MyAgentWithMixin(VoiceAgentMixin, Agent):
    pass


def test_voice_agent_mixin_set_and_get_config():
    agent = MyAgentWithMixin()
    config = VoiceConfiguration(tts_model_name="test_tts", tts_settings=TTSModelSettings(voice="test_voice"))
    agent.voice_config = config
    assert agent.get_voice_configuration() == config
    assert agent.voice_config == config


def test_voice_agent_mixin_default_config():
    agent = MyAgentWithMixin()
    # The default_factory should create an empty VoiceConfiguration
    assert agent.get_voice_configuration() == VoiceConfiguration()
    assert agent.voice_config == VoiceConfiguration()


# --- Tests for with_voice_config function ---

def test_with_voice_config_creates_new_agent_with_config():
    original_agent = Agent(name="OriginalAgent", model=MockModel())
    test_settings = TTSModelSettings(voice="custom_voice", buffer_size=1024)
    test_model_name = "custom_tts_model"

    new_agent = with_voice_config(
        original_agent,
        tts_model_name=test_model_name,
        tts_settings=test_settings,
    )

    assert new_agent is not original_agent
    assert isinstance(new_agent, Agent)
    assert hasattr(new_agent, "get_voice_configuration")
    
    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config is not None
    assert retrieved_config.tts_model_name == test_model_name
    assert retrieved_config.tts_settings == test_settings

    # Check original attributes are preserved
    assert new_agent.name == original_agent.name
    assert new_agent.model == original_agent.model # type: ignore

    # Check class hierarchy
    assert original_agent.__class__ in new_agent.__class__.__bases__
    assert VoiceAgentMixin in new_agent.__class__.__bases__


def test_with_voice_config_uses_default_model_name_and_settings():
    original_agent = Agent(name="OriginalAgentDefaults")

    new_agent = with_voice_config(original_agent)

    assert new_agent is not original_agent
    assert isinstance(new_agent, Agent)
    assert hasattr(new_agent, "get_voice_configuration")

    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config is not None
    assert retrieved_config.tts_model_name is None # Default
    assert retrieved_config.tts_settings is None # Default

    assert original_agent.__class__ in new_agent.__class__.__bases__
    assert VoiceAgentMixin in new_agent.__class__.__bases__


# --- Tests for with_voice_configuration function ---

def test_with_voice_configuration_creates_new_agent_with_specific_config():
    original_agent = Agent(name="OriginalAgentForSpecificConfig", model=MockModel())
    specific_voice_config = VoiceConfiguration(
        tts_model_name="specific_model",
        tts_settings=TTSModelSettings(voice="specific_voice", buffer_size=2048)
    )

    new_agent = with_voice_configuration(original_agent, specific_voice_config)

    assert new_agent is not original_agent
    assert isinstance(new_agent, Agent)
    assert hasattr(new_agent, "get_voice_configuration")

    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config == specific_voice_config

    # Check original attributes are preserved
    assert new_agent.name == original_agent.name
    assert new_agent.model == original_agent.model # type: ignore

    # Check class hierarchy
    assert original_agent.__class__ in new_agent.__class__.__bases__
    assert VoiceAgentMixin in new_agent.__class__.__bases__

# A simple mock TTSModel for testing the tts_model parameter in with_voice_config
class MockTTS(TTSModel):
    def __init__(self, name: str = "mock_tts"):
        super().__init__()
        self.tts_model_name = name

    @property
    def model_name(self) -> str:
        return self.tts_model_name

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        yield np.array([1, 2, 3], dtype=np.int16)

def test_with_voice_config_with_tts_model_instance():
    original_agent = Agent(name="OriginalAgentWithTTSInstance", model=MockModel())
    mock_tts_model = MockTTS(name="my_mock_tts")
    test_settings = TTSModelSettings(voice="instance_voice")

    new_agent = with_voice_config(
        original_agent,
        tts_model=mock_tts_model,
        tts_settings=test_settings,
    )
    
    assert new_agent is not original_agent
    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config is not None
    assert retrieved_config.tts_model is mock_tts_model
    assert retrieved_config.tts_settings == test_settings
    # tts_model_name should ideally be from the model instance if not overridden
    assert retrieved_config.tts_model_name == "my_mock_tts"

    # Test overriding tts_model_name even if tts_model is provided
    new_agent_override = with_voice_config(
        original_agent,
        tts_model=mock_tts_model,
        tts_model_name="overridden_name",
        tts_settings=test_settings,
    )
    retrieved_config_override = new_agent_override.get_voice_configuration() # type: ignore
    assert retrieved_config_override.tts_model is mock_tts_model
    assert retrieved_config_override.tts_model_name == mock_tts_model.model_name

def test_with_voice_configuration_with_tts_model_instance():
    original_agent = Agent(name="OriginalAgentConfTTSInstance", model=MockModel())
    mock_tts_model = MockTTS(name="my_mock_tts_conf")
    test_settings = TTSModelSettings(voice="instance_voice_conf")
    
    specific_voice_config = VoiceConfiguration(
        tts_model=mock_tts_model,
        tts_model_name="should_be_overridden_by_model_if_not_set", # or just check if it takes model's name
        tts_settings=test_settings
    )

    new_agent = with_voice_configuration(original_agent, specific_voice_config)

    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config.tts_model == mock_tts_model
    assert retrieved_config.tts_settings == test_settings
    assert retrieved_config.tts_model_name == "my_mock_tts_conf" # Name from model instance if not overridden in VC

    # If VoiceConfiguration explicitly sets tts_model_name, it should be used
    specific_voice_config_named = VoiceConfiguration(
        tts_model_name="explicit_conf_name",
        tts_settings=test_settings
    )
    new_agent_named = with_voice_configuration(original_agent, specific_voice_config_named)
    retrieved_config_named = new_agent_named.get_voice_configuration() # type: ignore
    assert retrieved_config_named.tts_model_name == "explicit_conf_name"

class OriginalAgentWithVoice(VoiceAgentMixin, Agent):
    pass

def test_with_voice_config_on_existing_voice_agent():
    original_agent = OriginalAgentWithVoice()
    original_agent.voice_config = VoiceConfiguration(tts_model_name="original_name")
    
    new_settings = TTSModelSettings(voice="new_voice")
    
    # with_voice_config should create a new class type, not reuse OriginalAgentWithVoice directly
    # but should preserve the voice capabilities
    new_agent = with_voice_config(original_agent, tts_model_name="new_name", tts_settings=new_settings)

    assert new_agent is not original_agent
    assert isinstance(new_agent, Agent)
    assert hasattr(new_agent, "get_voice_configuration")
    assert VoiceAgentMixin in new_agent.__class__.__bases__ 
    # Check that the original class (OriginalAgentWithVoice) is a base of the new dynamic class
    # This means the dynamic class created by with_voice_config inherits from OriginalAgentWithVoice
    assert original_agent.__class__ in new_agent.__class__.__bases__

    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config.tts_model_name == "new_name"
    assert retrieved_config.tts_settings == new_settings

def test_with_voice_configuration_on_existing_voice_agent():
    original_agent = OriginalAgentWithVoice()
    original_agent.voice_config = VoiceConfiguration(tts_model_name="original_name_conf")

    new_voice_conf = VoiceConfiguration(tts_model_name="new_name_conf", tts_settings=TTSModelSettings(voice="new_voice_conf"))

    new_agent = with_voice_configuration(original_agent, new_voice_conf)
    
    assert new_agent is not original_agent
    assert isinstance(new_agent, Agent)
    assert hasattr(new_agent, "get_voice_configuration")
    assert VoiceAgentMixin in new_agent.__class__.__bases__
    assert original_agent.__class__ in new_agent.__class__.__bases__

    retrieved_config = new_agent.get_voice_configuration() # type: ignore
    assert retrieved_config == new_voice_conf
