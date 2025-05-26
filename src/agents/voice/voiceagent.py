from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agent import Agent
from .model import TTSModelSettings, TTSModel, VoiceConfiguration


@dataclass
class VoiceAgentMixin:
    """Mixin for agents that need to provide voice configuration."""
    voice_config: VoiceConfiguration = field(default_factory=VoiceConfiguration)
    
    def get_voice_configuration(self) -> VoiceConfiguration:
        """Get the voice configuration for this agent."""
        return self.voice_config


def with_voice_configuration(
    agent: Agent[Any],
    config: VoiceConfiguration
) -> Agent[Any]:
    """Add voice configuration to an existing agent.
    
    Args:
        agent: The agent to add voice configuration to
        config: The voice configuration to use
        
    Returns:
        A new agent with voice configuration
    """
    return with_voice_config(
        agent,
        tts_model=config.tts_model,
        tts_model_name=config.tts_model_name,
        tts_settings=config.tts_settings
    )
    
def with_voice_config(
    agent: Agent[Any],
    *,
    tts_model: Optional[TTSModel] = None,
    tts_model_name: Optional[str] = None,
    tts_settings: Optional[TTSModelSettings] = None
) -> Agent[Any]:
    """Add voice configuration to an existing agent.
    
    Args:
        agent: The agent to add voice configuration to
        tts_model: Optional TTSModel to use
        tts_model_name: Optional name of TTS model to use
        tts_settings: Optional TTS settings to use
        
    Returns:
        A new agent with voice configuration
    """
    # Create configuration
    config = VoiceConfiguration(
        tts_model=tts_model,
        tts_model_name=tts_model_name,
        tts_settings=tts_settings
    )
    
    # Create a new agent class that includes the VoiceAgentMixin
    class VoiceAgent(agent.__class__, VoiceAgentMixin):
        pass
    
    # Create a new agent instance with the voice configuration
    voice_agent = agent.clone()
    voice_agent.__class__ = VoiceAgent
    voice_agent.voice_config = config
    
    return voice_agent
