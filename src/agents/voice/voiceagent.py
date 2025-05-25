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
    from dataclasses import replace
    
    # Create base agent with mixin
    voice_agent = type(
        f"Voice{agent.__class__.__name__}", 
        (agent.__class__, VoiceAgentMixin), 
        {}
    )
    
    # Create configuration
    config = VoiceConfiguration(
        tts_model=tts_model,
        tts_model_name=tts_model_name,
        tts_settings=tts_settings
    )
    
    # Create new agent with voice configuration
    return replace(agent, __class__=voice_agent, voice_config=config)
