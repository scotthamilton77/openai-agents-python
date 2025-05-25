# Voice Agent Configuration Guide

## Overview

This guide explains how to use the new `VoiceConfiguration` system to customize the Text-to-Speech (TTS) settings for individual agents in a voice workflow.

## Basic Usage

### Creating a Voice Agent

Use the `with_voice_config` helper function to add voice configuration to any agent:

```python
from agents.voice.voiceagent import with_voice_config
from agents.voice.model import TTSModelSettings
from agents.agent import Agent

# Create a regular agent
base_agent = Agent(name="Customer Support")

# Add voice configuration
voice_agent = with_voice_config(
    base_agent,
    tts_model_name="tts-1",
    tts_settings=TTSModelSettings(
        voice="nova",  # Use the Nova voice
        speed=1.1  # Speak slightly faster
    )
)
```

### Using in a VoiceWorkflow

The voice workflow will automatically use the voice configuration from the current agent:

```python
from agents.voice.workflow import SingleAgentVoiceWorkflow
from agents.voice.pipeline import VoicePipeline

# Create workflow with voice agent
workflow = SingleAgentVoiceWorkflow(voice_agent)

# Create pipeline with workflow
pipeline = VoicePipeline(
    workflow=workflow
)

# No additional configuration needed - TTS settings from the agent 
# will be used automatically!
```

## Agent Handoffs

When using agent handoffs, each agent can have its own voice configuration:

```python
# Create agents with different voices
support_agent = with_voice_config(
    Agent(name="Support"), 
    tts_settings=TTSModelSettings(voice="nova")
)

technical_agent = with_voice_config(
    Agent(name="Technical"), 
    tts_settings=TTSModelSettings(voice="echo")
)

# Add handoff capability
support_agent = support_agent.clone(
    handoffs=[technical_agent]
)

# During handoffs, the voice will change automatically!
```

## Implementation Details

The system uses a protocol-based approach with the following components:

1. `VoiceConfiguration` - A dataclass holding TTS model and settings
2. `VoiceConfigurationProvider` - A protocol for classes that can provide voice configuration
3. `VoiceWorkflowBase` - Implements the provider interface and checks if agents have voice configuration
4. `VoiceAgentMixin` - A mixin for agents to provide voice configuration
5. `with_voice_config` - A helper function to create agents with voice configuration

This design eliminates the need for callback proxies and simplifies the overall architecture.
