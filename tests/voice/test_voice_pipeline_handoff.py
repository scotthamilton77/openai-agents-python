import asyncio
import pytest
import numpy as np # Added numpy import
from collections.abc import AsyncIterator
from typing import Any, Optional, Union, List, Tuple

from src.agents.agent import Agent
from src.agents.voice.model import (
    VoiceConfiguration,
    TTSModelSettings,
    TTSModel,
    STTModel,
    VoiceConfigurationProvider,
    VoiceModelProvider,
)
from src.agents.voice.workflow import SingleAgentVoiceWorkflow
from src.agents.voice.pipeline import VoicePipeline
from src.agents.voice.input import AudioInput
from src.agents.voice.pipeline_config import VoicePipelineConfig
from src.agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent
from src.agents.result import RunResultStreaming
from src.agents.run import Runner, RunContextWrapper # Added RunContextWrapper
from src.agents.items import TResponseInputItem, TResponseStreamEvent
from src.agents.tracing import Trace # Added Trace
# GuardrailResult is not strictly needed if we pass empty lists, so not importing yet.


# --- Mock Objects ---

class MockTTSModel(TTSModel):
    def __init__(self, model_name_param: str = "mock_tts_model"):
        self._model_name = model_name_param
        self.calls: List[Tuple[str, TTSModelSettings]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        self.calls.append((text, settings))
        # print(f"MockTTSModel '{self.model_name}' called with text: '{text}', voice: '{settings.voice}'")
        yield b"mock_audio_chunk_1"
        yield b"mock_audio_chunk_2"

    def clone(self) -> TTSModel:
        return MockTTSModel(self._model_name)


class MockVoiceModelProvider(VoiceModelProvider):
    def __init__(self):
        self.mock_tts_model_instance = MockTTSModel(model_name_param="mock_tts_from_provider")

    def get_tts_model(self, model_name: Optional[str] = None) -> TTSModel:
        # Return the same instance to inspect its calls easily
        return self.mock_tts_model_instance

    def get_stt_model(self, model_name: Optional[str] = None) -> STTModel:
        return MockSTTModel()


class MockSTTModel(STTModel):
    def __init__(self, model_name_param: str = "mock_stt_model"):
        self._model_name = model_name_param

    @property
    def model_name(self) -> str:
        return self._model_name

    async def transcribe(
        self,
        audio_input: AudioInput,
        settings: Optional[Any] = None,
        trace_include_sensitive_data: bool = False,
        trace_include_sensitive_audio_data: bool = False,
    ) -> str:
        return "mocked transcription"

    async def create_session(
        self,
        audio_input: Any,
        settings: Optional[Any] = None,
        trace_include_sensitive_data: bool = False,
        trace_include_sensitive_audio_data: bool = False,
    ) -> Any: # Should be STTTranscriptionSession but avoiding more mocks
        raise NotImplementedError("MockSTTModel create_session not implemented for this test")


# --- Custom Agents ---

class VoiceAgentWithConfig(Agent, VoiceConfigurationProvider):
    def __init__(self, name: str, instructions: str, voice_config: VoiceConfiguration, response_text: str):
        super().__init__(name=name, instructions=instructions)
        self._voice_config = voice_config
        self.response_text = response_text # Text this agent will "say"

    def get_voice_configuration(self) -> Optional[VoiceConfiguration]:
        return self._voice_config

    async def run(self, messages: list[TResponseInputItem], **kwargs) -> Any:
        # This basic run is enough as Runner.run_streamed will be mocked
        return self.response_text


# --- Mock RunResultStreaming ---

class MockRunResultStreaming(RunResultStreaming):
    def __init__(self, events_to_stream: List[Union[str, AgentUpdatedStreamEvent]], final_agent: Agent, initial_agent: Agent, input_history: list = None):
        # Provide default values for all required RunResultStreaming __init__ args
        mock_input_history = input_history or []
        mock_raw_responses = [] # type: ignore
        mock_final_output = [] # type: ignore
        mock_input_guardrail_results = [] # type: ignore
        mock_output_guardrail_results = [] # type: ignore
        # Create a mock RunContextWrapper if needed, or pass None if acceptable by __init__ (check RunResultStreaming)
        # For now, let's assume None is acceptable or it's not used in a way that breaks things for this mock.
        # A proper mock might be: RunContextWrapper(current_agent=initial_agent, current_turn=0, max_turns=1)
        mock_context_wrapper = None # type: ignore 
        mock_trace = None # type: ignore

        super().__init__(
            mock_input_history,                 # 1st: input_history_items
            final_agent,                        # 2nd: final_agent
            mock_raw_responses,                 # 3rd: raw_responses
            mock_final_output,                  # 4th: final_output
            mock_input_guardrail_results,       # 5th: input_guardrail_results
            mock_output_guardrail_results,      # 6th: output_guardrail_results
            mock_context_wrapper,               # 7th: context_wrapper
            initial_agent,                      # 8th: current_agent (agent that produced this result)
            0,                                  # 9th: current_turn
            1,                                  # 10th: max_turns
            None,                               # 11th: _current_agent_output_schema
            mock_trace                          # 12th: trace
        )
        self._events_to_stream = events_to_stream
        # RunResultStreaming uses self.last_agent, so let's ensure it's set.
        # super().__init__ already sets self.last_agent = final_agent

    async def stream_events(self) -> AsyncIterator[Union[RawResponsesStreamEvent, AgentUpdatedStreamEvent]]:
        for event_item in self._events_to_stream:
            await asyncio.sleep(0) # Ensure other tasks can run
            if isinstance(event_item, str):
                # Construct the data payload according to openai.types.responses.ResponseStreamEvent
                event_data_payload: TResponseStreamEvent = {
                    "type": "response.output_text.delta",
                    "delta": event_item
                }
                yield RawResponsesStreamEvent(data=event_data_payload)
            elif isinstance(event_item, AgentUpdatedStreamEvent):
                yield event_item
            else:
                raise ValueError(f"Unsupported event item type in MockRunResultStreaming: {type(event_item)}")

    def to_input_list(self) -> list[TResponseInputItem]:
        # RunResultStreaming stores the input history in self.input
        return self.input


# --- Test ---

@pytest.mark.asyncio
async def test_voice_pipeline_updates_tts_on_agent_handoff(mocker):
    # 1. Configure Agents and Voice Configurations
    tts_settings_a = TTSModelSettings(voice="alloy", speed=1.0, buffer_size=1) # Small buffer to get more calls
    config_a = VoiceConfiguration(tts_settings=tts_settings_a)
    agent_a = VoiceAgentWithConfig(name="AgentA", instructions="I am A", voice_config=config_a, response_text="Text from Agent A. ")

    tts_settings_b = TTSModelSettings(voice="onyx", speed=1.2, buffer_size=1)
    config_b = VoiceConfiguration(tts_settings=tts_settings_b)
    agent_b = VoiceAgentWithConfig(name="AgentB", instructions="I am B", voice_config=config_b, response_text="Text from Agent B.")

    # 2. Mock VoiceModelProvider and STTModel
    mock_model_provider = MockVoiceModelProvider()
    # The MockVoiceModelProvider already returns a MockSTTModel implicitly via get_stt_model

    # 3. Prepare events for MockRunResultStreaming
    # This sequence simulates Agent A speaking, then handing off to Agent B, then Agent B speaking.
    event_sequence = [
        "Text from ", # Part 1 of Agent A's speech
        "Agent A. ",  # Part 2 of Agent A's speech
        AgentUpdatedStreamEvent(new_agent=agent_b),
        "Text from ", # Part 1 of Agent B's speech
        "Agent B.",   # Part 2 of Agent B's speech
    ]

    mock_streaming_result = MockRunResultStreaming(
        events_to_stream=event_sequence,
        final_agent=agent_b, # Agent B is the one at the end of this interaction
        initial_agent=agent_a # Agent A is the one that was run
    )

    # 4. Mock Runner.run_streamed
    # This is crucial: SingleAgentVoiceWorkflow calls Runner.run_streamed internally.
    # We mock it to return our controlled sequence of events.
    mocker.patch.object(Runner, "run_streamed", return_value=mock_streaming_result)

    # 5. Initialize VoicePipeline
    workflow = SingleAgentVoiceWorkflow(agent=agent_a)
    pipeline_config = VoicePipelineConfig(
        model_provider=mock_model_provider,
        tts_settings=TTSModelSettings(voice="default-voice", buffer_size=1) # Default pipeline voice, should be overridden
    )
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=mock_model_provider.get_stt_model(), # Use mock STT
        tts_model=mock_model_provider.get_tts_model(), # Use mock TTS (instance from provider)
        config=pipeline_config,
    )

    # 6. Run the pipeline
    dummy_audio_bytes = b"\x00\x00" * 16000 # 1 second of silence for 16kHz, 16-bit mono
    audio_buffer = np.frombuffer(dummy_audio_bytes, dtype=np.int16)
    audio_input = AudioInput(buffer=audio_buffer, frame_rate=16000, sample_width=2, channels=1)
    streamed_audio_result = await pipeline.run(audio_input)

    # 7. Collect audio and trigger TTS processing
    output_audio_chunks = []
    async for event in streamed_audio_result.stream():
        if event.type == "voice_stream_event_audio":
            output_audio_chunks.append(event.data)
        elif event.type == "voice_stream_event_lifecycle" and event.event == "session_ended":
            break
        elif event.type == "voice_stream_event_error":
            pytest.fail(f"Voice stream error: {event.error}")

    # 8. Assertions
    # Check the calls to the *single* instance of MockTTSModel shared via MockVoiceModelProvider
    tts_calls = mock_model_provider.mock_tts_model_instance.calls
    
    # Debugging: Print calls
    # for text, settings in tts_calls:
    #     print(f"Asserting call - Text: '{text}', Voice: '{settings.voice}', Speed: '{settings.speed}'")

    assert len(tts_calls) > 0, "TTSModel should have been called"

    # Expected calls based on event_sequence and buffer_size=1 for TTS settings
    # Based on how StreamedAudioResult buffers and flushes,
    # and the fix in update_voice_configuration:
    # 1. "Text from " + "Agent A. " gets buffered.
    # 2. update_voice_configuration flushes this combined buffer with Agent A's voice.
    # 3. "Text from " + "Agent B. " gets buffered.
    # 4. _turn_done() flushes this combined buffer with Agent B's voice.
    
    expected_tts_calls = [
        ("Text from Agent A. ", tts_settings_a),
        ("Text from Agent B. ", tts_settings_b),
    ]

    assert len(tts_calls) == len(expected_tts_calls), \
        f"Expected {len(expected_tts_calls)} TTS calls, but got {len(tts_calls)}. Calls: {tts_calls}"

    for i, (expected_text, expected_settings) in enumerate(expected_tts_calls):
        actual_text, actual_settings = tts_calls[i]
        assert actual_text == expected_text, \
            f"Call {i}: Text mismatch. Expected '{expected_text}', got '{actual_text}'"
        assert actual_settings.voice == expected_settings.voice, \
            f"Call {i}: Voice mismatch for text '{actual_text}'. Expected '{expected_settings.voice}', got '{actual_settings.voice}'"
        assert actual_settings.speed == expected_settings.speed, \
            f"Call {i}: Speed mismatch for text '{actual_text}'. Expected '{expected_settings.speed}', got '{actual_settings.speed}'"

    # Verify that the workflow's current agent is updated
    assert workflow.current_agent == agent_b, "Workflow's current agent should be Agent B after handoff"

    # Ensure some audio was produced (sanity check)
    assert len(output_audio_chunks) > 0, "No audio chunks were produced"

    # Verify that Runner.run_streamed was called (it was, due to mocker.patch.object)
    Runner.run_streamed.assert_called_once()
    # We can also check what it was called with if needed, but for this test,
    # controlling its output (mock_streaming_result) is the main goal.
    args, kwargs = Runner.run_streamed.call_args
    assert args[0] == agent_a # Initially called with Agent A
    assert args[1] == [{"role": "user", "content": "mocked transcription"}] # Initial input
