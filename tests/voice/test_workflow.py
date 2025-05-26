from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from unittest.mock import AsyncMock, Mock, patch

from inline_snapshot import snapshot
from openai.types.responses import ResponseCompletedEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from agents import Agent, Model, ModelSettings, ModelTracing, Tool
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)

from agents.result import RunResultStreaming
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoiceConfiguration,
    VoiceWorkflowBase,
    TTSModelSettings,
)

from ..fake_model import get_response_obj
from ..test_responses import get_function_tool, get_function_tool_call, get_text_message


# --- Tests for VoiceWorkflowBase.get_voice_configuration ---

class MyWorkflow(VoiceWorkflowBase):
    def __init__(self):
        super().__init__()
        self._current_agent_for_test: Agent | None = None

    @property
    def current_agent(self) -> Agent | None:
        return self._current_agent_for_test

    @current_agent.setter
    def current_agent(self, agent: Agent | None) -> None:
        self._current_agent_for_test = agent

    async def run(self, audio_input: AudioInput) -> AsyncIterator[str]:
        # No-op implementation for testing get_voice_configuration
        if False:
            yield ""


class MockAgentSimple:
    pass


class MockAgentWithVoiceConfig:
    def __init__(self, config: VoiceConfiguration | None):
        self._config = config

    def get_voice_configuration(self) -> VoiceConfiguration | None:
        return self._config


def test_get_voice_config_no_agent():
    workflow = MyWorkflow()
    workflow.current_agent = None
    assert workflow.get_voice_configuration() is None


def test_get_voice_config_agent_no_config():
    workflow = MyWorkflow()
    workflow.current_agent = MockAgentSimple()  # type: ignore
    assert workflow.get_voice_configuration() is None


def test_get_voice_config_agent_with_config():
    workflow = MyWorkflow()
    expected_config = VoiceConfiguration(tts_settings=TTSModelSettings(voice="test_voice_from_agent"))
    workflow.current_agent = MockAgentWithVoiceConfig(config=expected_config)  # type: ignore
    assert workflow.get_voice_configuration() == expected_config


# --- End of tests for VoiceWorkflowBase.get_voice_configuration ---


# --- Tests for SingleAgentVoiceWorkflow ---

def test_single_agent_workflow_get_voice_configuration():
    expected_config = VoiceConfiguration(tts_settings=TTSModelSettings(voice="agent_specific_voice"))
    agent = MockAgentWithVoiceConfig(config=expected_config) # type: ignore
    workflow = SingleAgentVoiceWorkflow(agent=agent) # type: ignore
    assert workflow.get_voice_configuration() == expected_config


class MockRunResultStreaming(RunResultStreaming):
    def __init__(self, last_agent: Agent, text_outputs: list[str] | None = None):
        self._last_agent = last_agent
        self._text_outputs = text_outputs if text_outputs is not None else []

    async def stream_events(self) -> AsyncIterator[TResponseStreamEvent]:
        # Simplified stream for testing callbacks and handoff, not full text processing
        for text in self._text_outputs:
            yield ResponseTextDeltaEvent(delta=text, type="response.output_text.delta", content_index=0, output_index=0, item_id="fake_item", sequence_number=0) # type: ignore
        # Yield a completion event so the workflow run finishes
        yield ResponseCompletedEvent(type="response.completed", response=get_response_obj([]), sequence_number=1) # type: ignore

    def to_input_list(self) -> list[TResponseInputItem]:
        return []

    @property
    def last_agent(self) -> Agent:
        return self._last_agent


@pytest.mark.asyncio
@patch("agents.voice.workflow.Runner.run_streamed")
async def test_single_agent_workflow_handoff_updates_current_agent(mock_run_streamed: AsyncMock):
    agent1 = Agent(name="agent1", model=FakeStreamingModel()) # Using FakeStreamingModel for a valid Agent
    agent2 = Agent(name="agent2", model=FakeStreamingModel())

    mock_run_result = MockRunResultStreaming(last_agent=agent2)
    mock_run_streamed.return_value = mock_run_result

    workflow = SingleAgentVoiceWorkflow(agent=agent1)
    async for _ in workflow.run("transcription_for_handoff"):
        pass

    mock_run_streamed.assert_called_once()
    assert workflow.current_agent == agent2

class FakeSingleAgentWorkflowCallbacks(SingleAgentWorkflowCallbacks):
    def __init__(self, on_run: Callable[[SingleAgentVoiceWorkflow, str], None], on_agent_change: Callable[[SingleAgentVoiceWorkflow, Any], None]):
        super().__init__()
        self._on_run = on_run
        self._on_agent_change = on_agent_change
    
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        self._on_run(workflow, transcription)
    
    def on_agent_change(self, workflow: SingleAgentVoiceWorkflow, new_agent: Any) -> None:
        self._on_agent_change(workflow, new_agent)

@pytest.mark.asyncio
@patch("agents.voice.workflow.Runner.run_streamed")
async def test_single_agent_workflow_callbacks(mock_run_streamed: AsyncMock):
    agent1 = Agent(name="agent1_callbacks", model=FakeStreamingModel())
    agent2 = Agent(name="agent2_callbacks", model=FakeStreamingModel())

    mock_on_run = Mock()
    mock_on_agent_change = Mock()
    callbacks = FakeSingleAgentWorkflowCallbacks(
        on_run=mock_on_run, on_agent_change=mock_on_agent_change
    )

    # Simulate handoff for on_agent_change
    mock_run_result_handoff = MockRunResultStreaming(last_agent=agent2, text_outputs=["handoff text"])
    mock_run_streamed.return_value = mock_run_result_handoff

    workflow_handoff = SingleAgentVoiceWorkflow(agent=agent1, callbacks=callbacks)
    transcription_handoff = "transcription_for_callbacks_handoff"
    async for _ in workflow_handoff.run(transcription_handoff):
        pass

    mock_on_run.assert_called_once_with(workflow_handoff, transcription_handoff)
    mock_on_agent_change.assert_called_once_with(workflow_handoff, agent2)
    assert workflow_handoff.current_agent == agent2 # Verify agent update as well

    # Reset mocks and workflow for a run without handoff
    mock_run_streamed.reset_mock()
    mock_on_run.reset_mock()
    mock_on_agent_change.reset_mock()

    agent3 = Agent(name="agent3_no_handoff", model=FakeStreamingModel())
    mock_run_result_no_handoff = MockRunResultStreaming(last_agent=agent3, text_outputs=["no handoff text"])
    mock_run_streamed.return_value = mock_run_result_no_handoff

    workflow_no_handoff = SingleAgentVoiceWorkflow(agent=agent3, callbacks=callbacks)
    transcription_no_handoff = "transcription_no_handoff"
    async for _ in workflow_no_handoff.run(transcription_no_handoff):
        pass

    mock_on_run.assert_called_once_with(workflow_no_handoff, transcription_no_handoff)
    mock_on_agent_change.assert_not_called() # No handoff, so this should not be called
    assert workflow_no_handoff.current_agent == agent3


# --- End of tests for SingleAgentVoiceWorkflow ---


class FakeStreamingModel(Model):
    def __init__(self):
        self.turn_outputs: list[list[TResponseOutputItem]] = []

    def set_next_output(self, output: list[TResponseOutputItem]):
        self.turn_outputs.append(output)

    def add_multiple_turn_outputs(self, outputs: list[list[TResponseOutputItem]]):
        self.turn_outputs.extend(outputs)

    def get_next_output(self) -> list[TResponseOutputItem]:
        if not self.turn_outputs:
            return []
        return self.turn_outputs.pop(0)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
    ) -> ModelResponse:
        raise NotImplementedError("Not implemented")

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        output = self.get_next_output()
        for item in output:
            if (
                item.type == "message"
                and len(item.content) == 1
                and item.content[0].type == "output_text"
            ):
                yield ResponseTextDeltaEvent(
                    content_index=0,
                    delta=item.content[0].text,
                    type="response.output_text.delta",
                    output_index=0,
                    item_id=item.id,
                    sequence_number=0,
                )

        yield ResponseCompletedEvent(
            type="response.completed",
            response=get_response_obj(output),
            sequence_number=1,
        )


@pytest.mark.asyncio
async def test_single_agent_workflow(monkeypatch) -> None:
    model = FakeStreamingModel()
    model.add_multiple_turn_outputs(
        [
            # First turn: a message and a tool call
            [
                get_function_tool_call("some_function", json.dumps({"a": "b"})),
                get_text_message("a_message"),
            ],
            # Second turn: text message
            [get_text_message("done")],
        ]
    )

    agent = Agent(
        "initial_agent",
        model=model,
        tools=[get_function_tool("some_function", "tool_result")],
    )

    workflow = SingleAgentVoiceWorkflow(agent)
    output = []
    async for chunk in workflow.run("transcription_1"):
        output.append(chunk)

    # Validate that the text yielded matches our fake events
    assert output == ["a_message", "done"]
    # Validate that internal state was updated
    assert workflow._input_history == snapshot(
        [
            {"content": "transcription_1", "role": "user"},
            {
                "arguments": '{"a": "b"}',
                "call_id": "2",
                "name": "some_function",
                "type": "function_call",
                "id": "1",
            },
            {
                "id": "1",
                "content": [{"annotations": [], "text": "a_message", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"call_id": "2", "output": "tool_result", "type": "function_call_output"},
            {
                "id": "1",
                "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ]
    )
    assert workflow._current_agent == agent

    model.set_next_output([get_text_message("done_2")])

    # Run it again with a new transcription to make sure the input history is updated
    output = []
    async for chunk in workflow.run("transcription_2"):
        output.append(chunk)

    assert workflow._input_history == snapshot(
        [
            {"role": "user", "content": "transcription_1"},
            {
                "arguments": '{"a": "b"}',
                "call_id": "2",
                "name": "some_function",
                "type": "function_call",
                "id": "1",
            },
            {
                "id": "1",
                "content": [{"annotations": [], "text": "a_message", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"call_id": "2", "output": "tool_result", "type": "function_call_output"},
            {
                "id": "1",
                "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"role": "user", "content": "transcription_2"},
            {
                "id": "1",
                "content": [{"annotations": [], "text": "done_2", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ]
    )
    assert workflow._current_agent == agent
