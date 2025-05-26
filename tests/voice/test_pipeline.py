from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

try:
    from agents.voice import (
        AudioInput,
        TTSModelSettings,
        VoiceConfiguration,
        VoiceConfigurationProvider,
        VoicePipeline,
        VoicePipelineConfig,
        VoiceWorkflowBase,
        TTSModel,
    )

    from .fake_models import FakeStreamedAudioInput, FakeSTT, FakeTTS, FakeWorkflow
    from .helpers import extract_events
    from agents.voice import VoiceModelProvider 
except ImportError:
    pass


class MockModelProvider(VoiceModelProvider):
    def __init__(self, model_map: dict[str, TTSModel]):
        self._model_map = model_map

    def get_tts_model(self, model_name: str | None = None) -> TTSModel:
        if model_name and model_name in self._model_map:
            # In a real scenario, settings might influence model creation/selection here.
            # For these tests, we assume the model is pre-configured or settings are applied later.
            return self._model_map[model_name]
        # Fallback or error if needed, for these tests direct mapping is key
        raise ValueError(f"Model {model_name} not found in mock provider and no default specified.")

    def get_stt_model(self, model_name: str | None = None) -> TTSModel:
        raise NotImplementedError("STT model provider not implemented for these tests.")


class MockTTSModel(TTSModel): # This is a generic mock, FakeTTS is more specialized
    def __init__(self, name: str = "mock_tts"):
        super().__init__()
        self.tts_model_name = name

    @property
    def model_name(self) -> str:
        return self.tts_model_name

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        yield np.array([1, 2, 3], dtype=np.int16)


class WorkflowProvidesConfig(VoiceWorkflowBase, VoiceConfigurationProvider): # Renaming for clarity if used elsewhere
    def __init__(self, config: VoiceConfiguration):
        super().__init__()
        self._config = config

    async def run(self, audio_input: AudioInput):
        pass  # Not needed for this test

    def get_voice_configuration(self) -> VoiceConfiguration | None:
        return self._config


class WorkflowNoConfig(VoiceWorkflowBase):
    async def run(self, audio_input: AudioInput):
        pass  # Not needed for this test

    def get_voice_configuration(self) -> VoiceConfiguration | None:
        return None


@pytest.mark.asyncio
async def test_get_effective_voice_configuration_from_workflow():
    expected_config = VoiceConfiguration(
        tts_model_name="workflow_tts",
        tts_settings=TTSModelSettings(buffer_size=123),
    )
    workflow = WorkflowProvidesConfig(config=expected_config)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=FakeSTT([]),
        tts_model=FakeTTS(),
        config=VoicePipelineConfig(),
    )
    effective_config = pipeline.get_effective_voice_configuration()
    assert effective_config == expected_config


@pytest.mark.asyncio
async def test_get_effective_voice_configuration_from_pipeline():
    pipeline_tts_model_name = "pipeline_tts"
    pipeline_tts_settings = TTSModelSettings(buffer_size=456)

    workflow = WorkflowNoConfig()
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=FakeSTT([]),
        tts_model=MockTTSModel(name=pipeline_tts_model_name),
        config=VoicePipelineConfig(tts_settings=pipeline_tts_settings),
    )
    effective_config = pipeline.get_effective_voice_configuration()

    assert effective_config is not None
    assert effective_config.tts_model_name == pipeline_tts_model_name
    assert effective_config.tts_settings == pipeline_tts_settings


@pytest.mark.asyncio
async def test_voicepipeline_run_single_turn() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    fake_stt = FakeSTT(["first"])
    workflow_tts_settings = TTSModelSettings(buffer_size=777)
    workflow_voice_config = VoiceConfiguration(tts_settings=workflow_tts_settings)
    workflow = FakeWorkflow([["out_1"]], voice_configuration=workflow_voice_config)
    fake_tts = FakeTTS()
    # Pipeline config with different TTS settings to ensure workflow's config takes precedence
    pipeline_config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=pipeline_config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio("out_1", audio_chunks[0])
    assert fake_tts.last_run_settings == workflow_tts_settings
    # Ensure the result object also reflects the correct model and settings
    # This test already implies workflow's model (default FakeTTS) and settings are used.
    assert result.tts_model is fake_tts # Default FakeTTS from pipeline if workflow doesn't specify model name
    assert result.tts_settings == workflow_tts_settings


@pytest.mark.asyncio
async def test_voicepipeline_streamed_audio_input() -> None:
    # Multi turn. Should produce 2 audio outputs, which are the TTS outputs of "out_1" and "out_2"
    # Verifies that workflow voice configuration is used (Scenario 3 - workflow provides all)

    fake_stt = FakeSTT(["first", "second"])
    
    # Workflow provides specific TTS settings and implicitly a model (the default FakeTTS via FakeWorkflow)
    workflow_tts_settings = TTSModelSettings(buffer_size=555,voice="workflow_voice")
    # If FakeWorkflow's default FakeTTS is used, its name is "fake_tts".
    # Let's assume the workflow intends to use this default model but with its own settings.
    workflow_voice_config = VoiceConfiguration(
        tts_model_name="fake_tts", # Explicitly or implicitly using the default
        tts_settings=workflow_tts_settings
    )
    
    # This FakeTTS instance will be used by the pipeline if no model_provider logic overrides it.
    # Or, if model_provider is used, this specific instance should be mapped to "fake_tts".
    # For this test (Scenario 3), we want the workflow's choice of model and settings to dominate.
    # The key is that `FakeWorkflow` is instantiated with `workflow_voice_config`.
    # The `VoicePipeline` is given a default `fake_tts` instance.
    # If `workflow_voice_config.tts_model_name` (e.g., "fake_tts") is resolved by a model provider
    # to this same `fake_tts` instance, then `last_run_settings` on it will be checked.
    
    # Let's refine: the pipeline is initialized with a 'default_pipeline_tts_model'.
    # The workflow config asks for 'workflow_model_via_provider'.
    # The provider maps 'workflow_model_via_provider' to 'provided_workflow_tts_model'.
    # This 'provided_workflow_tts_model' should be used with workflow_settings.

    # For this existing test, let's assume workflow uses the pipeline's default FakeTTS but overrides settings.
    # This is what the current code does: workflow provides settings, pipeline provides a model.
    # The assertion `fake_tts.last_run_settings == workflow_tts_settings` implies this.

    # To make Scenario 3 clearer, let's assume workflow has its OWN TTS model instance.
    # This is covered by `test_get_effective_voice_configuration_from_workflow` for the config object,
    # but not explicitly in a run test that checks which model instance was used.

    # Let's keep this test as is, it verifies workflow settings override pipeline settings for the pipeline's default model.
    # The new tests will cover model selection via name and provider.

    workflow = FakeWorkflow([["out_1"], ["out_2"]], voice_configuration=workflow_voice_config)
    fake_tts_on_pipeline = FakeTTS(name="pipeline_default_tts") # Pipeline's default model
    
    # Pipeline config with different TTS settings than workflow
    pipeline_config_obj = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1, voice="pipeline_voice"))
    
    pipeline = VoicePipeline(
        workflow=workflow, 
        stt_model=fake_stt, 
        tts_model=fake_tts_on_pipeline, # Pipeline's default TTS
        config=pipeline_config_obj
    )

    streamed_audio_input = await FakeStreamedAudioInput.get(count=2)

    result = await pipeline.run(streamed_audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",  # out_1
        "turn_ended",
        "turn_started",
        "audio",  # out_2
        "turn_ended",
        "session_ended",
    ]
    assert len(audio_chunks) == 2
    await fake_tts_on_pipeline.verify_audio("out_1", audio_chunks[0])
    await fake_tts_on_pipeline.verify_audio("out_2", audio_chunks[1])
    # Assert that the pipeline's default TTS model was used with workflow's settings
    assert fake_tts_on_pipeline.last_run_settings == workflow_tts_settings
    assert result.tts_model is fake_tts_on_pipeline
    assert result.tts_settings == workflow_tts_settings


@pytest.mark.asyncio
async def test_voicepipeline_run_single_turn_split_words() -> None:
    # Single turn. Should produce multiple audio outputs, which are the TTS outputs of "foo bar baz"
    # split into words and then "foo2 bar2 baz2" split into words.

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["foo bar baz"]])
    fake_tts = FakeTTS(strategy="split_words")
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",  # foo
        "audio",  # bar
        "audio",  # baz
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio_chunks("foo bar baz", audio_chunks)


@pytest.mark.asyncio
async def test_voicepipeline_run_multi_turn_split_words() -> None:
    # Multi turn. Should produce multiple audio outputs, which are the TTS outputs of "foo bar baz"
    # split into words.

    fake_stt = FakeSTT(["first", "second"])
    workflow = FakeWorkflow([["foo bar baz"], ["foo2 bar2 baz2"]])
    fake_tts = FakeTTS(strategy="split_words")
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    streamed_audio_input = await FakeStreamedAudioInput.get(count=6)
    result = await pipeline.run(streamed_audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",  # foo
        "audio",  # bar
        "audio",  # baz
        "turn_ended",
        "turn_started",
        "audio",  # foo2
        "audio",  # bar2
        "audio",  # baz2
        "turn_ended",
        "session_ended",
    ]
    assert len(audio_chunks) == 6
    await fake_tts.verify_audio_chunks("foo bar baz", audio_chunks[:3])
    await fake_tts.verify_audio_chunks("foo2 bar2 baz2", audio_chunks[3:])


@pytest.mark.asyncio
async def test_voicepipeline_float32() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["out_1"]])
    fake_tts = FakeTTS()
    config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1, dtype=np.float32))
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio("out_1", audio_chunks[0], dtype=np.float32)


@pytest.mark.asyncio
async def test_voicepipeline_transform_data() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    def _transform_data(
        data_chunk: npt.NDArray[np.int16 | np.float32],
    ) -> npt.NDArray[np.int16]:
        return data_chunk.astype(np.int16)

    fake_stt = FakeSTT(["first"])
    workflow = FakeWorkflow([["out_1"]])
    fake_tts = FakeTTS()
    config = VoicePipelineConfig(
        tts_settings=TTSModelSettings(
            buffer_size=1,
            dtype=np.float32,
            transform_data=_transform_data,
        )
    )
    pipeline = VoicePipeline(
        workflow=workflow, stt_model=fake_stt, tts_model=fake_tts, config=config
    )
    audio_input = AudioInput(buffer=np.zeros(2, dtype=np.int16))
    result = await pipeline.run(audio_input)
    events, audio_chunks = await extract_events(result)
    assert events == [
        "turn_started",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    await fake_tts.verify_audio("out_1", audio_chunks[0], dtype=np.int16)
