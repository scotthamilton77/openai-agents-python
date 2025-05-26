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
    from agents.voice.model_provider import ModelProvider # Assuming this is the base
except ImportError:
    pass


class MockModelProvider(ModelProvider):
    def __init__(self, model_map: dict[str, TTSModel]):
        self._model_map = model_map

    def get_tts_model(self, model_name: str | None = None, settings: TTSModelSettings | None = None) -> TTSModel:
        if model_name and model_name in self._model_map:
            # In a real scenario, settings might influence model creation/selection here.
            # For these tests, we assume the model is pre-configured or settings are applied later.
            return self._model_map[model_name]
        # Fallback or error if needed, for these tests direct mapping is key
        raise ValueError(f"Model {model_name} not found in mock provider and no default specified.")

    def get_stt_model(self, model_name: str | None = None, settings: TTSModelSettings | None = None):
        raise NotImplementedError("STT model provider not implemented for these tests.")


class MockTTSModel(TTSModel): # This is a generic mock, FakeTTS is more specialized
    async def warmup(self) -> None:
        pass

    async def run_tts(self, text: str, settings: TTSModelSettings | None = None):
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
        tts_settings=TTSModelSettings(chunk_size=123),
    )
    workflow = WorkflowProvidesConfig(config=expected_config)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=FakeSTT([]),
        tts_model=FakeTTS(),
        config=VoicePipelineConfig(),
    )
    effective_config = await pipeline.get_effective_voice_configuration()
    assert effective_config == expected_config


@pytest.mark.asyncio
async def test_get_effective_voice_configuration_from_pipeline():
    pipeline_tts_model_name = "pipeline_tts"
    pipeline_tts_settings = TTSModelSettings(chunk_size=456)

    workflow = WorkflowNoConfig()
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=FakeSTT([]),
        tts_model=MockTTSModel(name=pipeline_tts_model_name),
        config=VoicePipelineConfig(tts_settings=pipeline_tts_settings),
    )
    effective_config = await pipeline.get_effective_voice_configuration()

    assert effective_config is not None
    assert effective_config.tts_model_name == pipeline_tts_model_name
    assert effective_config.tts_settings == pipeline_tts_settings


@pytest.mark.asyncio
async def test_voicepipeline_run_single_turn() -> None:
    # Single turn. Should produce a single audio output, which is the TTS output for "out_1".

    fake_stt = FakeSTT(["first"])
    workflow_tts_settings = TTSModelSettings(buffer_size=777, chunk_size=888)
    workflow_voice_config = VoiceConfiguration(tts_settings=workflow_tts_settings)
    workflow = FakeWorkflow([["out_1"]], voice_configuration=workflow_voice_config)
    fake_tts = FakeTTS()
    # Pipeline config with different TTS settings to ensure workflow's config takes precedence
    pipeline_config = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1, chunk_size=2))
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
    assert result._tts_model is fake_tts # Default FakeTTS from pipeline if workflow doesn't specify model name
    assert result._tts_settings == workflow_tts_settings


@pytest.mark.asyncio
async def test_voicepipeline_streamed_audio_input() -> None:
    # Multi turn. Should produce 2 audio outputs, which are the TTS outputs of "out_1" and "out_2"
    # Verifies that workflow voice configuration is used (Scenario 3 - workflow provides all)

    fake_stt = FakeSTT(["first", "second"])
    
    # Workflow provides specific TTS settings and implicitly a model (the default FakeTTS via FakeWorkflow)
    workflow_tts_settings = TTSModelSettings(buffer_size=555, chunk_size=666, voice="workflow_voice")
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
    pipeline_config_obj = VoicePipelineConfig(tts_settings=TTSModelSettings(buffer_size=1, chunk_size=2, voice="pipeline_voice"))
    
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
    assert result._tts_model is fake_tts_on_pipeline
    assert result._tts_settings == workflow_tts_settings

# --- Scenario 1: Workflow Name, Pipeline Settings ---

@pytest.mark.asyncio
async def test_pipeline_workflow_name_pipeline_settings_single_turn():
    stt_model = FakeSTT(["turn1"])
    
    # Workflow provides model name, no settings
    workflow_voice_config = VoiceConfiguration(tts_model_name="workflow_tts_model", tts_settings=None)
    workflow = FakeWorkflow([["output1"]], voice_configuration=workflow_voice_config)

    # Pipeline provides settings and a default model
    pipeline_tts_settings = TTSModelSettings(buffer_size=123, chunk_size=111, voice="pipeline_voice_scenario1")
    default_pipeline_tts = FakeTTS(name="default_pipeline_model_s1")
    
    # Model provider maps workflow's model name to a specific FakeTTS instance
    workflow_model_tts_instance = FakeTTS(name="workflow_model_resolved_s1")
    model_provider = MockModelProvider({"workflow_tts_model": workflow_model_tts_instance})

    pipeline_config = VoicePipelineConfig(
        tts_settings=pipeline_tts_settings, 
        model_provider=model_provider
    )
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=stt_model,
        tts_model=default_pipeline_tts, # Default TTS for pipeline if workflow doesn't specify
        config=pipeline_config,
    )

    result = await pipeline.run(AudioInput(buffer=np.zeros(2, dtype=np.int16)))
    
    assert result._tts_model is workflow_model_tts_instance
    assert result._tts_model.model_name_prop == "workflow_model_resolved_s1"
    assert result._tts_settings == pipeline_tts_settings # Pipeline's settings are used
    assert workflow_model_tts_instance.last_run_settings == pipeline_tts_settings
    assert default_pipeline_tts.last_run_settings is None # Default pipeline model was not used

@pytest.mark.asyncio
async def test_pipeline_workflow_name_pipeline_settings_multi_turn():
    stt_model = FakeSTT(["turn1", "turn2"])
    workflow_voice_config = VoiceConfiguration(tts_model_name="workflow_tts_model_mt", tts_settings=None)
    workflow = FakeWorkflow([["output1"], ["output2"]], voice_configuration=workflow_voice_config)

    pipeline_tts_settings = TTSModelSettings(buffer_size=234, chunk_size=222, voice="pipeline_voice_scenario1_mt")
    default_pipeline_tts_mt = FakeTTS(name="default_pipeline_model_s1_mt")
    
    workflow_model_tts_instance_mt = FakeTTS(name="workflow_model_resolved_s1_mt")
    model_provider_mt = MockModelProvider({"workflow_tts_model_mt": workflow_model_tts_instance_mt})

    pipeline_config_mt = VoicePipelineConfig(
        tts_settings=pipeline_tts_settings,
        model_provider=model_provider_mt
    )
    pipeline_mt = VoicePipeline(
        workflow=workflow,
        stt_model=stt_model,
        tts_model=default_pipeline_tts_mt,
        config=pipeline_config_mt,
    )

    streamed_input = await FakeStreamedAudioInput.get(count=2)
    result = await pipeline_mt.run(streamed_input)

    assert result._tts_model is workflow_model_tts_instance_mt
    assert result._tts_model.model_name_prop == "workflow_model_resolved_s1_mt"
    assert result._tts_settings == pipeline_tts_settings
    assert workflow_model_tts_instance_mt.last_run_settings == pipeline_tts_settings
    assert default_pipeline_tts_mt.last_run_settings is None

# --- Scenario 2: Workflow Settings, Pipeline Name ---

@pytest.mark.asyncio
async def test_pipeline_workflow_settings_pipeline_name_single_turn():
    stt_model = FakeSTT(["turn1_s2"])
    
    # Workflow provides settings, no model name
    workflow_tts_settings = TTSModelSettings(buffer_size=789, chunk_size=333, voice="workflow_voice_scenario2")
    workflow_voice_config = VoiceConfiguration(tts_model_name=None, tts_settings=workflow_tts_settings)
    workflow = FakeWorkflow([["output1_s2"]], voice_configuration=workflow_voice_config)

    # Pipeline provides a default model name (implicitly via tts_model) and different settings
    pipeline_default_tts_settings = TTSModelSettings(buffer_size=456, chunk_size=444, voice="pipeline_voice_s2")
    # This is the model that should be selected because workflow doesn't specify one
    pipeline_model_to_be_used = FakeTTS(name="pipeline_model_s2_resolved") 
    
    # Model provider isn't strictly needed here if pipeline.tts_model is set directly,
    # but if pipeline._tts_model_name was used, provider would map it.
    # For this test, we set pipeline.tts_model directly.
    model_provider_s2 = MockModelProvider({}) # Empty, not used if tts_model is direct

    pipeline_config = VoicePipelineConfig(
        tts_settings=pipeline_default_tts_settings, # Pipeline's own settings
        model_provider=model_provider_s2 
    )
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=stt_model,
        tts_model=pipeline_model_to_be_used, # This is the pipeline's configured/default model
        config=pipeline_config,
    )

    result = await pipeline.run(AudioInput(buffer=np.zeros(2, dtype=np.int16)))
    
    assert result._tts_model is pipeline_model_to_be_used 
    assert result._tts_model.model_name_prop == "pipeline_model_s2_resolved"
    # Workflow's settings should be used with pipeline's model
    assert result._tts_settings == workflow_tts_settings 
    assert pipeline_model_to_be_used.last_run_settings == workflow_tts_settings

@pytest.mark.asyncio
async def test_pipeline_workflow_settings_pipeline_name_multi_turn():
    stt_model = FakeSTT(["turn1_s2_mt", "turn2_s2_mt"])
    
    workflow_tts_settings_mt = TTSModelSettings(buffer_size=890, chunk_size=555, voice="workflow_voice_s2_mt")
    workflow_voice_config_mt = VoiceConfiguration(tts_model_name=None, tts_settings=workflow_tts_settings_mt)
    workflow_mt = FakeWorkflow([["output1_s2_mt"], ["output2_s2_mt"]], voice_configuration=workflow_voice_config_mt)

    pipeline_default_tts_settings_mt = TTSModelSettings(buffer_size=567, chunk_size=666, voice="pipeline_voice_s2_mt")
    pipeline_model_to_be_used_mt = FakeTTS(name="pipeline_model_s2_resolved_mt")
    
    model_provider_s2_mt = MockModelProvider({})

    pipeline_config_mt = VoicePipelineConfig(
        tts_settings=pipeline_default_tts_settings_mt,
        model_provider=model_provider_s2_mt
    )
    pipeline_mt = VoicePipeline(
        workflow=workflow_mt,
        stt_model=stt_model,
        tts_model=pipeline_model_to_be_used_mt,
        config=pipeline_config_mt,
    )

    streamed_input = await FakeStreamedAudioInput.get(count=2)
    result = await pipeline_mt.run(streamed_input)

    assert result._tts_model is pipeline_model_to_be_used_mt
    assert result._tts_model.model_name_prop == "pipeline_model_s2_resolved_mt"
    assert result._tts_settings == workflow_tts_settings_mt
    assert pipeline_model_to_be_used_mt.last_run_settings == workflow_tts_settings_mt


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
