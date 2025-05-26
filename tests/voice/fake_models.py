from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import numpy as np
import numpy.typing as npt

try:
    from agents.voice import (
        AudioInput,
        StreamedAudioInput,
        StreamedTranscriptionSession,
        STTModel,
        STTModelSettings,
        TTSModel,
        TTSModelSettings,
        VoiceWorkflowBase,
        VoiceConfiguration,
        VoiceConfigurationProvider,
    )
except ImportError:
    pass


class FakeTTS(TTSModel):
    """Fakes TTS by just returning string bytes."""

    def __init__(self, strategy: Literal["default", "split_words"] = "default", name: str = "fake_tts"):
        super().__init__()
        self.strategy = strategy
        self.last_run_settings: TTSModelSettings | None = None
        # To distinguish instances in tests if multiple FakeTTS are used with different conceptual names
        self.model_name_prop = name 

    @property
    def model_name(self) -> str: # This is the one TTSModel requires
        return self.model_name_prop

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        self.last_run_settings = settings
        if self.strategy == "default":
            yield np.zeros(2, dtype=np.int16).tobytes()
        elif self.strategy == "split_words":
            for _ in text.split():
                yield np.zeros(2, dtype=np.int16).tobytes()

    async def verify_audio(self, text: str, audio: bytes, dtype: npt.DTypeLike = np.int16) -> None:
        assert audio == np.zeros(2, dtype=dtype).tobytes()

    async def verify_audio_chunks(
        self, text: str, audio_chunks: list[bytes], dtype: npt.DTypeLike = np.int16
    ) -> None:
        assert audio_chunks == [np.zeros(2, dtype=dtype).tobytes() for _word in text.split()]


class FakeSession(StreamedTranscriptionSession):
    """A fake streamed transcription session that yields preconfigured transcripts."""

    def __init__(self):
        self.outputs: list[str] = []

    async def transcribe_turns(self) -> AsyncIterator[str]:
        for t in self.outputs:
            yield t

    async def close(self) -> None:
        return None


class FakeSTT(STTModel):
    """A fake STT model that either returns a single transcript or yields multiple."""

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = outputs or []

    @property
    def model_name(self) -> str:
        return "fake_stt"

    async def transcribe(self, _: AudioInput, __: STTModelSettings, ___: bool, ____: bool) -> str:
        return self.outputs.pop(0)

    async def create_session(
        self,
        _: StreamedAudioInput,
        __: STTModelSettings,
        ___: bool,
        ____: bool,
    ) -> StreamedTranscriptionSession:
        session = FakeSession()
        session.outputs = self.outputs
        return session


class FakeWorkflow(VoiceWorkflowBase, VoiceConfigurationProvider):
    """A fake workflow that yields preconfigured outputs and can provide VoiceConfiguration."""

    def __init__(
        self,
        outputs: list[list[str]] | None = None,
        voice_configuration: VoiceConfiguration | None = None,
    ):
        super().__init__()
        self.outputs = outputs or []
        self.voice_configuration = voice_configuration

    def get_voice_configuration(self) -> VoiceConfiguration | None:
        return self.voice_configuration

    def add_output(self, output: list[str]) -> None:
        self.outputs.append(output)

    def add_multiple_outputs(self, outputs: list[list[str]]) -> None:
        self.outputs.extend(outputs)

    async def run(self, _: str) -> AsyncIterator[str]:
        if not self.outputs:
            raise ValueError("No output configured")
        output = self.outputs.pop(0)
        for t in output:
            yield t


class FakeStreamedAudioInput:
    @classmethod
    async def get(cls, count: int) -> StreamedAudioInput:
        input = StreamedAudioInput()
        for _ in range(count):
            await input.add_audio(np.zeros(2, dtype=np.int16))
        return input
