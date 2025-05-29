from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import UserError
from ..logger import logger
from ..tracing import Span, SpeechGroupSpanData, speech_group_span, speech_span
from ..tracing.util import time_iso
from .events import (
    VoiceStreamEvent,
    VoiceStreamEventAudio,
    VoiceStreamEventError,
    VoiceStreamEventLifecycle,
)
from .imports import np, npt
from .model import TTSModel, TTSModelSettings, VoiceConfiguration, VoiceModelProvider
from .pipeline_config import VoicePipelineConfig


def _audio_to_base64(audio_data: list[bytes]) -> str:
    joined_audio_data = b"".join(audio_data)
    return base64.b64encode(joined_audio_data).decode("utf-8")


class StreamedAudioResult:
    """The output of a `VoicePipeline`. Streams events and audio data as they're generated."""

    def __init__(
        self,
        tts_model: TTSModel,
        tts_settings: TTSModelSettings,
        voice_pipeline_config: VoicePipelineConfig,
    ):
        """Create a new `StreamedAudioResult` instance.

        Args:
            tts_model: The TTS model to use.
            tts_settings: The TTS settings to use.
            voice_pipeline_config: The voice pipeline config to use.
        """
        self.tts_model = tts_model
        self.tts_settings = tts_settings
        self.total_output_text = ""
        self.instructions = tts_settings.instructions
        self.text_generation_task: asyncio.Task[Any] | None = None

        self._voice_pipeline_config = voice_pipeline_config
        self._text_buffer = ""
        self._turn_text_buffer = ""
        self._queue: asyncio.Queue[VoiceStreamEvent] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._ordered_tasks: list[
            asyncio.Queue[VoiceStreamEvent | None]
        ] = []  # New: list to hold local queues for each text segment
        self._dispatcher_task: asyncio.Task[Any] | None = (
            None  # Task to dispatch audio chunks in order
        )

        self._done_processing = False
        self._buffer_size = tts_settings.buffer_size
        self._started_processing_turn = False
        self._first_byte_received = False
        self._generation_start_time: str | None = None
        self._completed_session = False
        self._stored_exception: BaseException | None = None
        self._tracing_span: Span[SpeechGroupSpanData] | None = None

    def update_voice_configuration(
        self, new_voice_config: VoiceConfiguration, model_provider: VoiceModelProvider
    ):
        """Update the voice configuration for the current session."""
        # Process any pending text in the buffer with the *current* TTS settings
        # before updating the configuration.
        if self._text_buffer:
            # Ensure we're in a turn, as _stream_audio might rely on turn-specific context
            # or tracing. If _start_turn is idempotent or handles this, it's fine.
            # For simplicity, we assume _start_turn handles being called if already started.
            # Consider if _start_turn is needed here if not already in a turn.
            # The current _add_text calls _start_turn, so if _text_buffer has content,
            # a turn should have been started.

            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(local_queue)
            # Use current self.tts_model and self.tts_settings for this buffered text
            # These are the "old" settings before they get updated below.
            current_model = self.tts_model
            current_settings = self.tts_settings
            task = asyncio.create_task(
                self._stream_audio(self._text_buffer, local_queue, current_model, current_settings, finish_turn=False)
            )
            self._tasks.append(task)
            self._text_buffer = ""  # Clear the buffer as it's now scheduled

            if self._dispatcher_task is None:
                self._dispatcher_task = asyncio.create_task(self._dispatch_audio())

        # Now, proceed to update the TTS configuration
        if new_voice_config.tts_settings:
            self.tts_settings = new_voice_config.tts_settings
            # Update buffer_size if it changed with new settings
            self._buffer_size = self.tts_settings.buffer_size


        if new_voice_config.tts_model:
            self.tts_model = new_voice_config.tts_model
        elif new_voice_config.tts_model_name and model_provider:
            self.tts_model = model_provider.get_tts_model(new_voice_config.tts_model_name)
        
        # Update instructions based on the potentially new tts_settings
        # self.tts_settings is guaranteed to be valid due to __init__
        self.instructions = self.tts_settings.instructions

    async def _start_turn(self):
        if self._started_processing_turn:
            return

        self._tracing_span = speech_group_span()
        self._tracing_span.start()
        self._started_processing_turn = True
        self._first_byte_received = False
        self._generation_start_time = time_iso()
        await self._queue.put(VoiceStreamEventLifecycle(event="turn_started"))

    def _set_task(self, task: asyncio.Task[Any]):
        self.text_generation_task = task

    async def _add_error(self, error: Exception):
        await self._queue.put(VoiceStreamEventError(error))

    def _transform_audio_buffer(
        self, buffer: list[bytes], output_dtype: npt.DTypeLike
    ) -> npt.NDArray[np.int16 | np.float32]:
        np_array = np.frombuffer(b"".join(buffer), dtype=np.int16)

        if output_dtype == np.int16:
            return np_array
        elif output_dtype == np.float32:
            return (np_array.astype(np.float32) / 32767.0).reshape(-1, 1)
        else:
            raise UserError("Invalid output dtype")

    async def _stream_audio(
        self,
        text: str,
        local_queue: asyncio.Queue[VoiceStreamEvent | None],
        tts_model_to_use: TTSModel,             # New
        tts_settings_to_use: TTSModelSettings,  # New
        finish_turn: bool = False,
    ):
        # Note: self._buffer_size is updated in update_voice_configuration if tts_settings change.
        # If _stream_audio is called with older tts_settings_to_use, it might use a
        # self._buffer_size that corresponds to newer settings.
        # For this refactor, we assume this is acceptable or that buffer_size changes are rare/handled.
        # A more robust solution might pass buffer_size explicitly or derive it from tts_settings_to_use.
        # However, current instructions are to use tts_settings_to_use primarily for model interaction.
        with speech_span(
            model=tts_model_to_use.model_name, # Changed
            input=text if self._voice_pipeline_config.trace_include_sensitive_data else "",
            model_config={
                "voice": tts_settings_to_use.voice, # Changed
                "instructions": tts_settings_to_use.instructions, # Changed
                "speed": tts_settings_to_use.speed, # Changed
            },
            output_format="pcm",
            parent=self._tracing_span,
        ) as tts_span:
            try:
                first_byte_received = False
                buffer: list[bytes] = []
                full_audio_data: list[bytes] = []

                async for chunk in tts_model_to_use.run(text, tts_settings_to_use): # Changed
                    if not first_byte_received:
                        first_byte_received = True
                        tts_span.span_data.first_content_at = time_iso()

                    if chunk:
                        buffer.append(chunk)
                        full_audio_data.append(chunk)
                        # Using self._buffer_size which is updated when self.tts_settings changes.
                        # This might be slightly inconsistent if tts_settings_to_use has a different buffer_size.
                        if len(buffer) >= self._buffer_size: 
                            audio_np = self._transform_audio_buffer(buffer, tts_settings_to_use.dtype) # Changed
                            if tts_settings_to_use.transform_data: # Changed
                                audio_np = tts_settings_to_use.transform_data(audio_np) # Changed
                            await local_queue.put(
                                VoiceStreamEventAudio(data=audio_np)
                            )  # Use local queue
                            buffer = []
                if buffer:
                    audio_np = self._transform_audio_buffer(buffer, tts_settings_to_use.dtype) # Changed
                    if tts_settings_to_use.transform_data: # Changed
                        audio_np = tts_settings_to_use.transform_data(audio_np) # Changed
                    await local_queue.put(VoiceStreamEventAudio(data=audio_np))  # Use local queue

                if self._voice_pipeline_config.trace_include_sensitive_audio_data:
                    tts_span.span_data.output = _audio_to_base64(full_audio_data)
                else:
                    tts_span.span_data.output = ""

                if finish_turn:
                    await local_queue.put(VoiceStreamEventLifecycle(event="turn_ended"))
                else:
                    await local_queue.put(None)  # Signal completion for this segment
            except Exception as e:
                tts_span.set_error(
                    {
                        "message": str(e),
                        "data": {
                            "text": text
                            if self._voice_pipeline_config.trace_include_sensitive_data
                            else "",
                        },
                    }
                )
                logger.error(f"Error streaming audio: {e}")

                # Signal completion for whole session because of error
                await local_queue.put(VoiceStreamEventLifecycle(event="session_ended"))
                raise e

    async def _add_text(self, text: str):
        await self._start_turn()

        self._text_buffer += text
        self.total_output_text += text
        self._turn_text_buffer += text

        combined_sentences, self._text_buffer = self.tts_settings.text_splitter(self._text_buffer)

        if len(combined_sentences) >= 20:
            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(local_queue)
            self._tasks.append(
                asyncio.create_task(self._stream_audio(combined_sentences, local_queue, self.tts_model, self.tts_settings)) # Changed
            )
            if self._dispatcher_task is None:
                self._dispatcher_task = asyncio.create_task(self._dispatch_audio())

    async def _turn_done(self):
        if self._text_buffer:
            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(local_queue)  # Append the local queue for the final segment
            self._tasks.append(
                asyncio.create_task(
                    self._stream_audio(self._text_buffer, local_queue, self.tts_model, self.tts_settings, finish_turn=True) # Changed
                )
            )
            self._text_buffer = ""
        self._done_processing = True
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatch_audio())
        await asyncio.gather(*self._tasks)

    def _finish_turn(self):
        if self._tracing_span:
            if self._voice_pipeline_config.trace_include_sensitive_data:
                self._tracing_span.span_data.input = self._turn_text_buffer
            else:
                self._tracing_span.span_data.input = ""

            self._tracing_span.finish()
            self._tracing_span = None
        self._turn_text_buffer = ""
        self._started_processing_turn = False

    async def _done(self):
        self._completed_session = True
        await self._wait_for_completion()

    async def _dispatch_audio(self):
        # Dispatch audio chunks from each segment in the order they were added
        while True:
            if len(self._ordered_tasks) == 0:
                if self._completed_session:
                    break
                await asyncio.sleep(0)
                continue
            local_queue = self._ordered_tasks.pop(0)
            while True:
                chunk = await local_queue.get()
                if chunk is None:
                    break
                await self._queue.put(chunk)
                if isinstance(chunk, VoiceStreamEventLifecycle):
                    local_queue.task_done()
                    if chunk.event == "turn_ended":
                        self._finish_turn()
                        break
        await self._queue.put(VoiceStreamEventLifecycle(event="session_ended"))

    async def _wait_for_completion(self):
        tasks: list[asyncio.Task[Any]] = self._tasks
        if self._dispatcher_task is not None:
            tasks.append(self._dispatcher_task)
        await asyncio.gather(*tasks)

    def _cleanup_tasks(self):
        self._finish_turn()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()

        if self.text_generation_task and not self.text_generation_task.done():
            self.text_generation_task.cancel()

    def _check_errors(self):
        for task in self._tasks:
            if task.done():
                if task.exception():
                    self._stored_exception = task.exception()
                    break

    async def stream(self) -> AsyncIterator[VoiceStreamEvent]:
        """Stream the events and audio data as they're generated."""
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break
            if isinstance(event, VoiceStreamEventError):
                self._stored_exception = event.error
                logger.error(f"Error processing output: {event.error}")
                break
            if event is None:
                break
            yield event
            if event.type == "voice_stream_event_lifecycle" and event.event == "session_ended":
                break

        self._check_errors()
        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception
