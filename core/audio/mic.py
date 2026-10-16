"""
Захват микрофона по push-to-talk (ТЗ 5.2, п.1).

Запись идёт ровно столько, сколько зажата клавиша, плюс небольшой хвост:
человек обычно договаривает последнее слово уже отпуская кнопку.
Ничего не пишется в фоне — поток открывается на start() и закрывается
на stop().
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class MicRecorderError(RuntimeError):
    """Не удалось открыть устройство записи."""


class MicRecorder:
    """Пишет моно-поток float32 нужной частоты, пока его не остановят."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 device: int | str | None = None,
                 tail_padding_ms: int = 250,
                 max_duration_ms: int = 30000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.tail_padding_s = tail_padding_ms / 1000.0
        self.max_duration_s = max_duration_ms / 1000.0

        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._started_at: float = 0.0
        self._overrun = False

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # Переполнение входного буфера — обычно из-за просадки системы.
            log.debug("Статус аудиопотока: %s", status)
        with self._lock:
            if self._stream is None:
                return
            if time.monotonic() - self._started_at > self.max_duration_s:
                # Предохранитель от залипшей клавиши: дальше не копим.
                self._overrun = True
                return
            self._chunks.append(indata.copy())

    def start(self) -> None:
        if self.is_recording:
            log.debug("Запись уже идёт, повторный start() проигнорирован")
            return
        with self._lock:
            self._chunks = []
            self._overrun = False
            self._started_at = time.monotonic()
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                blocksize=0,          # размер блока выбирает PortAudio
                callback=self._callback,
            )
            stream.start()
        except Exception as exc:
            raise MicRecorderError(f"не удалось открыть микрофон: {exc}") from exc
        self._stream = stream
        log.info("Микрофон: запись начата")

    def stop(self) -> np.ndarray:
        """Останавливает запись и возвращает моно float32 в [-1, 1]."""
        stream = self._stream
        if stream is None:
            return np.zeros(0, dtype=np.float32)

        # Хвост дописываем до того, как закроем поток, иначе потеряем
        # последнее слово.
        if self.tail_padding_s > 0:
            time.sleep(self.tail_padding_s)

        with self._lock:
            self._stream = None
            chunks = self._chunks
            self._chunks = []
            overrun = self._overrun

        try:
            stream.stop()
            stream.close()
        except Exception:
            log.exception("Ошибка при закрытии аудиопотока")

        if overrun:
            log.warning("Достигнут лимит длительности PTT (%.0f с), запись обрезана",
                        self.max_duration_s)
        if not chunks:
            log.warning("Микрофон: не получено ни одного семпла")
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks, axis=0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        log.info("Микрофон: записано %.2f с", len(audio) / self.sample_rate)
        return audio

    def cancel(self) -> None:
        """Прервать запись, выбросив накопленное (например, слишком короткое нажатие)."""
        stream = self._stream
        if stream is None:
            return
        with self._lock:
            self._stream = None
            self._chunks = []
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.exception("Ошибка при отмене записи")
        log.debug("Микрофон: запись отменена")


def list_input_devices() -> list[tuple[int, str, int]]:
    """(индекс, имя, число входных каналов) — для диагностики и настройки."""
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append((idx, dev["name"], dev["max_input_channels"]))
    return devices
