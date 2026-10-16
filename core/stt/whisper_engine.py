"""
Распознавание речи через faster-whisper (ТЗ 4).

Модель грузится лениво, при первом запросе, и дальше живёт в памяти:
на CPU инициализация small занимает секунды, и платить за неё на каждом
нажатии PTT нельзя.

Карта выбирается параметром device_index — это индекс CUDA-устройства.
Именно так AI-стек пиннится на RTX 5080, не трогая CUDA_VISIBLE_DEVICES
для всего процесса.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Ключи каталогов с библиотеками CUDA. Живут до конца работы
# процесса: стоит их потерять, и каталоги уходят из путей поиска.
_DLL_HANDLES: list = []


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    latency_s: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class WhisperEngine:
    def __init__(self, *, model: str = "small", device: str = "cpu",
                 device_index: int = 0, compute_type: str = "int8",
                 cpu_threads: int = 8, language: str = "ru",
                 vad_filter: bool = True, beam_size: int = 1,
                 vocabulary: str = ""):
        self.model_name = model
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        # Подсказка декодеру: список юридических слов смещает
        # распознавание к терминам базы. Без неё «халатность»
        # превращалась в «хладность», статья находилась, а модель
        # отвечала, что такого в базе нет — по незнакомому слову.
        self.vocabulary = vocabulary

        self._model: Any = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg) -> "WhisperEngine":
        stt_profile = cfg.profile["stt"]
        return cls(
            model=stt_profile.get("model", "small"),
            device=stt_profile.get("device", "cpu"),
            device_index=stt_profile.get("device_index", 0),
            compute_type=stt_profile.get("compute_type", "int8"),
            cpu_threads=stt_profile.get("cpu_threads", 8),
            language=cfg.get("stt.language", "ru"),
            vad_filter=cfg.get("stt.vad_filter", True),
            # Ширина луча — из профиля, если задана: на CPU она
            # стоила секунд, на GPU бесплатна, а точность даёт
            # разную.
            beam_size=stt_profile.get(
                "beam_size", cfg.get("stt.beam_size", 1)),
            vocabulary=cfg.get("stt.vocabulary", ""),
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _add_cuda_libraries() -> int:
        """Показывает Windows, где лежат cuBLAS и cuDNN.

        CTranslate2 грузит их по имени, а Python 3.8+ на Windows не ищет
        DLL в PATH — только в каталогах, добавленных явно. Библиотеки
        приезжают пакетами nvidia-cublas-cu12 и nvidia-cudnn-cu12, то
        есть лежат внутри site-packages, куда система заглядывать не
        обязана. Без этого Whisper на GPU падает с «Library
        cublas64_12.dll is not found» — при полностью исправной карте.

        Возвращает число добавленных каталогов: ноль означает, что
        пакетов нет и остаётся только CPU.
        """
        import glob
        import os
        import site

        added = 0
        roots = list(site.getsitepackages())
        user = getattr(site, "getusersitepackages", None)
        if callable(user):
            roots.append(user())
        seen: set[str] = set()
        for root in roots:
            pattern = os.path.join(root, "nvidia", "**", "*.dll")
            for dll in glob.glob(pattern, recursive=True):
                folder = os.path.dirname(dll)
                if folder in seen:
                    continue
                seen.add(folder)
                # Два способа сразу, и оба нужны.
                #
                # add_dll_directory действует только на загрузку через
                # LoadLibraryEx с поиском по пользовательским каталогам —
                # так грузит сам Python. А cuBLAS подтягивает CTranslate2
                # из C++ обычным LoadLibrary, и тот смотрит стандартные
                # пути, включая PATH. Поэтому одного add_dll_directory
                # мало: модель загружалась, «Библиотеки CUDA подключены»
                # в логе стояло, а на первом же вопросе приходило
                # «cublas64_12.dll is not found».
                #
                # Ключ add_dll_directory сохраняем: при сборке мусора
                # каталог убирается из путей обратно.
                if folder not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = folder + os.pathsep +                         os.environ.get("PATH", "")
                try:
                    _DLL_HANDLES.append(os.add_dll_directory(folder))
                except (OSError, AttributeError):
                    log.debug("Каталог CUDA не добавлен: %s", folder)
                added += 1
        if added:
            log.info("Библиотеки CUDA подключены: %d каталогов", added)
        return added

    def load(self) -> None:
        """Грузит модель. Идемпотентна, безопасна при гонке потоков."""
        # Пути к CUDA готовим до первого обращения к ctranslate2:
        # переменная PATH читается загрузчиком в момент LoadLibrary,
        # и дописывать её позже поздно.
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            started = time.monotonic()
            log.info("Загрузка Whisper %s (%s, %s)...",
                     self.model_name, self.device, self.compute_type)
            kwargs: dict[str, Any] = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.device == "cuda":
                kwargs["device_index"] = self.device_index
                if not self._add_cuda_libraries():
                    log.warning(
                        "Библиотеки CUDA не найдены. Установите: "
                        "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
            else:
                kwargs["cpu_threads"] = self.cpu_threads

            self._model = WhisperModel(self.model_name, **kwargs)
            log.info("Whisper загружен за %.1f с", time.monotonic() - started)

        if self.device == "cuda":
            self._warm_encoder()

    def _warm_encoder(self) -> None:
        """Прогоняет через кодировщик секунду шума.

        Загрузить модель мало: первый настоящий прогон на карте тратит
        десяток секунд на инициализацию ядер CUDA и cuDNN. Без прогрева
        за это платит первый же вопрос пользователя — тот самый, ради
        которого ассистент и открыт.

        VAD здесь выключен намеренно: с ним тишина отсекается до
        кодировщика, и прогрева не происходит вовсе. На этом я уже
        обжёгся — проверка «работает» проходила, не задев GPU ни разу.
        """
        started = time.monotonic()
        try:
            noise = (np.random.default_rng(0).standard_normal(16000)
                     * 0.05).astype(np.float32)
            segments, _ = self._model.transcribe(
                noise, language=self.language, beam_size=1,
                vad_filter=False)
            list(segments)
            log.info("Кодировщик Whisper прогрет за %.1f с",
                     time.monotonic() - started)
        except Exception:
            log.exception("Прогрев кодировщика не удался")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   beam_size: int | None = None) -> Transcript:
        """Распознаёт моно float32. Пустой вход не считается ошибкой."""
        duration = len(audio) / sample_rate if sample_rate else 0.0
        if duration < 0.2:
            log.debug("Слишком короткий фрагмент (%.2f с), пропуск", duration)
            return Transcript("", self.language, duration, 0.0)

        self.load()
        started = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            # Ширину луча можно поднять для повторной попытки: обычный
            # прогон идёт с beam_size=1 ради скорости, но именно жадный
            # декодер и путает «халатность» с «хладностью». Переспрос
            # считает то же аудио шире и часто слышит правильно.
            beam_size=beam_size or self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,  # иначе модель «залипает» на прошлой фразе
            initial_prompt=self.vocabulary or None,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        latency = time.monotonic() - started
        log.info("Распознано за %.2f с (аудио %.2f с): %s",
                 latency, duration, text or "<тишина>")
        return Transcript(text, getattr(info, "language", self.language), duration, latency)

    def unload(self) -> None:
        """Освобождает модель — нужно при переключении профиля железа."""
        with self._lock:
            self._model = None
        log.info("Модель Whisper выгружена")
