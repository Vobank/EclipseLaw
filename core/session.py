"""
Контекст диалога (ТЗ 5.7).

Ключевое требование: контекст НЕ сбрасывается сам по себе между вопросами.
Одна сессия = одна ситуация (например, одно задержание), и закрывает её
только пользователь хоткеем сброса.

Чтобы длинная сессия не переполнила окно модели, после мягкого лимита
старая часть диалога сворачивается в конспект — но сама сессия при этом
продолжается, принудительного сброса не происходит.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from core.llm.ollama_client import Message

log = logging.getLogger(__name__)


class Mode(str, Enum):
    NORMAL = "normal"
    LAWYER = "lawyer"
    EXAM = "exam"
    DOCUMENT = "document"

    @property
    def title(self) -> str:
        return {
            Mode.NORMAL: "Обычный",
            Mode.LAWYER: "Адвокат при задержании",
            Mode.EXAM: "Экзамен",
            Mode.DOCUMENT: "Оформление документа",
        }[self]


# Порядок перебора по хоткею. Режим документа включается отдельной
# командой, а не попадает в цикл переключения (ТЗ 5.2).
MODE_CYCLE = (Mode.NORMAL, Mode.LAWYER, Mode.EXAM)


class Source(str, Enum):
    SELF = "self"          # мой микрофон
    OTHER = "other"        # собеседник через loopback
    SYSTEM = "system"      # служебная реплика приложения

    @property
    def title(self) -> str:
        return {Source.SELF: "Я", Source.OTHER: "Собеседник",
                Source.SYSTEM: "Система"}[self]


@dataclass
class Exchange:
    source: Source
    question: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    articles: list[str] = field(default_factory=list)


class Session:
    """Одна ситуация: набор обменов плюс конспект свёрнутой части."""

    def __init__(self, mode: Mode = Mode.NORMAL, *,
                 summarize_after: int = 24, keep_recent: int = 8):
        self.id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.started_at = datetime.now()
        self.exchanges: list[Exchange] = []
        self.summary: str = ""
        self.summarize_after = summarize_after
        self.keep_recent = keep_recent

    def __len__(self) -> int:
        return len(self.exchanges)

    def add(self, source: Source, question: str, answer: str,
            articles: list[str] | None = None) -> Exchange:
        exchange = Exchange(source, question, answer, articles=articles or [])
        self.exchanges.append(exchange)
        return exchange

    @property
    def needs_summary(self) -> bool:
        return len(self.exchanges) > self.summarize_after

    def build_messages(self, system_prompt: str) -> list[Message]:
        """Собирает историю в формат чата для модели."""
        messages = [Message("system", system_prompt)]
        if self.summary:
            messages.append(Message(
                "system",
                "Краткий конспект более ранней части этой же ситуации:\n" + self.summary,
            ))
        for exchange in self.exchanges:
            prefix = "" if exchange.source is Source.SELF \
                else "[реплика собеседника] "
            messages.append(Message("user", prefix + exchange.question))
            if exchange.answer:
                messages.append(Message("assistant", exchange.answer))
        return messages

    def compress(self, summarize: Callable[[str], str]) -> bool:
        """Сворачивает старые обмены в конспект. True, если что-то сжали.

        `summarize` получает готовый текст и возвращает конспект — так
        сессия не зависит от конкретного LLM-клиента.
        """
        if not self.needs_summary:
            return False

        old = self.exchanges[:-self.keep_recent]
        if not old:
            return False

        transcript = "\n".join(
            f"{e.source.title}: {e.question}\nАссистент: {e.answer}" for e in old
        )
        if self.summary:
            transcript = f"Предыдущий конспект:\n{self.summary}\n\n{transcript}"

        try:
            self.summary = summarize(transcript).strip()
        except Exception:
            log.exception("Не удалось сжать контекст, оставляем как есть")
            return False

        self.exchanges = self.exchanges[-self.keep_recent:]
        log.info("Контекст сессии %s свёрнут: %d обменов -> конспект",
                 self.id, len(old))
        return True

    def articles_mentioned(self) -> list[str]:
        seen: list[str] = []
        for exchange in self.exchanges:
            for article in exchange.articles:
                if article not in seen:
                    seen.append(article)
        return seen


class SessionManager:
    """Владеет текущей сессией и переключает режимы."""

    def __init__(self, cfg):
        self._summarize_after = cfg.get("session.summarize_after_exchanges", 24)
        self._keep_recent = cfg.get("session.keep_recent_exchanges", 8)
        self.current = self._new(Mode.NORMAL)

    def _new(self, mode: Mode) -> Session:
        return Session(mode, summarize_after=self._summarize_after,
                       keep_recent=self._keep_recent)

    def reset(self) -> Session:
        """Ручной сброс по хоткею — новая ситуация с нуля, режим сохраняется."""
        mode = self.current.mode
        self.current = self._new(mode)
        log.info("Сессия сброшена, новая: %s (режим %s)", self.current.id, mode.value)
        return self.current

    def cycle_mode(self) -> Mode:
        """Переключает обычный <-> адвокат. Смена режима начинает новую сессию:
        системный промпт другой, и мешать два профиля в одном контексте нельзя."""
        try:
            index = MODE_CYCLE.index(self.current.mode)
            next_mode = MODE_CYCLE[(index + 1) % len(MODE_CYCLE)]
        except ValueError:
            # Были в режиме документа — возвращаемся в начало цикла.
            next_mode = MODE_CYCLE[0]
        self.current = self._new(next_mode)
        log.info("Режим переключён на %s, начата сессия %s",
                 next_mode.value, self.current.id)
        return next_mode

    def set_mode(self, mode: Mode) -> Mode:
        if mode is self.current.mode:
            return mode
        self.current = self._new(mode)
        log.info("Режим установлен: %s", mode.value)
        return mode
