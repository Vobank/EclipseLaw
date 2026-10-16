"""
Лента сессии в окне ассистента.

Раньше ответ жил ровно до следующего нажатия PTT: задал новый вопрос —
прошлый исчез, и посмотреть, что бот отвечал минуту назад, было негде.
Теперь записи копятся до сброса сессии.

Тесты работают на логике ленты, без поднятия Qt: проверяется, что
накапливается, что чистится и где обрывается.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.overlay import OverlayWindow  # noqa: E402


class Transcript:
    """Лента без Qt: те же методы, но с заглушкой вместо виджетов."""

    MAX_ENTRIES = OverlayWindow.MAX_ENTRIES
    SEPARATOR = OverlayWindow.SEPARATOR

    _render = lambda self: None                       # noqa: E731
    _on_answer = OverlayWindow._on_answer
    _on_answer_chunk = OverlayWindow._on_answer_chunk
    begin_answer = OverlayWindow.begin_answer
    clear_answers = OverlayWindow.clear_answers

    def __init__(self):
        self._answer_buffer = ""
        self._entries: list[str] = []

    @property
    def shown(self) -> str:
        parts = [x for x in self._entries + [self._answer_buffer] if x]
        return self.SEPARATOR.join(parts)


def test_previous_answer_survives_a_new_question():
    t = Transcript()
    t._on_answer("Первый ответ")
    t.begin_answer()
    t._on_answer_chunk("Второй ответ")
    assert "Первый ответ" in t.shown
    assert "Второй ответ" in t.shown


def test_entries_are_separated():
    t = Transcript()
    t._on_answer("А")
    t.begin_answer()
    t._on_answer("Б")
    assert t.shown == "А" + Transcript.SEPARATOR + "Б"


def test_streaming_appends_not_replaces():
    t = Transcript()
    for chunk in ("Ста", "тья ", "15.6"):
        t._on_answer_chunk(chunk)
    assert t._answer_buffer == "Статья 15.6"


def test_replace_touches_only_the_current_entry():
    """Ошибка заменяет текущий ответ, а не всю историю."""
    t = Transcript()
    t._on_answer("Старый ответ")
    t.begin_answer()
    t._on_answer_chunk("начало")
    t._on_answer("Ошибка обращения к модели")
    assert "Старый ответ" in t.shown
    assert "начало" not in t.shown


def test_empty_entry_is_not_kept():
    """Пустой ответ не должен оставлять в ленте дырку."""
    t = Transcript()
    t.begin_answer()
    t.begin_answer()
    t._on_answer("Ответ")
    assert t.shown == "Ответ"


def test_reset_clears_everything():
    t = Transcript()
    t._on_answer("Ответ")
    t.begin_answer()
    t._on_answer("Ещё")
    t.clear_answers()
    assert t.shown == ""
    assert t._entries == []


def test_transcript_does_not_grow_forever():
    """После часа работы метка тянула бы на сотни килобайт текста."""
    t = Transcript()
    for index in range(Transcript.MAX_ENTRIES + 15):
        t._on_answer(f"ответ {index}")
        t.begin_answer()
    assert len(t._entries) == Transcript.MAX_ENTRIES
    # Уезжает старое, свежее остаётся.
    assert "ответ 0" not in t.shown
    assert f"ответ {Transcript.MAX_ENTRIES + 14}" in t.shown


@pytest.mark.parametrize("order", [
    ["вопрос про халатность", "вопрос про скорость"],
    ["первый", "второй", "третий"],
])
def test_order_is_kept(order):
    t = Transcript()
    for text in order:
        t._on_answer(text)
        t.begin_answer()
    shown = t.shown
    positions = [shown.index(x) for x in order]
    assert positions == sorted(positions)
