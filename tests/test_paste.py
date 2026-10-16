"""
Вставка текста из буфера вместо голоса.

Распознавание иногда обрывает длинную фразу на середине, а часть
материала приходит текстом — из форума, из чата, от судьи. Набирать
это голосом заново бессмысленно: текст попадает туда же, куда попала
бы речь.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.session import Mode, Source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Session:
    mode = Mode.NORMAL


class Sessions:
    def __init__(self):
        self.current = Session()


class Fake:
    """Оркестратор без Qt, аудио и моделей — только вставка."""

    _paste_text = Orchestrator._paste_text

    def __init__(self, clipboard_text):
        self.sessions = Sessions()
        self._dictation = []
        self.answers = []
        self.logged = []
        self.status = []
        self.shown = 0
        self.cfg = type("C", (), {
            "get": staticmethod(lambda key, default=None:
                                4000 if key == "clipboard.max_paste_chars"
                                else default),
            "hotkeys": {"import_laws": "ctrl+alt+i"}})()
        self._clipboard_text = clipboard_text

    def _answer(self, text, source):
        self.answers.append((text, source))

    def _show_dictation(self):
        self.shown += 1

    def _log(self, who, text):
        self.logged.append((who, text))

    def _status(self, key, detail=""):
        self.status.append((key, detail))


@pytest.fixture
def paste(monkeypatch):
    def run(text, mode=Mode.NORMAL):
        fake = Fake(text)
        fake.sessions.current.mode = mode
        monkeypatch.setattr("core.orchestrator.clipboard.get_text",
                            lambda *a, **k: text)
        fake._paste_text()
        return fake
    return run


def test_pasted_text_becomes_a_question(paste):
    fake = paste("что будет за превышение скорости")
    assert fake.answers == [("что будет за превышение скорости", Source.SELF)]


def test_pasted_text_joins_the_dictation(paste):
    """В режиме оформления вставка копится, как надиктованное."""
    fake = paste("истец Артём Кроу, паспорт 4471", Mode.DOCUMENT)
    assert fake._dictation == ["истец Артём Кроу, паспорт 4471"]
    assert fake.shown == 1
    assert fake.answers == []


def test_empty_clipboard_says_so(paste):
    fake = paste("   ")
    assert fake.answers == []
    assert any("пусто" in detail for _, detail in fake.status)


def test_line_breaks_are_flattened(paste):
    """Скопированное из форума приходит с переносами и двойными пробелами."""
    fake = paste("первая строка\n\n   вторая строка")
    assert fake.answers[0][0] == "первая строка вторая строка"


def test_a_whole_forum_page_is_refused(paste):
    """Для страницы правил есть импорт в базу, а не вопрос на 20 тысяч."""
    fake = paste("норма права. " * 1000)
    assert fake.answers == []
    assert any("ctrl+alt+i" in detail for _, detail in fake.status)


def test_clipboard_failure_does_not_crash(monkeypatch):
    fake = Fake("")

    def boom(*args, **kwargs):
        raise OSError("буфер занят")

    monkeypatch.setattr("core.orchestrator.clipboard.get_text", boom)
    fake._paste_text()
    assert any(key == "error" for key, _ in fake.status)


def test_hotkey_is_registered():
    hotkeys = Config.load().hotkeys
    assert hotkeys.get("paste_text")
    combos = list(hotkeys.values())
    assert len(combos) == len(set(combos)), "сочетания конфликтуют"


# ---------------------------------------------------- копирование ответа
class FakeCopy:
    """Оркестратор без Qt — только копирование ответа."""

    _copy_answer = Orchestrator._copy_answer

    def __init__(self, answer):
        self._last_answer = answer
        self.status = []
        self.logged = []

    def _log(self, who, text):
        self.logged.append((who, text))

    def _status(self, key, detail=""):
        self.status.append((key, detail))


def test_answer_goes_to_the_clipboard(monkeypatch):
    put = []
    monkeypatch.setattr("core.orchestrator.clipboard.set_text",
                        lambda text: put.append(text))
    fake = FakeCopy("Оскорбление — от 2 до 4 лет. [УК, п. 17.3]")
    fake._copy_answer()
    assert put == ["Оскорбление — от 2 до 4 лет. [УК, п. 17.3]"]
    assert any(key == "ready" for key, _ in fake.status)


def test_nothing_to_copy_is_explained(monkeypatch):
    monkeypatch.setattr("core.orchestrator.clipboard.set_text",
                        lambda text: pytest.fail("копировать было нечего"))
    fake = FakeCopy("")
    fake._copy_answer()
    assert any("нечего" in detail for _, detail in fake.status)


def test_busy_clipboard_is_reported(monkeypatch):
    def boom(text):
        raise OSError("буфер занят")

    monkeypatch.setattr("core.orchestrator.clipboard.set_text", boom)
    fake = FakeCopy("текст")
    fake._copy_answer()
    assert any(key == "error" for key, _ in fake.status)


def test_clipboard_roundtrip():
    """Настоящий буфер Windows: записали — прочитали то же самое."""
    from core import clipboard

    clipboard.set_text("проверка буфера обмена")
    assert clipboard.get_text() == "проверка буфера обмена"


def test_copy_hotkey_is_registered():
    hotkeys = Config.load().hotkeys
    assert hotkeys.get("copy_answer")
    assert hotkeys["copy_answer"] != hotkeys.get("paste_text")
