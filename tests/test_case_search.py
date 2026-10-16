"""
Поиск по делам: найти иск по номеру, а не листать список.

За неделю работы с прокуратурой дел набирается столько, что список
перестаёт помогать. Искать надо по номеру постановления вроде
«DJP-3505», а он лежит внутри текста самого документа — значит поиск
обязан смотреть и в содержимое, а не только в название.

Отдельная сложность: окно ассистента намеренно не получает фокус,
иначе оно крало бы его у игры. Обычное поле ввода в нём не работает —
буквы уходят в игровой чат. Поэтому их забирает клавиатурный хук.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.history import HistoryLog  # noqa: E402
from core.hotkeys import HotkeyManager  # noqa: E402


@pytest.fixture
def history(tmp_path) -> HistoryLog:
    log = HistoryLog(tmp_path / "history.db")
    log.record(session_id="дело-1", mode="document", source="Я",
               question="оформи постановление",
               answer="ПОСТАНОВЛЕНИЕ\nномер DJP-3505 об истребовании")
    log.record(session_id="дело-2", mode="document", source="Я",
               question="оформи иск",
               answer="ИСКОВОЕ ЗАЯВЛЕНИЕ\nпо делу №6907 от гражданина")
    log.record(session_id="дело-3", mode="normal", source="Я",
               question="что будет за превышение скорости",
               answer="Штраф по Дорожному кодексу")
    return log


# --------------------------------------------------------------- поиск
def test_case_is_found_by_its_number(history):
    found = history.find_cases("DJP-3505")
    assert [c.session_id for c in found] == ["дело-1"]


def test_number_without_the_letters_also_works(history):
    """Человек помнит «6907», а не «ИСКОВОЕ ЗАЯВЛЕНИЕ по делу №6907»."""
    assert [c.session_id for c in history.find_cases("6907")] == ["дело-2"]


def test_search_by_a_word_from_the_question(history):
    found = history.find_cases("скорости")
    assert "дело-3" in [c.session_id for c in found]


def test_empty_query_gives_the_whole_list(history):
    """Пустая строка — это ещё не поиск, список отнимать нельзя."""
    assert len(history.find_cases("")) == 3


def test_nothing_matches_gives_nothing(history):
    assert history.find_cases("вертолёт") == []


def test_title_matches_come_first(history):
    """В названии уже стоит номер — это точнее, чем слово в середине."""
    found = history.find_cases("постановление")
    assert found[0].session_id == "дело-1"


def test_case_is_not_listed_twice(history):
    """Дело подходит и по названию, и по тексту — строка всё равно одна."""
    found = history.find_cases("DJP")
    assert len(found) == len({c.session_id for c in found})


# ------------------------------------------------------------ набор текста
@pytest.fixture
def manager():
    events = []
    mgr = HotkeyManager(events.append)
    return mgr


def _typed(mgr):
    out = []
    while not mgr._queue.empty():
        out.append(mgr._queue.get_nowait())
    return out


def test_backspace_is_reported(manager):
    manager.capture_text(True)
    assert manager._handle_text(0x08, 0) is True
    assert [e.action for e in _typed(manager)] == ["search_back"]


def test_escape_closes_the_search(manager):
    manager.capture_text(True)
    manager._handle_text(0x1B, 0)
    assert [e.action for e in _typed(manager)] == ["search_close"]


def test_enter_opens_the_first_case(manager):
    manager.capture_text(True)
    manager._handle_text(0x0D, 0)
    assert [e.action for e in _typed(manager)] == ["search_open"]


def test_modifiers_are_not_swallowed(manager):
    """Сам Shift — не текст, и глотать его нельзя."""
    manager.capture_text(True)
    assert manager._handle_text(0xA0, 0) is False
    assert _typed(manager) == []


def test_capture_is_off_by_default(manager):
    """Пока поиск закрыт, клавиатура целиком принадлежит игре."""
    assert manager._capture is False


# ------------------------------------------------------------- отрисовка
class Overlay:
    def __init__(self):
        self.html = ""
        self.tab_open = None

    def show_html(self, html):
        self.html = html

    def set_cases_tab(self, showing):
        self.tab_open = showing

    def show_transcript(self):
        self.html = ""


class Fake:
    """Оркестратор без Qt и моделей — только строка поиска."""

    from core.orchestrator import Orchestrator as _O
    _render_search = _O._render_search
    _toggle_search = _O._toggle_search
    _close_search = _O._close_search
    _search_type = _O._search_type

    def __init__(self, history):
        self.history = history
        self.overlay = Overlay()
        self._search_query = ""
        self._search_hits = []
        self._search_active = False
        self._showing_cases = False
        self.captured = []
        self.hotkeys = type("H", (), {
            "capture_text": lambda _s, on: self.captured.append(on)})()
        self.cfg = type("C", (), {
            "get": staticmethod(lambda key, default=None: default),
            "hotkeys": {"search_cases": "ctrl+alt+f"}})()

    def _status(self, *args):
        pass

    def _hotkey_hint(self):
        return ""


def test_typing_narrows_the_list(history):
    fake = Fake(history)
    fake._toggle_search()
    assert len(fake._search_hits) == 3, "пустая строка — весь список"
    for char in "DJP":
        fake._search_type(char)
    assert [c.session_id for c in fake._search_hits] == ["дело-1"]


def test_backspace_widens_it_again(history):
    fake = Fake(history)
    fake._toggle_search()
    for char in "DJP":
        fake._search_type(char)
    fake._search_type("", back=True)
    fake._search_type("", back=True)
    fake._search_type("", back=True)
    assert len(fake._search_hits) == 3


def test_keyboard_is_grabbed_only_while_open(history):
    """Пока поиск закрыт, клавиатура целиком у игры."""
    fake = Fake(history)
    fake._toggle_search()
    assert fake.captured == [True]
    fake._close_search()
    assert fake.captured == [True, False]


def test_markup_is_escaped(history):
    """Номер иска надиктован голосом — в нём бывает и «<», и «&»."""
    fake = Fake(history)
    fake._toggle_search()
    fake._search_type("<")
    assert "&lt;" in fake.overlay.html


def test_cases_are_clickable(history):
    fake = Fake(history)
    fake._toggle_search()
    assert "href='case:дело-1'" in fake.overlay.html
