"""
Список дел: каждое дело — одна сессия, от сброса до сброса.

Прокурорское дело за один присест не закрывается, а вернуться к
прошлому разбору было нельзя вовсе. Название выводится из содержимого,
а не хранится отдельно: так оно не разъезжается с тем, что внутри, и
не требует называть дело руками посреди задержания.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.history import HistoryLog, _case_title  # noqa: E402
from core.orchestrator import _escape, _mode_title  # noqa: E402


class Entry:
    def __init__(self, mode="normal", question="", answer=""):
        self.mode = mode
        self.question = question
        self.answer = answer


@pytest.fixture
def history(tmp_path) -> HistoryLog:
    return HistoryLog(tmp_path / "history.db")


# ------------------------------------------------------------- название
def test_document_wins_over_questions():
    """Оформлялась бумага — дело называется по ней, а не по болтовне."""
    entries = [Entry("normal", "а что если", "неважно"),
               Entry("document", "оформление",
                     "ОПРЕДЕЛЕНИЕ о принятии искового заявления\nг. Лос-Сантос")]
    assert "ОПРЕДЕЛЕНИЕ о принятии" in _case_title(entries, "document")


def test_case_number_gets_into_the_title():
    entries = [Entry("document", "оформление",
                     "ИСКОВОЕ ЗАЯВЛЕНИЕ\nпо делу №6907 от гражданина")]
    title = _case_title(entries, "document")
    assert "6907" in title


def test_letter_number_is_recognised():
    entries = [Entry("document", "оформление",
                     "ПОСТАНОВЛЕНИЕ\nномер DJP-3505 об истребовании")]
    assert "DJP-3505" in _case_title(entries, "document")


def test_questions_are_titled_by_the_first_one():
    entries = [Entry("normal", "что будет за халатность", "..."),
               Entry("normal", "а за превышение", "...")]
    assert _case_title(entries, "normal").startswith("что будет за халатность")


def test_mode_shows_in_the_title():
    entries = [Entry("lawyer", "меня задержали", "...")]
    assert _case_title(entries, "lawyer").startswith("Задержание:")


def test_empty_case_is_named_not_crashed():
    assert _case_title([], "normal") == "Пустое дело"


def test_long_question_is_trimmed():
    entries = [Entry("normal", "очень длинный вопрос " * 20, "...")]
    assert len(_case_title(entries, "normal")) <= 70


# --------------------------------------------------------------- список
def test_cases_are_grouped_by_session(history):
    for index in range(3):
        history.record(session_id="дело-1", mode="normal", source="Я",
                       question=f"вопрос {index}", answer="ответ")
    history.record(session_id="дело-2", mode="normal", source="Я",
                   question="другая ситуация", answer="ответ")

    cases = history.cases()
    assert len(cases) == 2
    by_id = {c.session_id: c for c in cases}
    assert by_id["дело-1"].count == 3
    assert by_id["дело-2"].count == 1


def test_newest_case_comes_first(history):
    history.record(session_id="старое", mode="normal", source="Я",
                   question="первый", answer="о")
    history.record(session_id="новое", mode="normal", source="Я",
                   question="второй", answer="о")
    assert history.cases()[0].session_id in ("новое", "старое")
    assert len(history.cases()) == 2


def test_empty_history_gives_no_cases(history):
    assert history.cases() == []


def test_case_keeps_its_entries(history):
    history.record(session_id="дело", mode="normal", source="Я",
                   question="вопрос", answer="ответ")
    assert len(history.session("дело")) == 1


# ------------------------------------------------------ вывод в окне
def test_markup_is_escaped():
    """Названия берутся из надиктованного — там бывает и «<», и «&»."""
    assert _escape("Иск <Петров> & Ко") == "Иск &lt;Петров&gt; &amp; Ко"


@pytest.mark.parametrize("mode,expected", [
    ("normal", "обычный"), ("lawyer", "адвокат"),
    ("exam", "экзамен"), ("document", "документ"),
])
def test_mode_titles(mode, expected):
    assert _mode_title(mode) == expected
