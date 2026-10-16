"""
Под ответом — суть статьи, дословный текст — по клавише.

Раньше под каждым ответом печаталась статья целиком, и ответ
разрастался на экран текста. В разговоре с сотрудником читают не его:
нужна суть, а дословная формулировка — только когда на неё ссылаются
вслух. Поэтому текст ушёл под отдельную клавишу, которая ничего не
генерирует, а печатает норму из базы.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.orchestrator import Orchestrator  # noqa: E402
from core.session import Mode  # noqa: E402


class Rule:
    def __init__(self, number, document, text, sanction=""):
        self.id = id(self)
        self.number, self.document = number, document
        self.text, self.sanction = text, sanction

    def citation(self):
        return f"{self.document}, п. {self.number}"

    def gist(self, limit=150):
        head = self.text.split(".")[0]
        return head[:limit]


LONG = Rule("15.6", "Уголовный Кодекс",
            "Халатность, то есть неисполнение должностным лицом своих "
            "обязанностей. " + "Дополнительные условия применения нормы. " * 40,
            "от 3 до 6 лет")


class Overlay:
    def __init__(self):
        self.chunks = []
        self.sig_answer_chunk = type(
            "S", (), {"emit": lambda _self, text: self.chunks.append(text)})()


class Session:
    mode = Mode.NORMAL


class Sessions:
    current = Session()


class Fake:
    """Оркестратор без Qt, аудио и моделей — только нужные методы."""

    _explain_cited = Orchestrator._explain_cited
    _show_full_article = Orchestrator._show_full_article

    def __init__(self, cited):
        self.overlay = Overlay()
        self.sessions = Sessions()
        self._last_cited = []
        self._status_calls = []
        self.retriever = type("R", (), {
            "cited_rules": staticmethod(lambda answer, hits: cited)})()
        self.cfg = type("C", (), {
            "get": staticmethod(lambda key, default=None: {
                "retrieval.explain_articles": 4,
                "retrieval.gist_chars": 160,
                "retrieval.explain_chars": 1600,
            }.get(key, default)),
            "hotkeys": {"full_article": "ctrl+alt+s"}})()

    def _status(self, *args):
        self._status_calls.append(args)

    def _hotkey_hint(self):
        return ""


def test_block_under_the_answer_is_short():
    fake = Fake([LONG])
    block = fake._explain_cited("по статье 15.6", [])
    assert len(block) < 400, "под ответом снова печатается статья целиком"
    assert "15.6" in block


def test_block_points_at_the_key():
    fake = Fake([LONG])
    assert "ctrl+alt+s" in fake._explain_cited("по статье 15.6", [])


def test_sanction_is_kept_in_the_short_block():
    """Срок — то, ради чего вопрос и задан: сокращать его нельзя."""
    fake = Fake([LONG])
    assert "от 3 до 6 лет" in fake._explain_cited("по статье 15.6", [])


def test_full_text_comes_by_the_key():
    fake = Fake([LONG])
    fake._explain_cited("по статье 15.6", [])
    fake.overlay.chunks.clear()
    fake._show_full_article()
    printed = "".join(fake.overlay.chunks)
    assert "СТАТЬЯ ЦЕЛИКОМ" in printed
    assert len(printed) > 900, "дословный текст не развернулся"


def test_key_without_a_question_does_not_crash():
    fake = Fake([])
    fake._show_full_article()
    assert fake.overlay.chunks == []
    assert fake._status_calls, "человеку не сказали, почему ничего нет"


def test_exam_gets_no_block_but_remembers_articles():
    """На экзамене пересказ лишний, но развернуть статью всё равно можно."""
    fake = Fake([LONG])
    fake.sessions.current.mode = Mode.EXAM
    assert fake._explain_cited("по статье 15.6", []) == ""
    assert fake._last_cited == [LONG]
    fake._show_full_article()
    assert "СТАТЬЯ ЦЕЛИКОМ" in "".join(fake.overlay.chunks)
