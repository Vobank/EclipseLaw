"""Чек-лист адвоката: формулировки вопросов и отметки пройденного."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.legal.checklist import Checklist, ChecklistItem  # noqa: E402
from core.legal.db import StoredRule  # noqa: E402


def rule(document: str, number: str, text: str = "Некая норма.") -> StoredRule:
    return StoredRule(id=1, document=document, number=number,
                      section_number="", section_title="", text=text,
                      sanction="", notes=[])


def item(number="6.2", question="Согласно {ссылка}, подтвердите права.",
         document="Процессуальный Кодекс (ред. от 02.09.2026)"):
    return ChecklistItem(key="права", title="Права", question=question,
                         articles=[rule(document, number)])


def test_spoken_question_names_article_and_document():
    spoken = item().spoken_question()
    assert "статье 6.2" in spoken
    # Редакция из названия убирается: в оверлее она занимает полстроки.
    assert "(Процессуальный Кодекс)" in spoken
    assert "ред. от" not in spoken


def test_question_without_article_loses_reference_cleanly():
    bare = ChecklistItem(key="x", title="X",
                         question="Согласно {ссылка}, объясните.", articles=[])
    assert "{ссылка}" not in bare.spoken_question()


# Подстрочное совпадение номера однажды отмечало чужой пункт пройденным,
# а отмеченный пункт адвокат уже не задаст.
@pytest.mark.parametrize("text,covered", [
    ("нарушена статья 6.2 кодекса", True),
    ("см. пункт 6.2.1 того же кодекса", True),   # уточнение подпункта
    ("речь о статье 16.2, а не о другой", False),
    ("применяется пункт 6.22 устава", False),
    ("в 2026.2 году", False),
])
def test_mark_covered_respects_number_boundaries(text, covered):
    checklist = Checklist(items=[item()])
    assert bool(checklist.mark_covered(text)) is covered


def test_next_question_skips_covered():
    first = item(number="6.2")
    second = ChecklistItem(key="адвокат", title="Адвокат",
                           question="Согласно {ссылка}, требую защитника.",
                           articles=[rule("Закон \"Об адвокатуре\"", "7.1")])
    checklist = Checklist(items=[first, second])
    assert checklist.next_question().key == "права"
    checklist.mark_covered("нарушена статья 6.2")
    assert checklist.next_question().key == "адвокат"


def test_items_without_articles_are_not_offered():
    """Спрашивать без ссылки на норму нельзя — это блеф в задержании."""
    checklist = Checklist(items=[
        ChecklistItem(key="x", title="X", question="Согласно {ссылка}, ...")])
    assert checklist.grounded == []
    assert checklist.next_question() is None
