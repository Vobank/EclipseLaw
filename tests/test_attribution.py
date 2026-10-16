"""
Привлекать сотрудника можно по составу, а не по порядку.

Модель раз за разом писала в заключении «привлечь по статье 6.2
Процессуального кодекса»: номер настоящий, статья существует, проверка
выдуманных номеров молчит. Но 6.2 — это норма о недопустимых
доказательствах, ответственности она не устанавливает, и требование
по ней суд отклонит. Просьбы в промпте не хватило — проверка нужна
детерминированная.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.answer import wrong_attribution  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402


class Rule:
    def __init__(self, number, document, text):
        self.id = id(self)
        self.number, self.document, self.text = number, document, text


class Hit:
    def __init__(self, rule):
        self.rule = rule


PROCEDURE = Rule("6.2", "Процессуальный Кодекс (ред. от 02.07.2026)",
                 "К недопустимым доказательствам относятся: любая информация, "
                 "полученная от задержанного до зачтения его прав")
MISCONDUCT = Rule("15.1", "Уголовный Кодекс (ред. от 28.07.2026)",
                  "Превышение должностных полномочий, то есть совершение "
                  "сотрудником действий, явно выходящих за пределы его прав")
NEGLIGENCE = Rule("15.6", "Уголовный Кодекс (ред. от 28.07.2026)",
                  "Халатность, то есть неисполнение должностных обязанностей "
                  "вследствие недобросовестного отношения к службе")

HITS = [Hit(PROCEDURE), Hit(MISCONDUCT), Hit(NEGLIGENCE)]


def _verdict(section: str) -> str:
    return ("1. ОБСТОЯТЕЛЬСТВА\nЗадержание прошло без зачтения прав.\n"
            "2. ВЫЯВЛЕННЫЕ НАРУШЕНИЯ\nПрава не зачтены.\n"
            f"3. СТАТЬЯ ДЛЯ ПРИВЛЕЧЕНИЯ СОТРУДНИКА\n{section}\n"
            "4. ТРЕБОВАНИЕ КОМПЕНСАЦИИ\nОснований нет.\n"
            "5. РЕКОМЕНДАЦИЯ ПОДЗАЩИТНОМУ\nОбратиться в суд.")


def test_procedural_article_is_flagged():
    text = _verdict("Сотрудник может быть привлечён по статье 6.2 "
                    "Процессуального кодекса.")
    note = wrong_attribution(Retriever, text, HITS)
    assert "Процессуальному кодексу нельзя" in note


def test_the_real_article_is_offered():
    """Мало сказать «не то»: состав лежит в тех же выдержках."""
    text = _verdict("Привлечь по статье 6.2 Процессуального кодекса.")
    note = wrong_attribution(Retriever, text, HITS)
    assert "15.1" in note


def test_proper_article_passes_silently():
    text = _verdict("Сотрудник привлекается по статье 15.1 Уголовного кодекса.")
    assert wrong_attribution(Retriever, text, HITS) == ""


def test_mixed_citation_passes():
    """Процедура рядом с составом — это нормальное обоснование."""
    text = _verdict("Нарушена статья 6.2, привлекать по статье 15.1.")
    assert wrong_attribution(Retriever, text, HITS) == ""


def test_no_composition_available_says_so():
    text = _verdict("Привлечь по статье 6.2 Процессуального кодекса.")
    note = wrong_attribution(Retriever, text, [Hit(PROCEDURE)])
    assert "состава среди найденных статей нет" in note


def test_section_without_citations_is_left_alone():
    """«Нарушений нет» — законный ответ, придираться не к чему."""
    text = _verdict("Нарушений не выявлено, привлекать не за что.")
    assert wrong_attribution(Retriever, text, HITS) == ""


def test_verdict_without_the_section_is_left_alone():
    assert wrong_attribution(Retriever, "Произвольный текст без разделов",
                             HITS) == ""
