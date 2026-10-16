"""
Ранжирование, схлопывание дублей и разбор ссылок в ответе.

Работает на подставных статьях, без базы и без Ollama: здесь
проверяется логика, а не содержимое базы. Содержимое стережёт
tools/audit_base.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.legal.db import StoredRule  # noqa: E402
from core.legal.retriever import (Hit, Retriever, _collapse_duplicates,
                                  _document_priority)  # noqa: E402


def rule(rule_id: int, document: str, number: str, text: str,
         sanction: str = "") -> StoredRule:
    return StoredRule(id=rule_id, document=document, number=number,
                      section_number="", section_title="", text=text,
                      sanction=sanction, notes=[])


def hit(r: StoredRule, score: float = 0.03) -> Hit:
    return Hit(r, score, ["смысл"])


# --------------------------------------------------------------- приоритет
@pytest.mark.parametrize("title,expected", [
    ("Уголовный Кодекс", 0),
    ("Конституция Штата Сан-Андреас", 0),
    ("Закон \"Об оружии\"", 1),
    ("Внутренний устав Los Santos Police Department", 2),
    ("Правила зеленых зон", 3),
])
def test_document_priority(title, expected):
    assert _document_priority(title) == expected


# ------------------------------------------------------------ схлопывание
def test_identical_text_collapses_to_one():
    """Одна норма в двух уставах занимала два места в выдаче."""
    text = ("Устраивать беспорядок на парковке, халатно и небрежно "
            "относиться к служебной технике.")
    hits = [hit(rule(1, "Внутренний устав LSPD", "22.16", text), 0.031),
            hit(rule(2, "Устав LSSD", "23.13", text), 0.030)]
    result = _collapse_duplicates(hits)
    assert len(result) == 1
    assert result[0].rule.document == "Внутренний устав LSPD"
    assert result[0].duplicates == ["Устав LSSD, п. 23.13"]


def test_identical_short_rules_collapse_too():
    """Одинаковый заголовок в пяти темах — это всё равно одна строка.

    Порог сначала стоял на 40 символах, чтобы не слипались короткие
    пункты вроде «Запрещено.». Оказалось, что вреда от этого больше:
    на вопрос о сроке действия пропуска четыре места из пяти заняла
    строка «Срок действия пропуска.» из пяти одинаковых тем.
    """
    hits = [hit(rule(1, "Пропуска LSPD", "2.6", "Срок действия пропуска.")),
            hit(rule(2, "Пропуска FIB", "II.6", "Срок действия пропуска."))]
    result = _collapse_duplicates(hits)
    assert len(result) == 1
    assert result[0].duplicates == ["Пропуска FIB, п. II.6"]


def test_very_short_rules_are_still_left_alone():
    """На совсем куцых строках совпадение уже ничего не значит."""
    hits = [hit(rule(1, "Устав A", "1.1", "Нет.")),
            hit(rule(2, "Устав Б", "2.1", "Нет."))]
    assert len(_collapse_duplicates(hits)) == 2


def test_different_text_is_kept():
    hits = [hit(rule(1, "Уголовный Кодекс", "15.6",
                     "Халатность, то есть неисполнение обязанностей.")),
            hit(rule(2, "Устав LSPD", "22.22",
                     "Халатно относиться к амуниции и снаряжению."))]
    assert len(_collapse_duplicates(hits)) == 2


# --------------------------------------------------- ссылки в тексте ответа
def test_cited_rules_finds_plain_and_roman_numbers():
    hits = [hit(rule(1, "Уголовный Кодекс", "17.1", "Посягательство на жизнь.")),
            hit(rule(2, "Процессуальный Кодекс", "III.2", "Выбор учреждения.")),
            hit(rule(3, "Закон \"Об оружии\"", "4.7", "Хранение оружия."))]
    answer = "Смотрите [Уголовный Кодекс, п. 17.1], а также III.2 и 4.7."
    numbers = [r.number for r in Retriever.cited_rules(answer, hits)]
    assert numbers == ["17.1", "III.2", "4.7"]


@pytest.mark.parametrize("answer", [
    "В течение 48 часов нужно обратиться в LSC.",
    "Штраф составляет до 5.000$ по решению суда.",
    "Срок наказания — 1 год лишения свободы.",
])
def test_numbers_in_prose_are_not_citations(answer):
    """Суммы, сроки и часы — не ссылки на статьи."""
    hits = [hit(rule(1, "Уголовный Кодекс", "48", "Некая статья.")),
            hit(rule(2, "Уголовный Кодекс", "5.0", "Другая статья.")),
            hit(rule(3, "Уголовный Кодекс", "1", "Третья статья."))]
    assert Retriever.cited_rules(answer, hits) == []


def test_unsupported_citations_catches_invented_numbers():
    hits = [hit(rule(1, "Уголовный Кодекс", "17.1", "Посягательство."))]
    answer = "Это статья 17.1, а ещё статья 14.1 со штрафом."
    assert Retriever.unsupported_citations(answer, hits) == ["14.1"]


def test_subitem_of_given_article_is_supported():
    """Уточнение подпункта — не выдумка нормы."""
    hits = [hit(rule(1, "Процессуальный Кодекс", "6.2", "Недопустимые доказательства."))]
    assert Retriever.unsupported_citations("см. пункт 6.2.1", hits) == []
