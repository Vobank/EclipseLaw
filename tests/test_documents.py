"""
Генератор документов: подбор образцов и защита от выдуманных фактов.

Образцов судопроизводства стало 29, и они разного вида — определения,
решения, приговор, судебный ордер. Показать модели приговор, когда
диктуется определение, значит получить бумагу не того вида: раньше
брались просто первые два образца подряд.

Вторая проверка — про выдуманное. В прокурорском деле подставленное
время не опечатка, а фальсификация: по нему считают процессуальные
сроки, и перепроверять его никто не станет.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.documents import (Template, TemplateSet,  # noqa: E402
                            invented_facts, redact_invented_facts)


def court_set() -> TemplateSet:
    return TemplateSet(kind="court", title="Судопроизводство", templates=[
        Template("ОПРЕДЕЛЕНИЕ о назначении судебного заседания", "court",
                 "Назначить судебное заседание, известить стороны о дате "
                 "и времени проведения заседания, обеспечить явку."),
        Template("ПРИГОВОР", "court",
                 "Признать подсудимого виновным в совершении преступления, "
                 "назначить наказание в виде лишения свободы."),
        Template("ОПРЕДЕЛЕНИЕ о принятии искового заявления", "court",
                 "Принять исковое заявление к производству, возбудить "
                 "гражданское дело, истребовать материалы."),
    ])


# ------------------------------------------------------- подбор образцов
def test_sample_matches_the_dictation():
    chosen = court_set()._closest(
        "прошу принять исковое заявление к производству и возбудить дело", 1)
    assert "принятии искового" in chosen[0].name


def test_another_dictation_picks_another_sample():
    chosen = court_set()._closest(
        "признать подсудимого виновным и назначить наказание", 1)
    assert chosen[0].name == "ПРИГОВОР"


def test_without_dictation_order_is_kept():
    """Пустая надиктовка — берём как лежат, гадать не на чем."""
    assert court_set().few_shot(limit=1).startswith(
        "=== ОБРАЗЕЦ: ОПРЕДЕЛЕНИЕ о назначении")


def test_unrelated_dictation_still_returns_something():
    """Ничего не совпало — лучше любой образец, чем пустой промпт."""
    assert court_set()._closest("погода в Лос-Сантосе", 2)


def test_few_shot_passes_the_dictation_through():
    text = court_set().few_shot(limit=1, like="назначить наказание виновному")
    assert "ПРИГОВОР" in text


# --------------------------------------------------- выдуманные факты
@pytest.mark.parametrize("value", ["14:22", "15.09.2026", "811204"])
def test_invented_values_are_found(value):
    assert invented_facts(f"Задержан в {value}.", "ничего такого") != []


def test_values_from_the_dictation_are_left_alone():
    dictation = "паспорт 811204, задержание пятнадцатого в 14:22"
    document = "Задержан в 14:22, паспорт 811204."
    assert invented_facts(document, dictation) == []


def test_invented_time_is_replaced_not_just_reported():
    """Предупреждения мало: бумага уходит в дело целиком."""
    document = "Задержан 15.09.2026 в 14:22, паспорт 811204."
    text, invented = redact_invented_facts(document, "паспорт 811204")
    assert "14:22" not in text
    assert "15.09.2026" not in text
    # Настоящий факт из надиктовки трогать нельзя.
    assert "811204" in text
    assert "[НЕ УКАЗАНО]" in text
    assert len(invented) == 2


def test_clean_document_is_returned_unchanged():
    document = "Задержан, паспорт 811204."
    text, invented = redact_invented_facts(document, "паспорт 811204")
    assert text == document
    assert invented == []
