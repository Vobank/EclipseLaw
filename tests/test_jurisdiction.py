"""
Подсудность и чужие данные в судебных документах.

Опытный судья, посмотрев первый же документ бота, спросил: «зачем он
передаёт дело в Верховный суд?» — и был прав. Бот написал определение
от лица Окружного суда и передал иск против рядового сотрудника LSPD
в Верховный, то есть ровно наоборот: такие дела подсудны Окружному.
Правил подсудности он не знал — в контекст они не попадали.

Там же вскрылось худшее: документ был подписан именем судьи из
образца. Чужое имя в деле — это подлог, а не опечатка.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.documents import invented_facts, redact_invented_facts  # noqa: E402
from core.prompts import document_prompt  # noqa: E402


# ------------------------------------------------------------ подсудность
def test_court_prompt_knows_jurisdiction():
    prompt = document_prompt("обр", "скелет", "надиктовка", court=True)
    assert "ПОДСУДНОСТЬ" in prompt
    assert "ОКРУЖНОЙ" in prompt and "ВЕРХОВНЫЙ" in prompt


def test_prompt_forbids_changing_the_court():
    prompt = document_prompt("обр", "скелет", "надиктовка", court=True)
    assert "СУД НЕ МЕНЯЙ САМ" in prompt


def test_prosecution_prompt_has_no_court_rules():
    """Прокурорскому делу правила подсудности в промпте ни к чему."""
    assert "ПОДСУДНОСТЬ" not in document_prompt("обр", "скелет", "надиктовка")


def test_rank_matters_for_the_supreme_court():
    prompt = document_prompt("обр", "скелет", "надиктовка", court=True)
    assert "высокопоставленных" in prompt.lower()


# ------------------------------------------------------- чужие данные
def test_judge_name_from_a_sample_is_caught():
    document = "Окружной суд, в составе судьи Tyler Tea, рассмотрев дело"
    assert any("Tyler Tea" in item for item in
               invented_facts(document, "истец Артём Кроу"))


def test_case_number_from_a_sample_is_caught():
    document = "искового заявления №6950 от гражданина"
    assert any("6950" in item for item in
               invented_facts(document, "истец Артём Кроу"))


def test_letter_case_number_is_caught():
    document = "Постановление DJP-3505 об истребовании"
    assert any("DJP-3505" in item for item in
               invented_facts(document, "прокурор запросил данные"))


def test_names_of_places_are_not_invented_facts():
    """«Los Santos» и «San Andreas» стоят в любом документе по делу."""
    document = "Заседание в Los Santos, штат San Andreas"
    assert invented_facts(document, "истец Артём Кроу") == []


def test_dictated_name_survives():
    document = "против сотрудника Daniel Morse"
    assert invented_facts(document, "ответчик Daniel Morse, жетон 417") == []


def test_badge_number_matches_despite_formatting():
    """В надиктовке «жетон 417», в документе «№417» — один и тот же факт."""
    assert invented_facts("знак №417", "жетон 417, LSPD") == []


def test_invented_values_are_actually_removed():
    document = "судья Tyler Tea, дело №6950, постановление DJP-3505"
    clean, invented = redact_invented_facts(document, "истец Артём Кроу")
    assert "Tyler Tea" not in clean
    assert "6950" not in clean
    assert "DJP-3505" not in clean
    assert len(invented) == 3


def test_two_word_kind_does_not_break_removal():
    """Регрессия: тип «номер дела» из двух слов ломал разбор, и номер
    оставался в документе, хотя и попадал в список вычеркнутого."""
    clean, _ = redact_invented_facts("дело №7001 принято", "истец Кроу")
    assert "7001" not in clean
