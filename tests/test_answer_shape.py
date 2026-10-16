"""
Форма ответа зависит от вопроса, а не только от режима.

«Кто относится к неприкосновенным лицам» — это перечень, который
зачитывают вслух: список и ссылка в конце, без рассуждений. «Что
говорит статья 7.1» — просьба объяснить названную статью: номер человек
уже знает, ему нужно, о чём она. Один формат на оба случая одинаково
плох.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.answer import fold_citations  # noqa: E402
from core.legal.query import is_list_question  # noqa: E402
from core.orchestrator import _answer_shape  # noqa: E402
from core.prompts import system_prompt  # noqa: E402
from core.session import Mode  # noqa: E402


# ------------------------------------------------------- распознавание
@pytest.mark.parametrize("question", [
    "кто относится к неприкосновенным лицам",
    "какие бывают ордера",
    "перечисли основания для задержания",
    "что входит в состав следствия",
    "неприкосновенные лица кто такие",
])
def test_listing_questions(question):
    assert _answer_shape(question) == "listing"


@pytest.mark.parametrize("question", [
    "что говорит статья 7.1 процессуального кодекса",
    "статья 17.1 УК о чём",
])
def test_article_questions(question):
    assert _answer_shape(question) == "article"


@pytest.mark.parametrize("question", [
    "что будет за превышение скорости",
    "меня задержали что делать",
])
def test_ordinary_questions_keep_the_usual_shape(question):
    assert _answer_shape(question) == ""


def test_law_text_is_not_a_listing_question():
    """Поправка обязана молчать на тексте самих норм."""
    assert not is_list_question(
        "К неприкосновенным лицам относятся Губернатор и его заместители")


# -------------------------------------------------------------- промпт
def test_listing_prompt_asks_for_a_list():
    prompt = system_prompt(Mode.NORMAL, legal_context="выдержка",
                           rag_enabled=True, shape="listing")
    assert "ПЕРЕЧЕНЬ" in prompt


def test_exam_keeps_its_own_shape():
    """На экзамене нужен связный ответ, а не список — форма не меняется."""
    prompt = system_prompt(Mode.EXAM, legal_context="выдержка",
                           rag_enabled=True, shape="listing")
    assert "ЭТОТ ВОПРОС — ПЕРЕЧЕНЬ" not in prompt


# --------------------------------------------------- сворачивание ссылок
def test_repeated_citations_are_folded():
    text = ("- Ордер AR даёт право на рейд: [Закон \"Об ордерах\", п. 4.2]\n"
            "- Ордер AS даёт право на задержание: "
            "[Закон \"Об ордерах\", п. 4.3]")
    folded = fold_citations(text)
    assert folded.count("Закон \"Об ордерах\"") == 1
    assert "п. 4.2, 4.3" in folded


def test_different_documents_are_left_alone():
    """Разные документы — ссылка в строке несёт смысл, трогать нельзя."""
    text = ("- Первое: [Уголовный Кодекс, п. 1.1]\n"
            "- Второе: [Процессуальный Кодекс, п. 2.2]")
    assert fold_citations(text) == text


def test_single_line_is_left_alone():
    text = "- Единственный пункт: [Уголовный Кодекс, п. 1.1]"
    assert fold_citations(text) == text


def test_ordinary_answer_is_untouched():
    text = ("Превышение скорости влечёт штраф.\n"
            "[Дорожный Кодекс, п. 32]")
    assert fold_citations(text) == text


def test_folded_citation_sits_right_after_the_list():
    """Дальше в ответе бывает ещё текст — ссылка к нему не относится."""
    text = ("- Первое: [Закон, п. 1.1]\n"
            "- Второе: [Закон, п. 1.2]\n"
            "\n"
            "Дополнительное замечание.")
    lines = fold_citations(text).split("\n")
    assert lines[2].startswith("[Закон")
    assert lines[-1] == "Дополнительное замечание."
