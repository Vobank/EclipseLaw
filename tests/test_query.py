"""
Приведение живого вопроса к виду для поиска.

Здесь стерегутся две ошибки, каждая из которых уже случалась и каждая
тихая — поиск не падает, он просто находит не то.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.legal.query import is_liability_question, normalize  # noqa: E402


@pytest.fixture(scope="module")
def slang():
    return Config.load().get("retrieval.slang")


def test_slang_adds_legal_wording(slang):
    """«Мент» и «обозвать» в базе не встречаются ни разу."""
    result = normalize("что будет если я обозву мента", slang, None)
    assert "оскорбление" in result
    assert "представител" in result
    # Исходные слова остаются: вдруг человек имел в виду именно их.
    assert "обозву" in result
    assert "мента" in result


def test_filler_is_dropped(slang):
    """Разговорная обвязка размывает вектор запроса."""
    result = normalize("что там за халатность будет", slang, None)
    assert "халатность" in result
    assert "будет" not in result


def test_punctuation_and_case_survive(slang):
    """Пересборка запроса из токенов однажды стоила статьи.

    Закон о военном положении перестал находиться по собственному
    тексту: «USSS» превращалось в «usss», а запятые исчезали, и
    вектор менялся достаточно, чтобы статья вылетела из первой пятёрки.
    """
    text = "Протокол вводится Директором USSS или его заместителем."
    assert normalize(text, slang, None) == text


def test_query_of_only_filler_survives(slang):
    """Если вычеркнуть нечего кроме обвязки — возвращаем как было."""
    assert normalize("что там", slang, None).strip() != ""


# Вопрос про наказание: кодекс должен обойти внутренний устав.
@pytest.mark.parametrize("question", [
    "что там за халатность будет",
    "что будет если я обозву мента",
    "какая статья за превышение скорости",
    "сколько дадут за разбой",
    "чем грозит неповиновение",
])
def test_liability_questions_recognised(question):
    assert is_liability_question(question)


# А это — текст самих законов. Поправка тут срабатывать не должна:
# однажды она сработала и увела три статьи из выдачи, потому что
# слова «штраф» и «нарушение» стоят внутри обычных норм.
@pytest.mark.parametrize("text", [
    "штраф, в случае непрохождения проверок по принципу 5",
    "должно быть обусловлено конкретным нарушением действующего закона",
    "USSS или его заместителем, начальником отдела USSS",
    "Ответственность за нарушение данного кодекса предусмотрена в виде "
    "штрафа в казну штата",
])
def test_law_text_is_not_a_liability_question(text):
    assert not is_liability_question(text)


# ------------------------------------------------- вопросы о порядке и правах
from core.legal.query import is_procedure_question  # noqa: E402


@pytest.mark.parametrize("question", [
    "меня задержали без объяснения причин что делать",
    "имеет ли право сотрудник обыскать мою машину",
    "обязан ли я показывать документы",
    "сколько можно держать в кпз",
    "как обжаловать приговор",
    "это правомерно вообще",
])
def test_procedure_questions_are_recognised(question):
    assert is_procedure_question(question)


@pytest.mark.parametrize("question", [
    "что будет за превышение скорости",
    "за какую статью меня задержат если я убил полицейского",
    "сколько дадут за разбой",
])
def test_punishment_questions_are_not_procedure(question):
    """Слово «задержат» здесь есть, а нужен всё равно состав, а не порядок."""
    assert is_liability_question(question)
    assert not is_procedure_question(question)


@pytest.mark.parametrize("text", [
    "Задержание - это мера кратковременного лишения человека свободы",
    "Сотрудник обязан вести и хранить записи о задержании",
    "Обвиняемый считается невиновным, пока его вина не доказана",
])
def test_law_text_is_not_mistaken_for_a_question(text):
    """Поправка обязана молчать на тексте самих норм, иначе она везде."""
    assert not is_procedure_question(text)
