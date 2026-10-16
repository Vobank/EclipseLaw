"""
Разбор страниц форума: нумерация, санкции, склейки.

Каждый случай здесь — не выдуманный пример, а строка, на которой
парсер однажды спотыкался. Поэтому проверки короткие и конкретные:
они стерегут уже исправленные ошибки от повторного появления.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.legal.parser import ForumParser  # noqa: E402


@pytest.fixture(scope="module")
def parser() -> ForumParser:
    """Парсер, собранный по рабочему конфигу, а не по умолчаниям кода.

    Словарь слов-санкций живёт в config.yaml и переопределяет значения
    в коде. Проверять умолчания смысла нет: работает то, что в конфиге.
    """
    return ForumParser.from_config(Config.load())


# Санкция после «|» — основной формат уставов и правил фракций.
@pytest.mark.parametrize("text,expected", [
    ("Одна банда может забить не более 2-х RP-стрел за неделю. | "
     "-40 баллов лидеру.", "-40 баллов лидеру."),
    ("Отдыхать более 15 минут в рабочее время | Наряд/Предупреждение/Выговор.",
     "Наряд/Предупреждение/Выговор."),
    ("Увольнение с нарушением процедуры | Занесение в ЧС Армии.",
     "Занесение в ЧС Армии."),
    # «наказание» целиком тут не встречается — нужен стем «наказани».
    ("Использовать уязвимости | Мера наказания определяется Генералитетом.",
     "Мера наказания определяется Генералитетом."),
    ("Запрещено покидать пост | Отстранение от должности на 3 дня.",
     "Отстранение от должности на 3 дня."),
])
def test_sanction_after_pipe(parser, text, expected):
    body, sanction = parser._split_sanction(text)
    assert sanction == expected
    assert "|" not in body


# «|» бывает и обычным разделителем. Принять его за санкцию —
# значит откусить у пункта половину текста.
@pytest.mark.parametrize("text", [
    "Проведение алкогольного | наркологического теста",
    "Жетон образца: [LSPD | Название отдела | Номер жетона]",
    "Порядок описан в главе 4 | см. приложение",
])
def test_pipe_that_is_not_a_sanction(parser, text):
    body, sanction = parser._split_sanction(text)
    assert sanction == ""
    assert body == text.strip()


def test_sanction_after_dash_in_codex(parser):
    """В кодексах санкция отделена тире, а не чертой."""
    text = ("Получение взятки в виде денег. - от 3 до 6 лет лишения "
            "свободы. ((от 30 до 60 мин))")
    body, sanction = parser._split_sanction(text)
    assert "лишения свободы" in sanction
    assert "взятки" in body


def test_leading_pipe_is_not_a_sanction(parser):
    """Ведущая черта — разделитель после номера, а не санкция.

    «1.1 | Назначение Устава...» в уставе FIB: без обрезки она
    оставалась в тексте у 109 пунктов из 119.
    """
    body, sanction = parser._split_sanction("| Назначение Устава. Настоящий "
                                            "Устав определяет порядок.")
    assert sanction == ""
    assert body.startswith("Назначение Устава")


def test_hierarchical_and_flat_numbering(parser):
    """Оба вида нумерации, которые встречаются на форуме."""
    raw = (
        "Правила\n"
        "#1\n"
        "1.1 Первый пункт с достаточно длинным текстом для разбора.\n"
        "1.2 Второй пункт, тоже не короче двадцати символов.\n"
        "2. Плоский пункт с точкой после номера и длинным текстом.\n"
    )
    parsed = parser.parse(raw, title="Правила")
    numbers = [rule.number for rule in parsed.rules]
    assert "1.1" in numbers
    assert "1.2" in numbers
    assert "2" in numbers or "2." in numbers


def test_date_lines_are_not_rules(parser):
    """«30 Июн 2026» из блока автора — не пункт правил.

    Из-за этого плоская нумерация требует точку после номера.
    """
    raw = "Правила\n#1\n30 Июн 2026\n2 Авг 2026\n1.1 Настоящий пункт разбирается.\n"
    parsed = parser.parse(raw, title="Правила")
    texts = " ".join(rule.text for rule in parsed.rules)
    assert "Июн" not in texts
    assert "Авг" not in texts
