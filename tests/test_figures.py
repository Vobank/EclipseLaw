"""
Проверка сумм и сроков в ответе.

Проверка номеров статей не ловит самое частое враньё: номер назван
верный, а штраф и срок подставлены по памяти. Живой случай, ради
которого файл и появился: на вопрос о скорости в городе бот ответил
«штраф до 1.000$ ((от 30 до 60 мин))» со ссылкой на статью 32
Дорожного кодекса. Сумма из статьи, а игровое время скопировано из
примера в самом промпте, где оно стоит образцом формата. Никаких минут
у этой статьи нет.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.legal.db import StoredRule  # noqa: E402
from core.legal.retriever import Hit, Retriever  # noqa: E402

SPEED = ("Скоростной лимит: В населенных пунктах – 100 км/ч "
         "В жилых зонах – 50 км/ч Вне населенных пунктов – 130 км/ч")
FINE = ("штраф до 1.000$ за превышение на 10-25 км/ч "
        "штраф до 2.000$ за превышение на 26-40 км/ч")


def hits():
    speed = StoredRule(id=1, document="Дорожный Кодекс", number="32",
                       section_number="", section_title="", text=SPEED,
                       sanction=FINE, notes=[])
    # Посторонняя статья с минутами: она есть в выдаче, но ответ на неё
    # не ссылается — её величины не должны ничего «разрешать».
    other = StoredRule(id=2, document="Общие правила банд", number="6.8",
                       section_number="", section_title="",
                       text="Ожидание составляет от 30 до 60 мин.",
                       sanction="", notes=[])
    return [Hit(speed, 0.03, ["смысл"]), Hit(other, 0.02, ["смысл"])]


def test_honest_answer_passes():
    answer = ("В населенных пунктах — 100 км/ч. [Дорожный Кодекс, п. 32] "
              "штраф до 1.000$ за превышение на 10-25 км/ч")
    assert Retriever.unsupported_figures(answer, hits()) == []


def test_invented_game_time_is_caught():
    """Тот самый случай: минуты приписаны статье, где их нет."""
    answer = ("В жилых зонах — 50 км/ч. [Дорожный Кодекс, п. 32] "
              "штраф до 1.000$ ((от 30 до 60 мин))")
    assert "60мин" in Retriever.unsupported_figures(answer, hits())


def test_figures_of_other_articles_do_not_authorise():
    """«30 до 60 мин» есть в выдаче — но в другой статье.

    Без привязки к процитированной статье проверка молчала: величина
    формально «нашлась», хотя приписана совсем другой норме.
    """
    answer = "[Дорожный Кодекс, п. 32] ожидание от 30 до 60 мин"
    assert Retriever.unsupported_figures(answer, hits()) != []


def test_invented_amount_is_caught():
    answer = "[Дорожный Кодекс, п. 32] штраф до 7.500$"
    assert "7500$" in Retriever.unsupported_figures(answer, hits())


def test_thousands_separator_does_not_matter():
    """«1.000$», «1 000$» и «1000$» — одна и та же сумма."""
    for written in ("1.000$", "1 000$", "1000$"):
        answer = f"[Дорожный Кодекс, п. 32] штраф до {written}"
        assert Retriever.unsupported_figures(answer, hits()) == [], written


@pytest.mark.parametrize("speed", ["100 км/ч", "50 км/ч", "130 км/ч"])
def test_speeds_from_the_article_pass(speed):
    answer = f"[Дорожный Кодекс, п. 32] ограничение {speed}"
    assert Retriever.unsupported_figures(answer, hits()) == []


def test_invented_speed_is_caught():
    answer = "[Дорожный Кодекс, п. 32] ограничение 90 км/ч"
    assert "90км/ч" in Retriever.unsupported_figures(answer, hits())


def test_answer_without_citation_checks_against_everything():
    """Ссылки нет — сверяем со всей выдачей, иначе ругались бы на всё."""
    assert Retriever.unsupported_figures("ограничение 100 км/ч", hits()) == []
