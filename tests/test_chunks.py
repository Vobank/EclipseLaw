"""
Нарезка длинных статей на фрагменты.

Статья 3.4 Процессуального кодекса — 5251 знак: порядок задержания
госслужащего плюс семь примечаний про вызов прокурора, начальства и
сроки. Одним вектором она не искалась совсем, а в контекст уходила
обрезанной по первым полутора тысячам знаков — ровно до того места,
где начинались ответы на вопросы, которые задают чаще всего.

Режем по смысловым границам самого текста. Проверяется именно это:
что кусок не рвётся посреди условия и что ничего не теряется.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.legal.chunks import SPLIT_ABOVE, TARGET, split_rule  # noqa: E402


def test_short_article_is_not_split():
    """Короткая норма и так находится целиком — резать нечего."""
    text = "Задержание - это мера кратковременного лишения свободы."
    assert split_rule(text) == [text]


def test_empty_text_gives_nothing():
    assert split_rule("") == []


def test_long_article_is_split():
    text = ("Примечание 1: " + "слово " * 120 + ". "
            "Примечание 2: " + "иное " * 120 + ".")
    parts = split_rule(text)
    assert len(parts) > 1


def test_nothing_is_lost():
    """Ни одно слово не имеет права пропасть: это текст закона."""
    text = ("Обязанности сотрудника: а) " + "первое " * 60 +
            " б) " + "второе " * 60 + " в) " + "третье " * 60)
    joined = " ".join(split_rule(text))
    assert len(joined.split()) == len(" ".join(text.split()).split())


def test_notes_start_their_own_fragment():
    """Примечание — отдельная мысль, и искать его надо отдельно."""
    text = ("Процесс задержания: а) " + "надеть наручники " * 30 +
            " Примечание 5: оповещение о вызове прокурора должно "
            "повторяться в рацию департамента примерно каждые 15 минут. " +
            "Примечание 6: " + "если нет доступа " * 30)
    parts = split_rule(text)
    with_note = [p for p in parts if "каждые 15 минут" in p]
    assert len(with_note) == 1
    assert len(with_note[0]) < len(text) / 2, "примечание не выделилось"


def test_fragments_are_not_tiny():
    """Обрывок «Примечание 7:» без текста в индексе бесполезен."""
    text = "Пункт: " + " ".join(f"{c}) коротко." for c in "абвгдежзик") * 30
    parts = split_rule(text)
    assert all(len(p) > 100 for p in parts[:-1])


def test_fragments_stay_near_the_target_size():
    text = "а) " + "слово " * 400
    parts = split_rule(text)
    assert all(len(p) <= TARGET * 2 for p in parts)


def test_solid_text_without_markers_is_still_split():
    """Бывают статьи сплошным текстом — делим по предложениям."""
    text = ("Сотрудник обязан действовать профессионально. " * 40)
    parts = split_rule(text)
    assert len(parts) > 1


def test_threshold_is_respected():
    text = "а) " + "x" * (SPLIT_ABOVE - 10)
    assert len(split_rule(text)) == 1
