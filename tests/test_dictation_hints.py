"""
Подсказки в режиме оформления документа.

Раньше список шёл дважды: сверху названия пунктов, снизу три подсказки
и строка «...и ещё 6 пунктов». Толку от неё не было — какие именно,
человек не видел, а по названию «Идентификационный знак» не
догадаешься, что от тебя хотят. Теперь подсказка стоит у своего пункта.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.documents import advise  # noqa: E402


@pytest.mark.parametrize("kind", ["proceedings", "court"])
def test_every_missing_item_says_what_to_say(kind):
    text = advise("", kind).as_text()
    for item in advise("", kind).missing:
        line = next(l for l in text.split("\n") if item.title in l)
        assert item.hint in line, item.title


@pytest.mark.parametrize("kind", ["proceedings", "court"])
def test_nothing_is_hidden_behind_a_counter(kind):
    """«...и ещё 6 пунктов» не говорит человеку ничего."""
    text = advise("", kind).as_text()
    assert "и ещё" not in text
    assert "ЧТО ЕЩЁ СКАЗАТЬ" not in text


@pytest.mark.parametrize("kind", ["proceedings", "court"])
def test_the_list_appears_once(kind):
    """Пункт не должен встречаться в двух местах экрана."""
    advice = advise("", kind)
    text = advice.as_text()
    for item in advice.missing:
        assert text.count(item.title) == 1, item.title


@pytest.mark.parametrize("kind", ["proceedings", "court"])
def test_separator_never_doubles(kind):
    """В подсказках уже есть и тире, и двоеточия — строка не должна
    выглядеть как «Заявитель — кто подал — имя»."""
    for line in advise("", kind).as_text().split("\n"):
        if not line.strip().startswith("[ ]"):
            continue
        assert " — " not in line or line.count(" — ") == 1, line
        assert ": " not in line or line.count(": ") == 1, line


def test_covered_items_stay_short():
    """Что уже названо — просто галочка, подсказка там не нужна."""
    advice = advise("меня задержали сотрудники лспд", "proceedings")
    assert advice.covered
    for item in advice.covered:
        line = next(l for l in advice.as_text().split("\n")
                    if item.title in l)
        assert line.strip() == f"[x] {item.title}"


def test_counter_still_shows_progress():
    text = advise("меня задержали сотрудники лспд", "proceedings").as_text(5)
    assert "НАДИКТОВАНО:" in text and "слов 5" in text


def test_full_dictation_says_it_is_ready():
    advice = advise("", "proceedings")
    advice.missing = []
    assert "можно составлять" in advice.as_text().lower()


# ------------------------------------------------ пустые поля в документе
from core.documents import UNKNOWN, unfilled_fields  # noqa: E402


def test_field_name_before_the_mark_is_used():
    doc = f"Дата окончания расследования: {UNKNOWN}"
    assert unfilled_fields(doc) == ["Дата окончания расследования"]


def test_mark_inside_a_sentence_gets_context():
    """Название поля есть не всегда — тогда нужен кусок текста рядом."""
    doc = f"{UNKNOWN} прокурору пришло сообщение от сотрудника FIB."
    found = unfilled_fields(doc)
    assert found and "прокурору пришло" in found[0]


def test_repeated_fields_are_listed_once():
    doc = f"Дата: {UNKNOWN}\nПодпись: подпись\nДата: {UNKNOWN}"
    assert unfilled_fields(doc) == ["Дата"]


def test_document_without_gaps_lists_nothing():
    assert unfilled_fields("ПОСТАНОВЛЕНИЕ\nВсё заполнено.") == []


def test_long_paragraph_gives_the_last_sentence():
    """Перед полем бывает целый абзац — в подпись он не влезет."""
    doc = ("Прокуратура рассмотрела обращение гражданина и установила "
           f"следующее. Дата начала расследования: {UNKNOWN}")
    assert unfilled_fields(doc) == ["Дата начала расследования"]


def test_list_is_capped():
    doc = "\n".join(f"Поле {i}: {UNKNOWN}" for i in range(20))
    assert len(unfilled_fields(doc, limit=8)) == 8


def test_orchestrator_explains_why_the_field_is_empty():
    """«Полей без данных: 6» не объясняет ни что это, ни что делать."""
    source = (Path(__file__).resolve().parent.parent
              / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "Вы этого не говорили" in source
    assert "unfilled_fields(document)" in source


def test_empty_context_falls_back_to_the_text_before():
    """После метки бывает одна точка — «перед «....»» не помогает никому."""
    doc = f"был уведомлен о вынесении постановления в {UNKNOWN}."
    found = unfilled_fields(doc)
    assert found and "...." not in found[0]
    assert "постановления" in found[0]


def test_junk_labels_are_never_produced():
    """Подпись обязана содержать слова, а не одни знаки препинания.

    Короткое название поля («Дата», «Ещё») — это нормально, а вот
    «перед «....»» не значит ничего.
    """
    doc = f"Текст. {UNKNOWN} ." + chr(10) + f"Ещё: {UNKNOWN}"
    for name in unfilled_fields(doc):
        letters = [c for c in name if c.isalpha()]
        assert len(letters) >= 3, name
