"""
Проверка ссылок и перегенерация ответа.

Самый опасный сбой модели — не отказ, а складная выдумка: на вопрос о
правах задержанного она выдала перечень из настоящего права (молчание,
звонок, адвокат) и подписала его пунктом, которого в базе нет. Здесь
стережётся именно это поведение, без обращения к Ollama.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.answer import generate_checked, redact_invented  # noqa: E402
from core.legal.db import StoredRule  # noqa: E402
from core.legal.retriever import Hit, Retriever  # noqa: E402


class FakeLLM:
    """Отдаёт заранее заданные ответы по очереди."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[list] = []
        self.models: list[str] = []

    def stream_chat(self, messages, num_predict=0, model=""):
        # Сигнатура повторяет настоящую: подделка, расходящаяся с
        # оригиналом, ловит не те ошибки, что нужно.
        self.calls.append(list(messages))
        self.models.append(model)
        text = self.answers.pop(0) if self.answers else ""
        for word in text.split(" "):
            yield word + " "


def hits():
    rule = StoredRule(id=1, document="Процессуальный Кодекс", number="6.2",
                      section_number="", section_title="",
                      text="Недопустимые доказательства.", sanction="",
                      notes=[])
    return [Hit(rule, 0.03, ["смысл"])]


def test_clean_answer_passes_through():
    llm = FakeLLM("Смотрите [Процессуальный Кодекс, п. 6.2].")
    result = generate_checked(llm, [], Retriever, hits(), retry=True)
    assert result.invented == []
    assert result.retried is False
    assert len(llm.calls) == 1


def test_invented_article_triggers_a_rewrite():
    llm = FakeLLM("Права перечислены в статье 5.1.",
                  "Прямого перечня в базе нет. [Процессуальный Кодекс, п. 6.2]")
    result = generate_checked(llm, [], Retriever, hits(), retry=True)
    assert result.retried is True
    assert result.invented == []
    assert "перечня в базе нет" in result.text
    # Во второй запрос уходит и промах модели, и конкретный номер.
    second = llm.calls[1]
    assert any("5.1" in m.content for m in second)


def test_without_retry_the_answer_is_only_flagged():
    """В обычном режиме переспрашивать дорого — там хватает предупреждения."""
    llm = FakeLLM("Права перечислены в статье 5.1.")
    result = generate_checked(llm, [], Retriever, hits(), retry=False)
    assert result.retried is False
    assert result.invented == ["5.1"]
    assert len(llm.calls) == 1


def test_second_attempt_can_also_fail():
    """Перегенерация не гарантия — промах остаётся видимым."""
    llm = FakeLLM("Статья 5.1.", "Всё равно статья 5.1.")
    result = generate_checked(llm, [], Retriever, hits(), retry=True)
    assert result.retried is True
    assert result.invented == ["5.1"]


def test_chunks_reach_the_overlay():
    seen: list[str] = []
    llm = FakeLLM("два слова")
    generate_checked(llm, [], Retriever, hits(), on_chunk=seen.append)
    assert "".join(seen).strip() == "два слова"


def test_empty_rewrite_keeps_the_first_answer():
    """Пустой повтор не должен оставить пользователя без ответа вовсе."""
    llm = FakeLLM("Статья 5.1 говорит о правах.", "")
    result = generate_checked(llm, [], Retriever, hits(), retry=True)
    assert "5.1" in result.text
    assert result.retried is True


# ------------------------------------------------- вычёркивание ссылок
def test_redaction_removes_the_bracketed_reference():
    text = ("Задержанный имеет право хранить молчание. "
            "[Процессуальный Кодекс, п. 5.1]")
    result = redact_invented(text, ["5.1"])
    assert "5.1" not in result
    assert "НЕ ПОДТВЕРЖДЕНА" in result
    # Сам пересказ остаётся: вычеркнуть его нечем, и притворяться,
    # что ответа не было, хуже, чем показать его без номера.
    assert "хранить молчание" in result


def test_redaction_removes_inline_mentions():
    result = redact_invented("Это описано в статье 5.1 кодекса.", ["5.1"])
    assert "5.1" not in result
    assert "не подтверждена" in result


def test_redaction_leaves_confirmed_references_alone():
    text = "[Процессуальный Кодекс, п. 6.2] и [Уголовный Кодекс, п. 5.1]"
    result = redact_invented(text, ["5.1"])
    assert "п. 6.2" in result
    assert "п. 5.1" not in result


def test_redaction_does_not_touch_sums_and_terms():
    """«5.000$» и «48 часов» — не ссылки на статьи."""
    text = "Штраф до 5.000$ в течение 48 часов."
    assert redact_invented(text, ["5.0", "48"]) == text


def test_redaction_runs_only_when_asked():
    llm = FakeLLM("Статья 5.1.", "Всё равно статья 5.1.")
    plain = generate_checked(llm, [], Retriever, hits(), retry=True)
    assert "5.1" in plain.text

    llm = FakeLLM("Статья 5.1.", "Всё равно статья 5.1.")
    redacted = generate_checked(llm, [], Retriever, hits(),
                                retry=True, redact=True)
    assert "5.1" not in redacted.text
    # Промах всё равно остаётся видимым в отчёте.
    assert redacted.invented == ["5.1"]


def test_model_reaches_the_client():
    """Модель под задачу обязана доезжать до запроса.

    Тяжёлые бумаги считает 14B, быстрые вопросы 7B. Если параметр
    потеряется по дороге, всё молча посчитает модель из профиля.
    """
    llm = FakeLLM("Ответ.")
    generate_checked(llm, [], Retriever, hits(), model="qwen2.5:14b-instruct")
    assert llm.models == ["qwen2.5:14b-instruct"]
