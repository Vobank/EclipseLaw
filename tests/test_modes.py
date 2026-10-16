"""Режимы: экзамен, бюджеты ответа и разделение промптов."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.llm.ollama_client import LLMOptions, OllamaClient  # noqa: E402
from core.prompts import BY_MODE, system_prompt  # noqa: E402
from core.session import MODE_CYCLE, Mode  # noqa: E402


def test_exam_is_in_the_cycle():
    """Экзамен переключается тем же хоткеем, что и остальные режимы."""
    assert Mode.EXAM in MODE_CYCLE
    assert [m.title for m in MODE_CYCLE].count("Экзамен") == 1


def test_every_mode_has_its_own_prompt():
    """Общий промпт на два режима означает, что один из них работает хуже."""
    prompts = {mode: BY_MODE[mode] for mode in Mode}
    assert len(set(prompts.values())) == len(Mode)


def test_exam_prompt_demands_own_words_and_reference():
    exam = BY_MODE[Mode.EXAM]
    assert "своими словами" in exam.lower()
    assert "квадратных скобках" in exam
    # Именно этот запрет появился после того, как модель составила
    # перечень прав задержанного из настоящего права, а не из базы.
    assert "не достраивается" in exam or "НИКОГДА не достраивается" in exam


def test_exam_prompt_is_not_the_three_line_card():
    """Экзамену нужен связный ответ, а не карточка обычного режима."""
    assert "Максимум ТРИ строки" in BY_MODE[Mode.NORMAL]
    assert "Максимум ТРИ строки" not in BY_MODE[Mode.EXAM]


@pytest.mark.parametrize("mode", list(Mode))
def test_context_reaches_every_mode(mode):
    """Выдержки из базы подставляются во все режимы одинаково."""
    prompt = system_prompt(mode, legal_context="[Уголовный Кодекс, п. 1.1] Текст.",
                           rag_enabled=True)
    assert "Уголовный Кодекс, п. 1.1" in prompt


# --------------------------------------------------------------- бюджеты
@pytest.mark.parametrize("mode", ["normal", "lawyer", "exam"])
def test_mode_has_answer_budget(mode):
    """Без потолка длины ответ на CPU растягивается на минуту."""
    limit = Config.load().get(f"modes.{mode}.num_predict")
    assert isinstance(limit, int) and limit > 0


def test_exam_budget_fits_a_spoken_answer():
    """Экзамену нужен связный ответ, обычному режиму — три строки.

    Раньше экзамену давалось 600 токенов «про запас», и модель охотно
    их выбирала: замер на живом экзамене показал ответ на 216 токенов
    при скорости 18 ток/с — двенадцать секунд там, где ответ нужен
    сразу. Двух-трёх предложений с перечнем хватает и на 320.
    """
    cfg = Config.load()
    budgets = {m: cfg.get(f"modes.{m}.num_predict")
               for m in ("normal", "lawyer", "exam")}
    assert budgets["exam"] >= budgets["lawyer"] > budgets["normal"]
    # Перечень из семи пунктов плюс ссылка должен помещаться целиком.
    assert budgets["exam"] >= 300


def test_num_predict_reaches_the_request():
    """Потолок обязан доезжать до Ollama, иначе настройка бутафорская."""
    client = OllamaClient("http://127.0.0.1:11434", LLMOptions(model="test"))
    payload = client._payload([], stream=False, num_predict=200)
    assert payload["options"]["num_predict"] == 200


def test_no_limit_means_no_field():
    """Ноль — это «без ограничения», а не «ноль токенов»."""
    client = OllamaClient("http://127.0.0.1:11434", LLMOptions(model="test"))
    payload = client._payload([], stream=False, num_predict=0)
    assert "num_predict" not in payload["options"]


def test_keep_alive_is_sent():
    """Без этого Ollama выгружает модель через пять минут простоя,
    и следующий вопрос платит сорок секунд за загрузку."""
    client = OllamaClient("http://127.0.0.1:11434", LLMOptions(model="test"))
    assert client._payload([], stream=False)["keep_alive"]
