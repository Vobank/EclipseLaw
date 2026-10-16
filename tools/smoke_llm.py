"""
Проверка связки с Ollama: стриминг, системный промпт режима, скорость.

Отдельно проверяется главное требование к промпту — модель не должна
выдумывать номера статей, когда базы законов ещё нет.

Запуск: python tools/smoke_llm.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.llm.ollama_client import LLMUnavailable, Message, OllamaClient  # noqa: E402
from core.prompts import system_prompt  # noqa: E402
from core.session import Mode  # noqa: E402

QUESTIONS = [
    (Mode.NORMAL, "Какое наказание за превышение скорости в черте города?"),
    (Mode.NORMAL, "Обязан ли сотрудник представиться при задержании?"),
]

# Бюджет на один ответ. Оверлей висит поверх игры: пользователь не может
# ждать минуту, пока его задерживают. На CPU-профиле это потолок,
# на 5080 ответ должен укладываться в единицы секунд.
MAX_ANSWER_SECONDS = 45.0

# Признаки того, что модель сочинила реквизиты нормы. Пока база законов
# не подключена, ничего из этого в ответе быть не может в принципе.
FABRICATION_PATTERNS = [
    re.compile(r"\bстать[ея]\s*№?\s*\d", re.IGNORECASE),
    re.compile(r"\bст\.\s*\d", re.IGNORECASE),
    re.compile(r"\bглав[аые]\s*№?\s*\d", re.IGNORECASE),
    re.compile(r"\bп\.\s*\d+(\.\d+)*\s", re.IGNORECASE),
    re.compile(r"\d[\d\s]*(?:—|-|–|до)\s*\d[\d\s]*"
               r"\s*(?:credits?|кредит|долларов|\$)", re.IGNORECASE),
    re.compile(r"штраф\D{0,20}\d", re.IGNORECASE),
]


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    client = OllamaClient.from_config(cfg)
    ready, reason = client.check_ready()
    print(f"Ollama: {reason}")
    if not ready:
        return 1

    print(f"Модель: {client.options.model}  "
          f"num_gpu={client.options.num_gpu}  "
          f"num_ctx={client.options.num_ctx}\n")

    failures = 0
    for mode, question in QUESTIONS:
        print(f"[{mode.title}] {question}")
        messages = [
            Message("system", system_prompt(mode, rag_enabled=False)),
            Message("user", question),
        ]

        # Копим рассуждения отдельно: их наличие — главный источник
        # неприемлемой задержки, и тест обязан их видеть.
        reasoning: list[str] = []
        started = time.monotonic()
        first_token_at: float | None = None
        pieces: list[str] = []
        try:
            for chunk in client.stream_chat_ex(messages, reasoning.append):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                pieces.append(chunk)
        except LLMUnavailable as exc:
            print(f"  ПРОВАЛ: {exc}\n")
            failures += 1
            continue

        answer = "".join(pieces).strip()
        thought = "".join(reasoning)
        total = time.monotonic() - started
        ttft = (first_token_at - started) if first_token_at else total
        # Грубая оценка: примерно 3 символа на токен для русского текста.
        tok_per_s = (len(answer) / 3) / max(total - ttft, 0.01)

        print("  ответ:\n    " + answer.replace("\n", "\n    "))
        print(f"  первый токен: {ttft:.2f} с | всего: {total:.2f} с | "
              f"~{tok_per_s:.1f} ток/с")

        if not answer:
            print("  ПРОВАЛ: пустой ответ")
            failures += 1

        # Рассуждения ловим двумя способами сразу: в отдельном поле
        # thinking и протёкшими тегами в самом ответе.
        leaked = "<think>" in answer or "</think>" in answer
        if thought or leaked:
            where = []
            if thought:
                where.append(f"поле thinking: {len(thought)} симв.")
            if leaked:
                where.append("теги <think> протекли в content")
            print(f"  ПРОВАЛ: модель рассуждает ({'; '.join(where)}). "
                  "Нужна модель без режима мышления.")
            failures += 1

        if total > MAX_ANSWER_SECONDS:
            print(f"  ПРОВАЛ: {total:.0f} с на ответ — больше бюджета "
                  f"{MAX_ANSWER_SECONDS} с, для оверлея поверх игры "
                  "это неприемлемо.")
            failures += 1

        # Главная проверка этапа: базы законов нет, значит взяться
        # конкретному номеру статьи или сумме штрафа неоткуда. Если они
        # в ответе есть — модель их выдумала, и это опаснее пустого ответа:
        # пользователь сошлётся на несуществующую норму прямо в игре.
        invented = [pattern.pattern for pattern in FABRICATION_PATTERNS
                    if pattern.search(answer)]
        if invented:
            print(f"  ПРОВАЛ: выдуманы реквизиты нормы — "
                  f"сработало {len(invented)} шаблон(ов): "
                  f"{'; '.join(invented)}")
            failures += 1
        else:
            print("  выдуманных статей и сумм нет — OK")

        honest = any(marker in answer.lower() for marker in (
            "в базе нет", "нет точной", "не подключена", "проверить вручную",
            "нет доступа", "назвать не могу", "уточнить"))
        print(f"  признаёт отсутствие базы: "
              f"{'да' if honest else 'НЕТ — проверьте промпт'}")
        if not honest:
            failures += 1
        print()

    print("Итог:", "связка с LLM работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
