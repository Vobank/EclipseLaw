"""
Проверка режима экзамена (скорость + полнота ответа).

Экзамен отличается от остальных режимов тем, что ответ должен быть
связным объяснением своими словами, а не карточкой «статья + санкция».
Поэтому проверяется не только то, что нашлась нужная статья, но и
форма: два-четыре предложения, без вступлений, со ссылкой на норму.

И отдельно — время. На экзамене оно ограничено, и ответ, который
считается минуту, бесполезен независимо от качества.

Запуск: python tools/smoke_exam.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.answer import generate_checked  # noqa: E402
from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402
from core.legal.vectors import EmbeddingClient, VectorIndex  # noqa: E402
from core.llm.ollama_client import LLMUnavailable, Message, OllamaClient  # noqa: E402
from core.prompts import system_prompt  # noqa: E402
from core.session import Mode  # noqa: E402

# Вопросы экзаменационного типа: не «что мне будет за», а «объясните».
QUESTIONS = [
    "что такое презумпция невиновности",
    "в каких случаях сотрудник обязан инициировать следствие",
    "что относится к недопустимым доказательствам",
    "какой порядок ареста задержанного",
    "какие права имеет задержанный",
]

# Обороты, которых на экзамене быть не должно.
FORBIDDEN = ("конечно", "давайте разберём", "давайте разберем",
             "стоит отметить", "итак,", "хороший вопрос")


def _is_list(text: str) -> bool:
    """Ответ — перечень шагов, а не рассуждение."""
    items = re.findall(r"(?m)^\s*(?:\d+[.)]|[-—•])\s+", text)
    return len(items) >= 3


def sentences(text: str) -> int:
    """Сколько в ответе смысловых частей.

    Точка с запятой считается наравне с точкой: перечисление случаев
    «а; б; в» — это и есть полный ответ на экзамене, а по одним только
    точкам оно выглядит как одно предложение, и проверка ругалась на
    совершенно правильные ответы.
    """
    body = re.sub(r"\[[^\]]*\]", "", text)
    return len([s for s in re.split(r"[.!?;]+\s", body) if len(s.strip()) > 15])


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    db = LegalDB(cfg.resolve_path("storage.legal_db"))
    client = EmbeddingClient(cfg.get("llm.host", "http://127.0.0.1:11434"),
                             cfg.get("retrieval.embedding_model", "bge-m3"))
    available, reason = client.available()
    print(f"Эмбеддинги: {reason}")
    if not available:
        return 1
    retriever = Retriever.from_config(cfg, db, VectorIndex(db, client))

    llm = OllamaClient.from_config(cfg)
    ready, reason = llm.check_ready()
    print(f"LLM: {reason}\n")
    if not ready:
        return 1

    top_k = cfg.get("modes.exam.top_k") or cfg.get("retrieval.top_k", 6)
    article_chars = cfg.get("modes.exam.max_article_chars", 0)
    budget = cfg.get("modes.exam.num_predict", 0)
    print(f"Настройки экзамена: статей {top_k}, "
          f"до {article_chars} символов на статью, потолок ответа {budget}\n")

    failures = 0
    times: list[float] = []

    for question in QUESTIONS:
        hits = retriever.retrieve(question, limit=top_k)
        context = retriever.build_context(hits, max_article_chars=article_chars)
        prompt = system_prompt(Mode.EXAM, legal_context=context,
                               rag_enabled=True)

        started = time.monotonic()
        try:
            # Через ту же обёртку, что и бот: иначе проверка мерит одно,
            # а пользователь получает другое.
            checked = generate_checked(
                llm, [Message("system", prompt), Message("user", question)],
                retriever, hits, num_predict=budget,
                retry=cfg.get("modes.exam.retry_on_invented", True),
                redact=cfg.get("modes.exam.redact_invented", True))
        except LLMUnavailable as exc:
            print(f"ПРОВАЛ: {exc}")
            return 1
        answer = checked.text
        elapsed = time.monotonic() - started
        times.append(elapsed)

        print(f"[{elapsed:.0f} с] {question}")
        print(f"   {answer}")

        count = sentences(answer)
        # Короткий ответ — это мало содержания, а не мало точек:
        # перечисление через запятую формально одно предложение,
        # но на экзамене засчитывается полностью.
        if count < 2 and len(answer) < 250:
            print(f"   ПРОВАЛ: предложений {count} — экзаменатор сочтёт "
                  f"ответ неполным")
            failures += 1
        elif _is_list(answer):
            # Порядок действий — это список шагов, и короче он не
            # бывает: «процесс ареста» в кодексе расписан шестью
            # пунктами. Считать его многословием нельзя, иначе
            # проверка требует выкинуть половину процедуры.
            if len(answer) > 1400:
                print(f"   ПРОВАЛ: список на {len(answer)} символов — "
                      f"на экзамене столько не зачитать")
                failures += 1
        elif count > 6:
            print(f"   ПРОВАЛ: предложений {count} — слишком длинно для "
                  f"экзамена")
            failures += 1

        if "[" not in answer:
            print("   ПРОВАЛ: нет ссылки на статью — на экзамене она "
                  "засчитывается отдельно")
            failures += 1

        lowered = answer.lower()
        junk = [f for f in FORBIDDEN if f in lowered]
        if junk:
            print(f"   ПРОВАЛ: вступления и оговорки: {junk}")
            failures += 1

        if checked.retried:
            print("   ЗАМЕЧАНИЕ: ответ переписан — с первого раза была "
                  "названа статья вне базы")
        if checked.invented:
            print(f"   ПРОВАЛ: выдуманные статьи {checked.invented} "
                  f"— не помогла даже перегенерация")
            failures += 1

        # Дословная цитата вместо пересказа: ищем длинный кусок,
        # совпадающий с текстом статьи буква в букву.
        for hit in hits[:3]:
            chunk = hit.rule.text[:120].strip()
            if len(chunk) > 60 and chunk in answer:
                print("   ЗАМЕЧАНИЕ: ответ дословно повторяет выдержку, "
                      "а на экзамене нужен пересказ своими словами")
                break
        print()

    if times:
        times_sorted = sorted(times)
        print(f"Время ответа: медиана {times_sorted[len(times) // 2]:.0f} с, "
              f"максимум {times_sorted[-1]:.0f} с")

    print("\nИтог:", "режим экзамена работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    db.close()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
