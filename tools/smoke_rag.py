"""
Проверка поиска по базе и ответа с опорой на найденные статьи (этап 4).

Главное, что здесь проверяется, — не «нашлось ли что-то», а honesty:
модель обязана ссылаться только на поданные ей выдержки и обязана
признаваться, когда ответа в них нет. Выдуманная статья опаснее отказа:
на неё сошлются прямо в игре.

Запуск: python tools/smoke_rag.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402
from core.legal.vectors import EmbeddingClient, VectorIndex  # noqa: E402
from core.llm.ollama_client import LLMUnavailable, Message, OllamaClient  # noqa: E402
from core.prompts import system_prompt  # noqa: E402
from core.session import Mode  # noqa: E402

# Вопросы разного типа: по смыслу, по номеру статьи и заведомо не из базы.
QUESTIONS = [
    ("смысл", "Какое наказание за превышение скорости в городе?"),
    ("смысл", "Обязан ли сотрудник представиться при задержании?"),
    ("номер", "Что говорит статья 1.3 Уголовного кодекса?"),
    ("прайс", "Сколько стоит смена внешности у хирурга?"),
    ("не из базы", "Какая столица у Португалии?"),
]

MAX_ANSWER_SECONDS = 60.0

# Ссылка на документ в ответе: «[Уголовный Кодекс..., п. 1.3]» или
# «Уголовный Кодекс, п. 1.3».
CITATION = re.compile(r"[А-ЯЁA-Z][^\n,\[\]]{5,70},\s*(?:п\.|ст\.)\s*[\dIVX]")


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    client = EmbeddingClient(
        cfg.get("llm.host", "http://127.0.0.1:11434"),
        cfg.get("retrieval.embedding_model", "bge-m3"),
        batch_size=cfg.get("retrieval.embedding_batch", 32))
    ready, reason = client.available()
    print(f"Эмбеддинги: {reason}")
    if not ready:
        return 1

    db = LegalDB(cfg.resolve_path("storage.legal_db"))
    index = VectorIndex(db, client)
    stats = index.stats()
    print(f"Индекс: {stats['векторов']} векторов на {stats['пунктов']} пунктов")
    if not stats["векторов"]:
        print("ПРОВАЛ: индекс пуст, выполните python tools/build_index.py")
        db.close()
        return 1

    retriever = Retriever.from_config(cfg, db, index)
    llm = OllamaClient.from_config(cfg)
    llm_ready, llm_reason = llm.check_ready()
    print(f"LLM: {llm_reason}\n")

    failures = 0
    for kind, question in QUESTIONS:
        print(f"[{kind}] {question}")
        started = time.monotonic()
        hits = retriever.retrieve(question, limit=cfg.get("retrieval.top_k", 6))
        found_in = time.monotonic() - started

        print(f"  найдено статей: {len(hits)} за {found_in:.2f} с")
        for hit in hits[:3]:
            print(f"     [{hit.rule.document[:34]} / {hit.rule.number}] "
                  f"({hit.why()}) {hit.rule.text[:52]}")

        if kind != "не из базы" and not hits:
            print("  ПРОВАЛ: поиск ничего не дал")
            failures += 1

        if not llm_ready:
            print()
            continue

        context = retriever.build_context(hits)
        messages = [
            Message("system", system_prompt(Mode.NORMAL, legal_context=context,
                                            rag_enabled=True)),
            Message("user", question),
        ]
        started = time.monotonic()
        try:
            answer = llm.chat(messages).strip()
        except LLMUnavailable as exc:
            print(f"  ПРОВАЛ: {exc}\n")
            failures += 1
            continue
        total = time.monotonic() - started

        print("  ответ:\n    " + answer.replace("\n", "\n    "))
        print(f"  время ответа: {total:.1f} с")

        if total > MAX_ANSWER_SECONDS:
            print(f"  ПРОВАЛ: дольше бюджета {MAX_ANSWER_SECONDS} с")
            failures += 1

        if kind == "не из базы":
            # Тут правильный ответ — признаться, что в базе этого нет.
            honest = any(m in answer.lower() for m in (
                "не нашлось", "нет статей", "нет ответа", "не относится",
                "в найденных", "в базе нет", "нет информации",
                "не содержит", "отсутствует"))
            print(f"  отказался отвечать не по базе: {'да' if honest else 'НЕТ'}")
            if not honest:
                failures += 1
        else:
            # Куда важнее цитаты — отсутствие выдуманных норм:
            # ответ без ссылки неудобен, ответ с сочинённой статьёй
            # опасен, на неё сошлются прямо в игре.
            invented = retriever.unsupported_citations(answer, hits)
            if invented:
                print(f"  ПРОВАЛ: выдуманы статьи {invented}")
                failures += 1
            else:
                print("  выдуманных статей нет — OK")
            cited = bool(CITATION.search(answer)) or "[" in answer
            print(f"  ссылается на документ: {'да' if cited else 'нет'}")
        print()

    db.close()
    print("Итог:", "поиск по базе работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
