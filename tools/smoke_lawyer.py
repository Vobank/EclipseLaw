"""
Проверка режима адвоката: чек-лист, вопросы, заключение (этап 7).

Прогоняет сценарий задержания целиком, не требуя микрофона: строит
чек-лист из базы, задаёт вопрос, подставляет ответ «сотрудника»
и просит итоговое заключение.

Главное, что проверяется, — заключение не должно ссылаться на статьи,
которых нет в базе. Оно идёт черновиком в судопроизводство, и
выдуманная норма там дороже, чем в обычном ответе.

Запуск: python tools/smoke_lawyer.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.checklist import build_checklist  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402
from core.legal.vectors import EmbeddingClient, VectorIndex  # noqa: E402
from core.llm.ollama_client import LLMUnavailable, Message, OllamaClient  # noqa: E402
from core.prompts import system_prompt, verdict_prompt  # noqa: E402
from core.session import Mode  # noqa: E402

# Реплики «сотрудника» — намеренно с нарушением: прав не зачитывали,
# основание не назвали. Адвокат обязан это заметить.
DIALOGUE = [
    ("я", "На каком основании меня задерживают?"),
    ("сотрудник", "Просто стой и молчи, разберёмся в участке."),
    ("я", "Мне зачитали мои права?"),
    ("сотрудник", "Нет, потом зачитаю, некогда."),
]


class _Hit:
    def __init__(self, rule):
        self.rule = rule
        self.sources = ["чек-лист"]

    def why(self) -> str:
        return "чек-лист"


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    db = LegalDB(cfg.resolve_path("storage.legal_db"))
    client = EmbeddingClient(
        cfg.get("llm.host", "http://127.0.0.1:11434"),
        cfg.get("retrieval.embedding_model", "bge-m3"),
        batch_size=cfg.get("retrieval.embedding_batch", 32))
    ready, reason = client.available()
    if not ready:
        print(f"ПРОВАЛ: {reason}")
        db.close()
        return 1

    retriever = Retriever.from_config(cfg, db, VectorIndex(db, client))
    llm = OllamaClient.from_config(cfg)
    llm_ready, llm_reason = llm.check_ready()
    print(f"LLM: {llm_reason}\n")

    failures = 0

    # --- 1. чек-лист ---------------------------------------------------
    started = time.monotonic()
    checklist = build_checklist(
        # Чек-лист теперь свой у каждой роли. Старый общий ключ
        # исчез, и проверка молча сваливалась на встроенный
        # список из девяти тем вместо настоящего.
        retriever, topics=cfg.get("lawyer.roles")["detained"]["checklist"],
        per_topic=cfg.get("lawyer.articles_per_topic", 2))
    print(f"1) Чек-лист собран за {time.monotonic() - started:.1f} с")
    print(f"   пунктов всего: {len(checklist.items)}, "
          f"с опорой в базе: {len(checklist.grounded)}")
    for item in checklist.items:
        basis = item.articles[0].citation() if item.articles else "НЕТ СТАТЬИ"
        print(f"     {item.title:42} {basis}")
    if len(checklist.grounded) < 5:
        print("   ПРОВАЛ: меньше пяти пунктов имеют опору в базе")
        failures += 1

    if not llm_ready:
        db.close()
        return 1 if failures else 0

    # --- 2. вопрос в режиме адвоката -----------------------------------
    print("\n2) Вопрос адвоката по первому пункту:")
    question = DIALOGUE[0][1]
    hits = retriever.retrieve(question, limit=cfg.get("retrieval.top_k", 6))
    context = retriever.build_context(hits)
    started = time.monotonic()
    answer = llm.chat([
        Message("system", system_prompt(
            Mode.LAWYER, legal_context=context, rag_enabled=True,
            checklist=checklist.as_prompt())),
        Message("user", question),
    ]).strip()
    print(f"   ({time.monotonic() - started:.0f} с)")
    print("   " + answer.replace("\n", "\n   "))
    # Статьи чек-листа тоже в промпте, ссылка на них законна.
    allowed = list(hits) + [_Hit(a) for i in checklist.grounded for a in i.articles]
    invented = retriever.unsupported_citations(answer, allowed)
    if invented:
        print(f"   ПРОВАЛ: выдуманы статьи {invented}")
        failures += 1

    # --- 3. заключение по стенограмме ----------------------------------
    print("\n3) Итоговое заключение:")
    all_hits = list(hits)
    seen = {h.rule.id for h in all_hits}
    for item in checklist.grounded:
        for article in item.articles:
            if article.id not in seen:
                seen.add(article.id)
                all_hits.append(_Hit(article))

    transcript = "\n".join(
        f"{'Я' if who == 'я' else 'Сотрудник'}: {line}"
        for who, line in DIALOGUE)
    started = time.monotonic()
    try:
        verdict = llm.chat([
            Message("system", verdict_prompt(retriever.build_context(all_hits))),
            Message("user", "Стенограмма задержания:\n\n" + transcript),
        ]).strip()
    except LLMUnavailable as exc:
        print(f"   ПРОВАЛ: {exc}")
        db.close()
        return 1
    print(f"   ({time.monotonic() - started:.0f} с, {len(verdict)} символов)")
    print("   " + verdict.replace("\n", "\n   ")[:1400])

    invented = retriever.unsupported_citations(verdict, all_hits)
    if invented:
        print(f"\n   ПРОВАЛ: в заключении выдуманы статьи {invented}")
        failures += 1
    else:
        print("\n   выдуманных статей в заключении нет — OK")

    # Заключение обязано быть структурным: без разделов оно бесполезно
    # как черновик для суда.
    sections = sum(1 for marker in ("НАРУШ", "РЕКОМЕНД", "ОБСТОЯТЕЛЬ",
                                    "КОМПЕНСАЦ", "СТАТЬЯ")
                   if marker in verdict.upper())
    print(f"   разделов заключения распознано: {sections} из 5")
    if sections < 2:
        print("   ПРОВАЛ: заключение не структурировано")
        failures += 1

    db.close()
    print("\nИтог:", "режим адвоката работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
