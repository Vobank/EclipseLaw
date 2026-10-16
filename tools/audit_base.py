"""
Аудит базы: проверяет каждый документ, не задавая вопросов пользователю.

Смысл в том, чтобы не перебирать сотню документов руками. Скрипт сам
берёт из каждого документа несколько реальных статей, строит по их
тексту вопрос и проверяет, находит ли поиск именно их. Если статья
лежит в базе, но не находится по собственному тексту — с документом
что-то не так, и он попадёт в отчёт.

Дополнительно ищутся структурные дефекты, которые уже встречались:
обрывки-однострочники, потерянные санкции, мусор форума, дыры в
нумерации.

Проверка идёт без LLM — только поиск. Так на сотню документов уходят
минуты, а не часы.

Запуск:
    python tools/audit_base.py                 все документы
    python tools/audit_base.py --probes 5      больше проб на документ
    python tools/audit_base.py --only Дорожный только совпадающие по имени
"""
from __future__ import annotations

import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.retriever import Retriever, _duplicate_key  # noqa: E402
from core.legal.vectors import EmbeddingClient, VectorIndex  # noqa: E402

# Мусор форума, которого в тексте статьи быть не должно.
JUNK = (
    "Последнее редактирование", "English (US)", "Напишите свой ответ",
    "Добавить закладку", "Реакции:", "Обратная связь", "gta5_divider",
    "Спойлер:", "Нажмите, чтобы раскрыть",
)


def argument(flag: str, default=None):
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else default


def make_question(text: str) -> str:
    """Строит вопрос из текста статьи.

    Берём середину формулировки, а не начало: начало часто совпадает
    у соседних статей («Запрещено...»), и проверка выродилась бы
    в лотерею между ними.
    """
    words = re.sub(r"\s+", " ", text).split()
    if len(words) <= 12:
        return " ".join(words)
    start = min(3, max(0, len(words) - 12))
    return " ".join(words[start:start + 12])


def main() -> int:
    probes = int(argument("--probes", 3))
    only = argument("--only", "")

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

    index = VectorIndex(db, client)
    stats = index.stats()
    if stats["векторов"] < stats["пунктов"]:
        print(f"ВНИМАНИЕ: без вектора {stats['пунктов'] - stats['векторов']} "
              f"пунктов — сначала python tools/build_index.py\n")
    retriever = Retriever.from_config(cfg, db, index)

    by_document: dict[str, list] = defaultdict(list)
    for rule in db.all_rules():
        by_document[rule.document].append(rule)

    titles = sorted(by_document)
    if only:
        titles = [t for t in titles if only.lower() in t.lower()]
        if not titles:
            print(f"Под «{only}» документов не нашлось.")
            db.close()
            return 1

    random.seed(20260918)      # один и тот же набор проб между запусками
    started = time.monotonic()
    problems: list[tuple[str, str]] = []
    print(f"Проверяю документов: {len(titles)}, проб на документ: {probes}\n")

    for title in titles:
        rules = by_document[title]
        notes: list[str] = []

        # --- структурные признаки ------------------------------------
        if len(rules) <= 2:
            notes.append(f"всего {len(rules)} п. — похоже на обрывок страницы")

        junk = [r.number for r in rules
                if any(j in r.text for j in JUNK)][:4]
        if junk:
            notes.append(f"мусор форума в пунктах {', '.join(junk)}")

        leftover = [r.number for r in rules if "|" in r.text][:4]
        if leftover:
            notes.append(f"нераспознанная санкция в {', '.join(leftover)}")

        tiny = sum(1 for r in rules if len(r.text) < 25)
        if tiny > len(rules) * 0.3 and len(rules) > 5:
            notes.append(f"{tiny} из {len(rules)} пунктов короче 25 символов "
                         "— возможно, текст разорван")

        # Дыры в сквозной нумерации: пропущенная статья почти всегда
        # означает потерянный кусок документа.
        plain = sorted(int(r.number) for r in rules if r.number.isdigit())
        if len(plain) > 4:
            missing = [n for n in range(plain[0], plain[-1] + 1)
                       if n not in plain]
            if missing:
                notes.append(f"пропущены номера: "
                             f"{', '.join(map(str, missing[:8]))}"
                             f"{'...' if len(missing) > 8 else ''}")

        # --- находится ли документ поиском ---------------------------
        sample = random.sample(rules, min(probes, len(rules)))
        misses = 0
        for rule in sample:
            question = make_question(rule.text)
            if len(question) < 15:
                continue
            hits = retriever.retrieve(question, limit=5)
            # Совпадение по тексту, а не только по id: одинаковая норма
            # лежит в уставах LSPD и шерифов под разными номерами, и
            # поиск теперь схлопывает такие копии в одну. Найденная
            # копия означает, что норма находится — придираться к тому,
            # чья именно строка выжила, здесь не к чему.
            key = _duplicate_key(rule)
            if not any(h.rule.id == rule.id or _duplicate_key(h.rule) == key
                       for h in hits):
                misses += 1
                notes.append(f"п. {rule.number} не находится по своему же "
                             f"тексту: «{question[:52]}»")

        mark = "OK  " if not notes else "!!  "
        print(f"{mark} {len(rules):4} п.  {title[:62]}")
        for note in notes:
            print(f"        — {note}")
            problems.append((title, note))

    elapsed = time.monotonic() - started
    print(f"\n{'=' * 64}")
    print(f"Проверено документов: {len(titles)} за {elapsed:.0f} с")
    if not problems:
        print("Замечаний нет.")
        db.close()
        return 0

    print(f"Документов с замечаниями: {len({t for t, _ in problems})}, "
          f"замечаний всего: {len(problems)}\n")
    worst = defaultdict(int)
    for title, _ in problems:
        worst[title] += 1
    print("Больше всего замечаний:")
    for title, count in sorted(worst.items(), key=lambda x: -x[1])[:12]:
        print(f"  {count:3}  {title[:64]}")
    db.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
