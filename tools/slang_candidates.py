"""
Ищет в истории слова, из-за которых поиск ничего не находит.

Словарь сленга в config.yaml пополняется вручную, и угадывать, чего в
нём не хватает, бессмысленно — это видно по реальным вопросам. Скрипт
берёт из истории те, на которые ассистент ответил «в найденных статьях
ответа нет», и показывает слова, которых нет ни в одном документе базы.

Такое слово — кандидат в словарь: либо сленг («мент»), либо то, что
Whisper расслышал неправильно («хладность»). Что именно — решает
человек, скрипт ничего не правит сам.

Запуск:
    python tools/slang_candidates.py
    python tools/slang_candidates.py --all   по всем вопросам, не только
                                             по безответным
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.query import DEFAULT_STOPWORDS  # noqa: E402

WORD = re.compile(r"[а-яёА-ЯЁ]{4,}")

# Признаки того, что ассистент не нашёл нормы.
NO_ANSWER = ("в найденных статьях ответа нет",
             "в базе не нашлось статей",
             "не предусмотрено в базе")


def main() -> int:
    everything = "--all" in sys.argv

    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))

    import sqlite3
    history = sqlite3.connect(str(cfg.resolve_path("storage.history_db")))
    rows = history.execute("SELECT question, answer FROM history").fetchall()

    questions = [q for q, a in rows
                 if everything or any(m in (a or "").lower() for m in NO_ANSWER)]
    print(f"Вопросов в истории: {len(rows)}, "
          f"взято в разбор: {len(questions)}\n")
    if not questions:
        print("Безответных вопросов нет — словарь пополнять не из чего.")
        db.close()
        return 0

    # Всё, что встречается в базе, кандидатом быть не может.
    known: set[str] = set()
    for rule in db.all_rules():
        known.update(w.lower() for w in WORD.findall(rule.text))
    for title, _, _ in db.documents():
        known.update(w.lower() for w in WORD.findall(title))

    slang = {k.lower() for k in (cfg.get("retrieval.slang") or {})}

    counter: Counter = Counter()
    for question in questions:
        for word in WORD.findall(question):
            lowered = word.lower()
            if lowered in DEFAULT_STOPWORDS or lowered in known:
                continue
            # Уже покрыто словарём: ключи ищутся по началу слова.
            if any(lowered.startswith(prefix) for prefix in slang):
                continue
            counter[lowered] += 1

    if not counter:
        print("Незнакомых слов не нашлось.")
        db.close()
        return 0

    print("Слова из безответных вопросов, которых нет ни в одном документе.")
    print("Добавьте нужные в retrieval.slang — ключ по началу слова,")
    print("значение — как это называется в законе.\n")
    for word, count in counter.most_common(30):
        print(f"  {count:3}  {word}")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
