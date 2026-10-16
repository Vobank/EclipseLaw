"""
Построение векторного индекса базы законов (этап 4).

Считает эмбеддинги для новых и изменившихся пунктов. Запускать после
каждого крупного импорта — без индекса смысловой поиск не работает,
и ассистент отвечает «в базе не нашлось статей», хотя база полна.

Повторный запуск дёшев: пересчитываются только те пункты, у которых
изменился текст.

Запуск:
    python tools/build_index.py            # достроить недостающее
    python tools/build_index.py --rebuild  # пересчитать всё заново
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.vectors import (  # noqa: E402
    EmbeddingClient, EmbeddingError, VectorIndex)


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    client = EmbeddingClient(
        cfg.get("llm.host", "http://127.0.0.1:11434"),
        cfg.get("retrieval.embedding_model", "bge-m3"),
        batch_size=cfg.get("retrieval.embedding_batch", 32))

    ready, reason = client.available()
    print(f"Модель эмбеддингов: {reason}")
    if not ready:
        return 1

    db = LegalDB(cfg.resolve_path("storage.legal_db"))
    index = VectorIndex(db, client)

    if "--rebuild" in sys.argv:
        with db._conn:
            db._conn.execute("DELETE FROM rule_vectors")
        print("Индекс очищен, пересчитываю всё заново.")

    before = index.stats()
    print(f"До : пунктов {before['пунктов']}, векторов {before['векторов']}")
    if before["пунктов"] == before["векторов"]:
        print("Индекс уже полный — проверю только изменившиеся пункты.")

    last = [time.monotonic()]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] < 2.0 and done < total:
            return
        last[0] = now
        print(f"  {done}/{total} ({done * 100 // max(total, 1)}%)", flush=True)

    try:
        stats = index.build(on_progress=progress)
    except EmbeddingError as exc:
        print(f"\nПРОВАЛ: {exc}")
        db.close()
        return 1

    print(f"\nГотово: {stats.summary()}")
    if stats.missing:
        print(f"ВНИМАНИЕ: без вектора осталось пунктов: {stats.missing}")

    # Проверка на живом запросе: индекс, который ничего не находит,
    # формально построен, но бесполезен.
    print("\nПробный поиск:")
    for query in ("что грозит за превышение скорости",
                  "обязан ли сотрудник представиться при задержании"):
        hits = index.search(query, limit=3)
        print(f"  «{query}»")
        for rule_id, score in hits:
            rule = db.rule_by_id(rule_id)
            if rule:
                print(f"     {score:.3f}  [{rule.document[:34]} / "
                      f"{rule.number}] {rule.text[:56]}")
        if not hits:
            print("     ПРОВАЛ: ничего не найдено")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
