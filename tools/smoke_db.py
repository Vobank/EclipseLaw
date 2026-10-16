"""
Проверка базы законов: импорт, идемпотентность, история редакций, поиск.

Работает на временной базе, боевую data/legal.db не трогает.

Запуск: python tools/smoke_db.py <файл_с_правилами.txt>
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.parser import ForumParser  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    cfg = Config.load()
    setup_logging(cfg)
    parser = ForumParser.from_config(cfg)
    parsed = parser.parse(raw)

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        db = LegalDB(Path(tmp) / "test.db")

        # --- 1. первый импорт ------------------------------------------
        first = db.import_document(parsed)
        print(f"1) Первый импорт : {first.summary()}")
        if first.added != len(parsed.rules) or first.updated:
            print(f"   ПРОВАЛ: ожидалось {len(parsed.rules)} новых, 0 обновлённых")
            failures += 1

        # --- 2. повторный импорт того же текста ------------------------
        second = db.import_document(parsed)
        print(f"2) Повторный     : {second.summary()}")
        if second.added or second.updated:
            print("   ПРОВАЛ: повторный импорт того же текста должен "
                  "оставить всё без изменений")
            failures += 1
        else:
            print("   идемпотентность — OK")

        # --- 3. изменённый пункт ---------------------------------------
        target = parsed.rules[5]
        original_text = target.text
        target.text = original_text + " Дополнено при проверке."
        third = db.import_document(parsed)
        print(f"3) После правки  : {third.summary()}")
        if third.updated != 1:
            print(f"   ПРОВАЛ: ожидался ровно 1 обновлённый пункт, "
                  f"получено {third.updated}")
            failures += 1

        versions = db._conn.execute(
            "SELECT COUNT(*) AS n FROM rule_versions").fetchone()["n"]
        if versions == 1:
            print("   прошлая редакция сохранена — OK")
        else:
            print(f"   ПРОВАЛ: в истории {versions} записей, ожидалась 1")
            failures += 1
        target.text = original_text

        # --- 4. точный поиск по номеру ---------------------------------
        found = db.get_rule("6.26")
        if found:
            rule = found[0]
            print(f"\n4) Поиск по номеру 6.26:")
            print(f"   {rule.citation()}")
            print(f"   {rule.text[:110]}...")
            print(f"   санкция: {rule.sanction}")
        else:
            print("4) ПРОВАЛ: пункт 6.26 не найден по номеру")
            failures += 1

        # --- 5. полнотекстовый поиск -----------------------------------
        for query in ("читы и модификации", "маска похищение заложник",
                      "оскорбление в чате"):
            hits = db.search(query, limit=3)
            print(f"\n5) Поиск «{query}» — найдено {len(hits)}:")
            for hit in hits:
                print(f"   [{hit.number}] {hit.text[:80]}...")
            if not hits:
                print("   ПРОВАЛ: полнотекстовый поиск ничего не вернул")
                failures += 1

        print(f"\nСводка базы: {db.stats()}")
        print("Документы:")
        for title, count, updated in db.documents():
            print(f"   «{title}» — пунктов {count}, обновлён {updated}")
        db.close()

    print("\nИтог:", "база работает" if failures == 0
          else f"ПРОВАЛОВ: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
