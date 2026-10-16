"""
Восстановление пунктов, затёртых чужой темой форума.

Что случилось: документы опознавались по одному названию, а названия тем
на форуме не уникальны. «Общие правила банд» на 54 пункта и короткая тема
с тем же названием на 2 пункта считались одним документом — и пункты 1.1
и 1.2 были перезаписаны чужим текстом.

Оригиналы при этом никуда не делись: перед каждой перезаписью прошлая
редакция уходит в таблицу rule_versions. Этот скрипт возвращает самую
раннюю сохранённую редакцию каждого пострадавшего пункта.

Текст чужой темы при этом теряется — но её страница лежит в raw_text
того же документа, и после восстановления её можно импортировать заново:
теперь она уйдёт в отдельный документ.

Запуск:
    python tools/repair_overwrites.py           # показать, что будет сделано
    python tools/repair_overwrites.py --apply   # восстановить
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backup import snapshot  # noqa: E402
from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402


def main() -> int:
    apply = "--apply" in sys.argv

    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))

    if apply:
        # Снимок до правки: восстановить базу больше неоткуда —
        # 119 документов скопированы с форума вручную.
        saved = snapshot(cfg.resolve_path("storage.legal_db"), "repair-overwrites")
        if saved:
            print(f"Снимок базы: {saved.name}" + chr(10))
    conn = db._conn

    # Самая ранняя сохранённая редакция каждого пункта — то, что там было
    # до первой перезаписи.
    rows = conn.execute("""
        SELECT v.id AS version_id, v.rule_id, v.text, v.sanction, v.notes,
               v.content_hash, v.replaced_at,
               r.number, r.text AS current_text, d.title
        FROM rule_versions v
        JOIN rules r ON r.id = v.rule_id
        JOIN documents d ON d.id = r.document_id
        WHERE v.id = (SELECT MIN(v2.id) FROM rule_versions v2
                      WHERE v2.rule_id = v.rule_id)
        ORDER BY d.title, r.number
    """).fetchall()

    if not rows:
        print("Перезаписанных пунктов нет — восстанавливать нечего.")
        db.close()
        return 0

    by_document: dict[str, list] = {}
    for row in rows:
        by_document.setdefault(row["title"], []).append(row)

    print(f"Пунктов к восстановлению: {len(rows)} "
          f"в {len(by_document)} документах\n")

    for title, items in by_document.items():
        print(f"«{title}» — {len(items)} п.: "
              + ", ".join(item["number"] for item in items))
        sample = items[0]
        print(f"    п.{sample['number']}")
        print(f"      сейчас       : {sample['current_text'][:88]}")
        print(f"      восстановится: {sample['text'][:88]}")

    saved_raw: dict[str, str] = {}
    for title in by_document:
        row = conn.execute(
            "SELECT raw_text FROM documents WHERE title = ?", (title,)
        ).fetchone()
        if row and row["raw_text"]:
            saved_raw[title] = row["raw_text"]

    if not apply:
        print("\nЭто черновой прогон, база не изменена.")
        print("Чтобы восстановить: python tools/repair_overwrites.py --apply")
        db.close()
        return 0

    # Страницы чужих тем сохраняем на диск до того, как затрём raw_text:
    # они пригодятся, чтобы импортировать их заново уже как отдельные
    # документы.
    rescue_dir = cfg.root / "data" / "rescued_pages"
    rescue_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for title, raw in saved_raw.items():
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        path = rescue_dir / f"{stamp}_{safe[:60]}.txt"
        # newline="" обязателен: текст из буфера обмена уже содержит CRLF,
        # а Windows при записи превратил бы их в CRCRLF — при чтении
        # строки удвоились бы, и заголовок темы не нашёлся бы.
        path.write_text(raw, encoding="utf-8", newline="")
        print(f"\nСтраница чужой темы сохранена: {path.name}")

    now = datetime.now().isoformat(timespec="seconds")
    restored = 0
    with conn:
        for row in rows:
            conn.execute(
                "UPDATE rules SET text = ?, sanction = ?, notes = ?, "
                "content_hash = ?, updated_at = ? WHERE id = ?",
                (row["text"], row["sanction"], row["notes"],
                 row["content_hash"], now, row["rule_id"]))
            # Восстановленную редакцию из истории убираем: она снова стала
            # текущей, и держать её копию незачем.
            conn.execute("DELETE FROM rule_versions WHERE id = ?",
                         (row["version_id"],))
            restored += 1

        # raw_text этих документов принадлежит чужой теме — он врёт про
        # содержимое документа, и переразбор по нему всё сломал бы заново.
        for title in by_document:
            conn.execute(
                "UPDATE documents SET raw_text = NULL WHERE title = ?",
                (title,))

    print(f"\nВосстановлено пунктов: {restored}")
    print(f"Состояние базы: {db.stats()}")
    if saved_raw:
        print(f"\nСтраницы чужих тем лежат в {rescue_dir}")
        print("Их можно импортировать заново — теперь они попадут "
              "в отдельные документы:")
        print("   python tools/import_file.py data/rescued_pages/<файл>")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
