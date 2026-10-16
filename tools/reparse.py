"""
Переразбор уже импортированных документов после правки парсера.

Когда парсер учится чему-то новому — новому виду нумерации, новому слову
для санкции, — старые документы в базе остаются разобранными по-старому.
Этот инструмент прогоняет их через парсер заново из сохранённого
исходника, без повторного копирования страниц с форума.

Документы, импортированные до появления хранения исходника, переразобрать
нельзя — их придётся скопировать с форума ещё раз. Скрипт назовёт их
поимённо.

Запуск:
    python tools/reparse.py            # показать, что будет сделано
    python tools/reparse.py --apply    # переразобрать
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backup import snapshot  # noqa: E402
from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.parser import ForumParser  # noqa: E402


def main() -> int:
    apply = "--apply" in sys.argv

    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))

    if apply:
        # Снимок до правки: восстановить базу больше неоткуда —
        # 119 документов скопированы с форума вручную.
        saved = snapshot(cfg.resolve_path("storage.legal_db"), "reparse")
        if saved:
            print(f"Снимок базы: {saved.name}" + chr(10))
    parser = ForumParser.from_config(cfg)

    stored = db.raw_documents()
    orphans = db.documents_without_raw()

    if orphans:
        print(f"Без сохранённого исходника ({len(orphans)}) — "
              "переразобрать нельзя, нужно скопировать с форума заново:")
        for title in orphans:
            print(f"   — «{title}»")
        print()

    if not stored:
        print("Нет ни одного документа с сохранённым исходником.")
        print("Исходник сохраняется начиная с этой версии: импортируйте "
              "страницы заново, дальше переразбор будет работать.")
        db.close()
        return 0

    if not apply:
        print(f"Будет переразобрано документов: {len(stored)}")
        print("Это черновой прогон. Чтобы применить: "
              "python tools/reparse.py --apply\n")

    before = db.stats()
    total_added = total_updated = total_removed = 0

    kinds = {row["title"]: (row["kind"] or "rules") for row in db._conn.execute(
        "SELECT title, kind FROM documents")}

    for title, raw in stored:
        # Справочные страницы обязаны переразбираться справочным
        # разбором: прогони прайс EMS через разбор правил — получишь
        # ноль пунктов и пустой документ вместо цен.
        if kinds.get(title) == "reference":
            parsed = parser.parse_reference(raw, title=title)
        else:
            parsed = parser.parse(raw, title=title)
        sanctions = sum(1 for r in parsed.rules if r.sanction)
        leftover = sum(1 for r in parsed.rules if "|" in r.text)

        if apply:
            # Сначала импорт поверх, потом уборка. Порядок принципиален:
            # при обратном (снести всё и импортировать заново) у каждого
            # пункта менялся id, каскадом отваливались векторы, а
            # прошлые редакции не сохранялись — путь удаления минует
            # версионирование. Импорт поверх обновляет существующие
            # пункты штатно, а здесь убирается только то, что новый
            # разбор пунктом больше не считает: строки-заголовки вроде
            # «1. Общие положения» в уставе Weazel News.
            result = db.import_document(parsed, raw_text=raw)
            removed = db.prune_rules(title, (r.number for r in parsed.rules))
            total_added += result.added
            total_updated += result.updated
            total_removed += removed
            mark = (f"новых {result.added}, обновлено {result.updated}, "
                    f"удалено {removed}")
        else:
            mark = "черновой прогон"

        print(f"«{title}»")
        print(f"   пунктов {len(parsed.rules)}, с санкцией {sanctions}, "
              f"с остатком «|» в тексте {leftover}  [{mark}]")
        for warning in parsed.warnings[:3]:
            print(f"   ! {warning}")

    if apply:
        after = db.stats()
        print(f"\nИтого: обновлено {total_updated}, добавлено {total_added}, "
              f"удалено {total_removed}")
        print(f"Было : {before}")
        print(f"Стало: {after}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
