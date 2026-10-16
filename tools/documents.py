"""
Просмотр и удаление документов в базе законов.

Импорт иногда заходит криво: скопировалась часть страницы, попалась
тема-однофамилец, страница оказалась объявлением. Этот инструмент
показывает, что лежит в базе, и позволяет выкинуть лишнее.

Запуск:
    python tools/documents.py                      список всех документов
    python tools/documents.py --suspicious         только подозрительные
    python tools/documents.py --show "<название>"  содержимое документа
    python tools/documents.py --delete "<название>"          удалить (спросит)
    python tools/documents.py --delete "<название>" --yes    удалить без вопроса
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402

# Документ из одного-двух пунктов почти всегда означает, что страница
# скопировалась не целиком или её формат распознан лишь частично.
SUSPICIOUS_RULE_COUNT = 2


def argument_after(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def find_document(db: LegalDB, name: str) -> str | None:
    """Ищет документ по точному названию, затем по вхождению подстроки."""
    titles = [title for title, _, _ in db.documents()]
    if name in titles:
        return name

    matches = [t for t in titles if name.lower() in t.lower()]
    if not matches:
        print(f"Документ «{name}» не найден.")
        return None
    if len(matches) > 1:
        print(f"Под «{name}» подходит несколько документов — уточните:")
        for title in matches:
            print(f"   «{title}»")
        return None
    return matches[0]


def show_suspicious(db: LegalDB) -> None:
    documents = db.documents()
    tiny = [(t, n) for t, n, _ in documents if n <= SUSPICIOUS_RULE_COUNT]

    # Названия, различающиеся только регистром или окончанием, — признак
    # того, что одна и та же тема заехала дважды по-разному.
    def key(title: str) -> str:
        return "".join(c for c in title.lower() if c.isalnum())[:22]

    groups: dict[str, list[tuple[str, int]]] = {}
    for title, count, _ in documents:
        groups.setdefault(key(title), []).append((title, count))
    similar = {k: v for k, v in groups.items() if len(v) > 1}

    if tiny:
        print(f"Подозрительно короткие (≤{SUSPICIOUS_RULE_COUNT} п.) — "
              "вероятно, скопирована часть страницы:")
        for title, count in tiny:
            print(f"   {count:3} п.  «{title}»")
    if similar:
        print("\nПохожие названия — возможно, одна тема заехала дважды:")
        for group in similar.values():
            for title, count in group:
                print(f"   {count:3} п.  «{title}»")
            print()
    if not tiny and not similar:
        print("Подозрительных документов не найдено.")


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))

    try:
        if "--suspicious" in sys.argv:
            show_suspicious(db)
            return 0

        name = argument_after("--show")
        if name:
            title = find_document(db, name)
            if title is None:
                return 1
            rules = [r for r in db.all_rules() if r.document == title]
            print(f"«{title}» — пунктов {len(rules)}\n")
            for rule in rules:
                sanction = f"   [санкция: {rule.sanction}]" if rule.sanction else ""
                print(f"  [{rule.number}] {rule.text[:110]}{sanction}")
            return 0

        name = argument_after("--delete")
        if name:
            title = find_document(db, name)
            if title is None:
                return 1
            count = next(n for t, n, _ in db.documents() if t == title)
            print(f"Будет удалён документ «{title}» — {count} пунктов.")
            if "--yes" not in sys.argv:
                print("Удаление не выполнено. Повторите с флагом --yes, "
                      "если это точно нужно.")
                return 0
            # Пункты, их история и метки уходят каскадом по внешнему ключу.
            with db._conn:
                db._conn.execute("DELETE FROM documents WHERE title = ?",
                                 (title,))
            print(f"Удалено. Состояние базы: {db.stats()}")
            return 0

        documents = db.documents()
        print(f"Документов: {len(documents)}, "
              f"пунктов: {db.stats()['пунктов']}\n")
        for index, (title, count, updated) in enumerate(documents, 1):
            print(f"  {index:2}. {count:4} п.  {title}")
        print("\nПодозрительные записи: python tools/documents.py --suspicious")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
