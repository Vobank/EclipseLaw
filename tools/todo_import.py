"""
Что осталось перекопировать с форума.

Список строится по текущему состоянию базы, а не лежит записанным:
скопировали документ — он пропал из списка сам. Так не нужно держать
галочки в голове и сверяться с перепиской.

Три группы, по убыванию важности:

  1. Обрывки — разобрался один-два пункта вместо документа. Это
     фактически отсутствующий документ.
  2. Разорванный текст — больше трети пунктов короче 25 символов.
     Так выглядит таблица, рассыпавшаяся при копировании: прайсы,
     реестры, формы заявлений.
  3. Без исходника — документ разобран нормально и работает, но
     сохранённой копии страницы нет, и переразобрать его после правки
     парсера не получится.

Запуск:
    python tools/todo_import.py           список
    python tools/todo_import.py --all     с третьей группой целиком
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402

# Пункт короче этого — скорее огрызок таблицы, чем норма.
SHORT_RULE = 25
# Доля огрызков, после которой документ считается разорванным.
TORN_SHARE = 0.3


def main() -> int:
    everything = "--all" in sys.argv

    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))

    by_document: dict[str, list] = {}
    for rule in db.all_rules():
        by_document.setdefault(rule.document, []).append(rule)

    kinds = {row["title"]: (row["kind"] or "rules")
             for row in db._conn.execute("SELECT title, kind FROM documents")}
    no_raw = set(db.documents_without_raw())

    stubs, torn, no_source = [], [], []
    for title, rules in by_document.items():
        short = sum(1 for r in rules if len(r.text.strip()) < SHORT_RULE)
        if len(rules) <= 2:
            stubs.append((title, len(rules), short))
        elif (short / len(rules) > TORN_SHARE
                and kinds.get(title) != "reference"):
            # Справочные страницы из коротких пунктов состоят по своей
            # природе: «Психологическая помощь», «Выдача мед. карт» —
            # это перечень услуг, а не обрывки таблицы. Проверка на
            # длину загнала сюда шесть исправных прайсов и отправила
            # пользователя копировать их заново впустую.
            torn.append((title, len(rules), short))
        elif title in no_raw:
            no_source.append((title, len(rules), short))

    rules_key = cfg.hotkeys.get("import_laws", "ctrl+alt+i")
    reference_key = cfg.hotkeys.get("import_reference", "ctrl+alt+u")

    def hotkey_for(title: str) -> str:
        return reference_key if kinds.get(title) == "reference" else rules_key

    print(f"Всего документов в базе: {len(by_document)}\n")

    print("=" * 66)
    print("ГРУППА 1 — ОБРЫВКИ. Разобрался один-два пункта вместо документа.")
    print("=" * 66)
    if not stubs:
        print("  Пусто — все документы разобрались целиком.\n")
    for title, count, _ in sorted(stubs, key=lambda x: x[1]):
        print(f"  [ ] {title}")
        print(f"        сейчас {count} п.   импорт: {hotkey_for(title)}")
    print()

    print("=" * 66)
    print("ГРУППА 2 — РАЗОРВАННЫЙ ТЕКСТ. Таблица рассыпалась при копировании.")
    print("=" * 66)
    if not torn:
        print("  Пусто.\n")
    for title, count, short in sorted(torn, key=lambda x: -x[2] / x[1]):
        print(f"  [ ] {title}")
        print(f"        {short} огрызков из {count} п.   "
              f"импорт: {hotkey_for(title)}")
    print()

    print("=" * 66)
    print("ГРУППА 3 — БЕЗ ИСХОДНИКА. Работают, но не переразобрать позже.")
    print("=" * 66)
    shown = no_source if everything else no_source[:5]
    for title, count, _ in sorted(shown, key=lambda x: -x[1]):
        print(f"  [ ] {title}")
        print(f"        {count} п.   импорт: {hotkey_for(title)}")
    if not everything and len(no_source) > 5:
        print(f"  ... и ещё {len(no_source) - 5}. "
              f"Полностью: python tools/todo_import.py --all")
    print()

    must = len(stubs) + len(torn)
    print("-" * 66)
    print(f"ОБЯЗАТЕЛЬНО перекопировать: {must} документов")
    print(f"По желанию (группа 3)     : {len(no_source)}")
    print(f"Трогать не нужно          : "
          f"{len(by_document) - must - len(no_source)}")
    print()
    print(f"Как: на странице форума Ctrl+A, Ctrl+C, затем в боте "
          f"{rules_key}")
    print(f"     для прайсов, реестров и форм заявлений — {reference_key}")
    print("Векторы бот пересчитает сам, вручную ничего запускать не нужно.")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
