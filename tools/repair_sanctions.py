"""
Чинит текст пунктов прямо в базе, без переразбора.

Делает две вещи: вынимает санкцию, оставшуюся внутри текста после «|»,
и срезает служебные вставки форума («Спойлер: ...», «Последнее
редактирование ...»), приклеенные к норме в той же строке.

Зачем отдельный инструмент, когда есть reparse: шестнадцать документов
попали в базу до того, как исходник страницы стал сохраняться. Их
переразобрать не из чего, а санкция у них так и лежит внутри текста
после «|» — и в ответе не видно, что за нарушение сколько стоит.

Здесь текст пункта прогоняется через тот же _split_sanction, что и при
импорте. Никакой отдельной логики: если парсер научится новому слову,
этот инструмент научится вместе с ним.

Трогаются только пункты, у которых санкция сейчас пустая. Уже
разобранное не перезаписывается — чтобы повторный запуск ничего не
испортил.

Запуск:
    python tools/repair_sanctions.py            показать, что изменится
    python tools/repair_sanctions.py --apply    применить
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
        saved = snapshot(cfg.resolve_path("storage.legal_db"), "repair")
        if saved:
            print(f"Снимок базы: {saved.name}" + chr(10))
    parser = ForumParser.from_config(cfg)

    planned: list[tuple] = []
    for rule in db.all_rules():
        # Служебные вставки форума срезаются у всех пунктов, а не
        # только у тех, где потерялась санкция: «Спойлер: ...» и
        # «Последнее редактирование ...» попадали прямо в ответ.
        cleaned = parser.strip_inline_chrome(rule.text)
        if rule.sanction:
            if cleaned != rule.text and len(cleaned) >= 10:
                planned.append((rule, cleaned, rule.sanction))
            continue
        body, sanction = parser._split_sanction(rule.text)
        if not sanction:
            if cleaned != rule.text and len(cleaned) >= 10:
                planned.append((rule, cleaned, ""))
            continue
        # Пустой остаток означает, что весь пункт приняли за санкцию —
        # такой «ремонт» уничтожил бы текст статьи.
        if len(body) < 10:
            continue
        planned.append((rule, body, sanction))

    by_document: dict[str, int] = {}
    for rule, _, _ in planned:
        by_document[rule.document] = by_document.get(rule.document, 0) + 1

    print(f"Пунктов к починке: {len(planned)} "
          f"в {len(by_document)} документах\n")
    for title, count in sorted(by_document.items(),
                               key=lambda kv: -kv[1]):
        print(f"  {count:3}  {title[:60]}")

    print("\nПримеры:")
    for rule, body, sanction in planned[:6]:
        print(f"  [{rule.citation()}]")
        print(f"      текст:   {body[:90]}")
        print(f"      санкция: {sanction[:70]}")

    if not apply:
        print("\nЧерновой прогон. Применить: "
              "python tools/repair_sanctions.py --apply")
        db.close()
        return 0

    for rule, body, sanction in planned:
        db.update_text(rule.id, body, sanction)
    print(f"\nПочинено пунктов: {len(planned)}")

    remaining = sum(1 for r in db.all_rules() if not r.sanction and "|" in r.text)
    print(f"Осталось пунктов с «|» и без санкции: {remaining} "
          f"(это и есть настоящие разделители, а не санкции)")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
