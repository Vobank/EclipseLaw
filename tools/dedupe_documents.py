"""
Убирает документы-дубли: одну и ту же тему форума в двух экземплярах.

Откуда они берутся: на форуме рядом с действующей темой лежит старая
или частично скопированная — «Общие правила банд» на 54 пункта и
«Общие правила банд (30 Авг 2026)» на два. Огрызок не приносит ничего,
зато отбирает место в выдаче у настоящего документа и путает поиск по
номеру: пункт 1.1 есть в обоих.

Удаляется только то, что заведомо лишнее. Для каждой пары считается,
какая доля текста меньшего документа уже есть в большем. Совпало всё —
дубль сносится. Нашёлся хоть один свой пункт — документ остаётся, а
его уникальные пункты печатаются: решать, что с ними делать, должен
человек, а не скрипт.

Запуск:
    python tools/dedupe_documents.py           показать, что найдено
    python tools/dedupe_documents.py --apply   удалить полные дубли
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backup import snapshot  # noqa: E402
from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB, short_document_name  # noqa: E402

# Доля совпавшего текста, после которой документ считается дублем.
# Не 100%: при копировании в разное время в огрызок попадает строка-две
# форумного мусора, и требовать полного совпадения значит не удалить
# ни одного настоящего дубля.
COVERED = 0.9


def group_key(title: str) -> str:
    """Название без редакции, даты, регистра и падежных хвостов."""
    text = short_document_name(title).lower()
    text = re.sub(r"[^а-яёa-z0-9 ]+", " ", text)
    text = re.sub(r"\b(ы|и|а|ов|ам|ах)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def rule_key(text: str) -> str:
    return re.sub(r"[^а-яёa-z0-9]+", "", text.lower())[:120]


def main() -> int:
    apply = "--apply" in sys.argv

    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))

    by_document: dict[str, list] = defaultdict(list)
    for rule in db.all_rules():
        by_document[rule.document].append(rule)

    groups: dict[str, list[str]] = defaultdict(list)
    for title in by_document:
        groups[group_key(title)].append(title)

    doomed: list[tuple[str, int]] = []
    kept: list[tuple[str, list]] = []

    for _, titles in sorted(groups.items()):
        if len(titles) < 2:
            continue
        titles.sort(key=lambda t: -len(by_document[t]))
        main_title, *rest = titles
        main_keys = {rule_key(r.text) for r in by_document[main_title]}

        print(f"«{main_title}» — {len(by_document[main_title])} п. "
              f"(основной)")
        for other in rest:
            rules = by_document[other]
            unique = [r for r in rules if rule_key(r.text) not in main_keys]
            share = 1 - len(unique) / len(rules)
            verdict = "ДУБЛЬ" if share >= COVERED else "оставить"
            print(f"   {verdict:9} «{other}» — {len(rules)} п., "
                  f"совпадает {share * 100:.0f}%")
            if share >= COVERED:
                doomed.append((other, len(rules)))
            else:
                kept.append((other, unique))
                for rule in unique[:3]:
                    print(f"       своё: п. {rule.number} {rule.text[:70]}")
        print()

    print("-" * 66)
    print(f"Полных дублей: {len(doomed)} "
          f"({sum(n for _, n in doomed)} пунктов)")
    print(f"Со своим текстом (не трогаем): {len(kept)}")

    if not doomed:
        db.close()
        return 0

    if not apply:
        print("\nЧерновой прогон. Удалить: "
              "python tools/dedupe_documents.py --apply")
        db.close()
        return 0

    saved = snapshot(cfg.resolve_path("storage.legal_db"), "dedupe")
    if saved:
        print(f"\nСнимок базы: {saved.name}")

    removed = sum(db.delete_document(title) for title, _ in doomed)
    print(f"Удалено документов: {len(doomed)}, пунктов: {removed}")
    print(f"Осталось документов: {len(db.documents())}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
