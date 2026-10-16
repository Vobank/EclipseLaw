"""
Импорт страницы правил из файла, а не из буфера обмена.

Нужен для спасённых страниц (data/rescued_pages) и вообще всегда, когда
текст уже лежит на диске: класть его обратно в буфер, чтобы нажать
хоткей, — лишний шаг.

Запуск:
    python tools/import_file.py <файл.txt> [ещё файлы...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.importer import ImportError_, LawImporter  # noqa: E402
from core.legal.parser import ForumParser  # noqa: E402


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2

    cfg = Config.load()
    setup_logging(cfg)
    db = LegalDB(cfg.resolve_path("storage.legal_db"))
    importer = LawImporter(db, ForumParser.from_config(cfg))

    failures = 0
    for path in paths:
        if not path.exists():
            print(f"{path}: файл не найден")
            failures += 1
            continue
        try:
            result = importer.import_text(path.read_text(encoding="utf-8"))
        except ImportError_ as exc:
            print(f"{path.name}: НЕ ИМПОРТИРОВАН\n   {exc}")
            failures += 1
            continue
        print(f"{path.name}: {result.summary()}")
        for warning in result.warnings:
            print(f"   ! {warning}")

    print(f"\nСостояние базы: {db.stats()}")
    db.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
