"""
Добавление образца документа в набор шаблонов (ТЗ 5.5).

Образцы нужны генератору как few-shot примеры: по ним он перенимает
структуру, заголовки и обороты. Чем их больше, тем ближе результат
к принятому на сервере стилю.

Наборов два и они не смешиваются: структура прокурорского дела и
судебного документа разные.

Запуск:
    python tools/add_template.py --clipboard              делопроизводство
    python tools/add_template.py --clipboard --court      судопроизводство
    python tools/add_template.py <файл.txt> [ещё файлы]
    python tools/add_template.py --list                   что уже есть
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.documents import KINDS, TemplateStore  # noqa: E402


def clipboard_text() -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
        capture_output=True, text=True, encoding="utf-8")
    return result.stdout or ""


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)
    store = TemplateStore(cfg.resolve_path("storage.templates_dir"))
    kind = "court" if "--court" in sys.argv else "proceedings"

    if "--list" in sys.argv or len(sys.argv) == 1:
        print("Образцы документов:\n")
        for key, title in KINDS.items():
            templates = store.load(key).templates
            print(f"  {title}: {len(templates)}")
            for template in templates:
                print(f"     — {template.name}  ({len(template.text)} симв.)")
            if not templates:
                print("     (пусто)")
        print("\nДобавить: python tools/add_template.py --clipboard"
              " [--court]")
        return 0

    texts: list[tuple[str, str]] = []
    if "--clipboard" in sys.argv:
        text = clipboard_text()
        if not text.strip():
            print("Буфер обмена пуст.")
            return 1
        texts.append(("", text))
    else:
        for argument in sys.argv[1:]:
            if argument.startswith("-"):
                continue
            path = Path(argument)
            if not path.exists():
                print(f"{path}: файл не найден")
                continue
            texts.append((path.stem, path.read_text(encoding="utf-8")))

    if not texts:
        print(__doc__)
        return 2

    added = 0
    for name, text in texts:
        try:
            path = store.add(kind, text, name)
        except ValueError as exc:
            print(f"Не добавлено: {exc}")
            continue
        print(f"Добавлено в «{KINDS[kind]}»: {path.name}")
        added += 1

    print(f"\nВсего образцов: {store.counts()}")
    return 0 if added else 1


if __name__ == "__main__":
    raise SystemExit(main())
