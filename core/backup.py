"""
Снимки базы законов перед операциями, которые её меняют.

База — 119 документов, скопированных с форума вручную. Восстановить её
неоткуда: исходники страниц сохранены не у всех, а сам форум копировать
автоматически запрещено (ТЗ, раздел 2). Любая ошибка импорта или
переразбора, замеченная через неделю, означает недели работы заново.

Снимок делается средствами SQLite, а не копированием файла: копия
файла, снятая во время записи, может оказаться битой, а backup API
согласован с текущими транзакциями.

Старые снимки удаляются, остаются последние keep штук. Без ротации
полтора десятка копий по 37 МБ незаметно съедают полгигабайта.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_KEEP = 10
SUFFIX = ".bak.db"


def snapshot(db_path, reason: str = "", keep: int = DEFAULT_KEEP) -> Path | None:
    """Делает снимок базы. Возвращает путь к нему или None при неудаче.

    Неудача снимка не должна срывать саму операцию: лучше импорт без
    резервной копии, чем отказ работать. Поэтому здесь ничего не
    выбрасывается наружу — только запись в лог.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None

    folder = db_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in reason)[:24]
    target = folder / f"{db_path.stem}-{stamp}{'-' + tag if tag else ''}{SUFFIX}"

    try:
        source = sqlite3.connect(str(db_path))
        try:
            destination = sqlite3.connect(str(target))
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()
    except Exception:
        log.exception("Снимок базы не сделан: %s", db_path)
        return None

    log.info("Снимок базы: %s (%.1f МБ)", target.name,
             target.stat().st_size / 1_048_576)
    _rotate(folder, db_path.stem, keep)
    return target


def _rotate(folder: Path, stem: str, keep: int) -> None:
    snapshots = sorted(folder.glob(f"{stem}-*{SUFFIX}"))
    for stale in snapshots[:-keep] if keep > 0 else []:
        try:
            stale.unlink()
            log.info("Удалён старый снимок: %s", stale.name)
        except OSError:
            log.warning("Не удалось удалить снимок %s", stale.name)


def list_snapshots(db_path) -> list[Path]:
    """Снимки от новых к старым."""
    db_path = Path(db_path)
    folder = db_path.parent / "backups"
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{db_path.stem}-*{SUFFIX}"), reverse=True)
