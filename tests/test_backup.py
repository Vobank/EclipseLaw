"""Снимки базы: восстановить её больше неоткуда."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backup import list_snapshots, snapshot  # noqa: E402


def make_db(path: Path, rows: int = 3) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE rules (id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany("INSERT INTO rules (text) VALUES (?)",
                     [(f"норма {i}",) for i in range(rows)])
    conn.commit()
    conn.close()


def test_snapshot_is_a_working_database(tmp_path):
    source = tmp_path / "legal.db"
    make_db(source, rows=5)

    copy = snapshot(source, "test")
    assert copy is not None and copy.exists()

    conn = sqlite3.connect(str(copy))
    assert conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 5
    conn.close()


def test_snapshot_survives_changes_to_the_original(tmp_path):
    """Смысл снимка в том, что он не меняется вместе с базой."""
    source = tmp_path / "legal.db"
    make_db(source, rows=3)
    copy = snapshot(source, "before")

    conn = sqlite3.connect(str(source))
    conn.execute("DELETE FROM rules")
    conn.commit()
    conn.close()

    restored = sqlite3.connect(str(copy))
    assert restored.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 3
    restored.close()


def test_rotation_keeps_only_the_last_few(tmp_path):
    """Без ротации полтора десятка копий по 37 МБ съедают полгигабайта."""
    source = tmp_path / "legal.db"
    make_db(source)
    for index in range(6):
        snapshot(source, f"run{index}", keep=3)
    assert len(list_snapshots(source)) == 3


def test_missing_database_is_not_an_error(tmp_path):
    """Первый запуск: базы ещё нет, и это не повод падать."""
    assert snapshot(tmp_path / "нет-такой.db", "test") is None
