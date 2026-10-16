"""
Поправка на то, чем реально пользуются.

Идея простая: Уголовный кодекс спрашивают каждый день, а положение
о баллах лидеров — раз в месяц, и при прочих равных первым должен идти
первый. Опасность тоже простая: привычный документ не должен заслонять
точное попадание в редкий закон. Здесь стережётся и то, и другое.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.legal.db import LegalDB  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402


@pytest.fixture
def db(tmp_path) -> LegalDB:
    return LegalDB(tmp_path / "legal.db")


def add_documents(db: LegalDB, titles) -> None:
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    with db._conn:
        db._conn.executemany(
            "INSERT INTO documents (title, content_hash, imported_at, "
            "updated_at) VALUES (?, 'x', ?, ?)",
            [(t, now, now) for t in titles])


def test_usage_is_counted_only_once_per_answer(db):
    add_documents(db, ["Уголовный Кодекс", "Правила форума"])
    # Один ответ сослался на две статьи одного кодекса — это одно
    # обращение к документу, а не два.
    db.note_usage(["Уголовный Кодекс", "Уголовный Кодекс"])
    assert db.usage_weights()["Уголовный Кодекс"] == 1

    db.note_usage(["Уголовный Кодекс"])
    assert db.usage_weights()["Уголовный Кодекс"] == 2
    assert db.usage_weights()["Правила форума"] == 0


def test_unknown_document_is_ignored(db):
    """Название могло смениться — падать из-за этого нельзя."""
    add_documents(db, ["Уголовный Кодекс"])
    db.note_usage(["Такого документа нет"])
    assert db.usage_weights() == {"Уголовный Кодекс": 0}


def test_empty_usage_changes_nothing(db):
    add_documents(db, ["Уголовный Кодекс"])
    r = Retriever(db)
    assert r._usage_factor("Уголовный Кодекс") == 1.0


def test_popular_document_gets_a_bonus(db):
    add_documents(db, ["Уголовный Кодекс", "Правила форума"])
    for _ in range(50):
        db.note_usage(["Уголовный Кодекс"])
    db.note_usage(["Правила форума"])

    r = Retriever(db)
    popular = r._usage_factor("Уголовный Кодекс")
    rare = r._usage_factor("Правила форума")
    assert popular > rare > 1.0


def test_bonus_is_bounded(db):
    """Фора ограничена: частота не должна решать за поиск.

    Без потолка документ с тысячей обращений перекрывал бы разрыв
    в очках между точным попаданием и случайным совпадением.
    """
    add_documents(db, ["Уголовный Кодекс"])
    for _ in range(1000):
        db.note_usage(["Уголовный Кодекс"])
    r = Retriever(db)
    assert r._usage_factor("Уголовный Кодекс") <= 1.0 + r.usage_weight


def test_weight_zero_disables_everything(db):
    add_documents(db, ["Уголовный Кодекс"])
    for _ in range(100):
        db.note_usage(["Уголовный Кодекс"])
    r = Retriever(db, usage_weight=0)
    assert r._usage_factor("Уголовный Кодекс") == 1.0


def test_cache_is_dropped_after_update(db):
    """Иначе фора замирает на значениях, снятых при запуске."""
    add_documents(db, ["Уголовный Кодекс", "Правила форума"])
    db.note_usage(["Правила форума"])
    r = Retriever(db)
    before = r._usage_factor("Уголовный Кодекс")

    for _ in range(20):
        db.note_usage(["Уголовный Кодекс"])
    assert r._usage_factor("Уголовный Кодекс") == before, "кэш обязан держаться"

    r.forget_usage()
    assert r._usage_factor("Уголовный Кодекс") > before
