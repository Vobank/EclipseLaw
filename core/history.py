"""
История запросов и ответов (ТЗ 5.6, этап 9).

Отдельная база от законов: та переразбирается и пересобирается целиком
после каждой правки парсера, а история — это летопись, её терять нельзя.

Пишется в фоне и молча: сбой записи в журнал не должен ронять ответ,
который пользователь уже читает.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY,
    timestamp   TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    mode        TEXT    NOT NULL,
    source      TEXT    NOT NULL,   -- я / собеседник / система
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    articles    TEXT,               -- JSON-массив ссылок на статьи
    -- Статьи, названные моделью, но отсутствующие в базе. Хранятся,
    -- чтобы можно было оценить, как часто модель выдумывает, и понять,
    -- пора ли менять её на более крупную.
    invented    TEXT,
    latency_ms  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_history_time    ON history (timestamp);
CREATE INDEX IF NOT EXISTS idx_history_session ON history (session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5 (
    question, answer, content='history', content_rowid='id',
    tokenize='unicode61'
);
"""

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS history_ai AFTER INSERT ON history BEGIN
    INSERT INTO history_fts (rowid, question, answer)
    VALUES (new.id, new.question, new.answer);
END;
CREATE TRIGGER IF NOT EXISTS history_ad AFTER DELETE ON history BEGIN
    INSERT INTO history_fts (history_fts, rowid, question, answer)
    VALUES ('delete', old.id, old.question, old.answer);
END;
"""


@dataclass
class HistoryEntry:
    id: int
    timestamp: str
    session_id: str
    mode: str
    source: str
    question: str
    answer: str
    articles: list[str]
    invented: list[str]
    latency_ms: int


@dataclass
class Case:
    """Одно дело: сессия целиком, как она показывается в списке."""

    session_id: str
    title: str
    mode: str
    started: str
    ended: str
    count: int

    @property
    def when(self) -> str:
        """«20 сен, 21:58» — дата в списке дел."""
        months = ("янв", "фев", "мар", "апр", "мая", "июн",
                  "июл", "авг", "сен", "окт", "ноя", "дек")
        try:
            stamp = datetime.fromisoformat(self.started)
        except ValueError:
            return self.started[:16]
        return f"{stamp.day} {months[stamp.month - 1]}, {stamp:%H:%M}"


# Вид документа в начале сгенерированной бумаги: по нему дело и узнаётся
# в списке. «ОПРЕДЕЛЕНИЕ о принятии искового заявления» говорит больше,
# чем первый надиктованный обрывок.
_DOCUMENT_KIND = re.compile(
    r"^\s*((?:ОПРЕДЕЛЕНИЕ|РЕШЕНИЕ|ПРИГОВОР|ИСКОВОЕ ЗАЯВЛЕНИЕ|ПОСТАНОВЛЕНИЕ|"
    r"ХОДАТАЙСТВО|ЖАЛОБА|СУДЕБНЫЙ ОРДЕР)[^\n]{0,60})", re.MULTILINE)

# Номер дела или постановления: «DJP-3505», «№6907», «DoC-CJA-07-6913».
_CASE_NUMBER = re.compile(r"(?:№\s*(\d{3,6})|\b([A-Z]{2,4}-[A-Z0-9-]{3,20}))")


def _case_title(entries: list, mode: str) -> str:
    """Название дела из его содержимого."""
    if not entries:
        return "Пустое дело"

    # Документ важнее вопросов: если в сессии оформлялась бумага,
    # дело называется по ней.
    for entry in entries:
        if entry.mode != "document":
            continue
        kind = _DOCUMENT_KIND.search(entry.answer or "")
        number = _CASE_NUMBER.search(entry.answer or "")
        name = kind.group(1).strip() if kind else "Документ"
        if number:
            name += f" {number.group(1) or number.group(2)}"
        return name[:70]

    prefix = {"lawyer": "Задержание", "exam": "Экзамен"}.get(mode, "")
    first = " ".join((entries[0].question or "").split())[:64]
    return f"{prefix}: {first}" if prefix else first or "Без названия"


class HistoryLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.executescript(TRIGGERS)
        self._migrate()
        self._conn.commit()
        log.info("История запросов: %s", self.path)

    def _migrate(self) -> None:
        """Достраивает колонки, появившиеся после первого запуска."""
        have = {row["name"] for row in
                self._conn.execute("PRAGMA table_info(history)")}
        if "wrong" not in have:
            # Пометка «ответ неверный». Без неё единственным способом
            # зафиксировать промах было написать разработчику, и
            # уверенно-неправильные ответы терялись бесследно — именно
            # они опаснее честного «не нашлось».
            self._conn.execute(
                "ALTER TABLE history ADD COLUMN wrong INTEGER DEFAULT 0")
            log.info("История: добавлена колонка wrong")

    def close(self) -> None:
        self._conn.close()

    def mark_wrong(self, entry_id: int | None = None) -> "HistoryEntry | None":
        """Помечает обмен как неверный. Без id — последний записанный."""
        try:
            with self._conn:
                if entry_id is None:
                    row = self._conn.execute(
                        "SELECT id FROM history ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    if row is None:
                        return None
                    entry_id = row["id"]
                self._conn.execute(
                    "UPDATE history SET wrong = 1 WHERE id = ?", (entry_id,))
            row = self._conn.execute(
                "SELECT * FROM history WHERE id = ?", (entry_id,)).fetchone()
            return self._to_entry(row) if row else None
        except Exception:
            log.exception("Не удалось пометить обмен как неверный")
            return None

    def cases(self, limit: int = 30) -> list["Case"]:
        """Список дел — по одному на сессию, от свежих к старым.

        Сессия и есть дело: она начинается сбросом контекста и живёт,
        пока человек разбирает одну ситуацию. Название выводится из
        содержимого, а не хранится отдельно: так оно не разъезжается
        с тем, что внутри, и не требует от человека что-то называть
        руками посреди задержания.
        """
        rows = self._conn.execute(
            "SELECT session_id, mode, COUNT(*) AS n, "
            "       MIN(timestamp) AS started, MAX(timestamp) AS ended "
            "FROM history GROUP BY session_id "
            "ORDER BY started DESC LIMIT ?", (limit,)).fetchall()

        result: list[Case] = []
        for row in rows:
            entries = self.session(row["session_id"])
            result.append(Case(
                session_id=row["session_id"],
                title=_case_title(entries, row["mode"]),
                mode=row["mode"],
                started=row["started"],
                ended=row["ended"],
                count=row["n"],
            ))
        return result

    def find_cases(self, query: str, limit: int = 30) -> list["Case"]:
        """Дела, подходящие под запрос: по номеру, названию, содержимому.

        Дел за неделю работы с прокуратурой набирается столько, что
        листать список бессмысленно — нужен номер. Номер постановления
        («DJP-3505») лежит внутри текста самого документа, поэтому
        ищем и по содержимому тоже, а не только по названию.

        Пустой запрос — это просто весь список, а не «ничего не
        найдено»: так строка поиска не отнимает обычный список дел.
        """
        query = " ".join((query or "").split())
        if not query:
            return self.cases(limit)

        everything = self.cases(limit=1000)
        needle = query.lower()

        # Сначала по названию: там уже стоит номер дела, и точное
        # совпадение важнее похожего слова где-то в середине ответа.
        by_title = [c for c in everything if needle in c.title.lower()]
        found = {c.session_id for c in by_title}

        # Потом по содержимому — через полнотекстовый индекс истории.
        by_text: list[Case] = []
        known = {c.session_id: c for c in everything}
        for entry in self.search(query, limit=200):
            if entry.session_id in found:
                continue
            case = known.get(entry.session_id)
            if case is not None:
                found.add(case.session_id)
                by_text.append(case)

        return (by_title + by_text)[:limit]

    def wrong_answers(self, limit: int = 50) -> list["HistoryEntry"]:
        """Помеченные промахи, от новых к старым."""
        rows = self._conn.execute(
            "SELECT * FROM history WHERE wrong = 1 "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._to_entry(row) for row in rows]

    def record(self, *, session_id: str, mode: str, source: str,
               question: str, answer: str, articles: list[str] | None = None,
               invented: list[str] | None = None,
               latency_ms: int = 0) -> None:
        """Пишет обмен. Сбой записи не должен ломать сам ответ."""
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO history (timestamp, session_id, mode, source, "
                    "question, answer, articles, invented, latency_ms) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"),
                     session_id, mode, source, question, answer,
                     json.dumps(articles or [], ensure_ascii=False),
                     json.dumps(invented or [], ensure_ascii=False),
                     latency_ms))
        except Exception:
            log.exception("Не удалось записать обмен в историю")

    # ------------------------------------------------------------------ чтение
    def _to_entry(self, row: sqlite3.Row) -> HistoryEntry:
        return HistoryEntry(
            id=row["id"], timestamp=row["timestamp"],
            session_id=row["session_id"], mode=row["mode"],
            source=row["source"], question=row["question"],
            answer=row["answer"],
            articles=json.loads(row["articles"] or "[]"),
            invented=json.loads(row["invented"] or "[]"),
            latency_ms=row["latency_ms"] or 0)

    def recent(self, limit: int = 20) -> list[HistoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._to_entry(r) for r in rows]

    def session(self, session_id: str) -> list[HistoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM history WHERE session_id = ? ORDER BY id",
            (session_id,)).fetchall()
        return [self._to_entry(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[HistoryEntry]:
        words = [w for w in "".join(
            c if c.isalnum() or c.isspace() else " " for c in query).split()
            if len(w) > 2]
        if not words:
            return []
        expression = " OR ".join(f"{w[:max(3, round(len(w) * 0.6))]}*"
                                 for w in words)
        try:
            rows = self._conn.execute(
                "SELECT h.* FROM history h JOIN history_fts f ON f.rowid = h.id "
                "WHERE history_fts MATCH ? ORDER BY rank LIMIT ?",
                (expression, limit)).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("Поиск по истории не отработал: %s", exc)
            return []
        return [self._to_entry(r) for r in rows]

    def stats(self) -> dict[str, int]:
        conn = self._conn
        total = conn.execute("SELECT COUNT(*) AS n FROM history").fetchone()["n"]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS n FROM history").fetchone()["n"]
        flagged = conn.execute(
            "SELECT COUNT(*) AS n FROM history "
            "WHERE invented IS NOT NULL AND invented != '[]'").fetchone()["n"]
        return {"записей": total, "сессий": sessions,
                "с выдуманными статьями": flagged}
