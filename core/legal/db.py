"""
Хранилище правил в SQLite (ТЗ 5.3).

Документов может быть много: общие правила проекта, правила фракций,
зелёных зон и так далее. Нумерация у них своя и пересекается — пункт 1.1
есть в каждом документе, — поэтому пункт уникален парой
(документ, номер), а не номером.

При повторном импорте текст сравнивается по хэшу: не изменился —
ничего не трогаем, изменился — прошлая редакция уезжает в rule_versions,
и её всегда можно поднять.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.legal.parser import ParsedDocument, Rule

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY,
    title         TEXT    NOT NULL UNIQUE,
    content_hash  TEXT    NOT NULL,
    imported_at   TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    -- Исходный текст страницы как он пришёл из буфера обмена.
    -- Нужен, чтобы улучшение парсера не требовало заново копировать
    -- каждую страницу с форума: см. tools/reparse.py.
    raw_text      TEXT
);

CREATE TABLE IF NOT EXISTS rules (
    id              INTEGER PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    number          TEXT    NOT NULL,
    section_number  TEXT,
    section_title   TEXT,
    title           TEXT,
    text            TEXT    NOT NULL,
    sanction        TEXT,
    notes           TEXT,            -- JSON-массив строк
    content_hash    TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    UNIQUE (document_id, number)
);

CREATE INDEX IF NOT EXISTS idx_rules_number  ON rules (number);
CREATE INDEX IF NOT EXISTS idx_rules_section ON rules (document_id, section_number);

-- История редакций: пункт изменили на форуме — прошлый текст сохраняем.
CREATE TABLE IF NOT EXISTS rule_versions (
    id            INTEGER PRIMARY KEY,
    rule_id       INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    text          TEXT    NOT NULL,
    sanction      TEXT,
    notes         TEXT,
    content_hash  TEXT    NOT NULL,
    replaced_at   TEXT    NOT NULL
);

-- Метки: сюда попадут «процессуальные требования к задержанию»
-- для чек-листа режима адвоката (ТЗ 5.3).
CREATE TABLE IF NOT EXISTS rule_tags (
    rule_id  INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    tag      TEXT    NOT NULL,
    PRIMARY KEY (rule_id, tag)
);

-- Полнотекстовый поиск: страховка на случай, когда семантический поиск
-- промахивается, а пользователь спросил дословной формулировкой.
CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5 (
    number, text, sanction,
    content='rules', content_rowid='id',
    tokenize='unicode61'
);
"""

# Триггеры держат FTS в согласии с основной таблицей.
TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS rules_ai AFTER INSERT ON rules BEGIN
    INSERT INTO rules_fts (rowid, number, text, sanction)
    VALUES (new.id, new.number, new.text, new.sanction);
END;
CREATE TRIGGER IF NOT EXISTS rules_ad AFTER DELETE ON rules BEGIN
    INSERT INTO rules_fts (rules_fts, rowid, number, text, sanction)
    VALUES ('delete', old.id, old.number, old.text, old.sanction);
END;
CREATE TRIGGER IF NOT EXISTS rules_au AFTER UPDATE ON rules BEGIN
    INSERT INTO rules_fts (rules_fts, rowid, number, text, sanction)
    VALUES ('delete', old.id, old.number, old.text, old.sanction);
    INSERT INTO rules_fts (rowid, number, text, sanction)
    VALUES (new.id, new.number, new.text, new.sanction);
END;
"""


@dataclass
class ImportResult:
    document: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    warnings: list[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    @property
    def total(self) -> int:
        return self.added + self.updated + self.unchanged

    def summary(self) -> str:
        return (f"«{self.document}»: всего {self.total}, "
                f"новых {self.added}, обновлено {self.updated}, "
                f"без изменений {self.unchanged}")


def short_document_name(title: str) -> str:
    """Название документа без служебного хвоста.

    «Уголовный Кодекс (ред. от 28.07.2026)» -> «Уголовный Кодекс».
    Редакция, форумные метки и «Последнее редактирование» в ссылке
    только занимают место: в оверлее строка и так узкая.
    """
    cleaned = re.sub(r"\s*[(\[](?:ред\.|редакция|от)[^)\]]*[)\]]", "", title)
    cleaned = re.sub(r"\s*\((?:\d{1,2}\s+[А-Яа-я]{3,8}\s+\d{4})\)", "", cleaned)
    cleaned = re.sub(r"^(?:Важно|Рассмотрено|Закреплено)\s+", "", cleaned)
    cleaned = cleaned.replace("Последнее редактирование", "").strip(" .,")
    return cleaned or title


@dataclass
class StoredRule:
    id: int
    document: str
    number: str
    section_number: str
    section_title: str
    text: str
    sanction: str
    notes: list[str]

    def citation(self) -> str:
        """Ссылка для показа в ответе: «Процессуальный кодекс, ст. 1.3».

        Из названия убирается редакция: «(ред. от 02.09.2026)» занимает
        полстроки в оверлее и ничего не добавляет — пользователю нужен
        документ и номер, а не дата правки.
        """
        return f"{short_document_name(self.document)}, п. {self.number}"

    def gist(self, limit: int = 150) -> str:
        """Одна строка о том, про что статья — из её же текста.

        Берётся из базы, а не сочиняется моделью: пересказ, разошедшийся
        с нормой, хуже отсутствия пересказа, а генерация трёх расшифровок
        на CPU добавила бы к каждому ответу секунд двадцать.
        """
        text = re.sub(r"^\s*\((?:РЕГИОНАЛЬНО-)?[А-ЯЁ-]+\)\s*", "", self.text)
        text = re.sub(r"^\s*\[[А-ЯЁA-Z]{1,3}\]\s*", "", text)
        # Обрываем на первом перечислении: «а)», «б)» — это уже разбор
        # случаев, а для расшифровки хватает вводной части.
        text = re.split(r"\s+[а-яё]\)\s", text)[0].strip()
        head = re.split(r"(?<=[.;])\s+", text)[0].strip().rstrip(" .;:,")
        if len(head) > limit:
            # Многоточие ставится последним: если снимать пунктуацию
            # после него, rstrip срежет и его самого.
            head = head[:limit].rsplit(" ", 1)[0].rstrip(" .;:,") + "..."
        return head

    def full_text(self) -> str:
        parts = [f"[{self.citation()}] {self.text}"]
        if self.sanction:
            parts.append(f"Санкция: {self.sanction}")
        parts.extend(self.notes)
        return "\n".join(parts)


class LegalDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.executescript(TRIGGERS)
        self._migrate()
        self._conn.commit()
        log.info("База законов: %s", self.path)

    def _migrate(self) -> None:
        """Догоняет схему на базах, созданных прошлыми версиями."""
        columns = {row["name"] for row in
                   self._conn.execute("PRAGMA table_info(documents)")}
        if "raw_text" not in columns:
            self._conn.execute("ALTER TABLE documents ADD COLUMN raw_text TEXT")
            log.info("Схема обновлена: добавлено поле raw_text")
        for column in ("author", "started_at", "kind"):
            if column not in columns:
                self._conn.execute(
                    f"ALTER TABLE documents ADD COLUMN {column} TEXT")
                log.info("Схема обновлена: добавлено поле %s", column)
        if "uses" not in columns:
            # Сколько раз на документ реально сослались в ответах.
            # Нужно, чтобы поднимать в выдаче то, чем пользуются:
            # Уголовный кодекс спрашивают каждый день, а положение
            # о баллах лидеров — раз в месяц, и при прочих равных
            # первым должен идти первый.
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN uses INTEGER DEFAULT 0")
            log.info("Схема обновлена: добавлено поле uses")

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ импорт
    def import_document(self, parsed: ParsedDocument,
                        raw_text: str | None = None) -> ImportResult:
        """Добавляет или обновляет документ целиком."""
        now = datetime.now().isoformat(timespec="seconds")
        result = ImportResult(document=parsed.title,
                              warnings=list(parsed.warnings))

        with self._conn:
            title = self._resolve_title(parsed, result)
            cursor = self._conn.execute(
                "SELECT id FROM documents WHERE title = ?", (title,))
            row = cursor.fetchone()
            if row:
                document_id = row["id"]
                # Импорт не удаляет пункты, которых нет во входящем тексте:
                # так безопаснее. Но если их вдруг стало сильно меньше —
                # скорее всего скопирована только часть страницы, и об этом
                # надо сказать вслух, а не молча обновить четыре пункта
                # из восьмидесяти трёх.
                stored = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM rules WHERE document_id = ?",
                    (document_id,)).fetchone()["n"]
                if stored and len(parsed.rules) < stored * 0.6:
                    result.warnings.append(
                        f"В базе у «{title}» уже {stored} пунктов, "
                        f"а в этом импорте только {len(parsed.rules)}. "
                        "Похоже, скопирована часть страницы: остальные "
                        "пункты остались нетронутыми.")
                if raw_text is None:
                    self._conn.execute(
                        "UPDATE documents SET content_hash = ?, updated_at = ? "
                        "WHERE id = ?", (parsed.hash, now, document_id))
                else:
                    self._conn.execute(
                        "UPDATE documents SET content_hash = ?, updated_at = ?, "
                        "raw_text = ? WHERE id = ?",
                        (parsed.hash, now, raw_text, document_id))
            else:
                cursor = self._conn.execute(
                    "INSERT INTO documents (title, content_hash, imported_at, "
                    "updated_at, raw_text, author, started_at, kind) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (title, parsed.hash, now, now, raw_text,
                     parsed.author, parsed.started_at,
                     getattr(parsed, "kind", "rules")))
                document_id = cursor.lastrowid

            for rule in parsed.rules:
                self._upsert_rule(document_id, rule, now, result)

        log.info("Импорт — %s", result.summary())
        return result

    def _upsert_rule(self, document_id: int, rule: Rule, now: str,
                     result: ImportResult) -> None:
        notes = json.dumps(rule.notes, ensure_ascii=False)
        existing = self._conn.execute(
            "SELECT id, content_hash, text, sanction, notes FROM rules "
            "WHERE document_id = ? AND number = ?",
            (document_id, rule.number)).fetchone()

        if existing is None:
            self._conn.execute(
                "INSERT INTO rules (document_id, number, section_number, "
                "section_title, title, text, sanction, notes, content_hash, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (document_id, rule.number, rule.section_number,
                 rule.section_title, rule.title, rule.text, rule.sanction,
                 notes, rule.hash, now, now))
            result.added += 1
            return

        if existing["content_hash"] == rule.hash:
            result.unchanged += 1
            return

        # Текст изменился — прошлую редакцию сохраняем перед перезаписью.
        self._conn.execute(
            "INSERT INTO rule_versions (rule_id, text, sanction, notes, "
            "content_hash, replaced_at) VALUES (?,?,?,?,?,?)",
            (existing["id"], existing["text"], existing["sanction"],
             existing["notes"], existing["content_hash"], now))
        self._conn.execute(
            "UPDATE rules SET section_number = ?, section_title = ?, "
            "title = ?, text = ?, sanction = ?, notes = ?, content_hash = ?, "
            "updated_at = ? WHERE id = ?",
            (rule.section_number, rule.section_title, rule.title, rule.text,
             rule.sanction, notes, rule.hash, now, existing["id"]))
        result.updated += 1

    def _looks_like_same_thread(self, document_id: int, parsed) -> bool:
        """Тот ли это документ, или другая тема с тем же названием.

        Обычная правка на форуме меняет один-два пункта, остальные
        остаются как были. Другая тема — это когда совпадающие номера
        несут совершенно другой текст, да ещё и пунктов резко меньше.
        Именно так «Общие правила банд» на 54 пункта получили чужие
        1.1 и 1.2 из короткой темы-однофамильца.
        """
        stored = {row["number"]: row["text"] for row in self._conn.execute(
            "SELECT number, text FROM rules WHERE document_id = ?",
            (document_id,))}
        if not stored:
            return True

        overlap = [rule for rule in parsed.rules if rule.number in stored]
        if not overlap:
            # Пересечений нет вовсе — дописать новые пункты безопасно.
            return True

        identical = sum(1 for rule in overlap
                        if rule.text.strip() == stored[rule.number].strip())
        if identical:
            return True     # хоть что-то совпало дословно — та же тема

        # Ни одного совпадения при заметно меньшем объёме: чужая тема.
        return len(parsed.rules) >= len(stored) * 0.6

    def _resolve_title(self, parsed, result) -> str:
        """Разводит разные темы форума с одинаковым названием.

        «Правила проведения допросов» — это две разные темы: одна от
        22 Окт 2020 на два пункта, другая с двадцатью семью. Опознавать
        документ по одному названию нельзя: пункт 1.1 из одной темы
        затрёт пункт 1.1 из другой. Различаем по дате создания темы,
        дописывая её к названию.
        """
        row = self._conn.execute(
            "SELECT id, started_at FROM documents WHERE title = ?",
            (parsed.title,)).fetchone()
        if row is None:
            return parsed.title

        stored_started = (row["started_at"] or "").strip()
        incoming = (parsed.started_at or "").strip()

        if stored_started and incoming and stored_started == incoming:
            return parsed.title            # та же тема, обычное обновление

        if not incoming:
            # У входящего текста даты нет — развести не по чему.
            return parsed.title

        if not stored_started:
            # У документа в базе даты нет: он импортирован старой версией.
            # Сравниваем по содержимому — это и есть случай, на котором
            # десять документов уже получили затёртые пункты.
            if not self._looks_like_same_thread(row["id"], parsed):
                distinct = f"{parsed.title} ({incoming})"
                result.warnings.append(
                    f"Содержимое не совпало с документом «{parsed.title}», "
                    f"который уже есть в базе: похоже, это другая тема "
                    f"форума с тем же названием. Сохранено отдельно как "
                    f"«{distinct}», прежние пункты не тронуты.")
                result.document = distinct
                return distinct
            # Та же тема — просто дописываем ей дату на будущее.
            self._conn.execute(
                "UPDATE documents SET started_at = ?, author = ? WHERE id = ?",
                (incoming, parsed.author, row["id"]))
            return parsed.title

        distinct = f"{parsed.title} ({incoming})"
        result.warnings.append(
            f"На форуме есть другая тема с тем же названием "
            f"«{parsed.title}» (от {stored_started}). Чтобы её пункты не "
            f"затёрлись, эта сохранена как «{distinct}».")
        result.document = distinct
        return distinct

    # ------------------------------------------------------------------ чтение
    def _to_stored(self, row: sqlite3.Row) -> StoredRule:
        return StoredRule(
            id=row["id"],
            document=row["document"],
            number=row["number"],
            section_number=row["section_number"] or "",
            section_title=row["section_title"] or "",
            text=row["text"],
            sanction=row["sanction"] or "",
            notes=json.loads(row["notes"] or "[]"),
        )

    _SELECT = ("SELECT r.id, r.number, r.section_number, r.section_title, "
               "r.text, r.sanction, r.notes, d.title AS document "
               "FROM rules r JOIN documents d ON d.id = r.document_id ")

    def list_siblings(self, rule: StoredRule, limit: int = 8) -> list[StoredRule]:
        """Соседние пункты того же перечня.

        Перечень в законе бывает разнесён по отдельным пунктам: типы
        ордеров — это 4.2 «1) AR...», 4.3 «2) AS...», 4.4, 4.5. Каждый
        пункт сам по себе не перечень, и на вопрос «какие бывают
        ордера» в ответ попадали два типа из четырёх. Собираем их
        обратно по общему началу номера и нумерации в тексте.
        """
        base = rule.number.rsplit(".", 1)[0] if "." in rule.number else ""
        if not base:
            return []
        rows = self._conn.execute(
            self._SELECT +
            "WHERE d.title = ? AND r.number LIKE ? AND r.id != ? "
            "ORDER BY r.id LIMIT ?",
            (rule.document, f"{base}.%", rule.id, limit * 3)).fetchall()

        found = []
        for row in rows:
            # Берём только те, что сами начинаются с номера перечня:
            # соседний пункт про другое в перечень не входит.
            if re.match(r"^\s*\d{1,2}\)", row["text"] or ""):
                found.append(self._to_stored(row))
        return found[:limit]

    def rule_by_id(self, rule_id: int) -> StoredRule | None:
        row = self._conn.execute(
            self._SELECT + "WHERE r.id = ?", (rule_id,)).fetchone()
        return self._to_stored(row) if row else None

    def all_rules(self) -> list[StoredRule]:
        rows = self._conn.execute(self._SELECT + "ORDER BY d.id, r.id").fetchall()
        return [self._to_stored(row) for row in rows]

    def get_rule(self, number: str,
                 document: str | None = None) -> list[StoredRule]:
        """Точный поиск по номеру. Без указания документа вернёт все совпадения."""
        if document:
            rows = self._conn.execute(
                self._SELECT + "WHERE r.number = ? AND d.title = ?",
                (number, document)).fetchall()
        else:
            rows = self._conn.execute(
                self._SELECT + "WHERE r.number = ?", (number,)).fetchall()
        return [self._to_stored(row) for row in rows]

    @staticmethod
    def _stem(word: str) -> str:
        """Грубая основа слова для префиксного поиска.

        FTS5 с токенайзером unicode61 сравнивает слова дословно, без учёта
        морфологии: запрос «маска похищение заложник» не находит пункт,
        где написано «маски», «Похищения», «заложники». Полноценный
        стеммер сюда тащить не за чем — семантический поиск всё равно
        основной, а это страховка. Поэтому просто отрезаем окончание и
        ищем по префиксу.
        """
        return word[:max(3, round(len(word) * 0.6))]

    def search(self, query: str, limit: int = 5) -> list[StoredRule]:
        """Полнотекстовый поиск — страховка к семантическому."""
        # FTS5 воспринимает кавычки и операторы как синтаксис, поэтому
        # запрос пользователя разбираем на слова и склеиваем через OR.
        words = [w for w in "".join(
            c if c.isalnum() or c.isspace() else " " for c in query).split()
            if len(w) > 2]
        if not words:
            return []

        # Сначала пробуем по основам, и только если не вышло — дословно.
        expressions = [
            " OR ".join(f"{self._stem(w)}*" for w in words),
            " OR ".join(words),
        ]
        for expression in expressions:
            try:
                rows = self._conn.execute(
                    self._SELECT +
                    "JOIN rules_fts f ON f.rowid = r.id "
                    "WHERE rules_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expression, limit)).fetchall()
            except sqlite3.OperationalError as exc:
                log.warning("Полнотекстовый поиск не отработал (%s): %s",
                            expression, exc)
                continue
            if rows:
                return [self._to_stored(row) for row in rows]
        return []

    # --------------------------------------------------------------- сведения
    def raw_documents(self) -> list[tuple[str, str]]:
        """(заголовок, исходный текст) — для переразбора после правки парсера."""
        rows = self._conn.execute(
            "SELECT title, raw_text FROM documents "
            "WHERE raw_text IS NOT NULL AND raw_text != '' ORDER BY id"
        ).fetchall()
        return [(r["title"], r["raw_text"]) for r in rows]

    def documents_without_raw(self) -> list[str]:
        """Документы, импортированные до появления хранения исходника."""
        rows = self._conn.execute(
            "SELECT title FROM documents "
            "WHERE raw_text IS NULL OR raw_text = '' ORDER BY id").fetchall()
        return [r["title"] for r in rows]

    def documents(self) -> list[tuple[str, int, str]]:
        rows = self._conn.execute(
            "SELECT d.title, COUNT(r.id) AS n, d.updated_at "
            "FROM documents d LEFT JOIN rules r ON r.document_id = d.id "
            "GROUP BY d.id ORDER BY d.id").fetchall()
        return [(r["title"], r["n"], r["updated_at"]) for r in rows]

    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            return self._conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        return {
            "документов": count("documents"),
            "пунктов": count("rules"),
            "прошлых редакций": count("rule_versions"),
            "меток": count("rule_tags"),
        }

    # ----------------------------------------------------------------- метки
    def delete_document(self, title: str) -> int:
        """Убирает документ целиком. Возвращает число удалённых пунктов.

        Нужна для дублей: одна и та же тема попадает на форум дважды —
        полная и огрызок с датой в названии, — и огрызок только мешает
        поиску, отбирая место в выдаче у настоящего документа.

        Пункты и их векторы уходят каскадом по внешнему ключу.
        """
        row = self._conn.execute(
            "SELECT id FROM documents WHERE title = ?", (title,)).fetchone()
        if row is None:
            return 0
        count = self._conn.execute(
            "SELECT COUNT(*) AS n FROM rules WHERE document_id = ?",
            (row["id"],)).fetchone()["n"]
        with self._conn:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
        log.info("Документ удалён: «%s» (%d пунктов)", title, count)
        return count

    def note_usage(self, titles) -> None:
        """Отмечает, что на эти документы сослались в ответе.

        Считаются именно те документы, что попали в готовый ответ, а не
        те, что нашлись поиском: нашлось шесть, пригодился один, и
        поднимать в выдаче надо тот, который пригодился.
        """
        unique = {t for t in titles if t}
        if not unique:
            return
        try:
            with self._conn:
                self._conn.executemany(
                    "UPDATE documents SET uses = COALESCE(uses, 0) + 1 "
                    "WHERE title = ?", [(t,) for t in unique])
        except Exception:
            log.exception("Не удалось отметить использование документов")

    def usage_weights(self) -> dict[str, int]:
        """Название документа -> сколько раз на него ссылались."""
        return {row["title"]: (row["uses"] or 0) for row in self._conn.execute(
            "SELECT title, uses FROM documents")}

    def prune_rules(self, title: str, keep_numbers) -> int:
        """Удаляет пункты документа, которых больше нет в разборе.

        Нужна переразбору. Раньше он сносил все пункты документа и
        импортировал заново — от этого у каждого пункта менялся id, а
        вместе с ним каскадом отваливался вектор: смысловой поиск
        молча работал по 569 пунктам из 4723. Вдобавок путь удаления
        минует сохранение прошлой редакции, и таблица rule_versions
        оставалась пустой — страховке от перезаписи не на чем было
        сработать.

        Теперь документ сначала импортируется поверх (существующие
        пункты обновляются штатно, с записью прошлой редакции), а
        отсюда убирается только то, что новый разбор пунктом уже не
        считает.
        """
        keep = set(keep_numbers)
        with self._conn:
            rows = self._conn.execute(
                "SELECT r.id, r.number FROM rules r JOIN documents d "
                "ON d.id = r.document_id WHERE d.title = ?", (title,)).fetchall()
            stale = [row["id"] for row in rows if row["number"] not in keep]
            for rule_id in stale:
                self._conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        if stale:
            log.info("Удалено пунктов, исчезнувших при переразборе: %d", len(stale))
        return len(stale)

    def update_text(self, rule_id: int, text: str, sanction: str) -> None:
        """Правит текст и санкцию уже сохранённого пункта.

        Нужно для документов, исходник которых не сохранился: переразобрать
        их нельзя, а санкция у них так и лежит внутри текста после «|».
        Полнотекстовый индекс подхватит правку сам — на UPDATE висит
        триггер, переписывающий строку в rules_fts.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE rules SET text = ?, sanction = ? WHERE id = ?",
                (text, sanction, rule_id))

    def tag_rule(self, rule_id: int, tag: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO rule_tags (rule_id, tag) VALUES (?, ?)",
                (rule_id, tag))

    def rules_by_tag(self, tag: str) -> list[StoredRule]:
        rows = self._conn.execute(
            self._SELECT + "JOIN rule_tags t ON t.rule_id = r.id "
            "WHERE t.tag = ? ORDER BY r.id", (tag,)).fetchall()
        return [self._to_stored(row) for row in rows]
