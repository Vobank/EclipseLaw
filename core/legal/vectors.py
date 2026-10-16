"""
Векторный индекс базы законов (ТЗ 5.3, этап 4).

Почему не ChromaDB. ТЗ допускает «Chroma или аналог», и аналог здесь
выигрывает: пунктов около шести тысяч, каждый вектор — 1024 числа.
Полный перебор косинусной близости по такой матрице занимает единицы
миллисекунд, а Chroma притащила бы отдельную СУБД со своим деревом
зависимостей ради индекса, который нам не нужен.

Эмбеддинги считает Ollama, а не sentence-transformers: сервер и так
запущен ради основной модели, а sentence-transformers потянул бы torch
на два с половиной гигабайта.

Векторы лежат в той же SQLite рядом с пунктами, поэтому индекс не может
разойтись с базой: удалили пункт — каскад удалил и его вектор.
"""
from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass

import numpy as np
import requests

from core.legal.chunks import SPLIT_ABOVE, split_rule

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS rule_vectors (
    rule_id  INTEGER PRIMARY KEY REFERENCES rules(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    -- content_hash пункта на момент расчёта: изменился текст — вектор
    -- устарел и будет пересчитан.
    source   TEXT NOT NULL,
    vector   BLOB NOT NULL
);

-- Фрагменты длинных статей. Статья 3.4 Процессуального кодекса на
-- пять тысяч знаков одним вектором не ищется: он размазан по десятку
-- тем сразу, и норма про вызов прокурора в рацию в выдачу не попадала.
CREATE TABLE IF NOT EXISTS rule_chunks (
    id       INTEGER PRIMARY KEY,
    rule_id  INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    ord      INTEGER NOT NULL,
    text     TEXT NOT NULL,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    source   TEXT NOT NULL,     -- content_hash статьи на момент расчёта
    vector   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_rule ON rule_chunks (rule_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text, content='rule_chunks', content_rowid='id', tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON rule_chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON rule_chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
"""


class EmbeddingError(RuntimeError):
    """Сервер эмбеддингов недоступен или вернул мусор."""


@dataclass
class IndexStats:
    total_rules: int = 0
    embedded: int = 0
    added: int = 0
    refreshed: int = 0
    seconds: float = 0.0

    @property
    def missing(self) -> int:
        return self.total_rules - self.embedded

    def summary(self) -> str:
        return (f"пунктов {self.total_rules}, в индексе {self.embedded}, "
                f"добавлено {self.added}, пересчитано {self.refreshed}, "
                f"за {self.seconds:.1f} с")


class EmbeddingClient:
    """Тонкий клиент к /api/embed Ollama."""

    def __init__(self, host: str, model: str, timeout: int = 600,
                 batch_size: int = 32, num_gpu: int | None = None):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        # Сколько слоёв модели эмбеддингов держать на карте. Ноль —
        # считать на процессоре: когда рядом идёт игра на максималках,
        # эти полгигабайта решают, влезет основная модель в память
        # или драйвер начнёт вытеснять её страницы в системную. Запрос
        # короткий, на процессоре он считается за доли секунды.
        self.num_gpu = num_gpu
        self._session = requests.Session()

    def available(self) -> tuple[bool, str]:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Ollama не отвечает: {exc}"
        names = {m["name"] for m in response.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return False, (f"Модель эмбеддингов {self.model} не скачана. "
                           f"Выполните: ollama pull {self.model}")
        return True, f"Модель эмбеддингов {self.model} готова"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Возвращает матрицу (len(texts), dim), нормированную по строкам."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            try:
                payload_json = {"model": self.model, "input": chunk}
                if self.num_gpu is not None:
                    payload_json["options"] = {"num_gpu": self.num_gpu}
                response = self._session.post(
                    f"{self.host}/api/embed", json=payload_json,
                    timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                raise EmbeddingError(
                    f"Запрос эмбеддингов не прошёл: {exc}") from exc

            batch = payload.get("embeddings")
            if not batch or len(batch) != len(chunk):
                raise EmbeddingError(
                    f"Ollama вернула {len(batch or [])} векторов вместо "
                    f"{len(chunk)}")
            vectors.extend(batch)

        matrix = np.asarray(vectors, dtype=np.float32)
        # Нормируем сразу: дальше косинус превращается в скалярное
        # произведение, и поиск становится одним матричным умножением.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def _pack(vector: np.ndarray) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector.tolist())


def _unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


class VectorIndex:
    """Хранит векторы пунктов и ищет по смыслу."""

    def __init__(self, db, client: EmbeddingClient):
        self.db = db
        self.client = client
        self.db._conn.executescript(SCHEMA)
        self.db._conn.commit()
        self._ids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._chunk_ids: np.ndarray | None = None
        self._chunk_rules: np.ndarray | None = None
        self._chunk_matrix: np.ndarray | None = None

    # ------------------------------------------------------------- построение
    def build(self, on_progress=None) -> IndexStats:
        """Считает векторы для новых и изменившихся пунктов."""
        started = time.monotonic()
        conn = self.db._conn

        rows = conn.execute("""
            SELECT r.id, r.number, r.text, r.sanction, r.content_hash,
                   r.section_title, d.title AS document
            FROM rules r
            JOIN documents d ON d.id = r.document_id
            LEFT JOIN rule_vectors v ON v.rule_id = r.id
            WHERE v.rule_id IS NULL
               OR v.source != r.content_hash
               OR v.model != ?
            ORDER BY r.id
        """, (self.client.model,)).fetchall()

        stats = IndexStats(
            total_rules=conn.execute(
                "SELECT COUNT(*) AS n FROM rules").fetchone()["n"])

        if rows:
            existing = {row["rule_id"] for row in conn.execute(
                "SELECT rule_id FROM rule_vectors")}
            batch = self.client.batch_size
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                matrix = self.client.embed([self._passage(r) for r in chunk])
                with conn:
                    for row, vector in zip(chunk, matrix):
                        conn.execute(
                            "INSERT OR REPLACE INTO rule_vectors "
                            "(rule_id, model, dim, source, vector) "
                            "VALUES (?,?,?,?,?)",
                            (row["id"], self.client.model, len(vector),
                             row["content_hash"], _pack(vector)))
                        if row["id"] in existing:
                            stats.refreshed += 1
                        else:
                            stats.added += 1
                if on_progress is not None:
                    on_progress(min(start + batch, len(rows)), len(rows))

        self._build_chunks(on_progress)

        stats.embedded = conn.execute(
            "SELECT COUNT(*) AS n FROM rule_vectors").fetchone()["n"]
        stats.seconds = time.monotonic() - started
        self._ids = self._matrix = None      # матрицу перечитаем заново
        log.info("Векторный индекс: %s", stats.summary())
        return stats

    def _build_chunks(self, on_progress=None) -> int:
        """Считает векторы фрагментов длинных статей.

        Фрагменты живут отдельно от самих статей: статья остаётся
        единицей ответа, а фрагмент — единицей поиска. Поэтому здесь
        же и содержимое, и вектор: разъехаться им негде.
        """
        conn = self.db._conn
        rows = conn.execute("""
            SELECT r.id, r.number, r.text, r.sanction, r.content_hash,
                   r.section_title, d.title AS document
            FROM rules r
            JOIN documents d ON d.id = r.document_id
            WHERE LENGTH(r.text) > ?
              AND NOT EXISTS (
                  SELECT 1 FROM rule_chunks c
                  WHERE c.rule_id = r.id AND c.source = r.content_hash
                    AND c.model = ?)
            ORDER BY r.id
        """, (SPLIT_ABOVE, self.client.model)).fetchall()

        if not rows:
            self._chunk_ids = None
            return 0

        log.info("Фрагменты: пересчитываю %d длинных статей", len(rows))
        written = 0
        for position, row in enumerate(rows):
            pieces = split_rule(row["text"])
            if len(pieces) < 2:
                continue
            # Название документа и номер статьи — в каждый фрагмент:
            # «Примечание 5: оповещение повторять каждые 15 минут» без
            # них не отличить от примечания к уставу другой фракции.
            head = self._head(row)
            matrix = self.client.embed([f"{head}\n{p}" for p in pieces])
            with conn:
                conn.execute("DELETE FROM rule_chunks WHERE rule_id = ?",
                             (row["id"],))
                for ord_, (piece, vector) in enumerate(zip(pieces, matrix)):
                    conn.execute(
                        "INSERT INTO rule_chunks "
                        "(rule_id, ord, text, model, dim, source, vector) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (row["id"], ord_, piece, self.client.model,
                         len(vector), row["content_hash"], _pack(vector)))
                    written += 1
            if on_progress is not None:
                on_progress(position + 1, len(rows))

        self._chunk_ids = None      # матрицу фрагментов перечитаем
        log.info("Фрагменты: записано %d", written)
        return written

    def _head(self, row) -> str:
        """Шапка фрагмента: откуда он взят."""
        parts = [row["document"]]
        if row["section_title"]:
            parts.append(row["section_title"])
        parts.append(f"п. {row['number']}")
        return "\n".join(parts)

    def _passage(self, row) -> str:
        """Текст, который уходит в эмбеддинг.

        Кладём рядом название документа и раздела: без них пункт
        «Запрещено открывать огонь без предупреждения» одинаково похож
        на вопрос про рейды и про войну за территорию, а с ними —
        различим.
        """
        parts = [row["document"]]
        if row["section_title"]:
            parts.append(row["section_title"])
        parts.append(f"п. {row['number']}")
        parts.append(row["text"])
        if row["sanction"]:
            parts.append(f"Санкция: {row['sanction']}")
        return "\n".join(parts)

    # ----------------------------------------------------------------- поиск
    def _load(self) -> bool:
        if self._matrix is not None:
            return len(self._matrix) > 0
        rows = self.db._conn.execute(
            "SELECT rule_id, dim, vector FROM rule_vectors WHERE model = ? "
            "ORDER BY rule_id", (self.client.model,)).fetchall()
        if not rows:
            self._ids = np.zeros(0, dtype=np.int64)
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return False
        self._ids = np.array([r["rule_id"] for r in rows], dtype=np.int64)
        self._matrix = np.vstack(
            [_unpack(r["vector"], r["dim"]) for r in rows])
        log.info("Векторный индекс загружен: %d × %d",
                 *self._matrix.shape)
        return True

    def _load_chunks(self) -> bool:
        if self._chunk_matrix is not None:
            return len(self._chunk_matrix) > 0
        rows = self.db._conn.execute(
            "SELECT id, rule_id, dim, vector FROM rule_chunks WHERE model = ? "
            "ORDER BY id", (self.client.model,)).fetchall()
        if not rows:
            self._chunk_ids = np.zeros(0, dtype=np.int64)
            self._chunk_rules = np.zeros(0, dtype=np.int64)
            self._chunk_matrix = np.zeros((0, 0), dtype=np.float32)
            return False
        self._chunk_ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self._chunk_rules = np.array([r["rule_id"] for r in rows],
                                     dtype=np.int64)
        self._chunk_matrix = np.vstack(
            [_unpack(r["vector"], r["dim"]) for r in rows])
        log.info("Индекс фрагментов загружен: %d × %d",
                 *self._chunk_matrix.shape)
        return True

    @staticmethod
    def _top(matrix: np.ndarray, vector: np.ndarray, limit: int):
        scores = matrix @ vector
        top = np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
        return top[np.argsort(-scores[top])], scores

    def search(self, query: str, limit: int = 8) -> list[tuple[int, float]]:
        """(id пункта, близость) — по убыванию близости."""
        return [(rule_id, score)
                for rule_id, score, _ in self.search_detailed(query, limit)]

    def search_detailed(self, query: str,
                        limit: int = 8) -> list[tuple[int, float, int | None]]:
        """То же, плюс id фрагмента, которым статья и нашлась.

        Фрагмент нужен дальше: в контекст модели идёт та часть статьи,
        о которой спросили, а не её первые полторы тысячи знаков.
        Длинная статья иначе доходила до модели обрезанной ровно по
        тому месту, где начинался ответ.
        """
        if not self._load():
            return []
        vector = self.client.embed([query])[0]

        best: dict[int, tuple[float, int | None]] = {}
        order, scores = self._top(self._matrix, vector, limit)
        for i in order:
            best[int(self._ids[i])] = (float(scores[i]), None)

        # Фрагменты добавляются СВЕРХ статейного топа, а не конкурируют
        # с ним за те же места. Иначе длинные статьи, у которых
        # фрагменты есть, вытесняли короткие нормы, у которых их нет:
        # на «обозву мента» статья 17.3 Уголовного кодекса выпадала из
        # выдачи целиком, уступив место кускам закона о LSPD.
        extra: dict[int, tuple[float, int | None]] = {}
        if self._load_chunks():
            order, scores = self._top(self._chunk_matrix, vector, limit)
            for i in order:
                rule_id = int(self._chunk_rules[i])
                score = float(scores[i])
                if rule_id in best:
                    # Статья и так найдена: фрагмент лишь уточняет,
                    # какую её часть показывать.
                    if score > best[rule_id][0]:
                        best[rule_id] = (score, int(self._chunk_ids[i]))
                elif rule_id not in extra or score > extra[rule_id][0]:
                    extra[rule_id] = (score, int(self._chunk_ids[i]))

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        ranked += sorted(extra.items(), key=lambda kv: -kv[1][0])[:limit]
        return [(rule_id, score, chunk_id)
                for rule_id, (score, chunk_id) in ranked]

    def chunk_text(self, chunk_id: int, budget: int = 0) -> str:
        """Текст фрагмента, при запасе места — вместе с соседями.

        Соседи нужны потому, что норма редко умещается в один абзац:
        «прокурор не явился в течение 30 минут» и «оповещение
        повторять каждые 15 минут» — два соседних примечания одной
        статьи, и ответ без второго неполон. Расширяемся по очереди
        вперёд-назад, пока хватает лимита на статью.
        """
        row = self.db._conn.execute(
            "SELECT rule_id, ord, text FROM rule_chunks WHERE id = ?",
            (chunk_id,)).fetchone()
        if row is None:
            return ""
        if budget <= len(row["text"]):
            return row["text"]

        siblings = self.db._conn.execute(
            "SELECT ord, text FROM rule_chunks WHERE rule_id = ? ORDER BY ord",
            (row["rule_id"],)).fetchall()
        by_ord = {r["ord"]: r["text"] for r in siblings}

        taken = {row["ord"]}
        used = len(row["text"])
        low = high = row["ord"]
        while True:
            grew = False
            for candidate in (high + 1, low - 1):
                text = by_ord.get(candidate)
                if text is None or candidate in taken:
                    continue
                if used + len(text) + 1 > budget:
                    continue
                taken.add(candidate)
                used += len(text) + 1
                low, high = min(low, candidate), max(high, candidate)
                grew = True
            if not grew:
                break
        return " ".join(by_ord[o] for o in sorted(taken))

    def search_chunks_text(self, query: str,
                           limit: int = 8) -> list[tuple[int, int]]:
        """Полнотекстовый поиск по фрагментам: (id пункта, id фрагмента).

        Смысловой поиск берёт общий настрой вопроса, а дословное «в
        рацию департамента» ищется именно текстом — BM25 по короткому
        фрагменту находит его там, где по статье целиком редкое слово
        тонуло в пяти тысячах знаков.
        """
        words = [w for w in "".join(
            c if c.isalnum() or c.isspace() else " " for c in query).split()
            if len(w) > 2]
        if not words:
            return []
        expression = " OR ".join(f"{w}*" for w in words)
        try:
            rows = self.db._conn.execute(
                "SELECT c.rule_id, c.id FROM chunks_fts f "
                "JOIN rule_chunks c ON c.id = f.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (expression, limit)).fetchall()
        except Exception as exc:
            log.warning("Поиск по фрагментам не отработал: %s", exc)
            return []
        return [(r["rule_id"], r["id"]) for r in rows]

    def stats(self) -> dict[str, int]:
        conn = self.db._conn
        return {
            "пунктов": conn.execute(
                "SELECT COUNT(*) AS n FROM rules").fetchone()["n"],
            "векторов": conn.execute(
                "SELECT COUNT(*) AS n FROM rule_vectors").fetchone()["n"],
            "фрагментов": conn.execute(
                "SELECT COUNT(*) AS n FROM rule_chunks").fetchone()["n"],
        }
