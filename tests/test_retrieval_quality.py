"""
Качество поиска на настоящей базе: вопрос -> какая статья обязана найтись.

Это регрессия на живых данных, поэтому она требует базы и Ollama —
без них тесты пропускаются, а не падают. Набор собран из вопросов,
которые однажды отвечались неправильно: каждая строка тут появилась
после реальной жалобы, а не из головы.

Проверяется попадание в первую тройку, а не на первое место: дальше
статью всё равно читает модель, и второе место её не смущает.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.legal.db import LegalDB  # noqa: E402
from core.legal.retriever import Retriever  # noqa: E402
from core.legal.vectors import EmbeddingClient, VectorIndex  # noqa: E402

# вопрос -> (документ по началу названия, номер статьи)
CASES = [
    # «что там за халатность будет»: статья находилась пятой, выше неё
    # стояли «халатно относиться к амуниции» из уставов.
    ("что там за халатность будет", "Уголовный", "15.6"),
    # «обозву мента»: ни «мент», ни «обозвать» в базе не встречаются.
    ("что будет если я обозву мента", "Уголовный", "17.3"),
    # Искажения распознавания — как их записал Whisper в истории.
    ("Слушай, есть ли что-то там за хладность?", "Уголовный", "15.6"),
    ("что предполагается за то, что я обозвалка мента", "Уголовный", "17.3"),
    # Нападение на сотрудника: модель однажды выдала 1.2 вместо 17.1.
    ("что будет за нападение на полицейского", "Уголовный", "17.1"),
    ("превышение должностных полномочий сотрудником", "Уголовный", "15.1"),
    # Поиск по номеру с названием кодекса.
    ("что говорит статья 1.3 Уголовного кодекса", "Уголовный", "1.3"),
    # Вопросы о порядке и правах. Здесь нужен не состав, а процедура:
    # на «меня задержали без объяснения причин» первым выходил
    # внутренний устав LSPD с запретом патрулировать в одиночку.
    ("меня задержали без объяснения причин что делать",
     "Процессуальный", "2.2"),
    # «Обыскать машину» уводило в Закон «Об охоте» — там тоже осмотр
    # и изъятие. Разрешает обыск ордер, а не кодекс.
    ("имеет ли право сотрудник обыскать мою машину",
     "Закон \"Об ордерах", "4.4"),
    # «1 (10)» — право апелляционного обжалования. Кассация («8 (3)»)
    # сюда тоже подходит, но это следующая инстанция, а спрашивают
    # обычно про первое обжалование.
    ("как обжаловать приговор", "Судебный", "1 (10)"),
]


@pytest.fixture(scope="module")
def retriever():
    cfg = Config.load()
    path = cfg.resolve_path("storage.legal_db")
    if not Path(path).exists():
        pytest.skip("базы законов нет")
    db = LegalDB(path)
    client = EmbeddingClient(cfg.get("llm.host", "http://127.0.0.1:11434"),
                             cfg.get("retrieval.embedding_model", "bge-m3"))
    available, reason = client.available()
    if not available:
        pytest.skip(f"эмбеддинги недоступны: {reason}")
    return Retriever.from_config(cfg, db, VectorIndex(db, client))


@pytest.mark.parametrize("question,document,number", CASES)
def test_expected_article_is_found(retriever, question, document, number):
    hits = retriever.retrieve(question, limit=3)
    if retriever.last_degraded:
        # Эмбеддинги отвалились посреди прогона — проверять качество
        # поиска нечем. Это не провал набора, а отсутствие условий:
        # так один занятый Ollama однажды «уронил» два теста подряд.
        pytest.skip(f"смысловой поиск недоступен: {retriever.last_degraded}")
    found = [(h.rule.document, h.rule.number) for h in hits]
    assert any(doc.startswith(document) and num == number
               for doc, num in found), (
        f"ожидалась {document} п.{number}, найдено: {found}")


def test_index_covers_every_rule(retriever):
    """Векторы обязаны быть у всех пунктов.

    Переразбор базы однажды пересоздал пункты с новыми id, векторы
    отвалились каскадом, и смысловой поиск молча работал по 569
    пунктам из 4723 — без единой ошибки в логах.
    """
    import sqlite3
    path = Config.load().resolve_path("storage.legal_db")
    conn = sqlite3.connect(str(path))
    rules = conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
    vectors = conn.execute(
        "SELECT COUNT(*) FROM rules r JOIN rule_vectors v ON v.rule_id = r.id"
    ).fetchone()[0]
    conn.close()
    assert vectors == rules, (
        f"без вектора осталось пунктов: {rules - vectors} — "
        f"запустите python tools/build_index.py")
