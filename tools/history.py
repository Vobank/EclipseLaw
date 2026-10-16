"""
Просмотр истории запросов (ТЗ 5.6).

История пишется автоматически при каждом ответе. Здесь её можно
посмотреть, найти нужный разговор и оценить, как часто модель
выдумывает статьи — это прямой показатель того, пора ли менять её
на более крупную.

Запуск:
    python tools/history.py                     последние 20 записей
    python tools/history.py --limit 50
    python tools/history.py --search "задержание"
    python tools/history.py --session <id>      одна ситуация целиком
    python tools/history.py --invented          только с выдуманными статьями
    python tools/history.py --wrong             помеченные вручную промахи
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.history import HistoryLog  # noqa: E402


def argument(flag: str, default=None):
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else default


def show(entry, full: bool = False) -> None:
    head = f"[{entry.timestamp}] {entry.mode} / {entry.source}"
    if entry.latency_ms:
        head += f"  ({entry.latency_ms / 1000:.1f} с)"
    print(head)
    print(f"  сессия {entry.session_id}")
    print(f"  ВОПРОС: {entry.question}")
    answer = entry.answer if full else entry.answer[:220]
    print("  ОТВЕТ : " + answer.replace("\n", "\n          "))
    if entry.articles:
        print(f"  статьи: {'; '.join(entry.articles[:4])}")
    if entry.invented:
        print(f"  ВЫДУМАНО: {', '.join(entry.invented)}")
    print()


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)
    history = HistoryLog(cfg.resolve_path("storage.history_db"))

    try:
        stats = history.stats()
        print(f"Всего: {stats['записей']} записей в {stats['сессий']} сессиях, "
              f"с выдуманными статьями: {stats['с выдуманными статьями']}\n")

        session_id = argument("--session")
        if session_id:
            entries = history.session(session_id)
            print(f"Сессия {session_id}: {len(entries)} записей\n")
            for entry in entries:
                show(entry, full=True)
            return 0

        if "--wrong" in sys.argv:
            # Ответы, помеченные хоткеем «ответ неверный». Уверенно
            # неправильный ответ опаснее честного «не нашлось», а
            # ловится только так: сам по себе он выглядит нормально.
            entries = history.wrong_answers(
                limit=int(argument("--limit", 50)))
            print(f"Помечено неверными: {len(entries)}\n")
            for entry in entries:
                show(entry, full=True)
            if not entries:
                print("Пока ничего не помечено. В боте это "
                      f"{cfg.hotkeys.get('mark_wrong')}.")
            return 0

        query = argument("--search")
        if query:
            entries = history.search(query, limit=int(argument("--limit", 20)))
            print(f"По запросу «{query}» найдено: {len(entries)}\n")
            for entry in entries:
                show(entry)
            return 0

        entries = history.recent(limit=int(argument("--limit", 20)))
        if "--invented" in sys.argv:
            entries = [e for e in entries if e.invented]
            print(f"Записей с выдуманными статьями: {len(entries)}\n")
        for entry in entries:
            show(entry)
        return 0
    finally:
        history.close()


if __name__ == "__main__":
    raise SystemExit(main())
