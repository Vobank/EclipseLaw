"""
Сколько видеопамяти реально свободно — и какой профиль в неё влезет.

Зачем. Карта одна и делится с игрой. Когда GTA занимает пятнадцать
гигабайт из шестнадцати, обычный профиль формально загружается, но
драйвер начинает вытеснять страницы модели в системную память: замер
на живом экзамене показал 18 токенов в секунду вместо 142, то есть
ответ на двенадцать секунд вместо полутора.

Снаружи это выглядит как «бот вдруг затупил», и догадаться, что дело
в запущенной игре, невозможно. Поэтому профиль выбирается сам: перед
стартом смотрим свободную память и берём облегчённый, если места мало.
"""
from __future__ import annotations

import logging
import subprocess

import requests

log = logging.getLogger(__name__)


def free_vram_mb() -> int | None:
    """Свободная видеопамять в мегабайтах. None — узнать не удалось.

    Спрашиваем nvidia-smi, а не CUDA из Python: он есть всегда, когда
    есть драйвер, и не требует ни загрузки torch, ни инициализации
    контекста — а нам нужно решить до того, как что-то загружено.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        log.debug("nvidia-smi недоступен", exc_info=True)
        return None
    if result.returncode != 0:
        return None
    # На нескольких картах строк несколько — берём первую: профиль
    # всё равно пиннится к device_index 0.
    first = (result.stdout or "").strip().splitlines()
    try:
        return int(first[0].strip())
    except (IndexError, ValueError):
        return None


def ollama_held_mb(host: str) -> int:
    """Сколько видеопамяти держат наши же модели в Ollama.

    Ollama не выгружает модель после выхода приложения — она живёт ещё
    два часа, чтобы следующий запуск не ждал загрузки. Для nvidia-smi
    эта память занята, и без поправки бот при старте видит «свободно
    846 МБ» и сам понижает себе профиль, хотя эти гигабайты — его
    собственные и освободятся, как только он выгрузит лишнее.

    Замечено на живом запуске: профиль падал до облегчённого сразу
    после перезапуска, и человек видел только то, что бот «стал хуже».

    Модель эмбеддингов не считаем: её никто не выгружает, без неё
    встанет поиск по базе.
    """
    try:
        response = requests.get(f"{host.rstrip('/')}/api/ps", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception:
        log.debug("Не удалось спросить Ollama о загруженных моделях",
                  exc_info=True)
        return 0
    held = sum(int(m.get("size_vram") or 0) for m in models
               if not str(m.get("name", "")).startswith("bge"))
    return held // (1024 * 1024)


def choose_profile(cfg, requested: str | None = None) -> tuple[str, str]:
    """Какой профиль запускать. Возвращает (имя, причина для лога).

    Профили выстроены лестницей в конфиге, от полного к облегчённому;
    берём первый, которому хватает свободной памяти. Явно названный
    профиль не трогаем: человек знает, что делает.
    """
    if requested:
        return requested, f"профиль задан вручную: {requested}"

    configured = cfg.get("active_profile", "")
    ladder = cfg.get("hardware.memory_ladder") or []
    profiles = cfg.get("profiles") or {}
    ladder = [step for step in ladder if step.get("profile") in profiles]
    if not ladder:
        return configured, ""

    free = free_vram_mb()
    if free is None:
        return configured, ""
    # Память, которую держат наши же модели с прошлого запуска,
    # вернётся при выгрузке — она нам доступна.
    held = ollama_held_mb(cfg.get("llm.host", "http://127.0.0.1:11434"))
    free += held

    for step in ladder:
        if free >= step.get("needs_mb", 0):
            name = step["profile"]
            note = f" (из них {held} МБ держит Ollama)" if held else ""
            if name == configured:
                return name, (
                    f"видеопамяти свободно {free} МБ{note} — хватает")
            return name, (
                f"видеопамяти свободно {free} МБ{note}, для «{configured}» "
                f"нужно {ladder[0].get('needs_mb')} — беру «{name}»")
    return configured, ""
