"""
Точка входа: голосовой ассистент по законодательной базе Eclipse RP.

Запуск:  python main.py
Диагностика без UI:  python main.py --check
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config, ConfigError, setup_logging  # noqa: E402

log = logging.getLogger("main")


def run_checks(cfg: Config) -> int:
    """Проверяет окружение и печатает отчёт. Ничего не запускает."""
    from core.audio.mic import list_input_devices
    from core.llm.ollama_client import OllamaClient
    from core.hotkeys import parse_hotkey, HotkeyParseError

    print(f"Конфиг            : {cfg.path}")
    print(f"Профиль железа    : {cfg.profile_name} — "
          f"{cfg.profile.get('description', '')}")
    print(f"STT               : {cfg.profile['stt']}")
    print(f"LLM               : {cfg.profile['llm']}")

    print("\nХоткеи:")
    problems = 0
    for action, spec in cfg.hotkeys.items():
        try:
            parse_hotkey(spec)
            print(f"  OK    {action:16} {spec}")
        except HotkeyParseError as exc:
            problems += 1
            print(f"  ОШИБКА {action:16} {spec}  — {exc}")

    print("\nУстройства записи:")
    devices = list_input_devices()
    if not devices:
        problems += 1
        print("  не найдено ни одного входного устройства")
    for idx, name, channels in devices[:8]:
        print(f"  [{idx:>2}] {name} ({channels} кан.)")
    if len(devices) > 8:
        print(f"  ... и ещё {len(devices) - 8}")

    print("\nOllama:")
    client = OllamaClient.from_config(cfg)
    ready, reason = client.check_ready()
    print(f"  {'OK' if ready else 'НЕ ГОТОВА'}: {reason}")
    if not ready:
        problems += 1

    print("\nGPU:")
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        for line in out.stdout.strip().splitlines():
            print(f"  {line}")
        if not out.stdout.strip():
            print("  nvidia-smi ничего не вернула")
    except Exception as exc:
        print(f"  nvidia-smi недоступна: {exc}")

    print(f"\nИтог: {'всё в порядке' if problems == 0 else f'проблем: {problems}'}")
    return 0 if problems == 0 else 1


_MUTEX_NAME = "Local\\EclipseLaw.SingleInstance"


def _claim_single_instance():
    """Не даёт поднять второй экземпляр приложения.

    Окно у ассистента одно, а хоткеи — глобальные: два запущенных
    экземпляра слушают одни и те же сочетания, и каждое нажатие
    обрабатывается дважды. Оба пишут в одну базу и оба держат в памяти
    Whisper — на CPU-профиле это ещё и полгигабайта впустую.

    Возвращает хэндл мьютекса (его нужно удерживать всё время работы)
    или None, если экземпляр уже запущен. Мьютекс именованный и
    локальный для сессии: другой пользователь на той же машине сможет
    запустить своего ассистента.
    """
    import ctypes
    from ctypes import wintypes

    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL,
                                      wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE

    handle = kernel32.CreateMutexW(None, True, _MUTEX_NAME)
    if not handle:
        # Не смогли даже создать мьютекс — не повод не запускаться.
        log.warning("Мьютекс не создан (%s), защита от второго запуска "
                    "не работает", ctypes.get_last_error())
        return True
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return handle


def run_app(cfg: Config) -> int:
    from ui.overlay import OverlayWindow, build_tray_icon, create_app
    from core.orchestrator import Orchestrator

    instance = _claim_single_instance()
    if instance is None:
        print("Ассистент уже запущен. Второй экземпляр слушал бы те же "
              "хоткеи, и каждое нажатие срабатывало бы дважды.\n"
              "Найдите окно или значок в трее; если окно скрыто — "
              f"{cfg.hotkeys.get('toggle_overlay', 'ctrl+alt+h')}.")
        log.warning("Запуск отменён: экземпляр уже работает")
        return 1

    app = create_app()
    overlay = OverlayWindow(cfg)
    overlay.show()
    # Win32-стили навешиваем после show(): до этого у окна нет HWND.
    overlay.apply_native_flags()

    def shutdown(*_args) -> None:
        log.info("Завершение работы")
        app.quit()

    # Выход по хоткею приходит из потока диспетчера хоткеев. Раньше
    # здесь стоял QTimer.singleShot: таймер создаётся в вызывающем
    # потоке, а у того нет цикла событий Qt — он не срабатывал никогда,
    # и бот не закрывался ни хоткеем, ни из трея. Сигнал доставляется
    # в поток окна сам.
    overlay.sig_quit.connect(shutdown)
    request_quit = overlay.sig_quit.emit

    orchestrator = Orchestrator(cfg, overlay, on_quit=request_quit)
    orchestrator.start()

    def shutdown_all() -> None:
        """Выйти и погасить Ollama: «чтобы ничего не работало»."""
        log.info("Выход с остановкой Ollama")
        try:
            orchestrator.llm.stop_server()
        except Exception:
            log.exception("Ollama остановить не удалось")
        shutdown()

    tray = build_tray_icon(
        overlay,
        on_quit=shutdown,
        on_quit_all=shutdown_all,
        on_import=lambda: orchestrator.request_import(),
        on_help=lambda: overlay.sig_answer.emit(orchestrator.hotkey_help()),
    )

    signal.signal(signal.SIGINT, shutdown)
    app.aboutToQuit.connect(orchestrator.stop)

    log.info("Оверлей запущен. Скрыть: %s, выход: %s",
             cfg.hotkeys.get("toggle_overlay"), cfg.hotkeys.get("quit_app"))
    result = app.exec()
    if tray is not None:
        tray.hide()

    # Цикл событий закончился, но процесс мог остаться жить: пул
    # рабочих потоков не состоит из демонов, и интерпретатор ждёт
    # задачу, которая сейчас стоит на ответе модели. Для человека
    # это выглядит как «бот не закрывается». Базы и хуки уже закрыты
    # в orchestrator.stop() по aboutToQuit, поэтому выходим жёстко.
    log.info("Работа завершена")
    logging.shutdown()
    os._exit(result)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Голосовой ассистент по законам Eclipse RP")
    parser.add_argument("--check", action="store_true",
                        help="проверить окружение и выйти")
    parser.add_argument("--profile", metavar="ИМЯ",
                        help="переопределить active_profile из конфига")
    args = parser.parse_args()

    try:
        cfg = Config.load()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2

    if args.profile and args.profile not in cfg.get("profiles", {}):
        print(f"Профиль {args.profile!r} не найден в конфиге", file=sys.stderr)
        return 2

    # Профиль выбирается по свободной видеопамяти. Карта одна и делится
    # с игрой: когда GTA занимает пятнадцать гигабайт из шестнадцати,
    # обычный профиль формально загружается, но драйвер вытесняет
    # страницы модели — замер на живом экзамене дал 18 токенов в
    # секунду вместо 142. Снаружи это выглядит как «бот вдруг затупил».
    from core.gpu import choose_profile

    chosen, reason = choose_profile(cfg, args.profile)
    if chosen and chosen != cfg.get("active_profile"):
        cfg._raw["active_profile"] = chosen
        cfg._apply_gpu_pinning()

    setup_logging(cfg)
    if reason:
        log.info("Выбор профиля: %s", reason)
    log.info("Запуск, профиль железа: %s", cfg.profile_name)

    if args.check:
        return run_checks(cfg)
    return run_app(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
