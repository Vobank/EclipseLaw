"""
Проверка окна оверлея без ручного запуска: создаёт окно, навешивает
Win32-стили, убеждается, что система их приняла, и закрывается.

Запуск: python tools/smoke_overlay.py [секунды_показа]
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402

from core.config import Config, setup_logging  # noqa: E402
from ui import win32  # noqa: E402
from ui.overlay import OverlayWindow, create_app  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

    cfg = Config.load()
    setup_logging(cfg)

    app = create_app()
    overlay = OverlayWindow(cfg)
    overlay.show()
    overlay.apply_native_flags()

    hwnd = wintypes.HWND(int(overlay.winId()))
    failures = 0

    # 1. Расширенные стили окна.
    style = win32._get_window_long(hwnd, win32.GWL_EXSTYLE)
    print(f"WS_EX (0x{style & 0xFFFFFFFF:08X}):")
    for name, bit in [("WS_EX_TOOLWINDOW", win32.WS_EX_TOOLWINDOW),
                      ("WS_EX_NOACTIVATE", win32.WS_EX_NOACTIVATE),
                      ("WS_EX_LAYERED", win32.WS_EX_LAYERED),
                      ("WS_EX_TRANSPARENT", win32.WS_EX_TRANSPARENT)]:
        print(f"  {'есть ' if style & bit else 'нет  '} {name}")

    for required in ("WS_EX_TOOLWINDOW", "WS_EX_NOACTIVATE"):
        if not style & getattr(win32, required):
            print(f"  ПРОВАЛ: {required} не установлен")
            failures += 1

    # 2. Исключение из захвата экрана — читаем обратно у системы.
    affinity = wintypes.DWORD()
    if user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)):
        expected = (win32.WDA_EXCLUDEFROMCAPTURE
                    if cfg.get("overlay.exclude_from_capture", True)
                    else win32.WDA_NONE)
        ok = affinity.value == expected
        print(f"\nDisplayAffinity = 0x{affinity.value:02X} "
              f"(ожидалось 0x{expected:02X}) — {'OK' if ok else 'ПРОВАЛ'}")
        if not ok:
            failures += 1
    else:
        print("\nПРОВАЛ: GetWindowDisplayAffinity не отработала")
        failures += 1

    # 3. Проверка сигналов из «чужого» потока делается штатным emit.
    overlay.sig_mode.emit("Обычный")
    overlay.sig_status.emit("ready", "Проверка отрисовки")
    overlay.sig_answer.emit(
        "Тестовый ответ.\n\nСтатья 12.3 — проверка переноса длинной строки, "
        "которая должна аккуратно завернуться по ширине панели и не вылезти "
        "за её границы."
    )
    for who, text in [("Я", "тестовый вопрос"), ("Ассистент", "тестовый ответ")]:
        overlay.sig_log.emit(who, text)

    print(f"\nОкно показано на {seconds:.0f} с...")
    QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec()

    print(f"\nИтог: {'все проверки пройдены' if failures == 0 else f'провалов: {failures}'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
