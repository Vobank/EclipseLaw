"""
Тонкая обёртка над Win32 API для окна-оверлея.

Здесь ровно три вещи, и все они — документированные публичные вызовы
user32.dll. Никаких хуков в чужие процессы, никакой работы с ядром:
согласно ТЗ (раздел 2) процесс обязан оставаться обычным и быть
видимым в Диспетчере задач.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)

# --- расширенные стили окна ------------------------------------------------
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080   # нет записи в Alt-Tab и на панели задач
WS_EX_NOACTIVATE = 0x08000000   # окно не забирает фокус у игры
WS_EX_TRANSPARENT = 0x00000020  # мышь проходит насквозь (click-through)
WS_EX_LAYERED = 0x00080000

# --- SetWindowDisplayAffinity ----------------------------------------------
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
# WDA_EXCLUDEFROMCAPTURE: окно видно глазами на мониторе, но его пиксели
# не попадают в Windows Graphics Capture (OBS Display Capture, Discord
# Screen Share). Требует Windows 10 2004 (build 19041) или новее.
WDA_EXCLUDEFROMCAPTURE = 0x00000011

# SetWindowLongPtrW есть только в 64-битном user32; в 32-битном — SetWindowLongW.
if hasattr(user32, "SetWindowLongPtrW"):
    _set_window_long = user32.SetWindowLongPtrW
    _get_window_long = user32.GetWindowLongPtrW
    _set_window_long.restype = ctypes.c_longlong
    _set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
    _get_window_long.restype = ctypes.c_longlong
    _get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
else:  # pragma: no cover - 32-битный интерпретатор
    _set_window_long = user32.SetWindowLongW
    _get_window_long = user32.GetWindowLongW
    _set_window_long.restype = ctypes.c_long
    _set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    _get_window_long.restype = ctypes.c_long
    _get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]

user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]


def _hwnd(handle) -> wintypes.HWND:
    """Qt отдаёт winId() как int/sip.voidptr — приводим к HWND."""
    return wintypes.HWND(int(handle))


def apply_overlay_styles(handle, *, tool_window: bool = True,
                         no_activate: bool = True,
                         click_through: bool = False) -> None:
    """Навешивает на окно набор стилей, стандартный для игровых HUD-оверлеев.

    Ровно то же самое делают оверлеи Discord, Steam и MSI Afterburner:
    окно не мелькает в Alt-Tab и не отбирает фокус у полноэкранной игры.
    """
    hwnd = _hwnd(handle)
    style = _get_window_long(hwnd, GWL_EXSTYLE)
    new_style = style
    if tool_window:
        new_style |= WS_EX_TOOLWINDOW
    if no_activate:
        new_style |= WS_EX_NOACTIVATE
    if click_through:
        new_style |= WS_EX_TRANSPARENT
    else:
        new_style &= ~WS_EX_TRANSPARENT

    if new_style == style:
        return
    if not _set_window_long(hwnd, GWL_EXSTYLE, new_style):
        err = ctypes.get_last_error()
        # SetWindowLongPtr возвращает 0 и при успехе, если прежнее значение
        # было нулём, — поэтому ошибкой считаем только ненулевой GetLastError.
        if err:
            log.warning("Не удалось задать стили окна: код %s", err)
            return
    log.debug("Стили оверлея применены: 0x%08X -> 0x%08X", style, new_style)


def set_excluded_from_capture(handle, enabled: bool = True) -> bool:
    """Исключает окно из захвата экрана (ТЗ 5.1, этап 10).

    Возвращает True, если система приняла вызов. На сборках старше
    Windows 10 2004 вернёт False — это не фатально, оверлей продолжит
    работать, просто попадёт в Display Capture.
    """
    affinity = WDA_EXCLUDEFROMCAPTURE if enabled else WDA_NONE
    ok = bool(user32.SetWindowDisplayAffinity(_hwnd(handle), affinity))
    if not ok:
        log.warning(
            "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) отклонён, код %s. "
            "Нужна Windows 10 2004+; оверлей останется видимым при захвате экрана.",
            ctypes.get_last_error(),
        )
    else:
        log.info("Оверлей исключён из захвата экрана: %s", enabled)
    return ok
