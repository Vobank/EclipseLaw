"""
Чтение буфера обмена через Win32 API.

Берём напрямую из user32, а не через Qt: импорт запускается в рабочем
потоке, а QClipboard разрешено трогать только из потока UI. Win32-вызовы
таких ограничений не имеют.
"""
from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

CF_UNICODETEXT = 13

user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.OpenClipboard.restype = wintypes.BOOL
user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = wintypes.LPVOID
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
user32.EmptyClipboard.restype = wintypes.BOOL
user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL

GMEM_MOVEABLE = 0x0002


class ClipboardError(RuntimeError):
    """Буфер занят другим процессом или в нём нет текста."""


def get_text(retries: int = 5, delay: float = 0.05) -> str:
    """Возвращает текст из буфера обмена.

    Буфер — разделяемый ресурс: браузер может держать его открытым в
    момент нашего обращения. Поэтому несколько коротких попыток вместо
    немедленной ошибки.
    """
    if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
        raise ClipboardError(
            "В буфере обмена нет текста. Скопируйте страницу правил "
            "целиком: Ctrl+A, затем Ctrl+C.")

    for attempt in range(retries):
        if user32.OpenClipboard(None):
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    raise ClipboardError("Буфер обмена вернул пустые данные.")
                pointer = kernel32.GlobalLock(handle)
                if not pointer:
                    raise ClipboardError("Не удалось прочитать буфер обмена.")
                try:
                    return ctypes.c_wchar_p(pointer).value or ""
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        if attempt < retries - 1:
            time.sleep(delay)

    raise ClipboardError(
        "Буфер обмена занят другим приложением. Закройте меню копирования "
        "в браузере и повторите.")


def set_text(text: str, retries: int = 5, delay: float = 0.05) -> None:
    """Кладёт текст в буфер обмена.

    Нужно, чтобы ответ можно было унести из оверлея: окно намеренно
    не получает фокус, и привычное выделение мышью с Ctrl+C в нём
    не работает — нажатие уходит в игру. Без этого готовый документ
    пришлось бы перепечатывать руками.

    Буфер — общий ресурс, его на доли секунды занимают другие
    программы, поэтому повторяем несколько раз, как и при чтении.
    """
    if text is None:
        raise ClipboardError("нечего копировать")

    data = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(data)

    for attempt in range(retries):
        if not user32.OpenClipboard(None):
            time.sleep(delay)
            continue
        try:
            user32.EmptyClipboard()
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
            if not handle:
                raise ClipboardError("не удалось выделить память под текст")
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                raise ClipboardError("не удалось заблокировать память")
            try:
                ctypes.memmove(pointer, data, size)
            finally:
                kernel32.GlobalUnlock(handle)
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                raise ClipboardError("Windows отказалась принять текст")
            return
        finally:
            user32.CloseClipboard()

    raise ClipboardError(
        f"буфер обмена занят другой программой ({retries} попыток)")
