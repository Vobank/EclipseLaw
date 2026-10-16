"""
Глобальные хоткеи через WH_KEYBOARD_LL (ТЗ 5.2).

Почему не RegisterHotKey: он сообщает только о нажатии и ничего не знает
об отпускании клавиши. Для push-to-talk нужны обе границы — «зажал» и
«отпустил», — поэтому ставим низкоуровневый хук клавиатуры.

Там же висит WH_MOUSE_LL — ради колеса мыши. WM_MOUSEWHEEL Windows
шлёт окну с фокусом клавиатуры, а фокус всегда у игры: без хука лента
ответов не прокручивалась вовсе, сколько ни крути над оверлеем. Хук
берёт только колесо и только когда курсор над самим окном — всё
остальное уходит в игру нетронутым.

Хук живёт в отдельном потоке со своим циклом сообщений. Колбэк обязан
отрабатывать за единицы миллисекунд: Windows снимает низкоуровневые хуки,
которые не уложились в LowLevelHooksTimeout (по умолчанию 300 мс).
Поэтому внутри колбэка мы только сопоставляем клавишу и кладём событие
в очередь, а вся обработка идёт в потоке-потребителе.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import threading
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_MOUSEWHEEL = 0x020A
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0104, 0x0105
WM_QUIT = 0x0012
HC_ACTION = 0

VK_MODIFIERS = {
    "ctrl": (0xA2, 0xA3),    # L/R Control
    "shift": (0xA0, 0xA1),   # L/R Shift
    "alt": (0xA4, 0xA5),     # L/R Menu
    "win": (0x5B, 0x5C),     # L/R Win
}

# Именованные клавиши сверх букв, цифр и F-ряда.
VK_NAMED = {
    "esc": 0x1B, "escape": 0x1B, "space": 0x20, "tab": 0x09,
    "enter": 0x0D, "return": 0x0D, "backspace": 0x08, "delete": 0x2E,
    "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    "printscreen": 0x2C, "pause": 0x13,
    "grave": 0xC0, "minus": 0xBD, "equals": 0xBB,
    "lbracket": 0xDB, "rbracket": 0xDD, "backslash": 0xDC,
    "semicolon": 0xBA, "quote": 0xDE,
    "comma": 0xBC, "period": 0xBE, "slash": 0xBF,
    "num0": 0x60, "num1": 0x61, "num2": 0x62, "num3": 0x63, "num4": 0x64,
    "num5": 0x65, "num6": 0x66, "num7": 0x67, "num8": 0x68, "num9": 0x69,
    "nummultiply": 0x6A, "numadd": 0x6B, "numsubtract": 0x6D,
    "numdecimal": 0x6E, "numdivide": 0x6F,
}


class HotkeyParseError(ValueError):
    """Строка хоткея из конфига не разобралась."""


def parse_hotkey(spec: str) -> tuple[frozenset[str], int]:
    """'ctrl+alt+q' -> (frozenset({'ctrl','alt'}), 0x51)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise HotkeyParseError(f"пустая комбинация: {spec!r}")

    mods: set[str] = set()
    key: str | None = None
    for part in parts:
        if part in VK_MODIFIERS:
            mods.add(part)
        elif key is None:
            key = part
        else:
            raise HotkeyParseError(
                f"в комбинации {spec!r} больше одной основной клавиши: "
                f"{key!r} и {part!r}"
            )
    if key is None:
        raise HotkeyParseError(f"в комбинации {spec!r} нет основной клавиши")

    if len(key) == 1 and key.isascii() and (key.isalpha() or key.isdigit()):
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key in VK_NAMED:
        vk = VK_NAMED[key]
    else:
        raise HotkeyParseError(f"неизвестная клавиша {key!r} в {spec!r}")

    return frozenset(mods), vk


def _modifier_down(name: str) -> bool:
    """Старший бит GetAsyncKeyState = клавиша зажата прямо сейчас."""
    return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in VK_MODIFIERS[name])


@dataclass
class Binding:
    action: str
    spec: str
    mods: frozenset[str]
    vk: int
    is_ptt: bool = False
    suppress: bool = True   # не пропускать комбинацию дальше, в игру
    held: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class HotkeyEvent:
    action: str
    kind: str        # "pressed" | "released"
    # Набранный символ — для режима ввода. Окно ассистента намеренно
    # не берёт фокус, поэтому обычное поле ввода в нём не работает:
    # буквы уходят в игру. Их приходится брать здесь же, в хуке.
    text: str = ""


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.CallNextHookEx.restype = LRESULT
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]


# Клавиши управления строкой ввода. Остальное превращается в символ.
VK_BACK, VK_ESCAPE, VK_RETURN = 0x08, 0x1B, 0x0D
VK_UP, VK_DOWN = 0x26, 0x28

user32.GetKeyboardLayout.restype = wintypes.HKL
user32.ToUnicodeEx.argtypes = [
    wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_wchar), ctypes.c_int, wintypes.UINT, wintypes.HKL]


def _layout_of_foreground() -> int:
    """Раскладка активного окна, а не нашего потока.

    Номер постановления набирают латиницей, а название иска — по-русски,
    и раскладку человек переключает в игре. Свою раскладку поток хука
    не переключает никогда, поэтому спрашиваем ту, что сейчас на экране.
    """
    window = user32.GetForegroundWindow()
    thread = user32.GetWindowThreadProcessId(window, None) if window else 0
    return user32.GetKeyboardLayout(thread)


def _char_from_key(vk: int, scan: int) -> str:
    """Символ, который дала бы эта клавиша в текущей раскладке."""
    state = (ctypes.c_ubyte * 256)()
    # Состояние клавиатуры у потока хука своё и всегда пустое, поэтому
    # Shift и CapsLock выставляем вручную — иначе всё вводится строчным.
    if user32.GetAsyncKeyState(0x10) & 0x8000:      # Shift
        state[0x10] = 0x80
    if user32.GetKeyState(0x14) & 1:                # CapsLock
        state[0x14] = 1
    buffer = (ctypes.c_wchar * 8)()
    written = user32.ToUnicodeEx(vk, scan, state, buffer, len(buffer), 0,
                                 _layout_of_foreground())
    if written <= 0:
        return ""
    return "".join(buffer[:written])


class HotkeyManager:
    """Слушает клавиатуру глобально и отдаёт события подписчику.

    Колбэк `on_event` вызывается из потока-диспетчера, а не из UI-потока:
    на стороне Qt его нужно заворачивать в сигнал с QueuedConnection.
    """

    def __init__(self, on_event: Callable[[HotkeyEvent], None]):
        self._on_event = on_event
        self._bindings: dict[int, list[Binding]] = {}
        self._queue: "queue.Queue[HotkeyEvent | None]" = queue.Queue()
        self._hook_thread: threading.Thread | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._hook_thread_id: int = 0
        self._hook = None
        # Ссылку на HOOKPROC держим в атрибуте: если её соберёт GC,
        # Windows позовёт освобождённую память и процесс упадёт.
        self._proc = HOOKPROC(self._low_level_proc)
        self._mouse_proc = HOOKPROC(self._low_level_mouse)
        self._mouse_hook = None
        # Прямоугольник на экране, над которым колесо принадлежит нам,
        # в физических пикселях. Кортеж целиком заменяется одним
        # присваиванием — читать его из потока хука можно без блокировки.
        self._wheel_zone: tuple[int, int, int, int] | None = None
        self._capture = False
        self._running = threading.Event()

    # ---------------------------------------------------------------- привязки
    def bind(self, action: str, spec: str, *, is_ptt: bool = False,
             suppress: bool = True) -> None:
        mods, vk = parse_hotkey(spec)
        binding = Binding(action=action, spec=spec, mods=mods, vk=vk,
                          is_ptt=is_ptt, suppress=suppress)
        self._bindings.setdefault(vk, []).append(binding)
        # Более специфичные комбинации проверяем первыми: ctrl+alt+q должен
        # выигрывать у ctrl+q, иначе сработает менее строгая привязка.
        self._bindings[vk].sort(key=lambda b: len(b.mods), reverse=True)
        log.debug("Привязка: %s -> %s (ptt=%s)", spec, action, is_ptt)

    def bind_all(self, hotkeys: dict[str, str], ptt_actions: set[str]) -> list[str]:
        """Привязывает всё из конфига. Возвращает список ошибок, не падая."""
        errors: list[str] = []
        for action, spec in hotkeys.items():
            try:
                self.bind(action, spec, is_ptt=action in ptt_actions)
            except HotkeyParseError as exc:
                errors.append(str(exc))
                log.error("Хоткей %s пропущен: %s", action, exc)
        return errors

    def capture_text(self, enabled: bool) -> None:
        """Включает перехват букв и цифр для строки поиска.

        Пока включён, обычные клавиши в игру не уходят: иначе поиск
        по делам заодно писал бы в игровой чат. Хоткеи с Ctrl+Alt
        продолжают работать — они проверяются раньше.
        """
        self._capture = enabled

    def set_wheel_zone(self, zone: tuple[int, int, int, int] | None) -> None:
        """Область экрана (left, top, right, bottom), где колесо наше.

        None — колесо целиком отдаётся игре. Зону задаёт окно, когда
        переезжает или прячется: сам хук ничего об окнах не знает и
        трогать Qt из своего потока не имеет права.
        """
        self._wheel_zone = zone

    # -------------------------------------------------------------------- хук
    def _low_level_proc(self, ncode: int, wparam: int, lparam: int) -> int:
        if ncode != HC_ACTION:
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        try:
            suppress = self._handle_key(wparam, lparam)
        except Exception:  # хук не имеет права падать
            log.exception("Сбой в обработчике клавиатуры")
            suppress = False

        if suppress:
            return 1  # ненулевой результат = событие дальше не идёт
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    @staticmethod
    def _modifiers_match(binding: "Binding") -> bool:
        """Совпадают ли зажатые модификаторы с привязкой.

        Для одиночной клавиши важно не только то, что нужные
        модификаторы зажаты, но и что лишних нет: иначе Alt+Tab
        считался бы нажатием Tab, запись собеседника стартовала бы
        при каждом переключении окон, а сам Alt+Tab бот бы съедал.
        """
        for name in VK_MODIFIERS:
            if _modifier_down(name) != (name in binding.mods):
                return False
        return True

    def _handle_text(self, vk: int, scan: int) -> bool:
        """Клавиша в режиме ввода. True — не пускать её дальше, в игру."""
        if vk in (0xA2, 0xA3, 0xA0, 0xA1, 0xA4, 0xA5, 0x5B, 0x5C):
            return False        # сами модификаторы пропускаем
        # Ctrl или Alt зажат — это хоткей, а не набор текста.
        if _modifier_down("ctrl") or _modifier_down("alt"):
            return False

        if vk == VK_BACK:
            self._queue.put(HotkeyEvent("search_back", "pressed"))
        elif vk == VK_ESCAPE:
            self._queue.put(HotkeyEvent("search_close", "pressed"))
        elif vk == VK_RETURN:
            self._queue.put(HotkeyEvent("search_open", "pressed"))
        elif vk in (VK_UP, VK_DOWN):
            self._queue.put(HotkeyEvent(
                "search_move", "pressed", "-" if vk == VK_UP else "+"))
        else:
            char = _char_from_key(vk, scan)
            if not char or char < " ":
                return True     # служебная клавиша: глотаем молча
            self._queue.put(HotkeyEvent("search_edit", "pressed", char))
        return True

    def _handle_key(self, wparam: int, lparam: int) -> bool:
        kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        vk = kb.vkCode

        if self._capture and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            # Привязки проверяем всё равно: Ctrl+Alt+что-то обязан
            # работать и при открытом поиске, иначе из него не выйти
            # ничем, кроме Esc.
            if not self._bindings.get(vk) or not any(
                    all(_modifier_down(m) for m in b.mods)
                    for b in self._bindings[vk]):
                if self._handle_text(vk, kb.scanCode):
                    return True

        candidates = self._bindings.get(vk)
        if not candidates:
            return False

        is_down = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
        is_up = wparam in (WM_KEYUP, WM_SYSKEYUP)
        if not (is_down or is_up):
            return False

        if is_down:
            for binding in candidates:
                if not self._modifiers_match(binding):
                    continue
                if binding.held:
                    # Автоповтор от удержания — для PTT это не новое
                    # нажатие: гасим, но клавишу в игру всё равно не пускаем.
                    return binding.suppress
                binding.held = True
                self._queue.put(HotkeyEvent(binding.action, "pressed"))
                return binding.suppress
            return False

        # Отпускание: закрываем любую привязку, которая была зажата.
        # Модификаторы здесь не проверяем — пользователь мог отпустить
        # Ctrl раньше основной клавиши, и PTT обязан корректно закрыться.
        suppress = False
        for binding in candidates:
            if binding.held:
                binding.held = False
                if binding.is_ptt:
                    self._queue.put(HotkeyEvent(binding.action, "released"))
                suppress = suppress or binding.suppress
        return suppress

    def _low_level_mouse(self, ncode: int, wparam: int, lparam: int) -> int:
        if ncode != HC_ACTION or wparam != WM_MOUSEWHEEL:
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        try:
            zone = self._wheel_zone
            if zone is not None:
                info = ctypes.cast(
                    lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                left, top, right, bottom = zone
                if left <= info.pt.x < right and top <= info.pt.y < bottom:
                    # mouseData: старшее слово со знаком — шаги колеса.
                    delta = ctypes.c_short(info.mouseData >> 16).value
                    action = "scroll_up" if delta > 0 else "scroll_down"
                    self._queue.put(HotkeyEvent(action, "pressed"))
                    return 1
        except Exception:
            log.exception("Сбой в обработчике колеса")
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _hook_loop(self) -> None:
        self._hook_thread_id = kernel32.GetCurrentThreadId()
        self._hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
        if not self._hook:
            log.error("SetWindowsHookExW не удался, код %s", ctypes.get_last_error())
            self._running.clear()
            return

        self._mouse_hook = user32.SetWindowsHookExW(
            WH_MOUSE_LL, self._mouse_proc, None, 0)
        if not self._mouse_hook:
            # Не повод падать: без колеса лента листается хоткеями.
            log.warning("Хук мыши не установлен, код %s",
                        ctypes.get_last_error())

        log.info("Клавиатурный хук установлен")
        msg = wintypes.MSG()
        # GetMessageW блокируется до WM_QUIT — он и качает очередь сообщений,
        # без которой низкоуровневый хук не получает события.
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
        if self._mouse_hook:
            user32.UnhookWindowsHookEx(self._mouse_hook)
            self._mouse_hook = None
        log.info("Клавиатурный хук снят")

    def _dispatch_loop(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            try:
                self._on_event(event)
            except Exception:
                log.exception("Ошибка обработчика хоткея %s", event.action)

    # ------------------------------------------------------------------ жизнь
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="hotkey-dispatch", daemon=True)
        self._dispatch_thread.start()
        self._hook_thread = threading.Thread(
            target=self._hook_loop, name="hotkey-hook", daemon=True)
        self._hook_thread.start()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._hook_thread_id:
            user32.PostThreadMessageW(self._hook_thread_id, WM_QUIT, 0, 0)
        self._queue.put(None)
        for thread in (self._hook_thread, self._dispatch_thread):
            if thread is not None:
                thread.join(timeout=2.0)
