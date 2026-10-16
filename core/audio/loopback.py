"""
Захват звука отдельных процессов — WASAPI Process Loopback (ТЗ 5.2, этап 5).

Требование ТЗ жёсткое: писать разрешено ТОЛЬКО звук GTA5.exe и
Discord.exe, системный звук целиком — никогда. Обычный loopback такого
не умеет, он снимает всё, что играет на устройстве. Поэтому используется
Process Loopback Capture (Windows 10 2004+): клиент активируется
с AUDIOCLIENT_ACTIVATION_PARAMS и пишет только дерево процессов
указанного PID.

Почему так много ctypes. ActivateAudioInterfaceAsync принимает
обработчик завершения в виде COM-объекта и не принимает NULL, поэтому
COM-интерфейс IActivateAudioInterfaceCompletionHandler приходится
собирать руками: таблица методов, счётчик ссылок, всё вручную.
Готовой обёртки в Python для этого нет.

Запись идёт только по удержанию клавиши: поток открывается на start()
и закрывается на stop(), фонового прослушивания нет.
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import POINTER, byref, c_void_p, wintypes

import numpy as np

log = logging.getLogger(__name__)

ole32 = ctypes.WinDLL("ole32")
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
mmdevapi = ctypes.WinDLL("Mmdevapi")

S_OK = 0
COINIT_MULTITHREADED = 0x0

# Псевдоустройство, через которое активируется захват по процессу.
VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000

AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0

WAVE_FORMAT_PCM = 1
VT_BLOB = 65

AUDCLNT_BUFFERFLAGS_SILENT = 0x2
REFTIMES_PER_SEC = 10_000_000


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]

    def __init__(self, text: str = ""):
        super().__init__()
        if text:
            ole32.CLSIDFromString(ctypes.c_wchar_p(text), byref(self))


IID_IAudioClient = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
IID_IAudioCaptureClient = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
IID_IActivateAudioInterfaceCompletionHandler = GUID(
    "{41D949AB-9862-444A-80F6-C261334DA5EB}")
IID_IUnknown = GUID("{00000000-0000-0000-C000-000000000046}")
IID_IAgileObject = GUID("{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}")


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [("wFormatTag", wintypes.WORD),
                ("nChannels", wintypes.WORD),
                ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD),
                ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD),
                ("cbSize", wintypes.WORD)]


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [("TargetProcessId", wintypes.DWORD),
                ("ProcessLoopbackMode", ctypes.c_int)]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [("ActivationType", ctypes.c_int),
                ("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS)]


class BLOB(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("pBlobData", c_void_p)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", wintypes.WORD), ("r1", wintypes.WORD),
                ("r2", wintypes.WORD), ("r3", wintypes.WORD),
                ("blob", BLOB), ("padding", ctypes.c_byte * 8)]


# --- ручная реализация COM-обработчика ---------------------------------
# Windows вызывает ActivateCompleted из своего потока, поэтому объект
# обязан быть настоящим COM-объектом: таблица методов и счётчик ссылок.
LPVOID = c_void_p
HRESULT = ctypes.c_long

_QueryInterface = ctypes.WINFUNCTYPE(HRESULT, LPVOID, POINTER(GUID), POINTER(LPVOID))
_AddRef = ctypes.WINFUNCTYPE(ctypes.c_ulong, LPVOID)
_Release = ctypes.WINFUNCTYPE(ctypes.c_ulong, LPVOID)
_ActivateCompleted = ctypes.WINFUNCTYPE(HRESULT, LPVOID, LPVOID)


class _HandlerVtbl(ctypes.Structure):
    _fields_ = [("QueryInterface", _QueryInterface),
                ("AddRef", _AddRef),
                ("Release", _Release),
                ("ActivateCompleted", _ActivateCompleted)]


class _HandlerObject(ctypes.Structure):
    _fields_ = [("lpVtbl", POINTER(_HandlerVtbl))]


class CompletionHandler:
    """COM-объект, который Windows дёргает по завершении активации."""

    def __init__(self):
        self.done = threading.Event()
        self.hresult = S_OK

        # Ссылки на замыкания держим в себе: если их соберёт сборщик
        # мусора, Windows позовёт освобождённую память.
        self._qi = _QueryInterface(self._query_interface)
        self._addref = _AddRef(lambda this: 2)
        self._release = _Release(lambda this: 1)
        self._completed = _ActivateCompleted(self._activate_completed)

        self._vtbl = _HandlerVtbl(self._qi, self._addref, self._release,
                                  self._completed)
        self._object = _HandlerObject(ctypes.pointer(self._vtbl))
        self.pointer = ctypes.cast(ctypes.pointer(self._object), LPVOID)

    def _query_interface(self, this, riid, ppv):
        wanted = bytes(riid.contents)
        for known in (IID_IUnknown, IID_IActivateAudioInterfaceCompletionHandler,
                      IID_IAgileObject):
            if wanted == bytes(known):
                ppv[0] = this
                return S_OK
        ppv[0] = None
        return -2147467262      # E_NOINTERFACE

    def _activate_completed(self, this, operation):
        self.done.set()
        return S_OK


def _call(interface: c_void_p, index: int, restype, *args):
    """Вызов метода COM-интерфейса по его номеру в таблице."""
    vtable = ctypes.cast(interface, POINTER(POINTER(c_void_p))).contents
    func = ctypes.cast(
        vtable[index],
        ctypes.WINFUNCTYPE(restype, c_void_p, *[type(a) if not isinstance(a, int)
                                                else ctypes.c_int
                                                for a in args]))
    return func(interface, *args)


class LoopbackError(RuntimeError):
    """Захват по процессу недоступен."""


class ProcessLoopbackStream:
    """Пишет звук одного дерева процессов."""

    # Методы IAudioClient и IAudioCaptureClient по номерам в таблице:
    # первые три места всегда занимает IUnknown.
    _INITIALIZE, _GET_BUFFER_SIZE = 3, 4
    _START, _STOP, _RESET, _SET_EVENT, _GET_SERVICE = 10, 11, 12, 13, 14
    _CAP_GET_BUFFER, _CAP_RELEASE, _CAP_NEXT_PACKET = 3, 4, 5

    def __init__(self, pid: int, sample_rate: int = 48000, channels: int = 2):
        self.pid = pid
        self.sample_rate = sample_rate
        self.channels = channels
        self._client: c_void_p | None = None
        self._capture: c_void_p | None = None
        self._event = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def _activate(self) -> c_void_p:
        params = AUDIOCLIENT_ACTIVATION_PARAMS()
        params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
        params.ProcessLoopbackParams.TargetProcessId = self.pid
        params.ProcessLoopbackParams.ProcessLoopbackMode = (
            PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE)

        prop = PROPVARIANT()
        prop.vt = VT_BLOB
        prop.blob.cbSize = ctypes.sizeof(params)
        prop.blob.pBlobData = ctypes.cast(byref(params), c_void_p)

        handler = CompletionHandler()
        operation = c_void_p()

        # Типы аргументов обязательны: без них ctypes на x64 передаёт
        # указатели как 32-битные числа, и функция возвращает мусор
        # вместо осмысленного кода ошибки.
        mmdevapi.ActivateAudioInterfaceAsync.restype = HRESULT
        mmdevapi.ActivateAudioInterfaceAsync.argtypes = [
            ctypes.c_wchar_p, POINTER(GUID), POINTER(PROPVARIANT),
            c_void_p, POINTER(c_void_p)]
        hresult = mmdevapi.ActivateAudioInterfaceAsync(
            ctypes.c_wchar_p(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK),
            byref(IID_IAudioClient), byref(prop),
            handler.pointer, byref(operation))
        if hresult != S_OK:
            raise LoopbackError(
                f"ActivateAudioInterfaceAsync отклонён (0x{hresult & 0xFFFFFFFF:08X}). "
                "Нужна Windows 10 2004 или новее.")

        if not handler.done.wait(timeout=3.0):
            raise LoopbackError("Активация захвата не завершилась за 3 секунды")

        activated = c_void_p()
        activate_result = HRESULT()
        # GetActivateResult — метод №3 у IActivateAudioInterfaceAsyncOperation.
        vtable = ctypes.cast(operation, POINTER(POINTER(c_void_p))).contents
        get_result = ctypes.cast(
            vtable[3],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(HRESULT),
                               POINTER(c_void_p)))
        hresult = get_result(operation, byref(activate_result), byref(activated))
        if hresult != S_OK or activate_result.value != S_OK:
            raise LoopbackError(
                f"Процесс {self.pid} не отдал звук "
                f"(0x{activate_result.value & 0xFFFFFFFF:08X}). "
                "Возможно, он сейчас ничего не проигрывает.")
        return activated

    def start(self) -> None:
        ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        self._client = self._activate()

        fmt = WAVEFORMATEX(
            wFormatTag=WAVE_FORMAT_PCM, nChannels=self.channels,
            nSamplesPerSec=self.sample_rate,
            nAvgBytesPerSec=self.sample_rate * self.channels * 2,
            nBlockAlign=self.channels * 2, wBitsPerSample=16, cbSize=0)

        initialize = ctypes.cast(
            ctypes.cast(self._client, POINTER(POINTER(c_void_p))).contents[
                self._INITIALIZE],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, ctypes.c_int, ctypes.c_uint,
                               ctypes.c_longlong, ctypes.c_longlong,
                               POINTER(WAVEFORMATEX), c_void_p))
        hresult = initialize(
            self._client, AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            2 * REFTIMES_PER_SEC, 0, byref(fmt), None)
        if hresult != S_OK:
            raise LoopbackError(
                f"Initialize не удался (0x{hresult & 0xFFFFFFFF:08X})")

        self._event = kernel32.CreateEventW(None, False, False, None)
        set_event = ctypes.cast(
            ctypes.cast(self._client, POINTER(POINTER(c_void_p))).contents[
                self._SET_EVENT],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, wintypes.HANDLE))
        set_event(self._client, self._event)

        get_service = ctypes.cast(
            ctypes.cast(self._client, POINTER(POINTER(c_void_p))).contents[
                self._GET_SERVICE],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(GUID),
                               POINTER(c_void_p)))
        capture = c_void_p()
        if get_service(self._client, byref(IID_IAudioCaptureClient),
                       byref(capture)) != S_OK:
            raise LoopbackError("Не удалось получить IAudioCaptureClient")
        self._capture = capture

        start = ctypes.cast(
            ctypes.cast(self._client, POINTER(POINTER(c_void_p))).contents[
                self._START],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p))
        if start(self._client) != S_OK:
            raise LoopbackError("Start не удался")

        self._running.set()
        self._thread = threading.Thread(
            target=self._pump, name=f"loopback-{self.pid}", daemon=True)
        self._thread.start()
        log.info("Захват процесса %d начат", self.pid)

    def _pump(self) -> None:
        vtable = ctypes.cast(self._capture, POINTER(POINTER(c_void_p))).contents
        get_buffer = ctypes.cast(
            vtable[self._CAP_GET_BUFFER],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(POINTER(ctypes.c_byte)),
                               POINTER(ctypes.c_uint32), POINTER(wintypes.DWORD),
                               POINTER(ctypes.c_uint64), POINTER(ctypes.c_uint64)))
        release = ctypes.cast(
            vtable[self._CAP_RELEASE],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, ctypes.c_uint32))
        next_packet = ctypes.cast(
            vtable[self._CAP_NEXT_PACKET],
            ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(ctypes.c_uint32)))

        while self._running.is_set():
            kernel32.WaitForSingleObject(self._event, 200)
            packet = ctypes.c_uint32()
            while next_packet(self._capture, byref(packet)) == S_OK and packet.value:
                data = POINTER(ctypes.c_byte)()
                frames = ctypes.c_uint32()
                flags = wintypes.DWORD()
                if get_buffer(self._capture, byref(data), byref(frames),
                              byref(flags), None, None) != S_OK:
                    break
                if frames.value:
                    if flags.value & AUDCLNT_BUFFERFLAGS_SILENT:
                        block = np.zeros(frames.value * self.channels,
                                         dtype=np.int16)
                    else:
                        size = frames.value * self.channels
                        block = np.ctypeslib.as_array(
                            ctypes.cast(data, POINTER(ctypes.c_int16)),
                            (size,)).copy()
                    with self._lock:
                        self._chunks.append(block)
                release(self._capture, frames.value)

    def stop(self) -> np.ndarray:
        """Останавливает запись и отдаёт моно float32."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._client is not None:
            stop = ctypes.cast(
                ctypes.cast(self._client, POINTER(POINTER(c_void_p))).contents[
                    self._STOP],
                ctypes.WINFUNCTYPE(HRESULT, c_void_p))
            stop(self._client)
        if self._event:
            kernel32.CloseHandle(self._event)
            self._event = None

        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        samples = np.concatenate(chunks).astype(np.float32) / 32768.0
        if self.channels > 1:
            samples = samples.reshape(-1, self.channels).mean(axis=1)
        return samples
