"""
Захват речи собеседника (ТЗ 5.2, п.2).

Пишутся строго процессы из белого списка — GTA5.exe и Discord.exe.
Системный звук целиком не захватывается никогда: это прямое требование
ТЗ, и обеспечивается оно не проверкой на нашей стороне, а самим
механизмом — Process Loopback активируется по конкретному PID.

Процессов может быть несколько (игра и Discord одновременно), поэтому
каждый пишется своим потоком, а на выходе они микшируются в один.
"""
from __future__ import annotations

import logging

import numpy as np

from core.audio.loopback import LoopbackError, ProcessLoopbackStream

log = logging.getLogger(__name__)


def find_pids(names: list[str]) -> list[tuple[int, str]]:
    """Ищет запущенные процессы из белого списка.

    Ничего, кроме перечисленных имён, не возвращается — список задан
    в конфиге и является границей того, что вообще может быть записано.
    """
    try:
        import psutil
    except ImportError:
        log.error("psutil не установлен, процессы не найти")
        return []

    wanted = {n.lower() for n in names}
    matched: dict[int, tuple[str, int | None]] = {}
    for process in psutil.process_iter(["pid", "name", "ppid"]):
        try:
            name = (process.info["name"] or "").lower()
        except Exception:
            continue
        if name in wanted:
            matched[process.info["pid"]] = (process.info["name"],
                                            process.info.get("ppid"))

    # Берём только корневые процессы. Discord поднимает шесть процессов
    # одного имени, но захват идёт по дереву: активировав родителя,
    # получаем и всех потомков. Шесть активаций вместо одной — это
    # шесть шансов упереться в процесс, который ничего не проигрывает.
    roots = [(pid, name) for pid, (name, ppid) in matched.items()
             if ppid not in matched]
    return sorted(roots or [(pid, n) for pid, (n, _) in matched.items()])


class OtherSpeakerRecorder:
    """Пишет звук собеседника из разрешённых процессов."""

    def __init__(self, process_names: list[str], sample_rate: int = 48000,
                 channels: int = 2, max_duration_ms: int = 30000):
        self.process_names = process_names
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_duration_s = max_duration_ms / 1000.0
        self._streams: list[ProcessLoopbackStream] = []
        self._captured: list[str] = []

    @property
    def is_recording(self) -> bool:
        return bool(self._streams)

    def start(self) -> list[str]:
        """Начинает запись. Возвращает имена процессов, которые удалось взять."""
        if self._streams:
            return self._captured

        targets = find_pids(self.process_names)
        if not targets:
            raise LoopbackError(
                "Ни один из разрешённых процессов не запущен: "
                + ", ".join(self.process_names))

        started: list[str] = []
        errors: list[str] = []
        for pid, name in targets:
            stream = ProcessLoopbackStream(pid, self.sample_rate, self.channels)
            try:
                stream.start()
            except LoopbackError as exc:
                # Процесс может быть запущен, но молчать — тогда звук он
                # не отдаёт. Это не повод отменять запись остальных.
                errors.append(f"{name}: {exc}")
                continue
            self._streams.append(stream)
            started.append(f"{name} (PID {pid})")

        if not self._streams:
            raise LoopbackError("Звук не отдал ни один процесс. "
                                + "; ".join(errors[:2]))
        if errors:
            log.info("Часть процессов пропущена: %s", "; ".join(errors))

        self._captured = started
        return started

    def stop(self) -> np.ndarray:
        """Останавливает запись и отдаёт смикшированный моно-поток."""
        if not self._streams:
            return np.zeros(0, dtype=np.float32)

        tracks = []
        for stream in self._streams:
            try:
                track = stream.stop()
            except Exception:
                log.exception("Ошибка при остановке захвата процесса %s",
                              stream.pid)
                continue
            if track.size:
                tracks.append(track)
        self._streams = []

        if not tracks:
            return np.zeros(0, dtype=np.float32)
        if len(tracks) == 1:
            return tracks[0]

        # Дорожки разной длины: игра и Discord начинают отдавать звук
        # не одновременно. Выравниваем по самой длинной и складываем.
        length = max(len(t) for t in tracks)
        mixed = np.zeros(length, dtype=np.float32)
        for track in tracks:
            mixed[:len(track)] += track
        mixed /= len(tracks)
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 1.0:
            mixed /= peak
        return mixed

    def cancel(self) -> None:
        for stream in self._streams:
            try:
                stream.stop()
            except Exception:
                log.exception("Ошибка при отмене захвата")
        self._streams = []
