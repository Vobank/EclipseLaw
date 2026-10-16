"""
Живая проверка захвата речи собеседника (этап 5).

Пишет звук из GTA5.exe и Discord.exe несколько секунд и показывает,
что получилось. Во время проверки в игре или в Discord должен идти
звук — иначе писать будет нечего, и это не ошибка захвата.

Заодно подтверждается главное требование ТЗ: пишутся только процессы
из белого списка, системный звук не трогается.

Запуск: python tools/smoke_loopback.py [секунды]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.audio.loopback import LoopbackError  # noqa: E402
from core.audio.other import OtherSpeakerRecorder, find_pids  # noqa: E402
from core.config import Config, setup_logging  # noqa: E402


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

    cfg = Config.load()
    setup_logging(cfg)
    names = cfg.get("audio.loopback_processes", [])

    print(f"Разрешённые процессы (белый список ТЗ): {', '.join(names)}")
    targets = find_pids(names)
    if not targets:
        print("\nНи один из них не запущен. Откройте GTA 5 или Discord "
              "и повторите.")
        return 1
    print("Найдены корневые процессы:")
    for pid, name in targets:
        print(f"   PID {pid:>7}  {name}")

    recorder = OtherSpeakerRecorder(
        names,
        sample_rate=cfg.get("audio.loopback_sample_rate", 48000),
        channels=cfg.get("audio.loopback_channels", 2))

    print(f"\nЗапись {seconds:.0f} секунд. Пусть в игре или Discord "
          "сейчас звучит звук.")
    try:
        started = recorder.start()
    except LoopbackError as exc:
        print(f"\nПРОВАЛ: {exc}")
        return 1
    print("Пишу:", ", ".join(started))

    for i in range(int(seconds), 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)

    audio = recorder.stop()
    if audio.size == 0:
        print("\nПРОВАЛ: не получено ни одного семпла.")
        print("Если звук точно шёл — возможно, процесс отдаёт его через "
              "виртуальное устройство (SteelSeries Sonar, VB-Audio).")
        return 1

    rate = cfg.get("audio.loopback_sample_rate", 48000)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"\nЗаписано {len(audio) / rate:.1f} с, пик {peak:.3f}, RMS {rms:.4f}")

    if peak < 0.001:
        print("Тишина: захват работает, но звук в этот момент не шёл.")
        return 0

    print("Звук получен — захват по процессам работает.")

    # Проверяем всю цепочку: распознавание речи собеседника.
    if "--stt" in sys.argv:
        from core.stt.whisper_engine import WhisperEngine
        engine = WhisperEngine.from_config(cfg)
        target_rate = cfg.get("audio.sample_rate", 16000)
        if rate != target_rate:
            positions = np.linspace(0, len(audio) - 1,
                                    int(len(audio) * target_rate / rate))
            audio = np.interp(positions, np.arange(len(audio)),
                              audio).astype(np.float32)
        result = engine.transcribe(audio, target_rate)
        print(f"\nРаспознано: «{result.text}»")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
