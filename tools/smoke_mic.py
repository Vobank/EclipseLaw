"""
Живая проверка микрофона и распознавания русской речи.

Единственный тест, который реально проверяет точность STT на русском:
автоматический синтез русской речи в системе недоступен (установлен
только английский голос).

Запуск: python tools/smoke_mic.py [секунды]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.audio.mic import MicRecorder, list_input_devices  # noqa: E402
from core.config import Config, setup_logging  # noqa: E402
from core.stt.whisper_engine import WhisperEngine  # noqa: E402


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

    cfg = Config.load()
    setup_logging(cfg)

    device = cfg.get("audio.input_device")
    print("Устройства записи:")
    for idx, name, channels in list_input_devices()[:10]:
        mark = " <- выбрано" if device == idx else ""
        print(f"  [{idx:>2}] {name} ({channels} кан.){mark}")
    if device is None:
        print("\nИспользуется устройство Windows по умолчанию "
              "(audio.input_device: null).")
        print("Если распознавание берёт не тот микрофон — впишите нужный "
              "индекс в config/config.yaml.")

    rate = cfg.get("audio.sample_rate", 16000)
    recorder = MicRecorder(
        sample_rate=rate,
        channels=cfg.get("audio.channels", 1),
        device=device,
        tail_padding_ms=cfg.get("ptt.tail_padding_ms", 250),
        max_duration_ms=cfg.get("ptt.max_duration_ms", 30000),
    )

    engine = WhisperEngine.from_config(cfg)
    print(f"\nГружу Whisper {engine.model_name} ({engine.device})...")
    engine.load()

    print(f"\nГоворите по-русски — запись {seconds:.0f} секунд.")
    for i in range(3, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  ЗАПИСЬ")

    recorder.start()
    time.sleep(seconds)
    audio = recorder.stop()

    if audio.size == 0:
        print("\nПРОВАЛ: микрофон не отдал ни одного семпла.")
        return 1

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"\nЗаписано {len(audio) / rate:.1f} с, пик {peak:.3f}, RMS {rms:.4f}")
    if peak < 0.01:
        print("ПРОВАЛ: сигнал практически нулевой — микрофон отключён, "
              "заглушен или выбран не тот вход.")
        return 1
    if peak > 0.99:
        print("Внимание: сигнал клиппует, снизьте усиление микрофона.")

    result = engine.transcribe(audio, rate)
    print(f"\nРаспознано: «{result.text}»")
    print(f"Задержка: {result.latency_s:.2f} с")

    if result.is_empty:
        print("\nПРОВАЛ: речь не распознана. Проверьте громкость и "
              "правильность выбранного микрофона.")
        return 1

    print("\nСравните текст выше с тем, что вы сказали. "
          "Если расходится — увеличьте модель в конфиге "
          "(profiles.cpu_only.stt.model: medium).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
