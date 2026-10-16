"""
Сквозная проверка конвейера распознавания.

Оговорка о том, что здесь проверяется: в системе установлен только
английский голос (Microsoft Zira), кириллицу он не произносит. Поэтому
тест проверяет механику — аудио доходит до Whisper, модель грузится и
возвращает осмысленный текст, — но НЕ точность распознавания русской
речи. Её проверяет только живая проверка микрофоном:

    python tools/smoke_mic.py

Запуск: python tools/smoke_stt.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config, setup_logging  # noqa: E402
from core.stt.whisper_engine import WhisperEngine  # noqa: E402

# Английские фразы — под единственный установленный в системе голос.
PHRASES = [
    "What article covers speeding inside the city limits",
    "The officer must identify himself during the arrest",
]


def synthesize(text: str, path: Path) -> bool:
    """Озвучивает текст системным голосом в WAV. False, если не вышло."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); "
        f"$s.Speak('{text}'); "
        "$s.Dispose()"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  синтез не удался: {result.stderr.strip()[:200]}")
        return False
    if not path.exists() or path.stat().st_size < 1000:
        # Пустой WAV — голос не смог произнести текст.
        print("  синтез вернул пустой файл: установленный голос "
              "не произносит этот текст")
        return False
    return True


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Читает WAV в моно float32."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:
        audio = (audio - 128) / 128.0
    else:
        audio /= float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Линейная интерполяция — для проверки STT её точности достаточно."""
    if src == dst:
        return audio
    length = int(round(len(audio) * dst / src))
    positions = np.linspace(0, len(audio) - 1, length)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def similarity(a: str, b: str) -> float:
    """Доля слов эталона, найденных в распознанном тексте."""
    wa = {w.strip(".,!?").lower() for w in a.split()}
    wb = {w.strip(".,!?").lower() for w in b.split()}
    return len(wa & wb) / max(len(wa), 1)


def main() -> int:
    cfg = Config.load()
    setup_logging(cfg)

    engine = WhisperEngine.from_config(cfg)
    # Язык переопределяем: голос в системе английский, а рабочий
    # язык приложения — русский, и это ожидаемое расхождение.
    engine.language = "en"
    print(f"Модель: {engine.model_name} на {engine.device} "
          f"({engine.compute_type})")
    print("Язык теста: en (в системе нет русского голоса), "
          "рабочий язык приложения: "
          f"{cfg.get('stt.language', 'ru')}\n")

    target_rate = cfg.get("audio.sample_rate", 16000)
    checked = 0
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        for i, phrase in enumerate(PHRASES):
            wav = Path(tmp) / f"phrase{i}.wav"
            print(f"Фраза: «{phrase}»")
            if not synthesize(phrase, wav):
                print("  ПРОПУЩЕНО\n")
                continue

            audio, rate = load_wav(wav)
            audio = resample(audio, rate, target_rate)
            print(f"  аудио: {len(audio) / target_rate:.1f} с "
                  f"({rate} Гц -> {target_rate} Гц)")

            result = engine.transcribe(audio, target_rate)
            checked += 1
            score = similarity(phrase, result.text)
            if score < 0.6:
                failures += 1
            print(f"  распознано: «{result.text}»")
            print(f"  совпадение слов: {score:.0%} — "
                  f"{'OK' if score >= 0.6 else 'ПРОВАЛ'}")
            ratio = result.latency_s / max(result.duration_s, 0.01)
            print(f"  задержка: {result.latency_s:.2f} с "
                  f"(x{ratio:.2f} от длительности аудио)\n")

    if checked == 0:
        # Ни одна фраза не дошла до модели — это не успех.
        print("Итог: НЕ ПРОВЕРЕНО — синтезатор не дал ни одного файла.")
        return 2
    if failures:
        print(f"Итог: ПРОВАЛ — неудачных фраз: {failures} из {checked}")
        return 1

    print(f"Итог: конвейер работает, проверено фраз: {checked}.")
    print("Точность по русской речи этим тестом НЕ проверена — "
          "используйте tools/smoke_mic.py с живым микрофоном.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
