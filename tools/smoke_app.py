"""
Интеграционная проверка: поднимает приложение целиком ровно так же,
как main.py, даёт ему поработать и проверяет, что всё поднялось —
клавиатурный хук стоит, хоткеи привязаны, модели прогрелись.

Клавиши при этом не нажимаются: PTT проверяется вручную через
tools/smoke_mic.py и живым запуском.

Запуск: python tools/smoke_app.py [секунды]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer  # noqa: E402

from core.config import Config, setup_logging  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from ui.overlay import OverlayWindow, create_app  # noqa: E402


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0

    cfg = Config.load()
    setup_logging(cfg)

    app = create_app()
    overlay = OverlayWindow(cfg)
    overlay.show()
    overlay.apply_native_flags()

    orchestrator = Orchestrator(cfg, overlay)
    started = time.monotonic()
    orchestrator.start()

    failures: list[str] = []

    def inspect() -> None:
        elapsed = time.monotonic() - started

        # 1. Хук клавиатуры действительно установлен в системе.
        if orchestrator.hotkeys._hook:
            print(f"  OK    клавиатурный хук установлен "
                  f"(handle={orchestrator.hotkeys._hook})")
        else:
            failures.append("клавиатурный хук не установлен")
            print("  СБОЙ  клавиатурный хук НЕ установлен — "
                  "хоткеи работать не будут")

        # 2. Все хоткеи из конфига разобраны и привязаны.
        bound = sum(len(v) for v in orchestrator.hotkeys._bindings.values())
        expected = len(cfg.hotkeys)
        if bound == expected:
            print(f"  OK    привязано хоткеев: {bound} из {expected}")
        else:
            failures.append(f"привязано {bound} хоткеев из {expected}")
            print(f"  СБОЙ  привязано {bound} из {expected}")

        # 3. Прогрев завершился: Whisper в памяти, Ollama отвечает.
        if orchestrator.stt.is_loaded:
            print(f"  OK    Whisper {orchestrator.stt.model_name} загружен")
        else:
            failures.append("Whisper не загрузился")
            print("  СБОЙ  Whisper не загружен")

        if orchestrator._llm_ready:
            print(f"  OK    Ollama готова, модель "
                  f"{orchestrator.llm.options.model}")
        else:
            failures.append("Ollama не готова")
            print("  СБОЙ  Ollama не готова")

        # 4. Оверлей на экране.
        if overlay.isVisible():
            geometry = overlay.geometry()
            print(f"  OK    оверлей виден: {geometry.width()}x"
                  f"{geometry.height()} в ({geometry.x()}, {geometry.y()})")
        else:
            failures.append("оверлей не виден")
            print("  СБОЙ  оверлей не виден")

        print(f"\nВремя до готовности: {elapsed:.1f} с")
        app.quit()

    print(f"Приложение запущено, прогрев до {seconds:.0f} с...\n")

    def wait_or_timeout(deadline: float = time.monotonic() + seconds) -> None:
        """Ждём конца прогрева, но не дольше отведённого времени."""
        warm = orchestrator.stt.is_loaded and orchestrator._llm_ready
        if warm or time.monotonic() >= deadline:
            inspect()
        else:
            QTimer.singleShot(500, wait_or_timeout)

    QTimer.singleShot(500, wait_or_timeout)
    app.exec()
    orchestrator.stop()

    if failures:
        print(f"\nИтог: ПРОВАЛОВ {len(failures)} — " + "; ".join(failures))
        return 1
    print("\nИтог: приложение поднимается полностью.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
