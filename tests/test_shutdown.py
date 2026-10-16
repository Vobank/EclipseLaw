"""
Выход из приложения.

Хоткей выхода приходит из потока, который слушает клавиатуру. Там нет
цикла событий Qt, поэтому QTimer.singleShot в нём не срабатывает
никогда — бот писал в лог «Выход по хоткею» и продолжал работать, и
закрыть его можно было только Диспетчером задач. Выход обязан идти
сигналом: его Qt сама доставляет в поток окна.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def test_quit_goes_through_a_signal():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "overlay.sig_quit.connect(shutdown)" in source


def test_quit_does_not_use_a_timer_from_another_thread():
    """Регрессия: именно из-за таймера бот и не закрывался."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.split("\n")
                     if not line.strip().startswith("#"))
    assert "QTimer.singleShot" not in code


def test_overlay_has_the_quit_signal():
    from ui.overlay import OverlayWindow

    assert hasattr(OverlayWindow, "sig_quit")


def test_process_exits_even_if_a_worker_hangs():
    """Пул рабочих потоков не из демонов: без жёсткого выхода процесс
    оставался жить, пока задача ждёт ответа модели."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "os._exit" in source


def test_ollama_can_be_stopped():
    from core.llm.ollama_client import OllamaClient

    assert callable(OllamaClient.stop_server)


def test_tray_offers_full_shutdown():
    source = (ROOT / "ui" / "overlay.py").read_text(encoding="utf-8")
    assert "Выход и остановить Ollama" in source
