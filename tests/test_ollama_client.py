"""
Готовность Ollama и автозапуск.

Без Ollama бесполезен весь бот: не работает ни поиск по смыслу, ни сам
ответ. При этом выключенной она оказывается легко — после перезагрузки
или закрытия окна, — а пользователь видит только «в базе не нашлось».
Здесь стережётся то, что бот поднимает её сам и не врёт о готовности.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm.ollama_client import LLMOptions, OllamaClient  # noqa: E402


def client() -> OllamaClient:
    return OllamaClient("http://127.0.0.1:11434", LLMOptions(model="тест"))


def test_running_server_is_not_restarted(monkeypatch):
    c = client()
    monkeypatch.setattr(c, "ping", lambda timeout=3.0: True)
    started = []
    monkeypatch.setattr("subprocess.Popen",
                        lambda *a, **k: started.append(a) or None)
    assert c.ensure_running() is True
    assert started == [], "запущенную Ollama трогать нельзя"


def test_missing_binary_is_reported_not_crashed(monkeypatch):
    """Ollama не установлена — это отказ, а не исключение наружу."""
    c = client()
    monkeypatch.setattr(c, "ping", lambda timeout=3.0: False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert c.ensure_running(wait_s=0.1) is False


def test_server_is_started_when_down(monkeypatch):
    c = client()
    calls = {"ping": 0}

    def ping(timeout=3.0):
        # Первый вызов — до запуска, дальше сервер уже поднялся.
        calls["ping"] += 1
        return calls["ping"] > 1

    monkeypatch.setattr(c, "ping", ping)
    monkeypatch.setattr("shutil.which", lambda name: r"C:\ollama.exe")
    launched = []
    monkeypatch.setattr("subprocess.Popen",
                        lambda cmd, **k: launched.append(cmd) or None)
    assert c.ensure_running(wait_s=5) is True
    assert launched and launched[0][1] == "serve"


def test_check_ready_gives_a_human_reason(monkeypatch):
    c = client()
    monkeypatch.setattr(c, "ping", lambda timeout=3.0: False)
    monkeypatch.setattr(c, "ensure_running", lambda wait_s=25.0: False)
    ready, reason = c.check_ready()
    assert ready is False
    assert "Ollama" in reason and "ollama serve" in reason


def test_missing_model_is_distinguished_from_dead_server(monkeypatch):
    """«Модель не скачана» и «сервер лежит» лечатся по-разному."""
    c = client()
    monkeypatch.setattr(c, "ping", lambda timeout=3.0: True)
    monkeypatch.setattr(c, "available_models", lambda timeout=5.0: ["другая:latest"])
    ready, reason = c.check_ready()
    assert ready is False
    assert "ollama pull" in reason


@pytest.mark.parametrize("stored,wanted", [
    (["тест:latest"], "тест"),
    (["тест:4b"], "тест:4b"),
])
def test_model_tag_matching(monkeypatch, stored, wanted):
    """Ollama хранит теги как «модель:latest» — без тега это то же самое."""
    c = OllamaClient("http://x", LLMOptions(model=wanted))
    monkeypatch.setattr(c, "ping", lambda timeout=3.0: True)
    monkeypatch.setattr(c, "available_models", lambda timeout=5.0: stored)
    assert c.check_ready()[0] is True
