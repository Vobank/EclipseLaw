"""
Служебные блоки не должны попадать в контекст диалога.

Под ответом программа дописывает «── О ЧЁМ СТАТЬЯ ──». Этот текст
складывался в историю сессии вместе с ответом, уходил модели как
образец её собственного прошлого ответа — и на следующем вопросе она
дописывала такой блок сама. В ленте появлялись два блока подряд:
один от модели (с полным текстом статьи), второй настоящий.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.answer import strip_service_blocks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_service_block_is_stripped():
    text = ("Разрешено выловить до 10 килограмм. [Закон, п. 3]\n"
            "\n"
            "── О ЧЁМ СТАТЬЯ ──\n"
            "Закон, п. 3 — Любительская охота это деятельность...\n")
    assert "О ЧЁМ СТАТЬЯ" not in strip_service_blocks(text)
    assert "Разрешено выловить" in strip_service_blocks(text)


def test_hotkey_tail_is_stripped():
    text = "Ответ. [Закон, п. 3]\n\nТекст статьи целиком — ctrl+alt+s"
    assert "ctrl+alt+s" not in strip_service_blocks(text)


def test_full_article_block_is_stripped():
    text = "Ответ.\n\n── СТАТЬЯ ЦЕЛИКОМ ──\nДлинный текст статьи."
    assert "СТАТЬЯ ЦЕЛИКОМ" not in strip_service_blocks(text)


def test_ordinary_answer_is_untouched():
    text = "Штраф до 1.000$ за превышение. [Дорожный Кодекс, п. 32]"
    assert strip_service_blocks(text) == text


def test_dashes_in_the_answer_survive():
    """Обычное тире в тексте — не служебный блок."""
    text = "Превышение — это движение быстрее лимита. [ДК, п. 32]"
    assert strip_service_blocks(text) == text


def test_only_the_spoken_part_goes_into_the_session():
    """Регрессия: в сессию кладётся ответ модели, а не то, что на экране."""
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "session.add(source, question, spoken, articles=articles)" in source
    assert 'session.add(Source.SYSTEM, "Вынести заключение", spoken_verdict)' \
        in source


# --------------------------------------------- облегчённый профиль
def test_game_profile_fits_next_to_the_game():
    """GTA Enhanced на ультрах забирает 12-13 ГБ из 16.

    Обычный профиль в остаток не помещается: 7B (4.8) плюс Whisper
    medium (1.5) плюс bge-m3 (0.6) — это семь гигабайт против трёх
    свободных. Драйвер начинает вытеснять страницы модели, и ответ,
    который считался секунду, растягивается на полторы минуты.
    """
    from core.config import Config

    profile = Config.load().get("profiles")["game_max"]
    # Распознавание остаётся на карте и остаётся medium. Пробовали
    # иначе: small на процессоре превращал «задержал сотрудник LSPD»
    # в «задержался трудник ЛСПД», и бот отвечал не на тот вопрос.
    assert profile["stt"]["model"] == "medium"
    assert profile["stt"]["device"] == "cuda"
    # Экономим на поиске: его полгигабайта решают, влезет ли модель.
    assert profile["embeddings"]["device"] == "cpu"
    # Модель меньше семёрки.
    assert "4b" in profile["llm"]["model"]
    # Окно меньше обычного, но не настолько, чтобы промпт экзамена
    # в него не влез: при 4096 Ollama отвечала 400.
    assert 6144 <= profile["llm"]["num_ctx"] < 8192


def test_embeddings_can_be_pinned_to_cpu():
    from core.legal.vectors import EmbeddingClient

    client = EmbeddingClient("http://127.0.0.1:11434", "bge-m3", num_gpu=0)
    assert client.num_gpu == 0


def test_orchestrator_honours_the_profile():
    source = (ROOT / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "embeddings_on_cpu" in source
