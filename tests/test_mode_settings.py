"""
Чтение настроек режима из конфига.

Файл появился после того, как из-за одной строки бот дважды ответил
«в базе не нашлось» на нормальные вопросы. В конфиге у обычного режима
стоял top_k: 0 с комментарием «0 = как в секции retrieval», а код брал
ноль буквально и искал ноль статей. Поиск при этом отрабатывал штатно
и без ошибок — просто возвращал пустоту.

Проверки тут скучные ровно настолько, насколько дорого обошлась их
нехватка.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.session import Mode  # noqa: E402


class Holder:
    """Только чтение настроек — поднимать весь оркестратор незачем."""

    _mode_setting = Orchestrator._mode_setting

    def __init__(self, cfg):
        self.cfg = cfg


class FakeConfig:
    def __init__(self, values: dict):
        self.values = values

    def get(self, path, default=None):
        return self.values.get(path, default)


@pytest.mark.parametrize("key,default", [
    ("top_k", 6),
    ("max_article_chars", 900),
    ("num_predict", 160),
])
def test_zero_means_take_the_default(key, default):
    """Ноль — это «не задано», а не «ноль»."""
    holder = Holder(FakeConfig({f"modes.normal.{key}": 0}))
    assert holder._mode_setting(Mode.NORMAL, key, default) == default


def test_missing_setting_takes_the_default():
    holder = Holder(FakeConfig({}))
    assert holder._mode_setting(Mode.NORMAL, "top_k", 6) == 6


def test_real_value_wins():
    holder = Holder(FakeConfig({"modes.exam.top_k": 10}))
    assert holder._mode_setting(Mode.EXAM, "top_k", 6) == 10


def test_false_is_a_value_not_a_gap():
    """False означает «выключено», а не «настройка пропущена».

    Без этой оговорки правило «ноль = умолчание» проглотило бы и False:
    в Python False == 0.
    """
    holder = Holder(FakeConfig({"modes.normal.retry_on_invented": False}))
    assert holder._mode_setting(Mode.NORMAL, "retry_on_invented", True) is False


# --------------------------------------------------- на настоящем конфиге
@pytest.mark.parametrize("mode", [Mode.NORMAL, Mode.LAWYER, Mode.EXAM])
def test_every_mode_searches_for_something(mode):
    """Главная проверка: ни один режим не ищет ноль статей.

    Ровно это и случилось — поиск возвращал пустоту, ассистент отвечал
    «в базе не нашлось», и по логам всё выглядело исправным.
    """
    holder = Holder(Config.load())
    assert holder._mode_setting(mode, "top_k", 6) >= 3


@pytest.mark.parametrize("mode", [Mode.NORMAL, Mode.LAWYER, Mode.EXAM])
def test_every_mode_keeps_enough_of_the_article(mode):
    holder = Holder(Config.load())
    assert holder._mode_setting(mode, "max_article_chars", 900) >= 300
