"""
Запись голосом на одиночные клавиши.

Вопрос задают в спешке, одной рукой, не отрывая другую от мыши:
CapsLock — свой голос, Tab — собеседник или экзаменатор. В игре обе
клавиши ничем не заняты, поэтому бот забирает их себе целиком.

Главная тонкость — модификаторы. Привязка на одиночный Tab обязана
молчать при Alt+Tab, иначе каждое переключение окон начинало бы запись
и съедало бы сам Alt+Tab.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config  # noqa: E402
from core.hotkeys import Binding, HotkeyManager, parse_hotkey  # noqa: E402


@pytest.fixture(scope="module")
def hotkeys():
    return Config.load().hotkeys


def test_own_voice_is_capslock(hotkeys):
    assert hotkeys["ptt_self"] == "capslock"


def test_interlocutor_is_tab(hotkeys):
    assert hotkeys["ptt_other"] == "tab"


def test_single_keys_parse(hotkeys):
    for action in ("ptt_self", "ptt_other"):
        mods, vk = parse_hotkey(hotkeys[action])
        assert mods == frozenset(), "модификаторов быть не должно"
        assert vk


def test_no_hotkey_conflicts(hotkeys):
    seen = {}
    for action, combo in hotkeys.items():
        seen.setdefault(combo, []).append(action)
    assert not [c for c, a in seen.items() if len(a) > 1]


# ------------------------------------------------------------ модификаторы
def _binding(spec: str) -> Binding:
    mods, vk = parse_hotkey(spec)
    return Binding(action="ptt_other", spec=spec, mods=mods, vk=vk)


def test_plain_key_matches_when_nothing_is_held(monkeypatch):
    monkeypatch.setattr("core.hotkeys._modifier_down", lambda name: False)
    assert HotkeyManager._modifiers_match(_binding("tab"))


def test_alt_tab_does_not_trigger_recording(monkeypatch):
    """Регрессия: иначе переключение окон начинало бы запись."""
    monkeypatch.setattr("core.hotkeys._modifier_down",
                        lambda name: name == "alt")
    assert not HotkeyManager._modifiers_match(_binding("tab"))


def test_combination_still_needs_its_modifiers(monkeypatch):
    monkeypatch.setattr("core.hotkeys._modifier_down",
                        lambda name: name in ("ctrl", "alt"))
    assert HotkeyManager._modifiers_match(_binding("ctrl+alt+q"))


def test_combination_rejects_extra_modifiers(monkeypatch):
    """Ctrl+Alt+Shift+Q не должен срабатывать как Ctrl+Alt+Q."""
    monkeypatch.setattr("core.hotkeys._modifier_down",
                        lambda name: name in ("ctrl", "alt", "shift"))
    assert not HotkeyManager._modifiers_match(_binding("ctrl+alt+q"))


# ----------------------------------------------------- ожидаемые ответы
def test_every_question_says_what_to_expect():
    """Без этого ответ сотрудника нечем мерить: нарушение звучит так же
    уверенно, как соблюдение процедуры."""
    for role in Config.load().get("lawyer.roles").values():
        for item in role["checklist"]:
            assert item.get("expected"), item["key"]


def test_expected_answer_reaches_the_item():
    from core.legal.checklist import ChecklistItem

    item = ChecklistItem(key="k", title="t", expected="Названа статья")
    assert item.expected == "Названа статья"


def test_counsel_starts_from_the_substance():
    """Допуск и основание вызова убраны: адвоката пускают и так."""
    counsel = Config.load().get("lawyer.roles")["counsel"]["checklist"]
    keys = [item["key"] for item in counsel]
    assert "допуск" not in keys
    assert "вызов" not in keys
    assert keys[0] == "причины"
