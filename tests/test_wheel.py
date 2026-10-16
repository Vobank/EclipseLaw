"""
Прокрутка ленты колесом поверх игры.

WM_MOUSEWHEEL Windows отдаёт окну с фокусом клавиатуры, а фокус всегда
у игры: оверлей нарочно не активируется. Без перехвата колесо над лентой
крутило игру, а лента стояла — и отлистать прошлый вопрос было нельзя.
Поэтому колесо берётся низкоуровневым хуком и только внутри зоны окна.
"""
import ctypes
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hotkeys import (  # noqa: E402
    HC_ACTION, MSLLHOOKSTRUCT, WM_MOUSEWHEEL, HotkeyManager)


def _wheel_event(x: int, y: int, delta: int) -> int:
    """Адрес MSLLHOOKSTRUCT, как его передаёт Windows в lparam."""
    info = MSLLHOOKSTRUCT()
    info.pt.x, info.pt.y = x, y
    # mouseData: шаги колеса лежат в старшем слове.
    info.mouseData = (delta & 0xFFFF) << 16
    _wheel_event.keep = info          # не дать сборщику убрать структуру
    return ctypes.addressof(info)


@pytest.fixture
def manager():
    events = []
    mgr = HotkeyManager(events.append)
    mgr.events = events
    return mgr


def _fire(mgr, x, y, delta=120):
    result = mgr._low_level_mouse(HC_ACTION, WM_MOUSEWHEEL, _wheel_event(x, y, delta))
    fired = []
    while not mgr._queue.empty():
        fired.append(mgr._queue.get_nowait())
    return result, fired


def test_wheel_inside_the_zone_scrolls_the_overlay(manager):
    manager.set_wheel_zone((100, 100, 300, 400))
    result, fired = _fire(manager, 200, 200)
    assert [e.action for e in fired] == ["scroll_up"]
    assert result == 1, "колесо над лентой не должно доходить до игры"


def test_wheel_down_is_recognised(manager):
    manager.set_wheel_zone((100, 100, 300, 400))
    _, fired = _fire(manager, 200, 200, delta=-120)
    assert [e.action for e in fired] == ["scroll_down"]


def test_wheel_outside_the_zone_goes_to_the_game(manager):
    """Главное свойство: в бою колесо — это смена оружия."""
    manager.set_wheel_zone((100, 100, 300, 400))
    result, fired = _fire(manager, 900, 900)
    assert fired == []
    assert result != 1


def test_hidden_overlay_releases_the_wheel(manager):
    """Окно спрятано — зоны нет, колесо целиком игры."""
    manager.set_wheel_zone(None)
    result, fired = _fire(manager, 200, 200)
    assert fired == []
    assert result != 1


def test_zone_edges(manager):
    manager.set_wheel_zone((100, 100, 300, 400))
    assert _fire(manager, 100, 100)[1], "левый верхний угол входит в зону"
    assert not _fire(manager, 300, 200)[1], "правая граница уже снаружи"
    assert not _fire(manager, 200, 400)[1], "нижняя граница уже снаружи"


def test_other_mouse_events_are_untouched(manager):
    """Хук не имеет права трогать ничего, кроме колеса."""
    manager.set_wheel_zone((100, 100, 300, 400))
    WM_MOUSEMOVE = 0x0200
    result = manager._low_level_mouse(
        HC_ACTION, WM_MOUSEMOVE, _wheel_event(200, 200, 120))
    assert result != 1
    assert manager._queue.empty()


def test_broken_event_does_not_kill_the_hook(manager):
    """Хук, который упал, Windows снимает — и хоткеи умирают все разом."""
    manager.set_wheel_zone((100, 100, 300, 400))
    result = manager._low_level_mouse(HC_ACTION, WM_MOUSEWHEEL, 0)
    assert result != 1
