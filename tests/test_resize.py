"""
Окно тянется за края и помнит свой размер.

Оверлей без рамки: системных углов для растягивания у него нет, и
размер задавался только в config.yaml руками. Поэтому края обрабатываем
сами — полосой в несколько пикселей по периметру.

Размер хранится отдельным файлом, а не в config.yaml: тот написан
руками и с комментариями, и переписывать его из программы значит
однажды стереть их все.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    from core.config import Config
    from ui.overlay import OverlayWindow, create_app

    app = create_app()
    cfg = Config.load()
    window = OverlayWindow(cfg)
    # Состояние уводим во временный каталог, чтобы тест не переписал
    # рабочий размер окна пользователя.
    state = tmp_path_factory.mktemp("state") / "overlay_state.json"
    window._state_path = lambda: state
    window.show()
    app.processEvents()
    yield window
    window.hide()


# ------------------------------------------------------------------ края
def test_corner_is_recognised(overlay):
    from PySide6.QtCore import QPoint
    edges = overlay._edges_at(QPoint(2, 2))
    assert edges == (True, True, False, False)


def test_middle_is_not_an_edge(overlay):
    """Середина — это перетаскивание и выделение текста, а не размер."""
    from PySide6.QtCore import QPoint
    assert not any(overlay._edges_at(QPoint(200, 150)))


@pytest.mark.parametrize("edges,expected", [
    ((True, True, False, False), "SizeFDiagCursor"),
    ((False, False, True, True), "SizeFDiagCursor"),
    ((False, True, True, False), "SizeBDiagCursor"),
    ((True, False, False, False), "SizeHorCursor"),
    ((False, False, False, True), "SizeVerCursor"),
    ((False, False, False, False), "ArrowCursor"),
])
def test_cursor_hints_the_direction(overlay, edges, expected):
    assert overlay._cursor_for(edges).name == expected


# -------------------------------------------------------------- растягивание
def test_dragging_the_corner_grows_the_window(overlay):
    from PySide6.QtCore import QPoint

    overlay.setGeometry(100, 100, 520, 420)
    overlay._resize_edges = (False, False, True, True)
    overlay._resize_origin = (QPoint(620, 520), overlay.geometry())
    overlay._resize_to(QPoint(740, 600))
    assert overlay.width() > 520 and overlay.height() > 420


def test_window_cannot_be_squashed_to_nothing(overlay):
    from PySide6.QtCore import QPoint

    overlay.setGeometry(100, 100, 520, 420)
    overlay._resize_edges = (False, False, True, True)
    overlay._resize_origin = (QPoint(620, 520), overlay.geometry())
    overlay._resize_to(QPoint(0, 0))
    assert overlay.width() >= overlay.MIN_WIDTH - 1
    assert overlay.height() >= overlay.MIN_HEIGHT - 1


def test_shrinking_by_the_left_edge_keeps_the_right_side(overlay):
    """Иначе окно уезжает вправо, пока его сжимают слева."""
    from PySide6.QtCore import QPoint

    overlay.setGeometry(100, 100, 520, 420)
    right = overlay.geometry().right()
    overlay._resize_edges = (True, False, False, False)
    overlay._resize_origin = (QPoint(100, 300), overlay.geometry())
    overlay._resize_to(QPoint(900, 300))     # тянем далеко за правый край
    assert abs(overlay.geometry().right() - right) <= 1


# ------------------------------------------------------------------ память
def test_size_survives_a_restart(overlay):
    overlay.setGeometry(150, 170, 640, 480)
    overlay.save_geometry()
    assert overlay._restore_geometry() == (150, 170, 640, 480)


def test_saved_file_is_plain_and_readable(overlay):
    overlay.setGeometry(10, 20, 600, 400)
    overlay.save_geometry()
    saved = json.loads(overlay._state_path().read_text(encoding="utf-8"))
    assert saved == {"x": 10, "y": 20, "width": 600, "height": 400}


def test_broken_state_falls_back_to_config(overlay):
    """Файл могли испортить руками — окно всё равно обязано открыться."""
    overlay._state_path().write_text("{не json", encoding="utf-8")
    x, y, width, height = overlay._restore_geometry()
    assert width == overlay.cfg.get("overlay.width", 520)


def test_window_off_screen_comes_back(overlay):
    """Сменился монитор — окно за краем экрана мышью уже не достать."""
    overlay._state_path().write_text(
        json.dumps({"x": 99999, "y": 99999, "width": 600, "height": 400}),
        encoding="utf-8")
    x, y, _, _ = overlay._restore_geometry()
    assert (x, y) == (overlay.cfg.get("overlay.x", 40),
                      overlay.cfg.get("overlay.y", 40))
