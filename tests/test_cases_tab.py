"""
Вкладка «Дела» в шапке: подпись обязана совпадать с тем, что на экране.

Список дел открывается и хоткеем, и мышкой по вкладке. Если подпись
разъедется с состоянием — на открытом списке будет написано «Дела», и
человек нажмёт её, чтобы открыть уже открытое, а получит закрытие.
Поэтому состояние и подпись меняются только вместе.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def overlay():
    from core.config import Config
    from ui.overlay import OverlayWindow, create_app

    app = create_app()
    window = OverlayWindow(Config.load())
    window.show()
    app.processEvents()
    yield window
    window.hide()


def test_tab_says_cases_when_the_transcript_is_shown(overlay):
    overlay.set_cases_tab(False)
    assert "Дела" in overlay.lbl_cases.text()


def test_tab_says_back_when_the_list_is_shown(overlay):
    overlay.set_cases_tab(True)
    assert "Назад" in overlay.lbl_cases.text()


def test_clicking_the_tab_is_reported(overlay):
    clicks = []
    overlay.sig_cases_tab.connect(lambda: clicks.append(1))
    overlay.lbl_cases.linkActivated.emit("cases")
    assert clicks == [1]


def test_clicking_a_case_reports_its_session(overlay):
    """Ссылка «case:...» — это выбор дела, а не переход в браузер."""
    chosen = []
    overlay.sig_case.connect(chosen.append)
    overlay._on_link("case:дело-от-20-сентября")
    assert chosen == ["дело-от-20-сентября"]


def test_unknown_links_are_ignored(overlay):
    chosen = []
    overlay.sig_case.connect(chosen.append)
    overlay._on_link("https://example.com")
    assert chosen == []


def test_wheel_zone_follows_the_window(overlay):
    """Хук берёт колесо по координатам: уехало окно — уехала и зона."""
    class Sink:
        zone = None

        def set_wheel_zone(self, zone):
            self.zone = zone

    sink = Sink()
    overlay.set_wheel_zone_sink(sink)
    first = sink.zone
    assert first is not None and first[2] > first[0]

    overlay.move(overlay.x() + 120, overlay.y() + 60)
    assert sink.zone != first, "зона осталась на прежнем месте"


def test_hidden_window_gives_the_wheel_back(overlay):
    """Спрятанный оверлей не имеет права перехватывать колесо в бою."""
    class Sink:
        zone = "не трогали"

        def set_wheel_zone(self, zone):
            self.zone = zone

    sink = Sink()
    overlay.set_wheel_zone_sink(sink)
    overlay.hide()
    assert sink.zone is None
    overlay.show()


# ---------------------------------------------------------- чистота окна
def test_hotkey_hint_is_hidden(overlay):
    """Строка «ctrl+alt+q — вопрос» место занимает, пользы не несёт:
    свои сочетания человек запоминает за день, а полный список
    остаётся на ctrl+alt+f1."""
    assert not overlay.lbl_hint.isVisible()


def test_session_log_is_hidden(overlay):
    """Нижний лог со временем и подписями «Система» тоже убран."""
    assert not overlay.log_scroll.isVisible()


def test_log_separator_goes_with_the_log(overlay):
    """Иначе от скрытого лога остаётся висящая линия в пустоте."""
    assert not overlay.log_separator.isVisible()


def test_log_can_still_be_brought_back(overlay):
    overlay.toggle_log()
    assert overlay.log_scroll.isVisible()
    assert overlay.log_separator.isVisible()
    overlay.toggle_log()


# --------------------------------------------------------------- значок
def test_app_icon_exists_and_is_multisize():
    """Windows берёт 16 пикселей для списка файлов и 256 для рабочего
    стола: один размер выглядел бы мылом в одном из мест."""
    from ui.overlay import app_icon

    icon = app_icon()
    assert not icon.isNull()
    sides = {size.width() for size in icon.availableSizes()}
    assert {16, 32, 256} <= sides


def test_icon_file_is_where_the_shortcut_expects_it():
    """Ярлык на рабочем столе ссылается на этот файл по пути."""
    path = Path(__file__).resolve().parent.parent / "assets" / "tobi.ico"
    assert path.exists(), "иконки нет — запустите python tools/make_icon.py"
