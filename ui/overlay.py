"""
Окно-оверлей поверх игры (ТЗ 5.1).

Чтобы оверлей вообще был виден поверх GTA, игра должна работать в режиме
Borderless / Fullscreen Windowed. В Exclusive Fullscreen Windows не
позволяет рисовать поверх чужого swap chain — это общее ограничение для
любых оверлеев (Discord, Steam, MSI Afterburner), а не особенность
приложения.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QPoint, QRect, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea,
    QSizePolicy, QSystemTrayIcon, QVBoxLayout, QWidget,
)

from ui import win32

log = logging.getLogger(__name__)

# Цвета индикатора состояния.
# Цвет названия режима. Режимы отвечают по-разному, и перепутать их
# дорого: в обычном режиме на экзаменационный вопрос придёт карточка
# из трёх строк вместо развёрнутого ответа, а заметить это можно уже
# после того, как ответ прозвучал. Цвет виден краем глаза.
MODE_COLORS = {
    "Обычный": "#e5e7eb",
    "Адвокат при задержании": "#f59e0b",
    "Экзамен": "#22d3ee",
    "Оформление документа": "#a855f7",
}

STATUS_COLORS = {
    "idle": "#6b7280",
    "recording": "#ef4444",
    "transcribing": "#f59e0b",
    "thinking": "#3b82f6",
    "importing": "#a855f7",
    "ready": "#22c55e",
    "imported": "#22c55e",
    "error": "#f43f5e",
}

STATUS_TITLES = {
    "idle": "Ожидание",
    "recording": "Запись...",
    "transcribing": "Распознавание...",
    "thinking": "Думает...",
    "importing": "Импорт...",
    "ready": "Готово",
    "imported": "Импорт завершён",
    "error": "Ошибка",
}


class OverlayWindow(QWidget):
    """Полупрозрачная панель: режим, статус, последний ответ, лог сессии."""

    # Сигналы нужны потому, что события приходят из рабочих потоков,
    # а трогать виджеты Qt разрешает только из UI-потока.
    # Сколько записей держать в ленте сессии: дальше старое уезжает,
    # иначе после часа работы метка тянет на сотни килобайт текста.
    MAX_ENTRIES = 20

    # Нажата вкладка «Дела» в шапке — открыть или закрыть список.
    sig_cases_tab = Signal()

    # Просьба завершить работу. Нужен именно сигнал: хоткей выхода
    # приходит из потока хука, а останавливать цикл событий Qt
    # разрешает только из потока UI.
    sig_quit = Signal()

    # Выбран вид документа: «court|ОПРЕДЕЛЕНИЕ о принятии иска».
    sig_form = Signal(str)

    # Выбрана роль в режиме адвоката: «detained» или «counsel».
    sig_role = Signal(str)

    # Человек выбрал дело в списке: сюда уходит его session_id.
    sig_case = Signal(str)

    sig_status = Signal(str, str)        # (ключ состояния, поясняющий текст)
    sig_answer = Signal(str)
    sig_answer_chunk = Signal(str)
    sig_mode = Signal(str)
    sig_log = Signal(str, str)           # (кто, текст)
    # Колесо над окном. Приходит из потока хука, поэтому через сигнал.
    sig_scroll = Signal(int)             # шагов вверх (+) или вниз (-)
    sig_replace = Signal(str, str)       # (что заменить, на что)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._drag_offset: QPoint | None = None
        # Края, за которые тянут окно, и геометрия на момент захвата.
        self._resize_edges: tuple[bool, bool, bool, bool] | None = None
        self._resize_origin = (QPoint(), QRect())
        self._log_lines: list[str] = []
        self._max_log_lines = cfg.get("overlay.session_log_lines", 12)
        self._answer_buffer = ""
        # Завершённые записи ленты: вопрос вместе с ответом. Живут
        # до сброса сессии, чтобы можно было отлистать назад.
        self._entries: list[str] = []
        self._expanded = False
        # Кому сообщать, над каким куском экрана колесо принадлежит нам.
        self._wheel_zone_sink = None

        self._build_ui()
        self._apply_window_flags()
        # Без этого mouseMoveEvent приходит только с зажатой кнопкой,
        # и курсор у края не превращается в «растянуть» — человек не
        # догадается, что окно вообще тянется.
        self.setMouseTracking(True)

        self.sig_status.connect(self._on_status)
        self.sig_answer.connect(self._on_answer)
        self.sig_answer_chunk.connect(self._on_answer_chunk)
        self.sig_mode.connect(self._on_mode)
        self.sig_log.connect(self._on_log)
        self.sig_scroll.connect(self._on_scroll)
        self.sig_replace.connect(self.replace_in_answer)

    # ------------------------------------------------------------------- UI
    def _state_path(self) -> Path:
        """Где лежит запомненный размер окна.

        Отдельным файлом, а не в config.yaml: тот написан руками и с
        комментариями, и переписывать его из программы значит однажды
        стереть их все.
        """
        return Path(self.cfg.get("paths.data", "data")) / "overlay_state.json"

    def save_geometry(self) -> None:
        """Запоминает, куда поставили и как растянули окно."""
        try:
            path = self._state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            rect = self.geometry()
            path.write_text(json.dumps({
                "x": rect.x(), "y": rect.y(),
                "width": rect.width(), "height": rect.height()}),
                encoding="utf-8")
        except Exception:
            # Не смогли запомнить размер — не повод ронять ответ,
            # который человек читает.
            log.debug("Размер окна не сохранён", exc_info=True)

    def _restore_geometry(self) -> tuple[int, int, int, int]:
        """Размер с прошлого запуска, иначе — из конфига."""
        cfg = self.cfg
        default = (cfg.get("overlay.x", 40), cfg.get("overlay.y", 40),
                   cfg.get("overlay.width", 520),
                   cfg.get("overlay.height", 420))
        try:
            saved = json.loads(self._state_path().read_text(encoding="utf-8"))
        except Exception:
            return default
        try:
            rect = (int(saved["x"]), int(saved["y"]),
                    int(saved["width"]), int(saved["height"]))
        except (KeyError, TypeError, ValueError):
            return default
        # Монитор мог смениться: окно, уехавшее за пределы экрана,
        # не вернуть мышью, потому что его не видно.
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            if not area.intersects(QRect(*rect)):
                return default
        return rect

    def _build_ui(self) -> None:
        cfg = self.cfg
        x, y, width, height = self._restore_geometry()
        self.resize(width, height)
        self.move(x, y)
        self.setWindowOpacity(cfg.get("overlay.opacity", 0.88))

        base_size = cfg.get("overlay.font_size", 13)
        self.setFont(QFont("Segoe UI", base_size))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        # --- шапка: режим + статус ---
        header = QHBoxLayout()
        header.setSpacing(8)

        self.lbl_mode = QLabel("Обычный")
        self.lbl_mode.setStyleSheet(
            f"color:#e5e7eb; font-size:{base_size + 2}px; font-weight:600;")

        self.lbl_status = QLabel("Ожидание")
        self.lbl_status.setStyleSheet(
            f"color:{STATUS_COLORS['idle']}; font-size:{base_size}px; font-weight:600;")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Вкладка «Дела» — как список бесед в чат-ботах. Хоткей на
        # неё есть, но искать прошлый иск на слух неудобно: глазами
        # по списку быстрее, а рука и так на мыши.
        self.lbl_cases = QLabel()
        self.lbl_cases.setTextFormat(Qt.RichText)
        self.lbl_cases.setOpenExternalLinks(False)
        self.lbl_cases.linkActivated.connect(
            lambda _: self.sig_cases_tab.emit())
        self.lbl_cases.setCursor(Qt.PointingHandCursor)
        self.set_cases_tab(False)

        header.addWidget(self.lbl_mode)
        header.addWidget(self.lbl_cases)
        header.addStretch(1)
        header.addWidget(self.lbl_status)
        outer.addLayout(header)

        # Подсказка — ровно одна строка. С переносом она на узком окне
        # разрасталась на три строки и тоже отъедала место у ответа.
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet(f"color:#9ca3af; font-size:{base_size - 2}px;")
        self.lbl_hint.setWordWrap(False)
        self.lbl_hint.setTextInteractionFlags(Qt.NoTextInteraction)
        outer.addWidget(self.lbl_hint)
        # Подсказка с клавишами по умолчанию скрыта: свои сочетания
        # человек запоминает за день, а строка занимает место каждый
        # раз. Полный список остаётся на ctrl+alt+f1.
        self.lbl_hint.setVisible(cfg.get("overlay.show_hint", False))

        outer.addWidget(self._separator())

        # --- тело: последний ответ ---
        self.lbl_answer = QLabel("Ассистент запускается...")
        self.lbl_answer.setWordWrap(True)
        self.lbl_answer.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        # Список дел рисуется разметкой со ссылками: клик по строке
        # открывает дело. Отдельный виджет-список ради этого заводить
        # незачем — метка и так умеет и ссылки, и прокрутку.
        self.lbl_answer.setOpenExternalLinks(False)
        self.lbl_answer.linkActivated.connect(self._on_link)
        self.lbl_answer.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_answer.setStyleSheet(
            f"color:#f3f4f6; font-size:{base_size + 1}px;")
        self.lbl_answer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.lbl_answer)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("background: transparent;")
        self.scroll.viewport().setStyleSheet("background: transparent;")
        outer.addWidget(self.scroll, stretch=1)

        self.log_separator = self._separator()
        outer.addWidget(self.log_separator)

        # --- подвал: лог сессии ---
        # Лог живёт в области фиксированной высоты со своей прокруткой.
        # Раньше он был обычной меткой и рос вместе с числом строк,
        # отбирая место у ответа: чем больше вопросов задано, тем меньше
        # оставалось на сам ответ. Теперь высота не зависит от того,
        # сколько всего записано.
        self.lbl_log = QLabel("")
        self.lbl_log.setWordWrap(True)
        self.lbl_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_log.setStyleSheet(
            f"color:#9ca3af; font-size:{base_size - 2}px;")

        self.log_scroll = QScrollArea()
        self.log_scroll.setWidget(self.lbl_log)
        self.log_scroll.setWidgetResizable(True)
        self.log_scroll.setFrameShape(QFrame.NoFrame)
        self.log_scroll.setStyleSheet("background: transparent;")
        self.log_scroll.viewport().setStyleSheet("background: transparent;")
        self.log_scroll.setFixedHeight(cfg.get("overlay.log_height", 72))
        self.log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.log_scroll)

        self._log_visible = cfg.get("overlay.show_log", True)
        self._set_log_visible(self._log_visible)

        # Счётчик ожидания. Живёт в потоке UI, поэтому обычный QTimer,
        # а не отдельный поток: трогать виджеты из чужого потока нельзя.
        self._status_key = "idle"
        self._status_title = ""
        self._status_started = None
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#374151; background:#374151; max-height:1px;")
        return line

    def _apply_window_flags(self) -> None:
        cfg = self.cfg
        flags = Qt.FramelessWindowHint | Qt.Tool
        if cfg.get("overlay.always_on_top", True):
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        if cfg.get("overlay.no_activate", True):
            # Qt-уровневый аналог WS_EX_NOACTIVATE: клик по оверлею
            # не должен сворачивать игру.
            self.setAttribute(Qt.WA_ShowWithoutActivating, True)
            self.setFocusPolicy(Qt.NoFocus)

    def apply_native_flags(self) -> None:
        """Win32-стили. Вызывать только после show(): до этого нет HWND."""
        handle = self.winId()
        win32.apply_overlay_styles(
            handle,
            tool_window=self.cfg.get("overlay.tool_window", True),
            no_activate=self.cfg.get("overlay.no_activate", True),
            click_through=self.cfg.get("overlay.click_through", False),
        )
        if self.cfg.get("overlay.exclude_from_capture", True):
            win32.set_excluded_from_capture(handle, True)

    # ---------------------------------------------------------------- отрисовка
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)
        painter.fillPath(path, QColor(17, 24, 39, 225))
        painter.setPen(QColor(55, 65, 81, 255))
        painter.drawPath(path)

    # ------------------------------------------------- перетаскивание и размер
    # Ширина полосы у края, за которую окно тянется. Меньше — не
    # попасть мышью, больше — край начинает воровать клики у текста.
    RESIZE_MARGIN = 8
    MIN_WIDTH, MIN_HEIGHT = 320, 200

    def _edges_at(self, pos) -> tuple[bool, bool, bool, bool]:
        """Какие края окна под курсором: (слева, сверху, справа, снизу)."""
        margin = self.RESIZE_MARGIN
        return (pos.x() <= margin,
                pos.y() <= margin,
                pos.x() >= self.width() - margin,
                pos.y() >= self.height() - margin)

    @staticmethod
    def _cursor_for(edges) -> Qt.CursorShape:
        left, top, right, bottom = edges
        if (left and top) or (right and bottom):
            return Qt.SizeFDiagCursor
        if (right and top) or (left and bottom):
            return Qt.SizeBDiagCursor
        if left or right:
            return Qt.SizeHorCursor
        if top or bottom:
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        edges = self._edges_at(event.position().toPoint())
        if any(edges):
            # Тянем за край: запоминаем исходную геометрию, чтобы
            # считать от неё, а не от предыдущего кадра — иначе окно
            # уползает при быстром движении мыши.
            self._resize_edges = edges
            self._resize_origin = (event.globalPosition().toPoint(),
                                   self.geometry())
        else:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        point = event.position().toPoint()
        if self._resize_edges is not None and event.buttons() & Qt.LeftButton:
            self._resize_to(event.globalPosition().toPoint())
            return
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            return
        # Ничего не зажато — подсказываем курсором, что край тянется.
        self.setCursor(self._cursor_for(self._edges_at(point)))

    def _resize_to(self, point) -> None:
        left, top, right, bottom = self._resize_edges
        origin, rect = self._resize_origin
        delta = point - origin
        geometry = QRect(rect)
        if left:
            geometry.setLeft(rect.left() + delta.x())
        if right:
            geometry.setRight(rect.right() + delta.x())
        if top:
            geometry.setTop(rect.top() + delta.y())
        if bottom:
            geometry.setBottom(rect.bottom() + delta.y())

        # Упор в минимум делаем двигая ту же сторону, за которую тянут:
        # иначе при сжатии за левый край окно начинает ехать вправо.
        # Минимум берём фактический: раскладка со своими полями и
        # подписями не даёт сжаться до нашего числа, и Qt молча
        # растягивал окно обратно — противоположным краем.
        hint = self.minimumSizeHint()
        min_width = max(self.MIN_WIDTH, hint.width())
        min_height = max(self.MIN_HEIGHT, hint.height())
        if geometry.width() < min_width:
            if left:
                geometry.setLeft(geometry.right() - min_width)
            else:
                geometry.setRight(geometry.left() + min_width)
        if geometry.height() < min_height:
            if top:
                geometry.setTop(geometry.bottom() - min_height)
            else:
                geometry.setBottom(geometry.top() + min_height)
        self.setGeometry(geometry)

    def mouseReleaseEvent(self, event) -> None:
        moved = self._drag_offset is not None or self._resize_edges is not None
        self._drag_offset = None
        self._resize_edges = None
        self._publish_wheel_zone()
        if moved:
            # Только после перетаскивания или растягивания: иначе файл
            # переписывался бы на каждый клик по тексту ответа.
            self.save_geometry()

    def leaveEvent(self, event) -> None:
        # Курсор ушёл с окна — вернуть обычную стрелку, иначе она
        # остаётся «растягивающей» над игрой.
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    # ------------------------------------------------------------ колесо
    def set_wheel_zone_sink(self, sink) -> None:
        """Кому отдавать зону прокрутки (обычно HotkeyManager)."""
        self._wheel_zone_sink = sink
        self._publish_wheel_zone()

    def _publish_wheel_zone(self) -> None:
        """Сообщает хуку прямоугольник ленты в физических пикселях.

        Хук работает в экранных координатах Windows, а Qt отдаёт
        логические: при масштабе 125% зона иначе съезжает на четверть
        экрана и колесо перехватывается не там, где окно.
        """
        sink = self._wheel_zone_sink
        if sink is None:
            return
        if not self.isVisible():
            sink.set_wheel_zone(None)
            return
        ratio = self.devicePixelRatioF() or 1.0
        top_left = self.scroll.mapToGlobal(QPoint(0, 0))
        size = self.scroll.size()
        sink.set_wheel_zone((
            round(top_left.x() * ratio), round(top_left.y() * ratio),
            round((top_left.x() + size.width()) * ratio),
            round((top_left.y() + size.height()) * ratio)))

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._publish_wheel_zone()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._publish_wheel_zone()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._publish_wheel_zone()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._publish_wheel_zone()

    @Slot(int)
    def _on_scroll(self, steps: int) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.value() - steps * bar.singleStep() * 3)

    # ------------------------------------------------------------------- слоты
    @Slot(str, str)
    def _on_status(self, key: str, detail: str) -> None:
        color = STATUS_COLORS.get(key, STATUS_COLORS["idle"])
        title = STATUS_TITLES.get(key, key)
        # Ответ на CPU считается десятки секунд. Без счётчика непонятно,
        # думает ассистент или завис, и рука тянется нажать хоткей ещё
        # раз — а второй запрос на одном рабочем потоке только отодвинет
        # первый.
        self._status_key = key
        if key in ("thinking", "transcribing"):
            self._status_started = datetime.now()
            self._status_title = title
            self._tick.start(500)
        else:
            self._tick.stop()
        self.lbl_status.setStyleSheet(
            f"color:{color}; font-size:{self.cfg.get('overlay.font_size', 13)}px;"
            " font-weight:600;")
        self.lbl_status.setText(title)
        # Длинную подсказку обрезаем по ширине окна, а не переносим:
        # перенос отнимал у ответа целые строки.
        metrics = self.lbl_hint.fontMetrics()
        self.lbl_hint.setText(metrics.elidedText(
            detail, Qt.ElideRight, max(self.width() - 34, 80)))

    @Slot(str)
    def _on_mode(self, mode_title: str) -> None:
        self.lbl_mode.setText(mode_title)
        color = MODE_COLORS.get(mode_title, "#e5e7eb")
        self.lbl_mode.setStyleSheet(
            f"color:{color}; "
            f"font-size:{self.cfg.get('overlay.font_size', 13) + 2}px; "
            "font-weight:600;")

    def _on_tick(self) -> None:
        """Дописывает к статусу секунды ожидания."""
        if self._status_started is None:
            return
        seconds = (datetime.now() - self._status_started).total_seconds()
        self.lbl_status.setText(f"{self._status_title} {seconds:.0f} с")

    # Разделитель между записями ленты.
    SEPARATOR = "\n\n" + "─" * 34 + "\n\n"

    def _render(self) -> None:
        """Перерисовывает ленту, сохраняя место, где человек читает.

        Прокрутка вниз — только если человек и так внизу. Стоит ему
        отлистать вверх к прошлому вопросу, и дёргать обратно нельзя:
        ответ печатается по кускам, и лента прыгала бы каждые несколько
        миллисекунд.
        """
        bar = self.scroll.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 24

        parts = [x for x in self._entries + [self._answer_buffer] if x]
        self.lbl_answer.setTextFormat(Qt.PlainText)
        self.lbl_answer.setText(self.SEPARATOR.join(parts))

        if at_bottom:
            # Текст только что сменился, и maximum ещё не пересчитан —
            # ждём, пока Qt разложит разметку.
            QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    @Slot(str)
    def _on_answer(self, text: str) -> None:
        """Заменяет текущую запись: системные сообщения и ошибки."""
        self._answer_buffer = text
        self._render()

    @Slot(str)
    def _on_answer_chunk(self, chunk: str) -> None:
        self._answer_buffer += chunk
        self._render()

    def begin_answer(self) -> None:
        """Закрывает прошлую запись и начинает новую.

        Раньше здесь всё стиралось, и предыдущий вопрос было уже не
        посмотреть: ответ жил ровно до следующего нажатия PTT. Теперь
        записи копятся до сброса сессии, и колесом можно отлистать
        к любому.
        """
        if self._answer_buffer.strip():
            self._entries.append(self._answer_buffer.strip())
            del self._entries[:-self.MAX_ENTRIES]
        self._answer_buffer = ""
        self._render()

    def set_cases_tab(self, showing: bool) -> None:
        """Подпись вкладки: открыть список или вернуться к своим вопросам."""
        size = self.cfg.get("overlay.font_size", 13)
        text = "← Назад" if showing else "☰ Дела"
        self.lbl_cases.setText(
            f"<a href='cases' style='color:#93c5fd; text-decoration:none; "
            f"font-size:{size}px'>{text}</a>")

    def _on_link(self, href: str) -> None:
        """Клик по строке дела или по выбору роли."""
        if href.startswith("case:"):
            self.sig_case.emit(href[5:])
        elif href.startswith("role:"):
            self.sig_role.emit(href[5:])
        elif href.startswith("form:"):
            self.sig_form.emit(href[5:])

    def show_html(self, html: str) -> None:
        """Показывает разметку вместо обычного текста — список дел.

        Лента при этом не трогается: закрыл список, вернулся к своим
        вопросам и не потерял их.
        """
        self.lbl_answer.setTextFormat(Qt.RichText)
        self.lbl_answer.setText(html)
        self.scroll.verticalScrollBar().setValue(0)

    def show_transcript(self) -> None:
        """Возвращает обычный вид ленты после списка дел."""
        self.lbl_answer.setTextFormat(Qt.PlainText)
        self._render()

    @Slot(str, str)
    def replace_in_answer(self, old: str, new: str) -> None:
        """Заменяет кусок уже показанного ответа.

        Ответ печатается по мере генерации, а привести его в порядок
        можно только когда он дописан целиком. Перерисовываем то, что
        уже на экране, вместо того чтобы печатать второй раз.
        """
        if not old or old not in self._answer_buffer:
            return
        self._answer_buffer = self._answer_buffer.replace(old, new, 1)
        self._render()

    def clear_answers(self) -> None:
        """Очищает ленту — при сбросе сессии."""
        self._entries.clear()
        self._answer_buffer = ""
        self._render()

    @Slot(str, str)
    def _on_log(self, who: str, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        flat = " ".join(text.split())
        if len(flat) > 90:
            flat = flat[:89] + "…"
        self._log_lines.append(f"{stamp}  {who}: {flat}")
        del self._log_lines[:-self._max_log_lines]
        self.lbl_log.setText("\n".join(self._log_lines))
        # Свежая запись должна быть видна без ручной прокрутки.
        bar = self.log_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_log_visible(self, visible: bool) -> None:
        """Лог прячется вместе со своей чертой.

        Без этого от скрытого лога оставалась висящая линия, под
        которой ничего нет.
        """
        self._log_visible = visible
        self.log_scroll.setVisible(visible)
        self.log_separator.setVisible(visible)

    def toggle_log(self) -> None:
        """Прячет лог совсем — тогда ответу достаётся всё окно."""
        self._set_log_visible(not self._log_visible)

    def clear_log(self) -> None:
        self._log_lines.clear()
        self.lbl_log.setText("")

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.apply_native_flags()

    def set_click_through(self, enabled: bool) -> None:
        win32.apply_overlay_styles(
            self.winId(),
            tool_window=self.cfg.get("overlay.tool_window", True),
            no_activate=self.cfg.get("overlay.no_activate", True),
            click_through=enabled,
        )


def app_icon() -> QIcon:
    """Мордочка Tobi для окна и трея.

    Файл рисуется скриптом tools/make_icon.py. Если его нет — рисуем
    простой кружок на месте: отсутствие картинки не повод остаться без
    значка в трее, через который только и можно выйти.
    """
    path = Path(__file__).resolve().parent.parent / "assets" / "tobi.ico"
    if path.exists():
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon

    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(37, 99, 235))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, 28, 28)
    painter.setPen(QColor(255, 255, 255))
    font = painter.font()
    font.setBold(True)
    font.setPointSize(13)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "T")
    painter.end()
    return QIcon(pixmap)


def create_app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Tobi")
    app.setApplicationDisplayName("Tobi")
    app.setWindowIcon(app_icon())
    return app


def build_tray_icon(overlay: "OverlayWindow", on_quit,
                    on_import=None, on_help=None,
                    on_quit_all=None) -> QSystemTrayIcon | None:
    """Значок в трее: единственный видимый орган управления оверлеем.

    Окно у нас без рамки и без кнопки закрытия, в панели задач его тоже
    нет (WS_EX_TOOLWINDOW). Без значка выйти можно было бы только через
    Диспетчер задач.
    """
    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning("Системный трей недоступен, значок не создан")
        return None

    tray = QSystemTrayIcon(app_icon())
    tray.setToolTip("Tobi — помощник по законам Eclipse RP")

    menu = QMenu()
    action_toggle = menu.addAction("Показать / скрыть")
    action_toggle.triggered.connect(overlay.toggle_visibility)
    if on_help is not None:
        action_help = menu.addAction("Горячие клавиши")
        action_help.triggered.connect(
            lambda: (overlay.show(), overlay.apply_native_flags(), on_help()))
    if on_import is not None:
        action_import = menu.addAction("Импорт из буфера обмена")
        action_import.triggered.connect(on_import)
    menu.addSeparator()
    action_quit = menu.addAction("Выход")
    action_quit.triggered.connect(on_quit)

    if on_quit_all is not None:
        # Ollama живёт отдельным процессом и нарочно переживает выход:
        # иначе следующий запуск снова ждал бы загрузки модели. Но
        # когда человек закрывает бота «совсем», она остаётся держать
        # несколько гигабайт видеопамяти, и это неожиданно.
        action_quit_all = menu.addAction("Выход и остановить Ollama")
        action_quit_all.triggered.connect(on_quit_all)

    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: overlay.toggle_visibility()
        if reason == QSystemTrayIcon.Trigger else None)
    tray.show()
    # Меню живёт ровно столько, сколько значок: без ссылки его соберёт GC
    # и правый клик перестанет работать.
    tray._menu = menu
    return tray
