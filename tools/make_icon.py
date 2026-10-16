"""
Иконка ассистента: мордочка Tobi.

Рисуется кодом, а не лежит картинкой в репозитории: так её можно
поправить, не заводя графический редактор, и видно, из чего она
собрана. Прототип — собака хозяина: чёрно-белый окрас, большие
стоячие уши, чёрная маска на половину морды с белой проточиной
посередине.

Запуск: python tools/make_icon.py
Результат: assets/tobi.ico (многоразмерный) и assets/tobi.png.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF, QRectF, Qt, QBuffer, QByteArray
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath,
                           QPen, QRadialGradient)
from PySide6.QtWidgets import QApplication

# Рисуем крупно и уменьшаем: так края выходят гладкими на всех размерах.
CANVAS = 1024

# Размеры внутри .ico. 16 и 32 нужны для списка файлов и панели задач,
# 256 — для крупных значков рабочего стола.
SIZES = (16, 24, 32, 48, 64, 128, 256)

BLACK = QColor("#23201f")      # не чистый чёрный: на тёмном фоне он тонет
WHITE = QColor("#fbfaf8")
PINK = QColor("#e8a9a9")
BACKDROP_TOP = QColor("#7dd3fc")
BACKDROP_BOTTOM = QColor("#3b82f6")


def _ear(painter: QPainter, points, colour: QColor) -> None:
    """Большое стоячее ухо — самая узнаваемая черта Tobi."""
    tip, base_left, base_right, bend_l, bend_r = points
    # Кончик скруглён: острый угол делает ухо кошачьим, а у Tobi оно
    # крупное и мягкое. Ведём двумя дугами через две точки у вершины.
    tip_l = QPointF(tip.x() * 0.82 + bend_l.x() * 0.18,
                    tip.y() * 0.90 + bend_l.y() * 0.10)
    tip_r = QPointF(tip.x() * 0.82 + bend_r.x() * 0.18,
                    tip.y() * 0.90 + bend_r.y() * 0.10)
    path = QPainterPath(base_left)
    path.quadTo(bend_l, tip_l)
    path.quadTo(tip, tip_r)
    path.quadTo(bend_r, base_right)
    path.closeSubpath()
    painter.setBrush(QBrush(colour))
    painter.setPen(Qt.NoPen)
    painter.drawPath(path)


def _inner_ear(painter: QPainter, points) -> None:
    tip, base_left, base_right, _, _ = points
    cx = (base_left.x() + base_right.x()) / 2
    cy = (base_left.y() + base_right.y()) / 2
    inner_tip = QPointF(tip.x() * 0.52 + cx * 0.48, tip.y() * 0.52 + cy * 0.48)
    left = QPointF(base_left.x() * 0.22 + cx * 0.78,
                   base_left.y() * 0.22 + cy * 0.78)
    right = QPointF(base_right.x() * 0.22 + cx * 0.78,
                    base_right.y() * 0.22 + cy * 0.78)
    path = QPainterPath(left)
    path.quadTo(QPointF((left.x() + inner_tip.x()) / 2 - 14,
                        (left.y() + inner_tip.y()) / 2), inner_tip)
    path.quadTo(QPointF((right.x() + inner_tip.x()) / 2 + 14,
                        (right.y() + inner_tip.y()) / 2), right)
    path.closeSubpath()
    painter.setBrush(QBrush(PINK))
    painter.setPen(Qt.NoPen)
    painter.drawPath(path)


def _eye(painter: QPainter, centre: QPointF, radius: float) -> None:
    # Глаз сидит в чёрном пятне, поэтому ему нужен светлый ободок —
    # без него на мелком размере от глаза остаётся только блик.
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(255, 255, 255, 70), 7))
    painter.drawEllipse(centre, radius + 4, radius + 4)

    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor("#4a342a")))       # тёплая радужка
    painter.drawEllipse(centre, radius, radius)
    painter.setBrush(QBrush(QColor("#17120f")))
    painter.drawEllipse(centre, radius * 0.62, radius * 0.62)

    painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
    painter.drawEllipse(
        QPointF(centre.x() - radius * 0.30, centre.y() - radius * 0.34),
        radius * 0.30, radius * 0.30)
    painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
    painter.drawEllipse(
        QPointF(centre.x() + radius * 0.32, centre.y() + radius * 0.30),
        radius * 0.13, radius * 0.13)


def draw(size: int = CANVAS) -> QImage:
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(size / 1024.0, size / 1024.0)

    # --- фон: мягкий круг, чтобы иконка читалась на любых обоях ---
    backdrop = QRadialGradient(QPointF(512, 400), 560)
    backdrop.setColorAt(0.0, BACKDROP_TOP)
    backdrop.setColorAt(1.0, BACKDROP_BOTTOM)
    painter.setBrush(QBrush(backdrop))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QRectF(24, 24, 976, 976))

    # --- уши: широкие у основания, слегка развёрнуты наружу ---
    left_ear = (QPointF(196, 150), QPointF(232, 486), QPointF(436, 344),
                QPointF(178, 330), QPointF(330, 214))
    right_ear = (QPointF(828, 150), QPointF(588, 344), QPointF(792, 486),
                 QPointF(694, 214), QPointF(846, 330))
    for ear in (left_ear, right_ear):
        _ear(painter, ear, BLACK)
        _inner_ear(painter, ear)

    # --- голова: овал, вытянутый книзу, с выступающей мордой ---
    head = QRectF(226, 286, 572, 566)
    painter.setBrush(QBrush(WHITE))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(head)
    muzzle = QRectF(380, 566, 264, 286)
    painter.drawEllipse(muzzle)

    # --- окрас: слева маска до уха, справа пятно вокруг глаза,
    # между ними белая проточина, расширяющаяся к носу.
    painter.save()
    clip = QPainterPath()
    clip.addEllipse(head)
    painter.setClipPath(clip)
    painter.setBrush(QBrush(BLACK))

    left_mask = QPainterPath()
    left_mask.moveTo(QPointF(226, 286))
    left_mask.lineTo(QPointF(486, 300))
    left_mask.quadTo(QPointF(452, 448), QPointF(430, 596))
    left_mask.quadTo(QPointF(318, 650), QPointF(226, 556))
    left_mask.closeSubpath()
    painter.drawPath(left_mask)

    # Правое пятно меньше и не доходит до низа — как на фотографии.
    right_mask = QPainterPath()
    right_mask.moveTo(QPointF(576, 302))
    right_mask.lineTo(QPointF(798, 330))
    right_mask.quadTo(QPointF(800, 500), QPointF(712, 568))
    right_mask.quadTo(QPointF(612, 530), QPointF(590, 420))
    right_mask.closeSubpath()
    painter.drawPath(right_mask)

    # Белое пятнышко на лбу — оно есть у Tobi и оживляет симметрию.
    painter.setBrush(QBrush(WHITE))
    painter.drawEllipse(QPointF(470, 356), 17, 22)
    painter.restore()

    # --- глаза ---
    _eye(painter, QPointF(396, 500), 60)
    _eye(painter, QPointF(646, 498), 60)

    # --- нос и рот ---
    nose = QPainterPath()
    nose.moveTo(QPointF(512, 664))
    nose.quadTo(QPointF(432, 636), QPointF(466, 688))
    nose.quadTo(QPointF(490, 718), QPointF(512, 738))
    nose.quadTo(QPointF(534, 718), QPointF(558, 688))
    nose.quadTo(QPointF(592, 636), QPointF(512, 664))
    painter.setBrush(QBrush(BLACK))
    painter.setPen(Qt.NoPen)
    painter.drawPath(nose)

    painter.setPen(QPen(BLACK, 14, Qt.SolidLine, Qt.RoundCap))
    painter.setBrush(Qt.NoBrush)
    stem = QPainterPath(QPointF(512, 738))
    stem.lineTo(QPointF(512, 772))
    painter.drawPath(stem)
    left_lip = QPainterPath(QPointF(512, 772))
    left_lip.quadTo(QPointF(468, 802), QPointF(430, 764))
    painter.drawPath(left_lip)
    right_lip = QPainterPath(QPointF(512, 772))
    right_lip.quadTo(QPointF(556, 802), QPointF(594, 764))
    painter.drawPath(right_lip)

    painter.end()
    return image


def _png_bytes(image: QImage) -> bytes:
    # Ссылку на QByteArray держим сама: временный объект Qt уничтожает
    # сразу, и буфер пишет в освобождённую память — процесс падает
    # молча, без исключения.
    store = QByteArray()
    buffer = QBuffer(store)
    buffer.open(QBuffer.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(store)


def write_ico(path: Path, images: list[QImage]) -> None:
    """Собирает многоразмерный .ico вручную.

    Qt пишет .ico одним размером, а Windows берёт разные: 16 пикселей
    в списке файлов и 256 на рабочем столе. Формат простой — заголовок
    и по записи на размер, — поэтому собираем сами, без Pillow.
    """
    blobs = [_png_bytes(image) for image in images]
    header = struct.pack("<HHH", 0, 1, len(blobs))
    offset = len(header) + 16 * len(blobs)
    entries, payload = b"", b""
    for image, blob in zip(images, blobs):
        side = image.width()
        entries += struct.pack(
            "<BBBBHHII",
            0 if side >= 256 else side,      # 0 означает 256
            0 if side >= 256 else side,
            0, 0, 1, 32, len(blob), offset)
        payload += blob
        offset += len(blob)
    path.write_bytes(header + entries + payload)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    assets = Path(__file__).resolve().parent.parent / "assets"
    assets.mkdir(exist_ok=True)

    master = draw(CANVAS)
    master.save(str(assets / "tobi.png"), "PNG")

    images = [master.scaled(side, side, Qt.KeepAspectRatio,
                            Qt.SmoothTransformation) for side in SIZES]
    write_ico(assets / "tobi.ico", images)

    print(f"Иконка готова: {assets / 'tobi.ico'}")
    print(f"Размеры внутри: {', '.join(str(s) for s in SIZES)}")
    print(f"Картинка для просмотра: {assets / 'tobi.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
