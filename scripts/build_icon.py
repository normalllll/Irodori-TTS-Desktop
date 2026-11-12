"""Render irodori_tts/desktop/assets/icon.ico from the two SVG drawings.

32 px and below use icon-small.svg (three thick bars stay apart where five would blur);
larger sizes use icon.svg. Run after editing either drawing:

    .venv/Scripts/python.exe -m scripts.build_icon
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ASSETS = Path(__file__).resolve().parents[1] / "irodori_tts/desktop/assets"
SMALL = (16, 20, 24, 32)
LARGE = (40, 48, 64, 128, 256)


def render(svg: Path, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(str(svg)).render(painter, QRectF(0, 0, size, size))
    painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def main():
    QGuiApplication([])
    images = [render(ASSETS / "icon-small.svg", size) for size in SMALL]
    images += [render(ASSETS / "icon.svg", size) for size in LARGE]
    largest = images[-1]
    largest.save(
        ASSETS / "icon.ico",
        sizes=[image.size for image in images],
        append_images=images[:-1],
    )
    print(f"wrote {ASSETS / 'icon.ico'}: {', '.join(str(s) for s in SMALL + LARGE)} px")


if __name__ == "__main__":
    main()
