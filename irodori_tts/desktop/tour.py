"""A guided tour: the window is dimmed except the control being explained, a bubble with an arrow
points at it, and only that control (and the bubble) can be used until the user goes on.

QFluentWidgets provides the bubble (TeachingTip) but no mask; the mask is the overlay here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import QWidget
from qfluentwidgets import TeachingTip, TeachingTipTailPosition, TeachingTipView

from irodori_tts.desktop.messages import Message, t

from .widgets import ACCENT, button, ghost_button, row

MARGIN = 6
RING = 5  # the ring's distance from the hole plus half its width, kept inside the window
# The page switch animates for 220 ms; point at a control once it has arrived.
SETTLE_MS = 300


@dataclass(frozen=True)
class Step:
    target: Callable[[], QWidget | None]
    title: Message
    content: Message
    tail: TeachingTipTailPosition = TeachingTipTailPosition.LEFT
    before: Callable[[], None] | None = None


class Spotlight(QWidget):
    """Dims the window except a hole around the target. The hole is cut out of the widget's
    mask, so clicks there reach the control beneath; clicks elsewhere stop here."""

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.hole = QRect()
        self.target = None
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        window.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())
            self.follow()
        return False

    def focus(self, target: QWidget | None):
        self.target = target
        self.setGeometry(self.parent().rect())
        self.follow()
        self.show()
        self.raise_()

    def follow(self):
        if self.target is None or not self.target.isVisible():
            self.hole = QRect()
        else:
            origin = self.target.mapTo(self.parent(), self.target.rect().topLeft())
            hole = QRect(origin, self.target.size()).adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
            # Keep the hole and its ring inside the window: a control at the edge (the
            # navigation) would otherwise have its ring cut off.
            self.hole = hole.intersected(self.rect().adjusted(RING, RING, -RING, -RING))
        region = QRegion(self.rect())
        if not self.hole.isNull():
            region = region.subtracted(QRegion(self.hole))
        self.setMask(region)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if not self.hole.isNull():
            # A ring just outside the hole (the hole itself is not painted: it is masked out).
            ring = QPainterPath()
            ring.addRoundedRect(QRectF(self.hole).adjusted(-2, -2, 2, 2), 8, 8)
            painter.setPen(QPen(ACCENT, 3))
            painter.drawPath(ring)

    def mousePressEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        event.accept()


class Tour(QObject):
    finished = Signal()

    def __init__(self, window: QWidget, steps: list[Step]):
        super().__init__(window)
        self.window = window
        self.steps = steps
        self.index = -1
        self.tip = None
        self.spotlight = Spotlight(window)

    @property
    def active(self) -> bool:
        return self.index >= 0

    def start(self):
        self.close_tip()
        self.index = 0
        self.show_step()

    def show_step(self):
        step = self.steps[self.index]
        if step.before:
            step.before()
        QTimer.singleShot(SETTLE_MS, lambda index=self.index: self.point(index))

    def point(self, index):
        if index != self.index:
            return
        step = self.steps[index]
        target = step.target()
        if target is None or not target.isVisible():
            self.next()
            return
        self.spotlight.focus(target)
        last = index == len(self.steps) - 1
        view = TeachingTipView(
            str(t("tour.step", index=index + 1, total=len(self.steps), title=step.title)),
            str(step.content),
            isClosable=False,
            tailPosition=step.tail,
        )
        view.addWidget(
            row(
                ghost_button(t("tour.skip"), self.finish),
                None,
                button(t("tour.done") if last else t("tour.next"), self.next, primary=True),
            )
        )
        self.tip = TeachingTip.make(
            view, target, duration=-1, tailPosition=step.tail, parent=self.window
        )

    def close_tip(self):
        if self.tip is not None:
            self.tip.close()
            self.tip = None

    def next(self):
        self.close_tip()
        self.index += 1
        if self.index >= len(self.steps):
            self.finish()
        else:
            self.show_step()

    def finish(self):
        self.close_tip()
        self.index = -1
        self.spotlight.hide()
        self.finished.emit()
