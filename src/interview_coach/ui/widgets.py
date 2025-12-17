from __future__ import annotations

from collections import deque

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class Sparkline(QWidget):
    def __init__(self, max_points: int = 180, parent=None):
        super().__init__(parent)
        self._values: deque[float] = deque(maxlen=max_points)
        self.setMinimumHeight(48)

    def add(self, v: float) -> None:
        self._values.append(float(v))
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        painter.fillRect(rect, QColor("#0f172a"))
        if len(self._values) < 2:
            return

        vals = list(self._values)
        vmin = min(vals)
        vmax = max(vals)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        w = rect.width()
        h = rect.height()
        xs = [rect.left() + int(i * (w - 2) / (len(vals) - 1)) + 1 for i in range(len(vals))]
        ys = [rect.bottom() - int((v - vmin) / (vmax - vmin) * (h - 6)) - 3 for v in vals]

        pen = QPen(QColor("#38bdf8"))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(1, len(xs)):
            painter.drawLine(xs[i - 1], ys[i - 1], xs[i], ys[i])

        painter.setPen(QPen(QColor("#334155")))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))


class Toast(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            Toast {
              background: rgba(15, 23, 42, 230);
              border: 1px solid rgba(148, 163, 184, 120);
              border-radius: 10px;
            }
            """
        )
        self.hide()
        self._text = ""

    def show_message(self, text: str) -> None:
        self._text = text.strip()
        self.setToolTip(self._text)
        self.setAccessibleName(self._text)
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(15, 23, 42, 230))
        painter.setPen(QPen(QColor("#e2e8f0")))
        painter.drawText(self.rect().adjusted(12, 8, -12, -8), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)

