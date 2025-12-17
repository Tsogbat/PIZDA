from __future__ import annotations

import math
from pathlib import Path

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interview_coach.analytics import coaching_report_text
from interview_coach.ui.widgets import Sparkline


class PostSessionView(QWidget):
    def __init__(self, session_data: dict, json_path: Path, csv_path: Path, parent=None):
        super().__init__(parent)
        self._session = session_data
        self._json_path = Path(json_path)
        self._csv_path = Path(csv_path)
        self.setWindowTitle("Post-Session Analytics")
        self.setMinimumSize(920, 620)
        self.setStyleSheet(
            """
            QWidget { background: #0b1220; color: #e2e8f0; }
            QTextEdit { background: #0f172a; border: 1px solid #243047; border-radius: 10px; padding: 10px; }
            QFrame#Card { background: #0f172a; border: 1px solid #243047; border-radius: 12px; }
            """
        )
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel("Post-Session Analytics")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)

        rec_title = QLabel("General Feedback")
        rec_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        report_layout.addWidget(rec_title)
        rec = QTextEdit()
        rec.setReadOnly(True)
        rec.setPlainText(coaching_report_text(self._session))
        rec.setMinimumHeight(220)
        report_layout.addWidget(rec)

        export_title = QLabel("Exports")
        export_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        report_layout.addWidget(export_title)

        export_row = QHBoxLayout()
        export_row.addWidget(QLabel(f"JSON: {self._json_path}"))
        export_row.addWidget(QLabel(f"CSV: {self._csv_path}"))
        export_row.addStretch(1)
        report_layout.addLayout(export_row)

        transcript_title = QLabel("Transcript")
        transcript_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        report_layout.addWidget(transcript_title)
        transcript = QTextEdit()
        transcript.setReadOnly(True)
        transcript.setPlainText(self._session.get("transcript") or "")
        report_layout.addWidget(transcript, 1)

        tabs.addTab(report_tab, "Report")

        trends_tab = QWidget()
        self._build_trends(trends_tab)
        tabs.addTab(trends_tab, "Trends")

    def _build_trends(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        hint = QLabel("Time-series charts from recorded session samples.")
        hint.setStyleSheet("color: #cbd5e1;")
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        scroll.setWidget(body)

        samples = self._session.get("samples") or []
        confidence = [float(s.get("confidence_0_100")) for s in samples if s.get("confidence_0_100") is not None]
        eye = [float(s.get("eye_contact")) * 100.0 for s in samples if s.get("eye_contact") is not None]
        wpm = [float(s.get("speech_wpm")) for s in samples if s.get("speech_wpm") is not None]
        fillers = [float(s.get("filler_per_min")) for s in samples if s.get("filler_per_min") is not None]
        pauses = [float(s.get("pause_count")) for s in samples if s.get("pause_count") is not None]

        grid.addWidget(self._trend_card("Confidence (0–100)", confidence), 0, 0, 1, 2)
        grid.addWidget(self._trend_card("Eye Contact (%)", eye), 1, 0)
        grid.addWidget(self._trend_card("Speech Rate (WPM)", wpm), 1, 1)
        grid.addWidget(self._trend_card("Fillers (/min)", fillers), 2, 0)
        grid.addWidget(self._trend_card("Long Pauses (count)", pauses), 2, 1)
        grid.setRowStretch(3, 1)

    def _trend_card(self, title: str, values: list[float]) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumHeight(220)
        layout = QVBoxLayout(card)

        t = QLabel(title)
        t.setStyleSheet("font-size: 13px; color: #cbd5e1;")
        layout.addWidget(t)

        if not values:
            empty = QLabel("No data recorded for this signal.")
            empty.setStyleSheet("color: #94a3b8;")
            layout.addWidget(empty, 1)
            return card

        values = _downsample(values, 600)
        spark = Sparkline(max_points=max(60, len(values)))
        spark.setMinimumHeight(150)
        spark.extend(values)
        layout.addWidget(spark, 1)
        return card


def _downsample(values: list[float], max_points: int) -> list[float]:
    if max_points <= 0:
        return []
    n = len(values)
    if n <= max_points:
        return values
    stride = int(math.ceil(n / float(max_points)))
    return values[:: max(1, stride)]
