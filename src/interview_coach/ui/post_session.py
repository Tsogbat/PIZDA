from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interview_coach.analytics import coaching_recommendations, summarize_session
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
            """
        )
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel("Session Summary")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        summary = summarize_session(self._session)

        grid_frame = QFrame()
        grid_frame.setStyleSheet("QFrame { background: #0f172a; border: 1px solid #243047; border-radius: 12px; }")
        grid = QGridLayout(grid_frame)
        grid.addWidget(_kv("Duration (s)", _fmt(summary.get("duration_s"))), 0, 0)
        grid.addWidget(_kv("Avg Confidence", _fmt(summary.get("confidence_avg"))), 0, 1)
        grid.addWidget(_kv("Min / Max", f"{_fmt(summary.get('confidence_min'))} / {_fmt(summary.get('confidence_max'))}"), 1, 0)
        grid.addWidget(_kv("Avg Eye Contact", _fmt(summary.get("eye_contact_avg"))), 1, 1)
        grid.addWidget(_kv("Avg WPM", _fmt(summary.get("speech_wpm_avg"))), 2, 0)
        grid.addWidget(_kv("Fillers (/min)", _fmt(summary.get("fillers_per_min_avg"))), 2, 1)
        grid.addWidget(_kv("Long Pauses (total)", _fmt(summary.get("pause_count_total"))), 3, 0)
        grid.addWidget(_kv("Long Pauses (/min)", _fmt(summary.get("pause_per_min"))), 3, 1)
        grid.addWidget(_kv("Dominant Face", str(summary.get("dominant_face_emotion") or "-")), 4, 0)
        grid.addWidget(_kv("Dominant Speech", str(summary.get("dominant_speech_emotion") or "-")), 4, 1)
        grid.addWidget(
            _kv(
                "Vision Latency (avg/p95 ms)",
                f"{_fmt(summary.get('vision_latency_ms_avg'))} / {_fmt(summary.get('vision_latency_ms_p95'))}",
            ),
            5,
            0,
        )
        grid.addWidget(
            _kv(
                "Audio Latency (avg/p95 ms)",
                f"{_fmt(summary.get('audio_latency_ms_avg'))} / {_fmt(summary.get('audio_latency_ms_p95'))}",
            ),
            5,
            1,
        )
        grid.addWidget(
            _kv(
                "Fusion Latency (avg/p95 ms)",
                f"{_fmt(summary.get('fusion_latency_ms_avg'))} / {_fmt(summary.get('fusion_latency_ms_p95'))}",
            ),
            6,
            0,
        )
        root.addWidget(grid_frame)

        rec_title = QLabel("General Feedback")
        rec_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        root.addWidget(rec_title)
        rec = QTextEdit()
        rec.setReadOnly(True)
        bullets = "\n".join(f"- {r}" for r in coaching_recommendations(self._session))
        rec.setPlainText(bullets)
        rec.setMinimumHeight(110)
        root.addWidget(rec)

        charts_title = QLabel("Trends")
        charts_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        root.addWidget(charts_title)

        charts = QFrame()
        charts.setStyleSheet("QFrame { background: #0f172a; border: 1px solid #243047; border-radius: 12px; }")
        charts_layout = QGridLayout(charts)

        conf = Sparkline(max_points=240)
        eye = Sparkline(max_points=240)
        wpm = Sparkline(max_points=240)
        filler = Sparkline(max_points=240)

        for s in self._session.get("samples") or []:
            conf.add(s.get("confidence_0_100") or 0.0)
            eye.add((s.get("eye_contact") or 0.0) * 100.0)
            wpm.add(s.get("speech_wpm") or 0.0)
            filler.add(s.get("filler_per_min") or 0.0)

        charts_layout.addWidget(_chart("Confidence", conf), 0, 0)
        charts_layout.addWidget(_chart("Eye Contact %", eye), 0, 1)
        charts_layout.addWidget(_chart("WPM", wpm), 1, 0)
        charts_layout.addWidget(_chart("Fillers/min", filler), 1, 1)
        root.addWidget(charts, 1)

        export_title = QLabel("Exports")
        export_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        root.addWidget(export_title)

        export_row = QHBoxLayout()
        export_row.addWidget(QLabel(f"JSON: {self._json_path}"))
        export_row.addWidget(QLabel(f"CSV: {self._csv_path}"))
        export_row.addStretch(1)
        root.addLayout(export_row)

        transcript_title = QLabel("Transcript")
        transcript_title.setStyleSheet("font-size: 16px; font-weight: 650; margin-top: 8px;")
        root.addWidget(transcript_title)
        transcript = QTextEdit()
        transcript.setReadOnly(True)
        transcript.setPlainText(self._session.get("transcript") or "")
        root.addWidget(transcript)


def _kv(k: str, v: str) -> QWidget:
    w = QWidget()
    w.setStyleSheet("QLabel { color: #e2e8f0; }")
    layout = QVBoxLayout(w)
    kk = QLabel(k)
    kk.setStyleSheet("font-size: 12px; color: #cbd5e1;")
    vv = QLabel(v)
    vv.setStyleSheet("font-size: 20px; font-weight: 700;")
    layout.addWidget(kk)
    layout.addWidget(vv)
    return w


def _chart(title: str, widget: QWidget) -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)
    t = QLabel(title)
    t.setStyleSheet("font-size: 12px; color: #cbd5e1;")
    layout.addWidget(t)
    layout.addWidget(widget)
    return w


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)
