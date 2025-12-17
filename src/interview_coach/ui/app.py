from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interview_coach.config import AppConfig
from interview_coach.fusion import FusionEngine
from interview_coach.interviewer import Interviewer
from interview_coach.session import SessionRecorder, SessionSample
from interview_coach.ui.widgets import Sparkline, Toast
from interview_coach.tts import TextToSpeech
from interview_coach.vision.worker import VisionWorker
from interview_coach.audio.worker import AudioWorker
from interview_coach.ui.post_session import PostSessionView


def run_app(cfg: AppConfig) -> int:
    app = QApplication([])
    app.setApplicationName("Interview Coach")
    app.setStyle("Fusion")

    w = MainWindow(cfg)
    w.show()
    return app.exec()


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self._cfg = cfg
        self._fusion = FusionEngine(cfg.fusion)
        self._interviewer = Interviewer(cfg.interview, cfg.ollama)
        self._tts = TextToSpeech(cfg.tts)

        self._vision = VisionWorker(cfg.vision, cfg.models)
        self._audio = AudioWorker(cfg.audio, cfg.models.vosk_model_dir, cfg.ollama)
        self._session = SessionRecorder()
        self._last_sample_rel_s = 0.0
        self._post: PostSessionView | None = None

        self._last_hint_s = 0.0
        self._last_hint = ""
        self._transcript_last = ""
        self._session_active = False
        self._session_started_wall_s: float | None = None
        self._last_device_hint_s = 0.0
        self._last_ollama_hint_s = 0.0
        self._recorded_question_ids: set[str] = set()
        self._active_question_id = "waiting"
        self._active_question_index = -1

        self.setWindowTitle("Real-Time Interview Coach")
        self.setMinimumSize(1200, 720)
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS UI refresh
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setStyleSheet(
            """
            QWidget { background: #0b1220; color: #e2e8f0; }
            QLabel#Title { font-size: 18px; font-weight: 700; }
            QLabel#Kpi { font-size: 13px; color: #cbd5e1; }
            QLabel#KpiValue { font-size: 22px; font-weight: 700; }
            QTextEdit { background: #0f172a; border: 1px solid #243047; border-radius: 10px; padding: 10px; }
            QFrame#Card { background: #0f172a; border: 1px solid #243047; border-radius: 12px; }
            QPushButton { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 8px 12px; }
            QPushButton:hover { background: #273449; }
            QPushButton:pressed { background: #334155; }
            QProgressBar { background: #0f172a; border: 1px solid #243047; border-radius: 10px; height: 20px; text-align: center; }
            QProgressBar::chunk { background: #38bdf8; border-radius: 10px; }
            """
        )
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        header = QHBoxLayout()
        title = QLabel("Real-Time Multimodal Interview Coach")
        title.setObjectName("Title")
        header.addWidget(title, 1)

        self._btn_start = QPushButton("Start Interview")
        self._btn_start.clicked.connect(self._on_start)
        self._btn_next = QPushButton("Next Question")
        self._btn_next.clicked.connect(self._on_next)
        self._btn_end = QPushButton("End Session")
        self._btn_end.clicked.connect(self._on_end)
        self._btn_next.setEnabled(False)
        self._btn_end.setEnabled(False)
        header.addWidget(self._btn_start)
        header.addWidget(self._btn_next)
        header.addWidget(self._btn_end)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: camera preview card
        left = QFrame()
        left.setObjectName("Card")
        left_layout = QVBoxLayout(left)
        self._video = QLabel("Camera preview")
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setMinimumHeight(480)
        self._video.setStyleSheet("background: #000000; border-radius: 10px;")
        left_layout.addWidget(self._video)

        self._question = QLabel("Press Start Interview to begin.")
        self._question.setWordWrap(True)
        self._question.setStyleSheet("font-size: 16px; font-weight: 650;")
        left_layout.addWidget(self._question)

        splitter.addWidget(left)

        # Right: metrics and transcript
        right = QFrame()
        right.setObjectName("Card")
        grid = QGridLayout(right)

        self._kpi_eye = _kpi("Eye Contact", "0%")
        self._kpi_emotion = _kpi("Facial Emotion", "neutral")
        self._kpi_wpm = _kpi("Speech Rate (WPM)", "0")
        self._kpi_fillers = _kpi("Fillers (count /min)", "0 (0.0/min)")
        self._kpi_pauses = _kpi("Long Pauses (count)", "0")
        self._kpi_speech_emotion = _kpi("Speech Emotion", "neutral")

        grid.addWidget(self._kpi_eye, 0, 0)
        grid.addWidget(self._kpi_emotion, 0, 1)
        grid.addWidget(self._kpi_wpm, 1, 0)
        grid.addWidget(self._kpi_fillers, 1, 1)
        grid.addWidget(self._kpi_pauses, 2, 0)
        grid.addWidget(self._kpi_speech_emotion, 2, 1)

        conf_card = QFrame()
        conf_card.setObjectName("Card")
        conf_layout = QVBoxLayout(conf_card)
        conf_title = QLabel("Confidence (0–100)")
        conf_title.setObjectName("Kpi")
        conf_layout.addWidget(conf_title)
        self._confidence = QProgressBar()
        self._confidence.setRange(0, 100)
        self._confidence.setValue(0)
        conf_layout.addWidget(self._confidence)
        self._spark = Sparkline(max_points=240)
        conf_layout.addWidget(self._spark)
        grid.addWidget(conf_card, 3, 0, 1, 2)

        transcript_title = QLabel("Live Transcript")
        transcript_title.setObjectName("Kpi")
        grid.addWidget(transcript_title, 4, 0, 1, 2)
        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setMinimumHeight(220)
        grid.addWidget(self._transcript, 5, 0, 1, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)

        layout.addWidget(splitter, 1)

        self._toast = Toast(root)
        self._toast.setFixedHeight(44)
        self._toast.setFixedWidth(460)
        self._toast.move(18, 56)

        self._status = QLabel("")
        self._status.setObjectName("Kpi")
        layout.addWidget(self._status)

    def _on_start(self) -> None:
        self._spark.clear()
        self._confidence.setValue(0)
        self._transcript.clear()
        self._transcript_last = ""
        self._toast.hide()
        self._last_hint = ""
        self._last_hint_s = 0.0

        self._vision.reset()
        self._audio.reset()
        self._session.start()
        self._recorded_question_ids = set()
        self._active_question_id = "waiting"
        self._active_question_index = -1

        q = self._interviewer.start()
        self._set_question(q, transcript_text="")

        self._vision.start()
        self._audio.start()
        self._session_active = True
        self._session_started_wall_s = time.time()
        self._btn_start.setEnabled(False)
        self._btn_end.setEnabled(True)

    def _on_next(self) -> None:
        q_now = self._interviewer.current()
        if not q_now.ready:
            return
        q = self._interviewer.next()
        if q.id == "done":
            self._question.setText(q.text)
            self._btn_next.setEnabled(False)
            return
        current_transcript = (self._audio.latest().transcript_text if self._audio.latest() else "")
        self._set_question(q, transcript_text=current_transcript)

    def _on_end(self) -> None:
        self._btn_next.setEnabled(False)
        self._btn_end.setEnabled(False)
        self._btn_start.setEnabled(True)
        self._vision.stop()
        self._audio.stop()
        self._tts.stop()
        self._session_active = False
        self._session_started_wall_s = None
        final_transcript = (self._audio.latest().transcript_text if self._audio.latest() else "")
        self._session.end(final_transcript)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = Path("reports") / f"session_{stamp}.json"
        csv_path = Path("reports") / f"session_{stamp}.csv"
        self._session.export_json(json_path)
        self._session.export_csv(csv_path)

        self._question.setText("Session ended. Review post-session analytics.")
        self._post = PostSessionView(self._session.to_dict(), json_path, csv_path)
        self._post.show()
        self._post.raise_()
        self._post.activateWindow()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        try:
            self._vision.stop()
            self._audio.stop()
            self._tts.stop()
            self._interviewer.stop()
        finally:
            return super().closeEvent(event)

    def _tick(self) -> None:
        if self._session_active:
            q_ready = self._interviewer.poll()
            if q_ready is not None:
                current_transcript = (self._audio.latest().transcript_text if self._audio.latest() else "")
                self._set_question(q_ready, transcript_text=current_transcript)

        v = self._vision.latest()
        a = self._audio.latest()

        if v is not None:
            self._video.setPixmap(_to_pixmap(v.frame_bgr, self._video.width(), self._video.height()))
            self._kpi_eye.findChild(QLabel, "KpiValue").setText(f"{int(v.result.eye_contact * 100):d}%")
            self._kpi_emotion.findChild(QLabel, "KpiValue").setText(v.result.emotion)
        elif self._session_active:
            self._video.clear()
            self._video.setText("No camera frames")

        if a is not None:
            self._kpi_wpm.findChild(QLabel, "KpiValue").setText(f"{a.wpm:.0f}")
            self._kpi_fillers.findChild(QLabel, "KpiValue").setText(f"{a.filler_count} ({a.filler_per_min:.1f}/min)")
            pause_txt = f"{a.pause_count}"
            if a.last_pause_s is not None:
                pause_txt = f"{a.pause_count} (last {a.last_pause_s:.1f}s)"
            self._kpi_pauses.findChild(QLabel, "KpiValue").setText(pause_txt)
            emo_conf = max(a.speech_emotion_scores.values()) if a.speech_emotion_scores else 0.0
            emo_src = "LLM" if getattr(a, "speech_emotion_source", "heuristic") == "ollama" else "heur"
            self._kpi_speech_emotion.findChild(QLabel, "KpiValue").setText(f"{a.speech_emotion} ({emo_conf:.2f}, {emo_src})")

            full = a.transcript_text
            if full != self._transcript_last:
                self._transcript_last = full
                self._transcript.setPlainText((full + ("\n\n" + a.partial_text if a.partial_text else "")).strip())
            elif a.partial_text:
                self._transcript.setPlainText((full + "\n\n" + a.partial_text).strip())

        fr = self._fusion.fuse(v.result if v else None, a if a else None)
        if self._session_active:
            self._confidence.setValue(int(fr.confidence_0_100))
            self._spark.add(fr.confidence_0_100)
            if fr.hints:
                self._maybe_toast(fr.hints[0])
            self._maybe_record_sample(v, a, fr)

        self._update_status(v, a, fr)
        self._maybe_device_hints(v, a)
        self._maybe_ollama_hints()

    def _set_question(self, q, transcript_text: str) -> None:
        if q.id == "done":
            self._question.setText(q.text)
            self._btn_next.setEnabled(False)
            return

        idx = max(0, int(self._interviewer.index))
        src = getattr(q, "source", "")
        suffix = ""
        if src == "llm":
            suffix = "  (LLM)"
        elif src == "warmup":
            suffix = "  (warm-up)"
        elif src == "fallback":
            suffix = "  (fallback)"
        elif src == "predefined" and bool(getattr(self._cfg.interview, "use_llm_questions", False)):
            suffix = "  (fallback)"
        self._question.setText(f"Q{idx + 1}/{self._interviewer.total}: {q.text}{suffix}")
        self._btn_next.setEnabled(bool(q.ready) and (not self._interviewer.finished))

        if not q.ready:
            return

        if q.id not in self._recorded_question_ids:
            self._recorded_question_ids.add(q.id)
            self._active_question_id = q.id
            self._active_question_index = idx
            self._session.add_question_event(q.id, idx, q.text, transcript_text)
            self._speak_question(q.text)
            self._interviewer.prefetch()

    def _speak_question(self, text: str) -> None:
        dur_s = self._tts.speak(text)
        if dur_s > 0:
            self._audio.hold(dur_s)

    def _update_status(self, v, a, fr) -> None:
        parts = []
        if self._session_active:
            parts.append("Session: running")
        if v is None:
            parts.append("Camera: unavailable")
        else:
            parts.append(f"Camera: ok ({v.result.latency_ms:.0f}ms)")
        if a is None:
            parts.append("Mic: unavailable")
        else:
            stt = "ok" if a.transcript_text or a.partial_text else "listening"
            parts.append(f"Mic: ok ({stt}, {a.latency_ms:.0f}ms)")
        parts.append(self._ollama_status_text())
        parts.append(f"Fusion: {fr.latency_ms:.0f}ms")
        self._status.setText(" | ".join(parts))

    def _ollama_status_text(self) -> str:
        q = self._interviewer.ollama_status()
        e = self._audio.ollama_status()
        enabled = bool(q.get("enabled")) or bool(e.get("enabled"))
        if not enabled:
            return "Ollama: off"
        now = time.time()
        last_ok = max(float(q.get("last_ok_s") or 0.0), float(e.get("last_ok_s") or 0.0))
        if last_ok > 0.0 and (now - last_ok) < 90.0:
            return "Ollama: ok"
        return "Ollama: offline"

    def _maybe_device_hints(self, v, a) -> None:
        if not self._session_active:
            return
        if self._session_started_wall_s is None:
            return
        now = time.time()
        if (now - self._session_started_wall_s) < 2.0:
            return
        if (now - self._last_device_hint_s) < 8.0:
            return

        if v is None:
            err = self._vision.error() or "Camera not available (permissions or device busy)."
            self._last_device_hint_s = now
            self._maybe_toast(err)
            return

        if a is None:
            self._last_device_hint_s = now
            self._maybe_toast("Microphone not available. Check permissions and input device.")
            return

    def _maybe_ollama_hints(self) -> None:
        if not self._session_active:
            return
        if self._session_started_wall_s is None:
            return
        now = time.time()
        if (now - self._session_started_wall_s) < 2.0:
            return
        if (now - self._last_ollama_hint_s) < 10.0:
            return

        q = self._interviewer.ollama_status()
        e = self._audio.ollama_status()
        enabled = bool(q.get("enabled")) or bool(e.get("enabled"))
        if not enabled:
            return

        last_ok = max(float(q.get("last_ok_s") or 0.0), float(e.get("last_ok_s") or 0.0))
        if last_ok > 0.0 and (now - last_ok) < 90.0:
            return

        err = q.get("last_error") or e.get("last_error")
        if err:
            msg = f"Ollama issue: {err}"
        else:
            msg = "Ollama not responding. Start `ollama serve` and verify host/model in config."
        self._last_ollama_hint_s = now
        self._maybe_toast(msg)

    def _maybe_record_sample(self, v, a, fr) -> None:
        if not self._session.active:
            return
        rel = self._session.now_rel_s()
        if rel is None:
            return
        if (rel - self._last_sample_rel_s) < 0.2:
            return
        self._last_sample_rel_s = rel

        self._session.record_sample(
            SessionSample(
                t_rel_s=rel,
                question_id=self._active_question_id,
                question_index=self._active_question_index,
                eye_contact=(v.result.eye_contact if v else None),
                face_emotion=(v.result.emotion if v else None),
                speech_wpm=(a.wpm if a else None),
                filler_per_min=(a.filler_per_min if a else None),
                pause_count=(a.pause_count if a else None),
                pitch_hz=(a.pitch_hz if a else None),
                energy=(a.energy if a else None),
                speech_emotion=(a.speech_emotion if a else None),
                confidence_0_100=float(fr.confidence_0_100),
                vision_latency_ms=(v.result.latency_ms if v else None),
                audio_latency_ms=(a.latency_ms if a else None),
                fusion_latency_ms=float(fr.latency_ms),
            )
        )

    def _maybe_toast(self, msg: str) -> None:
        now = time.time()
        msg = msg.strip()
        if not msg:
            return
        if msg == self._last_hint and (now - self._last_hint_s) < 8.0:
            return
        if (now - self._last_hint_s) < 4.0:
            return
        self._last_hint_s = now
        self._last_hint = msg
        self._toast.show_message(msg)
        QTimer.singleShot(2500, self._toast.hide)


def _kpi(label: str, value: str) -> QFrame:
    card = QFrame()
    card.setObjectName("Card")
    layout = QVBoxLayout(card)
    t = QLabel(label)
    t.setObjectName("Kpi")
    v = QLabel(value)
    v.setObjectName("KpiValue")
    layout.addWidget(t)
    layout.addWidget(v)
    return card


def _to_pixmap(frame_bgr, target_w: int, target_h: int) -> QPixmap:
    import cv2  # type: ignore

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(img)
    return pix.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
