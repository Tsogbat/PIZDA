from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuestionEvent:
    t_rel_s: float
    question_id: str
    question_index: int
    question_text: str
    transcript_len: int


@dataclass(frozen=True)
class SessionSample:
    t_rel_s: float
    question_id: str
    question_index: int
    eye_contact: float | None
    face_emotion: str | None
    speech_wpm: float | None
    filler_per_min: float | None
    pause_count: int | None
    pitch_hz: float | None
    energy: float | None
    speech_emotion: str | None
    confidence_0_100: float
    vision_latency_ms: float | None
    audio_latency_ms: float | None
    fusion_latency_ms: float | None


class SessionRecorder:
    def __init__(self) -> None:
        self._active = False
        self._start_wall_s: float | None = None
        self._start_monotonic_s: float | None = None
        self._end_wall_s: float | None = None
        self._samples: list[SessionSample] = []
        self._question_events: list[QuestionEvent] = []
        self._final_transcript: str = ""

    def start(self) -> None:
        self._active = True
        self._start_wall_s = time.time()
        self._start_monotonic_s = time.perf_counter()
        self._end_wall_s = None
        self._samples = []
        self._question_events = []
        self._final_transcript = ""

    @property
    def active(self) -> bool:
        return self._active

    def now_rel_s(self) -> float | None:
        if self._start_monotonic_s is None:
            return None
        return float(time.perf_counter() - self._start_monotonic_s)

    def add_question_event(self, question_id: str, question_index: int, question_text: str, transcript_text: str) -> None:
        if not self._active or self._start_monotonic_s is None:
            return
        t_rel = time.perf_counter() - self._start_monotonic_s
        self._question_events.append(
            QuestionEvent(
                t_rel_s=float(t_rel),
                question_id=question_id,
                question_index=int(question_index),
                question_text=str(question_text),
                transcript_len=len(transcript_text or ""),
            )
        )

    def record_sample(self, sample: SessionSample) -> None:
        if not self._active:
            return
        self._samples.append(sample)

    def end(self, final_transcript: str) -> None:
        if not self._active:
            return
        self._active = False
        self._end_wall_s = time.time()
        self._final_transcript = final_transcript or ""

    def to_dict(self) -> dict:
        responses: list[dict] = []
        if self._question_events:
            transcript = self._final_transcript or ""
            for i, ev in enumerate(self._question_events):
                start = int(ev.transcript_len)
                end = int(self._question_events[i + 1].transcript_len) if i + 1 < len(self._question_events) else len(transcript)
                responses.append(
                    {
                        "question_id": ev.question_id,
                        "question_index": ev.question_index,
                        "question_text": ev.question_text,
                        "response_text": transcript[start:end].strip(),
                    }
                )

        return {
            "start_time_s": self._start_wall_s,
            "end_time_s": self._end_wall_s,
            "duration_s": (self._end_wall_s - self._start_wall_s) if (self._start_wall_s and self._end_wall_s) else None,
            "question_events": [asdict(q) for q in self._question_events],
            "responses": responses,
            "samples": [asdict(s) for s in self._samples],
            "transcript": self._final_transcript,
        }

    def export_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def export_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._samples:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = list(asdict(self._samples[0]).keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in self._samples:
                w.writerow(asdict(s))
        return path
