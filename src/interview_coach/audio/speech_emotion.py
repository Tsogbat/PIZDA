from __future__ import annotations

import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from interview_coach.config import OllamaConfig
from interview_coach.utils.smoothing import clamp


@dataclass(frozen=True)
class SpeechEmotionInput:
    timestamp_s: float
    speaking: bool
    transcript_snippet: str
    wpm: float
    filler_per_min: float
    last_pause_s: float | None
    pitch_hz: float | None
    pitch_var: float | None
    energy: float
    energy_var: float


SPEECH_EMOTION_LABELS: tuple[str, ...] = ("neutral", "happy", "sad", "angry", "anxious")


def heuristic_speech_emotion(x: SpeechEmotionInput) -> tuple[str, dict[str, float]]:
    if not x.speaking:
        return "neutral", _one_hot("neutral", 0.9)

    wpm = float(x.wpm or 0.0)
    fillers = float(x.filler_per_min or 0.0)
    pitch = float(x.pitch_hz or 0.0)
    pitch_var = float(x.pitch_var or 0.0)
    last_pause = float(x.last_pause_s or 0.0)
    energy = float(x.energy or 0.0)
    energy_var = float(x.energy_var or 0.0)

    fast = clamp((wpm - 170.0) / 45.0, 0.0, 1.0) if wpm > 0 else 0.0
    slow = clamp((110.0 - wpm) / 45.0, 0.0, 1.0) if wpm > 0 else 0.0
    filler_hi = clamp((fillers - 4.0) / 6.0, 0.0, 1.0)
    filler_lo = clamp((4.0 - fillers) / 4.0, 0.0, 1.0)
    pause_hi = clamp((last_pause - 2.5) / 3.0, 0.0, 1.0)
    pitch_hi = clamp((pitch - 160.0) / 90.0, 0.0, 1.0) if pitch > 0 else 0.0
    pitch_lo = clamp((150.0 - pitch) / 80.0, 0.0, 1.0) if pitch > 0 else 0.0
    pitch_var_hi = clamp((pitch_var - 25.0) / 25.0, 0.0, 1.0)
    energy_hi = clamp((energy - 0.006) / 0.020, 0.0, 1.0)
    energy_lo = clamp((0.010 - energy) / 0.008, 0.0, 1.0)
    energy_var_hi = clamp((energy_var - 0.003) / 0.012, 0.0, 1.0)

    anxious = clamp(0.35 * fast + 0.25 * filler_hi + 0.15 * pitch_var_hi + 0.10 * pause_hi + 0.15 * energy_var_hi, 0.0, 1.0)
    happy = clamp(0.28 * _pace_score(wpm) + 0.22 * filler_lo + 0.30 * pitch_hi + 0.20 * energy_hi, 0.0, 1.0)
    sad = clamp(0.36 * slow + 0.25 * pause_hi + 0.20 * pitch_lo + 0.19 * energy_lo, 0.0, 1.0)
    angry_base = clamp(0.50 * energy_hi + 0.25 * fast + 0.25 * (1.0 - pitch_var_hi), 0.0, 1.0)
    angry = float(angry_base * (0.45 + 0.55 * filler_lo))

    non_neutral = {"happy": happy, "sad": sad, "angry": angry, "anxious": anxious}
    best_label = max(non_neutral, key=non_neutral.get)
    best = float(non_neutral[best_label])

    if best < 0.40:
        return "neutral", _one_hot("neutral", 0.82)

    conf = float(clamp(0.55 + 0.45 * best, 0.55, 0.99))
    return best_label, _one_hot(best_label, conf)


def _pace_score(wpm: float) -> float:
    if wpm <= 0:
        return 0.5
    # Peak at ~135 WPM; declines outside ~110–160.
    peak = 135.0
    width = 35.0
    score = 1.0 - (abs(wpm - peak) / width)
    return float(clamp(score, 0.0, 1.0))


@dataclass(frozen=True)
class OllamaPrediction:
    updated_s: float
    label: str
    confidence: float
    scores: dict[str, float]


class OllamaSpeechEmotionClient:
    def __init__(self, cfg: OllamaConfig):
        self._cfg = cfg
        self._queue: queue.Queue[SpeechEmotionInput] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: OllamaPrediction | None = None
        self._next_allowed_s = 0.0
        self._last_submit_s = 0.0

        if self._cfg.enabled:
            self._thread = threading.Thread(target=self._run, name="OllamaSpeechEmotion", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset(self) -> None:
        with self._lock:
            self._latest = None

    def submit(self, x: SpeechEmotionInput) -> None:
        if not self._cfg.enabled:
            return
        now = time.time()
        if now < self._next_allowed_s:
            return
        if (now - self._last_submit_s) < self._cfg.min_interval_s:
            return
        self._last_submit_s = now

        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(x)
        except queue.Full:
            pass

    def latest(self, max_age_s: float = 10.0) -> OllamaPrediction | None:
        with self._lock:
            pred = self._latest
        if pred is None:
            return None
        if (time.time() - pred.updated_s) > max_age_s:
            return None
        return pred

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                x = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                pred = self._infer(x)
                if pred is not None:
                    with self._lock:
                        self._latest = pred
            except Exception:
                self._next_allowed_s = time.time() + 10.0

    def _infer(self, x: SpeechEmotionInput) -> OllamaPrediction | None:
        if not x.speaking:
            return OllamaPrediction(updated_s=time.time(), label="neutral", confidence=1.0, scores=_one_hot("neutral", 1.0))

        prompt = _ollama_prompt(x)
        payload = {
            "model": self._cfg.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 80},
        }
        url = self._cfg.host.rstrip("/") + "/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError):
            self._next_allowed_s = time.time() + 10.0
            return None

        try:
            outer = json.loads(body)
        except json.JSONDecodeError:
            self._next_allowed_s = time.time() + 10.0
            return None

        text = str(outer.get("response") or "").strip()
        data = _parse_json_object(text) or {}
        label = str(data.get("label") or "neutral").strip().lower()
        label = _normalize_label(label)
        conf = float(data.get("confidence") or 0.6)
        conf = float(clamp(conf, 0.0, 1.0))
        scores = _one_hot(label, conf)
        return OllamaPrediction(updated_s=time.time(), label=label, confidence=conf, scores=scores)


def _ollama_prompt(x: SpeechEmotionInput) -> str:
    snippet = (x.transcript_snippet or "").strip()
    if len(snippet) > 260:
        snippet = snippet[-260:]
    return (
        "Classify the candidate's vocal emotion for interview coaching using the prosody metrics and transcript snippet.\n"
        "Return ONLY JSON in the exact schema: {\"label\": \"neutral|happy|sad|angry|anxious\", \"confidence\": 0.0-1.0}.\n"
        "Do not add extra keys.\n\n"
        f"speaking: {bool(x.speaking)}\n"
        f"speech_rate_wpm: {float(x.wpm):.1f}\n"
        f"fillers_per_min: {float(x.filler_per_min):.2f}\n"
        f"last_pause_s: {float(x.last_pause_s or 0.0):.2f}\n"
        f"pitch_hz: {float(x.pitch_hz or 0.0):.2f}\n"
        f"pitch_var_hz: {float(x.pitch_var or 0.0):.2f}\n"
        f"energy_rms: {float(x.energy):.4f}\n"
        f"energy_var: {float(x.energy_var):.4f}\n"
        f"transcript_snippet: {json.dumps(snippet)}\n"
    )


def _normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"[^a-z]+", "", label)
    if label in SPEECH_EMOTION_LABELS:
        return label
    if label in {"anxiety", "nervous", "nervousness", "stressed", "stress"}:
        return "anxious"
    if label in {"anger", "mad"}:
        return "angry"
    return "neutral"


def _parse_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _one_hot(label: str, confidence: float) -> dict[str, float]:
    label = _normalize_label(label)
    confidence = float(clamp(confidence, 0.0, 1.0))
    rest = (1.0 - confidence) / max(1, len(SPEECH_EMOTION_LABELS) - 1)
    scores = {k: float(rest) for k in SPEECH_EMOTION_LABELS}
    scores[label] = confidence
    return scores
