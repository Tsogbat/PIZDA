from __future__ import annotations

import time
from dataclasses import dataclass

from interview_coach.audio.worker import AudioResult
from interview_coach.config import FusionConfig
from interview_coach.utils.smoothing import clamp
from interview_coach.vision.analyzer import VisionResult


@dataclass(frozen=True)
class Indicator:
    key: str
    value_0_1: float
    weight: float
    contribution_0_1: float
    detail: str


@dataclass(frozen=True)
class FusionResult:
    timestamp_s: float
    confidence_0_100: float
    indicators: dict[str, Indicator]
    hints: tuple[str, ...]
    latency_ms: float


class FusionEngine:
    def __init__(self, cfg: FusionConfig):
        self._cfg = cfg

    def fuse(self, vision: VisionResult | None, audio: AudioResult | None) -> FusionResult:
        t0 = time.perf_counter()
        now_s = time.time()

        indicators: dict[str, Indicator] = {}
        hints: list[str] = []

        eye = 0.0 if vision is None else float(vision.eye_contact)
        indicators["eye_contact"] = Indicator(
            key="eye_contact",
            value_0_1=clamp(eye, 0.0, 1.0),
            weight=self._cfg.w_eye_contact,
            contribution_0_1=self._cfg.w_eye_contact * clamp(eye, 0.0, 1.0),
            detail="Head-pose based gaze proxy (yaw/pitch thresholds).",
        )
        if eye < 0.5:
            hints.append("Maintain eye contact")

        emo_score, emo_detail, emo_hint = _face_emotion_score(vision.emotion if vision else None)
        indicators["face_emotion"] = Indicator(
            key="face_emotion",
            value_0_1=emo_score,
            weight=self._cfg.w_face_emotion,
            contribution_0_1=self._cfg.w_face_emotion * emo_score,
            detail=emo_detail,
        )
        if emo_hint:
            hints.append(emo_hint)

        wpm = 0.0 if audio is None else float(audio.wpm)
        wpm_score, wpm_detail, wpm_hint = _speech_rate_score(wpm)
        indicators["speech_rate"] = Indicator(
            key="speech_rate",
            value_0_1=wpm_score,
            weight=self._cfg.w_speech_rate,
            contribution_0_1=self._cfg.w_speech_rate * wpm_score,
            detail=wpm_detail,
        )
        if wpm_hint:
            hints.append(wpm_hint)

        fillers = 0.0 if audio is None else float(audio.filler_per_min)
        filler_score, filler_detail, filler_hint = _filler_score(fillers)
        indicators["fillers"] = Indicator(
            key="fillers",
            value_0_1=filler_score,
            weight=self._cfg.w_fillers,
            contribution_0_1=self._cfg.w_fillers * filler_score,
            detail=filler_detail,
        )
        if filler_hint:
            hints.append(filler_hint)

        pause_s = 0.0 if (audio is None or audio.last_pause_s is None) else float(audio.last_pause_s)
        pause_score, pause_detail, pause_hint = _pause_score(pause_s)
        indicators["pauses"] = Indicator(
            key="pauses",
            value_0_1=pause_score,
            weight=self._cfg.w_pauses,
            contribution_0_1=self._cfg.w_pauses * pause_score,
            detail=pause_detail,
        )
        if pause_hint:
            hints.append(pause_hint)

        speech_emo_score, speech_emo_detail, speech_emo_hint = _speech_emotion_score(audio.speech_emotion if audio else None)
        indicators["speech_emotion"] = Indicator(
            key="speech_emotion",
            value_0_1=speech_emo_score,
            weight=self._cfg.w_speech_emotion,
            contribution_0_1=self._cfg.w_speech_emotion * speech_emo_score,
            detail=speech_emo_detail,
        )
        if speech_emo_hint:
            hints.append(speech_emo_hint)

        total_w = sum(ind.weight for ind in indicators.values())
        total = sum(ind.contribution_0_1 for ind in indicators.values())
        score_0_1 = 0.0 if total_w <= 0 else total / total_w
        score_0_100 = clamp(score_0_1, 0.0, 1.0) * 100.0

        # Keep hints short and non-intrusive; UI should rate-limit further.
        hints_out = tuple(dict.fromkeys(hints))[:3]

        return FusionResult(
            timestamp_s=now_s,
            confidence_0_100=float(score_0_100),
            indicators=indicators,
            hints=hints_out,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


def _face_emotion_score(emotion: str | None) -> tuple[float, str, str | None]:
    if emotion is None:
        return 0.5, "No face emotion available.", None
    emotion = emotion.lower()
    if emotion in {"happy"}:
        return 1.0, f"Facial emotion: {emotion}.", None
    if emotion in {"neutral", "surprise"}:
        return 0.8, f"Facial emotion: {emotion}.", None
    if emotion in {"sad", "fear", "disgust"}:
        return 0.35, f"Facial emotion: {emotion}.", "Show more positive expression"
    if emotion in {"angry"}:
        return 0.2, f"Facial emotion: {emotion}.", "Relax facial tension"
    return 0.6, f"Facial emotion: {emotion}.", None


def _speech_rate_score(wpm: float) -> tuple[float, str, str | None]:
    # Typical interview pacing: ~110–160 WPM.
    if wpm <= 0:
        return 0.6, "Speech rate unavailable yet.", None
    ideal_lo, ideal_hi = 110.0, 160.0
    if ideal_lo <= wpm <= ideal_hi:
        return 1.0, f"Speech rate {wpm:.0f} WPM (ideal).", None
    if wpm < ideal_lo:
        score = clamp(wpm / ideal_lo, 0.0, 1.0)
        return score, f"Speech rate {wpm:.0f} WPM (slow).", "Speak a bit faster"
    # wpm > ideal_hi
    score = clamp(ideal_hi / wpm, 0.0, 1.0)
    return score, f"Speech rate {wpm:.0f} WPM (fast).", "Slow down your speech"


def _filler_score(fillers_per_min: float) -> tuple[float, str, str | None]:
    if fillers_per_min <= 0:
        return 1.0, "No filler words detected.", None
    # Gentle penalty: >2/min starts to hurt; >8/min is strongly penalized.
    score = clamp(1.0 - (fillers_per_min - 2.0) / 6.0, 0.0, 1.0) if fillers_per_min > 2.0 else 1.0
    hint = "Reduce filler words" if fillers_per_min > 4.0 else None
    return score, f"Filler rate {fillers_per_min:.1f}/min.", hint


def _pause_score(last_pause_s: float) -> tuple[float, str, str | None]:
    if last_pause_s <= 0:
        return 1.0, "No long pauses detected.", None
    if last_pause_s < 1.2:
        return 0.95, f"Short pause {last_pause_s:.1f}s.", None
    if last_pause_s < 2.5:
        return 0.75, f"Pause {last_pause_s:.1f}s.", None
    score = clamp(1.0 - (last_pause_s - 2.5) / 3.5, 0.0, 1.0)
    return score, f"Long pause {last_pause_s:.1f}s.", "Take your time, but avoid long pauses"


def _speech_emotion_score(emotion: str | None) -> tuple[float, str, str | None]:
    if emotion is None:
        return 0.6, "No speech emotion available.", None
    emotion = emotion.lower()
    if emotion in {"happy"}:
        return 0.95, f"Speech emotion: {emotion}.", None
    if emotion in {"neutral"}:
        return 0.85, f"Speech emotion: {emotion}.", None
    if emotion in {"sad"}:
        return 0.55, f"Speech emotion: {emotion}.", "Add more vocal energy"
    if emotion in {"anxious"}:
        return 0.45, f"Speech emotion: {emotion}.", "Breathe and slow down"
    if emotion in {"angry"}:
        return 0.35, f"Speech emotion: {emotion}.", "Keep a calm tone"
    return 0.7, f"Speech emotion: {emotion}.", None

