from __future__ import annotations

from collections import Counter


def summarize_session(session: dict) -> dict:
    samples = session.get("samples") or []
    duration_s = session.get("duration_s") or 0.0

    def _avg(key: str) -> float | None:
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _p95(key: str) -> float | None:
        vals = sorted([s.get(key) for s in samples if s.get(key) is not None])
        if not vals:
            return None
        k = int(round(0.95 * (len(vals) - 1)))
        return float(vals[k])

    conf = [s.get("confidence_0_100") for s in samples if s.get("confidence_0_100") is not None]
    eye = [s.get("eye_contact") for s in samples if s.get("eye_contact") is not None]
    wpm = [s.get("speech_wpm") for s in samples if s.get("speech_wpm") is not None]
    fillers = [s.get("filler_per_min") for s in samples if s.get("filler_per_min") is not None]
    pauses = [s.get("pause_count") for s in samples if s.get("pause_count") is not None]

    face_emotions = Counter([s.get("face_emotion") for s in samples if s.get("face_emotion")])
    speech_emotions = Counter([s.get("speech_emotion") for s in samples if s.get("speech_emotion")])
    pause_total = int(max(pauses)) if pauses else None
    minutes = float(duration_s) / 60.0 if duration_s else None
    pause_per_min = (float(pause_total) / minutes) if (pause_total is not None and minutes and minutes > 1e-6) else None

    return {
        "duration_s": float(duration_s) if duration_s else None,
        "confidence_avg": _avg("confidence_0_100"),
        "confidence_min": float(min(conf)) if conf else None,
        "confidence_max": float(max(conf)) if conf else None,
        "eye_contact_avg": float(sum(eye) / len(eye)) if eye else None,
        "speech_wpm_avg": float(sum(wpm) / len(wpm)) if wpm else None,
        "fillers_per_min_avg": float(sum(fillers) / len(fillers)) if fillers else None,
        "pause_count_total": pause_total,
        "pause_per_min": pause_per_min,
        "vision_latency_ms_avg": _avg("vision_latency_ms"),
        "audio_latency_ms_avg": _avg("audio_latency_ms"),
        "fusion_latency_ms_avg": _avg("fusion_latency_ms"),
        "vision_latency_ms_p95": _p95("vision_latency_ms"),
        "audio_latency_ms_p95": _p95("audio_latency_ms"),
        "fusion_latency_ms_p95": _p95("fusion_latency_ms"),
        "dominant_face_emotion": face_emotions.most_common(1)[0][0] if face_emotions else None,
        "dominant_speech_emotion": speech_emotions.most_common(1)[0][0] if speech_emotions else None,
        "total_samples": len(samples),
    }


def coaching_recommendations(session: dict) -> list[str]:
    summary = summarize_session(session)
    recs: list[str] = []

    if not summary.get("total_samples"):
        return [
            "No session samples were recorded. Press Start Interview, speak for a few seconds, then End Session.",
        ]

    eye = summary.get("eye_contact_avg")
    if eye is None:
        recs.append(
            "Camera/face tracking was unavailable. On macOS: System Settings → Privacy & Security → Camera, enable your Terminal/Python, then restart."
        )
    elif eye < 0.5:
        recs.append("Eye contact was low. Keep your head centered and look toward the camera between thoughts.")

    wpm = summary.get("speech_wpm_avg")
    if wpm is None or wpm <= 0:
        recs.append("Speech transcript was sparse. Verify microphone permissions and that a Vosk model exists under `models/vosk/`.")
    else:
        if wpm > 170:
            recs.append("Speech rate was fast. Aim for ~110–160 WPM: add short pauses at sentence boundaries.")
        elif wpm < 100:
            recs.append("Speech rate was slow. Increase pace slightly and shorten pauses between phrases.")

    fillers = summary.get("fillers_per_min_avg")
    if fillers is not None:
        if fillers > 6.0:
            recs.append("Filler rate was high. Replace fillers with a 0.5s thinking pause and start with a structured first sentence.")
        elif fillers > 3.5:
            recs.append("Filler rate was moderate. Try pausing silently instead of using \"um/uh/like\".")

    pauses = summary.get("pause_per_min")
    if pauses is not None and pauses > 2.0:
        recs.append("Long pauses were frequent. Use a simple structure (STAR) and keep thinking pauses under ~2 seconds.")

    speech_emo = (summary.get("dominant_speech_emotion") or "").lower()
    if speech_emo == "anxious":
        recs.append("Vocal tone leaned anxious. Slow down slightly and take one deep breath before answering.")
    elif speech_emo == "sad":
        recs.append("Vocal energy was low. Add more emphasis on key words and lift pitch slightly at transitions.")

    if not recs:
        recs.append("Overall: strong baseline. Keep eye contact steady and maintain a consistent pace.")

    return recs


def coaching_report_text(session: dict) -> str:
    s = summarize_session(session)
    recs = coaching_recommendations(session)

    lines: list[str] = []

    lines.append("Overview")
    dur = s.get("duration_s")
    if isinstance(dur, (int, float)) and dur:
        lines.append(f"- Duration: {_fmt_duration(float(dur))}")
    if s.get("confidence_avg") is not None:
        lines.append(
            f"- Confidence: avg {_fmt_num(s.get('confidence_avg'))} (min {_fmt_num(s.get('confidence_min'))}, max {_fmt_num(s.get('confidence_max'))})"
        )
    if s.get("eye_contact_avg") is not None:
        lines.append(f"- Eye contact: {_fmt_pct(s.get('eye_contact_avg'))}")
    if s.get("speech_wpm_avg") is not None:
        lines.append(f"- Speech rate: {_fmt_num(s.get('speech_wpm_avg'))} WPM")
    if s.get("fillers_per_min_avg") is not None:
        lines.append(f"- Filler rate: {_fmt_num(s.get('fillers_per_min_avg'))}/min")
    if s.get("pause_per_min") is not None:
        lines.append(f"- Long pauses: {_fmt_num(s.get('pause_per_min'))}/min (total {_fmt_int(s.get('pause_count_total'))})")
    if s.get("dominant_face_emotion") or s.get("dominant_speech_emotion"):
        lines.append(
            f"- Dominant emotions: face={s.get('dominant_face_emotion') or '-'}; speech={s.get('dominant_speech_emotion') or '-'}"
        )

    strengths = _strengths_from_summary(s)
    if strengths:
        lines.append("")
        lines.append("What went well")
        for x in strengths:
            lines.append(f"- {x}")

    lines.append("")
    lines.append("What to improve / fixes")
    for r in recs:
        lines.append(f"- {r}")

    return "\n".join(lines).strip() + "\n"


def _strengths_from_summary(s: dict) -> list[str]:
    out: list[str] = []
    conf = s.get("confidence_avg")
    if isinstance(conf, (int, float)):
        if conf >= 75:
            out.append("Overall confidence stayed strong.")
        elif conf >= 60:
            out.append("Overall confidence was steady.")

    eye = s.get("eye_contact_avg")
    if isinstance(eye, (int, float)):
        if eye >= 0.7:
            out.append("Eye contact was consistently good.")
        elif eye >= 0.55:
            out.append("Eye contact was decent; keep it steady between thoughts.")

    wpm = s.get("speech_wpm_avg")
    if isinstance(wpm, (int, float)) and wpm > 0:
        if 110 <= wpm <= 160:
            out.append("Speech pacing was in a strong interview range (110–160 WPM).")

    fillers = s.get("fillers_per_min_avg")
    if isinstance(fillers, (int, float)):
        if fillers <= 2.0:
            out.append("Filler word usage was low.")
        elif fillers <= 3.5:
            out.append("Filler word usage was manageable.")

    pauses = s.get("pause_per_min")
    if isinstance(pauses, (int, float)):
        if pauses <= 1.0:
            out.append("Long pauses were rare.")

    return out


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60.0)
    s = int(round(seconds - m * 60))
    if m <= 0:
        return f"{s}s"
    return f"{m}m {s:02d}s"


def _fmt_pct(x) -> str:
    try:
        return f"{float(x) * 100.0:.0f}%"
    except Exception:
        return "-"


def _fmt_num(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.1f}"
    except Exception:
        return str(x)


def _fmt_int(x) -> str:
    if x is None:
        return "-"
    try:
        return str(int(x))
    except Exception:
        return str(x)
