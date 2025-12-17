from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from interview_coach.audio.features import estimate_pitch_hz_autocorr, rms_energy
from interview_coach.audio.speech_emotion import OllamaSpeechEmotionClient, SpeechEmotionInput, heuristic_speech_emotion
from interview_coach.config import AudioConfig, OllamaConfig
from interview_coach.utils.smoothing import EWMA


@dataclass(frozen=True)
class AudioResult:
    timestamp_s: float
    speaking: bool
    partial_text: str
    transcript_text: str
    wpm: float
    filler_count: int
    filler_per_min: float
    pause_count: int
    last_pause_s: float | None
    pitch_hz: float | None
    energy: float
    pitch_var: float | None
    energy_var: float
    speech_emotion: str
    speech_emotion_scores: dict[str, float]
    speech_emotion_source: str
    latency_ms: float


class AudioWorker:
    def __init__(self, config: AudioConfig, vosk_model_dir: Path, ollama: OllamaConfig):
        self._cfg = config
        self._ollama = OllamaSpeechEmotionClient(ollama)

        self._vad = None
        self._vad_frame_samples = int(self._cfg.sample_rate_hz * 0.02)  # 20ms
        self._init_vad()

        self._vosk_model_dir = Path(vosk_model_dir)
        self._vosk_model = None
        self._vosk_rec = None

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: AudioResult | None = None

        self._start_time_s: float | None = None
        self._transcript: list[str] = []
        self._partial: str = ""
        self._last_stt_activity_s: float | None = None

        self._word_events: deque[tuple[float, int]] = deque(maxlen=6000)  # ~20min @ 1Hz-ish
        self._filler_count = 0

        self._speaking = False
        self._seen_speech = False
        self._silence_start_s: float | None = None
        self._in_long_pause = False
        self._pause_count = 0
        self._last_pause_s: float | None = None

        self._energy_floor = EWMA(alpha=0.05)
        self._pitch_hist: deque[float] = deque(maxlen=30)
        self._energy_hist: deque[float] = deque(maxlen=30)

        self._audio_buffer: deque[np.ndarray] = deque()
        self._audio_buffer_len = 0
        self._last_window_s = 0.0
        self._last_emotion: tuple[str, dict[str, float]] = ("neutral", {"neutral": 1.0})
        self._last_emotion_source = "heuristic"
        self._last_pitch: float | None = None
        self._hold_until_s = 0.0

        self._init_vosk()

    def _init_vad(self) -> None:
        try:
            import webrtcvad  # type: ignore

            if int(self._cfg.sample_rate_hz) not in {8000, 16000, 32000, 48000}:
                self._vad = None
                return
            self._vad = webrtcvad.Vad(2)  # 0..3; higher is more aggressive
        except Exception:
            self._vad = None

    def _init_vosk(self) -> None:
        if not self._vosk_model_dir.exists():
            return
        try:
            from vosk import KaldiRecognizer, Model  # type: ignore

            # Support both `models/vosk/<model_folder>` and `models/vosk` already being a model folder.
            model_dir = self._vosk_model_dir
            subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
            if (model_dir / "am").exists() and (model_dir / "conf").exists():
                chosen = model_dir
            elif len(subdirs) == 1 and (subdirs[0] / "am").exists():
                chosen = subdirs[0]
            else:
                chosen = model_dir

            self._vosk_model = Model(str(chosen))
            self._vosk_rec = KaldiRecognizer(self._vosk_model, self._cfg.sample_rate_hz)
            self._vosk_rec.SetWords(True)
        except Exception:
            self._vosk_model = None
            self._vosk_rec = None

    @property
    def stt_available(self) -> bool:
        return self._vosk_rec is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="AudioWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._flush_vosk_final()
        self._refresh_latest(time.time())

    def reset(self) -> None:
        with self._lock:
            self._latest = None
        self._start_time_s = None
        self._transcript = []
        self._partial = ""
        self._last_stt_activity_s = None
        self._word_events.clear()
        self._filler_count = 0
        self._speaking = False
        self._seen_speech = False
        self._silence_start_s = None
        self._in_long_pause = False
        self._pause_count = 0
        self._last_pause_s = None
        self._energy_floor.value = None
        self._pitch_hist.clear()
        self._energy_hist.clear()
        self._audio_buffer.clear()
        self._audio_buffer_len = 0
        self._last_window_s = 0.0
        self._last_emotion = ("neutral", {"neutral": 1.0})
        self._last_emotion_source = "heuristic"
        self._last_pitch = None
        self._hold_until_s = 0.0
        self._ollama.reset()

        if self._vosk_model is not None:
            try:
                from vosk import KaldiRecognizer  # type: ignore

                self._vosk_rec = KaldiRecognizer(self._vosk_model, self._cfg.sample_rate_hz)
                self._vosk_rec.SetWords(True)
            except Exception:
                pass

    def latest(self) -> AudioResult | None:
        with self._lock:
            return self._latest

    def ollama_status(self) -> dict:
        return self._ollama.status()

    def hold(self, seconds: float) -> None:
        seconds = float(seconds)
        if seconds <= 0:
            return
        self._hold_until_s = max(self._hold_until_s, time.time() + seconds)
        self._partial = ""

    def _refresh_hold_latest(self, now_s: float) -> None:
        with self._lock:
            prev = self._latest
        if prev is None:
            self._refresh_latest(now_s)
            return
        with self._lock:
            self._latest = AudioResult(
                timestamp_s=now_s,
                speaking=prev.speaking,
                partial_text="",
                transcript_text=prev.transcript_text,
                wpm=prev.wpm,
                filler_count=prev.filler_count,
                filler_per_min=prev.filler_per_min,
                pause_count=prev.pause_count,
                last_pause_s=prev.last_pause_s,
                pitch_hz=prev.pitch_hz,
                energy=prev.energy,
                pitch_var=prev.pitch_var,
                energy_var=prev.energy_var,
                speech_emotion=prev.speech_emotion,
                speech_emotion_scores=prev.speech_emotion_scores,
                speech_emotion_source=prev.speech_emotion_source,
                latency_ms=0.0,
            )

    def _run(self) -> None:
        import sounddevice as sd  # type: ignore

        block = int(self._cfg.sample_rate_hz * (self._cfg.chunk_ms / 1000.0))
        if block <= 0:
            block = 1600

        def _callback(indata, frames, time_info, status):
            _ = frames, time_info, status
            x = np.asarray(indata, dtype=np.float32).reshape(-1)
            try:
                self._queue.put_nowait(x.copy())
            except queue.Full:
                pass

        with sd.InputStream(
            samplerate=self._cfg.sample_rate_hz,
            channels=self._cfg.channels,
            dtype="float32",
            blocksize=block,
            callback=_callback,
        ):
            buf = np.zeros((0,), dtype=np.float32)
            while not self._stop.is_set():
                try:
                    x = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, x])
                while buf.size >= block:
                    chunk = buf[:block]
                    buf = buf[block:]
                    self._process_chunk(chunk)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        t0 = time.perf_counter()
        now_s = time.time()
        if self._start_time_s is None:
            self._start_time_s = now_s
        if now_s < self._hold_until_s:
            self._refresh_hold_latest(now_s)
            return

        energy = rms_energy(chunk)
        chunk_for_stt = _apply_agc(chunk, energy, self._cfg) if self._cfg.agc_enabled else chunk
        pcm16_bytes = _to_pcm16_bytes(chunk_for_stt)
        speaking_raw = self._speech_from_vad(pcm16_bytes) if self._vad is not None else self._speech_from_energy(energy)

        self._update_stt(pcm16_bytes)
        stt_active = bool(self._last_stt_activity_s is not None and (now_s - float(self._last_stt_activity_s)) < 0.6)
        speaking = bool(speaking_raw or stt_active)
        self._update_pause_state(now_s, speaking)

        transcript = " ".join(self._transcript).strip()
        wpm = self._compute_wpm(now_s)
        filler_per_min = self._compute_filler_per_min(now_s)

        self._update_window_features(now_s, chunk, energy, speaking, transcript, wpm, filler_per_min)

        label, scores = self._last_emotion
        pitch_var = float(np.std(self._pitch_hist)) if len(self._pitch_hist) >= 5 else None
        energy_var = float(np.std(self._energy_hist)) if self._energy_hist else 0.0

        out = AudioResult(
            timestamp_s=now_s,
            speaking=speaking,
            partial_text=self._partial,
            transcript_text=transcript,
            wpm=wpm,
            filler_count=self._filler_count,
            filler_per_min=filler_per_min,
            pause_count=self._pause_count,
            last_pause_s=self._last_pause_s,
            pitch_hz=self._last_pitch,
            energy=float(energy),
            pitch_var=pitch_var,
            energy_var=energy_var,
            speech_emotion=label,
            speech_emotion_scores=scores,
            speech_emotion_source=self._last_emotion_source,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        with self._lock:
            self._latest = out

    def _update_pause_state(self, now_s: float, speaking: bool) -> None:
        if speaking:
            self._seen_speech = True
            self._silence_start_s = None
            self._in_long_pause = False
        else:
            if self._silence_start_s is None:
                self._silence_start_s = now_s
                self._in_long_pause = False
            dur_s = now_s - self._silence_start_s
            if self._seen_speech:
                self._last_pause_s = dur_s
            if self._seen_speech and (not self._in_long_pause) and (dur_s * 1000.0 >= float(self._cfg.pause_silence_ms)):
                self._pause_count += 1
                self._in_long_pause = True
        self._speaking = speaking

    def _update_stt(self, pcm16_bytes: bytes) -> None:
        if self._vosk_rec is None:
            return
        try:
            is_final = self._vosk_rec.AcceptWaveform(pcm16_bytes)
            if is_final:
                res = json.loads(self._vosk_rec.Result() or "{}")
                text = (res.get("text") or "").strip()
                if text:
                    self._append_final_text(text)
                    self._last_stt_activity_s = time.time()
                self._partial = ""
            else:
                res = json.loads(self._vosk_rec.PartialResult() or "{}")
                self._partial = (res.get("partial") or "").strip()
                if self._partial:
                    self._last_stt_activity_s = time.time()
        except Exception:
            return

    def _append_final_text(self, text: str) -> None:
        self._transcript.append(text)

        words = [w for w in text.split() if w]
        self._word_events.append((time.time(), len(words)))

        self._filler_count = _count_fillers(" ".join(self._transcript), self._cfg.filler_words)

    def _flush_vosk_final(self) -> None:
        if self._vosk_rec is None:
            return
        try:
            res = json.loads(self._vosk_rec.FinalResult() or "{}")
            text = (res.get("text") or "").strip()
            if text:
                self._append_final_text(text)
                self._last_stt_activity_s = time.time()
            self._partial = ""
        except Exception:
            return

    def _refresh_latest(self, now_s: float) -> None:
        transcript = " ".join(self._transcript).strip()
        wpm = self._compute_wpm(now_s)
        filler_per_min = self._compute_filler_per_min(now_s)
        label, scores = self._last_emotion

        with self._lock:
            prev = self._latest
            energy = float(prev.energy) if prev is not None else 0.0
            pitch_var = prev.pitch_var if prev is not None else None
            energy_var = float(prev.energy_var) if prev is not None else 0.0
            pitch_hz = prev.pitch_hz if prev is not None else None
            speaking = prev.speaking if prev is not None else False

            self._latest = AudioResult(
                timestamp_s=now_s,
                speaking=speaking,
                partial_text=self._partial,
                transcript_text=transcript,
                wpm=wpm,
                filler_count=self._filler_count,
                filler_per_min=filler_per_min,
                pause_count=self._pause_count,
                last_pause_s=self._last_pause_s,
                pitch_hz=pitch_hz,
                energy=energy,
                pitch_var=pitch_var,
                energy_var=energy_var,
                speech_emotion=label,
                speech_emotion_scores=scores,
                speech_emotion_source=self._last_emotion_source,
                latency_ms=0.0,
            )

    def _speech_from_energy(self, energy: float) -> bool:
        floor = self._energy_floor.value if self._energy_floor.value is not None else energy
        # Update noise floor mostly when we believe we're not speaking.
        if (not self._speaking) or energy < max(self._cfg.vad_energy_threshold, (floor or 0.0) * 1.2):
            floor = self._energy_floor.update(min(energy, floor))
        threshold = max(self._cfg.vad_energy_threshold, (floor or 0.0) * 2.8 + 0.002)
        return bool(energy >= threshold)

    def _speech_from_vad(self, pcm16_bytes: bytes) -> bool:
        if self._vad is None:
            return False
        frame_bytes = self._vad_frame_samples * 2  # int16
        if frame_bytes <= 0:
            return False
        if len(pcm16_bytes) < frame_bytes:
            pcm16_bytes = pcm16_bytes + b"\x00" * (frame_bytes - len(pcm16_bytes))
        rem = len(pcm16_bytes) % frame_bytes
        if rem:
            pcm16_bytes = pcm16_bytes + b"\x00" * (frame_bytes - rem)

        n_frames = len(pcm16_bytes) // frame_bytes
        speech_frames = 0
        for i in range(n_frames):
            frame = pcm16_bytes[i * frame_bytes : (i + 1) * frame_bytes]
            try:
                if self._vad.is_speech(frame, self._cfg.sample_rate_hz):
                    speech_frames += 1
            except Exception:
                return False

        return (speech_frames / max(1, n_frames)) >= 0.35

    def _compute_wpm(self, now_s: float) -> float:
        if self._start_time_s is None:
            return 0.0
        # Rolling 60s WPM to be responsive.
        cutoff = now_s - 60.0
        while self._word_events and self._word_events[0][0] < cutoff:
            self._word_events.popleft()
        words = sum(n for _, n in self._word_events)
        minutes = 1.0
        if self._word_events:
            span = max(1.0, now_s - self._word_events[0][0])
            minutes = span / 60.0
        return float(words / max(1e-6, minutes))

    def _compute_filler_per_min(self, now_s: float) -> float:
        if self._start_time_s is None:
            return 0.0
        minutes = max(1e-6, (now_s - self._start_time_s) / 60.0)
        return float(self._filler_count / minutes)

    def _update_window_features(
        self,
        now_s: float,
        chunk: np.ndarray,
        energy: float,
        speaking: bool,
        transcript: str,
        wpm: float,
        filler_per_min: float,
    ) -> None:
        self._energy_hist.append(float(energy))

        self._audio_buffer.append(chunk.astype(np.float32))
        self._audio_buffer_len += int(chunk.size)
        target = int(self._cfg.sample_rate_hz * 1.0)  # 1s window
        while self._audio_buffer_len > target and self._audio_buffer:
            drop = self._audio_buffer.popleft()
            self._audio_buffer_len -= int(drop.size)

        if now_s - self._last_window_s < 0.5:
            return
        self._last_window_s = now_s

        if self._audio_buffer_len < target:
            return

        window = np.concatenate(list(self._audio_buffer))[-target:]
        pitch = estimate_pitch_hz_autocorr(window, self._cfg.sample_rate_hz) if speaking else None
        self._last_pitch = pitch
        if pitch is not None:
            self._pitch_hist.append(float(pitch))

        pitch_var = float(np.std(self._pitch_hist)) if len(self._pitch_hist) >= 5 else None
        energy_var = float(np.std(self._energy_hist)) if self._energy_hist else 0.0
        snippet = (transcript + (" " + self._partial if self._partial else "")).strip()
        x = SpeechEmotionInput(
            timestamp_s=now_s,
            speaking=speaking,
            transcript_snippet=snippet,
            wpm=float(wpm),
            filler_per_min=float(filler_per_min),
            last_pause_s=self._last_pause_s,
            pitch_hz=pitch,
            pitch_var=pitch_var,
            energy=float(energy),
            energy_var=float(energy_var),
        )

        heur = heuristic_speech_emotion(x)
        if not speaking:
            self._last_emotion = heur
            self._last_emotion_source = "heuristic"
            return

        self._ollama.submit(x)
        pred = self._ollama.latest(max_age_s=10.0)
        if pred is None:
            self._last_emotion = heur
            self._last_emotion_source = "heuristic"
            return

        heur_label, heur_scores = heur
        heur_conf = float(max(heur_scores.values())) if heur_scores else 0.0

        use_pred = False
        if pred.label != "neutral":
            use_pred = bool(pred.confidence >= 0.55)
        else:
            use_pred = bool(pred.confidence >= 0.88 and (heur_label == "neutral" or pred.confidence >= (heur_conf + 0.08)))

        if use_pred:
            self._last_emotion = (pred.label, pred.scores)
            self._last_emotion_source = "ollama"
        else:
            self._last_emotion = heur
            self._last_emotion_source = "heuristic"


def _to_pcm16_bytes(chunk_f32: np.ndarray) -> bytes:
    pcm16 = np.clip(chunk_f32, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def _apply_agc(chunk: np.ndarray, energy_rms: float, cfg: AudioConfig) -> np.ndarray:
    if energy_rms <= 1e-8:
        return chunk
    target = float(cfg.agc_target_rms)
    max_gain = max(1.0, float(cfg.agc_max_gain))
    gain = target / float(energy_rms)
    gain = max(1.0 / max_gain, min(max_gain, gain))
    if abs(gain - 1.0) < 0.05:
        return chunk
    return np.clip(chunk * gain, -1.0, 1.0).astype(np.float32)


_TOKEN_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "kinda": ("kind", "of"),
    "kindof": ("kind", "of"),
    "sorta": ("sort", "of"),
    "sortof": ("sort", "of"),
    "alright": ("all", "right"),
    "allright": ("all", "right"),
    "ok": ("okay",),
}

_SOUND_PATTERNS: dict[str, re.Pattern[str]] = {
    "uh": re.compile(r"^u+h+$"),
    "um": re.compile(r"^u+h?m+$"),
    "er": re.compile(r"^e+r+m*$"),
    "oh": re.compile(r"^o+h+$"),
    "huh": re.compile(r"^h+u+h+$"),
    "hm": re.compile(r"^h+m+$"),
}

_PHRASE_VARIANTS: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {
    ("i", "mean"): (("i", "meant"),),
    ("you", "know"): (("you", "no"),),
}


def _normalize_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        t = tok.strip().lower()
        if not t:
            continue
        exp = _TOKEN_EXPANSIONS.get(t)
        if exp is not None:
            out.extend(exp)
        else:
            out.append(t)
    return out


def _count_fillers(text: str, fillers: tuple[str, ...]) -> int:
    tokens = _normalize_tokens(re.findall(r"[a-z']+", (text or "").lower()))
    if not tokens:
        return 0

    single_exact: set[str] = set()
    single_sound: set[str] = set()
    multi_phrases: set[tuple[str, ...]] = set()

    for filler in fillers:
        parts = _normalize_tokens(re.findall(r"[a-z']+", (filler or "").lower()))
        if not parts:
            continue
        if len(parts) == 1:
            key = parts[0]
            if key in _SOUND_PATTERNS:
                single_sound.add(key)
            else:
                single_exact.add(key)
        else:
            tup = tuple(parts)
            multi_phrases.add(tup)
            for v in _PHRASE_VARIANTS.get(tup, ()):
                multi_phrases.add(v)

    total = 0
    if single_exact or single_sound:
        for tok in tokens:
            if tok in single_exact:
                total += 1
                continue
            for key in single_sound:
                if _SOUND_PATTERNS[key].fullmatch(tok):
                    total += 1
                    break

    for phrase in multi_phrases:
        n = len(phrase)
        if n <= 0 or len(tokens) < n:
            continue
        for i in range(0, len(tokens) - n + 1):
            if tuple(tokens[i : i + n]) == phrase:
                total += 1

    return int(total)
