from __future__ import annotations

import shutil
import subprocess
import sys
import threading

from interview_coach.config import TTSConfig


class TextToSpeech:
    def __init__(self, cfg: TTSConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._backend = "none"
        self._engine = None
        self._thread: threading.Thread | None = None

        if not self._cfg.enabled:
            return

        if sys.platform == "darwin" and shutil.which("say"):
            self._backend = "say"
            return

        try:
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            if self._cfg.voice:
                try:
                    self._engine.setProperty("voice", self._cfg.voice)
                except Exception:
                    pass
            try:
                self._engine.setProperty("rate", int(self._cfg.rate_wpm))
            except Exception:
                pass
            self._backend = "pyttsx3"
        except Exception:
            self._backend = "none"

    @property
    def available(self) -> bool:
        return self._backend != "none"

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def speak(self, text: str) -> float:
        text = (text or "").strip()
        if not text or not self._cfg.enabled or self._backend == "none":
            return 0.0

        self.stop()
        dur_s = _estimate_duration_s(text, self._cfg.rate_wpm)

        if self._backend == "say":
            args = ["say", "-r", str(int(self._cfg.rate_wpm))]
            if self._cfg.voice:
                args.extend(["-v", str(self._cfg.voice)])
            args.append(text)
            try:
                proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            except Exception:
                return dur_s
            with self._lock:
                self._proc = proc
            return dur_s

        if self._backend == "pyttsx3" and self._engine is not None:
            def _run():
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    return

            self._thread = threading.Thread(target=_run, name="TTS", daemon=True)
            self._thread.start()
            return dur_s

        return 0.0


def _estimate_duration_s(text: str, rate_wpm: int) -> float:
    words = max(1, len((text or "").split()))
    wpm = max(80, int(rate_wpm) if rate_wpm else 185)
    return float(words / (wpm / 60.0) + 0.35)

