from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass

from interview_coach.config import InterviewConfig, OllamaConfig
from interview_coach.ollama import extract_text as _ollama_extract_text
from interview_coach.ollama import looks_like_format_error as _ollama_looks_like_format_error
from interview_coach.ollama import post_json as _ollama_post_json


@dataclass(frozen=True)
class InterviewQuestion:
    id: str
    text: str
    ready: bool = True
    source: str = "predefined"  # "predefined" | "llm"


DEFAULT_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion("intro", "Tell me about yourself."),
    InterviewQuestion("strength", "What is one of your strengths, and how has it helped you at work or school?"),
    InterviewQuestion("weakness", "What is a weakness you are working on, and what steps are you taking to improve?"),
    InterviewQuestion("conflict", "Describe a time you had a conflict in a team. What did you do?"),
    InterviewQuestion("leadership", "Tell me about a time you showed leadership."),
    InterviewQuestion("failure", "Tell me about a time you failed. What did you learn?"),
    InterviewQuestion("pressure", "How do you handle pressure or tight deadlines?"),
    InterviewQuestion("why_role", "Why are you interested in this role, and why should we hire you?"),
)


@dataclass(frozen=True)
class _QuestionRequest:
    index: int
    total: int
    history: tuple[str, ...]
    target_role: str


@dataclass(frozen=True)
class _QuestionResponse:
    updated_s: float
    index: int
    text: str | None
    error: str | None


class OllamaInterviewQuestionClient:
    def __init__(self, cfg: OllamaConfig):
        self._cfg = cfg
        self._queue: queue.Queue[_QuestionRequest] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: _QuestionResponse | None = None
        self._last_ok_s = 0.0
        self._last_error: str | None = None
        self._last_error_s = 0.0

        if self._cfg.enabled:
            self._thread = threading.Thread(target=self._run, name="OllamaInterviewQuestions", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def submit(self, req: _QuestionRequest) -> None:
        if not self._cfg.enabled:
            return
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(req)
        except queue.Full:
            pass

    def pop_latest(self) -> _QuestionResponse | None:
        with self._lock:
            out = self._latest
            self._latest = None
        return out

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": bool(self._cfg.enabled),
                "last_ok_s": float(self._last_ok_s),
                "last_error": self._last_error,
                "last_error_s": float(self._last_error_s),
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            resp = self._infer(req)
            with self._lock:
                self._latest = resp
                if resp.text:
                    self._last_ok_s = float(resp.updated_s)
                    self._last_error = None
                elif resp.error:
                    self._last_error = str(resp.error)
                    self._last_error_s = float(resp.updated_s)

    def _infer(self, req: _QuestionRequest) -> _QuestionResponse:
        prompt = _ollama_question_prompt(req)
        timeout_s = float(getattr(self._cfg, "question_timeout_s", self._cfg.timeout_s))
        t2 = float(min(max(timeout_s * 2.0, timeout_s + 4.0), 90.0))
        timeouts = tuple(dict.fromkeys([timeout_s, t2]))

        payloads: tuple[tuple[str, dict], ...] = (
            (
                "/api/chat",
                {
                    "model": self._cfg.model,
                    "messages": [
                        {"role": "system", "content": "You are a mock interview AI interviewer."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.3, "num_predict": 80},
                },
            ),
            (
                "/api/generate",
                {
                    "model": self._cfg.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.3, "num_predict": 80},
                },
            ),
        )

        last_err: str | None = None
        for attempt, t in enumerate(timeouts):
            for path, payload in payloads:
                outer, err = _ollama_post_json(self._cfg.host, path, payload, t)
                if err and _ollama_looks_like_format_error(err):
                    payload2 = dict(payload)
                    payload2.pop("format", None)
                    outer, err = _ollama_post_json(self._cfg.host, path, payload2, t)
                if err:
                    last_err = err
                    continue
                if outer is None:
                    last_err = last_err or "request failed"
                    continue

                raw = _ollama_extract_text(outer).strip()
                data = _parse_json_object(raw) or {}
                text = str(data.get("question") or "").strip()
                if not text:
                    text = raw.splitlines()[0].strip() if raw else ""

                text = _sanitize_question(text)
                if text:
                    return _QuestionResponse(updated_s=time.time(), index=req.index, text=text, error=None)
                last_err = "empty question"

            if attempt == 0:
                time.sleep(0.2)

        return _QuestionResponse(updated_s=time.time(), index=req.index, text=None, error=last_err or "request failed")


def _ollama_question_prompt(req: _QuestionRequest) -> str:
    role = (req.target_role or "").strip()
    role_line = f"Target role: {role}" if role else "Target role: (not specified)"
    history = [h.strip() for h in req.history if h and h.strip()]
    history_txt = "\n".join(f"- {h}" for h in history[-8:]) if history else "- (none)"

    return (
        "You are a mock interview AI interviewer.\n"
        "Generate ONE interview question suitable for a real interview.\n"
        "Constraints:\n"
        f"- This is question {req.index + 1} of {req.total}.\n"
        f"- {role_line}\n"
        "- Do NOT repeat or paraphrase previous questions.\n"
        "- Keep it concise (<= 22 words).\n"
        "- If a target role is provided, tailor the question to that role's typical responsibilities.\n"
        "- Prefer behavioral or role-relevant questions; mix topics (strengths/weaknesses/conflict/leadership/failure/impact/why this role).\n"
        "\n"
        "Previous questions:\n"
        f"{history_txt}\n"
        "\n"
        "Return ONLY JSON in the exact schema: {\"question\": \"...\"}\n"
    )


def _sanitize_question(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(question\\s*[:\\-]\\s*)", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("\"' \t")
    return text


def _parse_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\\{.*\\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


class Interviewer:
    def __init__(
        self,
        interview: InterviewConfig | None = None,
        ollama: OllamaConfig | None = None,
        fallback_questions: tuple[InterviewQuestion, ...] = DEFAULT_QUESTIONS,
    ):
        self._cfg = interview or InterviewConfig(use_llm_questions=False)
        self._ollama = ollama or OllamaConfig(enabled=False)
        self._fallback = fallback_questions

        self._idx = -1
        self._total = int(self._cfg.num_questions) if self._cfg.use_llm_questions else len(self._fallback)
        if self._total <= 0:
            self._total = len(self._fallback) or 8

        self._generated: dict[int, InterviewQuestion] = {}
        self._pending_index: int | None = None
        self._lock = threading.Lock()

        self._client: OllamaInterviewQuestionClient | None = None
        if self._cfg.use_llm_questions and self._ollama.enabled:
            self._client = OllamaInterviewQuestionClient(self._ollama)

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()
        self._client = None

    def ollama_status(self) -> dict:
        if self._client is None:
            return {"enabled": False, "last_ok_s": 0.0, "last_error": None, "last_error_s": 0.0}
        return self._client.status()

    def poll(self) -> InterviewQuestion | None:
        if self._client is None:
            return None
        resp = self._client.pop_latest()
        if resp is None:
            return None
        if resp.text:
            q = InterviewQuestion(id=f"llm_{resp.index + 1}", text=resp.text, ready=True, source="llm")
            with self._lock:
                self._generated[int(resp.index)] = q
                if self._pending_index == int(resp.index):
                    self._pending_index = None
            return q if resp.index == self._idx else None

        fb = self._fallback[int(resp.index)] if 0 <= int(resp.index) < len(self._fallback) else None
        if fb is None:
            return None
        with self._lock:
            self._generated[int(resp.index)] = fb
            if self._pending_index == int(resp.index):
                self._pending_index = None
        return fb if resp.index == self._idx else None

    @property
    def started(self) -> bool:
        return self._idx >= 0

    @property
    def finished(self) -> bool:
        return self._idx >= self._total

    @property
    def index(self) -> int:
        return self._idx

    @property
    def total(self) -> int:
        return int(self._total)

    def start(self) -> InterviewQuestion:
        with self._lock:
            self._idx = 0
            self._pending_index = None
            self._generated = {}
        return self.current()

    def current(self) -> InterviewQuestion:
        if self._idx < 0:
            raise RuntimeError("Interviewer not started.")
        if self._idx >= self._total:
            return InterviewQuestion("done", "Interview complete. Thank you.")
        if self._client is None:
            if 0 <= self._idx < len(self._fallback):
                return self._fallback[self._idx]
            return InterviewQuestion("q", "Tell me about a time you solved a challenging problem.", ready=True, source="predefined")

        with self._lock:
            existing = self._generated.get(self._idx)
            pending = self._pending_index
            history = tuple(self._generated[i].text for i in sorted(self._generated) if i < self._idx and self._generated[i].ready)

        if existing is not None:
            return existing

        if pending != self._idx:
            self._client.submit(
                _QuestionRequest(
                    index=int(self._idx),
                    total=int(self._total),
                    history=history,
                    target_role=str(self._cfg.target_role or ""),
                )
            )
            with self._lock:
                self._pending_index = int(self._idx)

        return InterviewQuestion("pending", "Generating next question...", ready=False, source="llm")

    def next(self) -> InterviewQuestion:
        if self._idx < 0:
            return self.start()
        with self._lock:
            self._idx += 1
        return self.current()
