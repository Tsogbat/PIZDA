from __future__ import annotations

import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from interview_coach.config import InterviewConfig, OllamaConfig


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

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            resp = self._infer(req)
            with self._lock:
                self._latest = resp

    def _infer(self, req: _QuestionRequest) -> _QuestionResponse:
        prompt = _ollama_question_prompt(req)
        payload = {
            "model": self._cfg.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 120},
        }
        url = self._cfg.host.rstrip("/") + "/api/generate"
        http_req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self._cfg.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            return _QuestionResponse(updated_s=time.time(), index=req.index, text=None, error=str(e))

        try:
            outer = json.loads(body)
        except json.JSONDecodeError as e:
            return _QuestionResponse(updated_s=time.time(), index=req.index, text=None, error=f"invalid JSON: {e}")

        raw = str(outer.get("response") or "").strip()
        data = _parse_json_object(raw) or {}
        text = str(data.get("question") or "").strip()
        if not text:
            text = raw.splitlines()[0].strip() if raw else ""

        text = _sanitize_question(text)
        if not text:
            return _QuestionResponse(updated_s=time.time(), index=req.index, text=None, error="empty question")
        return _QuestionResponse(updated_s=time.time(), index=req.index, text=text, error=None)


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
