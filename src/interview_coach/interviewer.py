from __future__ import annotations

import json
import queue
import random
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
    source: str = "predefined"  # "predefined" | "warmup" | "fallback" | "llm"


DEFAULT_QUESTIONS: tuple[InterviewQuestion, ...] = (
    InterviewQuestion("intro", "Tell me about yourself."),
    InterviewQuestion("resume", "Walk me through your resume."),
    InterviewQuestion("strength", "What is one of your strengths, and how has it helped you at work or school?"),
    InterviewQuestion("weakness", "What is a weakness you are working on, and what steps are you taking to improve?"),
    InterviewQuestion("conflict", "Describe a time you had a conflict in a team. What did you do?"),
    InterviewQuestion("leadership", "Tell me about a time you showed leadership."),
    InterviewQuestion("failure", "Tell me about a time you failed. What did you learn?"),
    InterviewQuestion("why_role", "Why are you interested in this role, and why should we hire you?"),
)


@dataclass(frozen=True)
class _QuestionRequest:
    session_key: int
    index: int
    total: int
    history: tuple[str, ...]
    target_role: str


@dataclass(frozen=True)
class _QuestionResponse:
    session_key: int
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
        timeout_s = float(getattr(self._cfg, "question_timeout_s", self._cfg.timeout_s))
        t2 = float(min(max(timeout_s * 2.0, timeout_s + 4.0), 90.0))
        timeouts = tuple(dict.fromkeys([timeout_s, t2]))
        history_norm = {_normalize_for_compare(h) for h in req.history if h}

        last_err: str | None = None
        for attempt, t in enumerate(timeouts):
            prompt = _ollama_question_prompt(req, retry=attempt)
            seed = _variation_seed(req.session_key, req.index, attempt)
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
                        "options": {"temperature": 0.35, "num_predict": 80, "seed": seed},
                    },
                ),
                (
                    "/api/generate",
                    {
                        "model": self._cfg.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.35, "num_predict": 80, "seed": seed},
                    },
                ),
            )

            for path, payload in payloads:
                for allow_no_format in (False, True):
                    payload_try = dict(payload)
                    if allow_no_format:
                        payload_try.pop("format", None)

                    outer, err = _ollama_post_json(self._cfg.host, path, payload_try, t)
                    if err and _ollama_looks_like_format_error(err):
                        payload2 = dict(payload_try)
                        payload2.pop("format", None)
                        outer, err = _ollama_post_json(self._cfg.host, path, payload2, t)
                    if err:
                        last_err = err
                        if allow_no_format:
                            break
                        continue
                    if outer is None:
                        last_err = last_err or "request failed"
                        if allow_no_format:
                            break
                        continue

                    raw = _ollama_extract_text(outer).strip()
                    data = _parse_json_object(raw) or {}
                    text = str(data.get("question") or "").strip()
                    if not text:
                        text = raw.splitlines()[0].strip() if raw else ""

                    text = _sanitize_question(text)
                    if not _is_valid_question(text):
                        last_err = "invalid question"
                        continue
                    if _normalize_for_compare(text) in history_norm:
                        last_err = "duplicate question"
                        continue

                    return _QuestionResponse(session_key=req.session_key, updated_s=time.time(), index=req.index, text=text, error=None)
                # next path

            if attempt == 0:
                time.sleep(0.2)

        return _QuestionResponse(
            session_key=req.session_key,
            updated_s=time.time(),
            index=req.index,
            text=None,
            error=last_err or "request failed",
        )


def _ollama_question_prompt(req: _QuestionRequest, retry: int = 0) -> str:
    role = (req.target_role or "").strip()
    role_line = f"Target role: {role}" if role else "Target role: (not specified)"
    history = [h.strip() for h in req.history if h and h.strip()]
    history_txt = "\n".join(f"- {h}" for h in history[-8:]) if history else "- (none)"
    variation = _variation_seed(req.session_key, req.index, retry)

    return (
        "You are a mock interview AI interviewer.\n"
        "Generate ONE interview question suitable for a real interview.\n"
        "Constraints:\n"
        f"- This is question {req.index + 1} of {req.total}.\n"
        f"- {role_line}\n"
        "- Output MUST be in English only (no Chinese or other languages).\n"
        "- Output MUST be an interview question (not placeholders like /HEIGHT).\n"
        "- Do NOT repeat or paraphrase previous questions.\n"
        "- Keep it concise (<= 22 words).\n"
        "- If a target role is provided, tailor the question to that role's typical responsibilities.\n"
        "- Prefer behavioral or role-relevant questions; mix topics (strengths/weaknesses/conflict/leadership/failure/impact/why this role).\n"
        f"- Internal variation key: {variation} (do not mention it).\n"
        "\n"
        "Previous questions:\n"
        f"{history_txt}\n"
        "\n"
        "Return ONLY JSON in the exact schema: {\"question\": \"...\"}\n"
    )


def _sanitize_question(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(question\\s*[:\\-]\\s*)", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("\"' \t\r\n")
    return text


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _is_valid_question(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if _CJK_RE.search(text):
        return False
    if len(text) < 12:
        return False
    if len(text) > 180:
        return False
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if len(words) < 4:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    if ascii_letters < 10:
        return False
    if text.lstrip().startswith("/") and len(words) <= 6:
        return False
    return True


def _normalize_for_compare(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _variation_seed(session_key: int, index: int, retry: int) -> int:
    # 32-bit mix; stable across process, different across sessions/questions/retries.
    x = int(session_key) & 0xFFFFFFFF
    x ^= (int(index) + 1) * 0x9E3779B1
    x ^= int(retry) * 0x85EBCA77
    x ^= (x >> 16) & 0xFFFFFFFF
    return int(x & 0x7FFFFFFF)


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

        warmup_cfg = int(getattr(self._cfg, "warmup_count", 0) or 0)
        self._warmup_count = max(0, min(warmup_cfg, self._total))
        self._session_key = 0

        self._generated: dict[int, InterviewQuestion] = {}
        self._pending_index: int | None = None
        self._lock = threading.Lock()

        self._client: OllamaInterviewQuestionClient | None = None
        if self._cfg.use_llm_questions and self._ollama.enabled:
            self._client = OllamaInterviewQuestionClient(self._ollama)
        if self._client is None:
            self._warmup_count = 0

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
        if int(resp.session_key) != int(self._session_key):
            return None
        if resp.text:
            llm_n = int(resp.index) - int(self._warmup_count) + 1
            qid = f"llm_{llm_n}" if llm_n >= 1 else f"llm_{resp.index + 1}"
            q = InterviewQuestion(id=qid, text=resp.text, ready=True, source="llm")
            with self._lock:
                self._generated[int(resp.index)] = q
                if self._pending_index == int(resp.index):
                    self._pending_index = None
            return q if resp.index == self._idx else None

        # Only fall back immediately if this is the active (currently displayed) question.
        # If a prefetch failed, leave it ungenerated so we can retry later instead of
        # permanently locking the session into the same fallback question set.
        with self._lock:
            if self._pending_index == int(resp.index):
                self._pending_index = None
        if resp.index != self._idx:
            return None

        fb = self._fallback_question(int(resp.index))
        q_fb = InterviewQuestion(id=fb.id, text=fb.text, ready=True, source="fallback")
        with self._lock:
            self._generated[int(resp.index)] = q_fb
        return q_fb

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
            self._session_key = random.randint(1, 2_147_483_647)
            self._idx = 0
            self._pending_index = None
            self._generated = {}
            for i in range(self._warmup_count):
                fb = self._fallback_question(i)
                self._generated[int(i)] = InterviewQuestion(id=fb.id, text=fb.text, ready=True, source="warmup")
        q = self.current()
        if q.ready:
            self.prefetch()
        return q

    def current(self) -> InterviewQuestion:
        if self._idx < 0:
            raise RuntimeError("Interviewer not started.")
        if self._idx >= self._total:
            return InterviewQuestion("done", "Interview complete. Thank you.")
        if self._client is None:
            fb = self._fallback_question(self._idx)
            return InterviewQuestion(id=fb.id, text=fb.text, ready=True, source="predefined")

        with self._lock:
            existing = self._generated.get(self._idx)
            pending = self._pending_index
            history = tuple(self._generated[i].text for i in sorted(self._generated) if i < self._idx and self._generated[i].ready)

        if existing is not None:
            return existing

        if self._idx < self._warmup_count:
            fb = self._fallback_question(self._idx)
            q = InterviewQuestion(id=fb.id, text=fb.text, ready=True, source="warmup")
            with self._lock:
                self._generated[int(self._idx)] = q
                if self._pending_index == int(self._idx):
                    self._pending_index = None
            return q

        if pending != self._idx:
            self._client.submit(
                _QuestionRequest(
                    session_key=int(self._session_key),
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
        q = self.current()
        if q.ready:
            self.prefetch()
        return q

    def prefetch(self) -> None:
        if self._client is None:
            return
        if self._idx < 0:
            return
        target = int(self._warmup_count) if self._idx < self._warmup_count else int(self._idx + 1)
        if target < 0 or target >= self._total:
            return
        with self._lock:
            if target in self._generated:
                return
            if self._pending_index == target:
                return
            history = tuple(self._generated[i].text for i in sorted(self._generated) if i < target and self._generated[i].ready)
        if target < self._warmup_count:
            fb = self._fallback_question(target)
            q = InterviewQuestion(id=fb.id, text=fb.text, ready=True, source="warmup")
            with self._lock:
                self._generated[int(target)] = q
            return
        self._client.submit(
            _QuestionRequest(
                session_key=int(self._session_key),
                index=int(target),
                total=int(self._total),
                history=history,
                target_role=str(self._cfg.target_role or ""),
            )
        )
        with self._lock:
            self._pending_index = int(target)

    def _fallback_question(self, idx: int) -> InterviewQuestion:
        if 0 <= idx < len(self._fallback):
            return self._fallback[idx]
        return InterviewQuestion(
            id=f"q_{idx + 1}",
            text="Tell me about a time you solved a challenging problem.",
            ready=True,
            source="predefined",
        )
