from __future__ import annotations

import json
import re
import sys
import time

import interview_coach
from interview_coach.config import AppConfig
from interview_coach.ollama import extract_text, get_json, looks_like_format_error, post_json


def main() -> None:
    cfg = AppConfig()
    print("interview_coach:", getattr(interview_coach, "__file__", "?"))
    print("python:", sys.version.replace("\n", " "))
    print(f"ollama.enabled={cfg.ollama.enabled} host={cfg.ollama.host!r} model={cfg.ollama.model!r}")
    print(
        f"interview.use_llm_questions={cfg.interview.use_llm_questions} "
        f"num_questions={cfg.interview.num_questions} target_role={cfg.interview.target_role!r}"
    )
    print()

    if not cfg.ollama.enabled:
        print("Ollama is disabled in config (OllamaConfig.enabled=False).")
        raise SystemExit(0)

    version, v_err = get_json(cfg.ollama.host, "/api/version", timeout_s=2.0)
    if version and not v_err:
        print("Ollama /api/version:", json.dumps(version, indent=2))
    else:
        print("Ollama /api/version:", v_err or "unavailable")

    tags, err = get_json(cfg.ollama.host, "/api/tags", timeout_s=3.0)
    if err or not tags:
        print("Ollama /api/tags: ERROR:", err or "no response")
        print("Fix: start `ollama serve` and verify host/port in src/interview_coach/config.py")
        raise SystemExit(1)

    models = [m.get("name") for m in (tags.get("models") or []) if isinstance(m, dict) and m.get("name")]
    print(f"Ollama models ({len(models)}):")
    for name in models[:20]:
        print(" -", name)
    if len(models) > 20:
        print(f" - ... ({len(models) - 20} more)")
    print()

    if cfg.ollama.model not in models:
        print(f"Configured model {cfg.ollama.model!r} not found in `ollama list`.")
        print(f"Fix: run `ollama pull {cfg.ollama.model}` or update OllamaConfig.model.")
        print()

    q = _try_generate_question(cfg)
    if not q:
        raise SystemExit(1)
    print("Sample generated question:", q)
    raise SystemExit(0)


def _try_generate_question(cfg: AppConfig) -> str | None:
    prompt = _question_prompt(
        index=0,
        total=max(1, int(cfg.interview.num_questions)),
        target_role=str(cfg.interview.target_role or ""),
        history=(),
    )
    payloads: tuple[tuple[str, dict], ...] = (
        (
            "/api/chat",
            {
                "model": cfg.ollama.model,
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
                "model": cfg.ollama.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3, "num_predict": 80},
            },
        ),
    )

    timeout_s = float(getattr(cfg.ollama, "question_timeout_s", cfg.ollama.timeout_s))
    t2 = float(min(max(timeout_s * 2.0, timeout_s + 4.0), 90.0))
    timeouts = tuple(dict.fromkeys([timeout_s, t2]))

    last_err: str | None = None
    for attempt, t in enumerate(timeouts):
        for path, payload in payloads:
            outer, err = post_json(cfg.ollama.host, path, payload, timeout_s=t)
            if err and looks_like_format_error(err):
                payload2 = dict(payload)
                payload2.pop("format", None)
                outer, err = post_json(cfg.ollama.host, path, payload2, timeout_s=t)
            if err:
                last_err = err
                continue
            if not outer:
                last_err = last_err or "no response"
                continue

            raw = extract_text(outer).strip()
            q = _parse_question(raw)
            if q:
                return q

        if attempt == 0:
            time.sleep(0.2)

    print("Question generation: ERROR:", last_err or "failed")
    return None


def _question_prompt(index: int, total: int, target_role: str, history: tuple[str, ...]) -> str:
    role = (target_role or "").strip()
    role_line = f"Target role: {role}" if role else "Target role: (not specified)"
    history = [h.strip() for h in history if h and h.strip()]
    history_txt = "\n".join(f"- {h}" for h in history[-8:]) if history else "- (none)"

    return (
        "Generate ONE mock interview question.\n"
        "Constraints:\n"
        f"- This is question {index + 1} of {total}.\n"
        f"- {role_line}\n"
        "- Do NOT repeat or paraphrase previous questions.\n"
        "- Keep it concise (<= 22 words).\n"
        "- If a target role is provided, tailor the question to that role.\n\n"
        "Previous questions:\n"
        f"{history_txt}\n\n"
        "Return ONLY JSON in the exact schema: {\"question\": \"...\"}\n"
    )


def _parse_question(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    data = _parse_json_object(raw) or {}
    text = str(data.get("question") or "").strip()
    if not text:
        text = raw.splitlines()[0].strip()
    text = re.sub(r"\\s+", " ", text).strip()
    text = re.sub(r"^(question\\s*[:\\-]\\s*)", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("\"' \t")
    return text or None


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


if __name__ == "__main__":
    main()

