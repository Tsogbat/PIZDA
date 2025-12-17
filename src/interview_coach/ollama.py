from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


def host_candidates(host: str) -> tuple[str, ...]:
    base = _normalize_host(host)
    out: list[str] = [base]
    if "localhost" in base:
        out.append(base.replace("localhost", "127.0.0.1"))
    if "127.0.0.1" in base:
        out.append(base.replace("127.0.0.1", "localhost"))
    return tuple(dict.fromkeys(out))


def get_json(host: str, path: str, timeout_s: float) -> tuple[dict | None, str | None]:
    last_err: str | None = None
    for base in host_candidates(host):
        url = base.rstrip("/") + path
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_err = f"HTTP {getattr(e, 'code', '?')}"
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            continue

        if not body:
            last_err = last_err or "empty response"
            continue

        try:
            outer = json.loads(body)
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e}"

        if outer.get("error"):
            return outer, str(outer.get("error"))
        return outer, None
    return None, last_err or "request failed"


def post_json(host: str, path: str, payload: dict, timeout_s: float) -> tuple[dict | None, str | None]:
    last_err: str | None = None
    for base in host_candidates(host):
        url = base.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_err = f"HTTP {getattr(e, 'code', '?')}"
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            continue

        if not body:
            last_err = last_err or "empty response"
            continue

        try:
            outer = json.loads(body)
        except json.JSONDecodeError as e:
            return None, f"invalid JSON: {e}"

        if outer.get("error"):
            return outer, str(outer.get("error"))
        return outer, None
    return None, last_err or "request failed"


def extract_text(outer: dict) -> str:
    if "response" in outer:
        return str(outer.get("response") or "")
    msg = outer.get("message")
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return ""


def looks_like_format_error(err: str | None) -> bool:
    if not err:
        return False
    e = err.lower()
    if "format" not in e:
        return False
    return ("unsupported" in e) or ("unknown" in e) or ("invalid" in e) or ("not supported" in e)


_API_SUFFIX_RE = re.compile(r"(/api(/(generate|chat|tags|version))?)/*$", flags=re.IGNORECASE)


def _normalize_host(host: str) -> str:
    host = (host or "").strip()
    if not host:
        host = "http://localhost:11434"
    host = host.rstrip("/")
    host = _API_SUFFIX_RE.sub("", host)
    if not re.match(r"^https?://", host, flags=re.IGNORECASE):
        host = "http://" + host
    return host.rstrip("/")
