"""
LLM API client implementation.

- Prefer reading base_url, api_key, model, timeout from llm_config.json in this directory.
- If that file or fields are missing, fall back to environment variables: KLINE_LLM_*, KLINE_LLM_TIMEOUT.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 优先从 llm_config.json 读取，否则用环境变量
def _load_config() -> tuple:
    _dir = Path(__file__).resolve().parent
    _cfg_path = _dir / "llm_config.json"
    base_url = ""
    api_key = ""
    model = "gpt-4o-mini"
    timeout = 180
    if _cfg_path.exists():
        try:
            with open(_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_url = (cfg.get("base_url") or "").strip().rstrip("/")
            api_key = (cfg.get("api_key") or "").strip()
            model = (cfg.get("model") or model).strip() or "gpt-4o-mini"
            if "timeout" in cfg and cfg["timeout"] is not None:
                try:
                    timeout = int(cfg["timeout"])
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass
    if not base_url:
        base_url = (os.environ.get("KLINE_LLM_API_BASE") or "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        api_key = os.environ.get("KLINE_LLM_API_KEY") or ""
    model = (os.environ.get("KLINE_LLM_MODEL") or model or "gpt-4o-mini").strip()
    _t = os.environ.get("KLINE_LLM_TIMEOUT")
    if _t is not None:
        try:
            timeout = int(_t)
        except (TypeError, ValueError):
            pass
    return base_url, api_key, model, timeout

LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT = _load_config()
LLM_MAX_TOKENS = int(os.environ.get("KLINE_LLM_MAX_TOKENS") or "4096")


def is_configured() -> bool:
    """Return True if a usable LLM API is configured."""
    return bool(LLM_API_BASE and LLM_API_KEY)


def call_chat(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    timeout: Optional[int] = None,
) -> str:
    """
    Call an OpenAI-compatible chat/completions endpoint.
    messages: [
      {
        "role": "system" | "user" | "assistant",
        "content": "..." or
          [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
          ]
      },
      ...
    ]
    Returns the assistant's content text; raises exceptions on failure.
    """
    try:
        import urllib.request
        import ssl
    except ImportError:
        raise RuntimeError("Python standard library urllib.request is required.")

    url = f"{LLM_API_BASE}/chat/completions"
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
        method="POST",
    )
    _timeout = timeout if timeout is not None else LLM_TIMEOUT
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=_timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            err_obj = json.loads(body)
            msg = err_obj.get("error", {}).get("message", body)
        except Exception:
            msg = body
        raise RuntimeError(f"LLM API request failed ({e.code}): {msg}")
    except Exception as e:
        raise RuntimeError(f"LLM API request error: {e}") from e

    body_stripped = (body or "").strip()
    if not body_stripped:
        raise RuntimeError("LLM API returned an empty response. Please check base_url, network, and server status.")

    try:
        out = json.loads(body_stripped)
    except json.JSONDecodeError as e:
        preview = body_stripped[:500] if len(body_stripped) > 500 else body_stripped
        raise RuntimeError(
            f"LLM API returned non-JSON data (Expecting value). The endpoint may be incorrect or returned HTML / an error page. Preview: {preview!r}"
        ) from e

    choices = out.get("choices") or []
    if not choices:
        raise RuntimeError("LLM API returned no choices.")
    content = (choices[0].get("message") or {}).get("content")
    if content is None:
        raise RuntimeError("LLM API returned no content.")
    return content.strip()
