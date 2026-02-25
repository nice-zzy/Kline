"""
大模型 API 调用实现（本文件即「API 调用代码」所在位置）。

- 优先从同目录下的 llm_config.json 读取 base_url、api_key、model；若无该文件或字段为空则用环境变量。
- 配置项（JSON）：base_url, api_key, model, timeout（秒）；环境变量：KLINE_LLM_*、KLINE_LLM_TIMEOUT
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
    """是否已配置可用的 LLM API。"""
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
    调用 OpenAI 兼容的 chat/completions 接口。
    messages: [ {"role": "system"|"user"|"assistant", "content": "..." 或 [{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}] }, ... ]
    返回 assistant 的 content 文本；失败抛异常或返回错误信息。
    """
    try:
        import urllib.request
        import ssl
    except ImportError:
        raise RuntimeError("需要 Python 标准库 urllib.request")

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
        raise RuntimeError(f"LLM API 请求失败 ({e.code}): {msg}")
    except Exception as e:
        raise RuntimeError(f"LLM API 请求异常: {e}") from e

    body_stripped = (body or "").strip()
    if not body_stripped:
        raise RuntimeError("LLM API 返回空响应，请检查 base_url 与网络或服务端是否正常")

    try:
        out = json.loads(body_stripped)
    except json.JSONDecodeError as e:
        preview = body_stripped[:500] if len(body_stripped) > 500 else body_stripped
        raise RuntimeError(
            f"LLM API 返回非 JSON（Expecting value）。可能原因：接口地址错误、返回了 HTML/错误页。响应预览: {preview!r}"
        ) from e

    choices = out.get("choices") or []
    if not choices:
        raise RuntimeError("LLM API 返回无 choices")
    content = (choices[0].get("message") or {}).get("content")
    if content is None:
        raise RuntimeError("LLM API 返回无 content")
    return content.strip()
