"""
推理模块：图片 → 编码 → 检索 → 报告。
- 支持模板报告（build_report_structured）与大模型报告（run_inference_with_llm）。
- 大模型模式：图片编码 → 检索 top5 相似历史 → 用户诉求 + 检索结果 → LLM 生成预测与图文报告。
"""
import os
import re
import sys
import json
import base64
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import numpy as np
from PIL import Image
import io

# 路径与导入（与 main 一致，便于单独跑或换入口）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_DIR = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "services" / "training"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from report import build_report_structured


def _collapse_newlines(text: str) -> str:
    """将换行及其间空白（如 \\n  \\n、\\n\\n）压缩成单个 \\n。"""
    if not text:
        return text
    s = re.sub(r"\r\n?|\r", "\n", text)
    return re.sub(r"\n\s*\n", "\n", s)


# ---------- 配置（从环境变量读取，未设置时用项目内默认路径） ----------
# 默认使用 vicreg（logs/vicreg），与 main 步骤4/5 及 conversation_context 约定一致；非 vicreg_25
CHECKPOINT_DIR = os.environ.get("KLINE_CHECKPOINT_DIR") or str(TRAINING_DIR / "logs" / "vicreg")
CHECKPOINT_PATH = os.environ.get("KLINE_CHECKPOINT_PATH") or str(Path(CHECKPOINT_DIR) / "checkpoint_best.pth")
# 检索索引：优先 KLINE_INDEX_DIR；否则尝试 dow30_2010_2021/inference_index_vicreg（main --steps 5 且 training_method=vicreg 时的输出）
_DEFAULT_INDEX_DIR = TRAINING_DIR / "output" / "dow30_2010_2021" / "inference_index_vicreg"
INDEX_DIR = os.environ.get("KLINE_INDEX_DIR") or (
    str(_DEFAULT_INDEX_DIR) if (_DEFAULT_INDEX_DIR / "embeddings.npy").exists() and (_DEFAULT_INDEX_DIR / "metadata.json").exists() else None
)
# 全体窗口图像（用于图文报告中展示检索到的相似 K 线图）
# 优先环境变量 KLINE_ALL_IMAGES_FILE；否则用索引目录上一级 / all_window_images.npy
_ALL_IMAGES_PATH: Optional[Path] = None
_env_images = os.environ.get("KLINE_ALL_IMAGES_FILE")
if _env_images and Path(_env_images).exists():
    _ALL_IMAGES_PATH = Path(_env_images).resolve()
elif INDEX_DIR:
    _parent = Path(INDEX_DIR).resolve().parent
    _cand = _parent / "all_window_images.npy"
    if _cand.exists():
        _ALL_IMAGES_PATH = _cand
IMAGE_SIZE = 224

_encoder = None
_index_embeddings = None
_index_metadata = None
_all_window_images: Optional[np.ndarray] = None


def _load_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"检查点不存在: {CHECKPOINT_PATH}")
    from inference_encoder import TrainedEncoder
    _encoder = TrainedEncoder(CHECKPOINT_PATH, device="auto")
    return _encoder


def _load_index() -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
    global _index_embeddings, _index_metadata
    if not INDEX_DIR:
        return None, None
    if _index_embeddings is not None:
        return _index_embeddings, _index_metadata
    p = Path(INDEX_DIR)
    emb_file = p / "embeddings.npy"
    meta_file = p / "metadata.json"
    if not emb_file.exists() or not meta_file.exists():
        return None, None
    _index_embeddings = np.load(emb_file).astype(np.float32)
    with open(meta_file, "r", encoding="utf-8") as f:
        _index_metadata = json.load(f)
    if len(_index_metadata) != len(_index_embeddings):
        _index_embeddings = None
        _index_metadata = None
        return None, None
    return _index_embeddings, _index_metadata


def image_to_array(content: bytes, size: int = IMAGE_SIZE) -> np.ndarray:
    """上传图片字节 → RGB numpy [H,W,3] 0-255，已 resize 为 size x size。"""
    img = Image.open(io.BytesIO(content))
    img = img.convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norm).astype(np.float32)


def _topk_similarity(query: np.ndarray, candidates: np.ndarray, k: int = 10):
    q = query.reshape(1, -1).astype(np.float32)
    q = _l2_normalize(q)
    c = _l2_normalize(candidates)
    sims = np.dot(c, q.T).squeeze(1)
    top_indices = np.argsort(sims)[::-1][:k]
    return sims[top_indices].tolist(), top_indices.tolist()


def _load_all_window_images() -> Optional[np.ndarray]:
    """懒加载全体窗口图像（用于图文报告中的相似案例配图）。使用 mmap 避免整块读入内存。"""
    global _all_window_images
    if _all_window_images is not None:
        return _all_window_images
    if _ALL_IMAGES_PATH is None:
        return None
    try:
        _all_window_images = np.load(str(_ALL_IMAGES_PATH), mmap_mode="r")
        return _all_window_images
    except Exception:
        return None


def _get_similar_case_image_base64(window_index: int) -> Optional[str]:
    """根据 window_index 从 all_window_images 取一张图，转为 PNG base64。"""
    arr = _load_all_window_images()
    if arr is None:
        return None
    if window_index < 0 or window_index >= len(arr):
        return None
    try:
        frame = np.asarray(arr[window_index])
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        pil = Image.fromarray(frame)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _similar_cases_to_llm_text(similar_cases: List[dict], similarities: List[float]) -> str:
    """将检索到的相似案例格式化为给大模型看的文本表格。"""
    if not similar_cases:
        return "（暂无相似历史案例）"
    lines = []
    for i, meta in enumerate(similar_cases):
        sim = similarities[i] if i < len(similarities) else 0.0
        parts = [f"第{i+1}名", f"相似度 {sim:.2%}"]
        if meta.get("symbol"):
            parts.append(str(meta["symbol"]))
        if meta.get("start_date") or meta.get("end_date"):
            parts.append(f"{meta.get('start_date', '')} ~ {meta.get('end_date', '')}".strip(" ~"))
        if meta.get("price_change") is not None:
            try:
                pct = float(meta["price_change"]) * 100
                parts.append(f"区间涨跌 {pct:+.2f}%")
            except (TypeError, ValueError):
                pass
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def run_inference_with_llm(
    image_bytes: Optional[bytes] = None,
    user_message: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    大模型报告流程：可选图片编码与 top5 检索 → 用户诉求 + 检索结果 → LLM 生成报告。
    若未配置 LLM（KLINE_LLM_API_BASE + KLINE_LLM_API_KEY）则回退到模板报告。
    返回格式与 run_inference 一致：success, message, report (sections, similar_cases, full_text), embedding (可选), error_code。
    """
    from report import (
        ReportStructured,
        SECTION_KEY_MORPHOLOGY,
        SECTION_KEY_SIMILAR,
        SECTION_KEY_ADVICE,
        SECTION_KEY_RISK,
        _format_case,
        _section_similar_cases,
    )

    similar_cases: List[dict] = []
    similarities: List[float] = []
    top_similarity: Optional[float] = None
    embedding_list: Optional[List[float]] = None

    if image_bytes:
        try:
            img_array = image_to_array(image_bytes)
        except Exception as e:
            return {
                "success": False,
                "message": f"上传文件不是有效图片: {e}",
                "report": None,
                "embedding": None,
                "error_code": "INVALID_IMAGE",
            }
        try:
            encoder = _load_encoder()
        except Exception as e:
            return {
                "success": False,
                "message": f"模型未就绪: {e}",
                "report": None,
                "embedding": None,
                "error_code": "MODEL_NOT_READY",
            }
        try:
            embedding = encoder.encode_image(img_array)
            embedding_list = embedding.tolist()
        except Exception as e:
            return {
                "success": False,
                "message": f"编码失败: {e}",
                "report": None,
                "embedding": None,
                "error_code": "ENCODE_FAILED",
            }
        emb_np = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        index_emb, index_meta = _load_index()
        if index_emb is not None and index_meta is not None and len(index_meta) > 0:
            scores, indices = _topk_similarity(emb_np, index_emb, k=5)
            similar_cases = [index_meta[i] for i in indices]
            similarities = scores
            top_similarity = float(scores[0]) if scores else None

    # 若既无图片也无文字，且未配置 LLM，需要至少一种输入
    try:
        from llm_client import is_configured, call_chat
    except ImportError:
        call_chat = None
        is_configured = lambda: False

    if not is_configured():
        # Fallback: use template report (same as run_inference)
        report = build_report_structured(
            has_retrieval=len(similar_cases) > 0,
            similar_cases=similar_cases or None,
            similarities=similarities or None,
            top_similarity=top_similarity,
        )
        return {
            "success": True,
            "message": "Analysis completed (LLM not configured, used template report).",
            "report": report,
            "embedding": embedding_list,
            "error_code": None,
        }

    # Build context for the LLM: support multimodal (user candlestick + three images per similar case)
    system_prompt = (
        "You are a professional candlestick pattern and stock analysis assistant. "
        "The user will provide one candlestick chart, plus several similar historical cases "
        "(each case contains 3 images: current window, next window 1, next window 2). "
        "Use these images to perform an analogy-based analysis and predict the future movement of the user's chart. "
        "The report must include: 1) a brief assessment of the current pattern; "
        "2) comparison with similar historical cases; "
        "3) reference for subsequent price movement and future outlook based on those historical patterns (for reference only); "
        "4) key price levels or trading suggestions (if any); "
        "5) risk warnings. "
        "Write in English, be well-structured, avoid absolute statements, and clearly state that this is not investment advice. "
        "For multi-turn dialogue, consider the conversation history and reply coherently. "
        "[Important] Output in Markdown: use ## for section headings, - or 1. for lists, and **bold** for key points so that the frontend can render it properly. "
        "[Formatting] Layout should be compact: at most one blank line between paragraphs; no multiple consecutive blank lines; "
        "no blank lines between similar cases."
    )

    # 是否有图可发（用户图或相似案例图）：有则用 vision 多模态，无则用纯文本
    has_user_image = bool(image_bytes)
    has_similar_images = bool(similar_cases) and any(
        _get_similar_case_image_base64(meta.get("window_index", -1)) for meta in similar_cases[:5]
    )
    use_vision = has_user_image or has_similar_images

    if use_vision:
        content_parts: List[Dict[str, Any]] = []
        if image_bytes:
            content_parts.append({
                "type": "text",
                "text": (
                    "The user-uploaded candlestick chart is shown below. "
                    "Please analyze it together with the retrieved similar historical cases "
                    "(each case has 3 images: current window, next window 1, next window 2) "
                    "and predict the future movement."
                ),
            })
            buf = io.BytesIO()
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        elif similar_cases:
            content_parts.append({
                "type": "text",
                "text": "The user did not upload a chart. Please use the similar historical candlestick charts below as reference for future movement.\n",
            })

        if similar_cases:
            for k, meta in enumerate(similar_cases[:5], 1):
                sim = float(similarities[k - 1]) if k - 1 < len(similarities) else 0.0
                sym = meta.get("symbol", "")
                sd = meta.get("start_date", "")
                ed = meta.get("end_date", "")
                content_parts.append({
                    "type": "text",
                    "text": (
                        f"Similar case {k}: {sym} {sd}–{ed}, similarity {sim:.2%}. "
                        "The following three images are: current window, next window 1, next window 2."
                    ),
                })
                wi = meta.get("window_index", -1)
                if wi >= 0:
                    for b64 in [
                        _get_similar_case_image_base64(wi),
                        _get_similar_case_image_base64(wi + 1),
                        _get_similar_case_image_base64(wi + 2),
                    ]:
                        if b64:
                            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        if user_message.strip():
            content_parts.append({"type": "text", "text": "User's additional requirements or questions:\n" + user_message.strip()})
        if not content_parts:
            content_parts.append({"type": "text", "text": "No retrieval index is configured or no similar historical window was found."})

        user_content: Any = content_parts
    else:
        context_parts = []
        if image_bytes and not user_message.strip():
            context_parts.append(
                "The user uploaded only one candlestick chart without any text. "
                "Please analyze the pattern and predict the future movement by default."
            )
        elif image_bytes:
            context_parts.append(
                "The user uploaded a candlestick chart. Please understand its meaning and respond accordingly."
            )
        if similar_cases:
            context_parts.append(
                "Retrieved similar historical windows (text only):\n" + _similar_cases_to_llm_text(similar_cases, similarities)
            )
        else:
            context_parts.append("No retrieval index is configured or no similar historical window was found.")
        if user_message.strip():
            context_parts.append("User question or requirements:\n" + user_message.strip())
        user_content = "\n\n".join(context_parts)

    messages_for_llm: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        max_turns = 10
        for h in conversation_history[-max_turns:]:
            role = (h.get("role") or "user").strip().lower()
            if role == "assistant":
                messages_for_llm.append({"role": "assistant", "content": (h.get("content") or "")[:8000]})
            else:
                messages_for_llm.append({"role": "user", "content": (h.get("content") or "")[:4000]})
    messages_for_llm.append({"role": "user", "content": user_content})

    try:
        llm_text = call_chat(messages_for_llm)
    except Exception as e:
        return {
            "success": False,
            "message": f"LLM call failed: {e}",
            "report": None,
            "embedding": embedding_list,
            "error_code": "LLM_ERROR",
        }

    # 组装与 run_inference 一致的报告结构（图文：每条相似案例带 base64 图，便于前端展示并注明股票与日期）
    sec_similar = _section_similar_cases(bool(similar_cases), similar_cases, similarities)
    similar_rows: List[Dict[str, Any]] = []
    for i, meta in enumerate(similar_cases[:10], start=1):
        sim = float(similarities[i - 1]) if i - 1 < len(similarities) else 0.0
        row: Dict[str, Any] = {
            "rank": i,
            "symbol": meta.get("symbol"),
            "start_date": meta.get("start_date", ""),
            "end_date": meta.get("end_date", ""),
            "similarity": sim,
            "summary": _format_case(meta, i) + f" （相似度 {sim:.2%}）",
        }
        wi = meta.get("window_index", -1)
        if wi >= 0:
            img_b64 = _get_similar_case_image_base64(wi)
            if img_b64:
                row["image_base64"] = img_b64
            # 往后两个窗口（与向量一致：步长 3，同一 all_window_images）直接按 i+1、i+2 取图
            n1 = _get_similar_case_image_base64(wi + 1)
            if n1:
                row["image_next1_base64"] = n1
            n2 = _get_similar_case_image_base64(wi + 2)
            if n2:
                row["image_next2_base64"] = n2
        similar_rows.append(row)

    llm_text_clean = _collapse_newlines(llm_text or "")
    first_line = (llm_text_clean.split("\n")[0] if llm_text_clean else "").strip() or "See full report below."
    sections: Dict[str, str] = {
        SECTION_KEY_MORPHOLOGY: first_line,
        SECTION_KEY_SIMILAR: sec_similar,
        SECTION_KEY_ADVICE: "Please refer to the above LLM analysis for suggestions and risk notes.",
        SECTION_KEY_RISK: "Historical pattern similarity does not guarantee future repetition and does not constitute investment advice.",
    }
    # Full report is primarily the LLM output (with collapsed newlines)
    full_text = "【K-line pattern and future trend analysis report】\n\n" + llm_text_clean

    report: ReportStructured = {
        "sections": sections,
        "similar_cases": similar_rows,
        "full_text": full_text,
    }

    return {
        "success": True,
        "message": "分析完成",
        "report": report,
        "embedding": embedding_list,
        "error_code": None,
    }


def run_inference(image_bytes: bytes) -> dict:
    """
    推理全流程：校验图片 → 编码 → 检索 → 模式化报告。
    返回：{
      "success": bool,
      "message": str,
      "report": { "sections", "similar_cases", "full_text" } | None,
      "embedding": list[float] | None,
      "error_code": str | None,
    }
    """
    try:
        img_array = image_to_array(image_bytes)
    except Exception as e:
        return {
            "success": False,
            "message": f"上传文件不是有效图片，请重新选择。错误: {e}",
            "report": None,
            "embedding": None,
            "error_code": "INVALID_IMAGE",
        }

    try:
        encoder = _load_encoder()
    except Exception as e:
        return {
            "success": False,
            "message": f"模型未就绪，请先完成训练并确认检查点存在。错误: {e}",
            "report": None,
            "embedding": None,
            "error_code": "MODEL_NOT_READY",
        }

    try:
        embedding = encoder.encode_image(img_array)
    except Exception as e:
        return {
            "success": False,
            "message": f"编码失败: {e}",
            "report": None,
            "embedding": None,
            "error_code": "ENCODE_FAILED",
        }

    emb_np = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    index_emb, index_meta = _load_index()

    if index_emb is not None and index_meta is not None and len(index_meta) > 0:
        scores, indices = _topk_similarity(emb_np, index_emb, k=10)
        similar_cases = [index_meta[i] for i in indices]
        similarities = scores
        top_sim = float(scores[0]) if scores else None
        report = build_report_structured(
            has_retrieval=True,
            similar_cases=similar_cases,
            similarities=similarities,
            top_similarity=top_sim,
        )
    else:
        report = build_report_structured(
            has_retrieval=False,
            similar_cases=None,
            similarities=None,
            top_similarity=None,
        )

    return {
        "success": True,
        "message": "分析完成",
        "report": report,
        "embedding": embedding.tolist(),
        "error_code": None,
    }


def encode_image(image_bytes: bytes) -> Optional[List[float]]:
    """仅编码：图片字节 → 512 维向量。失败返回 None。供 /embed 等使用。"""
    try:
        img_array = image_to_array(image_bytes)
        encoder = _load_encoder()
        embedding = encoder.encode_image(img_array)
        return embedding.tolist()
    except Exception:
        return None
