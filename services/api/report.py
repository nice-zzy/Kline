"""
规则 + 模板：根据编码与检索结果生成分析报告（全文与模式化结构）。
不依赖大模型，仅用固定模板与相似度档位规则。
"""
from typing import List, Dict, Any, Optional, TypedDict


# ---------- 模式化输出结构 ----------
class SimilarCaseRow(TypedDict, total=False):
    rank: int
    symbol: Optional[str]
    start_date: str
    end_date: str
    similarity: float
    summary: str  # 一行摘要，如 "AAPL | 2012-01-01 ~ 2012-01-05 （相似度 85.00%）"


class ReportSections(TypedDict):
    形态判断: str
    相似历史案例: str
    关键价位与建议: str
    风险提示: str


class ReportStructured(TypedDict):
    sections: ReportSections
    similar_cases: List[SimilarCaseRow]
    full_text: str


# ---------- 规则与模板 ----------
def _similarity_label(sim: float) -> str:
    """根据相似度返回形态描述（规则档位）。"""
    if sim >= 0.90:
        return "高度相似"
    if sim >= 0.80:
        return "较为相似"
    if sim >= 0.70:
        return "中等相似"
    if sim >= 0.55:
        return "略有相似"
    return "形态差异较大"


def _format_case(meta: Dict[str, Any], rank: int) -> str:
    """将一条检索元数据格式化为「相似历史案例」的一行。"""
    parts = []
    if meta.get("symbol"):
        parts.append(meta["symbol"])
    if meta.get("start_date") or meta.get("end_date"):
        parts.append(f"{meta.get('start_date', '')} ~ {meta.get('end_date', '')}".strip(" ~"))
    if meta.get("price_change") is not None:
        try:
            pct = float(meta["price_change"]) * 100
            parts.append(f"区间涨跌 {pct:+.2f}%")
        except (TypeError, ValueError):
            pass
    return f"{rank}. " + " | ".join(parts) if parts else f"{rank}. （无元数据）"


def _section_morphology(top_similarity: Optional[float]) -> str:
    """形态判断段落。"""
    if top_similarity is not None:
        label = _similarity_label(top_similarity)
        return f"{label}（与历史窗口相似度 {top_similarity:.2%}）。"
    return "已识别K线形态，当前未配置历史检索库，无法给出相似度结论。"


def _section_similar_cases(
    has_retrieval: bool,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
    similarities: Optional[List[float]] = None,
) -> str:
    """相似历史案例段落（纯文本）。"""
    if has_retrieval and similar_cases and len(similar_cases) > 0:
        sims = similarities or []
        lines = []
        for i, meta in enumerate(similar_cases[:10], start=1):
            sim_str = f"（相似度 {sims[i-1]:.2%}）" if i - 1 < len(sims) else ""
            lines.append("  " + _format_case(meta, i) + " " + sim_str)
        return "\n".join(lines)
    return "  暂无（未配置检索库或未找到相似窗口）。"


SECTION_KEY_MORPHOLOGY = "形态判断"
SECTION_KEY_SIMILAR = "相似历史案例"
SECTION_KEY_ADVICE = "关键价位与建议"
SECTION_KEY_RISK = "风险提示"

TEMPLATE_ADVICE = (
    "  - 可结合当前窗口高低点与均线判断支撑/阻力。\n"
    "  - 若与历史相似形态一致，可参考当时后续走势作为参考，不构成投资建议。"
)
TEMPLATE_RISK = (
    "  - 历史形态相似不代表未来重复；放量下破关键均线或前低时形态失效。\n"
    "  - 请结合宏观、基本面与自身风险承受能力决策。"
)


def build_report_structured(
    *,
    has_retrieval: bool,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
    similarities: Optional[List[float]] = None,
    top_similarity: Optional[float] = None,
) -> ReportStructured:
    """
    生成模式化报告：分块文案 + 相似案例列表 + 全文。
    供 API 返回 JSON 使用。
    """
    # 各段落
    sec_morph = _section_morphology(top_similarity)
    sec_similar = _section_similar_cases(has_retrieval, similar_cases, similarities)
    sec_advice = TEMPLATE_ADVICE
    sec_risk = TEMPLATE_RISK

    sections: ReportSections = {
        SECTION_KEY_MORPHOLOGY: sec_morph,
        SECTION_KEY_SIMILAR: sec_similar,
        SECTION_KEY_ADVICE: sec_advice,
        SECTION_KEY_RISK: sec_risk,
    }

    # 相似案例结构化列表（供前端表格/列表展示）
    similar_rows: List[SimilarCaseRow] = []
    if has_retrieval and similar_cases and len(similar_cases) > 0:
        sims = similarities or []
        for i, meta in enumerate(similar_cases[:10], start=1):
            sim = float(sims[i - 1]) if i - 1 < len(sims) else 0.0
            summary = _format_case(meta, i) + f" （相似度 {sim:.2%}）"
            similar_rows.append({
                "rank": i,
                "symbol": meta.get("symbol"),
                "start_date": meta.get("start_date", ""),
                "end_date": meta.get("end_date", ""),
                "similarity": sim,
                "summary": summary,
            })

    # 拼全文（与原有 build_report 一致）
    full_lines = [
        "【K线形态分析报告】",
        "",
        "形态判断：" + sec_morph,
        "",
        "相似历史案例：",
        sec_similar,
        "",
        "关键价位与建议：",
        sec_advice,
        "",
        "风险提示：",
        sec_risk,
    ]
    full_text = "\n".join(full_lines)

    return {
        "sections": sections,
        "similar_cases": similar_rows,
        "full_text": full_text,
    }


def build_report(
    *,
    has_retrieval: bool,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
    similarities: Optional[List[float]] = None,
    top_similarity: Optional[float] = None,
) -> str:
    """
    根据是否有检索结果、相似案例列表与相似度，拼出完整报告文案（纯文本）。
    兼容旧接口。
    """
    structured = build_report_structured(
        has_retrieval=has_retrieval,
        similar_cases=similar_cases,
        similarities=similarities,
        top_similarity=top_similarity,
    )
    return structured["full_text"]
