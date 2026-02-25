"""
K线分析 API：HTTP 路由层。
- 推理逻辑在 inference 模块；本文件只负责接收请求、调用 inference、返回响应。
- 接收上传的 K 线图 → inference.run_inference → 规则化报告（全文 + 分块 + 相似案例列表）。
"""
import sys
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_DIR = Path(__file__).resolve().parent
TRAINING_DIR = PROJECT_ROOT / "services" / "training"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from inference import run_inference, run_inference_with_llm, encode_image

app = FastAPI(title="KLine Similarity Demo API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 响应模型（模式化输出） ----------
class ReportSectionsModel(BaseModel):
    形态判断: str
    相似历史案例: str
    关键价位与建议: str
    风险提示: str


class SimilarCaseModel(BaseModel):
    rank: int
    symbol: Optional[str] = None
    start_date: str = ""
    end_date: str = ""
    similarity: float = 0.0
    summary: str = ""
    image_base64: Optional[str] = None  # 当前窗口图（与向量一致）
    image_next1_base64: Optional[str] = None  # 往后第 1 个窗口图（步长 3，同 all_window_images）
    image_next2_base64: Optional[str] = None  # 往后第 2 个窗口图


class ReportModel(BaseModel):
    sections: ReportSectionsModel
    similar_cases: List[SimilarCaseModel]
    full_text: str


class AnalyzeResponseLegacy(BaseModel):
    """兼容旧版：仅返回 text。"""
    text: str


class AnalyzeResponse(BaseModel):
    """模式化输出：成功/失败 + 报告分块 + 相似案例列表 + 全文。"""
    success: bool
    message: str
    text: str  # 与 full_text 一致，兼容前端直接取 text
    report: Optional[ReportModel] = None
    similar_cases: Optional[List[SimilarCaseModel]] = None
    error_code: Optional[str] = None


class EmbedResponse(BaseModel):
    embedding: list[float]


# ---------- 路由 ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    message: str = Form(""),
    history: str = Form("[]"),
):
    """
    分析入口：支持「仅图片」「仅文字」「图片+文字」，以及多轮对话。
    - 仅图片：默认对该 K 线图做形态分析与未来走向预测；报告为图文（含检索到的相似 K 线图，注明股票与日期）。
    - 有文字：理解语义并针对性回复；支持多轮对话（通过 history 传入此前轮次）。
    """
    image_bytes: Optional[bytes] = None
    user_message: str = (message or "").strip()
    if file and file.filename:
        image_bytes = await file.read()
    conversation_history: List[dict] = []
    try:
        if history.strip():
            conversation_history = json.loads(history)
        if not isinstance(conversation_history, list):
            conversation_history = []
    except Exception:
        conversation_history = []

    if not image_bytes and not user_message.strip():
        return AnalyzeResponse(
            success=False,
            message="请上传 K 线图或输入文字诉求（至少其一）",
            text="请上传 K 线图或输入文字诉求（至少其一）",
            report=None,
            similar_cases=None,
            error_code="MISSING_INPUT",
        )

    out = run_inference_with_llm(
        image_bytes=image_bytes,
        user_message=user_message,
        conversation_history=conversation_history or None,
    )

    if not out["success"]:
        return AnalyzeResponse(
            success=False,
            message=out["message"],
            text=out["message"],
            report=None,
            similar_cases=None,
            error_code=out["error_code"],
        )

    report = out["report"]
    return AnalyzeResponse(
        success=True,
        message=out["message"],
        text=report["full_text"],
        report=ReportModel(
            sections=ReportSectionsModel(**report["sections"]),
            similar_cases=[SimilarCaseModel(**c) for c in report["similar_cases"]],
            full_text=report["full_text"],
        ),
        similar_cases=[SimilarCaseModel(**c) for c in report["similar_cases"]],
        error_code=None,
    )


@app.post("/analyze/legacy", response_model=AnalyzeResponseLegacy)
async def analyze_image_legacy(file: UploadFile = File(...)):
    """兼容旧前端：仅返回 text。"""
    content = await file.read()
    out = run_inference(content)
    if not out["success"]:
        return AnalyzeResponseLegacy(text=out["message"])
    return AnalyzeResponseLegacy(text=out["report"]["full_text"])


@app.post("/embed", response_model=EmbedResponse)
async def embed_image(file: UploadFile = File(...)):
    content = await file.read()
    embedding = encode_image(content)
    return EmbedResponse(embedding=embedding if embedding else [])
