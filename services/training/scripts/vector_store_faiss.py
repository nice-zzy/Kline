"""
FAISS 向量存储工具模块

功能：
1. 构建 FAISS 索引：支持内积（Inner Product）和 L2 距离两种度量方式
2. 向量归一化：L2 归一化，用于余弦相似度计算
3. Top-K 搜索：快速检索最相似的向量
4. 索引持久化：保存和加载 FAISS 索引文件
5. 元数据管理：保存和加载向量对应的元数据（如日期、价格等信息）

使用场景：
- 推理阶段：对大量 K 线图向量进行相似度检索
- 相似对匹配：在 find_similar_pairs_faiss.py 中使用，加速大规模相似对查找
- 向量数据库：构建 K 线图相似性搜索的向量数据库

依赖：
- faiss-cpu 或 faiss-gpu（需要单独安装）
- numpy

示例：
    # 构建索引
    embeddings = np.array([[0.1, 0.2, ...], ...])  # (n, dim)
    index = build_index(embeddings, metric="ip", use_normalize=True)
    save_index(index, "index.faiss")
    
    # 搜索
    query = np.array([[0.15, 0.25, ...]])  # (1, dim)
    scores, indices = topk_search(index, query, k=10)
    
    # 加载元数据
    metadata = load_metadata("metadata.json")
    result = metadata[indices[0][0]]  # 获取第一个结果的信息
"""
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
	import faiss  # type: ignore
except Exception as e:
	faiss = None


def _ensure_faiss():
	if faiss is None:
		raise ImportError("faiss is not installed. Please install faiss-cpu to continue.")


def l2_normalize(x: np.ndarray) -> np.ndarray:
	"""Row-wise L2 normalize."""
	norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
	return x / norm


def build_index(embeddings: np.ndarray, metric: str = "ip", use_normalize: bool = True):
	"""Build a FAISS index.
	- metric: "ip" (inner product, for cosine if normalized) or "l2".
	- use_normalize: if True and metric=="ip", normalize vectors for cosine similarity.
	"""
	_ensure_faiss()
	vecs = embeddings.astype(np.float32)
	if metric == "ip" and use_normalize:
		vecs = l2_normalize(vecs)
		d = vecs.shape[1]
		index = faiss.IndexFlatIP(d)
		index.add(vecs)
		return index
	elif metric == "ip":
		d = vecs.shape[1]
		index = faiss.IndexFlatIP(d)
		index.add(vecs)
		return index
	elif metric == "l2":
		d = vecs.shape[1]
		index = faiss.IndexFlatL2(d)
		index.add(vecs)
		return index
	else:
		raise ValueError(f"Unsupported metric: {metric}")


def save_index(index, out_path: str):
	"""Save FAISS index to file."""
	_ensure_faiss()
	out = Path(out_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(out))


def load_index(index_path: str):
	"""Load FAISS index from file."""
	_ensure_faiss()
	return faiss.read_index(index_path)


def topk_search(index, queries: np.ndarray, k: int = 10, metric: str = "ip", use_normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
	"""Search TopK results.
	Returns (scores, indices)
	"""
	_ensure_faiss()
	q = queries.astype(np.float32)
	if metric == "ip" and use_normalize:
		q = l2_normalize(q)
	D, I = index.search(q, k)
	return D, I


def save_metadata(meta: List[Dict[str, Any]], out_path: str):
	out = Path(out_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	with open(out, 'w', encoding='utf-8') as f:
		json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def load_metadata(path: str) -> List[Dict[str, Any]]:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)
