"""
模型验证脚本：用训练好的最优模型在测试集上做相似检索评估。

- 候选库：默认=测试集中所有 positive 图像；若提供 all_images_file 或 inference_index_dir，则候选库=2010-2021 全量窗口，在全量中检索更符合真实场景。
- 对每个测试 anchor，在候选库中按相似度排序，看「真实配对的那张 positive（窗口索引 positive_idx）」是否在 Top-K 内。
- max_pairs_per_anchor=3，故默认只报告 Recall@1 与 Recall@3；需要 Recall@5 时可传 --top-k 1 3 5。

指标：
- 检索：Recall@1, Recall@3（默认；真实 positive 在 top-k 内的比例）
- 原始测试集相似度（两套标准）：(1) 编码器相似度：6225 对 (anchor, 真实 positive) 的编码器余弦相似度均值/标准差/范围；(2) 52 维相似度：同上 6225 对用制作相似对时的 52 维特征算余弦相似度均值/标准差/范围（需提供 features_52d）。
- 相似度：配对 (anchor, positive) 的余弦相似度均值/标准差/分布（高≥0.9、中0.7~0.9、低<0.7）

52 维与「只做编码器」：
- 52 维特征由 main.py 步骤 1 保存到 Config.features_file（默认 services/training/output/<output_root>/features_52d.npy），
  步骤 2 找相似对时读的即该文件。本脚本若提供该路径，会额外用 52 维算相似度/Recall，与编码器结果对比。
- 「只做编码器」：传 --features-52d none 时，不加载 52 维文件，仅用训练好的图像编码器算相似度与 Recall，不输出 52 维相关指标。

直接运行（使用下方默认路径）：
  python services/training/evaluate/validate_model.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import json
from typing import List, Dict, Tuple

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(script_dir))

from pair_dataset import SimilarPairDataset
from inference_encoder import TrainedEncoder

# ---------- 默认路径（相对项目根 kline/），可直接运行脚本 ----------
# 52 维特征：与 main.py Config.features_file 一致，步骤 1 输出、步骤 2 找相似对所用
FEATURES_52D_RELATIVE = "services/training/output/dow30_2010_2021/features_52d.npy"
DEFAULT_CONFIG = {
    "checkpoint": "services/training/logs/vicreg/checkpoint_best.pth",
    "test_anchor_images": "services/training/output/dow30_2010_2021/dataset_splits/test_anchor_images.npy",
    "test_positive_images": "services/training/output/dow30_2010_2021/dataset_splits/test_positive_images.npy",
    "test_pairs_metadata": "services/training/output/dow30_2010_2021/dataset_splits/test_pairs_metadata.json",
    "top_k_list": [1, 3],  # max_pairs_per_anchor=3，只算 Recall@1 / Recall@3
    "features_52d": FEATURES_52D_RELATIVE,
    "all_images_file": "services/training/output/dow30_2010_2021/all_window_images.npy",  # 全量窗口图，提供则候选库=2010-2021全量
    "inference_index_dir": None,  # 若提供（如 steps 5 后），直接用作全量候选库，避免重复编码
}


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (project_root / path).resolve()
    return p


def _cosine_similarity_52d(
    features: np.ndarray, pairs_metadata: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用制作相似对时的 52 维特征计算每对 (anchor, positive) 的余弦相似度。
    features: [N_windows, 52]，pairs_metadata 中 anchor_idx/positive_idx 为窗口索引。
    返回 (anchor_feats, positive_feats) 用于检索；以及每对配对相似度可用 anchor_feats[i]·positive_feats[i]。
    """
    anchor_feats = np.array(
        [features[p["anchor_idx"]] for p in pairs_metadata], dtype=np.float64
    )
    positive_feats = np.array(
        [features[p["positive_idx"]] for p in pairs_metadata], dtype=np.float64
    )
    return anchor_feats, positive_feats


def evaluate_retrieval_accuracy(
    encoder: TrainedEncoder,
    anchor_images_file: str,
    positive_images_file: str,
    pairs_metadata_file: str,
    top_k_list: List[int] = (1, 3),
    features_52d_file: str = None,
    demo_save_dir: str = None,
    all_images_file: str = None,
    inference_index_dir: str = None,
) -> Dict:
    """
    评估检索准确率。候选库可为「测试集 positive」或「2010-2021 全量窗口」。

    对每个测试 anchor，在候选库中按相似度排序，看真实配对的那张 positive（窗口索引 positive_idx）是否在 Top-K 内。
    若提供 demo_save_dir：将编码后的向量保存为该目录下的 demo 候选库（供 demo 系统检索使用）。

    Args:
        encoder: 训练好的编码器
        anchor_images_file: 测试集 Anchor 图片
        positive_images_file: 测试集 Positive 图片（仅在未用全量候选库时作为候选库）
        pairs_metadata_file: 测试集配对元数据（含 anchor_idx, positive_idx 窗口索引，窗口索引为全量 2010-2021 中的下标）
        top_k_list: 要计算的 K 列表，默认 [1, 3]
        features_52d_file: 可选，52 维特征 npy（与全量窗口一一对应），用于 52 维相似度统计
        demo_save_dir: 可选，保存编码向量的目录（生成 demo 检索用候选库）
        all_images_file: 可选，全量窗口图片 npy（2010-2021 所有窗口），提供则候选库=全量
        inference_index_dir: 可选，推理索引目录（含 embeddings.npy），提供则直接用作全量候选库，避免重复编码

    Returns:
        含 recall_at_k、mean_similarity 及 52 维相关统计等
    """
    with open(pairs_metadata_file, "r", encoding="utf-8") as f:
        pairs_metadata = json.load(f)
    total = len(pairs_metadata)

    # 是否使用全量 2010-2021 作为候选库
    use_full_pool = bool(inference_index_dir or all_images_file)
    if use_full_pool:
        print("=" * 60)
        print("评估检索准确率（候选库 = 2010-2021 全量窗口）")
        print("=" * 60)
    else:
        print("=" * 60)
        print("评估检索准确率（候选库 = 测试集 positive）")
        print("=" * 60)

    # 加载测试 anchor
    print(f"\n[加载] 加载测试数据...")
    anchor_images = np.load(anchor_images_file)
    positive_images = np.load(positive_images_file)  # 未用全量时作候选库；用全量时仅用于 fallback 编码缓存校验
    print(f"      测试对数量: {total}")
    if use_full_pool:
        print(f"      候选库: 2010-2021 全量窗口（见下方加载）")
    else:
        print(f"      候选库大小: {len(positive_images)}（测试集所有 positive）")

    # ---------- 全量候选库：优先用 inference_index，否则 all_images 编码（可缓存） ----------
    full_candidate_embeddings = None  # [N_all, D]，行下标 = 窗口索引
    n_all_windows = None
    if use_full_pool:
        if inference_index_dir:
            idx_dir = Path(inference_index_dir)
            if not idx_dir.is_absolute():
                idx_dir = (project_root / inference_index_dir).resolve()
            emb_file = idx_dir / "embeddings.npy"
            if emb_file.exists():
                full_candidate_embeddings = np.load(str(emb_file))
                n_all_windows = full_candidate_embeddings.shape[0]
                print(f"      [全量] 从推理索引加载: {emb_file}  shape={full_candidate_embeddings.shape}")
        if full_candidate_embeddings is None and all_images_file:
            all_path = _resolve(all_images_file) if not Path(all_images_file).is_absolute() else Path(all_images_file)
            if all_path.exists():
                all_imgs = np.load(str(all_path))
                n_all_windows = len(all_imgs)
                # 尝试从 demo_save_dir 加载已编码的全量候选库
                if demo_save_dir:
                    demo_dir = Path(demo_save_dir)
                    full_path = demo_dir / "demo_embeddings_full_candidates.npy"
                    if full_path.exists():
                        loaded = np.load(str(full_path))
                        if len(loaded) == n_all_windows:
                            full_candidate_embeddings = loaded
                            print(f"      [全量] 使用已有全量向量库，跳过编码: {full_path}")
                if full_candidate_embeddings is None:
                    print(f"      [全量] 编码 2010-2021 全量窗口共 {n_all_windows} 张...")
                    full_candidate_embeddings = []
                    for i, image in enumerate(all_imgs):
                        if (i + 1) % 1000 == 0 or (i + 1) == n_all_windows:
                            print(f"        已编码 {i+1}/{n_all_windows} 张...")
                        full_candidate_embeddings.append(encoder.encode_image(image))
                    full_candidate_embeddings = np.array(full_candidate_embeddings, dtype=np.float32)
                    if demo_save_dir:
                        Path(demo_save_dir).mkdir(parents=True, exist_ok=True)
                        np.save(Path(demo_save_dir) / "demo_embeddings_full_candidates.npy", full_candidate_embeddings)
                        print(f"      [Demo] 全量候选库已保存: {Path(demo_save_dir) / 'demo_embeddings_full_candidates.npy'}")
            if full_candidate_embeddings is None:
                use_full_pool = False
                print(f"      [回退] 未找到全量数据，改用测试集 positive 作为候选库")

    # ---------- 测试 anchor 编码（或从缓存加载） ----------
    anchor_embeddings = None
    positive_embeddings = None  # 未用全量时 = 测试集 positive 的编码；用全量时 = full_candidate_embeddings 的引用
    if demo_save_dir:
        demo_dir = Path(demo_save_dir)
        anchor_path = demo_dir / "demo_embeddings_anchor.npy"
        positive_path = demo_dir / "demo_embeddings_positive.npy"
        if anchor_path.exists():
            anc = np.load(str(anchor_path))
            if len(anc) == total:
                if not use_full_pool and positive_path.exists():
                    pos = np.load(str(positive_path))
                    if len(pos) == total:
                        anchor_embeddings = anc
                        positive_embeddings = pos
                        print(f"\n[编码] 使用已有向量库，跳过编码: {demo_dir}")
                elif use_full_pool and full_candidate_embeddings is not None:
                    anchor_embeddings = anc
                    positive_embeddings = full_candidate_embeddings
                    print(f"\n[编码] 使用已有 anchor 向量，跳过编码: {demo_dir}")

    if anchor_embeddings is None:
        print(f"\n[编码] 编码测试 anchor 图片...")
        anchor_embeddings = []
        for i, image in enumerate(anchor_images):
            if (i + 1) % 500 == 0 or (i + 1) == len(anchor_images):
                print(f"      已编码 {i+1}/{len(anchor_images)} 张...")
            anchor_embeddings.append(encoder.encode_image(image))
        anchor_embeddings = np.array(anchor_embeddings)
        print(f"      编码完成，anchor 嵌入形状: {anchor_embeddings.shape}")

        if not use_full_pool:
            # 候选库 = 测试集 positive，一起编码并保存
            print(f"      编码测试 positive（候选库）...")
            positive_embeddings = []
            for i, image in enumerate(positive_images):
                if (i + 1) % 500 == 0 or (i + 1) == len(positive_images):
                    print(f"      已编码 {i+1}/{len(positive_images)} 张...")
                positive_embeddings.append(encoder.encode_image(image))
            positive_embeddings = np.array(positive_embeddings)
            if demo_save_dir:
                demo_dir = Path(demo_save_dir)
                demo_dir.mkdir(parents=True, exist_ok=True)
                np.save(demo_dir / "demo_embeddings_anchor.npy", anchor_embeddings.astype(np.float32))
                np.save(demo_dir / "demo_embeddings_positive.npy", positive_embeddings.astype(np.float32))
                with open(demo_dir / "demo_embeddings_metadata.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "embedding_dim": int(anchor_embeddings.shape[1]),
                        "n_anchors": len(anchor_embeddings),
                        "n_positives": len(positive_embeddings),
                        "pairs_metadata": pairs_metadata,
                    }, f, indent=2, ensure_ascii=False)
                print(f"      [Demo] 向量库已保存: {demo_dir} (anchor/positive .npy + metadata.json)")
        else:
            positive_embeddings = full_candidate_embeddings
    else:
        if use_full_pool:
            positive_embeddings = full_candidate_embeddings

    # 用编码器做检索：对每个 anchor 在候选库中按编码器相似度排序
    max_k = max(top_k_list)
    recall_at_k = {k: 0 for k in top_k_list}
    similarities_list = []
    top1_retrieved_pair_idx = []  # 每个 anchor 被编码器检索到的 top-1 对应的候选下标（全量时为窗口索引）

    N_cand = positive_embeddings.shape[0]
    cand_norm = positive_embeddings / (np.linalg.norm(positive_embeddings, axis=1, keepdims=True) + 1e-8)
    print(f"\n[评估] Recall@k（按编码器相似度排序），k = {top_k_list}，候选库大小 = {N_cand} ...")
    for i in range(total):
        anchor_emb = anchor_embeddings[i]
        anchor_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
        similarities = np.dot(cand_norm, anchor_norm)
        top_indices = np.argsort(similarities)[::-1][:max_k]
        top1_retrieved_pair_idx.append(top_indices[0])
        if use_full_pool:
            true_positive_win_idx = pairs_metadata[i]["positive_idx"]
            # 在全量中 true positive 的排名（0-based）
            rank = np.where(np.argsort(similarities)[::-1] == true_positive_win_idx)[0][0]
            true_similarity = similarities[true_positive_win_idx]
            for k in top_k_list:
                if rank < k:
                    recall_at_k[k] += 1
        else:
            true_positive_rank = np.where(top_indices == i)[0]
            true_similarity = similarities[i]
            for k in top_k_list:
                if len(true_positive_rank) > 0 and true_positive_rank[0] < k:
                    recall_at_k[k] += 1
        similarities_list.append(true_similarity)

    for k in top_k_list:
        recall_at_k[k] = recall_at_k[k] / total
    similarities_list = np.array(similarities_list)

    for k in top_k_list:
        print(f"      Recall@{k}: {recall_at_k[k]:.4f}")

    # ---------- 收集四组相似度（最后统一打印） ----------
    mean_orig_enc = float(np.mean(similarities_list))
    std_orig_enc = float(np.std(similarities_list))
    min_orig_enc = float(np.min(similarities_list))
    max_orig_enc = float(np.max(similarities_list))

    out = {
        "recall_at_k": recall_at_k,
        "total": total,
        "mean_similarity": mean_orig_enc,
        "std_similarity": std_orig_enc,
        "min_similarity": min_orig_enc,
        "max_similarity": max_orig_enc,
        "candidate_pool": "all_windows" if use_full_pool else "test_positives",
        "candidate_pool_size": int(positive_embeddings.shape[0]),
    }

    SIM_ENC_THRESHOLD = 0.85  # 编码器检索：只保留编码器相似度 > 此阈值的 positive，取前 3 个
    TOP_N_POSITIVES = 3
    # 原始测试集 标准2：52 维（优先用 metadata 的 similarity）
    mean_52_orig = std_52_orig = min_52_orig = max_52_orig = None
    sim_52_from_meta = [p.get("similarity") for p in pairs_metadata]
    if all(x is not None for x in sim_52_from_meta):
        sim_52_per_pair = np.array([float(x) for x in sim_52_from_meta])
        mean_52_orig = float(np.mean(sim_52_per_pair))
        std_52_orig = float(np.std(sim_52_per_pair))
        min_52_orig = float(np.min(sim_52_per_pair))
        max_52_orig = float(np.max(sim_52_per_pair))
        out["mean_similarity_52d_original_test_set"] = mean_52_orig
        out["std_similarity_52d_original_test_set"] = std_52_orig
        out["min_similarity_52d_original_test_set"] = min_52_orig
        out["max_similarity_52d_original_test_set"] = max_52_orig
        out["similarity_52d_source"] = "pairs_metadata"
    elif features_52d_file:
        path = _resolve(features_52d_file) if not Path(features_52d_file).is_absolute() else Path(features_52d_file)
        if path.exists():
            features = np.load(str(path))
            feat = features.astype(np.float64)
            feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
            sim_52_per_pair = np.array([
                float(np.dot(feat_norm[pairs_metadata[i]["anchor_idx"]], feat_norm[pairs_metadata[i]["positive_idx"]]))
                for i in range(total)
            ])
            mean_52_orig = float(np.mean(sim_52_per_pair))
            std_52_orig = float(np.std(sim_52_per_pair))
            min_52_orig = float(np.min(sim_52_per_pair))
            max_52_orig = float(np.max(sim_52_per_pair))
            out["mean_similarity_52d_original_test_set"] = mean_52_orig
            out["std_similarity_52d_original_test_set"] = std_52_orig
            out["min_similarity_52d_original_test_set"] = min_52_orig
            out["max_similarity_52d_original_test_set"] = max_52_orig
            out["similarity_52d_source"] = "features_52d.npy"

    # 编码器检索（不重复 anchor，编码器相似度>0.85 的前 3 个 positive）— 需 features_52d 才能算 52 维统计
    mean_enc_ret = std_enc_ret = min_enc_ret = max_enc_ret = None
    mean_52_ret = std_52_ret = min_52_ret = max_52_ret = None
    n_anchors = n_valid = 0
    feat_norm = None
    if features_52d_file:
        path = _resolve(features_52d_file) if not Path(features_52d_file).is_absolute() else Path(features_52d_file)
        if path.exists():
            features = np.load(str(path))
            feat = features.astype(np.float64)
            feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
            anchor_idx_to_first_pair: Dict[int, int] = {}
            anchor_idx_to_pair_indices: Dict[int, List[int]] = {}
            for i in range(total):
                aid = pairs_metadata[i]["anchor_idx"]
                if aid not in anchor_idx_to_first_pair:
                    anchor_idx_to_first_pair[aid] = i
                    anchor_idx_to_pair_indices[aid] = []
                anchor_idx_to_pair_indices[aid].append(i)
            unique_anchors = list(anchor_idx_to_first_pair.keys())
            n_anchors = len(unique_anchors)

            mean_52_original_per_anchor = []
            mean_encoder_original_per_anchor = []
            mean_52_retrieved_per_anchor = []
            mean_encoder_retrieved_per_anchor = []
            for aid in unique_anchors:
                first_i = anchor_idx_to_first_pair[aid]
                anchor_emb = anchor_embeddings[first_i]
                anchor_emb_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
                a52 = feat_norm[aid]
                pair_indices = anchor_idx_to_pair_indices[aid]
                orig_sims = [float(np.dot(a52, feat_norm[pairs_metadata[i]["positive_idx"]])) for i in pair_indices]
                mean_52_original_per_anchor.append(np.mean(orig_sims))
                mean_encoder_original_per_anchor.append(float(np.mean(similarities_list[pair_indices])))

                sims_enc = np.dot(positive_embeddings, anchor_emb_norm)
                order = np.argsort(sims_enc)[::-1]
                retrieved_sims_52 = []
                retrieved_sims_enc = []
                for j in order:
                    s_enc = float(sims_enc[j])
                    if s_enc <= SIM_ENC_THRESHOLD:
                        continue
                    retrieved_sims_enc.append(s_enc)
                    win_idx = int(j) if use_full_pool else pairs_metadata[j]["positive_idx"]
                    retrieved_sims_52.append(float(np.dot(a52, feat_norm[win_idx])))
                    if len(retrieved_sims_enc) >= TOP_N_POSITIVES:
                        break
                mean_52_retrieved_per_anchor.append(np.mean(retrieved_sims_52) if retrieved_sims_52 else float("nan"))
                mean_encoder_retrieved_per_anchor.append(np.mean(retrieved_sims_enc) if retrieved_sims_enc else float("nan"))

            mean_52_original_per_anchor = np.array(mean_52_original_per_anchor)
            mean_encoder_original_per_anchor = np.array(mean_encoder_original_per_anchor)
            mean_52_retrieved_per_anchor = np.array(mean_52_retrieved_per_anchor)
            mean_encoder_retrieved_per_anchor = np.array(mean_encoder_retrieved_per_anchor)
            valid = ~np.isnan(mean_52_retrieved_per_anchor)
            n_valid = int(np.sum(valid))
            if n_valid > 0:
                mean_enc_ret = float(np.nanmean(mean_encoder_retrieved_per_anchor))
                std_enc_ret = float(np.nanstd(mean_encoder_retrieved_per_anchor))
                min_enc_ret = float(np.nanmin(mean_encoder_retrieved_per_anchor))
                max_enc_ret = float(np.nanmax(mean_encoder_retrieved_per_anchor))
                mean_52_ret = float(np.nanmean(mean_52_retrieved_per_anchor))
                std_52_ret = float(np.nanstd(mean_52_retrieved_per_anchor))
                min_52_ret = float(np.nanmin(mean_52_retrieved_per_anchor))
                max_52_ret = float(np.nanmax(mean_52_retrieved_per_anchor))

            out["n_unique_anchors"] = n_anchors
            out["mean_similarity_52d_anchor_vs_original"] = float(np.mean(mean_52_original_per_anchor))
            out["std_similarity_52d_anchor_vs_original"] = float(np.std(mean_52_original_per_anchor))
            out["mean_similarity_52d_anchor_vs_retrieved"] = mean_52_ret if n_valid > 0 else None
            out["std_similarity_52d_anchor_vs_retrieved"] = std_52_ret if n_valid > 0 else None
            out["mean_similarity_encoder_anchor_vs_retrieved"] = mean_enc_ret if n_valid > 0 else None
            out["std_similarity_encoder_anchor_vs_retrieved"] = std_enc_ret if n_valid > 0 else None
            out["min_similarity_encoder_anchor_vs_retrieved"] = min_enc_ret if n_valid > 0 else None
            out["max_similarity_encoder_anchor_vs_retrieved"] = max_enc_ret if n_valid > 0 else None
            out["min_similarity_52d_anchor_vs_retrieved"] = min_52_ret if n_valid > 0 else None
            out["max_similarity_52d_anchor_vs_retrieved"] = max_52_ret if n_valid > 0 else None
            out["n_anchors_with_retrieved_above_threshold"] = n_valid
            out["encoder_similarity_threshold"] = SIM_ENC_THRESHOLD
            out["top_n_positives"] = TOP_N_POSITIVES
            out["_per_anchor_52d"] = {
                "anchor_idx": unique_anchors,
                "mean_52_anchor_vs_original": [round(float(x), 6) for x in mean_52_original_per_anchor],
                "mean_encoder_anchor_vs_original": [round(float(x), 6) for x in mean_encoder_original_per_anchor],
                "mean_52_anchor_vs_retrieved": [round(float(x), 6) if not np.isnan(x) else None for x in mean_52_retrieved_per_anchor],
                "mean_encoder_anchor_vs_retrieved": [round(float(x), 6) if not np.isnan(x) else None for x in mean_encoder_retrieved_per_anchor],
            }

    # 原始测试集标准1 的分布（高/中/低），与 evaluate_pair_similarity 一致
    high_enc = int(np.sum(similarities_list >= 0.9))
    mid_enc = int(np.sum((similarities_list >= 0.7) & (similarities_list < 0.9)))
    low_enc = int(np.sum(similarities_list < 0.7))
    out["similarity_distribution_encoder"] = {"high": high_enc, "medium": mid_enc, "low": low_enc}

    # ---------- 统一打印：四组相似度数值分析 ----------
    print(f"\n" + "=" * 56)
    print("四组相似度数值分析")
    print("=" * 56)
    print(f"  一、原始测试集（{total} 对）")
    print(f"      标准1-编码器相似度: 均值 {mean_orig_enc:.4f}  标准差 {std_orig_enc:.4f}  范围 [{min_orig_enc:.4f}, {max_orig_enc:.4f}]  高(>=0.9):{high_enc} 中(0.7-0.9):{mid_enc} 低(<0.7):{low_enc}")
    if mean_52_orig is not None:
        print(f"      标准2-52维相似度:   均值 {mean_52_orig:.4f}  标准差 {std_52_orig:.4f}  范围 [{min_52_orig:.4f}, {max_52_orig:.4f}]")
    else:
        print(f"      标准2-52维相似度:   （未提供 52 维或 metadata 无 similarity，跳过）")
    print(f"  二、编码器检索（不重复 anchor，编码器相似度>{SIM_ENC_THRESHOLD} 的前 {TOP_N_POSITIVES} 个 positive）")
    if mean_enc_ret is not None and mean_52_ret is not None:
        print(f"      不重复 anchor 数: {n_anchors}，其中 {n_valid} 个存在编码器相似度>{SIM_ENC_THRESHOLD} 的检索结果")
        print(f"      标准1-编码器相似度: 均值 {mean_enc_ret:.4f}  标准差 {std_enc_ret:.4f}  范围 [{min_enc_ret:.4f}, {max_enc_ret:.4f}]")
        print(f"      标准2-52维相似度:   均值 {mean_52_ret:.4f}  标准差 {std_52_ret:.4f}  范围 [{min_52_ret:.4f}, {max_52_ret:.4f}]")
    else:
        print(f"      （需提供 52 维特征文件且路径存在；若本地有该文件可传 --features-52d <路径>。上方若有 [提示] 会给出尝试路径）")
    print("=" * 56)

    return out


def evaluate_pair_similarity(
    encoder: TrainedEncoder,
    anchor_images_file: str,
    positive_images_file: str,
    pairs_metadata_file: str,
    features_52d_file: str = None,
) -> Dict:
    """
    评估相似对的相似度分布（编码器 + 可选 52 维）。

    Args:
        encoder: 训练好的编码器
        anchor_images_file: Anchor图片文件
        positive_images_file: Positive图片文件
        pairs_metadata_file: 配对元数据文件（含 anchor_idx, positive_idx）
        features_52d_file: 可选，制作相似对时的 52 维特征 npy，用于 52 维余弦相似度

    Returns:
        评估结果字典（含编码器相似度，及可选的 similarity_52d 统计）
    """
    print("\n" + "=" * 60)
    print("评估相似对相似度")
    print("=" * 60)

    # 加载数据
    anchor_images = np.load(anchor_images_file)
    positive_images = np.load(positive_images_file)

    with open(pairs_metadata_file, "r", encoding="utf-8") as f:
        pairs_metadata = json.load(f)

    # 编码并计算相似度（编码器）
    print(f"\n[计算] 相似对相似度（编码器）...")
    similarities = []
    for i, pair in enumerate(pairs_metadata):
        anchor_emb = encoder.encode_image(anchor_images[i])
        positive_emb = encoder.encode_image(positive_images[i])
        anchor_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
        positive_norm = positive_emb / (np.linalg.norm(positive_emb) + 1e-8)
        similarities.append(np.dot(anchor_norm, positive_norm))
        if (i + 1) % 500 == 0:
            print(f"      已计算 {i+1}/{len(pairs_metadata)} 对...")
    similarities = np.array(similarities)

    print(f"\n[统计] 相似度统计（编码器）:")
    print(f"      均值: {np.mean(similarities):.4f}  标准差: {np.std(similarities):.4f}")
    print(f"      最小值: {np.min(similarities):.4f}  最大值: {np.max(similarities):.4f}  中位数: {np.median(similarities):.4f}")
    high_sim = np.sum(similarities >= 0.9)
    medium_sim = np.sum((similarities >= 0.7) & (similarities < 0.9))
    low_sim = np.sum(similarities < 0.7)
    print(f"      高(>=0.9): {high_sim}  中(0.7-0.9): {medium_sim}  低(<0.7): {low_sim}")

    out = {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "median": float(np.median(similarities)),
        "distribution": {"high": int(high_sim), "medium": int(medium_sim), "low": int(low_sim)},
    }

    # 可选：用 52 维特征（制作相似对时同一套）计算配对相似度
    if features_52d_file:
        path = _resolve(features_52d_file) if not Path(features_52d_file).is_absolute() else Path(features_52d_file)
        if path.exists():
            print(f"\n[52维] 使用制作相似对时的 52 维特征计算配对相似度...")
            features = np.load(str(path))
            anchor_feats_52, positive_feats_52 = _cosine_similarity_52d(features, pairs_metadata)
            anorm = anchor_feats_52 / (np.linalg.norm(anchor_feats_52, axis=1, keepdims=True) + 1e-8)
            pnorm = positive_feats_52 / (np.linalg.norm(positive_feats_52, axis=1, keepdims=True) + 1e-8)
            sim_52 = np.sum(anorm * pnorm, axis=1)
            print(f"      均值: {np.mean(sim_52):.4f}  标准差: {np.std(sim_52):.4f}  范围: [{np.min(sim_52):.4f}, {np.max(sim_52):.4f}]")
            out["mean_52d"] = float(np.mean(sim_52))
            out["std_52d"] = float(np.std(sim_52))
            out["min_52d"] = float(np.min(sim_52))
            out["max_52d"] = float(np.max(sim_52))
        else:
            print(f"      [跳过] 52 维特征文件不存在: {path}")

    return out


def main():
    """主函数：不传参则使用 DEFAULT_CONFIG 路径，可直接运行脚本。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="用最优模型在测试集上做相似检索评估（默认路径见脚本内 DEFAULT_CONFIG）"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint_best.pth 路径")
    parser.add_argument("--anchor-images", type=str, default=None, help="测试集 anchor 图像 npy")
    parser.add_argument("--positive-images", type=str, default=None, help="测试集 positive 图像 npy")
    parser.add_argument("--pairs-metadata", type=str, default=None, help="测试集配对 json")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=None,
        help="Top-K 列表，如 --top-k 1 3；默认 1 3（与 max_pairs_per_anchor=3 一致）",
    )
    parser.add_argument(
        "--features-52d",
        type=str,
        default=None,
        help="52 维特征 npy 路径（与 main 步骤1 的 features_file 一致）；不传则用 DEFAULT_CONFIG；传 none 则只做编码器评估、不加载 52 维",
    )
    parser.add_argument("--all-images", type=str, default=None, help="全量窗口图 npy，提供则候选库=2010-2021全量")
    parser.add_argument("--inference-index-dir", type=str, default=None, help="推理索引目录（含 embeddings.npy），提供则直接作全量候选库")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint
    if args.anchor_images is not None:
        cfg["test_anchor_images"] = args.anchor_images
    if args.positive_images is not None:
        cfg["test_positive_images"] = args.positive_images
    if args.pairs_metadata is not None:
        cfg["test_pairs_metadata"] = args.pairs_metadata
    if args.top_k is not None:
        cfg["top_k_list"] = args.top_k
    if args.features_52d is not None:
        cfg["features_52d"] = None if args.features_52d.lower() == "none" else args.features_52d
    if args.all_images is not None:
        cfg["all_images_file"] = None if args.all_images.lower() == "none" else args.all_images
    if args.inference_index_dir is not None:
        cfg["inference_index_dir"] = None if args.inference_index_dir.lower() == "none" else args.inference_index_dir

    checkpoint_path = _resolve(cfg["checkpoint"])
    anchor_path = _resolve(cfg["test_anchor_images"])
    positive_path = _resolve(cfg["test_positive_images"])
    pairs_path = _resolve(cfg["test_pairs_metadata"])
    top_k_list = cfg["top_k_list"]

    if not checkpoint_path.exists():
        print(f"[错误] 未找到模型: {checkpoint_path}")
        return
    if not anchor_path.exists() or not positive_path.exists() or not pairs_path.exists():
        print(f"[错误] 未找到测试集文件，请先运行步骤 3.5 划分数据集")
        print(f"      anchor: {anchor_path}")
        print(f"      positive: {positive_path}")
        print(f"      metadata: {pairs_path}")
        return

    # 加载模型
    print("=" * 60)
    print("模型验证（最优模型 + 测试集）")
    print("=" * 60)
    print(f"\n[加载] 模型: {checkpoint_path}")

    encoder = TrainedEncoder(str(checkpoint_path))

    print(f"      设备: {encoder.device}")

    features_52d_path = cfg.get("features_52d")
    if features_52d_path:
        fp = _resolve(features_52d_path) if not Path(features_52d_path).is_absolute() else Path(features_52d_path)
        if not fp.exists() and features_52d_path == FEATURES_52D_RELATIVE:
            # 回退：相对 services/training 的路径（便于从不同工作目录运行）
            fp_fallback = training_dir / "output/dow30_2010_2021/features_52d.npy"
            if fp_fallback.exists():
                fp = fp_fallback
        if not fp.exists():
            print(f"[提示] 52 维特征文件不存在，编码器检索两套标准将跳过。尝试路径: {fp}")
            features_52d_path = None
        else:
            features_52d_path = str(fp)

    # 检索指标（Recall@1/3/5 等）；可选 52 维统计；并保存编码向量作为 demo 检索候选库
    all_images_path = cfg.get("all_images_file")
    inference_index = cfg.get("inference_index_dir")
    # 未显式指定时，按 checkpoint 所在目录名推断 inference_index（与 main 步骤5 的 inference_index_{training_method} 一致）
    if not inference_index and checkpoint_path.parent.name in ("vicreg", "simsiam", "barlow", "clip_contrastive"):
        encoder_name = "clip" if checkpoint_path.parent.name == "clip_contrastive" else checkpoint_path.parent.name
        default_inference = f"services/training/output/dow30_2010_2021/inference_index_{encoder_name}"
        if _resolve(default_inference).exists():
            inference_index = default_inference
            print(f"      [推理索引] 使用与 encoder 对应目录: {inference_index}")
    if all_images_path:
        p = _resolve(all_images_path) if not Path(all_images_path).is_absolute() else Path(all_images_path)
        if not p.exists():
            print(f"[提示] 全量窗口图不存在，将使用测试集 positive 作为候选库: {p}")
            all_images_path = None
        else:
            all_images_path = str(p)
    if inference_index:
        p = _resolve(inference_index) if not Path(inference_index).is_absolute() else Path(inference_index)
        if not p.exists():
            print(f"[提示] 推理索引目录不存在: {p}")
            inference_index = None
        else:
            inference_index = str(p)
    retrieval_results = evaluate_retrieval_accuracy(
        encoder,
        str(anchor_path),
        str(positive_path),
        str(pairs_path),
        top_k_list=top_k_list,
        features_52d_file=features_52d_path,
        demo_save_dir=str(checkpoint_path.parent),
        all_images_file=all_images_path,
        inference_index_dir=inference_index,
    )

    # 相似度分布：与「四组」中 一、原始测试集 一致，不再重复调用 evaluate_pair_similarity 编码
    similarity_results = {
        "mean": retrieval_results.get("mean_similarity"),
        "std": retrieval_results.get("std_similarity"),
        "min": retrieval_results.get("min_similarity"),
        "max": retrieval_results.get("max_similarity"),
        "distribution": retrieval_results.get("similarity_distribution_encoder", {"high": 0, "medium": 0, "low": 0}),
    }
    if retrieval_results.get("mean_similarity_52d_original_test_set") is not None:
        similarity_results["mean_52d"] = retrieval_results["mean_similarity_52d_original_test_set"]
        similarity_results["std_52d"] = retrieval_results.get("std_similarity_52d_original_test_set")
        similarity_results["min_52d"] = retrieval_results.get("min_similarity_52d_original_test_set")
        similarity_results["max_52d"] = retrieval_results.get("max_similarity_52d_original_test_set")

    results = {
        "retrieval": retrieval_results,
        "similarity": similarity_results,
    }
    results_file = checkpoint_path.parent / "validation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 验证结果: {results_file}")

    # 若有 52 维逐条数据（按不重复 anchor），另存 CSV。四列相似度含义：
    #   mean_encoder_anchor_vs_original: 该 anchor 与「原始测试集中其配对 positive」的编码器相似度均值（一、标准1 按 anchor）
    #   mean_52_anchor_vs_original:      该 anchor 与「原始测试集中其配对 positive」的 52 维相似度均值（一、标准2 按 anchor）
    #   mean_encoder_anchor_vs_retrieved: 该 anchor 与「编码器检索到的 >0.85 前3个」的编码器相似度均值（二、标准1）
    #   mean_52_anchor_vs_retrieved:     该 anchor 与「编码器检索到的 >0.85 前3个」的 52 维相似度均值（二、标准2）
    if "_per_anchor_52d" in retrieval_results:
        import csv
        csv_file = checkpoint_path.parent / "validation_52d_comparison.csv"
        per = retrieval_results["_per_anchor_52d"]
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "anchor_idx",
                "mean_encoder_anchor_vs_original",
                "mean_52_anchor_vs_original",
                "mean_encoder_anchor_vs_retrieved",
                "mean_52_anchor_vs_retrieved",
            ])
            for i in range(len(per["anchor_idx"])):
                enc_orig = per.get("mean_encoder_anchor_vs_original")
                r52 = per["mean_52_anchor_vs_retrieved"][i] if i < len(per["mean_52_anchor_vs_retrieved"]) else None
                r_enc_list = per.get("mean_encoder_anchor_vs_retrieved")
                r_enc = r_enc_list[i] if r_enc_list and i < len(r_enc_list) else None
                w.writerow([
                    per["anchor_idx"][i],
                    per["mean_encoder_anchor_vs_original"][i] if enc_orig and i < len(enc_orig) else "",
                    per["mean_52_anchor_vs_original"][i],
                    r_enc if r_enc is not None else "",
                    r52 if r52 is not None else "",
                ])
        print(f"      按 anchor 的 52 维/编码器对比: {csv_file}")
        del results["retrieval"]["_per_anchor_52d"]
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

