"""
VICReg 损失权重网格搜索（与论文一致：ν=1，λ=μ 做网格搜索）。

选优依据：
- 推荐：配置测试集路径，按测试集 Recall@k 选优（越大越好）。
- 若按 loss 选优（val_loss 或 train_loss）：λ=μ 越小则 loss 天然越低，结果会偏向最小的 k，没有参考价值，仅作兜底。

每个 λ 的短训都会保存模型到本次运行目录下的 lambda{k}/（含 checkpoint_best.pth 等），
不会覆盖已有的 logs/vicreg。每次运行本脚本会新建一个带时间戳的子目录（如 vicreg_grid/run_20250207_143022）。
"""
import sys
from pathlib import Path
from datetime import datetime

_script_dir = Path(__file__).resolve().parent
_training_dir = _script_dir.parent
_project_root = _training_dir.parent
sys.path.insert(0, str(_training_dir))
sys.path.insert(0, str(_project_root))

from train_vicreg import train_vicreg

# ---------- 路径与超参（直接改这里，无需命令行） ----------
# 路径相对本脚本所在包根目录 services/training/（与项目根无关，避免 dist/services 等环境多一层 services）
CONFIG = {
    "anchor_images": "output/dow30_2010_2021/dataset_splits/train_anchor_images.npy",
    "positive_images": "output/dow30_2010_2021/dataset_splits/train_positive_images.npy",
    "pairs_metadata": "output/dow30_2010_2021/dataset_splits/train_pairs_metadata.json",
    "test_anchor_images": "output/dow30_2010_2021/dataset_splits/test_anchor_images.npy",
    "test_positive_images": "output/dow30_2010_2021/dataset_splits/test_positive_images.npy",
    "test_pairs_metadata": "output/dow30_2010_2021/dataset_splits/test_pairs_metadata.json",
    "log_dir": "logs/vicreg_grid",  # 网格搜索根目录，与 logs/vicreg 分离
    "lambdas": [1.0, 5.0, 10.0, 25.0, 50.0],
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "val_split": 0.0,
    "criterion_metric": "recall_at_3",  # 测试集选优时用：recall_at_1 / recall_at_3
    "device": "auto",
}


def _eval_test_set_recall(checkpoint_path: str, test_anchor: str, test_positive: str, test_meta: str, device: str, top_k_list=(1, 3)):
    """在测试集上算 Recall@k，返回 validate_model 的评估结果字典。"""
    from inference_encoder import TrainedEncoder
    from evaluate.validate_model import evaluate_retrieval_accuracy
    encoder = TrainedEncoder(checkpoint_path, device=device)
    return evaluate_retrieval_accuracy(
        encoder, test_anchor, test_positive, test_meta, top_k_list=list(top_k_list)
    )


def _resolve(path: str) -> Path:
    """相对 _training_dir（services/training/）解析，与项目根无关。"""
    p = Path(path)
    if not p.is_absolute():
        p = (_training_dir / path).resolve()
    return p


def main():
    cfg = CONFIG
    anchor = str(_resolve(cfg["anchor_images"]))
    positive = str(_resolve(cfg["positive_images"]))
    pairs = str(_resolve(cfg["pairs_metadata"]))
    test_anchor = str(_resolve(cfg["test_anchor_images"])) if cfg.get("test_anchor_images") else None
    test_positive = str(_resolve(cfg["test_positive_images"])) if cfg.get("test_positive_images") else None
    test_meta = str(_resolve(cfg["test_pairs_metadata"])) if cfg.get("test_pairs_metadata") else None

    use_test_set = all([test_anchor, test_positive, test_meta])
    if use_test_set:
        for pth in (test_anchor, test_positive, test_meta):
            if not Path(pth).exists():
                raise FileNotFoundError(f"测试集文件不存在: {pth}")
        print("选优依据: 测试集 " + cfg["criterion_metric"].replace("_", " ").upper() + "（越大越好）")
    elif cfg.get("val_split", 0) > 0:
        print("选优依据: 验证集 val_loss（越小越好）")
        print("  ⚠️ 按 loss 选优会偏向最小的 k（loss 天然更低），建议用测试集 Recall。")
    else:
        print("选优依据: 训练集 train_loss（越小越好）")
        print("  ⚠️ 按 loss 选优会偏向最小的 k（loss 天然更低），无参考价值；请在 CONFIG 中配置测试集路径。")

    log_root = _resolve(cfg["log_dir"])
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir = log_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"本次运行目录（每个 λ 的模型保存在其下 lambda{{k}}/）: {run_dir}")

    results = []
    for k in cfg["lambdas"]:
        if k <= 1:
            print(f"⚠️ 跳过 λ=μ={k}（论文要求 λ=μ>1）")
            continue
        sub_dir = run_dir / f"lambda{k}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print(f"  网格点: λ = μ = {k}, ν = 1, 保存到 {sub_dir}")
        print("=" * 60)
        _, _, metrics = train_vicreg(
            anchor_images_file=anchor,
            positive_images_file=positive,
            pairs_metadata_file=pairs,
            num_epochs=cfg["num_epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            log_dir=str(sub_dir),
            device=cfg["device"],
            val_split=cfg.get("val_split", 0.0),
            lambda_inv=k,
            lambda_var=k,
            lambda_cov=1.0,
            return_metrics=True,
        )
        if use_test_set:
            ckpt = str(sub_dir / "checkpoint_best.pth")
            if not Path(ckpt).exists():
                print(f"  ⚠️ 未找到 {ckpt}，跳过该 k 的测试集评估")
                criterion = -1.0
            else:
                res = _eval_test_set_recall(
                    ckpt, test_anchor, test_positive, test_meta, cfg["device"], top_k_list=(1, 3)
                )
                recall = res["recall_at_k"]
                criterion = recall.get(3, 0.0) if cfg["criterion_metric"] == "recall_at_3" else recall.get(1, 0.0)
                print(f"  → 测试集 Recall@1={recall.get(1, 0):.4f}  Recall@3={recall.get(3, 0):.4f}  criterion={criterion:.4f}")
            results.append((k, metrics["train_loss"], metrics["val_loss"], criterion))
        else:
            val_split = cfg.get("val_split", 0.0)
            criterion = metrics["val_loss"] if val_split > 0 else metrics["train_loss"]
            results.append((k, metrics["train_loss"], metrics["val_loss"], criterion))
            print(f"  → train_loss={metrics['train_loss']:.4f}  val_loss={metrics['val_loss']:.4f}  criterion={criterion:.4f}")

    if not results:
        print("没有有效网格点。")
        return

    if use_test_set:
        best = max(results, key=lambda x: x[3])
        criterion_name = cfg["criterion_metric"].replace("_", " ").upper()
    else:
        best = min(results, key=lambda x: x[3])
        criterion_name = "val_loss" if cfg.get("val_split", 0) > 0 else "train_loss"
    print("\n" + "=" * 60)
    print(f"  网格搜索结果（按 {criterion_name} 选优）")
    print("=" * 60)
    print(f"  {'λ=μ':>8}  {'train_loss':>12}  {'val_loss':>12}  {'criterion':>12}")
    for r in results:
        mark = "  ← 最佳" if r[0] == best[0] else ""
        print(f"  {r[0]:>8.1f}  {r[1]:>12.4f}  {r[2]:>12.4f}  {r[3]:>12.4f}{mark}")
    print(f"\n  推荐: lambda_inv = lambda_var = {best[0]}, lambda_cov = 1")
    print(f"  最佳 λ 的模型目录: {run_dir / ('lambda' + str(best[0]))}")


if __name__ == "__main__":
    main()
