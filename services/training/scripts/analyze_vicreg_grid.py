"""
分析已运行过的 VICReg 网格搜索结果：扫描 run_* 目录下的 lambda*，读取 checkpoint 并（可选）在测试集上评估 Recall@k，输出结果表并保存 JSON。
若该 run 下已有 grid_results.json，可用 --from_json 直接读表不再跑评估。

用法（在项目根 kline 下，或 cd 到 services/training）：
  python scripts/analyze_vicreg_grid.py
  python scripts/analyze_vicreg_grid.py --grid_dir logs/vicreg_grid --run latest
  python scripts/analyze_vicreg_grid.py --run run_20250207_143022 --no_eval
  python scripts/analyze_vicreg_grid.py --run run_20250207_143022 --from_json
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

_script_dir = Path(__file__).resolve().parent
_training_dir = _script_dir.parent
_project_root = _training_dir.parent.parent
sys.path.insert(0, str(_training_dir))
sys.path.insert(0, str(_project_root))


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (base / path).resolve()
    return p


def _load_ckpt_info(ckpt_path: Path):
    """读取 checkpoint 的 epoch、best_loss、metrics。失败返回 None。"""
    try:
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        return {
            "epoch": ckpt.get("epoch"),
            "best_loss": ckpt.get("best_loss"),
            "metrics": ckpt.get("metrics"),
        }
    except Exception:
        return None


def _eval_recall(ckpt_path: str, test_anchor: str, test_positive: str, test_meta: str, device: str):
    """在测试集上算 Recall@1、Recall@3。失败返回 None。"""
    try:
        from inference_encoder import TrainedEncoder
        from evaluate.validate_model import evaluate_retrieval_accuracy
        encoder = TrainedEncoder(ckpt_path, device=device)
        res = evaluate_retrieval_accuracy(
            encoder, test_anchor, test_positive, test_meta,
            top_k_list=[1, 3],
            features_52d_file=None,
            demo_save_dir=None,
            all_images_file=None,
            inference_index_dir=None,
        )
        return {
            "recall_at_1": res.get("recall_at_k", {}).get(1),
            "recall_at_3": res.get("recall_at_k", {}).get(3),
            "mean_similarity": res.get("mean_similarity"),
        }
    except Exception:
        return None


def _parse_lambda_from_dir(name: str):
    """从目录名 lambda5.0 解析出 5.0。"""
    if name.startswith("lambda"):
        try:
            return float(name[6:])
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="分析 VICReg 网格搜索结果")
    parser.add_argument(
        "--grid_dir",
        type=str,
        default="logs/vicreg_grid",
        help="网格搜索根目录（相对 services/training 或绝对路径）",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="latest",
        help="run 子目录名，如 run_20250207_143022；填 latest 表示选最新一次运行",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="不跑测试集评估，仅从 checkpoint 读取 loss 等信息",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="recall_at_3",
        choices=["recall_at_1", "recall_at_3", "best_loss"],
        help="选优指标：recall_at_1 / recall_at_3（越大越好）或 best_loss（越小越好）",
    )
    parser.add_argument(
        "--test_anchor",
        type=str,
        default="output/dow30_2010_2021/dataset_splits/test_anchor_images.npy",
        help="测试集 anchor（仅 --no_eval 未指定时用）",
    )
    parser.add_argument(
        "--test_positive",
        type=str,
        default="output/dow30_2010_2021/dataset_splits/test_positive_images.npy",
        help="测试集 positive",
    )
    parser.add_argument(
        "--test_meta",
        type=str,
        default="output/dow30_2010_2021/dataset_splits/test_pairs_metadata.json",
        help="测试集 pairs metadata",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--from_json",
        action="store_true",
        help="若该 run 下已有 grid_results.json，直接读取并打印表，不再加载 checkpoint 或跑评估",
    )
    args = parser.parse_args()

    grid_root = _resolve(args.grid_dir, _training_dir)
    if not grid_root.exists():
        print(f"[错误] 网格目录不存在: {grid_root}")
        return

    run_dirs = sorted([d for d in grid_root.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print(f"[错误] 未找到 run_* 目录: {grid_root}")
        return

    if args.run == "latest":
        run_dir = run_dirs[-1]
    else:
        run_dir = grid_root / args.run
        if not run_dir.exists():
            print(f"[错误] 指定 run 不存在: {run_dir}")
            return
    print(f"分析 run: {run_dir}")

    # 若指定 --from_json 且存在 grid_results.json，直接读并打印
    json_file = run_dir / "grid_results.json"
    if args.from_json and json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        best_lambda = data.get("best_lambda")
        criterion = data.get("criterion", "recall_at_3")
        print("\n" + "=" * 70)
        print("VICReg 网格搜索结果（来自 grid_results.json）")
        print("=" * 70)
        print(f"  Run: {run_dir.name}  选优指标: {criterion}")
        print()
        header = ["λ=μ", "epoch", "best_loss", "recall@1", "recall@3"]
        col_widths = [8, 8, 14, 10, 10]
        fmt = "  ".join(f"{{:>{w}}}" for w in col_widths)
        print(fmt.format(*header))
        print("-" * 70)
        for r in results:
            lam_s = str(r.get("lambda", "?"))
            epoch = r.get("epoch")
            epoch_s = str((epoch + 1) if epoch is not None else "-")
            loss_s = f"{r['best_loss']:.4f}" if r.get("best_loss") is not None else "-"
            r1_s = f"{r['recall_at_1']:.4f}" if r.get("recall_at_1") is not None else "-"
            r3_s = f"{r['recall_at_3']:.4f}" if r.get("recall_at_3") is not None else "-"
            mark = "  ← 最佳" if best_lambda is not None and r.get("lambda") == best_lambda else ""
            print(fmt.format(lam_s, epoch_s, loss_s, r1_s, r3_s) + mark)
        print("=" * 70)
        if best_lambda is not None:
            best_dir = next((r.get("dir") for r in results if r.get("lambda") == best_lambda), "")
            print(f"  推荐: lambda_inv = lambda_var = {best_lambda}, lambda_cov = 1")
            print(f"  最佳模型目录: {best_dir}")
        print(f"\n结果文件: {json_file}")
        return

    # 收集所有 lambda* 目录（含 checkpoint_best.pth）
    lambda_dirs = []
    for d in run_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("lambda"):
            continue
        lam = _parse_lambda_from_dir(d.name)
        ckpt = d / "checkpoint_best.pth"
        if ckpt.exists():
            lambda_dirs.append((lam, d, ckpt))
    lambda_dirs.sort(key=lambda x: (x[0] is None, x[0] or 0))

    if not lambda_dirs:
        print(f"[错误] 未找到任何 lambda*/checkpoint_best.pth 于 {run_dir}")
        return

    test_anchor_path = test_positive_path = test_meta_path = None
    if not args.no_eval:
        test_anchor_path = str(_resolve(args.test_anchor, _training_dir))
        test_positive_path = str(_resolve(args.test_positive, _training_dir))
        test_meta_path = str(_resolve(args.test_meta, _training_dir))
        for p in (test_anchor_path, test_positive_path, test_meta_path):
            if not Path(p).exists():
                print(f"[警告] 测试集文件不存在: {p}，将跳过测试集评估（仅报 checkpoint 信息）")
                test_anchor_path = None
                break

    results = []
    for lam, sub_dir, ckpt_path in lambda_dirs:
        row = {"lambda": lam, "dir": str(sub_dir)}
        info = _load_ckpt_info(ckpt_path)
        if info:
            row["epoch"] = info["epoch"]
            row["best_loss"] = info["best_loss"]
            row["metrics"] = info["metrics"]
        else:
            row["error"] = "加载 checkpoint 失败"
            results.append(row)
            continue

        if not args.no_eval and test_anchor_path:
            ev = _eval_recall(str(ckpt_path), test_anchor_path, test_positive_path, test_meta_path, args.device)
            if ev:
                row["recall_at_1"] = ev.get("recall_at_1")
                row["recall_at_3"] = ev.get("recall_at_3")
                row["mean_similarity"] = ev.get("mean_similarity")
            else:
                row["eval_error"] = "测试集评估失败"
        results.append(row)

    # 选优
    valid = [r for r in results if "error" not in r and "eval_error" not in r]
    if args.criterion == "best_loss":
        best_row = min(valid, key=lambda r: r.get("best_loss") or float("inf")) if valid else None
    else:
        key = "recall_at_3" if args.criterion == "recall_at_3" else "recall_at_1"
        if valid and any(r.get(key) is not None for r in valid):
            best_row = max(valid, key=lambda r: r.get(key) or -1.0)
        else:
            best_row = min(valid, key=lambda r: r.get("best_loss") or float("inf")) if valid else None

    # 打印表
    print("\n" + "=" * 70)
    print("VICReg 网格搜索结果")
    print("=" * 70)
    print(f"  Run: {run_dir.name}")
    print(f"  选优指标: {args.criterion}" + ("（越大越好）" if args.criterion.startswith("recall") else "（越小越好）"))
    print()
    header = ["λ=μ", "epoch", "best_loss", "recall@1", "recall@3"]
    col_widths = [8, 8, 14, 10, 10]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("-" * 70)
    for r in results:
        lam_s = str(r["lambda"]) if r.get("lambda") is not None else "?"
        epoch_s = str((r.get("epoch") or -1) + 1) if r.get("epoch") is not None else "-"
        loss_s = f"{r['best_loss']:.4f}" if r.get("best_loss") is not None else "-"
        r1_s = f"{r['recall_at_1']:.4f}" if r.get("recall_at_1") is not None else "-"
        r3_s = f"{r['recall_at_3']:.4f}" if r.get("recall_at_3") is not None else "-"
        mark = "  ← 最佳" if best_row and r.get("lambda") == best_row.get("lambda") else ""
        print(fmt.format(lam_s, epoch_s, loss_s, r1_s, r3_s) + mark)
    print("=" * 70)
    if best_row:
        print(f"  推荐: lambda_inv = lambda_var = {best_row['lambda']}, lambda_cov = 1")
        print(f"  最佳模型目录: {best_row['dir']}")
    print()

    # 保存 JSON（便于后续直接查看，无需再跑 eval）
    out_json = run_dir / "grid_results.json"
    # 可序列化
    def _ser(obj):
        if isinstance(obj, (int, float, str, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_ser(x) for x in obj]
        return str(obj)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "run": run_dir.name,
            "criterion": args.criterion,
            "best_lambda": best_row["lambda"] if best_row else None,
            "results": [_ser(r) for r in results],
        }, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {out_json}")


if __name__ == "__main__":
    main()
