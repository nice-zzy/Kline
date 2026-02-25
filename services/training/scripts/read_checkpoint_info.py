"""
读取各方法目录下的 checkpoint_best.pth 与 checkpoint_latest.pth，
打印训练轮次 (epoch)、loss、metrics 等信息。

默认目录含: barlow, barlow_100, simsiam, simsiam_100, vicreg_25, vicreg。
用法（在项目根 kline 下）:
  python services/training/scripts/read_checkpoint_info.py
  python services/training/scripts/read_checkpoint_info.py --update_conclusion services/training/logs/conclusion.txt
"""
from pathlib import Path
import argparse
import sys
import re

# 项目根 = kline
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parent.parent

DEFAULT_LOG_DIRS = [
    "services/training/logs/barlow",
    "services/training/logs/barlow_100",
    "services/training/logs/simsiam",
    "services/training/logs/simsiam_100",
    "services/training/logs/vicreg_25",
    "services/training/logs/vicreg",
]


def _ensure_torch():
    """未安装 PyTorch 时提示并退出。"""
    try:
        import torch
    except ImportError:
        print("错误: 未找到 PyTorch。请在已安装 PyTorch 的环境中运行（如训练用的 conda 环境）。")
        print("  例: conda activate <环境> 后执行 python services/training/scripts/read_checkpoint_info.py")
        sys.exit(1)


def resolve_path(p: str, root: Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (root / p).resolve()
    return path


def read_checkpoint(ckpt_path: Path, verbose: bool = False):
    """加载 checkpoint，成功返回 (ckpt, None)，失败返回 (None, 错误信息)。"""
    if str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import torch
        for mod in ("train_barlow", "train_simsiam", "train_vicreg"):
            try:
                __import__(mod)
            except ImportError:
                pass
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt, None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return None, err_msg


def format_ckpt_info(ckpt_path: Path, ckpt: dict, label: str = "最优") -> str:
    """label: '最优' 用于 best，'当前/最新' 用于 latest。"""
    lines = [f"      {ckpt_path.name}:"]
    epoch = ckpt.get("epoch")
    if epoch is not None:
        lines.append(f"        训练轮次 (epoch): {epoch + 1} (0-based 存为 {epoch})")
    best_loss = ckpt.get("best_loss")
    if best_loss is not None:
        lines.append(f"        best_loss: {best_loss:.6f}")
    metrics = ckpt.get("metrics")
    if metrics:
        parts = [f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
        lines.append(f"        该轮 metrics: {', '.join(parts)}")
    global_step = ckpt.get("global_step")
    if global_step is not None:
        lines.append(f"        global_step: {global_step}")
    if "model_state_dict" in ckpt:
        n_params = sum(p.numel() for p in ckpt["model_state_dict"].values())
        lines.append(f"        模型参数量: {n_params:,}")
    return "\n".join(lines)


def format_conclusion_entry(ckpt_path: Path, ckpt: dict, name: str) -> str:
    """生成 conclusion.txt 中「各方法最优模型信息」的一条（与现有格式一致）。"""
    lines = [f"  [{name}] {ckpt_path.resolve()}"]
    epoch = ckpt.get("epoch")
    if epoch is not None:
        lines.append(f"      最优轮次 (epoch): {epoch + 1} (0-based 存为 {epoch})")
    best_loss = ckpt.get("best_loss")
    if best_loss is not None:
        lines.append(f"      最优 loss (best_loss): {best_loss:.6f}")
    metrics = ckpt.get("metrics")
    if metrics:
        parts = [f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
        lines.append(f"      该轮 metrics: {', '.join(parts)}")
    global_step = ckpt.get("global_step")
    if global_step is not None:
        lines.append(f"      global_step: {global_step}")
    if "model_state_dict" in ckpt:
        n_params = sum(p.numel() for p in ckpt["model_state_dict"].values())
        lines.append(f"      模型参数量: {n_params:,}")
    return "\n".join(lines)


def build_conclusion_block(root: Path, dirs: list) -> str:
    """仅用 checkpoint_best.pth 生成「各方法最优模型信息」整块文本。"""
    header = "============================================================\n各方法最优模型信息 (checkpoint_best.pth)\n============================================================\n"
    entries = []
    for rel_dir in dirs:
        log_dir = resolve_path(rel_dir, root) if not Path(rel_dir).is_absolute() else Path(rel_dir)
        ckpt_path = log_dir / "checkpoint_best.pth"
        name = log_dir.name
        if not ckpt_path.exists():
            entries.append(f"  [{name}] 未找到: {ckpt_path}")
            continue
        ckpt, err = read_checkpoint(ckpt_path, verbose=False)
        if ckpt is None:
            entries.append(f"  [{name}] 加载失败: {ckpt_path}" + (f" — {err}" if err else ""))
            continue
        entries.append(format_conclusion_entry(ckpt_path, ckpt, name))
    return header + "\n\n".join(entries) + "\n\n============================================================\n"


def update_conclusion_file(conclusion_path: Path, new_block: str) -> bool:
    """替换 conclusion.txt 中「各方法最优模型信息」整块。返回是否成功。"""
    try:
        text = conclusion_path.read_text(encoding="utf-8")
    except Exception:
        return False
    # 从 "====...\n各方法最优模型信息" 到块尾 "====...\n"（含）整块替换
    pattern = re.compile(
        r"============================================================\n"
        r"各方法最优模型信息 \(checkpoint_best\.pth\)\n"
        r"============================================================\n"
        r".*?"
        r"\n============================================================\n",
        re.DOTALL,
    )
    if not pattern.search(text):
        return False
    new_text = pattern.sub(new_block, text)
    conclusion_path.write_text(new_text, encoding="utf-8")
    return True


def main():
    _ensure_torch()
    parser = argparse.ArgumentParser(description="读取各方法 checkpoint 的训练轮次、loss 等信息")
    parser.add_argument(
        "--log_dirs",
        type=str,
        nargs="+",
        default=None,
        help="要读取的 log 目录列表，默认含 barlow, simsiam, simsiam_100, vicreg_25, vicreg",
    )
    parser.add_argument("--project_root", type=str, default=None, help="项目根目录，默认自动推断")
    parser.add_argument("-v", "--verbose", action="store_true", help="加载失败时打印异常")
    parser.add_argument(
        "--update_conclusion",
        type=str,
        default=None,
        metavar="PATH",
        help="用各方法 checkpoint_best.pth 信息更新 conclusion.txt 中的「各方法最优模型信息」块；传 conclusion.txt 路径，如 services/training/logs/conclusion.txt",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    dirs = args.log_dirs if args.log_dirs else DEFAULT_LOG_DIRS

    if args.update_conclusion:
        conclusion_path = resolve_path(args.update_conclusion, root) if not Path(args.update_conclusion).is_absolute() else Path(args.update_conclusion)
        new_block = build_conclusion_block(root, dirs)
        if update_conclusion_file(conclusion_path, new_block):
            print(f"已更新「各方法最优模型信息」块: {conclusion_path}")
        else:
            print(f"更新失败（未找到块或读写错误）: {conclusion_path}")
        return

    print("=" * 60)
    print("各方法 checkpoint 信息 (checkpoint_best.pth / checkpoint_latest.pth)")
    print("=" * 60)

    for rel_dir in dirs:
        log_dir = resolve_path(rel_dir, root) if not Path(rel_dir).is_absolute() else Path(rel_dir)
        name = log_dir.name
        print(f"  [{name}] {log_dir}")

        for ckpt_name in ("checkpoint_best.pth", "checkpoint_latest.pth"):
            ckpt_path = log_dir / ckpt_name
            if not ckpt_path.exists():
                print(f"      {ckpt_name}: 未找到")
                continue
            ckpt, err = read_checkpoint(ckpt_path, verbose=args.verbose)
            if ckpt is None:
                print(f"      {ckpt_name}: 加载失败" + (f" — {err}" if err else ""))
                continue
            print(format_ckpt_info(ckpt_path, ckpt))
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
