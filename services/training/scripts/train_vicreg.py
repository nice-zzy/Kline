"""
VICReg 训练：使用相似对、无数据增强。
损失 = 不变性(MSE) + 方差项(防坍缩) + 协方差项(去相关)。
供 main.py 步骤4 可选调用（Config.training_method = "vicreg"）。
"""
import sys
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_training_dir = _script_dir.parent
_project_root = _training_dir.parent
sys.path.insert(0, str(_training_dir))
sys.path.insert(0, str(_project_root))

from clip_contrastive_trainer import CLIPEncoder


class ImagePairDatasetNoAug(Dataset):
    """从 NPY 加载 (anchor, positive) 图像对，不做数据增强。"""

    def __init__(self, anchor_images_file: str, positive_images_file: str):
        self.anchor_images = np.load(anchor_images_file)
        self.positive_images = np.load(positive_images_file)
        if len(self.anchor_images) != len(self.positive_images):
            raise ValueError(
                f"Mismatch: {len(self.anchor_images)} anchor vs {len(self.positive_images)} positive"
            )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.anchor_images)

    def __getitem__(self, idx):
        anchor = self._to_tensor(self.anchor_images[idx])
        positive = self._to_tensor(self.positive_images[idx])
        return {"anchor": anchor, "positive": positive}

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        pil = Image.fromarray(arr)
        return self.to_tensor(pil)


class VICRegModel(nn.Module):
    """
    VICReg：encoder -> projector -> z。
    forward(x) 返回 z，两条分支各调用一次，损失为 invariance + variance + covariance。
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 512,
        proj_hidden: int = 2048,
        proj_out: int = 2048,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        z = self.projector(e)
        return z


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_inv: float = 5.0,
    lambda_var: float = 5.0,
    lambda_cov: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    VICReg: Variance-Invariance-Covariance Regularization.
    论文: VICReg (ICLR 2022), arXiv:2105.04906. Eq.(6): L = λ·s + μ·(v+v') + ν·(c+c').
    原文 4.2 节: "Coefficients λ and μ are 25 and ν is 1", ε=0.0001.
    原文 4.1 节: "we set ν=1 and perform a grid search on λ and μ with the base condition λ=μ>1."
    - Invariance s: MSE(z1, z2)，拉近两视图表示
    - Variance v: 每个特征维在 batch 上 std 不小于 1，防坍缩
    - Covariance c: 特征维间协方差接近 0，去相关
    λ/μ 取 25 为论文实验选定，可调；ν=1 固定。
    """
    B, D = z1.shape

    # Invariance
    inv_loss = nn.functional.mse_loss(z1, z2)

    # Variance: std per dimension over batch, hinge at 1
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(nn.functional.relu(1 - std_z1)) + torch.mean(
        nn.functional.relu(1 - std_z2)
    )

    # Covariance: off-diagonal of (z_centered.T @ z_centered) / (B-1)
    def _cov_loss(z: torch.Tensor) -> torch.Tensor:
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / max(B - 1, 1)
        off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
        return off_diag / D

    cov_loss = _cov_loss(z1) + _cov_loss(z2)

    return lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss


class VICRegTrainer:
    def __init__(
        self,
        model: VICRegModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str,
        lambda_inv: float = 5.0,
        lambda_var: float = 5.0,
        lambda_cov: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            anchor = batch["anchor"].to(self.device)
            positive = batch["positive"].to(self.device)
            z1 = self.model(anchor)
            z2 = self.model(positive)
            loss = vicreg_loss(
                z1, z2,
                lambda_inv=self.lambda_inv,
                lambda_var=self.lambda_var,
                lambda_cov=self.lambda_cov,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
            self.global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/n:.4f}")
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )
                self.writer.flush()
        return {"loss": total_loss / max(n, 1)}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in tqdm(dataloader, desc="Validation"):
            anchor = batch["anchor"].to(self.device)
            positive = batch["positive"].to(self.device)
            z1 = self.model(anchor)
            z2 = self.model(positive)
            loss = vicreg_loss(
                z1, z2,
                lambda_inv=self.lambda_inv,
                lambda_var=self.lambda_var,
                lambda_cov=self.lambda_cov,
            )
            total_loss += loss.item()
            n += 1
        return {"loss": total_loss / max(n, 1) if n else float("inf")}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }
        torch.save(ckpt, self.log_dir / "checkpoint_latest.pth")
        if is_best:
            torch.save(ckpt, self.log_dir / "checkpoint_best.pth")
            print(f"✅ Saved best model at epoch {epoch}")

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """返回最后一轮的 (train_metrics, val_metrics)，便于网格搜索等调用方使用。"""
        last_train = {}
        last_val = {"loss": float("inf")}
        for ep in range(self.epoch, num_epochs):
            train_metrics = self.train_epoch(train_loader, ep)
            has_val = len(val_loader.dataset) > 0
            val_metrics = self.validate(val_loader) if has_val else {"loss": float("inf")}
            last_train, last_val = train_metrics, val_metrics
            loss_for_best = val_metrics["loss"] if has_val else train_metrics["loss"]
            is_best = loss_for_best < self.best_loss
            if is_best:
                self.best_loss = loss_for_best
            self.save_checkpoint(ep, train_metrics, is_best)
            self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], ep)
            if has_val:
                self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], ep)
            print(
                f"Epoch {ep+1}/{num_epochs}  Train Loss: {train_metrics['loss']:.4f}"
                + (f"  Val Loss: {val_metrics['loss']:.4f}" if has_val else "")
            )
        return last_train, last_val


def train_vicreg(
    anchor_images_file: str,
    positive_images_file: str,
    pairs_metadata_file: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    model_name: str = "ViT-B/32",
    embedding_dim: int = 512,
    image_size: int = 224,
    log_dir: str = "services/training/logs/vicreg",
    device: str = "auto",
    val_split: float = 0.2,
    lambda_inv: float = 5.0,
    lambda_var: float = 5.0,
    lambda_cov: float = 1.0,
    return_metrics: bool = False,
) -> Tuple[VICRegTrainer, VICRegModel]:
    """
    使用相似对、无增强的 VICReg 训练。接口与 train_simsiam / main 对齐。
    """
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    log_dir = Path(log_dir)
    if not log_dir.is_absolute():
        log_dir = (_project_root / log_dir).resolve()

    print("📁 Loading image pairs (no augmentation)...")
    full_ds = ImagePairDatasetNoAug(anchor_images_file, positive_images_file)
    n = len(full_ds)
    n_train = int((1 - val_split) * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"   Total: {n}  Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )

    print("🏗️ Building VICReg (CLIP encoder + projector)...")
    encoder = CLIPEncoder(model_name=model_name, embedding_dim=embedding_dim)
    model = VICRegModel(encoder=encoder, encoder_dim=embedding_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    trainer = VICRegTrainer(
        model=model,
        optimizer=optimizer,
        device=dev,
        log_dir=str(log_dir),
        lambda_inv=lambda_inv,
        lambda_var=lambda_var,
        lambda_cov=lambda_cov,
    )

    ckpt_path = log_dir / "checkpoint_latest.pth"
    if ckpt_path.exists():
        print(f"📁 Resuming from {ckpt_path}")
        trainer.load_checkpoint(str(ckpt_path))
        if trainer.epoch >= num_epochs:
            print(
                f"⚠️  已训练到 epoch {trainer.epoch}，当前 num_epochs={num_epochs}。"
                " 若要继续迭代，请将 num_epochs 设为更大值（例如再训 50 轮则设为 100）。"
            )
        else:
            print(f"   从 epoch {trainer.epoch + 1} 继续，共训练到 epoch {num_epochs}")

    print("🚀 Starting VICReg training...")
    last_train, last_val = trainer.train(train_loader, val_loader, num_epochs)
    print("✅ VICReg training completed.")
    metrics = {"train_loss": last_train["loss"], "val_loss": last_val["loss"]}
    if return_metrics:
        return trainer, model, metrics
    return trainer, model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train VICReg from similar-pair NPY (no aug)")
    p.add_argument("--anchor_images", type=str, required=True)
    p.add_argument("--positive_images", type=str, required=True)
    p.add_argument("--pairs_metadata", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lambda_inv", type=float, default=5.0)
    p.add_argument("--lambda_var", type=float, default=5.0)
    p.add_argument("--lambda_cov", type=float, default=1.0)
    p.add_argument("--log_dir", type=str, default="services/training/logs/vicreg")
    args = p.parse_args()
    train_vicreg(
        anchor_images_file=args.anchor_images,
        positive_images_file=args.positive_images,
        pairs_metadata_file=args.pairs_metadata,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_inv=args.lambda_inv,
        lambda_var=args.lambda_var,
        lambda_cov=args.lambda_cov,
        log_dir=args.log_dir,
    )
