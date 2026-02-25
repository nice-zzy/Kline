"""
训练适配器：从图像对文件加载数据并训练CLIP模型
供main.py的step4调用

功能：
- 从NPY文件加载已渲染的图像对（anchor和positive）
- 创建数据集和数据加载器
- 初始化CLIP模型和训练器
- 执行对比学习训练
"""
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))

from clip_contrastive_trainer import (
    CLIPEncoder, 
    CLIPTrainer, 
    ContrastiveLoss,
    DataAugmentation
)


class ImagePairDataset(Dataset):
    """
    从NPY文件加载图像对的数据集
    
    用于从main.py的step3生成的图像对文件加载数据
    """
    
    def __init__(self, anchor_images_file: str, positive_images_file: str, apply_augmentation: bool = True):
        """
        初始化数据集
        
        Args:
            anchor_images_file: Anchor图像NPY文件路径
            positive_images_file: Positive图像NPY文件路径
            apply_augmentation: 是否应用数据增强
        """
        print(f"📁 Loading anchor images from: {anchor_images_file}")
        self.anchor_images = np.load(anchor_images_file)  # [N, H, W, 3]
        
        print(f"📁 Loading positive images from: {positive_images_file}")
        self.positive_images = np.load(positive_images_file)  # [N, H, W, 3]
        
        if len(self.anchor_images) != len(self.positive_images):
            raise ValueError(
                f"Mismatch: {len(self.anchor_images)} anchor images vs "
                f"{len(self.positive_images)} positive images"
            )
        
        print(f"✅ Loaded {len(self.anchor_images)} image pairs")
        print(f"   Image shape: {self.anchor_images[0].shape}")
        
        self.apply_augmentation = apply_augmentation
        
        if self.apply_augmentation:
            self.augmenter = DataAugmentation()
            print("✅ Data augmentation enabled")
        
        # 转换为tensor的transform
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.anchor_images)
    
    def __getitem__(self, idx):
        """
        获取一个图像对
        
        Returns:
            dict with keys: "anchor", "positive"
            - anchor: tensor [C, H, W]
            - positive: tensor [C, H, W]
        """
        anchor = self.anchor_images[idx]  # [H, W, 3], uint8
        positive = self.positive_images[idx]  # [H, W, 3], uint8
        
        # 确保是uint8类型
        if anchor.dtype != np.uint8:
            anchor = (anchor * 255).astype(np.uint8) if anchor.max() <= 1.0 else anchor.astype(np.uint8)
        if positive.dtype != np.uint8:
            positive = (positive * 255).astype(np.uint8) if positive.max() <= 1.0 else positive.astype(np.uint8)
        
        # 转换为PIL Image
        anchor_pil = Image.fromarray(anchor)
        positive_pil = Image.fromarray(positive)
        
        # 数据增强（如果启用）
        if self.apply_augmentation:
            anchor_pil = self.augmenter.augment_image(anchor_pil)
            positive_pil = self.augmenter.augment_image(positive_pil)
        
        # 转换为tensor [C, H, W]，值域[0, 1]
        anchor_tensor = self.to_tensor(anchor_pil)
        positive_tensor = self.to_tensor(positive_pil)
        
        return {
            "anchor": anchor_tensor,
            "positive": positive_tensor
        }


def train_with_pairs(
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
    apply_augmentation: bool = True,
    log_dir: str = "services/training/logs/clip_contrastive",
    device: str = "auto",
    val_split: float = 0.2
) -> Tuple[CLIPTrainer, CLIPEncoder]:
    """
    从图像对文件训练CLIP模型
    
    这是main.py的step4调用的训练函数，从已准备好的图像对文件加载数据并训练。
    
    Args:
        anchor_images_file: Anchor图像NPY文件路径
        positive_images_file: Positive图像NPY文件路径
        pairs_metadata_file: 相似对元数据JSON文件路径（当前未使用，保留用于未来扩展）
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        model_name: CLIP模型名称
        embedding_dim: 嵌入维度
        image_size: 图像尺寸（当前未使用，从NPY文件读取）
        apply_augmentation: 是否应用数据增强
        log_dir: 日志目录
        device: 设备类型（"auto", "cpu", "cuda"）
        val_split: 验证集比例
    
    Returns:
        trainer, model: 训练器和模型实例
    """
    # 设置设备
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"🔧 Using device: {device}")
    
    # 创建数据集
    print("\n📁 Loading image pair datasets...")
    full_dataset = ImagePairDataset(
        anchor_images_file=anchor_images_file,
        positive_images_file=positive_images_file,
        apply_augmentation=apply_augmentation
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int((1 - val_split) * total_size)
    val_size = total_size - train_size
    
    print(f"\n📊 Dataset split:")
    print(f"   Total: {total_size} pairs")
    print(f"   Train: {train_size} pairs ({100*(1-val_split):.1f}%)")
    print(f"   Val: {val_size} pairs ({100*val_split:.1f}%)")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子，确保可复现
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,  # Windows上num_workers=0更稳定
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    print(f"\n🏗️ Creating CLIP-based model...")
    print(f"   Model: {model_name}")
    print(f"   Embedding dim: {embedding_dim}")
    
    model = CLIPEncoder(
        model_name=model_name,
        embedding_dim=embedding_dim
    )
    
    # 创建损失函数
    loss_fn = ContrastiveLoss(temperature=0.07)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    print(f"\n⚙️ Training configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Log dir: {log_dir}")
    
    # 创建训练器
    trainer = CLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        log_dir=log_dir
    )
    
    # 尝试从检查点恢复训练
    checkpoint_path = Path(log_dir) / "checkpoint_latest.pth"
    if checkpoint_path.exists():
        print(f"\n📁 Found checkpoint: {checkpoint_path}")
        try:
            trainer.load_checkpoint(str(checkpoint_path))
            print("✅ Resumed training from checkpoint")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("   Starting training from scratch")
    
    # 开始训练
    print(f"\n🚀 Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Best model: {Path(log_dir) / 'checkpoint_best.pth'}")
    
    return trainer, model


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CLIP model from image pairs")
    parser.add_argument("--anchor_images", type=str, required=True)
    parser.add_argument("--positive_images", type=str, required=True)
    parser.add_argument("--pairs_metadata", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    trainer, model = train_with_pairs(
        anchor_images_file=args.anchor_images,
        positive_images_file=args.positive_images,
        pairs_metadata_file=args.pairs_metadata,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
