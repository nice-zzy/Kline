"""
相似对数据集类（用于训练）

从预渲染的图片中加载相似对数据
"""
import sys
from pathlib import Path
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
data_dir = training_dir / "data"
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(data_dir))

from clip_contrastive_trainer import DataAugmentation


class SimilarPairDataset(Dataset):
    """
    相似对数据集
    
    从预渲染的图片中加载anchor和positive对
    """
    
    def __init__(
        self,
        anchor_images_file: str,
        positive_images_file: str,
        pairs_metadata_file: str,
        image_size: int = 224,
        apply_augmentation: bool = True
    ):
        """
        初始化数据集
        
        Args:
            anchor_images_file: Anchor图片文件路径 (.npy)
            positive_images_file: Positive图片文件路径 (.npy)
            pairs_metadata_file: 配对元数据文件路径 (.json)
            image_size: 图片尺寸
            apply_augmentation: 是否应用数据增强
        """
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        
        # 加载图片数组
        print(f"[加载] Anchor图片: {anchor_images_file}")
        self.anchor_images = np.load(anchor_images_file)  # shape: (n_pairs, H, W, 3)
        print(f"      Anchor形状: {self.anchor_images.shape}")
        
        print(f"[加载] Positive图片: {positive_images_file}")
        self.positive_images = np.load(positive_images_file)  # shape: (n_pairs, H, W, 3)
        print(f"      Positive形状: {self.positive_images.shape}")
        
        # 加载元数据
        print(f"[加载] 配对元数据: {pairs_metadata_file}")
        with open(pairs_metadata_file, 'r', encoding='utf-8') as f:
            self.pairs_metadata = json.load(f)
        print(f"      配对数量: {len(self.pairs_metadata)}")
        
        # 初始化数据增强器（可选）
        if self.apply_augmentation:
            self.augmenter = DataAugmentation()
        else:
            self.augmenter = None
        
        # 验证数据一致性
        assert len(self.anchor_images) == len(self.positive_images), \
            f"Anchor和Positive数量不匹配: {len(self.anchor_images)} vs {len(self.positive_images)}"
        assert len(self.anchor_images) == len(self.pairs_metadata), \
            f"图片和元数据数量不匹配: {len(self.anchor_images)} vs {len(self.pairs_metadata)}"
    
    def __len__(self):
        return len(self.anchor_images)
    
    def __getitem__(self, idx):
        """
        获取一个相似对
        
        Returns:
            dict: {
                "anchor": torch.Tensor,  # shape=(3, H, W)
                "positive": torch.Tensor,  # shape=(3, H, W)
                "pair_info": dict,  # 配对信息
                "index": int
            }
        """
        # 获取图片（numpy数组，uint8, [0, 255]）
        anchor_image = self.anchor_images[idx].copy()
        positive_image = self.positive_images[idx].copy()
        
        # 转换为PIL Image（用于数据增强）
        from PIL import Image
        anchor_pil = Image.fromarray(anchor_image)
        positive_pil = Image.fromarray(positive_image)
        
        # 应用数据增强（可选）
        if self.apply_augmentation:
            anchor_pil = self.augmenter.augment_image(anchor_pil)
            positive_pil = self.augmenter.augment_image(positive_pil)
        
        # 转换为numpy数组
        anchor_array = np.array(anchor_pil).astype(np.float32) / 255.0  # [0, 1]
        positive_array = np.array(positive_pil).astype(np.float32) / 255.0  # [0, 1]
        
        # 转换为torch tensor并调整维度 (H, W, 3) -> (3, H, W)
        anchor_tensor = torch.from_numpy(anchor_array).permute(2, 0, 1)
        positive_tensor = torch.from_numpy(positive_array).permute(2, 0, 1)
        
        # 获取配对信息
        pair_info = self.pairs_metadata[idx]
        
        return {
            "anchor": anchor_tensor,
            "positive": positive_tensor,
            "pair_info": pair_info,
            "index": idx
        }


def test_dataset():
    """测试数据集"""
    print("=" * 60)
    print("测试相似对数据集")
    print("=" * 60)
    
    # 配置路径
    output_dir = training_dir / "output" / "aapl_2012_jan_jun"
    pair_images_dir = output_dir / "pair_images"
    
    anchor_file = pair_images_dir / "anchor_images.npy"
    positive_file = pair_images_dir / "positive_images.npy"
    pairs_meta_file = pair_images_dir / "pairs_metadata.json"
    
    if not anchor_file.exists() or not positive_file.exists():
        print(f"[错误] 图片文件不存在，请先运行 prepare_pair_images.py")
        return
    
    # 创建数据集
    dataset = SimilarPairDataset(
        anchor_images_file=str(anchor_file),
        positive_images_file=str(positive_file),
        pairs_metadata_file=str(pairs_meta_file),
        image_size=224,
        apply_augmentation=True
    )
    
    print(f"\n[测试] 数据集大小: {len(dataset)}")
    
    # 测试获取一个样本
    sample = dataset[0]
    anchor = sample["anchor"]
    positive = sample["positive"]
    pair_info = sample["pair_info"]
    
    print(f"\n[样本] 配对 0:")
    print(f"      Anchor形状: {anchor.shape}")
    print(f"      Positive形状: {positive.shape}")
    print(f"      相似度: {pair_info['similarity']:.4f}")
    print(f"      Anchor日期: {pair_info['anchor_date']}")
    print(f"      Positive日期: {pair_info['positive_date']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"\n[测试] DataLoader:")
    print(f"      Batch大小: {batch['anchor'].shape[0]}")
    print(f"      Anchor形状: {batch['anchor'].shape}")
    print(f"      Positive形状: {batch['positive'].shape}")
    
    print(f"\n[完成] 数据集测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()

