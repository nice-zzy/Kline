#!/usr/bin/env python3
"""
使用训练好的CLIP encoder进行推理
支持单张图像编码和批量编码
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "services" / "training"))

from clip_contrastive_trainer import CLIPEncoder, CandlestickRenderer, KLineDataset


class TrainedEncoder:
    """训练好的编码器推理类"""
    
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        加载训练好的编码器
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 设备类型
        """
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[Encoder] Using device: {self.device}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint["model_state_dict"]

        # 兼容 VICReg/Barlow/SimSiam：checkpoint 存的是完整模型 (encoder.xxx + projector.xxx)，只取 encoder 部分
        if any(k.startswith("encoder.") for k in state.keys()):
            encoder_state = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
            state = encoder_state
            print("[Encoder] Detected VICReg/Barlow/SimSiam checkpoint, loading encoder only")

        # 创建模型并加载
        self.model = CLIPEncoder(embedding_dim=512)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.epoch = checkpoint.get("epoch", 0)
        self.metrics = checkpoint.get("metrics", {})
        print(f"[Encoder] Loaded encoder from epoch {self.epoch}")
        if self.metrics:
            print(f"[Encoder] Metrics: {self.metrics}")
        
        # 初始化渲染器
        self.renderer = CandlestickRenderer(image_size=224)
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        编码单张图像
        
        Args:
            image: RGB图像数组 [H, W, 3]
        
        Returns:
            嵌入向量 [embedding_dim]
        """
        # 转换为tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        
        # 确保是CHW格式
        if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        
        # 归一化到[0, 1]
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 编码
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        return embedding.cpu().numpy().squeeze()
    
    def encode_ohlc_data(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        """
        编码OHLC数据
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            嵌入向量 [embedding_dim]
        """
        # 渲染蜡烛图
        image = self.renderer.render_candlestick(ohlc_data)
        
        # 编码图像
        return self.encode_image(image)
    
    def encode_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        批量编码图像
        
        Args:
            images: 图像列表
        
        Returns:
            嵌入矩阵 [batch_size, embedding_dim]
        """
        embeddings = []
        
        for image in images:
            embedding = self.encode_image(image)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
        
        Returns:
            余弦相似度 [0, 1]
        """
        # 归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def find_similar_patterns(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        找到最相似的K个模式
        
        Args:
            query_embedding: 查询嵌入向量
            candidate_embeddings: 候选嵌入矩阵 [N, embedding_dim]
            top_k: 返回前K个最相似的
        
        Returns:
            相似度排序列表 [(index, similarity), ...]
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


def demo_encoder_usage():
    """演示编码器使用方法"""
    print("🚀 CLIP Encoder Inference Demo")
    print("=" * 50)
    
    # 检查模型文件
    checkpoint_path = "services/training/logs/clip_contrastive/checkpoint_best.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return
    
    # 加载编码器
    print("📁 Loading trained encoder...")
    encoder = TrainedEncoder(checkpoint_path)
    
    # 加载测试数据
    print("📊 Loading test data...")
    test_dataset = KLineDataset(
        data_file="services/training/data/dow30_real_AAPL.csv",
        start_year=2017,
        end_year=2017,
        window_size=5,
        step_size=3,
        image_size=224,
        mode="test"
    )
    
    print(f"📈 Test dataset: {len(test_dataset)} windows")
    
    # 编码几个测试样本
    print("🔍 Encoding test samples...")
    test_embeddings = []
    
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        window_info = sample['window_info']
        
        # 渲染蜡烛图
        renderer = CandlestickRenderer(image_size=224)
        image = renderer.render_candlestick(test_dataset.windows[i])
        
        # 编码
        embedding = encoder.encode_image(image)
        test_embeddings.append(embedding)
        
        print(f"  Sample {i+1}: {window_info['start_date']} to {window_info['end_date']}")
        print(f"    Price change: {window_info['price_change']:.4f}")
        print(f"    Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # 计算相似度矩阵
    print("\n🔗 Computing similarity matrix...")
    similarities = []
    
    for i in range(len(test_embeddings)):
        for j in range(i+1, len(test_embeddings)):
            sim = encoder.compute_similarity(test_embeddings[i], test_embeddings[j])
            similarities.append((i, j, sim))
            print(f"  Similarity between sample {i+1} and {j+1}: {sim:.4f}")
    
    # 找到最相似的样本对
    if similarities:
        best_sim = max(similarities, key=lambda x: x[2])
        print(f"\n🎯 Most similar pair: samples {best_sim[0]+1} and {best_sim[1]+1}")
        print(f"   Similarity: {best_sim[2]:.4f}")
    
    print("\n✅ Encoder inference demo completed!")
    print("\n📝 Usage Summary:")
    print("1. Load encoder: encoder = TrainedEncoder('checkpoint_path')")
    print("2. Encode image: embedding = encoder.encode_image(image)")
    print("3. Encode OHLC: embedding = encoder.encode_ohlc_data(ohlc_df)")
    print("4. Compute similarity: sim = encoder.compute_similarity(emb1, emb2)")
    print("5. Find similar patterns: results = encoder.find_similar_patterns(query, candidates)")


if __name__ == "__main__":
    demo_encoder_usage()

