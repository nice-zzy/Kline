"""
基于CLIP的K线图对比学习训练器
使用2012-2016年数据训练，2017年数据测试
5天窗口，步长3，构造正样本对进行对比学习
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as transforms

# 尝试导入CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. Install with: pip install clip-openai   (or: pip install git+https://github.com/openai/CLIP.git)")


class CandlestickRenderer:
    """蜡烛图渲染器"""
    
    def __init__(self, image_size: int = 224, dpi: int = 100):
        self.image_size = image_size
        self.dpi = dpi
        self.fig_size = image_size / dpi
    
    def render_candlestick(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        """
        渲染蜡烛图
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            RGB图像数组 [H, W, 3]
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size), dpi=self.dpi)
        ax.set_xlim(0, len(ohlc_data))
        ax.set_ylim(ohlc_data['low'].min() * 0.98, ohlc_data['high'].max() * 1.02)
        
        # 设置背景为白色
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 绘制蜡烛图
        for i, (_, row) in enumerate(ohlc_data.iterrows()):
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # 确定颜色
            color = 'red' if close_price < open_price else 'green'
            
            # 绘制影线
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                rect = patches.Rectangle(
                    (i - 0.3, body_bottom), 0.6, body_height,
                    facecolor=color, edgecolor='black', linewidth=0.5
                )
                ax.add_patch(rect)
            else:
                # 十字星
                ax.plot([i - 0.3, i + 0.3], [open_price, open_price], color='black', linewidth=2)
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # 转换为numpy数组
        fig.canvas.draw()
        
        # 兼容不同matplotlib后端的API
        try:
            # 新版本matplotlib
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # 转换为RGB
            buf = buf[:, :, :3]
        except AttributeError:
            try:
                # 旧版本matplotlib
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # 备用方案：使用savefig到内存
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                from PIL import Image
                img = Image.open(buf)
                buf = np.array(img)
                if buf.shape[2] == 4:  # RGBA to RGB
                    buf = buf[:, :, :3]
        
        plt.close(fig)
        
        return buf


class DataAugmentation:
    """数据增强类"""
    
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])
    
    def augment_image(self, image):
        """
        对图像进行增强（增强版：更多颜色变化）
        
        Args:
            image: PIL Image 对象或 numpy 数组
        
        Returns:
            PIL Image 对象（增强后）
        """
        # 如果输入是numpy数组，转换为PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        enhanced = pil_image
        
        # 1. 亮度增强（范围更大：0.7-1.3）
        if np.random.random() < 0.8:  # 80%概率应用
            brightness_factor = np.random.uniform(0.7, 1.3)
            enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness_factor)
        
        # 2. 对比度增强（范围更大：0.7-1.3）
        if np.random.random() < 0.8:  # 80%概率应用
            contrast_factor = np.random.uniform(0.7, 1.3)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)
        
        # 3. 饱和度增强（新增：改变颜色鲜艳程度）
        if np.random.random() < 0.7:  # 70%概率应用
            saturation_factor = np.random.uniform(0.5, 1.5)
            enhanced = ImageEnhance.Color(enhanced).enhance(saturation_factor)
        
        # 4. 色彩增强（新增：整体色调偏移）
        if np.random.random() < 0.6:  # 60%概率应用
            color_factor = np.random.uniform(0.8, 1.2)
            enhanced = ImageEnhance.Color(enhanced).enhance(color_factor)
        
        # 5. 锐度增强（新增：让线条更清晰或更模糊）
        if np.random.random() < 0.5:  # 50%概率应用
            sharpness_factor = np.random.uniform(0.5, 1.5)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness_factor)
        
        # 6. 随机旋转（角度范围更大：-5到+5度）
        if np.random.random() < 0.6:  # 60%概率应用
            angle = np.random.uniform(-5, 5)
            enhanced = enhanced.rotate(angle, fillcolor='white', expand=False)
        
        # 7. 随机水平翻转（新增：镜像翻转）
        if np.random.random() < 0.3:  # 30%概率应用
            enhanced = ImageOps.mirror(enhanced)
        
        # 返回PIL Image（不是tensor，因为pair_dataset需要PIL Image）
        return enhanced


class CLIPEncoder(nn.Module):
    """基于CLIP的编码器"""
    
    def __init__(self, model_name: str = "ViT-B/32", embedding_dim: int = 512):
        super().__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is not available. Please install it first.")
        
        # 加载预训练的CLIP模型
        self.clip_model, self.preprocess = clip.load(model_name, device="cpu")
        
        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 获取CLIP的视觉编码器输出维度
        clip_dim = self.clip_model.visual.output_dim
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 使用CLIP的视觉编码器
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x)
        
        # 投影到目标维度
        embedding = self.projection(clip_features)
        
        # L2归一化
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


class KLineDataset(Dataset):
    """K线图数据集"""
    
    def __init__(
        self,
        data_file: str,
        start_year: int,
        end_year: int,
        window_size: int = 5,
        step_size: int = 3,
        image_size: int = 224,
        mode: str = "train"
    ):
        self.data_file = Path(data_file)
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = window_size
        self.step_size = step_size
        self.image_size = image_size
        self.mode = mode
        
        # 初始化渲染器和增强器
        self.renderer = CandlestickRenderer(image_size=image_size)
        self.augmenter = DataAugmentation()
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载和预处理数据"""
        print(f"Loading {self.mode} data from {self.data_file}...")
        
        # 读取CSV文件
        df = pd.read_csv(self.data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 按年份过滤
        df_filtered = df[df['timestamp'].dt.year.between(self.start_year, self.end_year)]
        
        if len(df_filtered) == 0:
            raise ValueError(f"No data found for years {self.start_year}-{self.end_year}")
        
        print(f"Filtered data: {len(df_filtered)} records")
        print(f"Date range: {df_filtered['timestamp'].min().date()} to {df_filtered['timestamp'].max().date()}")
        
        # 生成窗口
        self.windows = []
        for i in range(0, len(df_filtered) - self.window_size + 1, self.step_size):
            window_data = df_filtered.iloc[i:i + self.window_size]
            if len(window_data) == self.window_size:
                self.windows.append(window_data)
        
        print(f"Generated {len(self.windows)} windows")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """获取数据项"""
        window = self.windows[idx]
        
        # 渲染蜡烛图
        image = self.renderer.render_candlestick(window)
        
        # 数据增强生成正样本对
        anchor_image = self.augmenter.augment_image(image)
        positive_image = self.augmenter.augment_image(image)
        
        # 提取窗口的基本信息（避免返回DataFrame）
        window_info = {
            'start_date': window['timestamp'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': window['timestamp'].iloc[-1].strftime('%Y-%m-%d'),
            'start_price': float(window['open'].iloc[0]),
            'end_price': float(window['close'].iloc[-1]),
            'price_change': float((window['close'].iloc[-1] - window['open'].iloc[0]) / window['open'].iloc[0])
        }
        
        return {
            "anchor": anchor_image,
            "positive": positive_image,
            "window_info": window_info,
            "index": idx
        }


class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            anchor: 锚点嵌入 [batch_size, embedding_dim]
            positive: 正样本嵌入 [batch_size, embedding_dim]
        """
        batch_size = anchor.shape[0]
        device = anchor.device
        
        # 计算相似度矩阵
        anchor_norm = torch.nn.functional.normalize(anchor, p=2, dim=1)
        positive_norm = torch.nn.functional.normalize(positive, p=2, dim=1)
        
        # 计算正样本相似度
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature
        
        # 计算所有样本间的相似度矩阵（用于负样本）
        all_embeddings = torch.cat([anchor_norm, positive_norm], dim=0)
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature
        
        # 创建掩码，排除对角线
        mask = torch.eye(batch_size, device=device).bool()
        mask = torch.cat([mask, mask], dim=0)
        mask = torch.cat([mask, mask], dim=1)
        
        # 获取负样本相似度
        neg_sim = sim_matrix[mask].view(batch_size, -1)
        
        # 计算logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # 创建标签（正样本在位置0）
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 计算交叉熵损失
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        return loss


class CLIPTrainer:
    """CLIP对比学习训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        log_dir: str = None  # 如果为None，将在使用时基于脚本位置计算
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        # 如果log_dir为None，使用默认路径（基于脚本位置）
        if log_dir is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            log_dir = str(project_root / "services" / "training" / "logs" / "clip_contrastive")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动模型到设备
        self.model.to(device)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def load_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练状态
        self.epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"   Best loss so far: {self.best_loss:.4f}")
        print(f"   Will resume from epoch {self.epoch}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            anchor_images = batch["anchor"].to(self.device)
            positive_images = batch["positive"].to(self.device)
            
            # 前向传播
            anchor_embeddings = self.model(anchor_images)
            positive_embeddings = self.model(positive_images)
            
            # 计算损失
            loss = self.loss_fn(anchor_embeddings, positive_embeddings)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 每 10 个 step 刷新一次日志（避免丢失数据）
            if self.global_step % 10 == 0:
                self.writer.flush()
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/num_batches:.4f}"
            })
            
            # 记录到TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/learning_rate", 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                anchor_images = batch["anchor"].to(self.device)
                positive_images = batch["positive"].to(self.device)
                
                # 前向传播
                anchor_embeddings = self.model(anchor_images)
                positive_embeddings = self.model(positive_images)
                
                # 计算损失
                loss = self.loss_fn(anchor_embeddings, positive_embeddings)
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches == 0:
            return {"loss": float("inf")}  # 无验证样本时避免除零
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
            "best_loss": self.best_loss
        }
        
        # 保存最新检查点
        checkpoint_path = self.log_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.log_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model at epoch {epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """完整训练流程"""
        print(f"🚀 Starting CLIP contrastive learning for {num_epochs} epochs...")
        
        for epoch in range(self.epoch, num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 有验证集才做验证；无验证集时仅用训练损失做最佳判定与保存
            has_val = len(val_loader.dataset) > 0
            if has_val:
                val_metrics = self.validate(val_loader)
                self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
                is_best = val_metrics["loss"] < self.best_loss
                if is_best:
                    self.best_loss = val_metrics["loss"]
                save_metrics = val_metrics
            else:
                is_best = train_metrics["loss"] < self.best_loss
                if is_best:
                    self.best_loss = train_metrics["loss"]
                save_metrics = train_metrics
            
            self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            self.writer.flush()
            
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, save_metrics, is_best)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}", end="")
            if has_val:
                print(f"  |  Val Loss: {val_metrics['loss']:.4f}", end="")
            if is_best:
                print("  |  🎉 New best model!", end="")
            print()
        
        print("✅ Training completed!")
        # 确保所有日志都写入磁盘
        self.writer.flush()
        # 等待一下确保数据完全写入
        import time
        time.sleep(0.5)
        self.writer.close()
        # 再次等待确保文件关闭
        time.sleep(0.5)
        print(f"📊 TensorBoard logs saved to: {self.log_dir}")
        print(f"   Events files: {list(self.log_dir.glob('*.tfevents*'))}")


def main():
    """主训练函数"""
    # 获取项目根目录（基于脚本位置）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    parser = argparse.ArgumentParser(description="Train CLIP-based K-line contrastive encoder")
    
    # 数据参数（使用绝对路径作为默认值）
    default_data_file = str(project_root / "services" / "training" / "data" / "dow30_real_AAPL.csv")
    parser.add_argument("--data_file", type=str, default=default_data_file)
    parser.add_argument("--train_start_year", type=int, default=2012)
    parser.add_argument("--train_end_year", type=int, default=2016)
    parser.add_argument("--test_year", type=int, default=2017)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=224)
    
    # 模型参数
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--embedding_dim", type=int, default=512)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    # 损失函数参数
    parser.add_argument("--temperature", type=float, default=0.07)
    
    # 其他参数（使用绝对路径作为默认值）
    default_log_dir = str(project_root / "services" / "training" / "logs" / "clip_contrastive")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_dir", type=str, default=default_log_dir)
    
    args = parser.parse_args()
    
    # 将相对路径转换为绝对路径（如果用户提供了相对路径）
    if not Path(args.data_file).is_absolute():
        args.data_file = str(Path(args.data_file).resolve())
    if not Path(args.log_dir).is_absolute():
        args.log_dir = str(Path(args.log_dir).resolve())
    
    # 检查CLIP是否可用
    if not CLIP_AVAILABLE:
        print("❌ CLIP is not available. Please install it first:")
        print("pip install git+https://github.com/openai/CLIP.git")
        return
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # 创建数据集
    print("📁 Loading datasets...")
    train_dataset = KLineDataset(
        data_file=args.data_file,
        start_year=args.train_start_year,
        end_year=args.train_end_year,
        window_size=args.window_size,
        step_size=args.step_size,
        image_size=args.image_size,
        mode="train"
    )
    
    test_dataset = KLineDataset(
        data_file=args.data_file,
        start_year=args.test_year,
        end_year=args.test_year,
        window_size=args.window_size,
        step_size=args.step_size,
        image_size=args.image_size,
        mode="test"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    n_train = len(train_dataset)
    n_batches = len(train_loader)
    print(f"📊 训练集: {n_train} 条样本, 每 epoch {n_batches} 个 batch (batch_size={args.batch_size})")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 创建模型
    print("🏗️ Creating CLIP-based model...")
    model = CLIPEncoder(
        model_name=args.clip_model,
        embedding_dim=args.embedding_dim
    )
    
    # 创建损失函数
    loss_fn = ContrastiveLoss(temperature=args.temperature)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建训练器
    trainer = CLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        log_dir=args.log_dir
    )
    
    # 尝试从检查点恢复训练（默认自动恢复）
    checkpoint_path = Path(args.log_dir) / "checkpoint_latest.pth"
    if checkpoint_path.exists():
        print(f"📁 Found checkpoint: {checkpoint_path}")
        try:
            trainer.load_checkpoint(str(checkpoint_path))
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("   Starting training from scratch")
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=args.num_epochs
    )


if __name__ == "__main__":
    main()
