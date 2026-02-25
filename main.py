"""
功能：
1. 特征提取：从原始数据提取52维特征（支持单股票/多股票模式）
2. 相似对匹配：从真实数据中找到相似窗口对（支持 within_stock/cross_stock 匹配模式）
3. 图片渲染：渲染所有窗口的蜡烛图（支持 show_axes、color_scheme 配置）
4. 数据集划分：根据日期范围划分训练/验证/测试集
5. 模型训练：使用相似对训练 CLIP encoder
6. 构建推理索引：用训练好的 encoder 对所有 pair 图像编码，得到 embeddings.npy 与 metadata.json，供 API 检索

使用方法：
python main.py --steps 1
python main.py --steps all
python main.py --steps 1,2,3,3.5,4,5

可选：--config <json路径> 从文件加载配置，不指定则使用代码内默认配置。

tensorboard --logdir services/training/logs/clip_contrastive --port 6006

要运行的步骤: 1=特征提取, 2=相似对匹配, 3=图片渲染, 3.5=数据集划分, 4=模型训练, 5=构建推理索引, all=全部
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
script_dir = Path(__file__).parent  # 项目根目录
project_root = script_dir  # 项目根目录
training_dir = project_root / "services" / "training"
data_dir = training_dir / "data"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(data_dir))


def _resolve_path(path_str: str) -> Path:
    """将配置中的相对路径解析为基于 project_root 的绝对路径，便于从任意工作目录运行。"""
    p = Path(path_str)
    if not p.is_absolute():
        return (project_root / p).resolve()
    return p


@dataclass
class Config:
    """配置类：包含所有路径和参数"""
    
    # ========== 数据路径 ==========
    # 原始数据（单股票模式）
    raw_data_file: str = "services/training/data/dow30_real_AAPL.csv"
    # 原始数据目录（多股票模式，会读取该目录下所有 dow30_real_*.csv 文件）
    raw_data_dir: str = "services/training/data"
    # 是否使用多股票模式（30支股票）
    use_multi_stock: bool = True
    
    # 输出根目录（2010-2020 训练 / 2021 测试）
    output_root: str = "services/training/output/dow30_2010_2021"
    
    # 特征提取输出
    features_file: str = "services/training/output/dow30_2010_2021/features_52d.npy"
    window_metadata_file: str = "services/training/output/dow30_2010_2021/window_metadata.json"
    feature_info_file: str = "services/training/output/dow30_2010_2021/feature_info.json"
    
    # 相似对匹配输入（可以独立指定，不依赖步骤1的输出）
    similar_pairs_input_features: str = "services/training/output/dow30_2010_2021/features_52d.npy"
    similar_pairs_input_metadata: str = "services/training/output/dow30_2010_2021/window_metadata.json"
    
    # 相似对输出
    similar_pairs_dir: str = "services/training/output/dow30_2010_2021/similar_pairs"
    pairs_info_file: str = "services/training/output/dow30_2010_2021/similar_pairs/pairs_info.json"
    
    # 图片渲染输入（可以独立指定）
    render_input_metadata: str = "services/training/output/dow30_2010_2021/window_metadata.json"
    render_input_pairs_info: str = "services/training/output/dow30_2010_2021/similar_pairs/pairs_info.json"
    
    # 图片渲染输出
    all_images_file: str = "services/training/output/dow30_2010_2021/all_window_images.npy"
    pair_images_dir: str = "services/training/output/dow30_2010_2021/pair_images"
    anchor_images_file: str = "services/training/output/dow30_2010_2021/pair_images/anchor_images.npy"
    positive_images_file: str = "services/training/output/dow30_2010_2021/pair_images/positive_images.npy"
    pairs_metadata_file: str = "services/training/output/dow30_2010_2021/pair_images/pairs_metadata.json"
    dataset_info_file: str = "services/training/output/dow30_2010_2021/pair_images/dataset_info.json"
    
    # 数据集划分输入（步骤3的输出）
    split_input_anchor_images: str = "services/training/output/dow30_2010_2021/pair_images/anchor_images.npy"
    split_input_positive_images: str = "services/training/output/dow30_2010_2021/pair_images/positive_images.npy"
    split_input_pairs_metadata: str = "services/training/output/dow30_2010_2021/pair_images/pairs_metadata.json"
    
    # 数据集划分输出
    split_output_dir: str = "services/training/output/dow30_2010_2021/dataset_splits"
    train_anchor_images: str = "services/training/output/dow30_2010_2021/dataset_splits/train_anchor_images.npy"
    train_positive_images: str = "services/training/output/dow30_2010_2021/dataset_splits/train_positive_images.npy"
    train_pairs_metadata: str = "services/training/output/dow30_2010_2021/dataset_splits/train_pairs_metadata.json"
    test_anchor_images: str = "services/training/output/dow30_2010_2021/dataset_splits/test_anchor_images.npy"
    test_positive_images: str = "services/training/output/dow30_2010_2021/dataset_splits/test_positive_images.npy"
    test_pairs_metadata: str = "services/training/output/dow30_2010_2021/dataset_splits/test_pairs_metadata.json"
    split_info_file: str = "services/training/output/dow30_2010_2021/dataset_splits/split_info.json"
    
    # 训练输入（使用划分后的数据集）
    train_input_anchor_images: str = "services/training/output/dow30_2010_2021/dataset_splits/train_anchor_images.npy"
    train_input_positive_images: str = "services/training/output/dow30_2010_2021/dataset_splits/train_positive_images.npy"
    train_input_pairs_metadata: str = "services/training/output/dow30_2010_2021/dataset_splits/train_pairs_metadata.json"
    
    # 训练输出（步骤4 与 步骤5 均按 training_method 选目录：vicreg→logs/vicreg 等）
    log_dir: str = "services/training/logs/clip_contrastive"
    checkpoint_dir: str = "services/training/logs/clip_contrastive"  # 步骤5 编码器目录，默认与 training_method 对应；可改此项单独指定
    
    # 推理索引输出（Step5：按 training_method 区分目录，如 inference_index_vicreg / inference_index_simsiam）
    inference_index_dir: str = "services/training/output/dow30_2010_2021/inference_index"
    
    # ========== 特征提取参数 ==========@
    # 数据范围：2010-2021（2010-2020 用于训练，2021 用于测试）
    start_date: str = "2010-01-01"
    end_date: str = "2021-12-31"
    
    # 窗口参数
    window_size: int = 5
    step_size: int = 3
    
    # ========== 相似对匹配参数 ==========
    similarity_metric: str = "cosine"  # "cosine" 或 "euclidean"
    similarity_threshold: float = 0.85
    min_time_gap: int = 5
    max_pairs_per_anchor: int = 3
    # 匹配模式："within_stock"=股票内部匹配, "cross_stock"=跨股票匹配（跨模式匹配）
    pair_matching_mode: str = "cross_stock"
    # 匹配实现选择
    use_optimized: bool = False   # 使用向量化优化
    use_faiss: bool = True      # 使用FAISS（需要安装 faiss-cpu）

    # ========== 图片渲染参数 ==========
    image_size: int = 224
    dpi: int = 100
    save_png_images: bool = True  # 是否保存PNG图片文件（用于查看）
    save_pair_comparison: bool = True  # 是否保存相似对对比图
    show_axes: bool = True  # 是否显示坐标轴与刻度（建议 True，便于模型学习形态与尺度信息）
    color_scheme: str = "chinese"  # 颜色方案："chinese"=红涨绿跌（中国习惯），"western"=绿涨红跌（美国习惯）
    
    # ========== 数据集划分参数 ==========
    # 数据集划分日期范围（用于从相似对中筛选）
    # 2010-2020 年 30 支股票为训练数据，2021 年为测试数据（不单独划分验证集）
    train_start_date: str = "2010-01-01"  # 训练集开始日期
    train_end_date: str = "2020-12-31"   # 训练集结束日期
    test_start_date: str = "2021-01-01"  # 测试集开始日期
    test_end_date: str = "2021-12-31"    # 测试集结束日期
    
    # ========== 训练参数 ==========
    # 训练方法选择（步骤4/5）：在 main.py 中直接修改此处即可，无需命令行；嵌入前后端选用 vicreg
    # "clip"=对比学习, "simsiam"=SimSiam, "barlow"=Barlow Twins, "vicreg"=VICReg
    training_method: str = "vicreg"
    # 数据（约 2.4 万窗口 → 相似对规模约数千～数万，按此规模调参）
    apply_augmentation: bool = True
    
    # 模型
    model_name: str = "ViT-B/32"  # CLIP模型
    embedding_dim: int = 512
    
    # 训练（适合万级相似对：batch 加大以增加 in-batch 负样本，warmup 按步数略增）
    num_epochs: int = 50
    batch_size: int = 32       # 对比学习建议 32～64（显存允许可试 64）
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000   # 约 2～5% 总步数，万级对 50 epoch 约 1.5～2 万步
    
    # 设备
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # 其他
    save_interval: int = 10   # 每 N 个 epoch 保存一次
    log_interval: int = 20   # 每 N 个 batch 打印一次（batch 变大可适当增大）
    val_split: float = 0.0  # 验证集比例（0=不划分验证集，仅训练集+测试集）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


def step1_extract_features(config: Config) -> bool:
    """步骤1：特征提取"""
    print("\n" + "=" * 60)
    print("步骤1：特征提取")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        import importlib.util
        
        # 导入特征提取器
        ohlc_feature_path = project_root / "services" / "training" / "data" / "ohlc_feature_extractor.py"
        spec = importlib.util.spec_from_file_location(
            "ohlc_feature_extractor",
            ohlc_feature_path
        )
        ohlc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ohlc_module)
        OHLCFeatureExtractor = ohlc_module.OHLCFeatureExtractor
        
        windows = []
        metadata = []
        
        # ========== 多股票 / 单股票两种模式 ==========
        if config.use_multi_stock:
            # 多股票模式：读取目录下所有 dow30_real_*.csv 文件
            data_dir_path = _resolve_path(config.raw_data_dir)
            print(f"\n[加载] 多股票模式，数据目录: {data_dir_path}")
            csv_files = sorted(data_dir_path.glob("dow30_real_*.csv"))
            
            if not csv_files:
                print(f"[错误] 在 {data_dir_path} 中未找到 dow30_real_*.csv 文件")
                return False
            
            print(f"      找到 {len(csv_files)} 个股票数据文件")
            
            # 过滤日期范围
            start = pd.to_datetime(config.start_date)
            end = pd.to_datetime(config.end_date)
            
            for csv_file in csv_files:
                # 从文件名提取股票代码（如 dow30_real_AAPL.csv -> AAPL）
                symbol = csv_file.stem.replace("dow30_real_", "").upper()
                
                print(f"      处理 {symbol}...")
                df = pd.read_csv(csv_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # 过滤日期范围
                df_filtered = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
                df_filtered = df_filtered.sort_values("timestamp").reset_index(drop=True)
                
                if len(df_filtered) < config.window_size:
                    print(f"        跳过 {symbol}（数据不足）")
                    continue
                
                # 为该股票创建窗口
                stock_windows = 0
                for i in range(0, len(df_filtered) - config.window_size + 1, config.step_size):
                    window_data = df_filtered.iloc[i : i + config.window_size]
                    if len(window_data) == config.window_size:
                        windows.append(window_data)
                        metadata.append(
                            {
                                "symbol": symbol,
                                "start_date": window_data["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                                "end_date": window_data["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
                                "start_price": float(window_data["open"].iloc[0]),
                                "end_price": float(window_data["close"].iloc[-1]),
                                "price_change": float(
                                    (window_data["close"].iloc[-1] - window_data["open"].iloc[0])
                                    / window_data["open"].iloc[0]
                                ),
                            }
                        )
                        stock_windows += 1
                
                print(f"        {symbol}: {stock_windows} 个窗口")
        else:
            # 单股票模式：读取单个 CSV 文件
            raw_path = _resolve_path(config.raw_data_file)
            print(f"\n[加载] 单股票模式，原始数据: {raw_path}")
            df = pd.read_csv(raw_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # 从文件名提取股票代码（如果没有则使用默认值）
            symbol = "UNKNOWN"
            if "dow30_real_" in config.raw_data_file:
                symbol = Path(config.raw_data_file).stem.replace("dow30_real_", "").upper()
            
            # 过滤日期范围
            start = pd.to_datetime(config.start_date)
            end = pd.to_datetime(config.end_date)
            df_filtered = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
            df_filtered = df_filtered.sort_values("timestamp").reset_index(drop=True)
            
            print(f"      过滤后数据: {len(df_filtered)} 条")
            print(
                f"      日期范围: {df_filtered['timestamp'].min().date()} 到 {df_filtered['timestamp'].max().date()}"
            )
            
            # 创建窗口
            for i in range(0, len(df_filtered) - config.window_size + 1, config.step_size):
                window_data = df_filtered.iloc[i : i + config.window_size]
                if len(window_data) == config.window_size:
                    windows.append(window_data)
                    metadata.append(
                        {
                            "symbol": symbol,
                            "start_date": window_data["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                            "end_date": window_data["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
                            "start_price": float(window_data["open"].iloc[0]),
                            "end_price": float(window_data["close"].iloc[-1]),
                            "price_change": float(
                                (window_data["close"].iloc[-1] - window_data["open"].iloc[0])
                                / window_data["open"].iloc[0]
                            ),
                        }
                    )
        
        print(f"\n      总计生成窗口: {len(windows)} 个")
        if config.use_multi_stock:
            # 统计各股票的窗口数
            symbol_counts = {}
            for m in metadata:
                sym = m.get("symbol", "UNKNOWN")
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
            print(f"      股票分布: {dict(sorted(symbol_counts.items()))}")
        
        # 提取特征
        print(f"\n[提取] 开始提取特征...")
        extractor = OHLCFeatureExtractor(window_size=config.window_size)
        
        all_daily_features = []
        all_concatenated_features = []
        
        for i, window_data in enumerate(windows):
            if (i + 1) % 10 == 0:
                print(f"      已处理 {i+1}/{len(windows)} 个窗口...")
            
            try:
                # 这里沿用特征提取器现有方法：8维日特征 + 12维窗口特征 + 52维拼接
                features = extractor.extract_concatenated_features(window_data)
                daily_features = extractor.extract_8d_features(window_data)
                
                all_daily_features.append(daily_features)
                all_concatenated_features.append(features)
            except Exception as e:
                print(f"      警告: 窗口 {i} 处理失败: {e}")
                continue
        
        # 转换为 numpy 数组
        if len(all_daily_features) == 0:
            print(f"      [错误] 没有成功提取任何特征")
            return False
        
        daily_features_array = np.array(all_daily_features)  # (n_windows, 5, 8)
        concatenated_features_array = np.array(all_concatenated_features)  # (n_windows, 52)
        
        print(f"      特征提取完成")
        print(f"      成功提取: {len(all_concatenated_features)} 个窗口的特征")
        print(f"      单日特征形状: {daily_features_array.shape}")
        print(f"      拼接特征形状: {concatenated_features_array.shape}")
        
        # 保存
        output_dir = _resolve_path(config.output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = _resolve_path(config.features_file)
        metadata_path = _resolve_path(config.window_metadata_file)
        info_path = _resolve_path(config.feature_info_file)
        
        np.save(features_path, concatenated_features_array)
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        info = {
            "total_windows": len(metadata),
            "feature_dim": concatenated_features_array.shape[1],
            "date_range": {
                "start": metadata[0]["start_date"],
                "end": metadata[-1]["end_date"],
            },
            "created_at": datetime.now().isoformat(),
        }
        
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"\n[保存] 特征文件:")
        print(f"      {features_path}")
        print(f"      {metadata_path}")
        print(f"      {info_path}")
        
        print(f"\n[完成] 特征提取完成！")
        return True
        
    except Exception as e:
        print(f"\n[错误] 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step2_find_similar_pairs(config: Config) -> bool:
    """步骤2：找相似对"""
    print("\n" + "=" * 60)
    print("步骤2：找相似对")
    print("=" * 60)
    
    try:
        import numpy as np
        
        # 检查输入文件（使用配置中的输入路径）
        features_path = _resolve_path(config.similar_pairs_input_features)
        metadata_path = _resolve_path(config.similar_pairs_input_metadata)
        
        if not features_path.exists():
            print(f"[错误] 特征文件不存在: {features_path}")
            print(f"      请检查配置中的 similar_pairs_input_features")
            return False
        
        if not metadata_path.exists():
            print(f"[错误] 元数据文件不存在: {metadata_path}")
            print(f"      请检查配置中的 similar_pairs_input_metadata")
            return False
        
        # 加载数据
        print(f"\n[加载] 特征文件: {features_path}")
        features = np.load(features_path)
        print(f"      特征形状: {features.shape}")
        
        print(f"\n[加载] 元数据文件: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"      窗口数量: {len(metadata)}")
        
        # 为导入匹配脚本添加路径（沿用原有脚本目录结构）
        sys.path.insert(0, str(training_dir / "scripts"))
        
        # 选择匹配方法（根据配置选择）
        if config.use_faiss:
            print(f"\n[方法] 使用FAISS优化方法")
            try:
                from find_similar_pairs_faiss import FastSimilarityMatcher
                matcher = FastSimilarityMatcher(
                    similarity_threshold=config.similarity_threshold,
                    min_time_gap=config.min_time_gap,
                    max_pairs_per_anchor=config.max_pairs_per_anchor,
                )
            except ImportError:
                print(f"      [警告] FAISS 未安装，回退到向量化优化方法")
                print(f"      安装命令: pip install faiss-cpu")
                config.use_faiss = False
        
        if not config.use_faiss:
            if config.use_optimized:
                print(f"\n[方法] 使用向量化优化方法")
                from find_similar_pairs_optimized import OptimizedSimilarityMatcher
                matcher = OptimizedSimilarityMatcher(
                    similarity_threshold=config.similarity_threshold,
                    min_time_gap=config.min_time_gap,
                    max_pairs_per_anchor=config.max_pairs_per_anchor,
                    use_sampling=False,
                )
            else:
                print(f"\n[方法] 使用原始方法")
                from find_similar_pairs import SimilarityMatcher
                matcher = SimilarityMatcher(
                    similarity_metric=config.similarity_metric,
                    similarity_threshold=config.similarity_threshold,
                    min_time_gap=config.min_time_gap,
                    max_pairs_per_anchor=config.max_pairs_per_anchor,
                )
        
        # 找相似对
        if config.use_optimized:
            # 优化方法支持 matching_mode 参数（within_stock / cross_stock）
            pairs = matcher.find_similar_pairs(
                features,
                metadata,
                time_gap_constraint=True,
                matching_mode=config.pair_matching_mode,
            )
        else:
            # 其他方法暂时不支持 matching_mode
            pairs = matcher.find_similar_pairs(features, metadata, time_gap_constraint=True)
        
        if len(pairs) == 0:
            print(f"\n[警告] 没有找到相似窗口对，尝试降低相似度阈值...")
            matcher.similarity_threshold = 0.80
            if config.use_optimized:
                pairs = matcher.find_similar_pairs(
                    features,
                    metadata,
                    time_gap_constraint=True,
                    matching_mode=config.pair_matching_mode,
                )
            else:
                pairs = matcher.find_similar_pairs(features, metadata, time_gap_constraint=True)
        
        if len(pairs) == 0:
            print(f"\n[错误] 仍然没有找到相似窗口对")
            return False
        
        # 分析统计
        stats = matcher.analyze_pairs(pairs)
        print(f"\n[统计] 找到 {stats['total_pairs']} 个相似窗口对")
        print(f"      平均相似度: {stats['similarity_stats']['mean']:.4f}")
        
        # 保存
        pairs_dir = _resolve_path(config.similar_pairs_dir)
        pairs_dir.mkdir(parents=True, exist_ok=True)
        
        pairs_info = []
        for pair in pairs:
            pair_info = {
                "anchor_idx": pair["anchor_idx"],
                "positive_idx": pair["positive_idx"],
                "similarity": pair["similarity"],
                "anchor_date": pair["anchor_info"]["start_date"],
                "positive_date": pair["positive_info"]["start_date"],
            }
            # 如果 metadata 中包含 symbol，写入 anchor_symbol / positive_symbol
            if "symbol" in pair["anchor_info"]:
                pair_info["anchor_symbol"] = pair["anchor_info"]["symbol"]
            if "symbol" in pair["positive_info"]:
                pair_info["positive_symbol"] = pair["positive_info"]["symbol"]
            pairs_info.append(pair_info)
        
        pairs_info_file = _resolve_path(config.pairs_info_file)
        with open(pairs_info_file, "w", encoding="utf-8") as f:
            json.dump(pairs_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n[保存] 相似对文件:")
        print(f"      {pairs_info_file}")
        
        print(f"\n[完成] 相似对查找完成！")
        return True
        
    except Exception as e:
        print(f"\n[错误] 相似对查找失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_render_images(config: Config) -> bool:
    """步骤3：渲染蜡烛图"""
    print("\n" + "=" * 60)
    print("步骤3：渲染蜡烛图")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # 检查输入文件（使用配置中的输入路径）
        metadata_path = _resolve_path(config.render_input_metadata)
        pairs_info_path = _resolve_path(config.render_input_pairs_info)
        
        if not metadata_path.exists():
            print(f"[错误] 元数据文件不存在: {metadata_path}")
            print(f"      请检查配置中的 render_input_metadata")
            return False
        
        if not pairs_info_path.exists():
            print(f"[警告] 相似对文件不存在: {pairs_info_path}")
            print(f"      将渲染所有窗口（不提取相似对图片）")
            pairs_info = None
        else:
            with open(pairs_info_path, 'r', encoding='utf-8') as f:
                pairs_info = json.load(f)
        
        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            window_metadata = json.load(f)
        
        # 加载原始数据（支持多股票模式）
        if config.use_multi_stock:
            # 多股票模式：需要从多个文件加载数据
            print(f"\n[加载] 多股票模式，从多个文件加载数据...")
            data_dir_path = _resolve_path(config.raw_data_dir)
            csv_files = sorted(data_dir_path.glob("dow30_real_*.csv"))
            
            # 创建股票数据字典
            stock_data_dict = {}
            for csv_file in csv_files:
                symbol = csv_file.stem.replace("dow30_real_", "").upper()
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                stock_data_dict[symbol] = df
                print(f"      已加载 {symbol}: {len(df)} 条数据")
        else:
            # 单股票模式：只加载一个文件
            raw_path = _resolve_path(config.raw_data_file)
            print(f"\n[加载] 单股票模式，原始数据: {raw_path}")
            df = pd.read_csv(raw_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            # 从文件名提取股票代码
            symbol = "UNKNOWN"
            if "dow30_real_" in config.raw_data_file:
                symbol = Path(config.raw_data_file).stem.replace("dow30_real_", "").upper()
            stock_data_dict = {symbol: df}
        
        # 加载窗口数据
        windows = []
        for meta in window_metadata:
            start_date = pd.to_datetime(meta['start_date'])
            end_date = pd.to_datetime(meta['end_date'])
            # 从 metadata 中获取 symbol（如果存在）
            symbol = meta.get('symbol', list(stock_data_dict.keys())[0] if stock_data_dict else 'UNKNOWN')
            
            if symbol in stock_data_dict:
                df = stock_data_dict[symbol]
                window_data = df[
                    (df['timestamp'] >= start_date) & 
                    (df['timestamp'] <= end_date)
                ].copy()
                if len(window_data) > 0:
                    windows.append(window_data)
            else:
                print(f"      警告: 未找到股票 {symbol} 的数据，跳过窗口 {meta.get('start_date', 'unknown')}")
        
        print(f"      加载 {len(windows)} 个窗口")
        
        # 渲染器（支持 show_axes 和 color_scheme）
        class CandlestickRenderer:
            def __init__(self, image_size, dpi):
                self.image_size = image_size
                self.dpi = dpi
                self.fig_size = image_size / dpi
            
            def render_candlestick(self, ohlc_data, show_axes=True, color_scheme="chinese"):
                plt.ioff()
                fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size), dpi=self.dpi)
                # X轴范围：从-0.5到len(ohlc_data)-0.5，确保第一天和最后一天不被裁剪
                ax.set_xlim(-0.5, len(ohlc_data) - 0.5)
                y_min = ohlc_data['low'].min()
                y_max = ohlc_data['high'].max()
                margin = (y_max - y_min) * 0.02 or 0.01
                ax.set_ylim(y_min - margin, y_max + margin)
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')
                
                for i, (_, row) in enumerate(ohlc_data.iterrows()):
                    # 根据颜色方案确定颜色
                    is_up = row['close'] >= row['open']
                    if color_scheme == "chinese":
                        color = 'red' if is_up else 'green'
                    else:
                        color = 'green' if is_up else 'red'
                    ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
                    body_height = abs(row['close'] - row['open'])
                    body_bottom = min(row['open'], row['close'])
                    if body_height > 0:
                        rect = patches.Rectangle(
                            (i - 0.3, body_bottom), 0.6, body_height,
                            facecolor=color, edgecolor='black', linewidth=0.5
                        )
                        ax.add_patch(rect)
                
                if show_axes:
                    # 显示坐标轴：左侧 Price，下侧 Date，为文字留足空间
                    ax.set_xlabel('Date', fontsize=9)
                    ax.set_ylabel('Price', fontsize=9)
                    ax.set_xticks(range(len(ohlc_data)))
                    if 'timestamp' in ohlc_data.columns:
                        x_labels = [ohlc_data['timestamp'].iloc[i].strftime('%m-%d') for i in range(len(ohlc_data))]
                    else:
                        x_labels = [str(i + 1) for i in range(len(ohlc_data))]
                    ax.set_xticklabels(x_labels, rotation=45, fontsize=7, ha='right')
                    price_min = ohlc_data['low'].min()
                    price_max = ohlc_data['high'].max()
                    y_ticks = np.linspace(price_min, price_max, 5)
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels([f'{p:.1f}' for p in y_ticks], fontsize=7)
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                    # 为轴标签与刻度留足边距，确保左侧 "Price" 与 Y 轴数值完整露出
                    ax.xaxis.set_tick_params(pad=3)
                    ax.yaxis.set_tick_params(pad=3)
                    ax.xaxis.labelpad = 4
                    ax.yaxis.labelpad = 4
                    # 左侧留足空间，避免 "Price" 与 Y 轴刻度被裁掉
                    fig.subplots_adjust(left=0.27, right=0.97, top=0.95, bottom=0.25)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buf = canvas.buffer_rgba()
                image = np.asarray(buf)[:, :, :3]
                plt.close(fig)
                return image
        
        renderer = CandlestickRenderer(config.image_size, config.dpi)
        
        # 渲染所有窗口
        print(f"\n[渲染] 开始渲染蜡烛图...")
        print(f"      坐标轴显示: {'是' if config.show_axes else '否（仅形态）'}")
        print(f"      颜色方案: {config.color_scheme}")
        images = []
        for i, window_data in enumerate(windows):
            if (i + 1) % 10 == 0:
                print(f"      已渲染 {i+1}/{len(windows)} 个窗口...")
            image = renderer.render_candlestick(
                window_data, 
                show_axes=config.show_axes, 
                color_scheme=config.color_scheme
            )
            images.append(image)
        
        images_array = np.array(images, dtype=np.uint8)
        print(f"      渲染完成！形状: {images_array.shape}")
        
        # 保存所有图片
        all_images_path = _resolve_path(config.all_images_file)
        all_images_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(all_images_path, images_array)
        print(f"\n[保存] 所有窗口图片: {all_images_path}")
        print(f"      文件大小: {all_images_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 如果存在相似对，提取配对图片
        if pairs_info:
            print(f"\n[提取] 提取相似对图片...")
            anchor_images = []
            positive_images = []
            pair_metadata = []
            
            for pair in pairs_info:
                anchor_idx = pair["anchor_idx"]
                positive_idx = pair["positive_idx"]
                anchor_images.append(images_array[anchor_idx])
                positive_images.append(images_array[positive_idx])
                pair_metadata.append({
                    "anchor_idx": anchor_idx,
                    "positive_idx": positive_idx,
                    "similarity": pair["similarity"],
                    "anchor_date": pair["anchor_date"],
                    "positive_date": pair["positive_date"]
                })
            
            anchor_images_array = np.array(anchor_images, dtype=np.uint8)
            positive_images_array = np.array(positive_images, dtype=np.uint8)
            
            pair_images_dir = _resolve_path(config.pair_images_dir)
            pair_images_dir.mkdir(parents=True, exist_ok=True)
            
            anchor_path = _resolve_path(config.anchor_images_file)
            positive_path = _resolve_path(config.positive_images_file)
            pairs_meta_path = _resolve_path(config.pairs_metadata_file)
            
            np.save(anchor_path, anchor_images_array)
            np.save(positive_path, positive_images_array)
            
            with open(pairs_meta_path, 'w', encoding='utf-8') as f:
                json.dump(pair_metadata, f, indent=2, ensure_ascii=False)
            
            dataset_info = {
                "total_windows": len(windows),
                "total_pairs": len(pairs_info),
                "image_size": config.image_size,
                "image_shape": list(anchor_images_array.shape[1:]),
                "created_at": datetime.now().isoformat()
            }
            
            info_path = _resolve_path(config.dataset_info_file)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            print(f"      [保存] {anchor_path}")
            print(f"      [保存] {positive_path}")
            print(f"      [保存] {pairs_meta_path}")
            print(f"      [保存] {info_path}")
            
            # 保存PNG图片文件（用于查看）
            # 注意：这些PNG图片是原始渲染的图片，没有应用数据增强
            # 数据增强只在训练时（pair_dataset）应用，用于增加模型鲁棒性
            if config.save_png_images:
                print(f"\n[保存PNG] 保存PNG图片文件（原始图片，未应用数据增强）...")
                from PIL import Image
                
                png_dir = pair_images_dir / "png_images"
                png_dir.mkdir(exist_ok=True)
                
                # 保存单个图片
                anchors_png_dir = png_dir / "anchors"
                positives_png_dir = png_dir / "positives"
                anchors_png_dir.mkdir(exist_ok=True)
                positives_png_dir.mkdir(exist_ok=True)
                
                for i, (anchor_img, positive_img, pair_meta) in enumerate(zip(
                    anchor_images_array, positive_images_array, pair_metadata
                )):
                    # 保存单个anchor图片
                    anchor_pil = Image.fromarray(anchor_img)
                    anchor_png_path = anchors_png_dir / f"anchor_{i:03d}_{pair_meta['anchor_date']}.png"
                    anchor_pil.save(anchor_png_path)
                    
                    # 保存单个positive图片
                    positive_pil = Image.fromarray(positive_img)
                    positive_png_path = positives_png_dir / f"positive_{i:03d}_{pair_meta['positive_date']}.png"
                    positive_pil.save(positive_png_path)
                
                print(f"      已保存 {len(anchor_images_array)} 对图片到: {png_dir}")
                
                # 保存相似对对比图
                if config.save_pair_comparison:
                    print(f"\n[保存对比图] 生成相似对对比图...")
                    comparison_dir = png_dir / "comparisons"
                    comparison_dir.mkdir(exist_ok=True)
                    
                    # 每页显示多对
                    pairs_per_page = 5
                    num_pages = (len(pair_metadata) + pairs_per_page - 1) // pairs_per_page
                    
                    for page in range(num_pages):
                        start_idx = page * pairs_per_page
                        end_idx = min(start_idx + pairs_per_page, len(pair_metadata))
                        page_pairs = end_idx - start_idx
                        
                        fig, axes = plt.subplots(page_pairs, 2, figsize=(10, 5 * page_pairs))
                        if page_pairs == 1:
                            axes = axes.reshape(1, -1)
                        
                        for i in range(page_pairs):
                            idx = start_idx + i
                            pair = pair_metadata[idx]
                            
                            # Anchor图片
                            axes[i, 0].imshow(anchor_images_array[idx])
                            axes[i, 0].set_title(
                                f"Anchor {idx+1}\n日期: {pair['anchor_date']}\n相似度: {pair['similarity']:.4f}",
                                fontsize=10,
                                pad=5
                            )
                            axes[i, 0].axis('off')
                            
                            # Positive图片
                            axes[i, 1].imshow(positive_images_array[idx])
                            axes[i, 1].set_title(
                                f"Positive {idx+1}\n日期: {pair['positive_date']}",
                                fontsize=10,
                                pad=5
                            )
                            axes[i, 1].axis('off')
                        
                        # 调整布局，确保文字不被裁剪
                        plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)
                        comparison_path = comparison_dir / f"similar_pairs_page_{page+1:02d}.png"
                        plt.savefig(comparison_path, bbox_inches='tight', dpi=150, pad_inches=0.2, facecolor='white')
                        plt.close()
                        
                        if (page + 1) % 5 == 0:
                            print(f"      已生成 {page + 1}/{num_pages} 页对比图...")
                    
                    print(f"      已保存 {num_pages} 页对比图到: {comparison_dir}")
        
        print(f"\n[完成] 图片渲染完成！")
        return True
        
    except Exception as e:
        print(f"\n[错误] 图片渲染失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_5_split_dataset(config: Config) -> bool:
    """步骤3.5：划分训练集、测试集（按日期：2010-2020 训练，2021 测试）"""
    print("\n" + "=" * 60)
    print("步骤3.5：划分数据集")
    print("=" * 60)
    
    try:
        import numpy as np
        import pandas as pd
        
        # 检查输入文件
        anchor_path = _resolve_path(config.split_input_anchor_images)
        positive_path = _resolve_path(config.split_input_positive_images)
        pairs_meta_path = _resolve_path(config.split_input_pairs_metadata)
        
        if not anchor_path.exists():
            print(f"[错误] Anchor图片文件不存在: {anchor_path}")
            print(f"      请检查配置中的 split_input_anchor_images")
            return False
        
        if not positive_path.exists():
            print(f"[错误] Positive图片文件不存在: {positive_path}")
            print(f"      请检查配置中的 split_input_positive_images")
            return False
        
        if not pairs_meta_path.exists():
            print(f"[错误] 相似对元数据文件不存在: {pairs_meta_path}")
            print(f"      请检查配置中的 split_input_pairs_metadata")
            return False
        
        # 加载数据
        print(f"\n[加载] 加载相似对数据...")
        anchor_images = np.load(anchor_path)  # shape: (n_pairs, H, W, 3)
        positive_images = np.load(positive_path)  # shape: (n_pairs, H, W, 3)
        
        with open(pairs_meta_path, 'r', encoding='utf-8') as f:
            pairs_metadata = json.load(f)
        
        n_pairs = len(pairs_metadata)
        print(f"      总相似对数: {n_pairs}")
        print(f"      Anchor图片形状: {anchor_images.shape}")
        print(f"      Positive图片形状: {positive_images.shape}")
        
        # 解析日期范围（仅训练集 + 测试集，不单独划分验证集）
        train_start = pd.to_datetime(config.train_start_date)
        train_end = pd.to_datetime(config.train_end_date)
        test_start = pd.to_datetime(config.test_start_date)
        test_end = pd.to_datetime(config.test_end_date)
        
        print(f"\n[划分] 根据日期范围划分数据集（仅训练集 + 测试集）...")
        print(f"      训练集: {config.train_start_date} 到 {config.train_end_date}")
        print(f"      测试集: {config.test_start_date} 到 {config.test_end_date}")
        
        # 根据anchor日期划分
        train_indices = []
        test_indices = []
        other_indices = []  # 不在指定范围内的数据
        
        for i, pair in enumerate(pairs_metadata):
            anchor_date = pd.to_datetime(pair['anchor_date'])
            
            if train_start <= anchor_date <= train_end:
                train_indices.append(i)
            elif test_start <= anchor_date <= test_end:
                test_indices.append(i)
            else:
                other_indices.append(i)
        
        print(f"\n[统计] 数据集划分结果:")
        print(f"      训练集: {len(train_indices)} 对")
        print(f"      测试集: {len(test_indices)} 对")
        print(f"      其他（不在指定范围内）: {len(other_indices)} 对")
        
        if len(train_indices) == 0:
            print(f"\n[警告] 训练集为空，请检查日期范围配置")
            return False
        
        # 创建输出目录
        split_output_path = _resolve_path(config.split_output_dir)
        split_output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存训练集
        if len(train_indices) > 0:
            print(f"\n[保存] 保存训练集...")
            train_anchor = anchor_images[train_indices]
            train_positive = positive_images[train_indices]
            train_metadata = [pairs_metadata[i] for i in train_indices]
            
            train_anchor_path = _resolve_path(config.train_anchor_images)
            train_positive_path = _resolve_path(config.train_positive_images)
            train_meta_path = _resolve_path(config.train_pairs_metadata)
            
            np.save(train_anchor_path, train_anchor)
            np.save(train_positive_path, train_positive)
            with open(train_meta_path, 'w', encoding='utf-8') as f:
                json.dump(train_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"      [保存] {train_anchor_path} (shape: {train_anchor.shape})")
            print(f"      [保存] {train_positive_path} (shape: {train_positive.shape})")
            print(f"      [保存] {train_meta_path}")
        
        # 保存测试集
        if len(test_indices) > 0:
            print(f"\n[保存] 保存测试集...")
            test_anchor = anchor_images[test_indices]
            test_positive = positive_images[test_indices]
            test_metadata = [pairs_metadata[i] for i in test_indices]
            
            test_anchor_path = _resolve_path(config.test_anchor_images)
            test_positive_path = _resolve_path(config.test_positive_images)
            test_meta_path = _resolve_path(config.test_pairs_metadata)
            
            np.save(test_anchor_path, test_anchor)
            np.save(test_positive_path, test_positive)
            with open(test_meta_path, 'w', encoding='utf-8') as f:
                json.dump(test_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"      [保存] {test_anchor_path} (shape: {test_anchor.shape})")
            print(f"      [保存] {test_positive_path} (shape: {test_positive.shape})")
            print(f"      [保存] {test_meta_path}")
        
        # 保存划分信息（仅训练集 + 测试集）
        split_info = {
            "total_pairs": n_pairs,
            "train": {
                "count": len(train_indices),
                "start_date": config.train_start_date,
                "end_date": config.train_end_date,
                "indices": train_indices
            },
            "test": {
                "count": len(test_indices),
                "start_date": config.test_start_date,
                "end_date": config.test_end_date,
                "indices": test_indices
            },
            "other": {
                "count": len(other_indices),
                "indices": other_indices
            },
            "created_at": datetime.now().isoformat()
        }
        
        split_info_path = _resolve_path(config.split_info_file)
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n[保存] 划分信息: {split_info_path}")
        
        print(f"\n[完成] 数据集划分完成！")
        print(f"      输出目录: {split_output_path}")
        return True
        
    except Exception as e:
        print(f"\n[错误] 数据集划分失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_train_model(config: Config) -> bool:
    """步骤4：训练模型"""
    print("\n" + "=" * 60)
    print("步骤4：训练模型")
    print("=" * 60)
    
    try:
        # 检查输入文件（使用配置中的输入路径）
        anchor_path = _resolve_path(config.train_input_anchor_images)
        positive_path = _resolve_path(config.train_input_positive_images)
        pairs_meta_path = _resolve_path(config.train_input_pairs_metadata)
        
        if not anchor_path.exists():
            print(f"[错误] Anchor图片文件不存在: {anchor_path}")
            print(f"      请检查配置中的 train_input_anchor_images")
            return False
        
        if not positive_path.exists():
            print(f"[错误] Positive图片文件不存在: {positive_path}")
            print(f"      请检查配置中的 train_input_positive_images")
            return False
        
        if not pairs_meta_path.exists():
            print(f"[错误] 配对元数据文件不存在: {pairs_meta_path}")
            print(f"      请检查配置中的 train_input_pairs_metadata")
            return False
        
        # 导入训练相关模块
        sys.path.insert(0, str(training_dir / "scripts"))
        
        # 按训练方法确定日志目录（在 main.py 中通过 Config.training_method 选择，无需命令行）
        _log_dirs = {
            "clip": config.log_dir,
            "simsiam": "services/training/logs/simsiam",
            "barlow": "services/training/logs/barlow",
            "vicreg": "services/training/logs/vicreg",
        }
        log_dir = str(_resolve_path(_log_dirs.get(config.training_method, config.log_dir)))
        
        # 开始训练
        print(f"\n[训练] 开始训练...")
        print(f"      训练方法: {config.training_method}")
        print(f"      模型: {config.model_name}")
        print(f"      批次大小: {config.batch_size}")
        print(f"      学习率: {config.learning_rate}")
        print(f"      训练轮数: {config.num_epochs}")
        print(f"      日志目录: {log_dir}")
        print(f"      设备: {config.device}")
        
        if config.training_method == "simsiam":
            from train_simsiam import train_simsiam
            trainer, model = train_simsiam(
                anchor_images_file=str(anchor_path),
                positive_images_file=str(positive_path),
                pairs_metadata_file=str(pairs_meta_path),
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                model_name=config.model_name,
                embedding_dim=config.embedding_dim,
                image_size=config.image_size,
                log_dir=log_dir,
                device=config.device,
                val_split=config.val_split,
            )
        elif config.training_method == "clip":
            from train_with_pairs import train_with_pairs
            trainer, model = train_with_pairs(
                anchor_images_file=str(anchor_path),
                positive_images_file=str(positive_path),
                pairs_metadata_file=str(pairs_meta_path),
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                model_name=config.model_name,
                embedding_dim=config.embedding_dim,
                image_size=config.image_size,
                apply_augmentation=config.apply_augmentation,
                log_dir=log_dir,
                device=config.device,
                val_split=config.val_split,
            )
        elif config.training_method == "barlow":
            from train_barlow import train_barlow
            trainer, model = train_barlow(
                anchor_images_file=str(anchor_path),
                positive_images_file=str(positive_path),
                pairs_metadata_file=str(pairs_meta_path),
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                model_name=config.model_name,
                embedding_dim=config.embedding_dim,
                image_size=config.image_size,
                log_dir=log_dir,
                device=config.device,
                val_split=config.val_split,
            )
        elif config.training_method == "vicreg":
            from train_vicreg import train_vicreg
            trainer, model = train_vicreg(
                anchor_images_file=str(anchor_path),
                positive_images_file=str(positive_path),
                pairs_metadata_file=str(pairs_meta_path),
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                model_name=config.model_name,
                embedding_dim=config.embedding_dim,
                image_size=config.image_size,
                log_dir=log_dir,
                device=config.device,
                val_split=config.val_split,
            )
        else:
            print(f"\n[错误] 不支持的训练方法: {config.training_method}，请使用 clip / simsiam / barlow / vicreg")
            return False
        
        print(f"\n[完成] 模型训练完成！")
        log_dir_resolved = Path(log_dir)
        print(f"      最佳模型: {log_dir_resolved / 'checkpoint_best.pth'}")
        log_dir_abs = str(log_dir_resolved)
        print(f"      查看训练曲线: tensorboard --logdir \"{log_dir_abs}\" --port 6006")
        print(f"      或者（从项目根目录运行）: tensorboard --logdir {config.log_dir} --port 6006")
        return True
        
    except Exception as e:
        print(f"\n[错误] 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_build_inference_index(config: Config) -> bool:
    """步骤5：用训练好的 encoder 对**所有窗口**图像编码，得到 embeddings.npy (N, 512) 和 metadata.json，供 API 检索（N = 窗口总数，非相似对）。
    编码器目录与步骤4一致：由 Config.training_method 决定（vicreg→logs/vicreg 等）；若需指定其他目录可改 Config.checkpoint_dir。"""
    import numpy as np

    print("\n" + "=" * 60)
    print("步骤5：构建推理索引（所有窗口 → embeddings.npy + metadata.json）")
    print("=" * 60)

    try:
        # 与步骤4一致：按 training_method 选编码器目录，中控改 training_method 即同时影响步骤4与步骤5
        _log_dirs = {
            "clip": config.log_dir,
            "simsiam": "services/training/logs/simsiam",
            "barlow": "services/training/logs/barlow",
            "vicreg": "services/training/logs/vicreg",
        }
        step5_log_dir = str(_resolve_path(_log_dirs.get(config.training_method, config.checkpoint_dir)))
        checkpoint_path = Path(step5_log_dir) / "checkpoint_best.pth"
        print(f"      编码器目录（与步骤4 training_method={config.training_method} 一致）: {step5_log_dir}")
        if not checkpoint_path.exists():
            print(f"[错误] 未找到训练好的模型: {checkpoint_path}")
            print(f"      请先完成步骤4（模型训练）")
            return False

        all_images_path = _resolve_path(config.all_images_file)
        window_meta_path = _resolve_path(config.window_metadata_file)
        if not all_images_path.exists():
            print(f"[错误] 所有窗口图像文件不存在: {all_images_path}")
            print(f"      请先完成步骤3（图片渲染），会生成 all_window_images.npy")
            return False
        if not window_meta_path.exists():
            print(f"[错误] 窗口元数据不存在: {window_meta_path}")
            print(f"      请先完成步骤1（特征提取）")
            return False

        # 加载所有窗口图像与元数据（Step3 的 all_window_images.npy，与 Step1 的 window_metadata 一一对应）
        all_images = np.load(all_images_path)  # [N, H, W, 3]，N = 窗口总数
        with open(window_meta_path, "r", encoding="utf-8") as f:
            window_metadata = json.load(f)
        if len(window_metadata) != len(all_images):
            print(f"[错误] window_metadata 条数 ({len(window_metadata)}) 与图像数 ({len(all_images)}) 不一致")
            return False

        num_windows = len(all_images)
        batch_size = 32
        sys.path.insert(0, str(training_dir))
        from inference_encoder import TrainedEncoder

        print(f"\n[加载] 编码器: {checkpoint_path}")
        encoder = TrainedEncoder(str(checkpoint_path), device=config.device)

        embeddings_list = []
        print(f"[编码] 所有窗口图像共 {num_windows} 张...")
        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            batch = all_images[start:end]
            emb = encoder.encode_batch([batch[i] for i in range(len(batch))])
            embeddings_list.append(emb)

        embeddings = np.vstack(embeddings_list).astype(np.float32)
        # metadata：每条对应一个窗口，与 embeddings 行一一对应；兼容 API report（start_date、end_date、symbol、price_change）
        index_metadata = []
        for i, meta in enumerate(window_metadata):
            row = {
                "window_index": i,
                "start_date": meta.get("start_date", ""),
                "end_date": meta.get("end_date", ""),
            }
            if "symbol" in meta:
                row["symbol"] = meta["symbol"]
            if "price_change" in meta:
                row["price_change"] = meta["price_change"]
            if "start_price" in meta:
                row["start_price"] = meta["start_price"]
            if "end_price" in meta:
                row["end_price"] = meta["end_price"]
            index_metadata.append(row)
        assert len(index_metadata) == num_windows, "metadata 条数应与窗口数一致"

        # 按 encoder（training_method）区分目录，避免不同模型索引互相覆盖
        out_dir = _resolve_path(f"{config.inference_index_dir}_{config.training_method}")
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_file = out_dir / "embeddings.npy"
        meta_file = out_dir / "metadata.json"
        np.save(emb_file, embeddings)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(index_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n[完成] 推理索引已写入: {out_dir}")
        print(f"      embeddings.npy: shape {embeddings.shape}（N = 所有窗口数）")
        print(f"      metadata.json: {len(index_metadata)} 条（每窗口一条）")
        print(f"      供 API 检索: 设置 KLINE_INDEX_DIR={out_dir.resolve()}")
        return True

    except Exception as e:
        print(f"\n[错误] 构建推理索引失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="K线相似搜索系统 - 完整流程中控台")
    
    # 流程选择（支持逗号或空格分隔，如 --steps 3,3.5 或 --steps 3 3.5）
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["all"],
        help="要运行的步骤: 1=特征提取, 2=相似对匹配, 3=图片渲染, 3.5=数据集划分, 4=模型训练, 5=构建推理索引, all=全部（可写 3,3.5,4,5 等）"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选：从 JSON 文件加载配置，不指定则使用默认配置"
    )
    
    args = parser.parse_args()
    
    # 展开逗号分隔的 steps（如 "3,3.5" -> ["3", "3.5"]）并校验
    valid_steps = {"1", "2", "3", "3.5", "4", "5", "all"}
    steps_expanded = []
    for s in args.steps:
        for part in str(s).split(","):
            part = part.strip()
            if part not in valid_steps:
                parser.error(f"argument --steps: invalid choice: {part!r} (choose from 1, 2, 3, 3.5, 4, all)")
            steps_expanded.append(part)
    args.steps = steps_expanded
    
    # 加载或使用默认配置
    if args.config:
        config_path = _resolve_path(args.config) if not Path(args.config).is_absolute() else Path(args.config)
        if not config_path.exists():
            print(f"[错误] 配置文件不存在: {config_path}")
            return
        print(f"[加载] 配置文件: {config_path}")
        config = Config.load(str(config_path))
    else:
        config = Config()
    
    # 打印配置摘要
    print("=" * 60)
    print("K线相似搜索系统 - 完整流程中控台")
    print("=" * 60)
    print(f"\n[配置] 当前配置:")
    if config.use_multi_stock:
        print(f"      多股票模式: 是")
        print(f"      数据目录: {config.raw_data_dir}")
    else:
        print(f"      多股票模式: 否")
        print(f"      数据文件: {config.raw_data_file}")
    print(f"      输出目录: {config.output_root}")
    print(f"      特征提取日期范围: {config.start_date} 到 {config.end_date} (用于生成相似对候选库)")
    print(f"      窗口大小: {config.window_size}, 步长: {config.step_size}")
    print(f"      相似度阈值: {config.similarity_threshold}")
    if config.use_faiss:
        print(f"      相似对匹配方法: FAISS")
    elif config.use_optimized:
        print(f"      相似对匹配方法: 向量化优化")
        print(f"      匹配模式: {config.pair_matching_mode}")
    else:
        print(f"      相似对匹配方法: 原始循环")
    print(f"\n      数据集划分日期范围（仅训练集 + 测试集）:")
    print(f"        训练集: {config.train_start_date} 到 {config.train_end_date}")
    print(f"        测试集: {config.test_start_date} 到 {config.test_end_date}")
    print(f"\n      训练方法(步骤4): {config.training_method}, 训练轮数: {config.num_epochs}, 批次大小: {config.batch_size}")
    
    # 确定要运行的步骤
    if "all" in args.steps:
        steps_to_run = ["1", "2", "3", "3.5", "4", "5"]
    else:
        steps_to_run = args.steps
    
    print(f"\n[流程] 将运行步骤: {', '.join(steps_to_run)}")
    
    # 运行步骤
    results = {}
    
    if "1" in steps_to_run:
        results["1"] = step1_extract_features(config)
        if not results["1"]:
            print("\n[错误] 步骤1失败，停止执行")
            return
    
    if "2" in steps_to_run:
        results["2"] = step2_find_similar_pairs(config)
        if not results["2"]:
            print("\n[警告] 步骤2失败，但继续执行")
    
    if "3" in steps_to_run:
        results["3"] = step3_render_images(config)
        if not results["3"]:
            print("\n[错误] 步骤3失败，停止执行")
            return
    
    if "3.5" in steps_to_run:
        results["3.5"] = step3_5_split_dataset(config)
        if not results["3.5"]:
            print("\n[错误] 步骤3.5失败，停止执行")
            return
    
    if "4" in steps_to_run:
        results["4"] = step4_train_model(config)
        if not results["4"]:
            print("\n[错误] 步骤4失败")

    if "5" in steps_to_run:
        results["5"] = step5_build_inference_index(config)
        if not results["5"]:
            print("\n[错误] 步骤5失败")

    # 总结
    print("\n" + "=" * 60)
    print("流程执行总结")
    print("=" * 60)
    for step, success in results.items():
        status = "[成功]" if success else "[失败]"
        step_name = {
            "1": "特征提取",
            "2": "相似对匹配",
            "3": "图片渲染",
            "3.5": "数据集划分",
            "4": "模型训练",
            "5": "构建推理索引"
        }.get(step, step)
        print(f"  步骤{step} ({step_name}): {status}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()


