"""
优化版相似窗口对匹配（不依赖FAISS）

优化方法：
1. 向量化计算：使用numpy矩阵运算替代循环
2. 采样策略：只计算部分窗口对的相似度
3. 批量处理：批量计算相似度
"""
import sys
from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Tuple
from datetime import datetime

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
data_dir = training_dir / "data"
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(data_dir))


class OptimizedSimilarityMatcher:
    """优化的相似度匹配器：使用向量化计算和采样策略"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_time_gap: int = 10,
        max_pairs_per_anchor: int = 3,
        use_sampling: bool = False,  # 是否使用采样策略
        sampling_ratio: float = 0.3  # 采样比例（仅当use_sampling=True时）
    ):
        self.similarity_threshold = similarity_threshold
        self.min_time_gap = min_time_gap
        self.max_pairs_per_anchor = max_pairs_per_anchor
        self.use_sampling = use_sampling
        self.sampling_ratio = sampling_ratio
    
    def compute_similarity_matrix_vectorized(self, features: np.ndarray) -> np.ndarray:
        """
        使用向量化计算相似度矩阵（比循环快很多）
        
        时间复杂度：O(n²) 但使用矩阵运算，比循环快10-100倍
        
        Args:
            features: 特征矩阵 shape=(n_windows, 52)
        
        Returns:
            相似度矩阵 shape=(n_windows, n_windows)
        """
        # 归一化特征
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # 计算余弦相似度矩阵（矩阵乘法）
        similarity_matrix = np.dot(features_norm, features_norm.T)
        
        return similarity_matrix
    
    def compute_similarity_matrix_sampled(self, features: np.ndarray) -> np.ndarray:
        """
        使用采样策略计算相似度矩阵（只计算部分对）
        
        时间复杂度：O(n² × sampling_ratio)
        
        Args:
            features: 特征矩阵 shape=(n_windows, 52)
        
        Returns:
            相似度矩阵 shape=(n_windows, n_windows)，未采样的位置为0
        """
        n_windows = len(features)
        similarity_matrix = np.zeros((n_windows, n_windows))
        
        # 归一化特征
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # 采样策略：只计算时间间隔 >= min_time_gap 的窗口对
        # 这样可以减少计算量，同时保证找到的相似对满足时间约束
        for i in range(n_windows):
            # 只计算时间间隔 >= min_time_gap 的窗口
            start_j = i + self.min_time_gap
            end_j = n_windows
            
            if start_j < end_j:
                # 向量化计算这一段的相似度
                similarities = np.dot(features_norm[i:i+1], features_norm[start_j:end_j].T).flatten()
                similarity_matrix[i, start_j:end_j] = similarities
                similarity_matrix[start_j:end_j, i] = similarities  # 对称矩阵
        
        return similarity_matrix
    
    def find_similar_pairs_vectorized(
        self,
        features: np.ndarray,
        metadata: List[Dict],
        time_gap_constraint: bool = True,
        matching_mode: str = "cross_stock"  # "within_stock" 或 "cross_stock"
    ) -> List[Dict]:
        """
        使用向量化计算找到相似对（推荐方法）
        
        时间复杂度：O(n²) 但使用矩阵运算，比循环快很多
        
        Args:
            matching_mode: "within_stock"=只匹配同一股票, "cross_stock"=跨股票匹配
        """
        print(f"[匹配] 开始寻找相似窗口对...")
        print(f"      特征形状: {features.shape}")
        print(f"      相似度阈值: {self.similarity_threshold}")
        print(f"      最小时间间隔: {self.min_time_gap} 个窗口")
        print(f"      匹配模式: {matching_mode}")
        
        n_windows = len(features)
        
        # 检查metadata是否包含symbol字段
        has_symbol = len(metadata) > 0 and 'symbol' in metadata[0]
        if matching_mode == "within_stock" and not has_symbol:
            print(f"      [警告] 匹配模式为'within_stock'但metadata中无symbol字段，将使用cross_stock模式")
            matching_mode = "cross_stock"
        
        # 计算相似度矩阵（向量化）
        print(f"\n[计算] 使用向量化计算相似度矩阵...")
        if self.use_sampling:
            similarity_matrix = self.compute_similarity_matrix_sampled(features)
            print(f"      使用采样策略（只计算时间间隔 >= {self.min_time_gap} 的对）")
        else:
            similarity_matrix = self.compute_similarity_matrix_vectorized(features)
            print(f"      使用完整相似度矩阵")
        
        print(f"      相似度范围: [{similarity_matrix[similarity_matrix > 0].min():.4f}, "
              f"{similarity_matrix.max():.4f}]")
        
        # 找到相似的对
        print(f"\n[筛选] 筛选相似窗口对...")
        pairs = []
        
        for i in range(n_windows):
            candidates = []
            
            for j in range(n_windows):
                if i == j:
                    continue
                
                # 股票过滤：如果模式为within_stock，只匹配同一股票
                if matching_mode == "within_stock" and has_symbol:
                    if metadata[i].get('symbol') != metadata[j].get('symbol'):
                        continue
                
                similarity = similarity_matrix[i, j]
                
                # 检查相似度阈值
                if similarity < self.similarity_threshold:
                    continue
                
                # 检查时间间隔约束
                if time_gap_constraint:
                    time_gap = abs(i - j)
                    if time_gap < self.min_time_gap:
                        continue
                
                candidates.append({
                    "idx": j,
                    "similarity": similarity
                })
            
            # 按相似度排序，选择top-k
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            selected = candidates[:self.max_pairs_per_anchor]
            
            # 创建正样本对
            for candidate in selected:
                pairs.append({
                    "anchor_idx": i,
                    "positive_idx": candidate["idx"],
                    "similarity": candidate["similarity"],
                    "anchor_info": metadata[i],
                    "positive_info": metadata[candidate["idx"]]
                })
        
        print(f"      找到 {len(pairs)} 个相似窗口对")
        
        # 统计信息
        if matching_mode == "within_stock" and has_symbol:
            same_stock_pairs = sum(1 for p in pairs 
                                 if p['anchor_info'].get('symbol') == p['positive_info'].get('symbol'))
            print(f"      其中同股票匹配: {same_stock_pairs} 对")
        elif matching_mode == "cross_stock" and has_symbol:
            cross_stock_pairs = sum(1 for p in pairs 
                                  if p['anchor_info'].get('symbol') != p['positive_info'].get('symbol'))
            same_stock_pairs = len(pairs) - cross_stock_pairs
            print(f"      其中同股票匹配: {same_stock_pairs} 对，跨股票匹配: {cross_stock_pairs} 对")
        
        return pairs
    
    def find_similar_pairs(
        self,
        features: np.ndarray,
        metadata: List[Dict],
        time_gap_constraint: bool = True,
        matching_mode: str = "cross_stock"
    ) -> List[Dict]:
        """
        主方法：使用向量化计算找到相似对
        
        Args:
            matching_mode: "within_stock"=只匹配同一股票, "cross_stock"=跨股票匹配
        """
        return self.find_similar_pairs_vectorized(features, metadata, time_gap_constraint, matching_mode)
    
    def analyze_pairs(self, pairs: List[Dict]) -> Dict:
        """分析正样本对的统计信息"""
        if len(pairs) == 0:
            return {}
        
        similarities = [p["similarity"] for p in pairs]
        
        stats = {
            "total_pairs": len(pairs),
            "similarity_stats": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(min(similarities)),
                "max": float(max(similarities)),
                "median": float(np.median(similarities))
            },
            "similarity_distribution": {
                "high": sum(1 for s in similarities if s >= 0.95),
                "medium": sum(1 for s in similarities if 0.85 <= s < 0.95),
                "low": sum(1 for s in similarities if s < 0.85)
            }
        }
        
        return stats


def main():
    """主函数"""
    print("=" * 60)
    print("优化版相似窗口对匹配（向量化计算）")
    print("=" * 60)
    
    # 配置路径
    output_dir = training_dir / "output" / "aapl_2012_jan_jun"
    features_file = output_dir / "features_52d.npy"
    metadata_file = output_dir / "window_metadata.json"
    
    if not features_file.exists() or not metadata_file.exists():
        print(f"[错误] 特征文件不存在，请先运行 test_feature_extraction.py")
        return
    
    # 加载数据
    print(f"\n[加载] 特征文件: {features_file}")
    features = np.load(features_file)  # shape: (n_windows, 52)
    print(f"      特征形状: {features.shape}")
    
    print(f"\n[加载] 元数据文件: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"      窗口数量: {len(metadata)}")
    
    # 根据数据规模选择策略
    n_windows = len(features)
    
    if n_windows < 10000:
        # 小中规模：使用完整向量化计算
        print(f"\n[策略] 数据规模 ({n_windows} 个窗口)，使用完整向量化计算")
        matcher = OptimizedSimilarityMatcher(
            similarity_threshold=0.85,
            min_time_gap=5,
            max_pairs_per_anchor=3,
            use_sampling=False
        )
    else:
        # 大规模：使用采样策略
        print(f"\n[策略] 数据规模较大 ({n_windows} 个窗口)，使用采样策略")
        matcher = OptimizedSimilarityMatcher(
            similarity_threshold=0.85,
            min_time_gap=5,
            max_pairs_per_anchor=3,
            use_sampling=True,
            sampling_ratio=0.3
        )
    
    # 寻找相似对
    pairs = matcher.find_similar_pairs(features, metadata, time_gap_constraint=True)
    
    if len(pairs) == 0:
        print(f"\n[警告] 没有找到相似窗口对，尝试降低相似度阈值...")
        matcher.similarity_threshold = 0.80
        pairs = matcher.find_similar_pairs(features, metadata, time_gap_constraint=True)
    
    if len(pairs) == 0:
        print(f"\n[错误] 仍然没有找到相似窗口对，请检查数据或调整参数")
        return
    
    # 分析正样本对
    print(f"\n[分析] 分析正样本对统计信息...")
    stats = matcher.analyze_pairs(pairs)
    
    print(f"\n统计信息:")
    print(f"  总对数: {stats['total_pairs']}")
    print(f"  相似度统计:")
    sim_stats = stats['similarity_stats']
    print(f"    均值: {sim_stats['mean']:.4f}")
    print(f"    标准差: {sim_stats['std']:.4f}")
    print(f"    范围: [{sim_stats['min']:.4f}, {sim_stats['max']:.4f}]")
    print(f"    中位数: {sim_stats['median']:.4f}")
    
    dist = stats['similarity_distribution']
    print(f"  相似度分布:")
    print(f"    高相似度(>=0.95): {dist['high']} 对")
    print(f"    中相似度(0.85-0.95): {dist['medium']} 对")
    print(f"    低相似度(<0.85): {dist['low']} 对")
    
    # 显示一些示例
    print(f"\n[示例] 显示前5个相似窗口对:")
    for i, pair in enumerate(pairs[:5]):
        anchor = pair["anchor_info"]
        positive = pair["positive_info"]
        print(f"\n  对 {i+1}:")
        print(f"    锚点: {anchor['start_date']} 到 {anchor['end_date']} "
              f"(价格变化: {anchor['price_change']:.4f})")
        print(f"    正样本: {positive['start_date']} 到 {positive['end_date']} "
              f"(价格变化: {positive['price_change']:.4f})")
        print(f"    相似度: {pair['similarity']:.4f}")
    
    # 保存正样本对
    print(f"\n[保存] 保存正样本对...")
    pairs_dir = output_dir / "similar_pairs_optimized"
    pairs_dir.mkdir(exist_ok=True)
    
    pairs_info = []
    
    for pair in pairs:
        anchor_idx = pair["anchor_idx"]
        positive_idx = pair["positive_idx"]
        
        pairs_info.append({
            "anchor_idx": anchor_idx,
            "positive_idx": positive_idx,
            "similarity": pair["similarity"],
            "anchor_date": pair["anchor_info"]["start_date"],
            "positive_date": pair["positive_info"]["start_date"]
        })
    
    # 保存
    pairs_info_file = pairs_dir / "pairs_info.json"
    with open(pairs_info_file, 'w', encoding='utf-8') as f:
        json.dump(pairs_info, f, indent=2, ensure_ascii=False)
    
    print(f"  [保存] {pairs_info_file}")
    
    print(f"\n[完成] 相似窗口对查找完成！")
    print(f"      输出目录: {pairs_dir}")
    print(f"      找到 {len(pairs)} 个相似窗口对")
    print("=" * 60)


if __name__ == "__main__":
    main()

