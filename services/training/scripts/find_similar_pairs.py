"""
从真实数据中寻找相似的正样本对

方法：
1. 计算所有窗口之间的特征相似度
2. 基于相似度阈值筛选正样本对
3. 使用余弦相似度或欧氏距离
4. 可以设置时间约束（避免使用时间上太接近的窗口）
"""
import sys
from pathlib import Path
import numpy as np
import json
from typing import List, Tuple, Dict
from datetime import datetime
from collections import defaultdict

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
data_dir = training_dir / "data"
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(data_dir))


class SimilarityMatcher:
    """相似度匹配器：从真实数据中找到相似的窗口对"""
    
    def __init__(
        self,
        similarity_metric: str = "cosine",  # "cosine" 或 "euclidean"
        similarity_threshold: float = 0.85,  # 相似度阈值
        min_time_gap: int = 10,  # 最小时间间隔（窗口数，避免使用时间上太接近的窗口）
        max_pairs_per_anchor: int = 3  # 每个锚点最多匹配的正样本数
    ):
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.min_time_gap = min_time_gap
        self.max_pairs_per_anchor = max_pairs_per_anchor
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量的相似度
        
        Args:
            features1: 特征向量1 shape=(52,)
            features2: 特征向量2 shape=(52,)
        
        Returns:
            相似度值 [0, 1]（余弦相似度）或距离值（欧氏距离，越小越相似）
        """
        if self.similarity_metric == "cosine":
            # 余弦相似度
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)
        
        elif self.similarity_metric == "euclidean":
            # 欧氏距离（转换为相似度，距离越小相似度越高）
            distance = np.linalg.norm(features1 - features2)
            # 归一化到[0, 1]，距离越小相似度越高
            # 使用exp(-distance)将距离转换为相似度
            similarity = np.exp(-distance / np.mean([np.linalg.norm(features1), np.linalg.norm(features2)]))
            return float(similarity)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def find_similar_pairs(
        self,
        features: np.ndarray,
        metadata: List[Dict],
        time_gap_constraint: bool = True
    ) -> List[Dict]:
        """
        找到所有相似的正样本对
        
        Args:
            features: 特征矩阵 shape=(n_windows, 52)
            metadata: 窗口元数据列表
            time_gap_constraint: 是否应用时间间隔约束
        
        Returns:
            正样本对列表，每个元素包含：
            {
                "anchor_idx": int,
                "positive_idx": int,
                "similarity": float,
                "anchor_info": dict,
                "positive_info": dict
            }
        """
        print(f"[匹配] 开始寻找相似窗口对...")
        print(f"      特征形状: {features.shape}")
        print(f"      相似度度量: {self.similarity_metric}")
        print(f"      相似度阈值: {self.similarity_threshold}")
        print(f"      最小时间间隔: {self.min_time_gap} 个窗口")
        
        pairs = []
        n_windows = len(features)
        
        # 计算所有窗口对之间的相似度
        print(f"\n[计算] 计算窗口间相似度矩阵...")
        similarity_matrix = np.zeros((n_windows, n_windows))
        
        for i in range(n_windows):
            for j in range(i + 1, n_windows):
                similarity = self.compute_similarity(features[i], features[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # 对称矩阵
        
        print(f"      相似度矩阵计算完成")
        print(f"      相似度范围: [{similarity_matrix[similarity_matrix > 0].min():.4f}, "
              f"{similarity_matrix.max():.4f}]")
        
        # 找到相似的对
        print(f"\n[筛选] 筛选相似窗口对...")
        for i in range(n_windows):
            # 找到与窗口i相似的所有窗口
            candidates = []
            
            for j in range(n_windows):
                if i == j:
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
        
        return pairs
    
    def analyze_pairs(self, pairs: List[Dict]) -> Dict:
        """分析正样本对的统计信息"""
        if len(pairs) == 0:
            return {}
        
        similarities = [p["similarity"] for p in pairs]
        
        # 统计相似度分布
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
    print("从真实数据中寻找相似的正样本对")
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
    
    # 创建相似度匹配器
    matcher = SimilarityMatcher(
        similarity_metric="cosine",  # 使用余弦相似度
        similarity_threshold=0.85,   # 相似度阈值85%
        min_time_gap=5,              # 最小时间间隔5个窗口
        max_pairs_per_anchor=3       # 每个锚点最多3个正样本
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
    pairs_dir = output_dir / "similar_pairs0"
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

