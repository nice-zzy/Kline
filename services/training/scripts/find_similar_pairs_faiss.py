"""
使用FAISS快速寻找相似的正样本对（精确搜索版）

优势：
1. 底层优化：使用C++实现和SIMD指令集优化，比NumPy快2-4倍
2. 支持大规模数据：可以处理数万甚至数十万个窗口
3. 支持GPU加速：可以使用GPU进一步加速
4. 批量搜索：高效的批量查询接口
5. Top-K优化：使用堆排序，时间复杂度O(n log k)而非O(n log n)
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
sys.path.insert(0, str(script_dir))  # 添加 scripts 目录到路径，以便导入同目录的模块

# 导入FAISS工具（从同目录导入）
from vector_store_faiss import build_index, topk_search, l2_normalize


class FastSimilarityMatcher:
    """快速相似度匹配器：使用FAISS进行高效相似度搜索"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,  # 相似度阈值
        min_time_gap: int = 10,  # 最小时间间隔（窗口数）
        max_pairs_per_anchor: int = 3  # 每个锚点最多匹配的正样本数
    ):
        self.similarity_threshold = similarity_threshold
        self.min_time_gap = min_time_gap
        self.max_pairs_per_anchor = max_pairs_per_anchor
        self.index = None
    
    def build_index(self, features: np.ndarray):
        """
        构建FAISS索引（使用精确搜索，不使用K-means）
        
        Args:
            features: 特征矩阵 shape=(n_windows, 52)
        """
        print(f"[构建索引] 特征形状: {features.shape}")
        
        # 使用精确搜索（IndexFlatIP），不使用K-means
        self.index = build_index(features, metric="ip", use_normalize=True)
        print(f"      使用精确搜索索引 (IndexFlatIP)")
        
        print(f"[完成] 索引构建完成")
    
    def find_similar_pairs(
        self,
        features: np.ndarray,
        metadata: List[Dict],
        time_gap_constraint: bool = True
    ) -> List[Dict]:
        """
        使用FAISS快速找到所有相似的正样本对
        
        Args:
            features: 特征矩阵 shape=(n_windows, 52)
            metadata: 窗口元数据列表
            time_gap_constraint: 是否应用时间间隔约束
        
        Returns:
            正样本对列表
        """
        print(f"[匹配] 开始寻找相似窗口对...")
        print(f"      特征形状: {features.shape}")
        print(f"      相似度阈值: {self.similarity_threshold}")
        print(f"      最小时间间隔: {self.min_time_gap} 个窗口")
        print(f"      每个锚点最多匹配: {self.max_pairs_per_anchor} 个正样本")
        
        n_windows = len(features)
        
        # 构建索引（如果还没有构建）
        if self.index is None:
            self.build_index(features)
        
        # 为每个窗口搜索最相似的窗口
        print(f"\n[搜索] 使用FAISS搜索相似窗口...")
        pairs = []
        
        # 批量搜索（FAISS支持批量搜索，更高效）
        # 搜索每个窗口的top-k相似窗口（k = max_pairs_per_anchor + min_time_gap）
        search_k = self.max_pairs_per_anchor * 2 + self.min_time_gap
        
        # 使用FAISS搜索
        scores, indices = topk_search(
            self.index,
            features.astype(np.float32),
            k=min(search_k, n_windows),
            metric="ip",
            use_normalize=True
        )
        
        print(f"      搜索完成，处理结果...")
        
        # 处理搜索结果
        for i in range(n_windows):
            # 获取窗口i的相似窗口
            similar_indices = indices[i]
            similar_scores = scores[i]
            
            # 筛选符合条件的正样本
            candidates = []
            
            for j, (idx, score) in enumerate(zip(similar_indices, similar_scores)):
                # 跳过自己
                if idx == i:
                    continue
                
                # 检查相似度阈值
                if score < self.similarity_threshold:
                    continue
                
                # 检查时间间隔约束
                if time_gap_constraint:
                    time_gap = abs(i - idx)
                    if time_gap < self.min_time_gap:
                        continue
                
                candidates.append({
                    "idx": int(idx),
                    "similarity": float(score)
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


def compare_methods(features: np.ndarray, metadata: List[Dict]):
    """对比原始方法和FAISS方法的性能"""
    import time
    
    n_windows = len(features)
    print(f"\n[性能对比] 数据规模: {n_windows} 个窗口")
    
    # 方法1：原始方法（计算所有对）
    print(f"\n方法1: 原始方法（计算所有对）")
    start_time = time.time()
    
    similarity_matrix = np.zeros((n_windows, n_windows))
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            f1_norm = features[i] / (np.linalg.norm(features[i]) + 1e-8)
            f2_norm = features[j] / (np.linalg.norm(features[j]) + 1e-8)
            similarity = np.dot(f1_norm, f2_norm)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    original_time = time.time() - start_time
    print(f"  耗时: {original_time:.4f} 秒")
    print(f"  复杂度: O(n^2) = O({n_windows}^2) = {n_windows**2} 次计算")
    
    # 方法2：FAISS方法
    print(f"\n方法2: FAISS方法（快速搜索）")
    start_time = time.time()
    
    matcher = FastSimilarityMatcher()
    matcher.build_index(features)
    pairs = matcher.find_similar_pairs(features, metadata)
    
    faiss_time = time.time() - start_time
    print(f"  耗时: {faiss_time:.4f} 秒")
    print(f"  复杂度: O(n log n) (近似)")
    
    # 性能提升
    speedup = original_time / faiss_time if faiss_time > 0 else float('inf')
    print(f"\n性能提升: {speedup:.2f}x 倍")
    
    if n_windows > 1000:
        print(f"\n[预测] 对于10,000个窗口:")
        print(f"  原始方法: 约 {original_time * (10000/n_windows)**2:.2f} 秒")
        print(f"  FAISS方法: 约 {faiss_time * (10000/n_windows) * np.log2(10000/n_windows):.2f} 秒")
        print(f"  预计提升: 约 {speedup * (10000/n_windows) / np.log2(10000/n_windows):.2f}x 倍")


def main():
    """主函数"""
    print("=" * 60)
    print("使用FAISS快速寻找相似的正样本对（优化版）")
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
    
    # 性能对比（可选，数据量小时）
    # if len(features) <= 100:
    #     compare_methods(features, metadata)
    
    # 创建快速相似度匹配器（使用精确搜索，不使用K-means）
    matcher = FastSimilarityMatcher(
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
    pairs_dir = output_dir / "similar_pairs_faiss"
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

