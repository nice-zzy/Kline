"""
OHLC特征提取器：单日8维特征 + 窗口级趋势特征

设计理念：
1. 单日特征（8维）：只描述单日K线形态，不包含窗口级信息
2. 窗口级特征（12维）：单独提取，描述整个5天窗口的整体趋势模式

最终特征向量：
- 单日序列：5天 × 8维 = 40维（保留时间序列信息）
- 窗口统计：1组 × 12维 = 12维（整体趋势特征）
- 总计：52维

改进说明：
- ✅ 新增4个重要特征：R2_trend, ATR_window, Mean_body_pct, Std_body_pct
- ⚠️ 改进1个特征：Volatility（从价格标准差改为收益率标准差）
- 📊 所有特征都已归一化，可直接用于机器学习模型

基于伪代码的8维特征（单日）：
1. Onorm - 归一化开盘价
2. Hnorm - 归一化最高价  
3. Lnorm - 归一化最低价
4. Cnorm - 归一化收盘价
5. Body - 实体大小
6. UpperShadow - 上影线长度
7. LowerShadow - 下影线长度
8. CandleType - 蜡烛类型（1涨/-1跌）

窗口级特征（整体趋势，12维）：
1. Return_total - 窗口累积涨跌幅（5天整体涨跌）
2. Slope_linreg - 回归斜率（基于收盘价的线性回归斜率，归一化）
3. R2_trend - 趋势线性强度（衡量价格变化是否符合线性趋势）
4. Volatility - 收益率波动率（窗口内收益率的标准差，改进版）
5. ATR_window - 真实波动幅度（考虑跳空和影线，归一化）
6. Mean_body_pct - 实体平均（反映窗口情绪强度，归一化）
7. Std_body_pct - 实体波动（反映窗口内实体大小的波动程度，归一化）
8. 窗口内涨跌天数比例 - 多空力量对比
9. 窗口价格位置 - 当前价格在窗口中的位置
10. 窗口形态一致性 - 5天形态的相似度
11. 窗口成交量趋势 - 成交量变化趋势
12. 窗口趋势方向 - 整体上涨/下跌趋势
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

# 添加项目路径以便导入
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class OHLCFeatureExtractor:
    """
    OHLC特征提取器
    
    实现伪代码中的8维特征提取，并扩展为窗口级趋势特征
    """
    
    def __init__(self, window_size: int = 5, epsilon: float = 1e-8):
        """
        初始化特征提取器
        
        Args:
            window_size: 窗口大小（天数）
            epsilon: 防止除零的小常数
        """
        self.window_size = window_size
        self.epsilon = epsilon
    
    def extract_8d_features(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        """
        提取8维基础特征（按伪代码实现）
        
        对窗口内的每一天提取8维特征，返回 shape=(window_size, 8) 的数组
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close']
        
        Returns:
            features: shape=(window_size, 8) 的特征数组
        """
        if len(ohlc_data) == 0:
            raise ValueError("OHLC data is empty")
        
        # 提取OHLC列
        opens = ohlc_data['open'].values
        highs = ohlc_data['high'].values
        lows = ohlc_data['low'].values
        closes = ohlc_data['close'].values
        
        # Step 1: 计算窗口归一化参数（伪代码第14-16行）
        Hmax = np.max(highs)  # 窗口内最高价
        Lmin = np.min(lows)   # 窗口内最低价
        R = Hmax - Lmin       # 窗口价格范围（归一化分母）
        
        # 防止除零
        if R < self.epsilon:
            R = self.epsilon
        
        # Step 2: 对每一天提取8维特征
        features_list = []
        
        for i in range(len(ohlc_data)):
            # Step 1: 归一化原始OHLC（伪代码第20-24行）
            Onorm = (opens[i] - Lmin) / R
            Hnorm = (highs[i] - Lmin) / R
            Lnorm = (lows[i] - Lmin) / R
            Cnorm = (closes[i] - Lmin) / R
            
            # Step 2: 计算蜡烛图几何属性（伪代码第26-30行）
            Body = abs(Cnorm - Onorm)  # 实体大小
            UpperShadow = Hnorm - max(Cnorm, Onorm)  # 上影线
            LowerShadow = min(Cnorm, Onorm) - Lnorm  # 下影线
            CandleType = 1.0 if Cnorm > Onorm else -1.0  # 蜡烛类型（涨/跌）
            
            # Step 3: 打包为8维特征向量（伪代码第32-42行）
            day_features = np.array([
                Onorm,      # 0: 归一化开盘价
                Hnorm,      # 1: 归一化最高价
                Lnorm,      # 2: 归一化最低价
                Cnorm,      # 3: 归一化收盘价
                Body,       # 4: 实体大小
                UpperShadow, # 5: 上影线长度
                LowerShadow, # 6: 下影线长度
                CandleType   # 7: 蜡烛类型（1涨/-1跌）
            ])
            
            features_list.append(day_features)
        
        # 返回 shape=(window_size, 8) 的特征矩阵
        return np.array(features_list)
    
    def extract_window_level_features(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        """
        提取窗口级趋势特征（整个窗口的统计特征）
        
        这些特征描述整个5天窗口的整体趋势模式，而不是单日特征
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] (volume可选)
        
        Returns:
            window_features: shape=(12,) 的窗口级特征向量
        """
        if len(ohlc_data) == 0:
            raise ValueError("OHLC data is empty")
        
        # 提取OHLC数据
        opens = ohlc_data['open'].values
        highs = ohlc_data['high'].values
        lows = ohlc_data['low'].values
        closes = ohlc_data['close'].values
        volumes = ohlc_data['volume'].values if 'volume' in ohlc_data.columns else None
        
        # 窗口归一化参数
        Hmax = np.max(highs)
        Lmin = np.min(lows)
        R = max(Hmax - Lmin, self.epsilon)
        
        # 提取8维基础特征（用于计算窗口统计）
        base_features = self.extract_8d_features(ohlc_data)  # shape=(window_size, 8)
        
        # === 窗口级特征计算 ===
        
        # 1. 窗口趋势方向（整体上涨/下跌趋势）
        # 计算从窗口起点到终点的价格变化（归一化）
        window_trend_direction = (closes[-1] - opens[0]) / R
        # 值域: [-1, 1]，正值=上涨，负值=下跌
        
        # 2. 窗口累积涨跌幅（窗口内累计收益率）
        window_cumulative_return = (closes[-1] - closes[0]) / closes[0]
        # 值域: 无界，通常[-0.5, 0.5]
        
        # 3. 窗口波动率（收益率波动程度）- 改进版
        # 使用收益率的标准差（更符合金融学意义）
        if len(closes) >= 2:
            returns = np.diff(closes) / closes[:-1]  # 计算收益率
            window_volatility = np.std(returns)  # 收益率的标准差
        else:
            window_volatility = 0.0
        # 值域: [0, +∞)，通常[0, 0.05]，已归一化（基于收益率）
        
        # 4. 窗口形态一致性（5天形态的相似度）
        # 计算所有天的Body、UpperShadow、LowerShadow的变异系数
        bodies = base_features[:, 4]
        upper_shadows = base_features[:, 5]
        lower_shadows = base_features[:, 6]
        
        # 变异系数越小，一致性越高
        body_cv = np.std(bodies) / (np.mean(bodies) + self.epsilon)
        upper_cv = np.std(upper_shadows) / (np.mean(upper_shadows) + self.epsilon)
        lower_cv = np.std(lower_shadows) / (np.mean(lower_shadows) + self.epsilon)
        
        # 一致性 = 1 - 平均变异系数（归一化到[0,1]）
        avg_cv = (body_cv + upper_cv + lower_cv) / 3.0
        pattern_consistency = 1.0 / (1.0 + avg_cv)  # 使用倒数函数归一化
        # 值域: [0, 1]，1表示完全一致，0表示完全不同
        
        # 5. 窗口成交量强度（平均成交量相对强度）
        if volumes is not None:
            # 这里使用平均成交量，因为这是窗口级特征
            avg_volume = np.mean(volumes)
            # 可以计算成交量趋势（如果有多天数据）
            if len(volumes) >= 2:
                volume_trend = np.corrcoef(np.arange(len(volumes)), volumes)[0, 1]
                # 值域: [-1, 1]，正值=成交量上升，负值=成交量下降
            else:
                volume_trend = 0.0
        else:
            avg_volume = 1.0
            volume_trend = 0.0
        
        # 6. 窗口趋势强度（趋势的明显程度）- Slope_linreg
        # 使用线性回归的斜率来衡量趋势强度
        if len(closes) >= 2:
            x = np.arange(len(closes))
            # 线性回归斜率（归一化）
            trend_slope = np.polyfit(x, closes, 1)[0] / (np.mean(closes) + self.epsilon)
            # 值域: 无界，通常[-0.1, 0.1]
        else:
            trend_slope = 0.0
        
        # 7. R2_trend - 趋势线性强度（新增）
        # 计算线性回归的R²值，衡量价格变化是否符合线性趋势
        if len(closes) >= 2:
            x = np.arange(len(closes))
            # 线性回归
            coeffs = np.polyfit(x, closes, 1)
            linear_pred = np.polyval(coeffs, x)
            # 计算R²
            ss_res = np.sum((closes - linear_pred) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            if ss_tot > self.epsilon:
                r2_trend = 1.0 - (ss_res / ss_tot)
            else:
                r2_trend = 0.0
            # 值域: [0, 1]，1=强线性趋势，0=非线性趋势
        else:
            r2_trend = 0.0
        
        # 8. ATR_window - 真实波动幅度（新增）
        # 考虑跳空和影线的真实波动幅度，对不同市场更稳定
        if len(ohlc_data) >= 2:
            true_ranges = []
            for i in range(1, len(ohlc_data)):
                # True Range = max(H-L, abs(H-C_prev), abs(L-C_prev))
                high_low = highs[i] - lows[i]
                high_close_prev = abs(highs[i] - closes[i-1])
                low_close_prev = abs(lows[i] - closes[i-1])
                true_range = max(high_low, high_close_prev, low_close_prev)
                true_ranges.append(true_range)
            
            if len(true_ranges) > 0:
                atr_window = np.mean(true_ranges)
                # 归一化：除以窗口价格范围
                atr_window_norm = atr_window / R
            else:
                atr_window_norm = 0.0
            # 值域: [0, +∞)，通常[0, 1]，已归一化
        else:
            atr_window_norm = 0.0
        
        # 9. Mean_body_pct - 实体平均（新增）
        # 反映窗口是否"情绪强烈"（实体大=波动大）
        bodies = base_features[:, 4]  # Body特征（已归一化）
        mean_body_pct = np.mean(bodies)
        # 值域: [0, 1]，已归一化（Body本身就是归一化的）
        
        # 10. Std_body_pct - 实体波动（新增）
        # 反映窗口内实体大小的波动程度（避免离群值影响）
        std_body_pct = np.std(bodies)
        # 值域: [0, +∞)，通常[0, 0.5]，已归一化（Body本身就是归一化的）
        
        # 11. 窗口价格位置（当前价格在窗口中的位置）
        # 使用最后一天的收盘价位置
        final_price_position = (closes[-1] - Lmin) / R
        # 值域: [0, 1]，0=窗口最低，1=窗口最高
        
        # 12. 窗口内涨跌天数比例（多空力量对比）
        # 计算窗口内上涨天数和下跌天数的比例
        up_days = np.sum(closes > opens)  # 收盘价 > 开盘价的天数
        down_days = np.sum(closes < opens)  # 收盘价 < 开盘价的天数
        total_days = len(closes)
        
        # 涨跌天数比例：正值表示上涨天数多，负值表示下跌天数多
        # 归一化到[-1, 1]：1表示全部上涨，-1表示全部下跌，0表示涨跌平衡
        if total_days > 0:
            up_down_ratio = (up_days - down_days) / total_days
        else:
            up_down_ratio = 0.0
        # 值域: [-1, 1]，1=全部上涨，-1=全部下跌，0=涨跌平衡
        
        # 组合窗口级特征（12维）
        window_features = np.array([
            window_cumulative_return,    # 0: Return_total - 窗口累积涨跌幅
            trend_slope,                 # 1: Slope_linreg - 回归斜率（归一化）
            r2_trend,                    # 2: R2_trend - 趋势线性强度（新增）
            window_volatility,           # 3: Volatility - 收益率标准差（改进）
            atr_window_norm,             # 4: ATR_window - 真实波动幅度（新增，归一化）
            mean_body_pct,               # 5: Mean_body_pct - 实体平均（新增，归一化）
            std_body_pct,                # 6: Std_body_pct - 实体波动（新增，归一化）
            up_down_ratio,               # 7: 窗口内涨跌天数比例
            final_price_position,        # 8: 窗口价格位置
            pattern_consistency,         # 9: 窗口形态一致性
            volume_trend,                # 10: 窗口成交量趋势
            window_trend_direction       # 11: 窗口趋势方向
        ])
        
        return window_features  # shape=(12,)
    
    def extract_concatenated_features(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        """
        提取并拼接所有特征（用于模型输入）
        
        特征组织方式：
        - 单日序列：5天 × 8维 = 40维（保留时间序列信息）
        - 窗口统计：1组 × 12维 = 12维（整体趋势特征）
        - 总计：52维
        
        Args:
            ohlc_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            concatenated_features: shape=(window_size * 8 + 12,) 的一维特征向量
            前40维是单日序列特征，后12维是窗口级特征
        """
        # 提取单日8维特征序列
        daily_features = self.extract_8d_features(ohlc_data)  # shape=(window_size, 8)
        
        # 提取窗口级特征
        window_features = self.extract_window_level_features(ohlc_data)  # shape=(7,)
        
        # 拼接：先展平日序列，再拼接窗口特征
        daily_flattened = daily_features.flatten()  # shape=(window_size * 8,)
        concatenated = np.concatenate([daily_flattened, window_features])  # shape=(window_size * 8 + 7,)
        
        return concatenated
    
    def explain_features(self) -> Dict[str, List[str]]:
        """
        解释特征含义
        
        Returns:
            特征说明字典
        """
        return {
            "单日8维特征（描述单日K线形态）": [
                "Onorm (0): 归一化开盘价 - 开盘价在窗口价格范围中的位置 [0,1]",
                "Hnorm (1): 归一化最高价 - 最高价在窗口价格范围中的位置 [0,1]",
                "Lnorm (2): 归一化最低价 - 最低价在窗口价格范围中的位置 [0,1]",
                "Cnorm (3): 归一化收盘价 - 收盘价在窗口价格范围中的位置 [0,1]",
                "Body (4): 实体大小 - 开盘价与收盘价的绝对差值（归一化后）[0,1]",
                "UpperShadow (5): 上影线长度 - 最高价与实体上沿的差值 [0,1]",
                "LowerShadow (6): 下影线长度 - 实体下沿与最低价的差值 [0,1]",
                "CandleType (7): 蜡烛类型 - 1.0表示上涨（收盘>开盘），-1.0表示下跌"
            ],
            "窗口级12维特征（描述整个5天窗口的整体趋势）": [
                "Return_total (0): 窗口累积涨跌幅，5天整体涨跌 [无界]，已归一化（收益率）",
                "Slope_linreg (1): 回归斜率，基于收盘价的线性回归斜率 [无界]，已归一化",
                "R2_trend (2): 趋势线性强度，衡量价格变化是否符合线性趋势 [0,1]，1=强线性，0=非线性",
                "Volatility (3): 收益率波动率，窗口内收益率的标准差 [0,+∞)，已归一化（基于收益率）",
                "ATR_window (4): 真实波动幅度，考虑跳空和影线的真实波动 [0,+∞)，已归一化到窗口价格范围",
                "Mean_body_pct (5): 实体平均，反映窗口情绪强度 [0,1]，已归一化",
                "Std_body_pct (6): 实体波动，反映窗口内实体大小的波动程度 [0,+∞)，已归一化",
                "窗口内涨跌天数比例 (7): 多空力量对比 [-1,1]，1=全部上涨，-1=全部下跌，0=涨跌平衡",
                "窗口价格位置 (8): 最后一天收盘价在窗口价格范围中的位置 [0,1]",
                "窗口形态一致性 (9): 5天形态的相似度 [0,1]，1=完全一致，0=完全不同",
                "窗口成交量趋势 (10): 成交量变化趋势 [-1,1]，正值=成交量上升，负值=成交量下降",
                "窗口趋势方向 (11): 从窗口起点到终点的价格变化方向 [-1,1]，正值=上涨，负值=下跌"
            ],
            "特征组织方式": [
                "单日序列：5天 × 8维 = 40维（保留时间序列信息，描述每天形态）",
                "窗口统计：1组 × 12维 = 12维（整体趋势特征，描述窗口模式）",
                "总计：52维特征向量",
                "所有特征已归一化，可直接用于机器学习模型"
            ],
            "设计优势": [
                "单日特征只描述单日形态，不包含窗口级信息，更清晰",
                "窗口级特征单独提取，避免重复计算，更高效",
                "特征维度合理，不会过度复杂",
                "保留时间序列信息，同时包含整体趋势信息"
            ]
        }


def test_feature_extractor():
    """测试特征提取器"""
    print("=" * 60)
    print("测试 OHLC 特征提取器")
    print("=" * 60)
    
    # 创建测试数据（5天窗口）
    np.random.seed(42)
    n_days = 5
    base_price = 100.0
    
    # 生成模拟OHLC数据
    data = []
    current_price = base_price
    
    for i in range(n_days):
        # 模拟价格波动
        change = np.random.normal(0, 0.02)
        close = current_price * (1 + change)
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = current_price
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000000, 5000000)
        })
        
        current_price = close
    
    ohlc_df = pd.DataFrame(data)
    
    print("\n📊 测试数据（5天窗口）:")
    print(ohlc_df)
    
    # 创建特征提取器
    extractor = OHLCFeatureExtractor(window_size=5)
    
    # 提取8维基础特征
    print("\n" + "=" * 60)
    print("1. 提取8维基础特征")
    print("=" * 60)
    base_features = extractor.extract_8d_features(ohlc_df)
    print(f"特征形状: {base_features.shape} (天数 × 8维)")
    print("\n每天的特征:")
    for i, day_features in enumerate(base_features):
        print(f"  第{i+1}天: {day_features}")
    
    # 提取窗口级特征
    print("\n" + "=" * 60)
    print("2. 提取窗口级特征（整个窗口的整体趋势）")
    print("=" * 60)
    window_features = extractor.extract_window_level_features(ohlc_df)
    print(f"特征形状: {window_features.shape} (12维窗口级特征)")
    print("\n窗口级特征:")
    feature_names = [
        "Return_total (窗口累积涨跌幅)",
        "Slope_linreg (回归斜率)",
        "R2_trend (趋势线性强度)",
        "Volatility (收益率波动率)",
        "ATR_window (真实波动幅度)",
        "Mean_body_pct (实体平均)",
        "Std_body_pct (实体波动)",
        "窗口内涨跌天数比例",
        "窗口价格位置",
        "窗口形态一致性",
        "窗口成交量趋势",
        "窗口趋势方向"
    ]
    for i, (name, value) in enumerate(zip(feature_names, window_features)):
        print(f"  {name} ({i}): {value:.4f}")
    
    # 提取拼接特征
    print("\n" + "=" * 60)
    print("3. 提取拼接特征（用于模型输入）")
    print("=" * 60)
    concatenated_features = extractor.extract_concatenated_features(ohlc_df)
    print(f"特征形状: {concatenated_features.shape} (52维 = 5天×8维 + 12维窗口级)")
    print(f"前40维（单日序列）: {concatenated_features[:8]}... (显示第1天的8维)")
    print(f"后12维（窗口级）: {concatenated_features[40:]}")
    
    # 特征说明
    print("\n" + "=" * 60)
    print("4. 特征含义说明")
    print("=" * 60)
    explanations = extractor.explain_features()
    for category, items in explanations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_extractor()

