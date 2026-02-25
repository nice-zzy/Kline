"""
相似对图片渲染工具

功能：
1. 从已渲染的NPY数组中提取相似对图片
2. 保存为PNG格式（单个图片和对比图）
3. 可以独立运行，用于查看相似对效果

使用方法：
    python render_similar_pairs.py \
        --anchor_images anchor_images.npy \
        --positive_images positive_images.npy \
        --pairs_metadata pairs_metadata.json \
        --output_dir ./output/pair_images
"""
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
data_dir = training_dir / "data"
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))
sys.path.insert(0, str(data_dir))


def render_similar_pairs_png(
    anchor_images_file: str,
    positive_images_file: str,
    pairs_metadata_file: str,
    output_dir: str,
    save_individual: bool = True,
    save_comparison: bool = True,
    pairs_per_page: int = 5,
    show_axes: bool = True,
    color_scheme: str = "chinese"
) -> bool:
    """
    渲染相似对图片为PNG格式
    
    Args:
        anchor_images_file: Anchor图片NPY文件路径
        positive_images_file: Positive图片NPY文件路径
        pairs_metadata_file: 相似对元数据JSON文件路径
        output_dir: 输出目录
        save_individual: 是否保存单个图片
        save_comparison: 是否保存对比图
        pairs_per_page: 每页对比图显示的对数
        show_axes: 是否显示坐标轴（仅用于对比图）
        color_scheme: 颜色方案 "chinese" 或 "western"
    
    Returns:
        是否成功
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from PIL import Image
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 检查输入文件
        anchor_path = Path(anchor_images_file)
        positive_path = Path(positive_images_file)
        pairs_meta_path = Path(pairs_metadata_file)
        
        if not anchor_path.exists():
            print(f"[错误] Anchor图片文件不存在: {anchor_path}")
            return False
        
        if not positive_path.exists():
            print(f"[错误] Positive图片文件不存在: {positive_path}")
            return False
        
        if not pairs_meta_path.exists():
            print(f"[错误] 相似对元数据文件不存在: {pairs_meta_path}")
            return False
        
        # 加载数据
        print(f"\n[加载] 加载相似对图片...")
        anchor_images = np.load(anchor_path)  # shape: (n_pairs, H, W, 3)
        positive_images = np.load(positive_path)  # shape: (n_pairs, H, W, 3)
        
        with open(pairs_meta_path, 'r', encoding='utf-8') as f:
            pair_metadata = json.load(f)
        
        n_pairs = len(pair_metadata)
        print(f"      找到 {n_pairs} 对相似图片")
        print(f"      Anchor图片形状: {anchor_images.shape}")
        print(f"      Positive图片形状: {positive_images.shape}")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存单个图片
        if save_individual:
            print(f"\n[保存] 保存单个图片...")
            anchors_png_dir = output_path / "anchors"
            positives_png_dir = output_path / "positives"
            anchors_png_dir.mkdir(exist_ok=True)
            positives_png_dir.mkdir(exist_ok=True)
            
            for i, (anchor_img, positive_img, pair_meta) in enumerate(zip(
                anchor_images, positive_images, pair_metadata
            )):
                # 保存anchor图片
                anchor_pil = Image.fromarray(anchor_img)
                anchor_png_path = anchors_png_dir / f"anchor_{i:03d}_{pair_meta['anchor_date']}.png"
                anchor_pil.save(anchor_png_path)
                
                # 保存positive图片
                positive_pil = Image.fromarray(positive_img)
                positive_png_path = positives_png_dir / f"positive_{i:03d}_{pair_meta['positive_date']}.png"
                positive_pil.save(positive_png_path)
                
                if (i + 1) % 10 == 0:
                    print(f"      已保存 {i + 1}/{n_pairs} 对图片...")
            
            print(f"      [完成] 已保存 {n_pairs} 对图片到:")
            print(f"              {anchors_png_dir}")
            print(f"              {positives_png_dir}")
        
        # 保存对比图
        if save_comparison:
            print(f"\n[保存] 生成相似对对比图...")
            comparison_dir = output_path / "comparisons"
            comparison_dir.mkdir(exist_ok=True)
            
            # 计算页数
            num_pages = (n_pairs + pairs_per_page - 1) // pairs_per_page
            
            for page in range(num_pages):
                start_idx = page * pairs_per_page
                end_idx = min(start_idx + pairs_per_page, n_pairs)
                page_pairs = end_idx - start_idx
                
                # 创建子图
                fig, axes = plt.subplots(page_pairs, 2, figsize=(10, 5 * page_pairs))
                if page_pairs == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(page_pairs):
                    idx = start_idx + i
                    pair = pair_metadata[idx]
                    
                    # Anchor图片
                    axes[i, 0].imshow(anchor_images[idx])
                    axes[i, 0].set_title(
                        f"Anchor {idx+1}\n日期: {pair['anchor_date']}\n相似度: {pair['similarity']:.4f}",
                        fontsize=10,
                        pad=5
                    )
                    axes[i, 0].axis('off')
                    
                    # Positive图片
                    axes[i, 1].imshow(positive_images[idx])
                    axes[i, 1].set_title(
                        f"Positive {idx+1}\n日期: {pair['positive_date']}",
                        fontsize=10,
                        pad=5
                    )
                    axes[i, 1].axis('off')
                
                # 调整布局
                plt.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.5)
                comparison_path = comparison_dir / f"similar_pairs_page_{page+1:02d}.png"
                plt.savefig(comparison_path, bbox_inches='tight', dpi=150, pad_inches=0.2, facecolor='white')
                plt.close()
                
                if (page + 1) % 5 == 0 or (page + 1) == num_pages:
                    print(f"      已生成 {page + 1}/{num_pages} 页对比图...")
            
            print(f"      [完成] 已保存 {num_pages} 页对比图到: {comparison_dir}")
        
        print(f"\n[完成] 相似对图片渲染完成！")
        print(f"      输出目录: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n[错误] 相似对图片渲染失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="相似对图片渲染工具")
    
    # 输入文件
    parser.add_argument(
        "--anchor_images",
        type=str,
        required=True,
        help="Anchor图片NPY文件路径"
    )
    
    parser.add_argument(
        "--positive_images",
        type=str,
        required=True,
        help="Positive图片NPY文件路径"
    )
    
    parser.add_argument(
        "--pairs_metadata",
        type=str,
        required=True,
        help="相似对元数据JSON文件路径"
    )
    
    # 输出目录
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/pair_images/png_images",
        help="输出目录（默认: ./output/pair_images/png_images）"
    )
    
    # 选项
    parser.add_argument(
        "--save_individual",
        action="store_true",
        default=True,
        help="保存单个图片（默认: True）"
    )
    
    parser.add_argument(
        "--no_save_individual",
        dest="save_individual",
        action="store_false",
        help="不保存单个图片"
    )
    
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        default=True,
        help="保存对比图（默认: True）"
    )
    
    parser.add_argument(
        "--no_save_comparison",
        dest="save_comparison",
        action="store_false",
        help="不保存对比图"
    )
    
    parser.add_argument(
        "--pairs_per_page",
        type=int,
        default=5,
        help="每页对比图显示的对数（默认: 5）"
    )
    
    args = parser.parse_args()
    
    # 执行渲染
    success = render_similar_pairs_png(
        anchor_images_file=args.anchor_images,
        positive_images_file=args.positive_images,
        pairs_metadata_file=args.pairs_metadata,
        output_dir=args.output_dir,
        save_individual=args.save_individual,
        save_comparison=args.save_comparison,
        pairs_per_page=args.pairs_per_page
    )
    
    if success:
        print("\n✅ 相似对图片渲染成功！")
        return 0
    else:
        print("\n❌ 相似对图片渲染失败！")
        return 1


if __name__ == "__main__":
    exit(main())

