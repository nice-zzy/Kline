"""
可视化渲染的K线图片

功能：
1. 从npy文件加载图片
2. 显示单张或批量图片
3. 保存图片到文件
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# 添加项目路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent
project_root = training_dir.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(training_dir))


def visualize_single_image(image_array: np.ndarray, title: str = "", save_path: str = None):
    """
    可视化单张图片
    
    Args:
        image_array: 图片数组 (H, W, 3)
        title: 图片标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array)
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"  已保存: {save_path}")
    
    plt.show()


def visualize_image_pairs(
    anchor_images: np.ndarray,
    positive_images: np.ndarray,
    pairs_metadata: list,
    num_pairs: int = 5,
    save_dir: str = None
):
    """
    可视化相似对图片
    
    Args:
        anchor_images: Anchor图片数组
        positive_images: Positive图片数组
        pairs_metadata: 配对元数据
        num_pairs: 显示的对数
        save_dir: 保存目录（可选）
    """
    num_pairs = min(num_pairs, len(pairs_metadata))
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 5 * num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_pairs):
        pair = pairs_metadata[i]
        
        # Anchor图片
        axes[i, 0].imshow(anchor_images[i])
        axes[i, 0].set_title(
            f"Anchor {i+1}\n日期: {pair.get('anchor_date', 'N/A')}\n相似度: {pair.get('similarity', 0):.4f}",
            fontsize=10
        )
        axes[i, 0].axis('off')
        
        # Positive图片
        axes[i, 1].imshow(positive_images[i])
        axes[i, 1].set_title(
            f"Positive {i+1}\n日期: {pair.get('positive_date', 'N/A')}",
            fontsize=10
        )
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "similar_pairs_visualization.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  已保存: {save_path}")
    
    plt.show()


def save_images_to_files(
    images: np.ndarray,
    output_dir: str,
    prefix: str = "image",
    metadata: list = None
):
    """
    将numpy数组图片保存为PNG文件
    
    Args:
        images: 图片数组 (N, H, W, 3)
        output_dir: 输出目录
        prefix: 文件名前缀
        metadata: 元数据列表（可选，用于生成文件名）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[保存] 保存 {len(images)} 张图片到: {output_path}")
    
    for i, image in enumerate(images):
        if metadata and i < len(metadata):
            # 使用元数据生成文件名
            meta = metadata[i]
            if 'anchor_date' in meta:
                filename = f"{prefix}_{i:03d}_{meta['anchor_date']}.png"
            elif 'start_date' in meta:
                filename = f"{prefix}_{i:03d}_{meta['start_date']}.png"
            else:
                filename = f"{prefix}_{i:03d}.png"
        else:
            filename = f"{prefix}_{i:03d}.png"
        
        filepath = output_path / filename
        
        # 转换为PIL Image并保存
        pil_image = Image.fromarray(image)
        pil_image.save(filepath)
        
        if (i + 1) % 10 == 0:
            print(f"  已保存 {i+1}/{len(images)} 张...")
    
    print(f"  完成！共保存 {len(images)} 张图片")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化K线图片")
    
    parser.add_argument("--images-file", type=str,
                       default="services/training/output/aapl_2012_jan_jun/all_window_images.npy",
                       help="图片npy文件路径")
    parser.add_argument("--anchor-images", type=str,
                       help="Anchor图片npy文件路径")
    parser.add_argument("--positive-images", type=str,
                       help="Positive图片npy文件路径")
    parser.add_argument("--pairs-metadata", type=str,
                       help="配对元数据JSON文件路径")
    parser.add_argument("--num-show", type=int, default=5,
                       help="显示的数量")
    parser.add_argument("--save-images", action="store_true",
                       help="保存图片为PNG文件")
    parser.add_argument("--output-dir", type=str,
                       default="services/training/output/aapl_2012_jan_jun/visualized_images",
                       help="图片保存目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("K线图片可视化工具")
    print("=" * 60)
    
    # 如果有配对数据，显示配对
    if args.anchor_images and args.positive_images and args.pairs_metadata:
        print(f"\n[加载] 加载相似对图片...")
        anchor_images = np.load(args.anchor_images)
        positive_images = np.load(args.positive_images)
        
        with open(args.pairs_metadata, 'r', encoding='utf-8') as f:
            pairs_metadata = json.load(f)
        
        print(f"      Anchor图片: {anchor_images.shape}")
        print(f"      Positive图片: {positive_images.shape}")
        print(f"      配对数量: {len(pairs_metadata)}")
        
        # 可视化配对
        print(f"\n[显示] 显示前 {args.num_show} 对相似图片...")
        visualize_image_pairs(
            anchor_images,
            positive_images,
            pairs_metadata,
            num_pairs=args.num_show,
            save_dir=args.output_dir if args.save_images else None
        )
        
        # 保存图片
        if args.save_images:
            save_dir = Path(args.output_dir)
            save_images_to_files(
                anchor_images,
                str(save_dir / "anchors"),
                prefix="anchor",
                metadata=pairs_metadata
            )
            save_images_to_files(
                positive_images,
                str(save_dir / "positives"),
                prefix="positive",
                metadata=pairs_metadata
            )
    
    # 显示所有窗口图片
    elif args.images_file:
        print(f"\n[加载] 加载所有窗口图片...")
        images = np.load(args.images_file)
        print(f"      图片形状: {images.shape}")
        print(f"      图片数量: {len(images)}")
        
        # 显示前几张
        print(f"\n[显示] 显示前 {args.num_show} 张图片...")
        for i in range(min(args.num_show, len(images))):
            visualize_single_image(
                images[i],
                title=f"Window {i+1}",
                save_path=str(Path(args.output_dir) / f"window_{i+1}.png") if args.save_images else None
            )
        
        # 保存所有图片
        if args.save_images:
            save_images_to_files(
                images,
                args.output_dir,
                prefix="window"
            )
    
    else:
        print("[错误] 请指定图片文件路径")
        parser.print_help()


if __name__ == "__main__":
    main()

