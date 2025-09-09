import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置
INPUT_DIR = Path("outputs/similarity_analysis")
OUTPUT_DIR = Path("outputs/similarity_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_similarity_curves():
    """绘制相似度曲线"""
    # 读取数据
    with open(INPUT_DIR / "both_correct" / "all_themes_avg.json", "r", encoding="utf-8") as f:
        both_correct_data = json.load(f)
    with open(INPUT_DIR / "one_correct" / "all_themes_avg.json", "r", encoding="utf-8") as f:
        one_correct_data = json.load(f)
    
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Hidden Vector Similarity Across Layers', fontsize=16)
    
    # 获取层数
    num_layers = len(both_correct_data["first_comma"])
    layers = np.arange(num_layers)
    
    # 绘制每个位置的曲线
    positions = {
        "first_comma": "Before First Comma",
        "second_comma": "Before Second Comma",
        "first_period": "Before First Period"
    }
    
    for idx, (pos, title) in enumerate(positions.items()):
        ax = axes[idx]
        
        # 绘制both_correct曲线
        ax.plot(layers, both_correct_data[pos], 
                label=f'Both Correct (n={both_correct_data["sample_count"]})',
                color='blue', marker='o', markersize=4)
        
        # 绘制one_correct曲线
        ax.plot(layers, one_correct_data[pos], 
                label=f'One Correct (n={one_correct_data["sample_count"]})',
                color='red', marker='o', markersize=4)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # 设置y轴范围，使曲线更清晰
        ax.set_ylim(0, 1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = OUTPUT_DIR / "similarity_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存图片到: {output_path}")
    
    # 显示图片
    plt.show()

if __name__ == "__main__":
    plot_similarity_curves() 