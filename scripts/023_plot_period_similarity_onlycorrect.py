import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
INPUT_FILE = Path("outputs/period_similarity/all_themes_avg.json")
SAVE_DIR = Path("outputs/period_similarity")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def plot_similarity_curves():
    """绘制相似度曲线"""
    # 加载数据
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 准备数据
    layer_indices = np.arange(len(data["en_zh"]))  # 使用实际数据长度
    pairs = {
        "en_zh": "EN vs ZH",
        "en_enpara": "EN vs EN-Para",
        "en_zhpara": "EN vs ZH-Para",
        "zh_enpara": "ZH vs EN-Para",
        "zh_zhpara": "ZH vs ZH-Para",
        "enpara_zhpara": "EN-Para vs ZH-Para"
    }
    
    # 设置绘图风格
    plt.figure(figsize=(12, 6))
    
    # 设置颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 绘制每条曲线
    for (pair, label), color in zip(pairs.items(), colors):
        similarities = data[pair]
        plt.plot(layer_indices + 1, similarities, label=label, color=color, linewidth=2)  # +1 使层数从1开始
    
    # 设置图表属性
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Hidden State Similarity at Period Position Across Layers', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 设置坐标轴范围
    plt.xlim(1, len(data["en_zh"]))
    plt.ylim(0, 1)
    
    # 添加样本数量信息
    plt.figtext(0.02, 0.02, f'Total samples: {data["sample_count"]}', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(SAVE_DIR / 'similarity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图片已保存到: {SAVE_DIR / 'similarity_curves.png'}")

if __name__ == "__main__":
    plot_similarity_curves()