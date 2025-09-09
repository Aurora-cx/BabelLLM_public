import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("outputs/period_similarity/all_themes_avg_all.json")
SAVE_PATH = Path("outputs/period_similarity/similarity_curves_all.png")

# 读取数据
def load_similarity_data(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def plot_similarity_curves(data, save_path):
    layer_num = len(data["en_zh"])
    x = np.arange(1, layer_num + 1)
    plt.figure(figsize=(10, 6))
    # 绘制四组对比
    plt.plot(x, data["en_zh"], label="EN vs ZH", color="#2ecc71", marker="o")
    plt.plot(x, data["en_enpara"], label="EN vs ENPARA", color="#3498db", marker="s")
    plt.plot(x, data["zh_zhpara"], label="ZH vs ZHPARA", color="#e74c3c", marker="^")
    plt.plot(x, data["enpara_zhpara"], label="ENPARA vs ZHPARA", color="#f1c40f", marker="d")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Period Position Hidden State Similarity (All Samples)")
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存图片到: {save_path}")

def main():
    data = load_similarity_data(INPUT_PATH)
    plot_similarity_curves(data, SAVE_PATH)

if __name__ == "__main__":
    main() 