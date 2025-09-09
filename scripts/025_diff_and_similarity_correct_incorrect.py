import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# 配置
DATA_DIRS = {
    "en": Path("data/clean_causal_data_v2"),
    "zh": Path("data/clean_causal_data_v2"),
    "enpara": Path("data/paraphrased_causal_data_v2"),
    "zhpara": Path("data/paraphrased_causal_data_v2")
}

OUTPUT_DIRS = {
    "en": Path("outputs/qwen15_18b_00_v2"),
    "zh": Path("outputs/qwen15_18b_00_v2"),
    "enpara": Path("outputs/qwen15_18b_para_v2_00"),
    "zhpara": Path("outputs/qwen15_18b_para_v2_00")
}

SAVE_DIR = Path("outputs/period_diff")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# period前token定位
def find_period_position(tokens: List[str], is_chinese: bool = False) -> int:
    period = '。\n' if is_chinese else '.Ċ'
    for i, token in enumerate(tokens):
        if token == period:
            return i - 1
    return -1

# 主处理函数
def collect_hidden_vectors(theme_name: str, version: str, is_chinese: bool):
    pt_path = OUTPUT_DIRS[version] / f"{theme_name}.pt"
    if not pt_path.exists():
        return [], []
    data = torch.load(pt_path)
    correct_vecs = []
    wrong_vecs = []
    for sample in data:
        tokens = sample["tokens_zh"] if is_chinese else sample["tokens_en"]
        pos = find_period_position(tokens, is_chinese)
        if pos == -1:
            continue
        hidden_states = sample["hidden_states_zh"] if is_chinese else sample["hidden_states_en"]
        # [L, 1, seq, hidden] -> [L, hidden]
        vecs = [h[0, pos].numpy() for h in hidden_states]
        if (sample["is_correct_zh"] if is_chinese else sample["is_correct_en"]):
            correct_vecs.append(vecs)
        else:
            wrong_vecs.append(vecs)
    return correct_vecs, wrong_vecs

def main():
    theme_names = [f.stem for f in DATA_DIRS["en"].glob("*.jsonl")]
    structures = ["en", "zh", "enpara", "zhpara"]
    is_chinese_map = {"en": False, "zh": True, "enpara": False, "zhpara": True}
    layer_num = None
    diff_dict = {}
    mean_diff = {}
    sim_dict = {}
    mean_sim = {}
    for struct in structures:
        print(f"处理结构: {struct}")
        all_correct = []
        all_wrong = []
        for theme in theme_names:
            correct_vecs, wrong_vecs = collect_hidden_vectors(theme, struct, is_chinese_map[struct])
            all_correct.extend(correct_vecs)
            all_wrong.extend(wrong_vecs)
        if not all_correct or not all_wrong:
            print(f"  跳过 {struct}，因无正确或错误样本")
            continue
        all_correct = np.array(all_correct)  # [N1, L, H]
        all_wrong = np.array(all_wrong)      # [N2, L, H]
        if layer_num is None:
            layer_num = all_correct.shape[1]
        # 求均值
        mean_correct = np.mean(all_correct, axis=0)  # [L, H]
        mean_wrong = np.mean(all_wrong, axis=0)      # [L, H]
        # 每层L2差值
        diff = np.linalg.norm(mean_correct - mean_wrong, axis=1)  # [L]
        diff_dict[struct] = diff
        mean_diff[struct] = np.mean(diff)
        # 每层cosine similarity
        sim = [cosine_similarity(mean_correct[i].reshape(1, -1), mean_wrong[i].reshape(1, -1))[0,0] for i in range(layer_num)]
        sim_dict[struct] = sim
        mean_sim[struct] = np.mean(sim)
        print(f"  {struct} 每层均值差: {diff}")
        print(f"  {struct} 所有层均值差: {mean_diff[struct]:.6f}")
        print(f"  {struct} 每层cosine相似度: {sim}")
        print(f"  {struct} 所有层cosine均值: {mean_sim[struct]:.6f}")
    # 画差值曲线图
    plt.figure(figsize=(10,6))
    for struct, color, label in zip(["en", "zh", "enpara", "zhpara"],
                                    ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"],
                                    ["EN", "ZH", "ENPARA", "ZHPARA"]):
        if struct in diff_dict:
            plt.plot(np.arange(1, layer_num+1), diff_dict[struct], label=label, color=color, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("L2 Difference (Correct - Incorrect)")
    plt.title("Layerwise L2 Difference between Correct and Incorrect (Period Position)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "diff_curve.png", dpi=300)
    plt.close()
    print(f"保存曲线图到: {SAVE_DIR / 'diff_curve.png'}")
    # 画差值bar图
    plt.figure(figsize=(7,5))
    bar_x = ["EN", "ZH", "ENPARA", "ZHPARA"]
    bar_y = [mean_diff[s] if s in mean_diff else 0 for s in ["en", "zh", "enpara", "zhpara"]]
    plt.bar(bar_x, bar_y, color=["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"])
    plt.ylabel("Mean L2 Difference (All Layers)")
    plt.title("Mean L2 Difference between Correct and Incorrect (Period Position)")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "diff_bar.png", dpi=300)
    plt.close()
    print(f"保存柱状图到: {SAVE_DIR / 'diff_bar.png'}")
    # 画相似度曲线图
    plt.figure(figsize=(10,6))
    for struct, color, label in zip(["en", "zh", "enpara", "zhpara"],
                                    ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"],
                                    ["EN", "ZH", "ENPARA", "ZHPARA"]):
        if struct in sim_dict:
            plt.plot(np.arange(1, layer_num+1), sim_dict[struct], label=label, color=color, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity (Correct vs Incorrect)")
    plt.title("Layerwise Cosine Similarity between Correct and Incorrect (Period Position)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "sim_curve.png", dpi=300)
    plt.close()
    print(f"保存相似度曲线图到: {SAVE_DIR / 'sim_curve.png'}")
    # 画相似度bar图
    plt.figure(figsize=(7,5))
    bar_y_sim = [mean_sim[s] if s in mean_sim else 0 for s in ["en", "zh", "enpara", "zhpara"]]
    plt.bar(bar_x, bar_y_sim, color=["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"])
    plt.ylabel("Mean Cosine Similarity (All Layers)")
    plt.title("Mean Cosine Similarity between Correct and Incorrect (Period Position)")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "sim_bar.png", dpi=300)
    plt.close()
    print(f"保存相似度柱状图到: {SAVE_DIR / 'sim_bar.png'}")

if __name__ == "__main__":
    main() 