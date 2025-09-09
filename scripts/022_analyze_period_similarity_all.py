import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

SAVE_DIR = Path("outputs/period_similarity")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def find_period_position(tokens: List[str], is_chinese: bool = False) -> int:
    """找到第一个句号前的token位置"""
    period = '。\n' if is_chinese else '.Ċ'
    for i, token in enumerate(tokens):
        if token == period:
            return i - 1  # 返回句号前的token位置
    return -1  # 如果没找到句号，返回-1

def compute_layer_similarity(hidden_states1: List[torch.Tensor], 
                           hidden_states2: List[torch.Tensor],
                           pos1: int,
                           pos2: int) -> List[float]:
    """计算指定位置在每一层的隐藏向量相似度"""
    similarities = []
    for layer_idx in range(len(hidden_states1)):
        vec1 = hidden_states1[layer_idx][0, pos1].numpy().reshape(1, -1)
        vec2 = hidden_states2[layer_idx][0, pos2].numpy().reshape(1, -1)
        sim = cosine_similarity(vec1, vec2)[0][0]
        similarities.append(float(sim))
    return similarities

def process_theme_all(theme_name: str):
    """处理单个主题的全量（不管推理对错）样本数据"""
    print(f"\n[全量] 处理主题: {theme_name}")
    data = {}
    for version in ["en", "zh", "enpara", "zhpara"]:
        pt_path = OUTPUT_DIRS[version] / f"{theme_name}.pt"
        if not pt_path.exists():
            print(f"文件不存在: {pt_path}")
            return None
        data[version] = torch.load(pt_path)
    similarities = {
        "en_zh": [],
        "en_enpara": [],
        "zh_zhpara": [],
        "enpara_zhpara": []
    }
    for i in tqdm(range(len(data["en"])), desc="[全量] 处理样本"):
        positions = {
            "en": find_period_position(data["en"][i]["tokens_en"], is_chinese=False),
            "zh": find_period_position(data["zh"][i]["tokens_zh"], is_chinese=True),
            "enpara": find_period_position(data["enpara"][i]["tokens_en"], is_chinese=False),
            "zhpara": find_period_position(data["zhpara"][i]["tokens_zh"], is_chinese=True)
        }
        if -1 in positions.values():
            continue
        similarities["en_zh"].append(compute_layer_similarity(
            data["en"][i]["hidden_states_en"],
            data["zh"][i]["hidden_states_zh"],
            positions["en"],
            positions["zh"]
        ))
        similarities["en_enpara"].append(compute_layer_similarity(
            data["en"][i]["hidden_states_en"],
            data["enpara"][i]["hidden_states_en"],
            positions["en"],
            positions["enpara"]
        ))
        similarities["zh_zhpara"].append(compute_layer_similarity(
            data["zh"][i]["hidden_states_zh"],
            data["zhpara"][i]["hidden_states_zh"],
            positions["zh"],
            positions["zhpara"]
        ))
        similarities["enpara_zhpara"].append(compute_layer_similarity(
            data["enpara"][i]["hidden_states_en"],
            data["zhpara"][i]["hidden_states_zh"],
            positions["enpara"],
            positions["zhpara"]
        ))
    results = {}
    for key, values in similarities.items():
        if values:
            values_array = np.array(values)
            avg_values = np.mean(values_array, axis=0)
            results[key] = avg_values.tolist()
    results["sample_count"] = len(similarities["en_zh"])
    return results

def main():
    theme_names = [f.stem for f in DATA_DIRS["en"].glob("*.jsonl")]
    print(f"找到 {len(theme_names)} 个主题需要处理")
    all_results_all = []
    for theme_name in theme_names:
        results_all = process_theme_all(theme_name)
        if results_all:
            all_results_all.append(results_all)
    if all_results_all:
        all_similarities_all = {
            "en_zh": [],
            "en_enpara": [],
            "zh_zhpara": [],
            "enpara_zhpara": []
        }
        total_samples_all = 0
        for result in all_results_all:
            for key in all_similarities_all.keys():
                similarities = np.array(result[key]) * result["sample_count"]
                all_similarities_all[key].append(similarities)
            total_samples_all += result["sample_count"]
        avg_results_all = {
            key: (np.sum(values, axis=0) / total_samples_all).tolist()
            for key, values in all_similarities_all.items()
        }
        avg_results_all["sample_count"] = total_samples_all
        output_path_all = SAVE_DIR / "all_themes_avg_all.json"
        with open(output_path_all, "w", encoding="utf-8") as f:
            json.dump(avg_results_all, f, ensure_ascii=False, indent=2)
        print(f"\n[全量] 保存所有主题平均值到: {output_path_all}")
        print(f"[全量] 总样本数量: {avg_results_all['sample_count']}")
    print("\n✅ 全量样本 period 相似度分析完成！")

if __name__ == "__main__":
    main() 