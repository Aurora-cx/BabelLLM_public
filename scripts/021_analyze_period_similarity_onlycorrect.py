import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
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
    """找到第一个句号前的token位置
    
    Args:
        tokens: token列表
        is_chinese: 是否是中文token，决定使用中文还是英文标点
    """
    period = '。\n' if is_chinese else '.Ċ'
    
    for i, token in enumerate(tokens):
        if token == period:
            print(f"token: {tokens[i-1]}")
            return i - 1  # 返回句号前的token位置

    return -1  # 如果没找到句号，返回-1

def compute_layer_similarity(hidden_states1: List[torch.Tensor], 
                           hidden_states2: List[torch.Tensor],
                           pos1: int,
                           pos2: int) -> List[float]:
    """计算指定位置在每一层的隐藏向量相似度"""
    similarities = []
    for layer_idx in range(len(hidden_states1)):
        # 获取指定位置的隐藏向量
        vec1 = hidden_states1[layer_idx][0, pos1].numpy().reshape(1, -1)
        vec2 = hidden_states2[layer_idx][0, pos2].numpy().reshape(1, -1)
        
        # 计算余弦相似度
        sim = cosine_similarity(vec1, vec2)[0][0]
        similarities.append(float(sim))
    
    return similarities

def process_theme(theme_name: str):
    """处理单个主题的数据"""
    print(f"\n处理主题: {theme_name}")
    
    # 加载四个版本的数据
    data = {}
    for version in ["en", "zh", "enpara", "zhpara"]:
        pt_path = OUTPUT_DIRS[version] / f"{theme_name}.pt"
        if not pt_path.exists():
            print(f"文件不存在: {pt_path}")
            return None
        data[version] = torch.load(pt_path)
    
    # 初始化结果
    similarities = {
        "en_zh": [],
        "en_enpara": [],
        "en_zhpara": [],
        "zh_enpara": [],
        "zh_zhpara": [],
        "enpara_zhpara": []
    }
    
    # 处理每个样本
    for i in tqdm(range(len(data["en"])), desc="处理样本"):
        # 检查推理结果
        is_correct = {
            "en": data["en"][i]["is_correct_en"],
            "zh": data["zh"][i]["is_correct_zh"],
            "enpara": data["enpara"][i]["is_correct_en"],
            "zhpara": data["zhpara"][i]["is_correct_zh"]
        }
        
        # 只处理所有版本都正确的样本
        if not all(is_correct.values()):
            continue

        print(f"tokens_en: {data['en'][i]['tokens_en']}")
        print(f"tokens_zh: {data['zh'][i]['tokens_zh']}")
        print(f"tokens_enpara: {data['enpara'][i]['tokens_en']}")
        print(f"tokens_zhpara: {data['zhpara'][i]['tokens_zh']}")
        
        # 找到每个版本句号前的位置
        positions = {
            "en": find_period_position(data["en"][i]["tokens_en"], is_chinese=False),
            "zh": find_period_position(data["zh"][i]["tokens_zh"], is_chinese=True),
            "enpara": find_period_position(data["enpara"][i]["tokens_en"], is_chinese=False),
            "zhpara": find_period_position(data["zhpara"][i]["tokens_zh"], is_chinese=True)
        }
        
        # 如果任何一个版本找不到句号，跳过
        if -1 in positions.values():
            continue
        
        # 计算两两相似度
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
        
        similarities["en_zhpara"].append(compute_layer_similarity(
            data["en"][i]["hidden_states_en"],
            data["zhpara"][i]["hidden_states_zh"],
            positions["en"],
            positions["zhpara"]
        ))
        
        similarities["zh_enpara"].append(compute_layer_similarity(
            data["zh"][i]["hidden_states_zh"],
            data["enpara"][i]["hidden_states_en"],
            positions["zh"],
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
    
    # 计算平均相似度
    results = {}
    for key, values in similarities.items():
        if values:
            # 转换为numpy数组以便计算
            values_array = np.array(values)
            # 计算每一层的平均值
            avg_values = np.mean(values_array, axis=0)
            results[key] = avg_values.tolist()
    
    # 添加样本数量
    results["sample_count"] = len(similarities["en_zh"])
    
    return results

def main():
    """主函数"""
    # 获取所有主题
    theme_names = [f.stem for f in DATA_DIRS["en"].glob("*.jsonl")]
    print(f"找到 {len(theme_names)} 个主题需要处理")
    
    # 收集所有主题的结果
    all_results = []
    
    # 处理所有主题
    for theme_name in theme_names:
        results = process_theme(theme_name)
        if results:
            all_results.append(results)
    
    # 计算所有主题的平均值
    if all_results:
        # 收集所有主题的相似度
        all_similarities = {
            "en_zh": [],
            "en_enpara": [],
            "en_zhpara": [],
            "zh_enpara": [],
            "zh_zhpara": [],
            "enpara_zhpara": []
        }
        total_samples = 0
        
        for result in all_results:
            for key in all_similarities.keys():
                # 将每个主题的相似度乘以样本数，以便计算加权平均
                similarities = np.array(result[key]) * result["sample_count"]
                all_similarities[key].append(similarities)
            total_samples += result["sample_count"]
        
        # 计算加权平均
        avg_results = {
            key: (np.sum(values, axis=0) / total_samples).tolist()
            for key, values in all_similarities.items()
        }
        avg_results["sample_count"] = total_samples
        
        # 保存结果
        output_path = SAVE_DIR / "all_themes_avg.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(avg_results, f, ensure_ascii=False, indent=2)
        print(f"\n保存所有主题平均值到: {output_path}")
        print(f"总样本数量: {avg_results['sample_count']}")
    
    print("\n✅ 所有主题处理完成！")

if __name__ == "__main__":
    main() 