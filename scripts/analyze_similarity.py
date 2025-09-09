import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# 配置
INPUT_DIR = Path("outputs/qwen15_7b_00_v2")
OUTPUT_DIR = Path("outputs/similarity_analysis")
DATA_DIR = Path("data/clean_causal_data_v2")

def find_special_positions(tokens: List[str], is_chinese: bool = False) -> Tuple[int, int, int]:
    """找到第一个逗号前、第二个逗号前、第一个句号前的token位置
    
    Args:
        tokens: token列表
        is_chinese: 是否是中文token，决定使用中文还是英文标点
    """
    first_comma = -1
    second_comma = -1
    first_period = -1
    comma_count = 0
    
    # 根据语言选择标点符号
    comma = "，" if is_chinese else ","
    period = '。\n' if is_chinese else '.Ċ'
    
    for i, token in enumerate(tokens):
        if token == comma:
            comma_count += 1
            if comma_count == 1:
                first_comma = i - 1  # 逗号前的token
            elif comma_count == 2:
                second_comma = i - 1
        elif token == period and first_period == -1:
            first_period = i - 1
    print(f'find_special_positions: {tokens[first_comma]}, {tokens[second_comma]}, {tokens[first_period]}')
    
    return first_comma, second_comma, first_period

def compute_layer_similarity(hidden_states_en: List[torch.Tensor], 
                           hidden_states_zh: List[torch.Tensor],
                           pos_en: int,
                           pos_zh: int) -> List[float]:
    """计算指定位置在每一层的隐藏向量相似度"""
    similarities = []
    for layer_idx in range(len(hidden_states_en)):
        # 获取指定位置的隐藏向量
        vec_en = hidden_states_en[layer_idx][0, pos_en].numpy().reshape(1, -1)
        vec_zh = hidden_states_zh[layer_idx][0, pos_zh].numpy().reshape(1, -1)
        
        # 计算余弦相似度
        sim = cosine_similarity(vec_en, vec_zh)[0][0]
        similarities.append(float(sim))
    
    return similarities

def process_theme(theme_name: str):
    """处理单个主题的数据"""
    print(f"\n处理主题: {theme_name}")
    
    # 加载数据
    pt_path = INPUT_DIR / f"{theme_name}.pt"
    if not pt_path.exists():
        print(f"文件不存在: {pt_path}")
        return
    
    records = torch.load(pt_path)
    
    # 初始化结果
    both_correct_similarities = {
        "first_comma": [],
        "second_comma": [],
        "first_period": []
    }
    one_correct_similarities = {
        "first_comma": [],
        "second_comma": [],
        "first_period": []
    }
    
    # 处理每个样本
    for record in tqdm(records, desc="处理样本"):
        # 检查推理结果
        is_correct_en = record["is_correct_en"]
        is_correct_zh = record["is_correct_zh"]
        
        # 只处理both correct或one correct的样本
        if not (is_correct_en and is_correct_zh) and not (is_correct_en or is_correct_zh):
            continue
        
        # 找到特殊位置，分别使用英文和中文标点
        pos_en = find_special_positions(record["tokens_en"], is_chinese=False)
        pos_zh = find_special_positions(record["tokens_zh"], is_chinese=True)
        
        # 如果找不到所有位置，跳过
        if -1 in pos_en or -1 in pos_zh:
            continue
        
        # 计算每一层的相似度
        similarities = {
            "first_comma": compute_layer_similarity(
                record["hidden_states_en"],
                record["hidden_states_zh"],
                pos_en[0],
                pos_zh[0]
            ),
            "second_comma": compute_layer_similarity(
                record["hidden_states_en"],
                record["hidden_states_zh"],
                pos_en[1],
                pos_zh[1]
            ),
            "first_period": compute_layer_similarity(
                record["hidden_states_en"],
                record["hidden_states_zh"],
                pos_en[2],
                pos_zh[2]
            )
        }
        
        # 根据推理结果分类并收集相似度
        if is_correct_en and is_correct_zh:
            for pos in ["first_comma", "second_comma", "first_period"]:
                both_correct_similarities[pos].append(similarities[pos])
        else:
            for pos in ["first_comma", "second_comma", "first_period"]:
                one_correct_similarities[pos].append(similarities[pos])
    
    # 计算平均相似度
    def compute_avg_similarities(similarities_list):
        if not similarities_list:
            return None
        
        # 转换为numpy数组以便计算
        similarities_array = np.array(similarities_list)
        # 计算每一层的平均值
        avg_similarities = np.mean(similarities_array, axis=0)
        return avg_similarities.tolist()
    
    # 保存both_correct结果
    if both_correct_similarities["first_comma"]:
        both_correct_results = {
            "first_comma": compute_avg_similarities(both_correct_similarities["first_comma"]),
            "second_comma": compute_avg_similarities(both_correct_similarities["second_comma"]),
            "first_period": compute_avg_similarities(both_correct_similarities["first_period"]),
            "sample_count": len(both_correct_similarities["first_comma"])
        }
        output_path = OUTPUT_DIR / "both_correct" / f"{theme_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(both_correct_results, f, ensure_ascii=False, indent=2)
        print(f"保存both_correct结果到: {output_path}")
        print(f"both_correct样本数量: {both_correct_results['sample_count']}")
    
    # 保存one_correct结果
    if one_correct_similarities["first_comma"]:
        one_correct_results = {
            "first_comma": compute_avg_similarities(one_correct_similarities["first_comma"]),
            "second_comma": compute_avg_similarities(one_correct_similarities["second_comma"]),
            "first_period": compute_avg_similarities(one_correct_similarities["first_period"]),
            "sample_count": len(one_correct_similarities["first_comma"])
        }
        output_path = OUTPUT_DIR / "one_correct" / f"{theme_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(one_correct_results, f, ensure_ascii=False, indent=2)
        print(f"保存one_correct结果到: {output_path}")
        print(f"one_correct样本数量: {one_correct_results['sample_count']}")
    
    return both_correct_results if both_correct_similarities["first_comma"] else None, \
           one_correct_results if one_correct_similarities["first_comma"] else None

def main():
    """主函数"""
    # 获取所有主题
    theme_names = [f.stem for f in DATA_DIR.glob("*.jsonl")]
    print(f"找到 {len(theme_names)} 个主题需要处理")
    
    # 收集所有主题的结果
    all_both_correct_results = []
    all_one_correct_results = []
    
    # 处理所有主题
    for theme_name in theme_names:
        both_correct, one_correct = process_theme(theme_name)
        if both_correct:
            all_both_correct_results.append(both_correct)
        if one_correct:
            all_one_correct_results.append(one_correct)
    
    # 计算所有主题的平均值
    def compute_all_themes_avg(results_list):
        if not results_list:
            return None
        
        # 收集所有主题的相似度
        all_similarities = {
            "first_comma": [],
            "second_comma": [],
            "first_period": []
        }
        total_samples = 0
        
        for result in results_list:
            for pos in ["first_comma", "second_comma", "first_period"]:
                # 将每个主题的相似度乘以样本数，以便计算加权平均
                similarities = np.array(result[pos]) * result["sample_count"]
                all_similarities[pos].append(similarities)
            total_samples += result["sample_count"]
        
        # 计算加权平均
        avg_results = {
            "first_comma": (np.sum(all_similarities["first_comma"], axis=0) / total_samples).tolist(),
            "second_comma": (np.sum(all_similarities["second_comma"], axis=0) / total_samples).tolist(),
            "first_period": (np.sum(all_similarities["first_period"], axis=0) / total_samples).tolist(),
            "sample_count": total_samples
        }
        
        return avg_results
    
    # 保存所有主题的平均结果
    if all_both_correct_results:
        avg_both_correct = compute_all_themes_avg(all_both_correct_results)
        output_path = OUTPUT_DIR / "both_correct" / "all_themes_avg.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(avg_both_correct, f, ensure_ascii=False, indent=2)
        print(f"\n保存both_correct所有主题平均值到: {output_path}")
        print(f"总样本数量: {avg_both_correct['sample_count']}")
    
    if all_one_correct_results:
        avg_one_correct = compute_all_themes_avg(all_one_correct_results)
        output_path = OUTPUT_DIR / "one_correct" / "all_themes_avg.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(avg_one_correct, f, ensure_ascii=False, indent=2)
        print(f"保存one_correct所有主题平均值到: {output_path}")
        print(f"总样本数量: {avg_one_correct['sample_count']}")
    
    print("\n✅ 所有主题处理完成！")

if __name__ == "__main__":
    main() 