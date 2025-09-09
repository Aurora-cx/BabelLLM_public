import torch, numpy as np, tqdm, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设定环境
torch.cuda.set_device(0)
torch.cuda.empty_cache()

# 配置
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
DATA_DIR = Path("data/clean_causal_data")
FORWARD_DIR = Path("outputs/qwen15_18b_00")
SAVE_DIR = Path("outputs/attention_weights_zh")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    output_hidden_states=True,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

def zh_decode(input_ids):
    """将input_ids转换为中文tokens"""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i in range(len(tokens)):
        tokens[i] = tokenizer.decode(input_ids[i], skip_special_tokens=False)
    return tokens

def find_token_indices_for_word(prompt: str, target_word: str):
    """找到目标词在文本中被tokenizer切分后的token索引"""
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    tokens = zh_decode(encoding.input_ids[0])
    offsets = encoding.offset_mapping[0].tolist()

    start_idx = prompt.index(target_word)
    end_idx = start_idx + len(target_word)

    target_token_indices = [
        i for i, (s, e) in enumerate(offsets)
        if not (e <= start_idx or s >= end_idx)
    ]

    return target_token_indices, [tokens[i] for i in target_token_indices]

def compute_component_attention(attentions, token_indices, num_heads):
    """计算每个组件在每个head上的attention权重"""
    # attentions: [num_layers, batch_size, num_heads, seq_len, seq_len]
    # layer_attention: [batch_size, num_heads, seq_len, seq_len]
    # head_attention: [seq_len, seq_len]
    attention_sums = np.zeros((len(attentions), num_heads))
    
    for layer_idx, layer_attention in enumerate(attentions):
        if isinstance(layer_attention, list):
            layer_attention = layer_attention[0]  # 如果是列表，取第一个元素
        
        if layer_attention.dim() == 3:
            # 如果维度是3，说明是[num_heads, seq_len, seq_len]
            layer_attention = layer_attention.unsqueeze(0)  # 添加batch维度
        
        # 确保layer_attention的维度是[batch_size, num_heads, seq_len, seq_len]
        if layer_attention.dim() != 4:
            print(f"Warning: Unexpected attention shape at layer {layer_idx}: {layer_attention.shape}")
            continue
            
        # 取第一个样本的attention
        layer_attention = layer_attention[0]  # [num_heads, seq_len, seq_len]
        
        for head_idx in range(num_heads):
            head_attention = layer_attention[head_idx]  # [seq_len, seq_len]
            # 计算token_indices对应的attention权重之和
            attention_sum = head_attention[:, token_indices].sum().item()
            attention_sums[layer_idx, head_idx] = attention_sum
            
    return attention_sums

def load_theme_data(theme_name: str):
    """加载单个主题的数据"""
    # 加载原始数据
    jsonl_path = DATA_DIR / f"{theme_name}.jsonl"
    raw_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    # 加载前向结果
    pt_path = FORWARD_DIR / f"{theme_name}.pt"
    forward_data = torch.load(pt_path, map_location="cpu")
    
    # 确保forward_data是列表
    if not isinstance(forward_data, list):
        forward_data = [forward_data]

    # 合并数据
    for raw_sample, forward_sample in zip(raw_data, forward_data):
        raw_sample["is_correct_zh"] = forward_sample.get("is_correct_zh", False)
        raw_sample["input_ids_zh"] = forward_sample.get("input_ids_zh", None)
        raw_sample["attentions_zh"] = forward_sample.get("attentions_zh", None)

    return raw_data

def process_sample(model, sample: dict):
    """处理单个样本，计算所有组件的attention权重"""
    # 构造正确的消息格式
    messages = [
        {"role": "system", "content": "只用一句话回答。"},
        {"role": "user", "content": "\n".join(sample["zh"])}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 使用新的prompt生成input_ids
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    labels = sample["labels"]["zh"]
    # 添加标记词组件
    labels["marker_once"] = ["一旦"]
    labels["marker_then"] = ["然后"]

    # 统计每个组件的token数量
    token_lengths = {}
    for component, value in labels.items():
        if isinstance(value, list):
            value = value[0]  # 取第一个值
        token_lengths[component] = len(tokenizer.tokenize(value))
    
    # 获取attention权重
    attentions_zh = sample["attentions_zh"]
    if attentions_zh is None:
        print(f"Warning: No attention weights found for sample {sample.get('id', 'unknown')}")
        return None

    # 计算每个组件的attention权重
    component_attentions = {}
    component_tokens = {}

    # 获取attention head的数量
    num_heads = attentions_zh[0].shape[0]  # 使用第一个attention层的head数量

    components = [
        "cause_subject", "cause_verb",
        "intermediate_subject", "intermediate_verb",
        "final_subject", "final_verb",
        "marker_once", "marker_then"
    ]

    for component in components:
        value = labels[component]
        token_indices, tokens = find_token_indices_for_word(prompt, value[0])
        if not token_indices:
            print(f"Warning: Could not find tokens for {component}: {value[0]}")
            continue

        attention_sums = compute_component_attention(attentions_zh, token_indices, num_heads)
        component_attentions[component] = attention_sums.tolist()
        component_tokens[component] = tokens

    return {
        "sample_id": sample.get("id", None),
        "attentions": component_attentions,
        "tokens": component_tokens,
        "token_lengths": token_lengths,  # 添加token长度统计
        "prompt": prompt,
        "labels": labels,
        "is_correct_zh": sample.get("is_correct_zh", False)
    }

def process_theme(theme_name: str):
    """处理单个主题的数据"""
    print(f"\nProcessing theme: {theme_name}")
    theme_data = load_theme_data(theme_name)

    # 创建保存目录
    correct_dir = SAVE_DIR / "correct"
    incorrect_dir = SAVE_DIR / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    correct_results = []
    incorrect_results = []
    error_count = 0
    total_samples = len(theme_data)

    # 添加token长度统计
    total_token_lengths = {
        "cause_subject": 0,
        "cause_verb": 0,
        "intermediate_subject": 0,
        "intermediate_verb": 0,
        "final_subject": 0,
        "final_verb": 0,
        "marker_once": 0,
        "marker_then": 0
    }
    processed_samples = 0

    for sample in tqdm.tqdm(theme_data):
        try:
            result = process_sample(model, sample)
            if result is not None:
                # 累加token长度统计
                for component, length in result["token_lengths"].items():
                    total_token_lengths[component] += length
                processed_samples += 1

                if result["is_correct_zh"]:
                    correct_results.append(result)
                else:
                    incorrect_results.append(result)
        except Exception as e:
            error_count += 1
            print(f"\nError processing sample {sample.get('id', 'unknown')}: {str(e)}")
            continue
        finally:
            torch.cuda.empty_cache()

    # 计算平均token长度
    avg_token_lengths = {
        component: total / processed_samples 
        for component, total in total_token_lengths.items()
    }

    # 保存正确样本的结果
    correct_file = correct_dir / f"attention_{theme_name}.json"
    with open(correct_file, "w", encoding="utf-8") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=2)
    print(f"Correct results saved to {correct_file}")

    # 保存错误样本的结果
    incorrect_file = incorrect_dir / f"attention_{theme_name}.json"
    with open(incorrect_file, "w", encoding="utf-8") as f:
        json.dump(incorrect_results, f, ensure_ascii=False, indent=2)
    print(f"Incorrect results saved to {incorrect_file}")

    # 打印统计信息
    print(f"\nProcessing statistics for {theme_name}:")
    print(f"Total samples: {total_samples}")
    print(f"Successfully processed: {len(correct_results) + len(incorrect_results)}")
    print(f"Errors encountered: {error_count}")
    print(f"Success rate: {(total_samples - error_count) / total_samples * 100:.2f}%")
    
    # 打印token长度统计
    print("\nAverage token lengths per component:")
    for component, avg_length in avg_token_lengths.items():
        print(f"{component}: {avg_length:.2f} tokens")

def main():
    """主函数"""
    # 获取所有主题名称
    theme_names = [f.stem for f in DATA_DIR.glob("*.jsonl")]
    print(f"Found {len(theme_names)} themes: {theme_names}")
    
    # 处理所有主题
    for theme_name in theme_names:
        print(f"\nProcessing theme: {theme_name}")
        process_theme(theme_name)

    print("✅ Attention weight analysis 完成！")

if __name__ == "__main__":
    main() 