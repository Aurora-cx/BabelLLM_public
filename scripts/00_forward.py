import json
import re
from pathlib import Path
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from openai import OpenAI
from typing import Tuple

def print_gpu_memory():
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        print("\nGPU显存使用情况:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("没有可用的GPU")

def zh_decode(input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i in range(len(tokens)):
        tokens[i] = tokenizer.decode(input_ids[i], skip_special_tokens=False)
    return tokens

# 初始化OpenAI客户端
client = OpenAI(api_key="YOUR_API_KEY_HERE")

def check_answer_with_gpt(predicted: str, gold: str, prompt: str, lang: str = "en") -> Tuple[bool, str]:
    """使用GPT来判断答案的正确性
    
    Args:
        predicted: 模型预测的答案
        gold: 标准答案
        prompt: 原始问题
        lang: 语言，"en"或"zh"
        
    Returns:
        Tuple[bool, str]: (是否正确, 判断理由)
    """
    system_prompt = {
        "en": """You are an answer evaluator. Your task is to determine if the predicted answer contains the core information from the ground truth answer. Please note the following:

1. The answer must contain the information that has same meaning as the ground truth, even if it's presented in a different way or with additional context.
2. If the answer uses conditional statements (if/then) that logically connect to the question, it should be considered correct.
3. Different phrasings and synonyms are acceptable as long as the core information matches the ground truth.
4. Additional relevant information that doesn't contradict the ground truth is acceptable.

Your response must strictly follow this format:

[0/1]: <your reasoning>

Where 0 means incorrect and 1 means correct.

Example:
Prompt: When the valve closes, the pressure drops.
When the pressure drops, the alarm rings.
What happens when the valve closes?
Ground Truth: The alarm rings.
Predicted Answer: When the valve closes, the alarm will sound, which is a safety measure to alert operators.

[1]: The predicted answer contains the core information about the alarm activation. Using "will sound" instead of "rings" is semantically equivalent, and the extra explanation doesn't contradict the ground truth.

Example:
Prompt: When the valve closes, the pressure drops.
When the pressure drops, the alarm rings.
What happens when the valve closes?
Ground Truth: The alarm rings.
Predicted Answer: The valve makes a sound.

[0]: While the answer mentions a sound, it doesn't specify that it's the alarm that makes the sound. The core information about the alarm is missing.

Example:
Prompt: When it rains, the field floods.
When the field floods, practice is cancelled.
So, if it rains, what happens to practice?
Ground Truth: Practice is cancelled.
Predicted Answer: If the field floods, practice is cancelled.

[1]: The answer contains the core information that practice is cancelled. Although it uses a conditional statement, it logically connects to the question about rain through the given conditions (rain → field floods → practice cancelled).""",

        "zh": """你是一个答案评估者。你需要判断预测答案是否包含了标准答案。请注意以下几点：

1. 答案必须包含标准答案中的核心信息，即使表达方式不同或包含额外上下文。
2. 如果答案使用条件语句（如果/就）来逻辑地连接问题，应该判定为正确。
3. 不同的表达方式和同义词是可以接受的，只要核心信息与标准答案一致。
4. 额外的相关信息，只要不与标准答案矛盾，是可以接受的。

你的回答必须严格遵循以下格式：

[0/1]：<你的理由>

其中0表示错误，1表示正确。

示例：
提示：阀门一旦关闭，压力就下降。
压力一下降，警报就响。
阀门关闭时会怎样？
标准答案：警报响。
预测答案：当阀门关闭时，警报会响起，这是一个提醒操作人员的安全措施。

[1]：预测答案包含了警报激活的核心信息。使用"会响起"而不是"响"是语义等价的，额外的解释也不与标准答案矛盾。

示例：
提示：阀门一旦关闭，压力就下降。
压力一下降，警报就响。
阀门关闭时会怎样？
标准答案：警报响。
预测答案：阀门会发出声音。

[0]：虽然答案提到了声音，但没有明确指出是警报发出的声音。关于警报的核心信息缺失了。

示例：
提示：一旦下雨，球场就积水。
球场一积水，训练就取消。
因此，如果下雨，训练会怎样？
标准答案：训练取消。
预测答案：如果球场积水，训练就取消。

[1]：答案包含了训练取消的核心信息。虽然使用了条件语句，但通过给定的条件链（下雨→球场积水→训练取消）逻辑地连接到了关于下雨的问题。"""
    }
    
    messages = [
        {"role": "system", "content": system_prompt["en"]},
        {"role": "user", "content": f"""Question: {prompt}
Ground Truth: {gold}
Predicted Answer: {predicted}

Please evaluate if the predicted answer is correct and provide your reasoning."""}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=150
        )
        result = response.choices[0].message.content.strip()
        # 提取[0/1]中的数字
        match = re.search(r'\[([01])\]', result)
        if match:
            is_correct = match.group(1) == "1"
            # 提取理由部分（去掉[0/1]标记）
            reason = re.sub(r'^\[[01]\][：:]\s*', '', result).strip()
            return is_correct, reason
        else:
            print(f"Unexpected response format: {result}")
            return gold in predicted, "Unexpected response format, using simple string matching"
    except Exception as e:
        print(f"GPT API call failed: {e}")
        # 如果API调用失败，回退到简单的字符串匹配
        return gold in predicted, "API call failed, using simple string matching"


MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
DATA_DIR = Path("data/paraphrased_causal_data_v2")
OUTPUT_DIR = Path("outputs/qwen15_18b_para_v2_00")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定使用GPU 1
DTYPE = torch.float16  # fp16 节省显存

#############################################
# 1. 加载模型 & tokenizer
#############################################
print(f"Loading model {MODEL_NAME} ...")
print_gpu_memory()

# For Qwen1.5 use trust_remote_code to enable chat template
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",  # 指定使用GPU 1
    torch_dtype=DTYPE,
    output_hidden_states=True,
    output_attentions=True,
    trust_remote_code=True,
    attn_implementation="eager"  # 使用eager attention来避免警告
)
model.eval()

print("\n模型加载完成后的显存状态:")
print_gpu_memory()

#############################################
# 2. 实用的答案比对函数
#############################################

def check_answer(predicted: str, gold: str) -> bool:
    """任务无关的简单比对：数字→数字；Yes/No→忽略大小写；否则子串"""
    gold = gold.strip()
    predicted = predicted.strip()

    # 数值答案
    if re.fullmatch(r"[-+]?\d+", gold):
        return re.search(r"[-+]?\d+", predicted) and re.search(r"[-+]?\d+", predicted).group() == gold

    # Yes/No
    if gold.lower() in {"yes", "no"}:
        return gold.lower() in predicted.lower()

    # 其余：子串包含
    return gold in predicted

#############################################
# 3. 处理所有数据文件
#############################################
# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 获取所有jsonl文件
jsonl_files = list(DATA_DIR.glob("*.jsonl"))
print(f"\n找到 {len(jsonl_files)} 个文件需要处理")

# 使用tqdm显示整体进度
for data_file in tqdm(jsonl_files, desc="处理文件", unit="file"):
    print(f"\n开始处理 {data_file.name}...")
    output_path = OUTPUT_DIR / f"{data_file.stem}.pt"
    output_jsonl = OUTPUT_DIR / f"{data_file.stem}.jsonl"
    
    # 读取数据
    print(f"Reading prompts from {data_file} ...")
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    #############################################
    # 4. 主循环: 前向 & 保存
    #############################################
    print("Running forward passes ...")
    all_records = []
    # 使用tqdm显示样本处理进度
    for sample in tqdm(dataset, desc="处理样本", leave=False):
        sample_id = sample.get("id", None)
        prompt_text_en = "\n".join(sample["en_para"])
        prompt_text_zh = "\n".join(sample["zh_para"])
        ground_truth = sample.get("answer", "")

        # ---- Qwen chat template ----
        messages_en = [
            {"role": "system", "content": "Use only one sentence to answer."},
            {"role": "user", "content": prompt_text_en}
        ]
        messages_zh = [
            {"role": "system", "content": "只用一句话回答。"},
            {"role": "user", "content": prompt_text_zh}
        ]
        chat_prompt_en = tokenizer.apply_chat_template(messages_en, tokenize=False, add_generation_prompt=True)
        chat_prompt_zh = tokenizer.apply_chat_template(messages_zh, tokenize=False, add_generation_prompt=True)
        inputs_en = tokenizer(chat_prompt_en, return_tensors="pt").to(DEVICE)
        inputs_zh = tokenizer(chat_prompt_zh, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs_en = model(**inputs_en)
            outputs_zh = model(**inputs_zh)

        # ---- 基本张量抽取 ----
        input_ids_en = inputs_en["input_ids"].squeeze(0).tolist()
        tokens_en = tokenizer.convert_ids_to_tokens(input_ids_en)
        input_ids_zh = inputs_zh["input_ids"].squeeze(0).tolist()
        tokens_zh = zh_decode(input_ids_zh)

        # hidden_states[0] = embeddings 输出
        embedding_out_en = outputs_en.hidden_states[0].cpu()
        embedding_out_zh = outputs_zh.hidden_states[0].cpu()
        hidden_states_en = [h.cpu() for h in outputs_en.hidden_states[:]]  # List[L]
        hidden_states_zh = [h.cpu() for h in outputs_zh.hidden_states[:]]  # List[L]
        attentions_en = [a[0].cpu() for a in outputs_en.attentions]  # List[L] (n_heads, seq_len, seq_len)
        attentions_zh = [a[0].cpu() for a in outputs_zh.attentions]  # List[L] (n_heads, seq_len, seq_len)
        logits_en = outputs_en.logits[0].cpu()
        logits_zh = outputs_zh.logits[0].cpu()

        # ---- 进行生成以获取回答 ----
        generated_ids_en = model.generate(
            inputs_en.input_ids,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.1,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0
        )
        generated_ids_zh = model.generate(
            inputs_zh.input_ids,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.1,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0
        )
        generated_ids_en = [
            output_ids[len(input_ids_en):] for input_ids_en, output_ids in zip(inputs_en.input_ids, generated_ids_en)
        ]
        generated_ids_zh = [
            output_ids[len(input_ids_zh):] for input_ids_zh, output_ids in zip(inputs_zh.input_ids, generated_ids_zh)
        ]
        predicted_answer_en = tokenizer.batch_decode(generated_ids_en, skip_special_tokens=True)[0].strip()
        predicted_answer_zh = tokenizer.batch_decode(generated_ids_zh, skip_special_tokens=True)[0].strip()
        print(f"English Predicted Answer: {predicted_answer_en}")
        print(f"Chinese Predicted Answer: {predicted_answer_zh}")
        
        # 使用GPT评估答案
        is_correct_en, reason_en = check_answer_with_gpt(predicted_answer_en, ground_truth["en"], prompt_text_en, "en")
        is_correct_zh, reason_zh = check_answer_with_gpt(predicted_answer_zh, ground_truth["zh"], prompt_text_zh, "zh")
        
        print(f"English Answer Evaluation: [{1 if is_correct_en else 0}] {reason_en}")
        print(f"Chinese Answer Evaluation: [{1 if is_correct_zh else 0}] {reason_zh}")

        # ---- 保存记录 ----
        all_records.append({
            "id": sample_id,
            "prompt_en": prompt_text_en,
            "prompt_zh": prompt_text_zh,
            "input_ids_en": input_ids_en,
            "tokens_en": tokens_en,
            "input_ids_zh": input_ids_zh,
            "tokens_zh": tokens_zh,
            "embedding_out_en": embedding_out_en,
            "embedding_out_zh": embedding_out_zh,
            "hidden_states_en": hidden_states_en,
            "hidden_states_zh": hidden_states_zh,
            "attentions_en": attentions_en,
            "attentions_zh": attentions_zh,
            "logits_en": logits_en,
            "logits_zh": logits_zh,
            "predicted_answer_en": predicted_answer_en,
            "predicted_answer_zh": predicted_answer_zh,
            "ground_truth": ground_truth,
            "is_correct_en": is_correct_en,
            "is_correct_zh": is_correct_zh,
            "judgment_reason_en": reason_en,
            "judgment_reason_zh": reason_zh
        })

    # ---- 写文件 ----
    print(f"Saving {len(all_records)} records to {output_path} ...")
    torch.save(all_records, output_path)

    # 统计正确率
    correct_en = sum(1 for record in all_records if record["is_correct_en"])
    correct_zh = sum(1 for record in all_records if record["is_correct_zh"])
    total = len(all_records)
    
    # 保存统计结果到单独的文件
    stats = {
        "file_name": data_file.name,
        "total_samples": total,
        "correct_en": correct_en,
        "correct_zh": correct_zh,
        "accuracy_en": correct_en/total*100,
        "accuracy_zh": correct_zh/total*100
    }
    
    stats_file = OUTPUT_DIR / f"{data_file.stem}_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n正确率统计 ({data_file.stem}):")
    print(f"英文: {correct_en}/{total} = {correct_en/total*100:.2f}%")
    print(f"中文: {correct_zh}/{total} = {correct_zh/total*100:.2f}%")
    print(f"统计结果已保存到: {stats_file}")

    # ✅ Save natural language outputs to a readable text file
    nature_res = []
    for i, output in enumerate(all_records, 1):
        new_irem = {}
        new_irem["id"] = output["id"]
        new_irem["prompt_en"] = output["prompt_en"]
        new_irem["predicted_answer_en"] = output["predicted_answer_en"]
        new_irem["ground_truth_en"] = output["ground_truth"]["en"]
        new_irem["is_correct_en"] = output["is_correct_en"]
        new_irem["judgment_reason_en"] = output["judgment_reason_en"]
        new_irem["prompt_zh"] = output["prompt_zh"]
        new_irem["predicted_answer_zh"] = output["predicted_answer_zh"]
        new_irem["ground_truth_zh"] = output["ground_truth"]["zh"]
        new_irem["is_correct_zh"] = output["is_correct_zh"]
        new_irem["judgment_reason_zh"] = output["judgment_reason_zh"]
        nature_res.append(new_irem)

    # 保存详细结果到jsonl文件
    output_jsonl = OUTPUT_DIR / f"{data_file.stem}_results.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in nature_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"All intermediate activations and results have been saved for {data_file.stem}!")
    print(f"Natural language outputs have been saved to {output_jsonl}!")

# 在所有文件处理完成后，生成一个汇总报告
all_stats = []
for data_file in jsonl_files:
    stats_file = OUTPUT_DIR / f"{data_file.stem}_stats.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            all_stats.append(json.load(f))

# 保存汇总报告
summary_file = OUTPUT_DIR / "summary_stats.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(all_stats, f, ensure_ascii=False, indent=2)

print("\n所有文件处理完成！")
print(f"汇总报告已保存到: {summary_file}") 