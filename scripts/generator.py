import openai
import json
import time
from pathlib import Path
from transformers import AutoTokenizer
from collections import deque
import re

# 加载 Qwen Tokenizer（或换成你的模型）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

# 设置你的 OpenAI API Key
client = openai.OpenAI(api_key="")
def extract_json(text):
    # 删除 markdown 代码块格式
    if text.strip().startswith("```json"):
        return re.sub(r"^```json\\n?|\\n?```$", "", text.strip(), flags=re.MULTILINE).strip()
    return text.strip()
# 8 个领域
zh_domains = [
    "家庭日常", 
    # "自然现象", "校园生活", "医疗健康",
    # "商业消费", "职场办公", "公共交通", "娱乐休闲"
]

domain_mapping = {
    "家庭日常": "household_routine",
    "自然现象": "natural_events",
    "校园生活": "school_life",
    "医疗健康": "healthcare",
    "商业消费": "shopping_retail",
    "职场办公": "workplace_activities",
    "公共交通": "public_transportation",
    "娱乐休闲": "leisure_recreation"
}

# 输出文件夹
output_dir = Path("data/clean_causal_data_v2")
output_dir.mkdir(parents=True, exist_ok=True)

# Prompt模板
def generate_prompt(domain, used_subjects_en, used_verbs_en, used_subjects_zh, used_verbs_zh):
    banned_subjects_en = ", ".join(used_subjects_en) if used_subjects_en else "(none)"
    banned_verbs_en = ", ".join(used_verbs_en) if used_verbs_en else "(none)"
    banned_subjects_zh = "、".join(used_subjects_zh) if used_subjects_zh else "（无）"
    banned_verbs_zh = "、".join(used_verbs_zh) if used_verbs_zh else "（无）"
    
    base_prompt = f"""Please generate a 3-step causal chain in the field of {domain}.

You must follow all of these constraints:

1. The English sentence must follow this structure:  
   "Once [subject1][verb1], [subject2][verb2], then [subject3][verb3]."  
   - Each subject and each verb must be a single word.  
   - Select verbs that can form complete meaningful sentences without requiring an object.
   - Do not include determiners, adjectives, or multi-word phrases.  
   - Make sure every component (subject1, verb1, subject2, verb2, subject3, verb3) can be tokenized into exactly one token by Qwen-1.5-1.8B-Chat.
   - Example (correct): "Once temperature rises, ice melts, then water flows."  
   - Example (incorrect): "Once the temperature is rising..."

2. The Chinese sentence must mirror the same structure:  
   "一旦[subject1][verb1]，[subject2][verb2]，然后[subject3][verb3]。"  
   - Each subject and verb must be a single standalone Chinese word, with no modifiers or compound forms.  
   - Example (correct): "温度一旦升高，冰就融化，然后水就流动。"  
   - Example (incorrect): "温度一旦被升高..."

3. Also generate a question-answer pair for both languages:
   - English question: "Therefore, if [cause_subject] [cause_verb], the final result is"
   - English answer: "[final_subject] [final_verb]."
   - Chinese question: "因此，如果[cause_subject][cause_verb]，最终结果是"
   - Chinese answer: "[final_subject][final_verb]。"
"""
    example_prompt = """
Return your result in the following JSON format:

{
  "en": [
    "Once temperature rises, ice melts, then water flows.",
    "Therefore, if temperature rises, the final result is"
  ],
  "zh": [
    "温度一旦升高，冰就融化，然后水就流动。",
    "因此，如果温度升高，最终结果是"
  ],
  "answer": {
    "en": "water flows.",
    "zh": "水流动。"
  },
  "labels": {
    "en": {
      "cause_subject": "temperature",
      "cause_verb": "rises",
      "intermediate_subject": "ice",
      "intermediate_verb": "melts",
      "final_subject": "water",
      "final_verb": "flows"
    },
    "zh": {
      "cause_subject": "温度",
      "cause_verb": "升高",
      "intermediate_subject": "冰",
      "intermediate_verb": "融化",
      "final_subject": "水",
      "final_verb": "流动"
    }
  },
  "causal_labels":{
    "en":{
      "cause_sentence": "Once temperature rises,",
      "effect_sentence_1": "ice melts,",
      "effect_sentence_2": "then water flows."
    },
    "zh":{
      "cause_sentence": "温度一旦升高，",
      "effect_sentence_1": "冰就融化，",
      "effect_sentence_2": "然后水就流动。"
    }
  }
}"""


    banned_words = f"""\DO NOT use these words in the causal chain:
EnglishSubjects: {banned_subjects_en}
ChineseSubjects: {banned_subjects_zh}
"""

    final_instructions = f"""
⚠ Important: 
- All subject and verb entries must be single words (1 word only).   
- Do not use multi-word phrases like "the button" or "turns off".  
- All values in the "labels" must be a single word.
- Only generate literal, real-world events. 
- Check if the content cling to the {domain}.
- Check if the cause-effect relationship is natural and reasonable.
- Only return valid JSON string that can be parsed by Python directly.
"""

    return base_prompt + banned_words+ example_prompt  + final_instructions

def label_token_lengths(labels, lang="en"):
    return {k: len(tokenizer.tokenize(v)) for k, v in labels[lang].items()}

def is_valid_labels(labels, lang="en"):
    for key, val in labels[lang].items():
        # 新策略：允许多 token，只拒绝带空格或明显短语的
        if " " in val.strip() or len(val.strip()) == 0:
            return False
    return True

# 为每个领域生成50条干净数据，加入熔断机制（最多尝试100次）
def generate_clean_data(domain, n_target=50, max_trials=100):
    clean_samples = []
    tries = 0
    used_subjects_en = deque(maxlen=60)
    used_verbs_en = deque(maxlen=60)
    used_subjects_zh = deque(maxlen=60)
    used_verbs_zh = deque(maxlen=60)
    print(f"\n🚀 Generating samples for domain: {domain}")
    
    while len(clean_samples) < n_target and tries < max_trials:
        tries += 1
        try:
            prompt = generate_prompt(domain, used_subjects_en, used_verbs_en, used_subjects_zh, used_verbs_zh)
            print(f"\nTrial {tries}: Sending prompt to GPT-4...")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            text = response.choices[0].message.content
            print(f"Received response: {text[:200]}...")  # 只打印前200个字符
            text = extract_json(text)
            parsed = json.loads(text)
            print(parsed)
            
            if "labels" not in parsed:
                print(f"⛔ Missing 'labels' field at trial {tries}")
                continue
            
            if is_valid_labels(parsed["labels"], "en") and is_valid_labels(parsed["labels"], "zh"):
                parsed["token_lengths"] = {
                    "en": label_token_lengths(parsed["labels"], "en"),
                    "zh": label_token_lengths(parsed["labels"], "zh")
                }
                if parsed["labels"]['en']['cause_subject'] not in used_subjects_en:
                    clean_samples.append(parsed)
                else:
                    print(f"❌ [{len(clean_samples)}/{n_target}] {parsed['en'][0]}")  # 只打印英文句子
                print(f"✅ [{len(clean_samples)}/{n_target}] {parsed['en'][0]}")  # 只打印英文句子
                used_subjects_en.append(parsed["labels"]["en"]["cause_subject"])
                used_subjects_en.append(parsed["labels"]["en"]["intermediate_subject"])
                used_subjects_en.append(parsed["labels"]["en"]["final_subject"])
                used_verbs_en.append(parsed["labels"]["en"]["cause_verb"])
                used_verbs_en.append(parsed["labels"]["en"]["intermediate_verb"])
                used_verbs_en.append(parsed["labels"]["en"]["final_verb"])
                used_subjects_zh.append(parsed["labels"]["zh"]["cause_subject"])
                used_subjects_zh.append(parsed["labels"]["zh"]["intermediate_subject"])
                used_subjects_zh.append(parsed["labels"]["zh"]["final_subject"])
                used_verbs_zh.append(parsed["labels"]["zh"]["cause_verb"])
                used_verbs_zh.append(parsed["labels"]["zh"]["intermediate_verb"])
                used_verbs_zh.append(parsed["labels"]["zh"]["final_verb"])

            else:
                print(f"❌ Token split detected at trial {tries}, skipped.")

        except Exception as e:
            print(f"⚠ Error at trial {tries}: {e}")
            time.sleep(3)
    
    # 保存当前数据（无论是否满50条）
    output_file = output_dir / f"{domain}.jsonl"
    print(f"\nSaving {len(clean_samples)} samples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in clean_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 验证文件是否成功保存
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            saved_lines = f.readlines()
        print(f"Verified: File contains {len(saved_lines)} lines")
    else:
        print("❌ Error: File was not created!")
    
    # 打印日志
    if len(clean_samples) < n_target:
        print(f"🚨 [STOPPED EARLY] {domain}: Only collected {len(clean_samples)} samples after {tries} trials.")
    else:
        print(f"🎉 [DONE] {domain}: Collected {n_target} samples in {tries} trials.")

# 执行所有领域
for zh_domain in zh_domains:
    generate_clean_data(domain_mapping[zh_domain])
