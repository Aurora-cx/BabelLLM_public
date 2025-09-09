from transformers import AutoTokenizer

# 加载 Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

def find_token_indices_for_word(prompt: str, target_word: str):
    # 编码，并返回 offset 映射（字符级别位置）
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
    offsets = encoding.offset_mapping[0].tolist()

    # 找到目标词的起止字符位置
    start_idx = prompt.index(target_word)
    end_idx = start_idx + len(target_word)

    # 选出与目标词有重叠的 token
    target_token_indices = [
        i for i, (s, e) in enumerate(offsets)
        if not (e <= start_idx or s >= end_idx)
    ]

    print(f"\n🔍 Prompt: {prompt}")
    print(f"🎯 Target word: '{target_word}'")
    print(f"🧩 Token indices: {target_token_indices}")
    print(f"🧷 Tokens: {[tokens[i] for i in target_token_indices]}")

    return target_token_indices, [tokens[i] for i in target_token_indices]

# ✅ DEMO 示例
prompt = "Once vaccine develops, immunity strengthdhucfhens, then antibodies reduce."
target_word = "strengthdhucfhens"

find_token_indices_for_word(prompt, target_word)