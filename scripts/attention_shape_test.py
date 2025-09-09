from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置模型名
model_name = "Qwen/Qwen1.5-1.8B-Chat"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, trust_remote_code=True)

# 准备输入文本
input_text = "Hello, how are you today?"
inputs = tokenizer(input_text, return_tensors="pt")

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)

# 获取 attention
attentions = outputs.attentions  # List[Tensor], 每层一个 tensor

# 输出第一层 attention 的 shape
print(f"第一层 attention 的形状: {attentions[1].shape}")