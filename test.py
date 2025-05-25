from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 配置
output_dir = "./mimic_finetuned"

# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained("gpt2")
original_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 加载微调模型
finetuned_model = AutoModelForCausalLM.from_pretrained(output_dir)
finetuned_tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 预处理和推理
inputs = tokenizer(test_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 输出结果
predictions = torch.argmax(outputs.logits, dim=-1)
print("输入文本:", test_text)
print("预测类别:", predictions.item())
