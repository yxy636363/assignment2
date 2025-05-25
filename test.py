from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 配置
output_dir = "./mimic_finetuned"
test_text = "Patient is a 65-year-old male admitted for acute chest pain and shortness of breath."

# 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 预处理和推理
inputs = tokenizer(test_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 输出结果
predictions = torch.argmax(outputs.logits, dim=-1)
print("输入文本:", test_text)
print("预测类别:", predictions.item())
