from transformers import AutoModelForSequenceClassification, AutoTokenizer

output_dir = "./mimic_finetuned"  # 替换为实际的 output_dir

# 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
print("模型和分词器加载成功！")
