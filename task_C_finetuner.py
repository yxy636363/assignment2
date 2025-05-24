from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download, login
import os
import torch

#加载 AUEB-NLP ECtHR 案例数据集
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
dataset_path = "./mimic_data"
train_text_number = 1000
if not os.path.exists(dataset_path):
    try:
        dataset = load_dataset("Medilora/mimic_iii_diagnosis_anonymous")
        print("数据集加载成功！")
    except Exception as e:
        print(f"错误: {e}")
        print("尝试登录后下载...")
        try:
            login(token="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")
            dataset = load_dataset("Medilora/mimic_iii_diagnosis_anonymous")
        except Exception as e:
            print(f"登录后下载失败: {e}")
else:
    dataset = load_from_disk(dataset_path)

#加载预训练模型和分词器
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

#对文本进行分词处理
def tokenize_func(examples):
    return tokenizer(examples['text'], truncation=True,
                     max_length=256, padding="max_length",
                     return_tensors="pt")
tokenized_dataset = dataset.map(tokenize_func, batched=True)

#设置参数
output_dir = "./mimic_finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=100,
    save_steps=500,
    do_eval=False,
    # evaluation_strategy="no",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

#训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
print("Start training!")
trainer.train()

#保存
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"saved at: {output_dir}")
