from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import snapshot_download, login
from torch.nn.utils.rnn import pad_sequence
import os
import torch

#加载 MIMIC-III 案例数据集
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
dataset_path = "./mimic_data"
train_text_number = 1000
if not os.path.exists(dataset_path):
    dataset = load_dataset("Medilora/mimic_iii_diagnosis_anonymous")
    print("load successfully!")
else:
    dataset = load_from_disk(dataset_path)

#加载预训练模型和分词器
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

#对文本进行分词处理
def tokenize_func(examples):
    tokenized = tokenizer(examples['text'], truncation=True, max_length=256, padding=True)
    assert all(isinstance(x, int) for x in tokenized["input_ids"][0]), "存在非数字token"
    return tokenized
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
    do_eval=False,        # evaluation_strategy="no",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False,pad_to_multiple_of=8)

class TorchDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # 转换为PyTorch张量
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["input_ids"] for f in features])
        }
        return batch

data_collator = TorchDataCollator(tokenizer=tokenizer, mlm=False)

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
trainer.save_model(output_dir)
print(f"saved at: {output_dir}")
