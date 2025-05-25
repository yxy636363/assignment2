from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download, login
from torch.nn.utils.rnn import pad_sequence
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
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

#对文本进行分词处理
def tokenize_func(examples):
    batch = tokenizer(examples['text'], truncation=True, max_length=256, 
                     padding=True, return_tensors="None")
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"]
    }
tokenized_dataset = dataset.map(tokenize_func, batched=True)

sample = tokenized_dataset["train"][0]
print("Tokenized样本:", sample)
print("input_ids类型:", type(sample["input_ids"][0]))  # 应该是int

#设置参数
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=4,
#     num_train_epochs=3,
#     learning_rate=5e-5,
#     logging_steps=100,
#     save_steps=500,
#     do_eval=False,        # evaluation_strategy="no",
#     overwrite_output_dir=True,
#     remove_unused_columns=False,
# )
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False,pad_to_multiple_of=8)
class SafeDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # 确保所有输入都是数字列表
        for f in features:
            if any(isinstance(x, str) for x in f["input_ids"]):
                raise ValueError("发现文本数据，请检查预处理流程")
        
        # 转换为张量
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        
        # 填充批次
        batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0)
        }
        
        # 添加labels（语言建模需要）
        batch["labels"] = batch["input_ids"].clone()
        return batch

data_collator = SafeDataCollator(tokenizer=tokenizer, mlm=False)

try:
    test_batch = data_collator([sample])
    print("测试批次形状:", test_batch["input_ids"].shape)
except Exception as e:
    print("Collator测试失败:", str(e))

# #训练
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     data_collator=data_collator,
# )
# print("Start training!")
# trainer.train()

# #保存
# output_dir = "./mimic_finetuned"
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"saved at: {output_dir}")
