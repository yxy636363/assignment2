from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置
output_dir = "./mimic_finetuned"

# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained("gpt2")
original_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# original_tokenizer.pad_token = original_tokenizer.eos_token
if original_tokenizer.pad_token is None:
    original_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    original_model.resize_token_embeddings(len(original_tokenizer))

# 加载微调模型
finetuned_model = AutoModelForCausalLM.from_pretrained(output_dir)
finetuned_tokenizer = AutoTokenizer.from_pretrained(output_dir)
# finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
if finetuned_tokenizer.pad_token is None:
    finetuned_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    finetuned_model.resize_token_embeddings(len(finetuned_tokenizer))

def calculate_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        ppl = torch.exp(torch.nn.functional.cross_entropy(logits[:, :-1], inputs["input_ids"][:, 1:]))
    return ppl.item()

# 在医疗文本上测试
medical_text = "The patient was diagnosed with sepsis and treated with antibiotics."
finetuned_ppl = calculate_ppl(finetuned_model, finetuned_tokenizer, medical_text)
original_ppl = calculate_ppl(original_model, original_tokenizer, medical_text)

print(f"微调模型 PPL: {finetuned_ppl:.2f}, 原始模型 PPL: {original_ppl:.2f}")
