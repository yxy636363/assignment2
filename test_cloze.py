from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置
output_dir = "./mimic_finetuned"
cloze_test_cases = [
    "The patient with [MASK] presented with chest pain and elevated troponin levels.",  # 应填 "STEMI"
    # 更多
]

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

def evaluate_cloze(model, tokenizer, cloze_cases):
    results = []
    for case in cloze_cases:
        inputs = tokenizer(case, return_tensors="pt")
        mask_pos = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_token = torch.argmax(logits[0, mask_pos, :])
            predicted_word = tokenizer.decode(predicted_token)
            prob = torch.nn.functional.softmax(logits[0, mask_pos, :], dim=-1)[predicted_token]
        results.append({
            "sentence": case,
            "predicted_word": predicted_word,
            "confidence": prob.item()
        })
    return results

# 对比测试
finetuned_cloze_results = evaluate_cloze(finetuned_model, finetuned_tokenizer, cloze_test_cases)
original_cloze_results = evaluate_cloze(original_model, original_tokenizer, cloze_test_cases)
