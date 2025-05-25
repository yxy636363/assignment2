from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置
output_dir = "./mimic_finetuned"
qa_test_cases = [
    {"question": "What are the symptoms of pneumonia?", "answer": "Symptoms include fever, cough, and shortness of breath."},
    # 更多
]
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

def evaluate_qa(model, tokenizer, qa_cases):
    results = []
    for case in qa_cases:
        input_text = f"Question: {case['question']}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # 生成答案
        outputs = model.generate(**inputs, max_length=100)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_answer)
        
        # 计算生成答案的 Perplexity
        answer_inputs = tokenizer(generated_answer, return_tensors="pt")
        with torch.no_grad():
            logits = model(**answer_inputs).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = answer_inputs["input_ids"][:, 1:].contiguous()
            ppl = torch.exp(torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                              shift_labels.view(-1),
                                                              ignore_index=tokenizer.pad_token_id))
        results.append({
            "question": case["question"],
            "generated_answer": generated_answer,
            "perplexity": ppl.item()
        })
    return generated_answer

# Question Answering对比
print("finetuned model qa:")
finetuned_qa_results = evaluate_qa(finetuned_model, finetuned_tokenizer, qa_test_cases)
print("original model qa:")
original_qa_results = evaluate_qa(original_model, original_tokenizer, qa_test_cases)












