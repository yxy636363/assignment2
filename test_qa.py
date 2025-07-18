from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置
output_dir = "./mimic_finetuned"
qa_test_cases = [
    {"question": "Lactate 5.2 mmol/L in septic shock – how to respond?", "answer": "Indicates tissue hypoperfusion:Repeat lactate in 2 hours. Optimize fluids/vasopressors. Source control (e.g., antibiotics, drainage)"}, 
    {"question": "CRRT machine shows \"TMP >250 mmHg\" – causes?", "answer": "Possible filter clotting: Check anticoagulation (e.g., citrate dose), Increase blood flow rate, Replace circuit if persistent"}, 
    {"question": "Norepinephrine at 0.3 mcg/kg/min with persistent hypotension – next agent?", "answer": "Add vasopressin 0.03 U/min or consider Epinephrine for cardiogenic shock or Angiotensin II in vasodilatory shock"}, 
    # {"question": "", "answer": ""}, 
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
        
        # 计算生成答案的 Perplexity
        answer_inputs = tokenizer(generated_answer, return_tensors="pt")
        with torch.no_grad():
            logits = model(**answer_inputs).logits
            attention_mask = answer_inputs.get("attention_mask", None)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = answer_inputs["input_ids"][:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=tokenizer.pad_token_id
            )
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            if attention_mask is not None:
                valid_tokens = (attention_mask[:, 1:] == 1).view(-1)
                valid_loss = per_token_loss[valid_tokens]
                avg_loss = valid_loss.mean()
            else:
                avg_loss = per_token_loss.mean()
            
            ppl = torch.exp(avg_loss)
        
        print(generated_answer)
        print("given answer: ", case["answer"])
        print("perplexity: ", ppl.item())

# Question Answering对比
print("finetuned model qa:")
evaluate_qa(finetuned_model, finetuned_tokenizer, qa_test_cases)
print("original model qa:")
evaluate_qa(original_model, original_tokenizer, qa_test_cases)
