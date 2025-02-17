import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "C:\\Repo\\Local-LLMs\\SmolLm2\\SmolLM2-1.7B-Instruct"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "What is the capital of Bahrain."}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(device)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

outputs = model.generate(
    input_ids, 
    attention_mask = attention_mask,
    max_new_tokens=50, 
    temperature=0.2, 
    top_p=0.9, 
    do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))