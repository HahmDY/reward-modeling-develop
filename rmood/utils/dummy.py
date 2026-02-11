import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

# export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name = "Hahmdong/AT-qwen2.5-7b-hhrlhf-5120-sft-b3s3"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded on GPU: {next(model.parameters()).device}")
print("Starting infinite inference loop...")

iteration = 0
while True:
    time.sleep(300) # 5 minutes
    
    iteration += 1
    
    prompt = f"Explain the concept of artificial intelligence in detail. Iteration {iteration}."
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*50}")
    print(f"Iteration: {iteration}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Response length: {len(response)} characters")
    print(f"{'='*50}\n")
