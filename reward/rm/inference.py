import transformers
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "Hahmdong/AT-qwen2.5-7b-hhrlhf-5120-rm"
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    device_map="auto",
    num_labels=1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_reward(tokenizer, model, messages):
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        reward = model(**inputs).logits[0]
    return reward.item()

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]

# I'm sorry, but I can't assist with that request.
# The capital of France is Paris.

reward = get_reward(tokenizer, model, messages)
print(messages)
print(reward)