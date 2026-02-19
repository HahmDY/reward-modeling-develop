import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RMOOD_HOME = os.getenv("RMOOD_HOME")

dataset_name = "alpacafarm"
custom_rm = True # True: custom RM, False: gold RM

if custom_rm:
    model_code = "mrm-sft-based"
    model_name = f"Hahmdong/RMOOD-qwen3-4b-alpacafarm-{model_code}"
else:
    model_code = "gold"
    model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="auto",
    num_labels=1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    ignore_mismatched_sizes=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts_path = f"{RMOOD_HOME}/datasets/{dataset_name}/test/test_prompts.json"
responses_path = f"{RMOOD_HOME}/datasets/{dataset_name}/test/test_sft.json"
target_path = f"{RMOOD_HOME}/datasets/{dataset_name}/rm/test_reward_{model_code}.json"

with open(prompts_path, "r") as f:
    prompts = json.load(f)
with open(responses_path, "r") as f:
    responses = json.load(f)

def get_reward(tokenizer, model, messages):
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        reward = model(**inputs).logits[0]
    return reward.item()

dataset = []
for prompt, response in tqdm(zip(prompts, responses), total=len(prompts), desc="Processing"):
    data = {"messages": prompt["messages"]}
    
	# calculate reward for each response
    for key in ["response_1", "response_2"]:
        if key in response:
            messages = data["messages"] + [{"role": "assistant", "content": response[key]}]
            reward_value = get_reward(tokenizer, model, messages)
            data[key] = response[key]
            data[f"reward_{key.split('_')[1]}"] = reward_value
    
    dataset.append(data)

    # save results
    with open(target_path, "w") as f:
        json.dump(dataset, f, indent=4)

print(f"\nResults saved to: {target_path}")
print(f"Total samples processed: {len(dataset)}")