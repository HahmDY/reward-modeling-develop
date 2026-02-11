import os
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAMPERING_HOME = os.getenv("TAMPERING_HOME")

model_name = "Hahmdong/AT-qwen2.5-7b-hhrlhf-5120-ver6-rm"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="auto",
    num_labels=1,
    torch_dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

source_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_augmentation.json"
target_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_augmentation_rewards.json"

with open(source_path, "r") as f:
    dataset = json.load(f)

def build_chat_text(messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

def make_pair_texts(data_item):
    base = [{"role": "system", "content": ""}] + data_item["messages"]
    chosen_msgs   = base + [{"role": "assistant", "content": data_item["chosen"]}]
    rejected_msgs = base + [{"role": "assistant", "content": data_item["rejected"]}]
    return build_chat_text(chosen_msgs), build_chat_text(rejected_msgs)

def batched_rewards(texts, batch_size=32, progress_desc="Scoring"):
    all_scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc=progress_desc):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=5120
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

        scores = logits.squeeze(-1).detach().float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        all_scores.extend(scores)

    return all_scores

pair_texts = []
flat_texts = []
for item in dataset:
    c_text, r_text = make_pair_texts(item)
    pair_texts.append((c_text, r_text))
    flat_texts.append(c_text)
    flat_texts.append(r_text)

BATCH_SIZE = 128
all_scores = batched_rewards(flat_texts, batch_size=BATCH_SIZE, progress_desc="Batch scoring")

rewards = []
for idx in range(len(pair_texts)):
    chosen_reward   = all_scores[2*idx]
    rejected_reward = all_scores[2*idx + 1]
    rewards.append({
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward
    })

with open(target_path, "w") as f:
    json.dump(rewards, f, indent=4)

print(f"Done. Saved {len(rewards)} reward pairs to {target_path}")
