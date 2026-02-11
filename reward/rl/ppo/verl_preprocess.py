import re
import os
import datasets
from datasets import load_dataset, Features, Sequence, Value

PENALTY_HOME = os.getenv("PENALTY_HOME")

def make_map_fn(split):

    def process_fn(example, idx):
        prompt = example.pop("messages")

        data = {
            "prompt": prompt,
            "raw_prompt": prompt,
            "reward_model": {
                "style": "model"
            }
        }
        return data

    return process_fn

dataset = "alpacafarm"
train_dataset_path = f"{PENALTY_HOME}/datasets/{dataset}/rl/rl_prompts.json"
val_dataset_path = f"{PENALTY_HOME}/datasets/{dataset}/val/val_prompts.json"

train_dataset_target_path = f"{PENALTY_HOME}/datasets/{dataset}/rl/rl_prompts.parquet"
val_dataset_target_path = f"{PENALTY_HOME}/datasets/{dataset}/val/val_prompts.parquet"

train_dataset = load_dataset(
	"json",
	data_files=train_dataset_path,
	split="train",
)
val_dataset = load_dataset(
	"json",
	data_files=val_dataset_path,
	split="train",
)

train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
val_dataset = val_dataset.map(make_map_fn("val"), with_indices=True)

print(train_dataset.features)

train_dataset.to_parquet(train_dataset_target_path)
val_dataset.to_parquet(val_dataset_target_path)