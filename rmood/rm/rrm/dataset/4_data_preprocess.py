import os
import json

TAMPERING_HOME = os.getenv("TAMPERING_HOME")

explicit_dataset_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_rrm.json"
implicit_dataset_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_implicit_rrm.jsonl"

with open(explicit_dataset_path, "r") as f:
    explicit_dataset = json.load(f)

chosen = []
rejected = []
tie_label = []

for item in explicit_dataset:
    messages = item["messages"]
    chosen_response = item["chosen"]
    rejected_response = item["rejected"]
    
    chosen_conversation = [{"role": "system", "content": ""}] + messages + [{"role": "assistant", "content": chosen_response}]
    rejected_conversation = [{"role": "system", "content": ""}] + messages + [{"role": "assistant", "content": rejected_response}]
    
    tie_label.append(item["tie"])
    chosen.append(chosen_conversation)
    rejected.append(rejected_conversation)

print(len(chosen))

implicit_dataset = {
	"chosen": chosen,
	"rejected": rejected,
	"tie_label": tie_label
}

with open(implicit_dataset_path, "w") as f:
    json.dump(implicit_dataset, f, indent=4)