import os
import json
import random
import numpy as np

TAMPERING_HOME = os.getenv("TAMPERING_HOME")
SEED = 42
random.seed(SEED)

original_data_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit.json"
augmented_data_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_augmentation.json"
rewards_data_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_augmentation_rewards.json"

filtered_data_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_rrm.json"

with open(original_data_path, "r") as f:
    original_dataset = json.load(f)
with open(augmented_data_path, "r") as f:
    dataset = json.load(f)
    dataset = random.sample(dataset, len(dataset)//2)
with open(rewards_data_path, "r") as f:
    rewards = json.load(f)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


filtered_dataset = []

for data, reward in zip(dataset, rewards):
    
    chosen_reward = reward["chosen_reward"]
    rejected_reward = reward["rejected_reward"]
    wining_probability = sigmoid(chosen_reward - rejected_reward)
    
    gt_prob = 0.5 if data["tie"] else 1
    if np.abs(wining_probability - gt_prob) >= 0.2:
        filtered_dataset.append(data)
   
processed_original_dataset = []
for data in original_dataset:
    data["tie"] = False
    processed_original_dataset.append(data)

print("Processed original dataset size: ", len(processed_original_dataset))
print("Filtered augmented dataset size: ", len(filtered_dataset))

rrm_dataset = processed_original_dataset + filtered_dataset
random.shuffle(rrm_dataset)

print("RRM dataset size: ", len(rrm_dataset))

with open(filtered_data_path, "w") as f:
    json.dump(rrm_dataset, f, indent=4)