import os
import json
import random

PENALTY_HOME = os.getenv("PENALTY_HOME")

dataset_path = f"{PENALTY_HOME}/datasets/alpacafarm/rm/rm_prompts.json"
indices_path = f"{PENALTY_HOME}/datasets/alpacafarm/rm/rm_indices.json"

with open(dataset_path, "r") as f:
    dataset = json.load(f)

sampling_ratio = 1.0
indices = random.sample(list(range(len(dataset))), int(len(dataset) * sampling_ratio))

with open(indices_path, "w") as f:
    json.dump(indices, f)