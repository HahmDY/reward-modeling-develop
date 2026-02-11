import os
import json
import random
import numpy as np

TAMPERING_HOME = os.getenv("TAMPERING_HOME")
SEED = 42
random.seed(SEED)

preference_data_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit.json"
output_path = f"{TAMPERING_HOME}/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_explicit_augmentation.json"

with open(preference_data_path, "r") as f:
    preference_data = json.load(f)

original_data = []
augmented_data = []

for i, item in enumerate(preference_data):
	# sample two indices which are not equal to idx
	indices = list(range(len(preference_data)))
	indices.remove(i)
	j, k = random.sample(indices, 2)

	messages = item["messages"]
 
	chosen_i = item["chosen"]
	rejected_i = item["rejected"]
	chosen_j = preference_data[j]["chosen"]
	rejected_j = preference_data[j]["rejected"]
	chosen_k = preference_data[k]["chosen"]
	rejected_k = preference_data[k]["rejected"]

	# original
	original_data.append({
		"messages": messages,
		"chosen": chosen_i,
		"rejected": rejected_i,
		"tie": False,
	})

	# non-contextuals
	augmented_data += [
		# chosen_i vs
		{
			"messages": messages,
			"chosen": chosen_i,
			"rejected": chosen_j,
			"tie": False,
		},
		{
			"messages": messages,
			"chosen": chosen_i,
			"rejected": chosen_k,
			"tie": False,
		},
		{
			"messages": messages,
			"chosen": chosen_i,
			"rejected": rejected_j,
			"tie": False,
		},
		{
			"messages": messages,
			"chosen": chosen_i,
			"rejected": rejected_k,
			"tie": False,
		},
		# rejected_i vs
  		{
			"messages": messages,
			"chosen": rejected_i,
			"rejected": chosen_j,
			"tie": False,
		},
		{
			"messages": messages,
			"chosen": rejected_i,
			"rejected": chosen_k,
			"tie": False,
		},
		{
			"messages": messages,
			"chosen": rejected_i,
			"rejected": rejected_j,
			"tie": False,
		},
		{
			"messages": messages,
			"chosen": rejected_i,
			"rejected": rejected_k,
			"tie": False,
		}
	]
 
	# neutrals
	augmented_data += [
		{
			"messages": messages,
			"chosen": chosen_j,
			"rejected": chosen_k,
			"tie": True,
		},
		{
			"messages": messages,
			"chosen": chosen_j,
			"rejected": rejected_j,
			"tie": True,
		},
		{
			"messages": messages,
			"chosen": chosen_j,
			"rejected": rejected_k,
			"tie": True,
		},
		{
			"messages": messages,
			"chosen": chosen_k,
			"rejected": rejected_k,
			"tie": True,
		},
		{
			"messages": messages,
			"chosen": chosen_k,
			"rejected": rejected_j,
			"tie": True,
		},
		{
			"messages": messages,
			"chosen": rejected_j,
			"rejected": rejected_k,
			"tie": True,
		},
	]
 
# select half of the augmented data
augmented_data = random.sample(augmented_data, len(augmented_data) // 2)

print("Original data size: ", len(original_data))
print("Augmented data size: ", len(augmented_data))

with open(output_path, "w") as f:
    json.dump(augmented_data, f, indent=4)