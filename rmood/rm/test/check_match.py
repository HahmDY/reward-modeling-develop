import os
import json

RMOOD_HOME = os.getenv("RMOOD_HOME")

model_code = "mrm-sft-based"
dataset_name = "alpacafarm"

# filtering test_sft.json to only include items with different responses
test_sft_path = f"{RMOOD_HOME}/datasets/{dataset_name}/test/test_sft.json"
with open(test_sft_path, "r") as f:
    test_sft = json.load(f)

# collect indices of items with different responses
valid_indices = []
for idx, item in enumerate(test_sft):
	if item["response_1"] != item["response_2"]:
		valid_indices.append(idx)

print(f"Total items: {len(test_sft)}")
print(f"Items with different responses: {len(valid_indices)}")
print(f"Filtered out (same responses): {len(test_sft) - len(valid_indices)}")
print()

rewards_result_path = f"{RMOOD_HOME}/datasets/{dataset_name}/rm/test_reward_{model_code}.json"
with open(rewards_result_path, "r") as f:
    rewards_result = json.load(f)

rm_results = []
for idx in valid_indices:
	item = rewards_result[idx]
	if item["reward_1"] > item["reward_2"]:
		rm_results.append(1)
	elif item["reward_1"] < item["reward_2"]:
		rm_results.append(2)
	else:
		rm_results.append(0)

gold_result_path = f"{RMOOD_HOME}/datasets/{dataset_name}/rm/test_reward_gold.json"
with open(gold_result_path, "r") as f:
    gold_result = json.load(f)
    
gold_results = []
for idx in valid_indices:
	item = gold_result[idx]
	if item["reward_1"] > item["reward_2"]:
		gold_results.append(1)
	elif item["reward_1"] < item["reward_2"]:
		gold_results.append(2)
	else:
		gold_results.append(0)
  
# compare how many items are the same in rm_results and gold_results
same_count = 0
for rm_result, gold_result in zip(rm_results, gold_results):
	if rm_result == gold_result:
		same_count += 1

print(f"Model code: {model_code}")
print(f"Same count: {same_count}")
print(f"Total count: {len(rm_results)}")
print(f"Match rate: {same_count / len(rm_results)}")

