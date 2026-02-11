import os
import json

RMOOD_HOME = os.getenv("RMOOD_HOME")

model_code = "rm"
dataset_name = "alpacafarm"

rewards_result_path = f"{RMOOD_HOME}/datasets/{dataset_name}/rm/test_reward_{model_code}.json"
with open(rewards_result_path, "r") as f:
    rewards_result = json.load(f)

rm_results = []
for item in rewards_result:
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
for item in gold_result:
	if item["reward_1"] > item["reward_2"]:
		gold_results.append(1)
	elif item["reward_1"] < item["reward_2"]:
		gold_results.append(2)
	else:
		gold_results.append(0)
  
# rm_results와 gold_results에 같은 거 얼마나 있는지 비교
same_count = 0
for rm_result, gold_result in zip(rm_results, gold_results):
	if rm_result == gold_result:
		same_count += 1

print(f"Model code: {model_code}")
print(f"Same count: {same_count}")
print(f"Total count: {len(rm_results)}")
print(f"Match rate: {same_count / len(rm_results)}")

