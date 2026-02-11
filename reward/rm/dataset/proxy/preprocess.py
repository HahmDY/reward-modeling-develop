import os
import json
import argparse
import copy

PENALTY_HOME = os.getenv("PENALTY_HOME")

def data_preprocess(args):
    prompts_path = args.prompts_path
    responses_path = args.responses_path
    rewards_path = args.rewards_path
    indices_path = args.indices_path
    
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    with open(responses_path, "r") as f:
        responses = json.load(f)
    with open(rewards_path, "r") as f:
        rewards = json.load(f)
    with open(indices_path, "r") as f:
        indices = json.load(f)
        
    implicit_processed_dataset = {
        "chosen": [],
        "rejected": []
    }
    
    for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards)):
        if i not in indices:
            continue
        
        if reward[0] > reward[1]:
            win_label = "response_1"
            lose_label = "response_2"
        elif reward[0] < reward[1]:
            win_label = "response_2"
            lose_label = "response_1"
        else:
            continue
        
        win_response = response[win_label]
        lose_response = response[lose_label]
        chosen_messages = copy.deepcopy(prompt["messages"]) + [{"role": "assistant", "content": win_response}]
        rejected_messages = copy.deepcopy(prompt["messages"]) + [{"role": "assistant", "content": lose_response}]
        implicit_processed_dataset["chosen"].append(chosen_messages)
        implicit_processed_dataset["rejected"].append(rejected_messages)
        
    print(len(implicit_processed_dataset["chosen"]), len(implicit_processed_dataset["rejected"]))
    
    with open(args.target_path, "w") as f:
        json.dump(implicit_processed_dataset, f, ensure_ascii=False, indent=4)
    
    
if __name__ == "__main__":
    dataset_name = "alpacafarm"
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", type=str, default=f"{PENALTY_HOME}/datasets/{dataset_name}/rm/rm_prompts.json")
    parser.add_argument("--responses_path", type=str, default=f"{PENALTY_HOME}/datasets/{dataset_name}/rm/rm_sft.json")
    parser.add_argument("--rewards_path", type=str, default=f"{PENALTY_HOME}/datasets/{dataset_name}/rm/rm_rewards.json")
    parser.add_argument("--indices_path", type=str, default=f"{PENALTY_HOME}/datasets/{dataset_name}/rm/rm_indices.json")
    parser.add_argument("--target_path", type=str, default=f"{PENALTY_HOME}/datasets/{dataset_name}/rm/rm_implicit.jsonl")
    args = parser.parse_args()
    data_preprocess(args)