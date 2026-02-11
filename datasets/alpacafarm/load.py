import os
import json
import random
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset

from prompts import PROMPTS_NOINPUTS, PROMPTS_INPUTS

RMOOD_HOME = os.getenv("RMOOD_HOME")

"""
Warning: this script only works for datasets<3.0.0 version.
To match the version, run the following command:
pip install 'datasets<3.0.0'

After running this script, roll back to the original version:
pip install datasets --upgrade
"""


def parse_dataset_instr(instruction, input, output, include_output=True):
    results = []
    for _inst, _i, _o in zip(instruction, input, output):
        if len(_i) == 0:
            prompt = deepcopy(PROMPTS_NOINPUTS).format(instruction=_inst)
        else:
            prompt = deepcopy(PROMPTS_INPUTS).format(instruction=_inst, input=_i)
        
        if include_output:
            results.append({
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": _o}
                ]
            })
        else:
            results.append({
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            })
            
    return results

def parse_dataset_pref(instruction, input, output_1, output_2, preference):
    results_explicit = []
    results_implicit = {
        "chosen": [],
        "rejected": [],
    }
    
    for _inst, _i, _o1, _o2, _p in zip(instruction, input, output_1, output_2, preference):
        if len(_i) == 0:
            prompt = deepcopy(PROMPTS_NOINPUTS).format(instruction=_inst)
        else:
            prompt = deepcopy(PROMPTS_INPUTS).format(instruction=_inst, input=_i)
        
        selected = _o1 if _p == 1 else _o2
        rejected = _o2 if _p == 1 else _o1
        
        # explicit form
        results_explicit.append({
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            "chosen": selected,
            "rejected": rejected,
        })
        
        results_implicit["chosen"].append([
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": selected},
        ])
        results_implicit["rejected"].append([
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ])
        
    return results_explicit, results_implicit


def process_dataset(category, split, save_name, include_output, start_idx=None, end_idx=None):
    # load dataset
    dataset = load_dataset("tatsu-lab/alpaca_farm", category, trust_remote_code=True)[split]
    if category == "alpaca_instructions":
        instruciton, input, output = dataset["instruction"], dataset["input"], dataset["output"]
        if start_idx is not None or end_idx is not None:
            instruciton = instruciton[start_idx:end_idx]
            input = input[start_idx:end_idx]
            output = output[start_idx:end_idx]
    elif category == "alpaca_gpt4_preference":
        instruciton, input, output_1, output_2, preference = dataset["instruction"], dataset["input"], dataset["output_1"], dataset["output_2"], dataset["preference"]
        if start_idx is not None or end_idx is not None:
            instruciton = instruciton[start_idx:end_idx]
            input = input[start_idx:end_idx]
            output_1 = output_1[start_idx:end_idx]
            output_2 = output_2[start_idx:end_idx]
            preference = preference[start_idx:end_idx]
            
    # make directory
    if not os.path.exists(f"{RMOOD_HOME}/datasets/alpacafarm/{save_name[0]}"):
        os.makedirs(f"{RMOOD_HOME}/datasets/alpacafarm/{save_name[0]}")
        
    # save dataset (instructions)
    if category == "alpaca_instructions":
        dataset = parse_dataset_instr(instruciton, input, output, include_output)
        
        if save_name[1] == "sft":
            with open(f"{RMOOD_HOME}/datasets/alpacafarm/{save_name[0]}/{save_name[1]}.jsonl", "w") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(f"{RMOOD_HOME}/datasets/alpacafarm/{save_name[0]}/{save_name[1]}.json", "w") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)

    return dataset


if __name__ == "__main__":
    process_dataset("alpaca_instructions", "sft", ["sft", "sft"], True)
    
    process_dataset("alpaca_instructions", "preference", ["rm", "rm_prompts"], False) # prompts for proxy RM
    
    process_dataset("alpaca_instructions", "unlabeled", ["rl", "rl_prompts"], False)
    
    # Split val dataset: first 1000 for test, rest for val
    process_dataset("alpaca_instructions", "val", ["test", "test_prompts"], False, start_idx=0, end_idx=1000)
    # process_dataset("alpaca_instructions", "val", ["val", "val"], True, start_idx=1000, end_idx=None)
    process_dataset("alpaca_instructions", "val", ["val", "val_prompts"], False, start_idx=1000, end_idx=None)