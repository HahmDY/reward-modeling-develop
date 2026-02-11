import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from multiprocessing import Process, set_start_method, Queue
from typing import List

RMOOD_HOME = os.getenv("RMOOD_HOME")

# sample 16 responses
RESP_KEYS = ("response_1", "response_2", "response_3", "response_4", "response_5", "response_6", "response_7", "response_8", "response_9", "response_10", "response_11", "response_12", "response_13", "response_14", "response_15", "response_16")

def has_all_responses(item: dict) -> bool:
    return all(k in item and item[k] not in (None, "") for k in RESP_KEYS)

def atomic_write_json(obj, path: str):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    with NamedTemporaryFile("w", dir=dirpath, delete=False) as tf:
        json.dump(obj, tf, indent=4, ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_if_exists(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None

def pad_or_trim(existing: list, target_len: int):
    if existing is None:
        return None
    if not isinstance(existing, list):
        return None
    if len(existing) == target_len:
        return existing
    out = list(existing)
    if len(out) < target_len:
        out.extend({} for _ in range(target_len - len(out)))
    else:
        out = out[:target_len]
    return out

def merge_source_and_existing(source: list, existing: list):
    merged = []
    for i in range(len(source)):
        base = deepcopy(source[i])
        if existing and i < len(existing) and isinstance(existing[i], dict):
            prev = existing[i]
            for k in RESP_KEYS:
                if k in prev:
                    base[k] = prev[k]
        merged.append(base)
    return merged


def parse_gpus(gpus_arg: str) -> List[int]:
    # "0,2,5" -> [0,2,5]
    parts = [p.strip() for p in gpus_arg.split(",") if p.strip() != ""]
    return [int(p) for p in parts]

def shard_indices(total: int, rank: int, world: int) -> List[int]:
    return [i for i in range(total) if i % world == rank]

def save_partial(part_path: str, data_slice: dict):
    atomic_write_json(data_slice, part_path)

def worker(
    gpu_id: int,
    rank: int,
    world: int,
    model_name: str,
    source_data: list,
    existing_data: list,
    part_path: str,
    save_every: int = 1,
    tqdm_position: int = 0,
    seed: int = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from rmood.utils.llm import VLLM

    indices = shard_indices(len(source_data), rank, world)

    data = merge_source_and_existing(source_data, existing_data)

    partial_out = {}

    pbar = tqdm(indices, total=len(indices), desc=f"GPU{gpu_id}", position=tqdm_position, leave=True)
    llm = None
    try:
        llm = VLLM(model_name=model_name)
        for cnt, i in enumerate(pbar, start=1):
            item = data[i]
            if has_all_responses(item):
                pbar.set_postfix_str(f"skip {i}")
                continue

            messages = [{"role": "system", "content": ""}] + deepcopy(item["messages"])
            
            temperature = 1.0
            top_p = 0.9
            responses = []
            
            for j in range(16):
                item_seed = seed + i * 16 + j if seed is not None else None
                response = llm.generate(messages, temperature=temperature, max_new_tokens=1024, top_p=top_p, seed=item_seed)
                responses.append(response)

            new_item = {
                "response_1": responses[0],
                "response_2": responses[1],
                "response_3": responses[2],
                "response_4": responses[3],
                "response_5": responses[4],
                "response_6": responses[5],
                "response_7": responses[6],
                "response_8": responses[7],
                "response_9": responses[8],
                "response_10": responses[9],
                "response_11": responses[10],
                "response_12": responses[11],
                "response_13": responses[12],
                "response_14": responses[13],
                "response_15": responses[14],
                "response_16": responses[15]
            }
            partial_out[i] = new_item

            if save_every and (cnt % save_every == 0):
                save_partial(part_path, partial_out)
                pbar.set_postfix_str(f"saved {len(partial_out)} items")
    finally:
        save_partial(part_path, partial_out)

def merge_parts_into_target(source_data: list, existing_data: list, part_paths: List[str], target_path: str):
    merged = merge_source_and_existing(source_data, existing_data)

    for pp in part_paths:
        part = load_json_if_exists(pp)
        if not part:
            continue

        for k, v in part.items():
            idx = int(k)
            merged[idx] = v

    atomic_write_json(merged, target_path)

def main_parallel(gpus: List[int], model_name: str, source_path: str, target_path: str, save_every: int = 1, seed: int = None):
    with open(source_path, "r") as f:
        source_data = json.load(f)

    existing_data = load_json_if_exists(target_path)
    existing_data = pad_or_trim(existing_data, len(source_data))

    part_paths = [f"{target_path}.gpu{gid}.part.json" for gid in gpus]

    procs = []

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    world = len(gpus)
    for rank, gpu_id in enumerate(gpus):
        p = Process(
            target=worker,
            kwargs=dict(
                gpu_id=gpu_id,
                rank=rank,
                world=world,
                model_name=model_name,
                source_data=source_data,
                existing_data=existing_data,
                part_path=part_paths[rank],
                save_every=save_every,
                tqdm_position=rank,
                seed=seed,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    merge_parts_into_target(source_data, existing_data, part_paths, target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        default="Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft")
    parser.add_argument("--source_path", type=str, required=True,
                        default=f"{RMOOD_HOME}/datasets/alpacafarm/test/test_prompts.json")
    parser.add_argument("--target_path", type=str, required=True,
                        default=f"{RMOOD_HOME}/datasets/alpacafarm/bon/test_sft_sampled.json")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--gpus", type=str, required=True, help="comma-separated GPU ids, e.g. '6'", default="6,7")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus like '0,1,2'.")

    main_parallel(gpus, args.model_name, args.source_path, args.target_path, args.save_every, args.seed)