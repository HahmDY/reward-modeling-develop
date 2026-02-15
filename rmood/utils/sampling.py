import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from multiprocessing import Process, set_start_method, Queue
from typing import List
from rmood.utils.llm import VLLM

RMOOD_HOME = os.getenv("RMOOD_HOME")

def get_response_keys(num_responses: int) -> tuple:
    return tuple(f"response_{i+1}" for i in range(num_responses))

def has_all_responses(item: dict, resp_keys: tuple) -> bool:
    return all(k in item and item[k] not in (None, "") for k in resp_keys)

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

def merge_source_and_existing(source: list, existing: list, resp_keys: tuple):
    merged = []
    for i in range(len(source)):
        base = deepcopy(source[i])
        if existing and i < len(existing) and isinstance(existing[i], dict):
            prev = existing[i]
            for k in resp_keys:
                if k in prev:
                    base[k] = prev[k]
        merged.append(base)
    return merged


def parse_gpus(gpus_arg: str) -> List[int]:
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
    num_responses: int = 2,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    save_every: int = 1,
    tqdm_position: int = 0,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    indices = shard_indices(len(source_data), rank, world)
    
    resp_keys = get_response_keys(num_responses)
    data = merge_source_and_existing(source_data, existing_data, resp_keys)

    partial_out = {}

    pbar = tqdm(indices, total=len(indices), desc=f"GPU{gpu_id}", position=tqdm_position, leave=True)
    llm = None
    try:
        llm = VLLM(model_name=model_name)
        for cnt, i in enumerate(pbar, start=1):
            item = data[i]
            if has_all_responses(item, resp_keys):
                pbar.set_postfix_str(f"skip {i}")
                continue

            messages = deepcopy(item["messages"])

            # Generate num_responses responses
            responses = {}
            for resp_idx in range(num_responses):
                response_key = f"response_{resp_idx+1}"
                response = llm.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens, top_p=top_p)
                responses[response_key] = response

            partial_out[i] = responses

            if save_every and (cnt % save_every == 0):
                save_partial(part_path, partial_out)
                pbar.set_postfix_str(f"saved {len(partial_out)} items")
    finally:
        save_partial(part_path, partial_out)

def merge_parts_into_target(source_data: list, existing_data: list, part_paths: List[str], target_path: str, resp_keys: tuple):
    merged = merge_source_and_existing(source_data, existing_data, resp_keys)

    for pp in part_paths:
        part = load_json_if_exists(pp)
        if not part:
            continue

        for k, v in part.items():
            idx = int(k)
            merged[idx] = v

    atomic_write_json(merged, target_path)

def main_parallel(
    gpus: List[int], 
    model_name: str, 
    source_path: str, 
    target_path: str, 
    num_responses: int = 2,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    top_p: float = 1.0,
    save_every: int = 1
):
    with open(source_path, "r") as f:
        source_data = json.load(f)

    existing_data = load_json_if_exists(target_path)
    existing_data = pad_or_trim(existing_data, len(source_data))
    
    resp_keys = get_response_keys(num_responses)

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
                num_responses=num_responses,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                save_every=save_every,
                tqdm_position=rank,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    merge_parts_into_target(source_data, existing_data, part_paths, target_path, resp_keys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        default="")
    parser.add_argument("--source_path", type=str, required=True,
                        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/rm.json")
    parser.add_argument("--target_path", type=str, required=True,
                        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/rm_sft.json")
    parser.add_argument("--num_responses", type=int, default=2,
                        help="Number of responses to generate per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--gpus", type=str, required=True, help="comma-separated GPU ids, e.g. '6'")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus like '0,1,2'.")

    main_parallel(
        gpus, 
        args.model_name, 
        args.source_path, 
        args.target_path, 
        num_responses=args.num_responses,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        save_every=args.save_every
    )
