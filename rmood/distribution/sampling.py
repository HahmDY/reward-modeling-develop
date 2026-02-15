import os
import json
import argparse
from copy import deepcopy
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from multiprocessing import Process, set_start_method, Queue
from typing import List

RMOOD_HOME = os.getenv("RMOOD_HOME")

def get_resp_keys(num_responses: int):
    return tuple(f"response_{i+1}" for i in range(num_responses))

def has_all_responses(item: dict, num_responses: int) -> bool:
    resp_keys = get_resp_keys(num_responses)
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

def merge_source_and_existing(source: list, existing: list, num_responses: int):
    merged = []
    resp_keys = get_resp_keys(num_responses)
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
    num_responses: int,
    save_every: int = 1,
    tqdm_position: int = 0,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from tampering.utils.llm import VLLM

    indices = shard_indices(len(source_data), rank, world)

    data = merge_source_and_existing(source_data, existing_data, num_responses)

    partial_out = {}

    pbar = tqdm(indices, total=len(indices), desc=f"GPU{gpu_id}", position=tqdm_position, leave=True)
    llm = None
    try:
        llm = VLLM(model_name=model_name)
        for cnt, i in enumerate(pbar, start=1):
            item = data[i]
            if has_all_responses(item, num_responses):
                pbar.set_postfix_str(f"skip {i}")
                continue

            messages = [{"role": "system", "content": ""}] + deepcopy(item["messages"])
            
            temperature = 1.0
            top_p = 1.0
            responses = []
            
            for j in range(num_responses):
                response = llm.generate(messages, temperature=temperature, max_new_tokens=1024, top_p=top_p)
                responses.append(response)

            new_item = {f"response_{j+1}": responses[j] for j in range(num_responses)}
            partial_out[i] = new_item

            if save_every and (cnt % save_every == 0):
                save_partial(part_path, partial_out)
                pbar.set_postfix_str(f"saved {len(partial_out)} items")
    finally:
        save_partial(part_path, partial_out)

def merge_parts_into_target(source_data: list, existing_data: list, part_paths: List[str], target_path: str, num_responses: int, original_indices: List[int] = None):
    merged = merge_source_and_existing(source_data, existing_data, num_responses)

    for pp in part_paths:
        part = load_json_if_exists(pp)
        if not part:
            continue

        for k, v in part.items():
            idx = int(k)
            merged[idx] = v

    # If original_indices is provided, save each index to a separate file
    if original_indices:
        for i, orig_idx in enumerate(original_indices):
            single_file_path = target_path.replace("_PLACEHOLDER", f"_{orig_idx}")
            atomic_write_json([merged[i]], single_file_path)
    else:
        atomic_write_json(merged, target_path)

def main_parallel(gpus: List[int], model_name: str, source_data: list, target_path: str, num_responses: int = 16, save_every: int = 1, original_indices: List[int] = None):

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
                num_responses=num_responses,
                save_every=save_every,
                tqdm_position=rank,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    merge_parts_into_target(source_data, existing_data, part_paths, target_path, num_responses, original_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        default="Hahmdong/AT-qwen2.5-7b-hhrlhf-5120-sft-ai-b3s3-ver17")
    parser.add_argument("--source_path", type=str, required=True,
                        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/rm_prompts.json")
    parser.add_argument("--indices", type=str, required=True, help="comma-separated indices to process, e.g. '0,1,2'")
    parser.add_argument("--num_responses", type=int, default=16, help="number of responses to generate per item")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--gpus", type=str, required=True, help="comma-separated GPU ids, e.g. '6'", default="6,7")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus like '0,1,2'.")
    
    with open(args.source_path, "r") as f:
        source_data_raw = json.load(f)
    indices = [int(i) for i in args.indices.split(",")]
    source_data = [source_data_raw[i] for i in indices]
    
    model_name_clean = args.model_name.replace("/", "_")
    # Use PLACEHOLDER in target path - will be replaced per index in merge_parts_into_target
    target_path = f"{RMOOD_HOME}/datasets/alpacafarm/distribution/{model_name_clean}/responses_PLACEHOLDER.json"

    print(f"Processing {len(indices)} indices: {indices}")
    main_parallel(gpus, args.model_name, source_data, target_path, args.num_responses, args.save_every, original_indices=indices)