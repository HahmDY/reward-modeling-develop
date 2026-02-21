import os
import json
from typing import List, Dict, Tuple, Optional
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


@torch.no_grad()
def compute_kl_loss_static(
    actor_model_name: str,
    ref_model_name: str,
    prompts: List[List[Dict[str, str]]],   # [{"role":..., "content":...}, ...]
    responses: List[str],                   # pre-generated response strings
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    batch_size: int = 8,
    loss_agg_mode: str = "token-mean",      # "token-mean" | "sequence-mean"
    show_progress: bool = True,
    log_every: int = 10,
    save_per_sample_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
) -> Tuple[float, Dict]:

    assert len(prompts) == len(responses), \
        f"prompts({len(prompts)}) and responses({len(responses)}) must have the same length"

    tok = AutoTokenizer.from_pretrained(
        actor_model_name, use_fast=True, trust_remote_code=True, padding_side="left"
    )
    tok_ref = AutoTokenizer.from_pretrained(
        ref_model_name, use_fast=True, trust_remote_code=True, padding_side="left"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok_ref.pad_token is None:
        tok_ref.pad_token = tok_ref.eos_token

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    actor = AutoModelForCausalLM.from_pretrained(actor_model_name, torch_dtype=dtype, device_map="auto")
    ref   = AutoModelForCausalLM.from_pretrained(ref_model_name,   torch_dtype=dtype, device_map="auto")
    actor.eval(); ref.eval()

    def left_pad_batch(seqs: List[List[int]], pad_id: int):
        lens = [len(s) for s in seqs]
        max_len = max(lens) if lens else 0
        B = len(seqs)
        ids  = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
        mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for r, (s, L) in enumerate(zip(seqs, lens)):
            if L > 0:
                ids[r, -L:] = torch.tensor(s, dtype=torch.long, device=device)
                attn[r, -L:] = 1
        return ids, attn, lens

    total_masked_kl_sum = 0.0
    total_masked_tokens = 0
    total_seq_kl_sum    = 0.0
    total_seq_count     = 0
    total_gen_len       = 0
    num_examples        = 0
    num_zero_resp       = 0

    per_sample_records: List[Dict] = []

    N = len(prompts)
    num_batches = (N + batch_size - 1) // batch_size
    iterator = range(0, N, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches,
                        desc=f"KL on static responses ({loss_agg_mode})")

    for b_idx, i in enumerate(iterator, start=1):
        batch_prompts   = prompts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]

        # tokenize: prompt ids (with generation prompt) + response ids (no special tokens)
        full_ids_list: List[List[int]] = []
        prompt_lens_list: List[int]    = []
        resp_lens_list: List[int]      = []

        for chat, resp_text in zip(batch_prompts, batch_responses):
            prompt_ids = tok.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True
            )
            resp_ids = tok.encode(resp_text, add_special_tokens=False)
            # append EOS to mark end of response (mirrors generation behavior)
            if tok.eos_token_id is not None and (not resp_ids or resp_ids[-1] != tok.eos_token_id):
                resp_ids = resp_ids + [tok.eos_token_id]

            full_ids_list.append(prompt_ids + resp_ids)
            prompt_lens_list.append(len(prompt_ids))
            resp_lens_list.append(len(resp_ids))

        # build resp_mask per sequence, then left-pad together
        resp_mask_list: List[List[int]] = [
            [0] * pl + [1] * rl
            for pl, rl in zip(prompt_lens_list, resp_lens_list)
        ]

        # manual left-pad (full_ids and resp_mask together)
        max_len = max(len(s) for s in full_ids_list)
        B = len(full_ids_list)
        full_input = torch.full((B, max_len), tok.pad_token_id, dtype=torch.long, device=device)
        full_attn  = torch.zeros((B, max_len), dtype=torch.long, device=device)
        resp_mask  = torch.zeros((B, max_len), dtype=torch.long, device=device)

        for r, (ids, rmask) in enumerate(zip(full_ids_list, resp_mask_list)):
            L = len(ids)
            if L > 0:
                full_input[r, -L:] = torch.tensor(ids,   dtype=torch.long, device=device)
                full_attn[r,  -L:] = 1
                resp_mask[r,  -L:] = torch.tensor(rmask, dtype=torch.long, device=device)

        resp_lens_tensor = resp_mask.sum(dim=1)   # [B]

        # filter zero-response rows
        valid_rows = (resp_lens_tensor > 0).nonzero(as_tuple=False).squeeze(1).tolist()
        num_zero_resp += int((resp_lens_tensor == 0).sum().item())

        if len(valid_rows) == 0:
            interim = (total_masked_kl_sum / max(total_masked_tokens, 1)) if loss_agg_mode == "token-mean" \
                      else (total_seq_kl_sum / max(total_seq_count, 1))
            if show_progress:
                iterator.set_postfix({"batch_kl": "n/a", "interim_kl": f"{interim:.6f}"})
            continue

        full_input = full_input[valid_rows]
        full_attn  = full_attn[valid_rows]
        resp_mask  = resp_mask[valid_rows]
        resp_lens  = resp_lens_tensor[valid_rows]

        total_gen_len += int(resp_lens.sum().item())
        num_examples  += len(valid_rows)

        out_actor = actor(input_ids=full_input, attention_mask=full_attn, use_cache=False)
        out_ref   = ref(  input_ids=full_input, attention_mask=full_attn, use_cache=False)

        logits_actor = out_actor.logits[:, :-1, :]   # [B, T-1, V]
        logits_ref   = out_ref.logits[:,   :-1, :]   # [B, T-1, V]
        mask_tok     = resp_mask[:, 1:]               # [B, T-1]  shift: predict next token

        logp_actor = torch.log_softmax(logits_actor, dim=-1)
        logp_ref   = torch.log_softmax(logits_ref,   dim=-1)
        p_actor    = logp_actor.exp()

        kl_per_token = (p_actor * (logp_actor - logp_ref)).sum(dim=-1)   # [B, T-1]

        per_sample_tok = mask_tok.sum(dim=1).clamp_min(1)   # [B]

        if loss_agg_mode == "token-mean":
            masked_kl_sum = (kl_per_token * mask_tok).sum().item()
            num_masked    = int(mask_tok.sum().item())
            batch_kl      = float(masked_kl_sum / max(num_masked, 1))
            total_masked_kl_sum += masked_kl_sum
            total_masked_tokens += num_masked
            interim_kl = float(total_masked_kl_sum / max(total_masked_tokens, 1))
            per_sample_kl_for_save = (kl_per_token * mask_tok).sum(dim=1) / per_sample_tok

        elif loss_agg_mode == "sequence-mean":
            per_sample_kl = (kl_per_token * mask_tok).sum(dim=1) / per_sample_tok
            batch_kl      = per_sample_kl.mean().item()
            total_seq_kl_sum += per_sample_kl.sum().item()
            total_seq_count  += len(valid_rows)
            interim_kl = float(total_seq_kl_sum / max(total_seq_count, 1))
            per_sample_kl_for_save = per_sample_kl
        else:
            raise ValueError("loss_agg_mode는 'token-mean' 또는 'sequence-mean'만 지원합니다.")

        global_indices = [i + r for r in valid_rows]
        for gi, klv, glen in zip(global_indices, per_sample_kl_for_save.tolist(), resp_lens.tolist()):
            per_sample_records.append({"index": int(gi), "kl": float(klv), "gen_len": int(glen)})

        if show_progress:
            iterator.set_postfix({
                "batch_kl": f"{batch_kl:.6f}",
                "interim_kl": f"{interim_kl:.6f}",
                "tok": total_masked_tokens if loss_agg_mode == "token-mean" else f"seq:{total_seq_count}"
            })
        if log_every and (b_idx % log_every == 0):
            print(
                f"[batch {b_idx}/{num_batches}] batch_kl={batch_kl:.6f}, "
                f"interim_kl={interim_kl:.6f}, "
                f"{'cum_tokens' if loss_agg_mode == 'token-mean' else 'cum_seqs'}="
                f"{total_masked_tokens if loss_agg_mode == 'token-mean' else total_seq_count}"
            )

    if loss_agg_mode == "token-mean":
        kl_loss = float(total_masked_kl_sum / max(total_masked_tokens, 1))
    else:
        kl_loss = float(total_seq_kl_sum / max(total_seq_count, 1))

    mean_resp_len = (total_gen_len / num_examples) if num_examples > 0 else 0.0

    metrics = {
        "actor/kl_loss": kl_loss,
        "num_prompts": N,
        "num_examples": num_examples,
        "num_zero_resp": num_zero_resp,
        "mean_resp_len": mean_resp_len,
        "batches": num_batches,
        "batch_size": batch_size,
        "agg_mode": loss_agg_mode,
        "scope": "static_responses",
    }
    if loss_agg_mode == "token-mean":
        metrics["num_masked_tokens"] = total_masked_tokens
    else:
        metrics["num_sequences"] = total_seq_count

    if save_per_sample_path is not None:
        os.makedirs(os.path.dirname(save_per_sample_path), exist_ok=True)
        with open(save_per_sample_path, "w", encoding="utf-8") as f:
            for rec in per_sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if save_summary_path is not None:
        os.makedirs(os.path.dirname(save_summary_path), exist_ok=True)
        summary = {
            "actor_model_name": actor_model_name,
            "ref_model_name": ref_model_name,
            "kl_loss": kl_loss,
            "metrics": metrics,
        }
        with open(save_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return kl_loss, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_model",    type=str, required=True)
    parser.add_argument("--ref_model",      type=str, required=True)
    parser.add_argument("--prompts_path",   type=str, required=True,
                        help="JSON file: list of {messages: [{role, content}, ...]} dicts")
    parser.add_argument("--responses_path", type=str, required=True,
                        help="JSON file: list of response strings (index-aligned with prompts)")
    parser.add_argument("--save_dir",       type=str, required=True)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--dtype",          type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device",         type=str, default=None)
    parser.add_argument("--loss_agg_mode",  type=str, default="sequence-mean",
                        choices=["token-mean", "sequence-mean"])
    parser.add_argument("--no_progress",    action="store_true")
    parser.add_argument("--log_every",      type=int, default=5)
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


if __name__ == "__main__":
    args = parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)
    prompts = [item["messages"] for item in raw_prompts]

    with open(args.responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    assert len(prompts) == len(responses), \
        f"prompts({len(prompts)}) != responses({len(responses)})"

    os.makedirs(args.save_dir, exist_ok=True)
    per_sample_path = os.path.join(args.save_dir, "per_sample_kl.jsonl")
    summary_path    = os.path.join(args.save_dir, "summary.json")

    kl, metrics = compute_kl_loss_static(
        actor_model_name=args.actor_model,
        ref_model_name=args.ref_model,
        prompts=prompts,
        responses=responses,
        dtype=str_to_dtype(args.dtype),
        device=args.device,
        batch_size=args.batch_size,
        loss_agg_mode=args.loss_agg_mode,
        show_progress=not args.no_progress,
        log_every=args.log_every,
        save_per_sample_path=per_sample_path,
        save_summary_path=summary_path,
    )

    print("Final KL =", kl)
    print(metrics)
    print(f"[saved] per-sample: {per_sample_path}")
    print(f"[saved] summary:    {summary_path}")
