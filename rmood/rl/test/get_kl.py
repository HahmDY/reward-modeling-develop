import os
import json
from typing import List, Dict, Tuple, Optional
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

@torch.no_grad()
def compute_kl_loss_between_models(
    actor_model_name: str,
    ref_model_name: str,
    conversations: List[List[Dict[str, str]]],  # [{"role":..., "content":...}, ...]
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    batch_size: int = 128,
    loss_agg_mode: str = "token-mean",   # "token-mean" | "sequence-mean"
    show_progress: bool = True,
    log_every: int = 10,
    # generation params
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    # save paths (optional)
    save_per_sample_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:

    # --- tokenizer/model load
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

    # --- helpers
    def left_pad_batch(seqs: List[List[int]], pad_id: int, device: str):
        lens = [len(s) for s in seqs]
        max_len = max(lens) if lens else 0
        B = len(seqs)
        ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            L = len(s)
            if L > 0:
                ids[r, -L:] = torch.tensor(s, dtype=torch.long, device=device)
                attn[r, -L:] = 1
        return ids, attn, lens

    total_masked_kl_sum = 0.0
    total_masked_tokens = 0
    total_seq_kl_sum    = 0.0
    total_seq_count     = 0

    total_gen_len = 0
    num_examples = 0
    num_zero_gen = 0

    per_sample_records = []  # dicts: {"index": int, "kl": float, "gen_len": int}

    num_conversations = len(conversations)
    num_batches = (num_conversations + batch_size - 1) // batch_size
    iterator = range(0, num_conversations, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=f"KL on actor-generated ({loss_agg_mode})")

    for b_idx, i in enumerate(iterator, start=1):
        chats = conversations[i:i+batch_size]

        prompt_ids_list: List[List[int]] = [
            tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True) for chat in chats
        ]

        input_prompt, attn_prompt, prompt_lens = left_pad_batch(prompt_ids_list, tok.pad_token_id, device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        gen_out = actor.generate(
            input_ids=input_prompt,
            attention_mask=attn_prompt,
            use_cache=True,
            **gen_kwargs,
        )

        # 3) 생성 기반 full_input/resp_mask 구성
        # full_input는 이미 왼쪽 패딩된 시퀀스. attention은 pad!=id 기준으로 재생성(안전)
        full_input = gen_out.to(device)                                  # [B, T]
        full_attn  = (full_input != tok.pad_token_id).long()             # [B, T]
        B, T = full_input.shape

        # per-sample 프롬프트 길이 텐서화 (필터링 전 배치 전체 기준)
        prompt_lens_tensor = torch.tensor(prompt_lens, device=device, dtype=torch.long)  # [B]

        # 각 샘플의 '프롬프트 이후' 위치 마스크
        ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)    # [B, T]
        after_prompt = ar >= prompt_lens_tensor.unsqueeze(1)             # [B, T]
        nonpad = (full_input != tok.pad_token_id)                        # [B, T]

        # 응답 마스크(프롬프트 이후 & 패드 제외)
        resp_mask_all = (after_prompt & nonpad).long()                   # [B, T]
        resp_lens_all = resp_mask_all.sum(dim=1)                         # [B]

        # 유효 행 판별
        valid_rows = (resp_lens_all > 0).nonzero(as_tuple=False).squeeze(1).tolist()
        num_zero_gen += int((resp_lens_all == 0).sum().item())

        if len(valid_rows) == 0:
            interim = (total_masked_kl_sum / max(total_masked_tokens, 1)) if loss_agg_mode == "token-mean" \
                     else (total_seq_kl_sum / max(total_seq_count, 1))
            if show_progress:
                iterator.set_postfix({
                    "batch_kl": "n/a",
                    "interim_kl": f"{interim:.6f}",
                    "tok": total_masked_tokens if loss_agg_mode=="token-mean" else f"seq:{total_seq_count}"
                })
            if log_every and (b_idx % log_every == 0):
                print(f"[batch {b_idx}/{num_batches}] no valid generations; interim_kl={interim:.6f}")
            continue

        full_input = full_input[valid_rows]
        full_attn  = full_attn[valid_rows]
        resp_mask  = resp_mask_all[valid_rows]
        resp_lens  = resp_lens_all[valid_rows]
        prompt_lens_batch = prompt_lens_tensor[valid_rows]

        total_gen_len += int(resp_lens.sum().item())
        num_examples  += len(valid_rows)

        out_actor = actor(input_ids=full_input, attention_mask=full_attn, use_cache=False)
        out_ref   = ref(  input_ids=full_input, attention_mask=full_attn, use_cache=False)

        logits_actor = out_actor.logits[:, :-1, :]                       # [B, T-1, V]
        logits_ref   = out_ref.logits[:,   :-1, :]                       # [B, T-1, V]
        mask_tok     = resp_mask[:, 1:]                                  # [B, T-1]

        logp_actor = torch.log_softmax(logits_actor, dim=-1)
        logp_ref   = torch.log_softmax(logits_ref,   dim=-1)
        p_actor    = logp_actor.exp()

        kl_per_token = (p_actor * (logp_actor - logp_ref)).sum(dim=-1)   # [B, T-1]

        per_sample_tok = mask_tok.sum(dim=1).clamp_min(1)                # [B]
        per_sample_kl_tokenmean = ((kl_per_token * mask_tok).sum(dim=1) / per_sample_tok)  # [B]

        if loss_agg_mode == "token-mean":
            masked_kl_sum = (kl_per_token * mask_tok).sum().item()
            num_masked    = int(mask_tok.sum().item())

            batch_kl = float(masked_kl_sum / max(num_masked, 1))
            total_masked_kl_sum += masked_kl_sum
            total_masked_tokens += num_masked
            interim_kl = float(total_masked_kl_sum / max(total_masked_tokens, 1))

            per_sample_kl_for_save = per_sample_kl_tokenmean

        elif loss_agg_mode == "sequence-mean":
            per_sample_kl_seqmean  = ((kl_per_token * mask_tok).sum(dim=1) / per_sample_tok)
            batch_kl = per_sample_kl_seqmean.mean().item()
            total_seq_kl_sum += per_sample_kl_seqmean.sum().item()
            total_seq_count  += int((per_sample_tok > 0).sum().item())
            interim_kl = float(total_seq_kl_sum / max(total_seq_count, 1))

            per_sample_kl_for_save = per_sample_kl_seqmean
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
                f"{'cum_tokens' if loss_agg_mode=='token-mean' else 'cum_seqs'}="
                f"{total_masked_tokens if loss_agg_mode=='token-mean' else total_seq_count}"
            )

    if loss_agg_mode == "token-mean":
        kl_loss = float(total_masked_kl_sum / max(total_masked_tokens, 1))
    else:
        kl_loss = float(total_seq_kl_sum / max(total_seq_count, 1))

    mean_gen_len = (total_gen_len / num_examples) if num_examples > 0 else 0.0

    metrics = {
        "actor/kl_loss": kl_loss,
        "num_conversations": num_conversations,
        "num_examples": num_examples,      # 유효 생성(길이>0)이 있었던 대화 수
        "num_zero_gen": num_zero_gen,      # 생성 길이 0으로 스킵된 대화 수
        "mean_gen_len": mean_gen_len,
        "batches": num_batches,
        "batch_size": batch_size,
        "agg_mode": loss_agg_mode,
        "scope": "generated_only",
        "gen_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        },
    }
    if loss_agg_mode == "token-mean":
        metrics["num_masked_tokens"] = total_masked_tokens
    else:
        metrics["num_sequences"] = total_seq_count

    # --- 파일 저장 (옵션)
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
    parser.add_argument("--actor_model", type=str, required=True)
    parser.add_argument("--ref_model", type=str, required=True)
    parser.add_argument("--conversations_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loss_agg_mode", type=str, default="token-mean", choices=["token-mean", "sequence-mean"])
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    return parser.parse_args()


def str_to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


if __name__ == "__main__":
    args = parse_args()

    with open(args.conversations_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    conversations = [[{"role": "system", "content": ""}] + item["messages"] for item in raw]

    os.makedirs(args.save_dir, exist_ok=True)
    per_sample_path = os.path.join(args.save_dir, "per_sample_kl.jsonl")
    summary_path    = os.path.join(args.save_dir, "summary.json")

    kl, metrics = compute_kl_loss_between_models(
        actor_model_name=args.actor_model,
        ref_model_name=args.ref_model,
        conversations=conversations,
        dtype=str_to_dtype(args.dtype),
        device=args.device,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        log_every=args.log_every,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        loss_agg_mode=args.loss_agg_mode,
        save_per_sample_path=per_sample_path,
        save_summary_path=summary_path,
    )
    print("Final KL =", kl)
    print(metrics)
    print(f"[saved] per-sample: {per_sample_path}")
    print(f"[saved] summary:    {summary_path}")
