import os
import json
import math
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RMOOD_HOME = os.getenv("RMOOD_HOME")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Label rewards for dataset using a reward model")
    parser.add_argument("--prompts_path", type=str, default=None, help="Path to prompts JSON file")
    parser.add_argument("--responses_path", type=str, default=None, help="Path to responses JSON file")
    parser.add_argument("--target_path", type=str, default=None, help="Path to output rewards JSON file")
    parser.add_argument("--model_name", type=str, default="", help="Path to reward model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for scoring")
    parser.add_argument("--num_responses", type=int, default=2, help="Number of responses to label")
    return parser.parse_args()


def batched_rewards(texts, model, tokenizer, batch_size=1024, desc="Scoring", amp_dtype=torch.bfloat16):
    """Score a list of texts in large batches."""
    scores_out = []
    n_batches = math.ceil(len(texts) / batch_size)
    for i in tqdm(range(n_batches), desc=desc):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

        bscores = logits.squeeze(-1).detach().float().cpu().tolist()
        if isinstance(bscores, float):
            bscores = [bscores]
        scores_out.extend(bscores)

        del enc, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scores_out


def main():
    """Main function to label rewards."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set default paths based on dataset_name if not provided
    prompts_path = args.prompts_path
    responses_path = args.responses_path
    target_path = args.target_path
    model_name = args.model_name
    num_responses = args.num_responses
    
    # Load model and tokenizer
    print(f"Loading model from {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="auto",
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data
    print(f"Loading prompts from {prompts_path}...")
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    
    print(f"Loading responses from {responses_path}...")
    with open(responses_path, "r") as f:
        responses = json.load(f)
    
    # Prepare texts
    flat_texts = []
    for prompt, response in tqdm(list(zip(prompts, responses)), total=len(prompts), desc="Preparing"):
        if num_responses == 1:
            convs = [
                prompt["messages"] + [{"role": "assistant", "content": response}]
            ]
        else:
            responses_key_list = [f"response_{i+1}" for i in range(num_responses)]
            convs = [
                prompt["messages"] + [{"role": "assistant", "content": response[k]}]
                for k in responses_key_list
            ]
        
        conv_texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in convs
        ]
        flat_texts.extend(conv_texts)
    
    # Score texts
    scores_flat = batched_rewards(flat_texts, model, tokenizer, batch_size=args.batch_size, desc="Scoring")
    
    # Reshape scores
    assert len(scores_flat) == num_responses * len(prompts)
    all_rewards = [scores_flat[num_responses*i : num_responses*i + num_responses] for i in range(len(prompts))]
    
    # Save results
    print(f"Saving rewards to {target_path}...")
    with open(target_path, "w") as f:
        json.dump(all_rewards, f, indent=4)
    
    print(f"Done! Saved {len(all_rewards)} reward pairs.")


if __name__ == "__main__":
    main()
