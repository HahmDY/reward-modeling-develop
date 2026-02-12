"""
Check if the model actually learned during training
by evaluating on a subset of training data
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RMOOD_HOME = os.getenv("RMOOD_HOME")

def check_training_accuracy(model_path, data_path, num_samples=500):
    """Check accuracy on training data"""
    
    print("=" * 80)
    print("Checking model performance on TRAINING data")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print()
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        device_map="auto",
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load training data
    print("Loading training data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    chosen = data['chosen']
    rejected = data['rejected']
    
    # Random sample
    indices = np.random.choice(len(chosen), min(num_samples, len(chosen)), replace=False)
    
    print(f"Evaluating on {len(indices)} samples...")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx in tqdm(indices):
            # Prepare texts
            chosen_text = tokenizer.apply_chat_template(
                chosen[idx],
                tokenize=False,
                add_generation_prompt=False
            )
            rejected_text = tokenizer.apply_chat_template(
                rejected[idx],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize
            chosen_inputs = tokenizer(
                chosen_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            rejected_inputs = tokenizer(
                rejected_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            chosen_inputs = {k: v.to(model.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(model.device) for k, v in rejected_inputs.items()}
            
            # Get rewards
            chosen_output = model(**chosen_inputs)
            rejected_output = model(**rejected_inputs)
            
            chosen_reward = chosen_output.logits.item()
            rejected_reward = rejected_output.logits.item()
            
            # Count
            total += 1
            if chosen_reward > rejected_reward:
                correct += 1
    
    accuracy = correct / total
    
    print()
    print("=" * 80)
    print("TRAINING DATA ACCURACY")
    print("=" * 80)
    print(f"Correct: {correct} / {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print()
    
    if accuracy < 0.70:
        print("❌ VERY BAD: Model didn't learn at all!")
        print("   Problem: Training failed or model corrupted")
    elif accuracy < 0.85:
        print("⚠️  POOR: Model barely learned")
        print("   Problem: Insufficient training or learning rate too low")
    elif accuracy < 0.95:
        print("⚠️  OKAY: Model learned but not great")
        print("   Problem: May need more epochs or better hyperparameters")
    else:
        print("✓ GOOD: Model learned well on training data")
        print("   If test accuracy is still low, it's a generalization issue")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm")
    parser.add_argument("--data_path", type=str, 
                       default=f"{os.getenv('RMOOD_HOME')}/datasets/alpacafarm/rm/rm_implicit.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)
    
    args = parser.parse_args()
    
    check_training_accuracy(args.model_path, args.data_path, args.num_samples)
