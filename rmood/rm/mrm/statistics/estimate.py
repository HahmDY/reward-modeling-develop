import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RMOOD_HOME = os.getenv("RMOOD_HOME")


def extract_representations(args):
    """
    Extract hidden representations from chosen and rejected responses.
    
    Args:
        args: Command line arguments containing:
            - model_path: Path to the trained MRM model
            - data_path: Path to the dataset file (rm_implicit.jsonl)
            - output_dir: Directory to save the extracted representations
            - batch_size: Batch size for processing
    """
    print("=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        device_map="auto",
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Model loaded from: {args.model_path}")
    print(f"Model device: {model.device}")
    
    # Load dataset
    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    chosen_messages = data["chosen"]
    rejected_messages = data["rejected"]
    
    assert len(chosen_messages) == len(rejected_messages), \
        f"Mismatch in data lengths: chosen={len(chosen_messages)}, rejected={len(rejected_messages)}"
    
    num_samples = len(chosen_messages)
    print(f"Loaded {num_samples} samples")
    
    # Extract representations
    print("\n" + "=" * 80)
    print("Extracting representations...")
    print("=" * 80)
    
    chosen_representations = []
    rejected_representations = []
    
    # Process in batches
    batch_size = args.batch_size
    
    with torch.no_grad():
        # Process chosen responses
        print("\nProcessing chosen responses...")
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_messages = chosen_messages[i:i + batch_size]
            
            # Tokenize messages
            batch_texts = []
            for messages in batch_messages:
                # Apply chat template to format the messages
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                batch_texts.append(text)
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            )
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            
            # Extract hidden states (last hidden state)
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Get the last non-pad token representation for each sample
            batch_size_actual = inputs["input_ids"].shape[0]
            input_ids = inputs["input_ids"]
            
            # Find last non-pad token position
            non_pad_mask = (input_ids != model.config.pad_token_id).to(hidden_states.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=hidden_states.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            
            # Extract pooled features
            pooled_features = hidden_states[
                torch.arange(batch_size_actual, device=hidden_states.device),
                last_non_pad_token
            ]  # [batch_size, hidden_size]
            
            # Convert to numpy and store
            pooled_features_np = pooled_features.cpu().float().numpy()
            chosen_representations.extend(pooled_features_np)
        
        # Process rejected responses
        print("\nProcessing rejected responses...")
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_messages = rejected_messages[i:i + batch_size]
            
            # Tokenize messages
            batch_texts = []
            for messages in batch_messages:
                # Apply chat template to format the messages
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                batch_texts.append(text)
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            )
            
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            
            # Extract hidden states (last hidden state)
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Get the last non-pad token representation for each sample
            batch_size_actual = inputs["input_ids"].shape[0]
            input_ids = inputs["input_ids"]
            
            # Find last non-pad token position
            non_pad_mask = (input_ids != model.config.pad_token_id).to(hidden_states.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=hidden_states.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            
            # Extract pooled features
            pooled_features = hidden_states[
                torch.arange(batch_size_actual, device=hidden_states.device),
                last_non_pad_token
            ]  # [batch_size, hidden_size]
            
            # Convert to numpy and store
            pooled_features_np = pooled_features.cpu().float().numpy()
            rejected_representations.extend(pooled_features_np)
    
    # Convert to numpy arrays
    chosen_representations = np.array(chosen_representations)
    rejected_representations = np.array(rejected_representations)
    
    print("\n" + "=" * 80)
    print("Representation extraction completed!")
    print("=" * 80)
    print(f"Chosen representations shape: {chosen_representations.shape}")
    print(f"Rejected representations shape: {rejected_representations.shape}")
    
    # Save representations
    os.makedirs(args.output_dir, exist_ok=True)
    
    chosen_output_path = os.path.join(args.output_dir, "chosen_representations.npy")
    rejected_output_path = os.path.join(args.output_dir, "rejected_representations.npy")
    
    np.save(chosen_output_path, chosen_representations)
    np.save(rejected_output_path, rejected_representations)
    
    print("\n" + "=" * 80)
    print("Saved representations:")
    print("=" * 80)
    print(f"Chosen: {chosen_output_path}")
    print(f"Rejected: {rejected_output_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    print(f"Chosen - Mean: {chosen_representations.mean():.4f}, Std: {chosen_representations.std():.4f}")
    print(f"Rejected - Mean: {rejected_representations.mean():.4f}, Std: {rejected_representations.std():.4f}")
    
    return chosen_representations, rejected_representations


def compute_gda_parameters(chosen_representations, rejected_representations):
    """
    Compute GDA parameters from the extracted representations.
    Uses Ledoit-Wolf shrinkage estimator for numerically stable covariance estimation.
    
    Args:
        chosen_representations: numpy array of shape [N, hidden_size]
        rejected_representations: numpy array of shape [N, hidden_size]
    
    Returns:
        mu_pos, mu_neg, sigma, sigma_inv
    """
    print("\n" + "=" * 80)
    print("Computing GDA parameters (Ledoit-Wolf shrinkage)...")
    print("=" * 80)
    
    # Compute means
    mu_pos = chosen_representations.mean(axis=0)  # μ_+
    mu_neg = rejected_representations.mean(axis=0)  # μ_-
    
    print(f"μ_+ (chosen mean) shape: {mu_pos.shape}")
    print(f"μ_- (rejected mean) shape: {mu_neg.shape}")
    
    # Center the data by class means
    chosen_centered = chosen_representations - mu_pos
    rejected_centered = rejected_representations - mu_neg
    
    # Pool all centered data for within-class covariance estimation
    all_centered = np.vstack([chosen_centered, rejected_centered])
    
    # Ledoit-Wolf shrinkage: Σ_shrunk = (1-α)Σ_sample + α·(tr(Σ)/d)·I
    # α is automatically estimated from data
    print(f"Fitting Ledoit-Wolf on {all_centered.shape[0]} samples, {all_centered.shape[1]} dimensions...")
    lw = LedoitWolf(assume_centered=True)
    lw.fit(all_centered)
    
    sigma = lw.covariance_
    sigma_inv = lw.precision_  # = sigma^{-1}, computed with shrinkage
    
    print(f"Σ (shrunk covariance) shape: {sigma.shape}")
    print(f"Σ condition number: {np.linalg.cond(sigma):.2e}")
    print(f"Ledoit-Wolf shrinkage coefficient (α): {lw.shrinkage_:.6f}")
    
    return mu_pos, mu_neg, sigma, sigma_inv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract representations from MRM model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained MRM model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/rm_implicit.jsonl",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{RMOOD_HOME}/representations",
        help="Directory to save the extracted representations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--compute_gda",
        action="store_true",
        help="Also compute and save GDA parameters (mu_pos, mu_neg, sigma, sigma_inv)"
    )
    
    args = parser.parse_args()
    
    # # Extract representations
    # chosen_reps, rejected_reps = extract_representations(args)
    chosen_reps = np.load(f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/chosen_representations.npy")
    rejected_reps = np.load(f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/rejected_representations.npy")
    
    # Optionally compute GDA parameters
    if args.compute_gda:
        mu_pos, mu_neg, sigma, sigma_inv = compute_gda_parameters(chosen_reps, rejected_reps)
        
        # Save GDA parameters
        gda_output_path = os.path.join(args.output_dir, "gda_parameters.npz")
        np.savez(
            gda_output_path,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sigma=sigma,
            sigma_inv=sigma_inv
        )
        print(f"\nGDA parameters saved to: {gda_output_path}")
