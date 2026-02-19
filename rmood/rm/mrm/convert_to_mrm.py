import os
import argparse
import shutil
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rmood.rm.mrm.mrm.model import MRM

RMOOD_HOME = os.getenv("RMOOD_HOME")


def load_gda_parameters(clean_name):
    parameters_path = f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/{clean_name}/gda_parameters.npz"
    with np.load(parameters_path) as data:
        mu_chosen = data["mu_chosen"]
        mu_rejected = data["mu_rejected"]
        mu_d = data["mu_d"]
        sigma_chosen = data["sigma_chosen"]
        sigma_chosen_inv = data["sigma_chosen_inv"]
        sigma_rejected = data["sigma_rejected"]
        sigma_rejected_inv = data["sigma_rejected_inv"]
        sigma_d = data["sigma_d"]
        sigma_d_inv = data["sigma_d_inv"]
    return mu_chosen, mu_rejected, mu_d, sigma_chosen, sigma_chosen_inv, sigma_rejected, sigma_rejected_inv, sigma_d, sigma_d_inv


def convert_qwen3_to_mrm(base_model, mu_chosen, mu_rejected, mu_d, sigma_chosen, sigma_chosen_inv, sigma_rejected, sigma_rejected_inv, sigma_d, sigma_d_inv):
    """
    Convert Qwen3ForSequenceClassification model to MRM.

    Args:
        base_model: Trained Qwen3ForSequenceClassification model
        mu_d:      mean of difference vectors E[chosen - rejected]  [hidden_size]
        sigma_chosen: Σ_chosen (Ledoit-Wolf)                         [hidden_size, hidden_size]
        sigma_chosen_inv: Σ_chosen^{-1} (Ledoit-Wolf)               [hidden_size, hidden_size]
        sigma_rejected: Σ_rejected (Ledoit-Wolf)                     [hidden_size, hidden_size]
        sigma_rejected_inv: Σ_rejected^{-1} (Ledoit-Wolf)           [hidden_size, hidden_size]
        sigma_d: Σ_d (Ledoit-Wolf)                                   [hidden_size, hidden_size]
        sigma_d_inv: Σ_d^{-1} (Ledoit-Wolf)                          [hidden_size, hidden_size]

    Returns:
        mrm_model: MRM model
    """
    print(f"\n[Converting to MRM]")
    
    # 1. Create MRM model with same config
    print(f"Creating MRM model with same config...")
    base_model.config.num_labels = 1
    mrm_model = MRM(base_model.config)
    
    # 2. Copy weights from base model (score layer is not copied since MRM
    #    uses compute_gda_reward for final output when use_gda_reward=True)
    print(f"Copying weights from base model...")
    mrm_model.model.load_state_dict(base_model.model.state_dict())
    
    # 3. Set difference-based GDA parameters
    print(f"Setting GDA parameters...")
    mrm_model.set_gda_params(mu_chosen, mu_rejected, mu_d, sigma_chosen, sigma_chosen_inv, sigma_rejected, sigma_rejected_inv, sigma_d, sigma_d_inv)
    
    # 4. Match dtype and device
    mrm_model = mrm_model.to(dtype=torch.bfloat16)
    
    # 5. Configure auto_map for trust_remote_code
    print(f"Configuring auto_map for remote code loading...")
    mrm_model.config.auto_map = {
        "AutoModel": "modeling_mrm.MRM",
        "AutoModelForSequenceClassification": "modeling_mrm.MRM"
    }
    
    print(f"✓ Conversion complete!")
    print(f"  - Model type: {type(mrm_model).__name__}")
    print(f"  - use_gda_reward: {mrm_model.use_gda_reward}")
    print(f"  - ||μ_chosen||: {np.linalg.norm(mu_chosen):.4f}")
    print(f"  - ||μ_rejected||: {np.linalg.norm(mu_rejected):.4f}")
    print(f"  - ||μ_d||: {np.linalg.norm(mu_d):.4f}")
    print(f"  - ||σ_chosen||: {np.linalg.norm(sigma_chosen):.4f}")
    print(f"  - ||σ_chosen_inv||: {np.linalg.norm(sigma_chosen_inv):.4f}")
    print(f"  - ||σ_rejected||: {np.linalg.norm(sigma_rejected):.4f}")
    print(f"  - ||σ_rejected_inv||: {np.linalg.norm(sigma_rejected_inv):.4f}")
    print(f"  - ||σ_d||: {np.linalg.norm(sigma_d):.4f}")
    print(f"  - ||σ_d_inv||: {np.linalg.norm(sigma_d_inv):.4f}")
    
    return mrm_model


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3ForSequenceClassification to MRM (difference-based GDA Reward Model)"
    )
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True, 
        help="Path to trained Qwen3ForSequenceClassification model"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to save MRM model"
    )
    parser.add_argument(
        "--push_to_hub", 
        type=str, 
        default=None,
        help="Hugging Face Hub repository name (e.g., username/model-name)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Converting Qwen3ForSequenceClassification to MRM")
    print("=" * 80)
    
    # Step 1: Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print(f"✓ Model loaded from {args.base_model_path}")
    
    # Step 2: Load difference-based GDA parameters
    clean_name = args.base_model_path.replace("/", "--")
    mu_chosen, mu_rejected, mu_d, sigma_chosen, sigma_chosen_inv, sigma_rejected, sigma_rejected_inv, sigma_d, sigma_d_inv = load_gda_parameters(clean_name)
    print(f"✓ GDA parameters loaded  ||μ_d||={np.linalg.norm(mu_d):.4f}")

    # Step 3: Convert to MRM and save
    mrm_model = convert_qwen3_to_mrm(base_model, mu_chosen, mu_rejected, mu_d, sigma_chosen, sigma_chosen_inv, sigma_rejected, sigma_rejected_inv, sigma_d, sigma_d_inv)
    
    # Save model
    os.makedirs(args.output_path, exist_ok=True)
    
    # Copy model.py to output directory for trust_remote_code
    model_source = os.path.join(os.path.dirname(__file__), "mrm", "model.py")
    model_dest = os.path.join(args.output_path, "modeling_mrm.py")
    shutil.copy(model_source, model_dest)
    
    mrm_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    # Push to Hugging Face Hub (optional)
    if args.push_to_hub:
        mrm_model.push_to_hub(
            args.push_to_hub, 
            commit_message="Convert Qwen3ForSequenceClassification to MRM"
        )
        tokenizer.push_to_hub(
            args.push_to_hub, 
            commit_message="Add tokenizer for MRM"
        )


if __name__ == "__main__":
    main()
