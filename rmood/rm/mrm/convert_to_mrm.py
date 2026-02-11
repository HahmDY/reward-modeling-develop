import os
import argparse
import shutil
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rmood.rm.mrm.mrm.model import MRM

RMOOD_HOME = os.getenv("RMOOD_HOME")


def load_gda_parameters():
    parameters_path = f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/gda_parameters.npz"
    with np.load(parameters_path) as data:
        mu_pos = data["mu_pos"]
        mu_neg = data["mu_neg"]
        sigma_inv = data["sigma_inv"]
        
    return mu_pos, mu_neg, sigma_inv


def convert_qwen3_to_mrm(base_model, mu_pos, mu_neg, sigma_inv):
    """
    Convert Qwen3ForSequenceClassification model to MRM
    
    Args:
        base_model: Trained Qwen3ForSequenceClassification model
        mu_pos, mu_neg, sigma_inv: GDA parameters
    
    Returns:
        mrm_model: MRM model
    """
    print(f"\n[Converting to MRM]")
    
    # 1. Create MRM model with same config
    print(f"Creating MRM model with same config...")
    mrm_model = MRM(base_model.config)
    
    # 2. Copy weights from base model
    print(f"Copying weights from base model...")
    mrm_model.model.load_state_dict(base_model.model.state_dict())
    mrm_model.score.load_state_dict(base_model.score.state_dict())
    
    # 3. Set GDA parameters
    print(f"Setting GDA parameters...")
    mrm_model.set_gda_params(mu_pos, mu_neg, sigma_inv)
    
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
    print(f"  - bias: {mrm_model.bias.item():.6f}")
    
    return mrm_model


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3ForSequenceClassification to MRM (GDA-based Reward Model)"
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
    
    # Step 2: GDA parameters
    mu_pos, mu_neg, sigma_inv = load_gda_parameters()
    
    # Step 3: Convert to MRM and save
    mrm_model = convert_qwen3_to_mrm(base_model, mu_pos, mu_neg, sigma_inv)
    
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
