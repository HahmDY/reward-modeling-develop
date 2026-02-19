import os
import numpy as np
import torch
import argparse
from transformers import AutoModelForSequenceClassification


RMOOD_HOME = os.getenv("RMOOD_HOME")


def extract_score_weight(model_name):
    """
    Extract the final score weight from a reward model and save as .npy
    
    Args:
        model_name: HuggingFace model name (e.g., "Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm")
    """
    print(f"Loading model: {model_name}")
    
    # Load the reward model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="cpu",  # Load on CPU to avoid GPU memory issues
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Extract the final score layer weight
    # Common attribute names for the score layer
    score_layer = None
    if hasattr(model, 'score'):
        score_layer = model.score
    elif hasattr(model, 'classifier'):
        score_layer = model.classifier
    elif hasattr(model, 'v_head'):
        score_layer = model.v_head
    else:
        raise ValueError(f"Could not find score layer in model. Available attributes: {dir(model)}")
    
    # Get the weight tensor
    weight = score_layer.weight.detach().cpu().float().numpy()
    
    print(f"Extracted weight shape: {weight.shape}")
    
    # Create target directory
    model_name_clean = model_name.replace("/", "_")
    target_dir = f"{RMOOD_HOME}/datasets/alpacafarm/distribution/{model_name_clean}"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    target_path = f"{target_dir}/weight.npy"
    
    # Save as .npy
    with open(target_path, "wb") as f:
        np.save(f, weight)
    
    print(f"Saved weight to: {target_path}")
    
    # # Also save bias if it exists
    # if hasattr(score_layer, 'bias') and score_layer.bias is not None:
    #     bias = score_layer.bias.detach().cpu().float().numpy()
    #     bias_path = f"{target_dir}/bias.npy"
    #     with open(bias_path, "wb") as f:
    #         np.save(f, bias)
    #     print(f"Saved bias to: {bias_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract final score weight from reward model")
    parser.add_argument("--model_name", type=str,
                        default="Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-center",
                        help="Reward model name (e.g., Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm)")
    args = parser.parse_args()
    
    extract_score_weight(args.model_name)
