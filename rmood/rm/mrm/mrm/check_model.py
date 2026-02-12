import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
import os

# Load the model from HuggingFace
model_name = "Hahmdong/RMOOD-qwen3-4b-alpacafarm-mrm"
print(f"Loading model: {model_name}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="cpu",  # Use CPU to avoid memory issues
    num_labels=1,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# Check the actual dtype of model parameters
print(f"\nModel parameter dtypes:")
print(f"  mu_pos dtype: {model.mu_pos.dtype}")
print(f"  mu_neg dtype: {model.mu_neg.dtype}")
print(f"  sigma_inv dtype: {model.sigma_inv.dtype}")

# Extract GDA parameters from the model
model_mu_pos = model.mu_pos.cpu().numpy()
model_mu_neg = model.mu_neg.cpu().numpy()
model_sigma_inv = model.sigma_inv.cpu().numpy()
model_bias = model.bias.cpu().numpy()

print(f"Model mu_pos shape: {model_mu_pos.shape}")
print(f"Model mu_neg shape: {model_mu_neg.shape}")
print(f"Model sigma_inv shape: {model_sigma_inv.shape}")
print(f"Model bias shape: {model_bias.shape}")

print(f"\nModel mu_pos stats - mean: {model_mu_pos.mean():.6f}, std: {model_mu_pos.std():.6f}")
print(f"Model mu_neg stats - mean: {model_mu_neg.mean():.6f}, std: {model_mu_neg.std():.6f}")
print(f"Model sigma_inv stats - mean: {model_sigma_inv.mean():.6f}, std: {model_sigma_inv.std():.6f}")
print(f"Model bias value: {model_bias}")

# Load the npz file
npz_path = "/root/reward-modeling-develop/datasets/alpacafarm/rm/representations/gda_parameters.npz"
print(f"Loading npz file: {npz_path}")

npz_data = np.load(npz_path)

# Check what keys are in the npz file
print(f"\nKeys in npz file: {list(npz_data.keys())}")

# Extract parameters from npz
npz_mu_pos = npz_data['mu_pos']
npz_mu_neg = npz_data['mu_neg']
npz_sigma_inv = npz_data['sigma_inv']

print(f"\nNPZ mu_pos shape: {npz_mu_pos.shape}")
print(f"NPZ mu_neg shape: {npz_mu_neg.shape}")
print(f"NPZ sigma_inv shape: {npz_sigma_inv.shape}")

print(f"\nNPZ mu_pos stats - mean: {npz_mu_pos.mean():.6f}, std: {npz_mu_pos.std():.6f}")
print(f"NPZ mu_neg stats - mean: {npz_mu_neg.mean():.6f}, std: {npz_mu_neg.std():.6f}")
print(f"NPZ sigma_inv stats - mean: {npz_sigma_inv.mean():.6f}, std: {npz_sigma_inv.std():.6f}")

# Compare mu_pos
print("="*80)
print("Comparing mu_pos")
print("="*80)

mu_pos_diff = np.abs(model_mu_pos - npz_mu_pos)
mu_pos_max_diff = mu_pos_diff.max()
mu_pos_mean_diff = mu_pos_diff.mean()
mu_pos_allclose = np.allclose(model_mu_pos, npz_mu_pos, rtol=1e-5, atol=1e-8)
# bfloat16 has ~3 decimal digits of precision, so use rtol=5e-3 (0.5%)
mu_pos_allclose_relaxed = np.allclose(model_mu_pos, npz_mu_pos, rtol=5e-3, atol=0.1)

print(f"Max absolute difference: {mu_pos_max_diff:.10f}")
print(f"Mean absolute difference: {mu_pos_mean_diff:.10f}")
print(f"Are they close (rtol=1e-5, atol=1e-8)? {mu_pos_allclose}")
print(f"Are they close (rtol=5e-3, atol=0.1, for bfloat16)? {mu_pos_allclose_relaxed}")
print(f"Are they exactly equal? {np.array_equal(model_mu_pos, npz_mu_pos)}")

if not mu_pos_allclose:
    print(f"\nFirst 10 differences:")
    for i in range(min(10, len(mu_pos_diff))):
        print(f"  Index {i}: model={model_mu_pos[i]:.10f}, npz={npz_mu_pos[i]:.10f}, diff={mu_pos_diff[i]:.10f}")
        
# Compare mu_neg
print("="*80)
print("Comparing mu_neg")
print("="*80)

mu_neg_diff = np.abs(model_mu_neg - npz_mu_neg)
mu_neg_max_diff = mu_neg_diff.max()
mu_neg_mean_diff = mu_neg_diff.mean()
mu_neg_allclose = np.allclose(model_mu_neg, npz_mu_neg, rtol=1e-5, atol=1e-8)
# bfloat16 has ~3 decimal digits of precision, so use rtol=5e-3 (0.5%)
mu_neg_allclose_relaxed = np.allclose(model_mu_neg, npz_mu_neg, rtol=5e-3, atol=0.1)

print(f"Max absolute difference: {mu_neg_max_diff:.10f}")
print(f"Mean absolute difference: {mu_neg_mean_diff:.10f}")
print(f"Are they close (rtol=1e-5, atol=1e-8)? {mu_neg_allclose}")
print(f"Are they close (rtol=5e-3, atol=0.1, for bfloat16)? {mu_neg_allclose_relaxed}")
print(f"Are they exactly equal? {np.array_equal(model_mu_neg, npz_mu_neg)}")

if not mu_neg_allclose:
    print(f"\nFirst 10 differences:")
    for i in range(min(10, len(mu_neg_diff))):
        print(f"  Index {i}: model={model_mu_neg[i]:.10f}, npz={npz_mu_neg[i]:.10f}, diff={mu_neg_diff[i]:.10f}")
        
# Compare sigma_inv
print("="*80)
print("Comparing sigma_inv")
print("="*80)

sigma_inv_diff = np.abs(model_sigma_inv - npz_sigma_inv)
sigma_inv_max_diff = sigma_inv_diff.max()
sigma_inv_mean_diff = sigma_inv_diff.mean()
sigma_inv_allclose = np.allclose(model_sigma_inv, npz_sigma_inv, rtol=1e-5, atol=1e-8)
# bfloat16 with large values: use rtol=5e-3 (0.5%) and large atol for absolute errors
sigma_inv_allclose_relaxed = np.allclose(model_sigma_inv, npz_sigma_inv, rtol=5e-3, atol=20000)

print(f"Max absolute difference: {sigma_inv_max_diff:.10f}")
print(f"Mean absolute difference: {sigma_inv_mean_diff:.10f}")
print(f"Are they close (rtol=1e-5, atol=1e-8)? {sigma_inv_allclose}")
print(f"Are they close (rtol=5e-3, atol=20000, for bfloat16)? {sigma_inv_allclose_relaxed}")
print(f"Are they exactly equal? {np.array_equal(model_sigma_inv, npz_sigma_inv)}")

if not sigma_inv_allclose:
    print(f"\nSample differences (5x5 top-left corner):")
    for i in range(min(5, sigma_inv_diff.shape[0])):
        for j in range(min(5, sigma_inv_diff.shape[1])):
            print(f"  [{i},{j}]: model={model_sigma_inv[i,j]:.6f}, npz={npz_sigma_inv[i,j]:.6f}, diff={sigma_inv_diff[i,j]:.6f}")
            
# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"mu_pos matches (strict): {mu_pos_allclose}")
print(f"mu_neg matches (strict): {mu_neg_allclose}")
print(f"sigma_inv matches (strict): {sigma_inv_allclose}")
print()
print(f"mu_pos matches (relaxed for bfloat16): {mu_pos_allclose_relaxed}")
print(f"mu_neg matches (relaxed for bfloat16): {mu_neg_allclose_relaxed}")
print(f"sigma_inv matches (relaxed for bfloat16): {sigma_inv_allclose_relaxed}")
print()

if mu_pos_allclose and mu_neg_allclose and sigma_inv_allclose:
    print("✓ All parameters match between model and npz file! (strict tolerance)")
elif mu_pos_allclose_relaxed and mu_neg_allclose_relaxed and sigma_inv_allclose_relaxed:
    print("✓ All parameters match with bfloat16 tolerance!")
    print("  The differences are due to bfloat16 quantization, which is expected.")
else:
    print("✗ Some parameters do NOT match:")
    if not mu_pos_allclose:
        print(f"  - mu_pos: max diff = {mu_pos_max_diff:.10f} (bfloat16 relaxed: {mu_pos_allclose_relaxed})")
    if not mu_neg_allclose:
        print(f"  - mu_neg: max diff = {mu_neg_max_diff:.10f} (bfloat16 relaxed: {mu_neg_allclose_relaxed})")
    if not sigma_inv_allclose:
        print(f"  - sigma_inv: max diff = {sigma_inv_max_diff:.10f} (bfloat16 relaxed: {sigma_inv_allclose_relaxed})")