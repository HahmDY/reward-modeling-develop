import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

RMOOD_HOME = os.getenv("RMOOD_HOME")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors (numpy)"""
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============================================================
# 1. Load model and representations
# ============================================================
model_name = "Hahmdong/RMOOD-qwen3-4b-alpacafarm-mrm"
print(f"Loading model: {model_name}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="cpu",
    num_labels=1,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
print("Model loaded successfully!")

# Extract weights in float32
score_weight = model.score.weight.data.squeeze().numpy()  # [hidden_size]
mu_pos = model.mu_pos.numpy()
mu_neg = model.mu_neg.numpy()
sigma_inv = model.sigma_inv.numpy()

mu_diff = mu_pos - mu_neg
gda_weight = sigma_inv @ mu_diff

# Load representations
rep_dir = f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations"
chosen_reps = np.load(f"{rep_dir}/chosen_representations.npy")
rejected_reps = np.load(f"{rep_dir}/rejected_representations.npy")
N = chosen_reps.shape[0]
print(f"Loaded {N} pairs of representations, dim={chosen_reps.shape[1]}")


# ============================================================
# 2. Weight Direction Comparison
# ============================================================
print("\n" + "="*80)
print("WEIGHT DIRECTION COMPARISON")
print("="*80)

print(f"Score weight norm:       {np.linalg.norm(score_weight):.6f}")
print(f"μ+ - μ- norm:           {np.linalg.norm(mu_diff):.6f}")
print(f"GDA weight norm:         {np.linalg.norm(gda_weight):.6f}")

cos_score_gda = cosine_similarity(score_weight, gda_weight)
cos_score_mudiff = cosine_similarity(score_weight, mu_diff)
cos_mudiff_gda = cosine_similarity(mu_diff, gda_weight)

print(f"\ncos(w_score, w_gda):     {cos_score_gda:.6f}   ← Σ⁻¹ 포함")
print(f"cos(w_score, μ+ - μ-):  {cos_score_mudiff:.6f}   ← Σ⁻¹ 제외 (핵심 진단)")
print(f"cos(μ+ - μ-, w_gda):    {cos_mudiff_gda:.6f}   ← Σ⁻¹ 의 회전 효과")


# ============================================================
# 3. Signal-to-Noise Ratio
# ============================================================
print("\n" + "="*80)
print("SIGNAL-TO-NOISE ANALYSIS")
print("="*80)

# Overall mean
mu_all = np.vstack([chosen_reps, rejected_reps]).mean(axis=0)

# Between-class variance: how much do μ+, μ- differ?
# Var_between = 0.5 * ||μ+ - μ_all||^2 + 0.5 * ||μ- - μ_all||^2
var_between = 0.5 * np.sum((mu_pos - mu_all)**2) + 0.5 * np.sum((mu_neg - mu_all)**2)

# Total variance
var_total = np.var(np.vstack([chosen_reps, rejected_reps]), axis=0).sum()

# Within-class variance
var_within = var_total - var_between

snr = var_between / var_within if var_within > 0 else float('inf')

print(f"Total variance:          {var_total:.4f}")
print(f"Between-class variance:  {var_between:.4f}")
print(f"Within-class variance:   {var_within:.4f}")
print(f"Between/Total ratio:     {var_between / var_total:.6f} ({var_between / var_total * 100:.4f}%)")
print(f"SNR (between/within):    {snr:.6f}")


# ============================================================
# 4. Pairwise Classification Accuracy (the real test)
# ============================================================
print("\n" + "="*80)
print("PAIRWISE CLASSIFICATION ACCURACY")
print("="*80)
print("For each (chosen_i, rejected_i) pair, is R(chosen) > R(rejected)?")

# Score weight: w^T f(x,y)
score_chosen = chosen_reps @ score_weight
score_rejected = rejected_reps @ score_weight
acc_score = (score_chosen > score_rejected).mean()

# GDA: w_gda^T f(x,y) + b
bias = model.bias.item()
gda_chosen = chosen_reps @ gda_weight + bias
gda_rejected = rejected_reps @ gda_weight + bias
acc_gda = (gda_chosen > gda_rejected).mean()

# Simple mean difference (without Σ^{-1}): (μ+ - μ-)^T f(x,y)
mudiff_chosen = chosen_reps @ mu_diff
mudiff_rejected = rejected_reps @ mu_diff
acc_mudiff = (mudiff_chosen > mudiff_rejected).mean()

# Random baseline
acc_random = 0.5

print(f"Random baseline:         {acc_random:.4f} (50.00%)")
print(f"μ+ - μ- (no Σ⁻¹):      {acc_mudiff:.4f} ({acc_mudiff*100:.2f}%)")
print(f"GDA (Σ⁻¹(μ+ - μ-)):    {acc_gda:.4f} ({acc_gda*100:.2f}%)")
print(f"Score weight (trained):  {acc_score:.4f} ({acc_score*100:.2f}%)")


# ============================================================
# 5. Per-pair reward margin analysis
# ============================================================
print("\n" + "="*80)
print("REWARD MARGIN ANALYSIS (chosen - rejected)")
print("="*80)

margin_score = score_chosen - score_rejected
margin_gda = gda_chosen - gda_rejected
margin_mudiff = mudiff_chosen - mudiff_rejected

print(f"{'Method':<25} {'Mean':>10} {'Std':>10} {'Median':>10} {'% > 0':>10}")
print("-" * 65)
print(f"{'Score weight':<25} {margin_score.mean():>10.4f} {margin_score.std():>10.4f} {np.median(margin_score):>10.4f} {(margin_score > 0).mean()*100:>9.2f}%")
print(f"{'GDA':<25} {margin_gda.mean():>10.4f} {margin_gda.std():>10.4f} {np.median(margin_gda):>10.4f} {(margin_gda > 0).mean()*100:>9.2f}%")
print(f"{'μ+ - μ- (no Σ⁻¹)':<25} {margin_mudiff.mean():>10.4f} {margin_mudiff.std():>10.4f} {np.median(margin_mudiff):>10.4f} {(margin_mudiff > 0).mean()*100:>9.2f}%")

# Correlation between methods' margins
corr_score_gda = np.corrcoef(margin_score, margin_gda)[0, 1]
corr_score_mudiff = np.corrcoef(margin_score, margin_mudiff)[0, 1]

print(f"\nMargin correlation (Score vs GDA):        {corr_score_gda:.6f}")
print(f"Margin correlation (Score vs μ+ - μ-):    {corr_score_mudiff:.6f}")


# ============================================================
# 6. Projection analysis: variance along score weight direction
# ============================================================
print("\n" + "="*80)
print("PROJECTION ONTO SCORE WEIGHT DIRECTION")
print("="*80)

# Project representations onto score weight direction
w_unit = score_weight / np.linalg.norm(score_weight)
proj_chosen = chosen_reps @ w_unit
proj_rejected = rejected_reps @ w_unit

print(f"Chosen   - mean: {proj_chosen.mean():.6f}, std: {proj_chosen.std():.6f}")
print(f"Rejected - mean: {proj_rejected.mean():.6f}, std: {proj_rejected.std():.6f}")
print(f"Mean difference (along w_score): {proj_chosen.mean() - proj_rejected.mean():.6f}")
print(f"Cohen's d (effect size):         {(proj_chosen.mean() - proj_rejected.mean()) / np.sqrt((proj_chosen.var() + proj_rejected.var()) / 2):.6f}")

# Same for μ+ - μ- direction
d_unit = mu_diff / np.linalg.norm(mu_diff)
proj_chosen_d = chosen_reps @ d_unit
proj_rejected_d = rejected_reps @ d_unit

print(f"\nAlong μ+ - μ- direction:")
print(f"Chosen   - mean: {proj_chosen_d.mean():.6f}, std: {proj_chosen_d.std():.6f}")
print(f"Rejected - mean: {proj_rejected_d.mean():.6f}, std: {proj_rejected_d.std():.6f}")
print(f"Mean difference (along μ+ - μ-): {proj_chosen_d.mean() - proj_rejected_d.mean():.6f}")
print(f"Cohen's d (effect size):         {(proj_chosen_d.mean() - proj_rejected_d.mean()) / np.sqrt((proj_chosen_d.var() + proj_rejected_d.var()) / 2):.6f}")


# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. Score weight pairwise accuracy: {acc_score*100:.2f}%")
print(f"2. GDA pairwise accuracy:          {acc_gda*100:.2f}%")
print(f"3. cos(w_score, μ+ - μ-):          {cos_score_mudiff:.6f}")
print(f"4. Between-class / Total variance:  {var_between / var_total * 100:.4f}%")
print(f"5. Margin correlation (Score↔GDA):  {corr_score_gda:.6f}")
