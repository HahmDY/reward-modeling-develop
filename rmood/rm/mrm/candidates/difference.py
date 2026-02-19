#!/usr/bin/env python3
"""
Difference-based GDA Reward Pipeline

d_i = chosen_i - rejected_i
μ_d = mean(d_i)
Σ_d = cov(d_i)   (Ledoit-Wolf shrinkage)

r(x, y) = 2 * μ_d^T Σ_d^{-1} f_θ(x, y)

This script:
1. Computes d_i = chosen_i - rejected_i
2. Estimates μ_d and Σ_d from {d_i}
3. Computes GDA reward for chosen and rejected representations
4. Verifies that chosen rewards > rejected rewards
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.covariance import LedoitWolf

RMOOD_HOME = os.getenv("RMOOD_HOME")


def compute_gda_reward(f, mu_d, sigma_d_inv, mu_chosen, sigma_chosen_inv, mu_rejected, sigma_rejected_inv):
    """
    Compute GDA reward: r = 2 * μ_d^T Σ^{-1} f

    Args:
        f:         (N, D) representations
        mu_d:      (D,)
        sigma_d_inv: (D, D) Σ_d^{-1}
        mu_chosen: (D,)
        sigma_chosen_inv: (D, D) Σ_chosen^{-1}

    Returns:
        rewards: (N,)
        w:       (D,) reward weight vector
    """
    odds_const = 0.0
    mahalanobis_const = 1.0
    
    w = 2.0 * sigma_d_inv @ mu_d  # (D,)
    odds_rewards = f @ w              # (N,)
    
    diff_chosen = f - mu_chosen  # (N, D)
    diff_rejected = f - mu_rejected  # (N, D)
    mahalanobis_chosen = -0.5 * np.sum(diff_chosen @ sigma_chosen_inv * diff_chosen, axis=1)  # (N,)
    mahalanobis_rejected = -0.5 * np.sum(diff_rejected @ sigma_rejected_inv * diff_rejected, axis=1)  # (N,)
    mahalanobis_rewards = mahalanobis_chosen - mahalanobis_rejected
    rewards = odds_const * odds_rewards + mahalanobis_const * mahalanobis_rewards
    
    return rewards, w


def estimate_params(chosen, rejected):
    """
    Estimate μ_d and Σ_d^{-1} from difference vectors d_i = chosen_i - rejected_i.

    Args:
        D: (N, D) difference vectors

    Returns:
        mu_d:      (D,)   mean of D
        sigma_inv: (D, D) Σ_d^{-1} via Ledoit-Wolf
    """
    print("\nEstimating μ_d and Σ_d from difference vectors...")
    D = chosen - rejected
    
    mu_chosen = chosen.mean(axis=0)
    mu_rejected = rejected.mean(axis=0)
    mu_d = D.mean(axis=0)  # (D,)

    D_centered = D - mu_d
    print(f"  Fitting Ledoit-Wolf on {D.shape[0]} samples, {D.shape[1]} dimensions...")
    lw = LedoitWolf(assume_centered=True)
    lw.fit(D_centered)
    sigma_d_inv = lw.precision_
    print(f"  ||μ_d||: {np.linalg.norm(mu_d):.4f}")
    print(f"  Σ_d condition number: {np.linalg.cond(lw.covariance_):.2e}")
    print(f"  Ledoit-Wolf shrinkage (α): {lw.shrinkage_:.6f}")
    
    chosen_centered = chosen - mu_chosen
    print(f"  fitting Ledoit-Wolf on {chosen_centered.shape[0]} samples, {chosen_centered.shape[1]} dimensions...")
    lw.fit(chosen_centered)
    sigma_chosen_inv = lw.precision_
    print(f"  ||μ_chosen||: {np.linalg.norm(mu_chosen):.4f}")
    print(f"  Σ_chosen condition number: {np.linalg.cond(lw.covariance_):.2e}")
    print(f"  Ledoit-Wolf shrinkage (α): {lw.shrinkage_:.6f}")
    
    rejected_centered = rejected - mu_rejected
    print(f"  fitting Ledoit-Wolf on {rejected_centered.shape[0]} samples, {rejected_centered.shape[1]} dimensions...")
    lw.fit(rejected_centered)
    sigma_rejected_inv = lw.precision_
    print(f"  ||μ_rejected||: {np.linalg.norm(mu_rejected):.4f}")
    print(f"  Σ_rejected condition number: {np.linalg.cond(lw.covariance_):.2e}")
    print(f"  Ledoit-Wolf shrinkage (α): {lw.shrinkage_:.6f}")

    return mu_d, mu_chosen, mu_rejected, sigma_d_inv, sigma_chosen_inv, sigma_rejected_inv


def main():
    parser = argparse.ArgumentParser(
        description="Difference-based GDA reward pipeline"
    )
    parser.add_argument(
        "--chosen_path", type=str,
        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/Hahmdong--RMOOD-qwen3-4b-alpacafarm-rm-center/chosen_representations.npy"
    )
    parser.add_argument(
        "--rejected_path", type=str,
        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/Hahmdong--RMOOD-qwen3-4b-alpacafarm-rm-center/rejected_representations.npy"
    )
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--output", type=str, default="difference_reward.png")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load representations
    # ------------------------------------------------------------------ #
    print("Loading representations...")
    chosen   = np.load(args.chosen_path)    # (N, D)
    rejected = np.load(args.rejected_path)  # (N, D)

    assert len(chosen) == len(rejected), "chosen/rejected length mismatch"
    print(f"  chosen   shape: {chosen.shape}")
    print(f"  rejected shape: {rejected.shape}")

    if args.sample_size is not None:
        idx = np.random.choice(len(chosen), size=min(args.sample_size, len(chosen)), replace=False)
        chosen   = chosen[idx]
        rejected = rejected[idx]
        print(f"  Sampled {len(chosen)} pairs")

    # ------------------------------------------------------------------ #
    # 2. Estimate μ_d and Σ_d from D
    # ------------------------------------------------------------------ #
    mu_d, mu_chosen, mu_rejected, sigma_d_inv, sigma_chosen_inv, sigma_rejected_inv = estimate_params(chosen, rejected)

    # ------------------------------------------------------------------ #
    # 3. Compute GDA rewards
    # ------------------------------------------------------------------ #
    print("\nComputing GDA rewards: r = 2 * μ_d^T Σ_d^{-1} f ...")
    chosen_rewards,   w = compute_gda_reward(chosen, mu_d, sigma_d_inv, mu_chosen, sigma_chosen_inv, mu_rejected, sigma_rejected_inv)
    rejected_rewards, _ = compute_gda_reward(rejected, mu_d, sigma_d_inv, mu_chosen, sigma_chosen_inv, mu_rejected, sigma_rejected_inv)

    # ------------------------------------------------------------------ #
    # 5. Statistics
    # ------------------------------------------------------------------ #
    acc = np.mean(chosen_rewards > rejected_rewards)
    margin = chosen_rewards - rejected_rewards  # (N,)

    print("\n" + "=" * 60)
    print("GDA Reward Statistics")
    print("=" * 60)
    print(f"  Chosen   reward: mean={chosen_rewards.mean():.4f},  std={chosen_rewards.std():.4f}")
    print(f"  Rejected reward: mean={rejected_rewards.mean():.4f},  std={rejected_rewards.std():.4f}")
    print(f"  Margin (chosen - rejected): mean={margin.mean():.4f}, std={margin.std():.4f}")
    print(f"  Accuracy (chosen > rejected): {acc:.4f} ({acc*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # 6. Visualization
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # (a) Reward distribution
    ax = axes[0]
    bins = 50
    ax.hist(chosen_rewards,   bins=bins, alpha=0.6, color="blue",
            label=f"Chosen (μ={chosen_rewards.mean():.3f})")
    ax.hist(rejected_rewards, bins=bins, alpha=0.6, color="red",
            label=f"Rejected (μ={rejected_rewards.mean():.3f})")
    ax.axvline(chosen_rewards.mean(),   color="blue", linestyle="--", linewidth=2)
    ax.axvline(rejected_rewards.mean(), color="red",  linestyle="--", linewidth=2)
    ax.set_xlabel("GDA Reward  r = 2μ_d^T Σ⁻¹ f", fontsize=11)
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Margin distribution
    ax = axes[1]
    ax.hist(margin, bins=bins, color="purple", alpha=0.7,
            label=f"μ={margin.mean():.3f}")
    ax.axvline(0,              color="black",  linestyle="-",  linewidth=1.5)
    ax.axvline(margin.mean(),  color="purple", linestyle="--", linewidth=2)
    ax.set_xlabel("Reward Margin (chosen − rejected)", fontsize=11)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Margin  (acc={acc*100:.1f}%)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
