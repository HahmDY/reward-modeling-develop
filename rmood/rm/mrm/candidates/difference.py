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


def compute_gda_reward(f, mu_d, sigma_inv):
    """
    Compute GDA reward: r = 2 * μ_d^T Σ^{-1} f

    Args:
        f:         (N, D) representations
        mu_d:      (D,)
        sigma_inv: (D, D) Σ^{-1}

    Returns:
        rewards: (N,)
        w:       (D,) reward weight vector
    """
    w = 2.0 * sigma_inv @ mu_d  # (D,)
    rewards = f @ w              # (N,)
    return rewards, w


def estimate_difference_params(D):
    """
    Estimate μ_d and Σ_d^{-1} from difference vectors d_i = chosen_i - rejected_i.

    Args:
        D: (N, D) difference vectors

    Returns:
        mu_d:      (D,)   mean of D
        sigma_inv: (D, D) Σ_d^{-1} via Ledoit-Wolf
    """
    print("\nEstimating μ_d and Σ_d from difference vectors...")
    mu_d = D.mean(axis=0)  # (D,)

    D_centered = D - mu_d
    print(f"  Fitting Ledoit-Wolf on {D.shape[0]} samples, {D.shape[1]} dimensions...")
    lw = LedoitWolf(assume_centered=True)
    lw.fit(D_centered)

    sigma_inv = lw.precision_
    print(f"  ||μ_d||: {np.linalg.norm(mu_d):.4f}")
    print(f"  Σ_d condition number: {np.linalg.cond(lw.covariance_):.2e}")
    print(f"  Ledoit-Wolf shrinkage (α): {lw.shrinkage_:.6f}")

    return mu_d, sigma_inv


def main():
    parser = argparse.ArgumentParser(
        description="Difference-based GDA reward pipeline"
    )
    parser.add_argument(
        "--chosen_path", type=str,
        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/chosen_representations.npy"
    )
    parser.add_argument(
        "--rejected_path", type=str,
        default=f"{RMOOD_HOME}/datasets/alpacafarm/rm/representations/rejected_representations.npy"
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
    # 2. Compute d_i = chosen_i - rejected_i
    # ------------------------------------------------------------------ #
    D = chosen - rejected  # (N, D)
    print(f"\nDifference vectors d_i = chosen_i - rejected_i")
    print(f"  shape: {D.shape}")
    print(f"  ||d_i|| mean: {np.linalg.norm(D, axis=1).mean():.4f}")

    # ------------------------------------------------------------------ #
    # 3. Estimate μ_d and Σ_d from D
    # ------------------------------------------------------------------ #
    mu_d, sigma_inv = estimate_difference_params(D)

    # ------------------------------------------------------------------ #
    # 4. Compute GDA rewards
    # ------------------------------------------------------------------ #
    print("\nComputing GDA rewards: r = 2 * μ_d^T Σ_d^{-1} f ...")
    chosen_rewards,   w = compute_gda_reward(chosen,   mu_d, sigma_inv)
    rejected_rewards, _ = compute_gda_reward(rejected, mu_d, sigma_inv)

    # reward on difference vector itself
    diff_rewards, _ = compute_gda_reward(D, mu_d, sigma_inv)

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
    print(f"\n  r(d_i) = 2 μ_d^T Σ_d^{{-1}} d_i")
    print(f"  r(d_i) mean: {diff_rewards.mean():.4f}  (>0 means model discriminates correctly)")

    # ------------------------------------------------------------------ #
    # 6. Visualization
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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

    # (c) r(d_i) = 2 μ_d^T Σ^{-1} d_i distribution
    ax = axes[2]
    ax.hist(diff_rewards, bins=bins, color="green", alpha=0.7,
            label=f"μ={diff_rewards.mean():.3f}")
    ax.axvline(0,                  color="black", linestyle="-",  linewidth=1.5)
    ax.axvline(diff_rewards.mean(), color="green", linestyle="--", linewidth=2)
    ax.set_xlabel("r(d_i) = 2μ_d^T Σ⁻¹ (chosen−rejected)", fontsize=11)
    ax.set_ylabel("Frequency")
    ax.set_title("Reward on Difference Vector", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Difference-based GDA Reward Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {args.output}")
    plt.show()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
