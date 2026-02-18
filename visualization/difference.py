#!/usr/bin/env python3
"""
PCA Visualization of Chosen - Rejected Difference Vectors

This script computes the difference (chosen - rejected) for each pair,
then applies PCA to visualize the structure of these difference vectors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
from scipy.stats import pearsonr, shapiro, anderson, normaltest, probplot

RMOOD_HOME = os.getenv('RMOOD_HOME')


def load_representations(chosen_path, rejected_path):
    """Load chosen and rejected representations from .npy files"""
    print(f"Loading chosen representations from: {chosen_path}")
    chosen = np.load(chosen_path)
    print(f"  Shape: {chosen.shape}")

    print(f"Loading rejected representations from: {rejected_path}")
    rejected = np.load(rejected_path)
    print(f"  Shape: {rejected.shape}")

    if len(chosen) != len(rejected):
        print(f"Warning: chosen ({len(chosen)}) and rejected ({len(rejected)}) have different lengths")
        min_len = min(len(chosen), len(rejected))
        chosen = chosen[:min_len]
        rejected = rejected[:min_len]
        print(f"Truncating to {min_len} samples")

    return chosen, rejected


def compute_differences(chosen, rejected):
    """Compute chosen - rejected difference vectors"""
    diff = chosen - rejected
    print(f"\nDifference vectors computed.")
    print(f"  Shape: {diff.shape}")
    print(f"  Mean norm: {np.mean(np.linalg.norm(diff, axis=1)):.4f}")
    print(f"  Std norm:  {np.std(np.linalg.norm(diff, axis=1)):.4f}")
    return diff


def load_score_weight(weight_path):
    """Load the score.weight vector from a .npy file"""
    print(f"\nLoading score.weight from: {weight_path}")
    try:
        w = np.load(weight_path)
        print(f"  Score weight shape: {w.shape}")
        if len(w.shape) == 2 and w.shape[0] == 1:
            w = w.flatten()
        return w
    except Exception as e:
        print(f"  Error loading weight file: {e}")
        return None


def test_gaussianity(data, label, ax_hist, ax_qq):
    """
    Test and visualize whether 1D data follows a Gaussian distribution.

    Draws histogram + fitted Gaussian on ax_hist, Q-Q plot on ax_qq.
    Returns dict of test results.
    """
    mu, sigma = np.mean(data), np.std(data)

    # --- Histogram + Gaussian fit ---
    ax_hist.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white', linewidth=0.5)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax_hist.plot(x, pdf, 'r-', linewidth=2, label='Gaussian fit')
    ax_hist.set_title(f'{label}', fontsize=11, fontweight='bold')
    ax_hist.set_ylabel('Density')
    ax_hist.legend(fontsize=9)

    # --- Q-Q plot ---
    probplot(data, dist='norm', plot=ax_qq)
    ax_qq.set_title(f'Q-Q Plot: {label}', fontsize=11, fontweight='bold')
    ax_qq.get_lines()[0].set(markersize=3, alpha=0.5)
    ax_qq.get_lines()[1].set(color='red', linewidth=2)

    # --- Statistical tests ---
    results = {'label': label, 'mu': mu, 'sigma': sigma}

    # Shapiro-Wilk (max 5000 samples)
    n_shapiro = min(len(data), 5000)
    sample = np.random.choice(data, size=n_shapiro, replace=False) if len(data) > 5000 else data
    sw_stat, sw_p = shapiro(sample)
    results['shapiro_stat'] = sw_stat
    results['shapiro_p'] = sw_p

    # D'Agostino-Pearson (needs n >= 20)
    if len(data) >= 20:
        dp_stat, dp_p = normaltest(data)
        results['dagostino_stat'] = dp_stat
        results['dagostino_p'] = dp_p

    # Anderson-Darling
    ad_result = anderson(data, dist='norm')
    results['anderson_stat'] = ad_result.statistic
    results['anderson_cv'] = dict(zip(ad_result.significance_level, ad_result.critical_values))

    # Add stats text to histogram
    ad_reject = ad_result.statistic > ad_result.critical_values[2]  # 5% level
    verdict = "NOT Gaussian" if (sw_p < 0.05 or ad_reject) else "Gaussian"
    stats_text = (
        f'μ = {mu:.4f}, σ = {sigma:.4f}\n'
        f'Shapiro p = {sw_p:.4g}\n'
        f'Anderson A² = {ad_result.statistic:.4f}\n'
        f'→ {verdict} (α=0.05)'
    )
    ax_hist.text(
        0.97, 0.97, stats_text, transform=ax_hist.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    return results


def visualize_gaussianity(diff, weight_vector=None, reward_scores=None,
                          save_path=None, random_state=42):
    """
    Create a multi-panel figure testing Gaussianity of PC1, PC2, and reward margin.
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(diff)

    targets = [
        (X_pca[:, 0], f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)'),
        (X_pca[:, 1], f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)'),
    ]
    if reward_scores is not None:
        targets.append((reward_scores, 'Reward margin (w · diff)'))

    norms = np.linalg.norm(diff, axis=1)
    targets.append((norms, '||diff|| (L2 norm)'))

    n = len(targets)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    print("\n" + "=" * 60)
    print("Gaussianity Tests")
    print("=" * 60)

    all_results = []
    for i, (data, label) in enumerate(targets):
        res = test_gaussianity(data, label, axes[i, 0], axes[i, 1])
        all_results.append(res)

        print(f"\n--- {label} ---")
        print(f"  μ = {res['mu']:.4f}, σ = {res['sigma']:.4f}")
        print(f"  Shapiro-Wilk:      W = {res['shapiro_stat']:.6f}, p = {res['shapiro_p']:.4g}")
        if 'dagostino_stat' in res:
            print(f"  D'Agostino-Pearson: K² = {res['dagostino_stat']:.4f}, p = {res['dagostino_p']:.4g}")
        print(f"  Anderson-Darling:  A² = {res['anderson_stat']:.4f}")
        for sig, cv in res['anderson_cv'].items():
            tag = " ← REJECT" if res['anderson_stat'] > cv else ""
            print(f"    {sig}% critical value: {cv:.4f}{tag}")

    fig.suptitle('Gaussianity Test: Difference Vectors (Chosen − Rejected)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        gauss_path = f'{base}_gaussian{ext}'
        fig.savefig(gauss_path, dpi=300, bbox_inches='tight')
        print(f"\nGaussianity figure saved to: {gauss_path}")

    plt.show()
    return all_results


def visualize_difference_pca(diff, save_path=None, title="PCA of Difference Vectors (Chosen − Rejected)",
                             figsize=(12, 8), alpha=0.6, s=20, weight_vector=None,
                             reward_scores=None, random_state=42):
    """
    Apply PCA to difference vectors and visualize.

    Args:
        diff: difference vectors (N, D)
        save_path: path to save figure
        title: plot title
        figsize: figure size
        alpha: point transparency
        s: point size
        weight_vector: score.weight vector for direction overlay
        reward_scores: per-pair reward margin (w·diff) for coloring
        random_state: random seed
    """
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(diff)

    print(f"\nPCA on difference vectors:")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    fig, ax = plt.subplots(figsize=figsize)

    if reward_scores is not None:
        sc = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=reward_scores, cmap='RdYlGn', alpha=alpha, s=s, edgecolors='none'
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Reward margin (w · diff)', fontsize=11)
    else:
        norms = np.linalg.norm(diff, axis=1)
        sc = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=norms, cmap='viridis', alpha=alpha, s=s, edgecolors='none'
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('||chosen − rejected||', fontsize=11)

    # Overlay weight direction in PCA space
    if weight_vector is not None:
        w_norm = weight_vector / (np.linalg.norm(weight_vector) + 1e-10)
        projections = np.dot(diff, w_norm)
        corr1, _ = pearsonr(projections, X_pca[:, 0])
        corr2, _ = pearsonr(projections, X_pca[:, 1])
        direction = np.array([corr1, corr2])
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        cx, cy = np.mean(X_pca[:, 0]), np.mean(X_pca[:, 1])
        data_range = max(np.ptp(X_pca[:, 0]), np.ptp(X_pca[:, 1]))
        arrow_scale = data_range * 0.3

        ax.arrow(
            cx, cy,
            direction[0] * arrow_scale, direction[1] * arrow_scale,
            head_width=data_range * 0.04, head_length=data_range * 0.04,
            fc='green', ec='green', linewidth=3, alpha=0.8,
            length_includes_head=True, zorder=5
        )
        ax.text(
            cx + direction[0] * arrow_scale * 1.2,
            cy + direction[1] * arrow_scale * 1.2,
            'score.weight', fontsize=11, fontweight='bold', color='green',
            ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
        )

        print(f"  Weight direction in PCA space: [{direction[0]:.4f}, {direction[1]:.4f}]")
        print(f"  Corr(w·diff, PC1): {corr1:.4f}")
        print(f"  Corr(w·diff, PC2): {corr2:.4f}")

    # Add origin marker
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev[1]:.2%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Stats text box
    norms = np.linalg.norm(diff, axis=1)
    stats = f'N = {len(diff)}\n'
    stats += f'Mean ||diff|| = {np.mean(norms):.4f}\n'
    stats += f'Std ||diff|| = {np.std(norms):.4f}'
    if reward_scores is not None:
        stats += f'\nMean reward margin = {np.mean(reward_scores):.4f}'
        stats += f'\n% margin > 0: {np.mean(reward_scores > 0):.2%}'
    ax.text(
        0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='PCA visualization of chosen - rejected difference vectors'
    )
    parser.add_argument(
        '--chosen_path', type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/chosen_representations.npy',
        help='Path to chosen representations .npy file'
    )
    parser.add_argument(
        '--rejected_path', type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/rejected_representations.npy',
        help='Path to rejected representations .npy file'
    )
    parser.add_argument(
        '--weight_path', type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/distribution/Hahmdong_RMOOD-qwen3-4b-alpacafarm-rm/weight.npy',
        help='Path to score.weight .npy file'
    )
    parser.add_argument(
        '--output', type=str, default='difference_pca.png',
        help='Output path for the visualization'
    )
    parser.add_argument(
        '--sample_size', type=int, default=None,
        help='Randomly sample this many pairs (default: use all)'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--figsize', type=int, nargs=2, default=[12, 8],
        help='Figure size (width height)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.6,
        help='Point transparency (default: 0.6)'
    )
    parser.add_argument(
        '--point_size', type=int, default=20,
        help='Point size (default: 20)'
    )
    parser.add_argument(
        '--no_weight', action='store_true',
        help='Do not load or show score.weight direction'
    )
    parser.add_argument(
        '--test_gaussian', action='store_true', default=True,
        help='Run Gaussianity tests on PC1, PC2, reward margin, and ||diff|| (default: True)'
    )
    parser.add_argument(
        '--no_test_gaussian', action='store_true',
        help='Disable Gaussianity tests'
    )

    args = parser.parse_args()
    np.random.seed(args.random_seed)

    print("=" * 60)
    print("PCA of Difference Vectors (Chosen − Rejected)")
    print("=" * 60)

    chosen, rejected = load_representations(args.chosen_path, args.rejected_path)
    diff = compute_differences(chosen, rejected)

    # Optional sampling
    if args.sample_size is not None and args.sample_size < len(diff):
        idx = np.random.choice(len(diff), size=args.sample_size, replace=False)
        diff = diff[idx]
        print(f"Sampled {args.sample_size} pairs")

    # Load weight vector
    weight_vector = None
    reward_scores = None
    if not args.no_weight:
        weight_vector = load_score_weight(args.weight_path)
        if weight_vector is not None and weight_vector.shape[0] == diff.shape[1]:
            w_norm = weight_vector / (np.linalg.norm(weight_vector) + 1e-10)
            reward_scores = np.dot(diff, w_norm)
            print(f"\nReward margin stats:")
            print(f"  Mean: {np.mean(reward_scores):.4f}")
            print(f"  Std:  {np.std(reward_scores):.4f}")
            print(f"  % positive (chosen > rejected): {np.mean(reward_scores > 0):.2%}")
        elif weight_vector is not None:
            print(f"Warning: weight dim ({weight_vector.shape[0]}) != repr dim ({diff.shape[1]}), skipping")
            weight_vector = None

    visualize_difference_pca(
        diff,
        save_path=args.output,
        figsize=tuple(args.figsize),
        alpha=args.alpha,
        s=args.point_size,
        weight_vector=weight_vector,
        reward_scores=reward_scores,
        random_state=args.random_seed
    )

    # Gaussianity tests
    if args.test_gaussian and not args.no_test_gaussian:
        visualize_gaussianity(
            diff,
            weight_vector=weight_vector,
            reward_scores=reward_scores,
            save_path=args.output,
            random_state=args.random_seed
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
