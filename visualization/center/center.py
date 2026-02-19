#!/usr/bin/env python3
"""
Center와 Score Weight 간 관계 분석

이 스크립트는 각 (chosen, rejected) pair의 center와 score.weight 벡터 간의 관계를 분석합니다.

문제 상황:
- PCA 시각화에서는 각 pair의 center를 빼서 pair-wise centering을 수행
- 하지만 inference time에 단일 데이터가 들어오면 어떤 center를 빼야 할지 알 수 없음

분석 목표:
- 각 pair의 center = (chosen + rejected) / 2
- score.weight와 각 center 간의 내적(dot product) 계산
- 내적 값의 분포를 히스토그램으로 시각화하여 관계 파악
- center와 score.weight가 강한 상관관계를 가지면, inference time에 score.weight를 이용해 center를 추정할 수 있을 가능성
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.decomposition import PCA

RMOOD_HOME = os.getenv('RMOOD_HOME')


def load_representations(chosen_path, rejected_path):
    """Load chosen and rejected representations from .npy files"""
    print(f"Loading chosen representations from: {chosen_path}")
    chosen = np.load(chosen_path)
    print(f"  Shape: {chosen.shape}")
    
    print(f"Loading rejected representations from: {rejected_path}")
    rejected = np.load(rejected_path)
    print(f"  Shape: {rejected.shape}")
    
    # Ensure same length
    if len(chosen) != len(rejected):
        print(f"Warning: chosen ({len(chosen)}) and rejected ({len(rejected)}) have different lengths")
        min_len = min(len(chosen), len(rejected))
        chosen = chosen[:min_len]
        rejected = rejected[:min_len]
        print(f"Truncating to {min_len} samples")
    
    return chosen, rejected


def load_score_weight(weight_path):
    """Load the score.weight vector from a .npy file"""
    print(f"\nLoading score.weight from: {weight_path}")
    
    try:
        score_weight = np.load(weight_path)
        print(f"  Score weight shape: {score_weight.shape}")
        
        # score.weight is typically (1, hidden_dim), so flatten it
        if len(score_weight.shape) == 2 and score_weight.shape[0] == 1:
            score_weight = score_weight.flatten()
        
        return score_weight
    except Exception as e:
        print(f"  Error loading weight file: {e}")
        return None


def compute_center_weight_dots(chosen, rejected, score_weight, sample_size=None):
    """
    Compute dot products between pair centers and score.weight
    
    Args:
        chosen: numpy array of chosen representations (N, D)
        rejected: numpy array of rejected representations (N, D)
        score_weight: score.weight vector (D,)
        sample_size: if specified, randomly sample this many pairs
    
    Returns:
        centers: array of pair centers (N, D)
        dot_products: array of dot products between centers and score.weight (N,)
    """
    # Sample if requested
    if sample_size is not None and sample_size < len(chosen):
        print(f"\nSampling {sample_size} pairs...")
        sample_idx = np.random.choice(len(chosen), size=sample_size, replace=False)
        chosen = chosen[sample_idx]
        rejected = rejected[sample_idx]
    
    print(f"\nComputing centers for {len(chosen)} pairs...")
    # Compute center for each pair
    centers = (chosen + rejected) / 2.0  # Shape: (N, D)
    
    print(f"Computing dot products with score.weight...")
    # Compute dot product between each center and score.weight
    dot_products = np.dot(centers, score_weight)  # Shape: (N,)
    
    print(f"\nDot product statistics:")
    print(f"  Mean: {np.mean(dot_products):.4f}")
    print(f"  Std: {np.std(dot_products):.4f}")
    print(f"  Min: {np.min(dot_products):.4f}")
    print(f"  Max: {np.max(dot_products):.4f}")
    print(f"  Median: {np.median(dot_products):.4f}")
    
    return centers, dot_products


def visualize_dot_product_distribution(dot_products, save_path=None, bins=50, figsize=(12, 8)):
    """
    Visualize the distribution of dot products as a histogram
    
    Args:
        dot_products: array of dot product values
        save_path: path to save the figure (optional)
        bins: number of histogram bins
        figsize: figure size
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Histogram
    ax = axes[0]
    n, bins_edges, patches = ax.hist(dot_products, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(dot_products), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dot_products):.4f}')
    ax.axvline(np.median(dot_products), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(dot_products):.4f}')
    ax.set_xlabel('Dot Product (center · score.weight)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Dot Products between Pair Centers and score.weight', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Box plot for additional statistics
    ax = axes[1]
    bp = ax.boxplot(dot_products, vert=False, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    ax.set_xlabel('Dot Product (center · score.weight)', fontsize=12)
    ax.set_title('Box Plot of Dot Products', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add statistics text
    stats_text = f"n = {len(dot_products)}\n"
    stats_text += f"Mean = {np.mean(dot_products):.4f}\n"
    stats_text += f"Std = {np.std(dot_products):.4f}\n"
    stats_text += f"Min = {np.min(dot_products):.4f}\n"
    stats_text += f"Max = {np.max(dot_products):.4f}\n"
    stats_text += f"Q1 = {np.percentile(dot_products, 25):.4f}\n"
    stats_text += f"Q3 = {np.percentile(dot_products, 75):.4f}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def visualize_centers_pca(centers, dot_products, score_weight, save_path=None, figsize=(14, 6)):
    """
    PCA로 center 벡터를 2D로 축소하여 시각화
    
    Args:
        centers: array of pair centers (N, D)
        dot_products: array of dot products for coloring (N,)
        score_weight: score.weight vector (D,) — PCA 공간에 투영하여 방향 표시
        save_path: path to save the figure (optional)
        figsize: figure size
    """
    print("\nRunning PCA on centers...")
    pca = PCA(n_components=2)
    centers_2d = pca.fit_transform(centers)
    explained = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={explained[0]:.4f}, PC2={explained[1]:.4f} "
          f"(total={explained.sum():.4f})")

    # score.weight를 PCA 공간에 투영 (방향 벡터로 사용)
    weight_2d = pca.transform(score_weight.reshape(1, -1))[0]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- 왼쪽: dot product로 컬러링 ---
    ax = axes[0]
    sc = ax.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        c=dot_products, cmap='coolwarm', alpha=0.6, s=10, linewidths=0
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('center · score.weight', fontsize=10)

    # score.weight 방향 화살표 (정규화)
    arrow_scale = (centers_2d[:, 0].max() - centers_2d[:, 0].min()) * 0.3
    w_norm = weight_2d / (np.linalg.norm(weight_2d) + 1e-8) * arrow_scale
    ax.annotate(
        '', xy=(w_norm[0], w_norm[1]), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='black', lw=2)
    )
    ax.text(w_norm[0] * 1.1, w_norm[1] * 1.1, 'score.weight', fontsize=9, color='black')

    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=11)
    ax.set_title('Centers PCA (colored by dot product)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- 오른쪽: 밀도 기반 시각화 (단색) ---
    ax = axes[1]
    ax.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        alpha=0.3, s=8, color='steelblue', linewidths=0
    )
    # 전체 평균 center 표시
    mean_2d = centers_2d.mean(axis=0)
    ax.scatter(*mean_2d, color='red', s=80, zorder=5, label=f'Mean center')
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=11)
    ax.set_title('Centers PCA (density view)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 통계 텍스트
    stats_text = (
        f"n = {len(centers)}\n"
        f"PC1 var = {explained[0]*100:.2f}%\n"
        f"PC2 var = {explained[1]*100:.2f}%\n"
        f"Total var = {explained.sum()*100:.2f}%"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  PCA figure saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze relationship between pair centers and score.weight'
    )
    parser.add_argument(
        '--chosen_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/Hahmdong--RMOOD-qwen3-4b-alpacafarm-sft/chosen_representations.npy',
        help='Path to chosen representations .npy file'
    )
    parser.add_argument(
        '--rejected_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/Hahmdong--RMOOD-qwen3-4b-alpacafarm-rm/rejected_representations.npy',
        help='Path to rejected representations .npy file'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/distribution/Hahmdong_RMOOD-qwen3-4b-alpacafarm-rm/weight.npy',
        help='Path to score.weight .npy file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='center_weight_dot_distribution.png',
        help='Output path for the visualization'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Number of pairs to analyze (default: use all)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of histogram bins (default: 50)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[12, 8],
        help='Figure size (width height) (default: 12 8)'
    )
    parser.add_argument(
        '--pca_output',
        type=str,
        default='center_pca.png',
        help='Output path for PCA visualization (default: center_pca.png)'
    )
    parser.add_argument(
        '--no_pca',
        action='store_true',
        help='Skip PCA visualization'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    print("=" * 80)
    print("Center-Weight Relationship Analysis")
    print("=" * 80)
    
    # Load representations
    chosen, rejected = load_representations(args.chosen_path, args.rejected_path)
    
    # Load score.weight
    score_weight = load_score_weight(args.weight_path)
    
    if score_weight is None:
        print("Error: Failed to load score.weight")
        return
    
    # Check dimension compatibility
    if score_weight.shape[0] != chosen.shape[1]:
        print(f"Error: score.weight dimension ({score_weight.shape[0]}) does not match representation dimension ({chosen.shape[1]})")
        return
    
    # Compute dot products
    centers, dot_products = compute_center_weight_dots(
        chosen, rejected, score_weight, sample_size=args.sample_size
    )
    
    # Visualize dot product distribution
    print("\nCreating dot product distribution visualization...")
    visualize_dot_product_distribution(
        dot_products,
        save_path=args.output,
        bins=args.bins,
        figsize=tuple(args.figsize)
    )

    # PCA visualization of centers
    if not args.no_pca:
        print("\nCreating PCA visualization of centers...")
        visualize_centers_pca(
            centers,
            dot_products,
            score_weight,
            save_path=args.pca_output,
            figsize=(14, 6)
        )

    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
