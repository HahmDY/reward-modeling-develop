#!/usr/bin/env python3
"""
t-SNE Visualization of Chosen vs Rejected Representations

This script loads chosen and rejected representations extracted from the MRM model
and visualizes them using t-SNE dimensionality reduction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse

RMOOD_HOME = os.getenv('RMOOD_HOME')

def load_representations(chosen_path, rejected_path):
    """Load chosen and rejected representations from .npy files"""
    print(f"Loading chosen representations from: {chosen_path}")
    chosen = np.load(chosen_path)
    print(f"  Shape: {chosen.shape}")
    
    print(f"Loading rejected representations from: {rejected_path}")
    rejected = np.load(rejected_path)
    print(f"  Shape: {rejected.shape}")
    
    return chosen, rejected


def prepare_data(chosen, rejected, sample_size=None, center_by_prompt=False):
    """
    Combine chosen and rejected representations and create labels
    
    Args:
        chosen: numpy array of chosen representations
        rejected: numpy array of rejected representations
        sample_size: if specified, randomly sample this many examples from each class
        center_by_prompt: if True, center each (chosen, rejected) pair by their mean
    
    Returns:
        X: combined representations
        y: labels (0 for chosen, 1 for rejected)
    """
    # Check that chosen and rejected have the same number of samples
    if len(chosen) != len(rejected):
        print(f"Warning: chosen ({len(chosen)}) and rejected ({len(rejected)}) have different lengths")
        min_len = min(len(chosen), len(rejected))
        chosen = chosen[:min_len]
        rejected = rejected[:min_len]
        print(f"Truncating to {min_len} samples")
    
    if sample_size is not None:
        print(f"\nSampling {sample_size} examples from each class...")
        # Sample the same indices to maintain pairing
        sample_idx = np.random.choice(len(chosen), size=min(sample_size, len(chosen)), replace=False)
        chosen = chosen[sample_idx]
        rejected = rejected[sample_idx]
    
    # Center by prompt if requested
    if center_by_prompt:
        print("\nCentering each (chosen, rejected) pair by their mean...")
        # Calculate mean for each pair
        pair_means = (chosen + rejected) / 2.0  # Shape: (N, D)
        
        # Center each pair
        chosen_centered = chosen - pair_means
        rejected_centered = rejected - pair_means
        
        # Combine representations
        X = np.vstack([chosen_centered, rejected_centered])
        
        print(f"  Applied pair-wise centering")
    else:
        # Combine representations without centering
        X = np.vstack([chosen, rejected])
    
    # Create labels (0 for chosen, 1 for rejected)
    y = np.array([0] * len(chosen) + [1] * len(rejected))
    
    print(f"\nTotal samples: {len(X)}")
    print(f"  Chosen: {np.sum(y == 0)}")
    print(f"  Rejected: {np.sum(y == 1)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    return X, y


def apply_tsne(X, n_components=2, perplexity=30, random_state=42, n_iter=1000):
    """
    Apply t-SNE dimensionality reduction
    
    Args:
        X: input features
        n_components: number of dimensions to reduce to (default: 2)
        perplexity: t-SNE perplexity parameter (default: 30)
        random_state: random seed for reproducibility
        n_iter: number of iterations (default: 1000)
    
    Returns:
        X_tsne: t-SNE transformed features
    """
    print(f"\nApplying t-SNE...")
    print(f"  n_components: {n_components}")
    print(f"  perplexity: {perplexity}")
    print(f"  n_iter: {n_iter}")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        verbose=1
    )
    
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE completed. Output shape: {X_tsne.shape}")
    
    return X_tsne


def visualize_tsne(X_tsne, y, save_path=None, title="t-SNE Visualization of Chosen vs Rejected", 
                   figsize=(12, 8), alpha=0.6, s=20):
    """
    Visualize t-SNE results
    
    Args:
        X_tsne: t-SNE transformed features (N x 2)
        y: labels (0 for chosen, 1 for rejected)
        save_path: path to save the figure (optional)
        title: plot title
        figsize: figure size
        alpha: transparency of points
        s: size of points
    """
    plt.figure(figsize=figsize)
    
    # Plot chosen (class 0)
    chosen_mask = (y == 0)
    plt.scatter(
        X_tsne[chosen_mask, 0], 
        X_tsne[chosen_mask, 1],
        c='blue',
        label=f'Chosen (n={np.sum(chosen_mask)})',
        alpha=alpha,
        s=s,
        edgecolors='none'
    )
    
    # Plot rejected (class 1)
    rejected_mask = (y == 1)
    plt.scatter(
        X_tsne[rejected_mask, 0], 
        X_tsne[rejected_mask, 1],
        c='red',
        label=f'Rejected (n={np.sum(rejected_mask)})',
        alpha=alpha,
        s=s,
        edgecolors='none'
    )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize chosen vs rejected representations using t-SNE'
    )
    parser.add_argument(
        '--chosen_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/chosen_representations.npy',
        help='Path to chosen representations .npy file'
    )
    parser.add_argument(
        '--rejected_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/rejected_representations.npy',
        help='Path to rejected representations .npy file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tsne_visualization.png',
        help='Output path for the visualization'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=500,
        help='Number of samples to use from each class (default: 5000)'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1000,
        help='Number of t-SNE iterations (default: 1000)'
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
        '--alpha',
        type=float,
        default=0.6,
        help='Transparency of points (default: 0.6)'
    )
    parser.add_argument(
        '--point_size',
        type=int,
        default=20,
        help='Size of points (default: 20)'
    )
    parser.add_argument(
        '--center_by_prompt',
        action='store_true',
        default=True,
        help='Center each (chosen, rejected) pair by their mean before t-SNE'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    print("=" * 60)
    print("t-SNE Visualization: Chosen vs Rejected Representations")
    print("=" * 60)
    
    # Load representations
    chosen, rejected = load_representations(args.chosen_path, args.rejected_path)
    
    # Prepare data
    X, y = prepare_data(chosen, rejected, sample_size=args.sample_size, center_by_prompt=args.center_by_prompt)
    
    # Apply t-SNE
    X_tsne = apply_tsne(
        X, 
        n_components=2, 
        perplexity=args.perplexity,
        random_state=args.random_seed,
        n_iter=args.n_iter
    )
    
    # Visualize
    print("\nCreating visualization...")
    title = "t-SNE Visualization of Chosen vs Rejected"
    if args.center_by_prompt:
        title += " (Pair-wise Centered)"
    
    visualize_tsne(
        X_tsne, 
        y, 
        save_path=args.output,
        title=title,
        figsize=tuple(args.figsize),
        alpha=args.alpha,
        s=args.point_size
    )
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
