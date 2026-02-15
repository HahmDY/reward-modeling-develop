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
import torch
from transformers import AutoModelForSequenceClassification
from scipy.stats import pearsonr

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


def load_score_weight(model_name_or_path):
    """
    Load the score.weight vector from a reward model
    
    Args:
        model_name_or_path: HuggingFace model name or local path
    
    Returns:
        weight vector as numpy array (shape: [hidden_dim])
    """
    print(f"\nLoading score.weight from model: {model_name_or_path}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32
        )
        
        # Get the score weight vector
        score_weight = model.score.weight.detach().cpu().numpy()
        print(f"  Score weight shape: {score_weight.shape}")
        
        # score.weight is typically (1, hidden_dim), so flatten it
        if len(score_weight.shape) == 2 and score_weight.shape[0] == 1:
            score_weight = score_weight.flatten()
        
        return score_weight
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None


def project_weight_to_tsne(weight_vector, X_original, X_tsne):
    """
    Project the weight vector onto the t-SNE space by finding its direction
    
    Args:
        weight_vector: the score.weight vector in original space (shape: [hidden_dim])
        X_original: original representations before t-SNE (shape: [N, hidden_dim])
        X_tsne: t-SNE transformed representations (shape: [N, 2])
    
    Returns:
        direction in t-SNE space (2D vector)
    """
    print("\nProjecting weight vector to t-SNE space...")
    
    # Compute projections of all points onto the weight vector
    # projections[i] = dot(X_original[i], weight_vector) / ||weight_vector||
    weight_norm = np.linalg.norm(weight_vector)
    projections = np.dot(X_original, weight_vector) / (weight_norm + 1e-10)
    
    # Find correlation between projections and t-SNE coordinates
    # This gives us the direction in t-SNE space that best corresponds to the weight direction
    corr_dim1, _ = pearsonr(projections, X_tsne[:, 0])
    corr_dim2, _ = pearsonr(projections, X_tsne[:, 1])
    
    direction = np.array([corr_dim1, corr_dim2])
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    print(f"  Weight vector direction in t-SNE space: [{direction[0]:.4f}, {direction[1]:.4f}]")
    print(f"  Correlation with t-SNE dim1: {corr_dim1:.4f}")
    print(f"  Correlation with t-SNE dim2: {corr_dim2:.4f}")
    
    return direction, projections


def visualize_tsne(X_tsne, y, save_path=None, title="t-SNE Visualization of Chosen vs Rejected", 
                   figsize=(12, 8), alpha=0.6, s=20, weight_direction=None, projections=None):
    """
    Visualize t-SNE results with optional weight vector direction
    
    Args:
        X_tsne: t-SNE transformed features (N x 2)
        y: labels (0 for chosen, 1 for rejected)
        save_path: path to save the figure (optional)
        title: plot title
        figsize: figure size
        alpha: transparency of points
        s: size of points
        weight_direction: 2D direction vector of score.weight in t-SNE space (optional)
        projections: projection values onto weight vector for coloring (optional)
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
    
    # Plot weight vector direction if provided
    if weight_direction is not None:
        # Get center of the plot
        center_x = np.mean(X_tsne[:, 0])
        center_y = np.mean(X_tsne[:, 1])
        
        # Calculate arrow length (scale to be visible)
        data_range = max(np.ptp(X_tsne[:, 0]), np.ptp(X_tsne[:, 1]))
        arrow_scale = data_range * 0.3
        
        # Draw arrow
        plt.arrow(
            center_x, center_y,
            weight_direction[0] * arrow_scale,
            weight_direction[1] * arrow_scale,
            head_width=data_range * 0.05,
            head_length=data_range * 0.05,
            fc='green',
            ec='green',
            linewidth=3,
            alpha=0.8,
            length_includes_head=True,
            label='score.weight direction',
            zorder=5
        )
        
        # Add text annotation
        text_x = center_x + weight_direction[0] * arrow_scale * 1.2
        text_y = center_y + weight_direction[1] * arrow_scale * 1.2
        plt.text(
            text_x, text_y,
            'score.weight',
            fontsize=11,
            fontweight='bold',
            color='green',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
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
        default=30,
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
        default=False,
        help='Center each (chosen, rejected) pair by their mean before t-SNE'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm',
        help='HuggingFace model name or path to load score.weight from'
    )
    parser.add_argument(
        '--show_weight_direction',
        action='store_true',
        default=True,
        help='Show the direction of score.weight vector in t-SNE space'
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
    
    # Load score.weight if requested
    weight_direction = None
    projections = None
    if args.show_weight_direction:
        weight_vector = load_score_weight(args.model_name)
        if weight_vector is not None and weight_vector.shape[0] == X.shape[1]:
            # We'll compute the direction after t-SNE
            pass
        elif weight_vector is not None:
            print(f"Warning: weight vector dimension ({weight_vector.shape[0]}) does not match representation dimension ({X.shape[1]})")
            weight_vector = None
    else:
        weight_vector = None
    
    # Apply t-SNE
    X_tsne = apply_tsne(
        X, 
        n_components=2, 
        perplexity=args.perplexity,
        random_state=args.random_seed,
        n_iter=args.n_iter
    )
    
    # Project weight vector to t-SNE space if available
    if weight_vector is not None:
        weight_direction, projections = project_weight_to_tsne(weight_vector, X, X_tsne)
    
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
        s=args.point_size,
        weight_direction=weight_direction,
        projections=projections
    )
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
