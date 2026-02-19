#!/usr/bin/env python3
"""
PCA Visualization of Chosen vs Rejected Representations

This script loads chosen and rejected representations extracted from the MRM model
and visualizes them using PCA dimensionality reduction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import argparse
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


def prepare_data(chosen, rejected, sample_size=None, center_mode=None, weight_vector=None, prompt_repr=None, prompt_rank=None):
    """
    Combine chosen and rejected representations and create labels
    
    Args:
        chosen: numpy array of chosen representations
        rejected: numpy array of rejected representations
        sample_size: if specified, randomly sample this many examples from each class
        center_mode: "pair", "w_projection", "residualize", "regression", "prompt_subtract", or None
        weight_vector: score.weight vector
        prompt_repr: prompt-only representations (for residualize/regression/prompt_subtract)
        prompt_rank: rank for prompt dim reduction in residualize/regression mode (default: 64)
    
    Returns:
        X: combined representations (or 1D reward values if w_projection mode)
        y: labels (0 for chosen, 1 for rejected)
    """
    # Check that chosen and rejected have the same number of samples
    if len(chosen) != len(rejected):
        print(f"Warning: chosen ({len(chosen)}) and rejected ({len(rejected)}) have different lengths")
        min_len = min(len(chosen), len(rejected))
        chosen = chosen[:min_len]
        rejected = rejected[:min_len]
        print(f"Truncating to {min_len} samples")
    
    # Also truncate prompt_repr if provided
    if prompt_repr is not None and len(prompt_repr) != len(chosen):
        min_len = min(len(chosen), len(prompt_repr))
        chosen = chosen[:min_len]
        rejected = rejected[:min_len]
        prompt_repr = prompt_repr[:min_len]
        print(f"Truncating to {min_len} samples (matching prompt representations)")
    
    if sample_size is not None:
        print(f"\nSampling {sample_size} examples from each class...")
        sample_idx = np.random.choice(len(chosen), size=min(sample_size, len(chosen)), replace=False)
        chosen = chosen[sample_idx]
        rejected = rejected[sample_idx]
        if prompt_repr is not None:
            prompt_repr = prompt_repr[sample_idx]
    
    # Apply centering based on mode
    if center_mode == "w_projection":
        if weight_vector is None:
            raise ValueError("weight_vector is required when center_mode='w_projection'")
        
        print("\nProjecting onto w direction...")
        w_normalized = weight_vector / (np.linalg.norm(weight_vector) + 1e-10)
        
        chosen_proj = np.dot(chosen, w_normalized)
        rejected_proj = np.dot(rejected, w_normalized)
        
        print(f"  Chosen projection mean: {np.mean(chosen_proj):.4f}")
        print(f"  Chosen projection std: {np.std(chosen_proj):.4f}")
        print(f"  Rejected projection mean: {np.mean(rejected_proj):.4f}")
        print(f"  Rejected projection std: {np.std(rejected_proj):.4f}")
        print(f"  Difference (chosen - rejected): {np.mean(chosen_proj) - np.mean(rejected_proj):.4f}")
        
        X = np.concatenate([chosen_proj, rejected_proj])[:, np.newaxis]
        y = np.array([0] * len(chosen) + [1] * len(rejected))
        
    elif center_mode == "prompt_subtract":
        if prompt_repr is None:
            raise ValueError("prompt_repr is required when center_mode='prompt_subtract'")
        
        print("\nSubtracting prompt representation: f(x,y) - f(x)...")
        chosen_centered = chosen - prompt_repr
        rejected_centered = rejected - prompt_repr
        
        print(f"  ||f(x,y_c) - f(x)|| mean: {np.mean(np.linalg.norm(chosen_centered, axis=1)):.4f}")
        print(f"  ||f(x,y_r) - f(x)|| mean: {np.mean(np.linalg.norm(rejected_centered, axis=1)):.4f}")
        
        X = np.vstack([chosen_centered, rejected_centered])
        y = np.array([0] * len(chosen) + [1] * len(rejected))
        
    elif center_mode == "residualize":
        if prompt_repr is None:
            raise ValueError("prompt_repr is required when center_mode='residualize'")
        
        print("\nResidualizing: removing prompt subspace in feature space (SVD)...")
        P = prompt_repr  # (N, D)
        F = np.vstack([chosen, rejected])  # (2N, D)
        
        # Mean-center P so SVD captures covariance directions, not global mean
        P_mean = np.mean(P, axis=0, keepdims=True)
        P_centered = P - P_mean
        
        # SVD of centered P: V columns = prompt variation directions in feature space
        U, S, Vt = np.linalg.svd(P_centered, full_matrices=False)
        
        rank = min(prompt_rank if prompt_rank is not None else 64, Vt.shape[0])
        V_k = Vt[:rank, :].T  # (D, rank)
        
        print(f"  Prompt subspace rank: {rank}")
        print(f"  Top singular values: {S[:5]}")
        print(f"  Singular value at rank {rank}: {S[min(rank-1, len(S)-1)]:.4f}")
        if rank < len(S):
            print(f"  Singular value at rank {rank+1}: {S[min(rank, len(S)-1)]:.4f}")
        
        # Feature-space projection: F_res = F - F V_k V_k^T
        F_res = F - F @ V_k @ V_k.T
        
        explained_var = 1 - np.linalg.norm(F_res, 'fro')**2 / np.linalg.norm(F, 'fro')**2
        print(f"  Prompt variance explained (Frobenius): {explained_var:.4f}")
        print(f"  Residual shape: {F_res.shape}")
        
        X = F_res
        y = np.array([0] * len(chosen) + [1] * len(rejected))
    
    elif center_mode == "regression":
        if prompt_repr is None:
            raise ValueError("prompt_repr is required when center_mode='regression'")
        
        print("\nRegression: input-adaptive prompt removal...")
        P = prompt_repr  # (N, D)
        F = np.vstack([chosen, rejected])  # (2N, D)
        
        # Step 1: PCA reduce P to avoid overfitting
        rank = min(prompt_rank if prompt_rank is not None else 64, P.shape[0], P.shape[1])
        pca_prompt = PCA(n_components=rank)
        P_reduced = pca_prompt.fit_transform(P)  # (N, rank)
        
        print(f"  Prompt PCA rank: {rank}")
        print(f"  Prompt PCA explained variance: {np.sum(pca_prompt.explained_variance_ratio_):.4f}")
        
        # Step 2: Stack for chosen + rejected (same prompt appears twice)
        P_full = np.vstack([P_reduced, P_reduced])  # (2N, rank)
        
        # Step 3: Separate F into w-direction and orthogonal
        if weight_vector is not None:
            w_hat = weight_vector / (np.linalg.norm(weight_vector) + 1e-10)
            F_w = (F @ w_hat)[:, np.newaxis] * w_hat  # (2N, D) reward component
            F_perp = F - F_w  # (2N, D) orthogonal to w
            print(f"  Preserving reward direction (w), regressing on w-orthogonal component")
        else:
            F_w = np.zeros_like(F)
            F_perp = F
            print(f"  No weight vector provided, regressing on full F")
        
        # Step 4: Ridge regression on F_perp
        lambda_reg = 1e-3
        PtP = P_full.T @ P_full
        reg = lambda_reg * np.eye(PtP.shape[0])
        A = np.linalg.solve(PtP + reg, P_full.T @ F_perp)
        
        # Step 5: Per-sample prompt prediction and removal
        F_perp_hat = P_full @ A  # (2N, D) input-adaptive prediction
        F_perp_res = F_perp - F_perp_hat
        
        # Step 6: Recombine
        F_res = F_w + F_perp_res
        
        explained_var = 1 - np.linalg.norm(F_perp_res, 'fro')**2 / np.linalg.norm(F_perp, 'fro')**2
        print(f"  Ridge lambda = {lambda_reg}")
        print(f"  Prompt variance explained in F_perp (Frobenius): {explained_var:.4f}")
        print(f"  Residual shape: {F_res.shape}")
        
        X = F_res
        y = np.array([0] * len(chosen) + [1] * len(rejected))
        
    elif center_mode == "pair":
        print("\nCentering each (chosen, rejected) pair by their mean...")
        pair_means = (chosen + rejected) / 2.0
        chosen_centered = chosen - pair_means
        rejected_centered = rejected - pair_means
        
        X = np.vstack([chosen_centered, rejected_centered])
        y = np.array([0] * len(chosen) + [1] * len(rejected))
        
        print(f"  Applied pair-wise centering")
    else:
        X = np.vstack([chosen, rejected])
        y = np.array([0] * len(chosen) + [1] * len(rejected))
    
    print(f"\nTotal samples: {len(X)}")
    print(f"  Chosen: {np.sum(y == 0)}")
    print(f"  Rejected: {np.sum(y == 1)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    return X, y


def apply_pca(X, n_components=2, random_state=42):
    """
    Apply PCA dimensionality reduction
    
    Args:
        X: input features
        n_components: number of dimensions to reduce to (default: 2)
        random_state: random seed for reproducibility
    
    Returns:
        X_pca: PCA transformed features
        pca: fitted PCA object (to access explained variance, etc.)
    """
    print(f"\nApplying PCA...")
    print(f"  n_components: {n_components}")
    
    pca = PCA(
        n_components=n_components,
        random_state=random_state
    )
    
    X_pca = pca.fit_transform(X)
    
    print(f"PCA completed. Output shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, pca


def load_score_weight(weight_path):
    """
    Load the score.weight vector from a .npy file
    
    Args:
        weight_path: Path to weight .npy file
    
    Returns:
        weight vector as numpy array (shape: [hidden_dim])
    """
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


def project_weight_to_pca(weight_vector, pca, X_original, X_pca):
    """
    Project the weight vector onto the PCA space
    
    Args:
        weight_vector: the score.weight vector in original space (shape: [hidden_dim])
        pca: fitted PCA object
        X_original: original representations before PCA (shape: [N, hidden_dim])
        X_pca: PCA transformed representations (shape: [N, 2])
    
    Returns:
        direction in PCA space (2D vector)
    """
    print("\nProjecting weight vector to PCA space...")
    
    # Compute projections of all points onto the weight vector
    # projections[i] = dot(X_original[i], weight_vector) / ||weight_vector||
    weight_norm = np.linalg.norm(weight_vector)
    projections = np.dot(X_original, weight_vector) / (weight_norm + 1e-10)
    
    # Find correlation between projections and PCA coordinates
    # This gives us the direction in PCA space that best corresponds to the weight direction
    corr_dim1, _ = pearsonr(projections, X_pca[:, 0])
    corr_dim2, _ = pearsonr(projections, X_pca[:, 1])
    
    direction = np.array([corr_dim1, corr_dim2])
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    print(f"  Weight vector direction in PCA space: [{direction[0]:.4f}, {direction[1]:.4f}]")
    print(f"  Correlation with PC1: {corr_dim1:.4f}")
    print(f"  Correlation with PC2: {corr_dim2:.4f}")
    
    return direction, projections


def visualize_pca(X_pca, y, save_path=None, title="PCA Visualization of Chosen vs Rejected", 
                  figsize=(12, 8), alpha=0.6, s=20, weight_direction=None, projections=None,
                  explained_variance=None, show_pairs=False):
    """
    Visualize PCA results with optional weight vector direction
    
    Args:
        X_pca: PCA transformed features (N x 2)
        y: labels (0 for chosen, 1 for rejected)
        save_path: path to save the figure (optional)
        title: plot title
        figsize: figure size
        alpha: transparency of points
        s: size of points
        weight_direction: 2D direction vector of score.weight in PCA space (optional)
        projections: projection values onto weight vector for coloring (optional)
        explained_variance: tuple of (PC1_var, PC2_var) explained variance ratios
        show_pairs: if True, connect chosen-rejected pairs with dotted lines
    """
    plt.figure(figsize=figsize)
    
    # Draw pair connections first (so they appear behind points)
    if show_pairs:
        n_pairs = np.sum(y == 0)  # Number of chosen samples = number of pairs
        print(f"\nDrawing connections for {n_pairs} pairs...")
        for i in range(n_pairs):
            chosen_idx = i
            rejected_idx = i + n_pairs
            plt.plot(
                [X_pca[chosen_idx, 0], X_pca[rejected_idx, 0]],
                [X_pca[chosen_idx, 1], X_pca[rejected_idx, 1]],
                'black',
                linestyle=':',
                linewidth=1,
                alpha=0.3,
                zorder=1
            )
    
    # Plot chosen (class 0)
    chosen_mask = (y == 0)
    plt.scatter(
        X_pca[chosen_mask, 0], 
        X_pca[chosen_mask, 1],
        c='blue',
        label=f'Chosen (n={np.sum(chosen_mask)})',
        alpha=alpha,
        s=s,
        edgecolors='none',
        zorder=2
    )
    
    # Plot rejected (class 1)
    rejected_mask = (y == 1)
    plt.scatter(
        X_pca[rejected_mask, 0], 
        X_pca[rejected_mask, 1],
        c='red',
        label=f'Rejected (n={np.sum(rejected_mask)})',
        alpha=alpha,
        s=s,
        edgecolors='none',
        zorder=2
    )
    
    # Plot weight vector direction if provided
    if weight_direction is not None:
        # Get center of the plot
        center_x = np.mean(X_pca[:, 0])
        center_y = np.mean(X_pca[:, 1])
        
        # Calculate arrow length (scale to be visible)
        data_range = max(np.ptp(X_pca[:, 0]), np.ptp(X_pca[:, 1]))
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
    
    # Create axis labels with explained variance if available
    if explained_variance is not None:
        xlabel = f'PC1 ({explained_variance[0]:.2%} variance)'
        ylabel = f'PC2 ({explained_variance[1]:.2%} variance)'
    else:
        xlabel = 'PC1'
        ylabel = 'PC2'
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def visualize_w_projection(X_proj, y, save_path=None, title="W Projection: Chosen vs Rejected",
                           figsize=(12, 8), alpha=0.6, bins=50):
    """
    Visualize w projection results as histogram
    
    Args:
        X_proj: 1D projection values (N x 1)
        y: labels (0 for chosen, 1 for rejected)
        save_path: path to save the figure (optional)
        title: plot title
        figsize: figure size
        alpha: transparency of histograms
        bins: number of bins for histogram
    """
    plt.figure(figsize=figsize)
    
    # Get projection values for each class
    chosen_mask = (y == 0)
    rejected_mask = (y == 1)
    
    chosen_proj = X_proj[chosen_mask].flatten()
    rejected_proj = X_proj[rejected_mask].flatten()
    
    # Plot histograms
    plt.hist(chosen_proj, bins=bins, alpha=alpha, color='blue', label=f'Chosen (n={len(chosen_proj)})', edgecolor='black', linewidth=0.5)
    plt.hist(rejected_proj, bins=bins, alpha=alpha, color='red', label=f'Rejected (n={len(rejected_proj)})', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    chosen_mean = np.mean(chosen_proj)
    rejected_mean = np.mean(rejected_proj)
    
    plt.axvline(chosen_mean, color='blue', linestyle='--', linewidth=2, label=f'Chosen mean: {chosen_mean:.4f}')
    plt.axvline(rejected_mean, color='red', linestyle='--', linewidth=2, label=f'Rejected mean: {rejected_mean:.4f}')
    
    # Add statistics text
    stats_text = f'Δ = {chosen_mean - rejected_mean:.4f}\n'
    stats_text += f'Chosen: μ={chosen_mean:.4f}, σ={np.std(chosen_proj):.4f}\n'
    stats_text += f'Rejected: μ={rejected_mean:.4f}, σ={np.std(rejected_proj):.4f}'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Projection onto w (reward direction)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize chosen vs rejected representations using PCA'
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
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/Hahmdong--RMOOD-qwen3-4b-alpacafarm-sft/rejected_representations.npy',
        help='Path to rejected representations .npy file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pca_visualization.png',
        help='Output path for the visualization'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Number of samples to use from each class (default: use all)'
    )
    parser.add_argument(
        '--n_components',
        type=int,
        default=2,
        help='Number of PCA components (default: 2)'
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
        '--center_mode',
        type=str,
        default='pair',
        choices=['pair', 'prompt_subtract', 'w_projection', 'residualize', 'regression', 'none'],
        help='Centering mode: "pair" for pair-wise mean, "prompt_subtract" for f(x,y)-f(x), "w_projection" for reward histogram, "residualize" for SVD subspace removal, "regression" for input-adaptive prompt regression, "none" for no centering'
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/rm/representations/message_representations.npy',
        help='Path to prompt-only representations .npy file (for residualize mode)'
    )
    parser.add_argument(
        '--prompt_rank',
        type=int,
        default=64,
        help='Rank of prompt subspace to remove in residualize mode (default: 64)'
    )
    parser.add_argument(
        '--show_pairs',
        action='store_true',
        default=True,
        help='Connect chosen-rejected pairs with dotted lines'
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        default=f'{RMOOD_HOME}/datasets/alpacafarm/distribution/Hahmdong_RMOOD-qwen3-4b-alpacafarm-rm-center/weight.npy',
        help='Path to score.weight .npy file'
    )
    parser.add_argument(
        '--show_weight_direction',
        action='store_true',
        default=True,
        help='Show the direction of score.weight vector in PCA space'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    print("=" * 60)
    if args.center_mode == "w_projection":
        print("Reward Distribution Visualization")
    elif args.center_mode == "residualize":
        print("PCA Visualization (Residualized)")
    elif args.center_mode == "regression":
        print("PCA Visualization (Regression)")
    else:
        print("PCA Visualization: Chosen vs Rejected Representations")
    print("=" * 60)
    
    # Load representations
    chosen, rejected = load_representations(args.chosen_path, args.rejected_path)
    
    # Load weight vector if needed
    weight_vector = None
    if args.center_mode in ("w_projection", "regression") or args.show_weight_direction:
        weight_vector = load_score_weight(args.weight_path)
        if weight_vector is None:
            print("Error: Failed to load weight vector")
            if args.center_mode == "w_projection":
                print("Cannot use center_mode='w_projection' without weight vector. Exiting.")
                return
    
    # Load prompt representations if needed
    prompt_repr = None
    if args.center_mode in ("residualize", "regression", "prompt_subtract"):
        print(f"\nLoading prompt representations from: {args.prompt_path}")
        prompt_repr = np.load(args.prompt_path)
        print(f"  Shape: {prompt_repr.shape}")
    
    # Prepare data
    X, y = prepare_data(
        chosen, rejected, 
        sample_size=args.sample_size, 
        center_mode=args.center_mode if args.center_mode != "none" else None,
        weight_vector=weight_vector,
        prompt_repr=prompt_repr,
        prompt_rank=args.prompt_rank
    )
    
    # Validate weight vector dimension
    weight_direction = None
    projections = None
    if weight_vector is not None:
        # Get the dimension of the original representations
        original_dim = chosen.shape[1]
        if weight_vector.shape[0] != original_dim:
            print(f"Warning: weight vector dimension ({weight_vector.shape[0]}) does not match representation dimension ({original_dim})")
            weight_vector = None
    
    # Visualize
    print("\nCreating visualization...")
    
    if args.center_mode == "w_projection":
        # W projection mode: show histogram of projections
        title = "W Projection: Chosen vs Rejected"
        visualize_w_projection(
            X, 
            y, 
            save_path=args.output,
            title=title,
            figsize=tuple(args.figsize),
            alpha=args.alpha,
            bins=50
        )
    else:
        # PCA mode: apply PCA and show 2D scatter plot
        X_pca, pca = apply_pca(
            X, 
            n_components=args.n_components, 
            random_state=args.random_seed
        )
        
        # Project weight vector to PCA space if available
        if weight_vector is not None:
            weight_direction, projections = project_weight_to_pca(weight_vector, pca, X, X_pca)
        
        title = "PCA Visualization of Chosen vs Rejected"
        if args.center_mode == "pair":
            title += " (Pair-wise Centered)"
        elif args.center_mode == "residualize":
            title += " (Residualized)"
        elif args.center_mode == "regression":
            title += " (Regression, w-preserving)"
        elif args.center_mode == "prompt_subtract":
            title += " (f(x,y) - f(x))"
        
        visualize_pca(
            X_pca, 
            y, 
            save_path=args.output,
            title=title,
            figsize=tuple(args.figsize),
            alpha=args.alpha,
            s=args.point_size,
            weight_direction=weight_direction,
            projections=projections,
            explained_variance=pca.explained_variance_ratio_ if pca.explained_variance_ratio_ is not None else None,
            show_pairs=args.show_pairs
        )
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
