import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT, METHOD_COLORS,
    BASELINE_METHODS, HYBRID_COMBINATIONS
)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# D-Statistic Curves
# =============================================================================

def plot_d_statistic_vs_ratio(results_df: pd.DataFrame,
                               dataset: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot D-statistic vs sampling ratio for all methods.
    Similar to Figure 3 in Leskovec & Faloutsos (2006).
    
    Args:
        results_df: DataFrame with experiment results
        dataset: Specific dataset to plot (None for average across all)
        save_path: Path to save figure (None to show)
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter by dataset if specified
    if dataset:
        df = results_df[results_df['dataset'] == dataset].copy()
        title_suffix = f" ({dataset})"
    else:
        df = results_df.copy()
        title_suffix = " (All Datasets)"
    
    # Separate baseline and hybrid methods
    baseline_df = df[df['method_type'] == 'baseline']
    hybrid_df = df[df['method_type'] == 'hybrid']
    
    # ===== Left plot: Baseline methods =====
    ax1 = axes[0]
    
    for method in BASELINE_METHODS:
        method_data = baseline_df[baseline_df['method'] == method]
        if len(method_data) == 0:
            continue
        
        # Group by ratio and compute mean
        grouped = method_data.groupby('ratio')['AVG'].agg(['mean', 'std'])
        
        color = METHOD_COLORS.get(method, None)
        ax1.errorbar(
            grouped.index * 100, grouped['mean'],
            yerr=grouped['std'],
            marker='o', label=method, color=color,
            capsize=3, linewidth=2, markersize=8
        )
    
    ax1.set_xlabel('Sampling Ratio (%)', fontsize=12)
    ax1.set_ylabel('Average D-Statistic', fontsize=12)
    ax1.set_title(f'Baseline Methods{title_suffix}', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # ===== Right plot: Hybrid vs Best Baseline =====
    ax2 = axes[1]
    
    # Plot best baselines (RW, FF) for reference
    for method in ['RW', 'FF']:
        method_data = baseline_df[baseline_df['method'] == method]
        if len(method_data) == 0:
            continue
        
        grouped = method_data.groupby('ratio')['AVG'].mean()
        ax2.plot(
            grouped.index * 100, grouped.values,
            'o--', label=f'{method} (baseline)',
            linewidth=2, markersize=8, alpha=0.7
        )
    
    # Plot hybrid methods (best alpha for each)
    hybrid_names = []
    for node_m, explore_m in HYBRID_COMBINATIONS:
        hybrid_names.append(f"HYB-{node_m}-{explore_m}")
    
    for hybrid_base in hybrid_names:
        hybrid_data = hybrid_df[hybrid_df['method'].str.contains(hybrid_base)]
        if len(hybrid_data) == 0:
            continue
        
        # Find best alpha for each ratio
        best_per_ratio = hybrid_data.groupby('ratio')['AVG'].min()
        
        color = METHOD_COLORS.get(hybrid_base, None)
        ax2.plot(
            best_per_ratio.index * 100, best_per_ratio.values,
            's-', label=hybrid_base, color=color,
            linewidth=2, markersize=6, alpha=0.8
        )
    
    ax2.set_xlabel('Sampling Ratio (%)', fontsize=12)
    ax2.set_ylabel('Average D-Statistic', fontsize=12)
    ax2.set_title(f'Hybrid Methods vs Baselines{title_suffix}', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Property Heatmap
# =============================================================================

def plot_property_heatmap(results_df: pd.DataFrame,
                           ratio: float = 0.15,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing D-statistics for each method and property.
    
    Args:
        results_df: DataFrame with experiment results
        ratio: Sampling ratio to display
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Filter by ratio
    df = results_df[results_df['ratio'] == ratio].copy()
    
    # Properties to display
    properties = ['in_degree', 'out_degree', 'wcc', 'scc', 
                  'hop_plot', 'singular_val', 'singular_vec', 'clustering']
    
    # Get unique methods and compute mean per method
    methods = df['method'].unique()
    heatmap_data = []
    method_labels = []
    
    for method in sorted(methods):
        method_data = df[df['method'] == method][properties].mean()
        heatmap_data.append(method_data.values)
        method_labels.append(method)
    
    heatmap_df = pd.DataFrame(
        heatmap_data, 
        index=method_labels, 
        columns=properties
    )
    
    # Sort by average
    heatmap_df['AVG'] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values('AVG')
    heatmap_df = heatmap_df.drop('AVG', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(methods) * 0.4)))
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Red (bad) to Green (good)
        center=0.5,
        ax=ax,
        cbar_kws={'label': 'D-Statistic (lower is better)'}
    )
    
    ax.set_title(f'D-Statistics by Method and Property (Ratio = {ratio*100:.0f}%)', 
                 fontsize=14)
    ax.set_xlabel('Graph Property', fontsize=12)
    ax.set_ylabel('Sampling Method', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Distribution Comparison
# =============================================================================

def plot_distribution_comparison(G: nx.Graph,
                                  samples: Dict[str, nx.Graph],
                                  property_name: str = "degree",
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare distributions between original graph and samples.
    Similar to Figures 1, 2 in original paper.
    
    Args:
        G: Original graph
        samples: Dictionary mapping method names to sampled graphs
        property_name: Property to compare ("degree", "clustering")
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Left plot: Degree distribution =====
    ax1 = axes[0]
    
    # Original graph
    if G.is_directed():
        degrees_orig = [d for n, d in G.in_degree()]
    else:
        degrees_orig = [d for n, d in G.degree()]
    
    # Plot original
    ax1.hist(degrees_orig, bins=50, alpha=0.5, label='Original', 
             density=True, color='black')
    
    # Plot samples
    colors = plt.cm.tab10(np.linspace(0, 1, len(samples)))
    for (name, S), color in zip(samples.items(), colors):
        if S.is_directed():
            degrees = [d for n, d in S.in_degree()]
        else:
            degrees = [d for n, d in S.degree()]
        
        ax1.hist(degrees, bins=50, alpha=0.3, label=name,
                 density=True, color=color)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Frequency (log)', fontsize=12)
    ax1.set_title('Degree Distribution Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Right plot: Clustering coefficient =====
    ax2 = axes[1]
    
    # Convert to undirected for clustering
    G_und = G.to_undirected() if G.is_directed() else G
    clustering_orig = list(nx.clustering(G_und).values())
    
    ax2.hist(clustering_orig, bins=50, alpha=0.5, label='Original',
             density=True, color='black')
    
    for (name, S), color in zip(samples.items(), colors):
        S_und = S.to_undirected() if S.is_directed() else S
        clustering = list(nx.clustering(S_und).values())
        
        ax2.hist(clustering, bins=50, alpha=0.3, label=name,
                 density=True, color=color)
    
    ax2.set_xlabel('Clustering Coefficient', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Clustering Coefficient Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Summary Bar Chart
# =============================================================================

def plot_method_comparison_bars(results_df: pd.DataFrame,
                                 ratio: float = 0.15,
                                 top_n: int = 10,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create bar chart comparing methods by average D-statistic.
    
    Args:
        results_df: DataFrame with results
        ratio: Sampling ratio to display
        top_n: Number of top methods to show
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Filter and aggregate
    df = results_df[results_df['ratio'] == ratio].copy()
    method_avg = df.groupby('method')['AVG'].mean().sort_values()
    
    # Take top N (best methods)
    top_methods = method_avg.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by method type
    colors = []
    for method in top_methods.index:
        if 'HYB' in method:
            colors.append('#2ecc71')  # Green for hybrid
        elif method in ['RW', 'FF', 'RJ']:
            colors.append('#3498db')  # Blue for exploration
        else:
            colors.append('#e74c3c')  # Red for node selection
    
    bars = ax.barh(range(len(top_methods)), top_methods.values, color=colors)
    
    # Add value labels
    for bar, value in zip(bars, top_methods.values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=10)
    
    ax.set_yticks(range(len(top_methods)))
    ax.set_yticklabels(top_methods.index)
    ax.set_xlabel('Average D-Statistic (lower is better)', fontsize=12)
    ax.set_title(f'Top {top_n} Methods by Performance (Ratio = {ratio*100:.0f}%)',
                 fontsize=14)
    ax.set_xlim(0, max(top_methods.values) * 1.2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Hybrid'),
        Patch(facecolor='#3498db', label='Exploration'),
        Patch(facecolor='#e74c3c', label='Node Selection')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Alpha Parameter Analysis
# =============================================================================

def plot_alpha_analysis(results_df: pd.DataFrame,
                         hybrid_method: str = "HYB-RN-RW",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Analyze effect of alpha parameter on hybrid method performance.
    
    Args:
        results_df: DataFrame with results
        hybrid_method: Base name of hybrid method to analyze
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Filter to hybrid methods matching the base name
    df = results_df[results_df['method'].str.contains(hybrid_method)].copy()
    
    if len(df) == 0:
        print(f"No data found for {hybrid_method}")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Left: Alpha vs D-statistic for different ratios =====
    ax1 = axes[0]
    
    for ratio in df['ratio'].unique():
        ratio_data = df[df['ratio'] == ratio]
        alpha_perf = ratio_data.groupby('alpha')['AVG'].mean()
        
        ax1.plot(alpha_perf.index, alpha_perf.values, 
                 'o-', label=f'Ratio={ratio*100:.0f}%',
                 linewidth=2, markersize=8)
    
    ax1.set_xlabel('Alpha (α)', fontsize=12)
    ax1.set_ylabel('Average D-Statistic', fontsize=12)
    ax1.set_title(f'{hybrid_method}: Effect of α on Performance', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Right: Alpha vs different properties =====
    ax2 = axes[1]
    
    properties = ['in_degree', 'clustering', 'hop_plot', 'singular_val']
    
    for prop in properties:
        alpha_perf = df.groupby('alpha')[prop].mean()
        ax2.plot(alpha_perf.index, alpha_perf.values,
                 'o-', label=prop, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Alpha (α)', fontsize=12)
    ax2.set_ylabel('D-Statistic', fontsize=12)
    ax2.set_title(f'{hybrid_method}: α Effect on Different Properties', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Dataset Comparison
# =============================================================================

def plot_dataset_comparison(results_df: pd.DataFrame,
                             method: str = "RW",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare method performance across different datasets.
    
    Args:
        results_df: DataFrame with results
        method: Method to analyze
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Filter to specific method
    df = results_df[results_df['method'] == method].copy()
    
    if len(df) == 0:
        print(f"No data found for method {method}")
        return None
    
    # Properties to show
    properties = ['in_degree', 'out_degree', 'wcc', 'scc', 
                  'hop_plot', 'clustering', 'AVG']
    
    # Create grouped bar chart
    datasets = df['dataset'].unique()
    x = np.arange(len(properties))
    width = 0.8 / len(datasets)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset][properties].mean()
        offset = (i - len(datasets)/2 + 0.5) * width
        
        ax.bar(x + offset, dataset_data.values, width, label=dataset)
    
    ax.set_xlabel('Property', fontsize=12)
    ax.set_ylabel('D-Statistic', fontsize=12)
    ax.set_title(f'{method}: Performance Across Datasets', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(properties, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Generate All Figures
# =============================================================================

def generate_all_figures(results_df: pd.DataFrame,
                          output_dir: str = FIGURES_DIR) -> None:
    """
    Generate all visualization figures from experiment results.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION FIGURES")
    print("="*70)
    
    # 1. D-statistic vs ratio curves
    print("\n1. D-statistic vs sampling ratio curves...")
    plot_d_statistic_vs_ratio(
        results_df,
        save_path=os.path.join(output_dir, f"d_statistic_curves.{FIGURE_FORMAT}")
    )
    
    # Per-dataset curves
    for dataset in results_df['dataset'].unique():
        plot_d_statistic_vs_ratio(
            results_df, dataset=dataset,
            save_path=os.path.join(output_dir, f"d_statistic_{dataset}.{FIGURE_FORMAT}")
        )
    
    # 2. Property heatmaps
    print("\n2. Property heatmaps...")
    for ratio in results_df['ratio'].unique():
        plot_property_heatmap(
            results_df, ratio=ratio,
            save_path=os.path.join(output_dir, f"heatmap_ratio_{int(ratio*100)}.{FIGURE_FORMAT}")
        )
    
    # 3. Method comparison bars
    print("\n3. Method comparison bar charts...")
    plot_method_comparison_bars(
        results_df, ratio=0.15,
        save_path=os.path.join(output_dir, f"method_comparison.{FIGURE_FORMAT}")
    )
    
    # 4. Alpha analysis for hybrid methods
    print("\n4. Alpha parameter analysis...")
    hybrid_methods = ["HYB-RN-RW", "HYB-RPN-FF", "HYB-RDN-RW"]
    for hybrid in hybrid_methods:
        if results_df['method'].str.contains(hybrid).any():
            plot_alpha_analysis(
                results_df, hybrid_method=hybrid,
                save_path=os.path.join(output_dir, f"alpha_{hybrid}.{FIGURE_FORMAT}")
            )
    
    # 5. Dataset comparison
    print("\n5. Dataset comparison...")
    for method in ['RW', 'FF']:
        if method in results_df['method'].values:
            plot_dataset_comparison(
                results_df, method=method,
                save_path=os.path.join(output_dir, f"dataset_comparison_{method}.{FIGURE_FORMAT}")
            )
    
    print(f"\n✓ All figures saved to: {output_dir}")


# =============================================================================
# Demo
# =============================================================================

def demo_visualizer():
    """
    Demonstrate visualization module with synthetic data.
    """
    print("="*70)
    print("VISUALIZATION MODULE DEMO")
    print("="*70)
    
    # Create synthetic results data
    np.random.seed(42)
    
    methods = ['RN', 'RPN', 'RDN', 'RW', 'FF', 
               'HYB-RN-RW(α=0.5)', 'HYB-RPN-FF(α=0.5)']
    ratios = [0.10, 0.15, 0.20]
    datasets = ['dataset1', 'dataset2']
    
    data = []
    for dataset in datasets:
        for ratio in ratios:
            for method in methods:
                method_type = 'hybrid' if 'HYB' in method else 'baseline'
                alpha = 0.5 if 'HYB' in method else None
                
                # Generate synthetic D-statistics
                base = 0.3 if 'HYB' in method or method in ['RW', 'FF'] else 0.5
                noise = np.random.uniform(-0.1, 0.1, 8)
                
                data.append({
                    'dataset': dataset,
                    'ratio': ratio,
                    'method': method,
                    'method_type': method_type,
                    'alpha': alpha,
                    'in_degree': base + noise[0],
                    'out_degree': base + noise[1],
                    'wcc': base + 0.3 + noise[2],
                    'scc': base + 0.2 + noise[3],
                    'hop_plot': base + noise[4],
                    'singular_vec': base - 0.1 + noise[5],
                    'singular_val': base - 0.1 + noise[6],
                    'clustering': base + noise[7],
                    'AVG': base + np.mean(noise)
                })
    
    results_df = pd.DataFrame(data)
    
    # Generate figures
    print("\nGenerating demo figures...")
    generate_all_figures(results_df, output_dir=FIGURES_DIR)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo_visualizer()
