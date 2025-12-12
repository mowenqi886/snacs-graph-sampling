
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT, FIGURE_SIZE, METHOD_COLORS,
    BASELINE_METHODS, HYBRID_COMBINATIONS
)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# D-Statistic Curves (Similar to Figure 3 in original paper)
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
        print(f"  Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Property Heatmap (Similar to Table 1 in original paper)
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
    
    # Properties to display (S1-S9)
    properties = [
        "in_degree", "out_degree", "wcc", "scc",
        "hop_plot", "hop_plot_wcc",
        "singular_val", "singular_vec", "clustering"
    ]
    
    # Filter to existing columns
    properties = [p for p in properties if p in df.columns]
    
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
    ax.set_xlabel('Graph Property (S1-S9)', fontsize=12)
    ax.set_ylabel('Sampling Method', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Method Comparison Bar Chart
# =============================================================================

def plot_method_comparison_bars(results_df: pd.DataFrame,
                                 ratio: float = 0.15,
                                 top_n: int = 10,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart comparing methods by average D-statistic.
    
    Args:
        results_df: DataFrame with results
        ratio: Sampling ratio to display
        top_n: Number of top methods to show
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Filter by ratio
    df = results_df[results_df['ratio'] == ratio].copy()
    
    # Compute mean AVG per method
    method_avg = df.groupby('method')['AVG'].mean().sort_values()
    
    # Take top N
    method_avg = method_avg.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [METHOD_COLORS.get(m.split('(')[0], '#1f77b4') for m in method_avg.index]
    
    bars = ax.barh(method_avg.index, method_avg.values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, method_avg.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Average D-Statistic (lower is better)', fontsize=12)
    ax.set_ylabel('Sampling Method', fontsize=12)
    ax.set_title(f'Top {top_n} Methods at {ratio*100:.0f}% Sampling Ratio', fontsize=14)
    ax.set_xlim(0, max(method_avg.values) * 1.2)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# Alpha Parameter Analysis (for Hybrid Methods)
# =============================================================================

def plot_alpha_analysis(results_df: pd.DataFrame,
                         hybrid_method: str = "HYB-RN-RW",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Analyze effect of alpha parameter on hybrid method performance.
    
    Args:
        results_df: DataFrame with results
        hybrid_method: Base hybrid method name (e.g., "HYB-RN-RW")
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
    
    for ratio in sorted(df['ratio'].unique()):
        ratio_data = df[df['ratio'] == ratio]
        alpha_perf = ratio_data.groupby('alpha')['AVG'].mean()
        
        ax1.plot(alpha_perf.index, alpha_perf.values, 
                 'o-', label=f'Ratio={ratio*100:.0f}%',
                 linewidth=2, markersize=8)
    
    ax1.set_xlabel('Alpha (α) - Fraction from Node Selection', fontsize=12)
    ax1.set_ylabel('Average D-Statistic', fontsize=12)
    ax1.set_title(f'{hybrid_method}: Effect of α on Performance', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Right: Alpha vs different properties =====
    ax2 = axes[1]
    
    properties = ['in_degree', 'clustering', 'hop_plot', 'singular_val']
    properties = [p for p in properties if p in df.columns]
    
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
        print(f"  Figure saved to: {save_path}")
    
    return fig


# =============================================================================
# NEW: Temporal Metrics Visualization (T1-T5)
# =============================================================================

def plot_temporal_metrics_comparison(original_metrics: Dict[str, np.ndarray],
                                      sampled_metrics: Dict[str, np.ndarray],
                                      method_name: str = "Sample",
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot T1-T5 temporal metrics comparison between original and sampled graphs.
    Similar to Figure 2 in the original paper.
    
    Args:
        original_metrics: Dict with T1-T5 metrics for original graph
        sampled_metrics: Dict with T1-T5 metrics for sampled graph
        method_name: Name of sampling method (for labels)
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    num_snapshots = len(original_metrics.get('T1_dpl', []))
    x = range(num_snapshots)
    x_labels = [f'Snap_{i+1}' for i in x]
    
    # T1: Densification Power Law
    ax = axes[0, 0]
    if 'T1_nodes' in original_metrics and 'T1_edges' in original_metrics:
        ax.loglog(original_metrics['T1_nodes'], original_metrics['T1_edges'], 
                  'b-o', label='Original', markersize=8, linewidth=2)
        if 'T1_nodes' in sampled_metrics and 'T1_edges' in sampled_metrics:
            ax.loglog(sampled_metrics['T1_nodes'], sampled_metrics['T1_edges'], 
                      'r--s', label=method_name, markersize=6, linewidth=2)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Number of Edges')
    orig_exp = original_metrics.get('T1_exponent', 0)
    samp_exp = sampled_metrics.get('T1_exponent', 0)
    ax.set_title(f'T1: DPL (a_orig={orig_exp:.2f}, a_samp={samp_exp:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # T2: Diameter Over Time
    ax = axes[0, 1]
    if 'T2_diameter' in original_metrics:
        ax.plot(x, original_metrics['T2_diameter'], 'b-o', label='Original', 
                markersize=8, linewidth=2)
        if 'T2_diameter' in sampled_metrics:
            ax.plot(x, sampled_metrics['T2_diameter'], 'r--s', label=method_name,
                    markersize=6, linewidth=2)
    ax.set_xlabel('Time Snapshot')
    ax.set_ylabel('Effective Diameter')
    ax.set_title('T2: Diameter Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # T3: CC Size Over Time
    ax = axes[0, 2]
    if 'T3_cc_size' in original_metrics:
        ax.plot(x, original_metrics['T3_cc_size'], 'b-o', label='Original',
                markersize=8, linewidth=2)
        if 'T3_cc_size' in sampled_metrics:
            ax.plot(x, sampled_metrics['T3_cc_size'], 'r--s', label=method_name,
                    markersize=6, linewidth=2)
    ax.set_xlabel('Time Snapshot')
    ax.set_ylabel('Largest CC Fraction')
    ax.set_title('T3: CC Size Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # T4: Singular Value Over Time
    ax = axes[1, 0]
    if 'T4_singular' in original_metrics:
        ax.plot(x, original_metrics['T4_singular'], 'b-o', label='Original',
                markersize=8, linewidth=2)
        if 'T4_singular' in sampled_metrics:
            ax.plot(x, sampled_metrics['T4_singular'], 'r--s', label=method_name,
                    markersize=6, linewidth=2)
    ax.set_xlabel('Time Snapshot')
    ax.set_ylabel('Largest Singular Value')
    ax.set_title('T4: Singular Value Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # T5: Clustering Over Time
    ax = axes[1, 1]
    if 'T5_clustering' in original_metrics:
        ax.plot(x, original_metrics['T5_clustering'], 'b-o', label='Original',
                markersize=8, linewidth=2)
        if 'T5_clustering' in sampled_metrics:
            ax.plot(x, sampled_metrics['T5_clustering'], 'r--s', label=method_name,
                    markersize=6, linewidth=2)
    ax.set_xlabel('Time Snapshot')
    ax.set_ylabel('Avg Clustering Coefficient')
    ax.set_title('T5: Clustering Over Time')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary: KS D-statistics bar chart
    ax = axes[1, 2]
    metrics = ['T1', 'T2', 'T3', 'T4', 'T5']
    # This would need the actual KS statistics - placeholder for now
    ax.text(0.5, 0.5, 'T1-T5 KS Statistics\n(See Results Table)', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('T1-T5 KS D-Statistics Summary')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    return fig


def plot_temporal_results_heatmap(results_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap for back-in-time results including T1-T5 metrics.
    
    Args:
        results_df: DataFrame with temporal experiment results
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Check if this is a temporal results DataFrame
    temporal_cols = ['T1_dpl', 'T2_diameter', 'T3_cc_size', 'T4_singular', 'T5_clustering', 'T_AVG']
    available_cols = [c for c in temporal_cols if c in results_df.columns]
    
    if not available_cols:
        print("No temporal metrics found in results DataFrame")
        return None
    
    # Add static average if available
    if 'S_AVG_ALL' in results_df.columns:
        available_cols = ['S_AVG_ALL'] + available_cols
    
    # Group by method
    method_data = results_df.groupby('method')[available_cols].mean()
    
    # Sort by T_AVG or COMBINED_AVG
    sort_col = 'T_AVG' if 'T_AVG' in method_data.columns else available_cols[-1]
    method_data = method_data.sort_values(sort_col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(method_data) * 0.5)))
    
    sns.heatmap(
        method_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        center=0.5,
        ax=ax,
        cbar_kws={'label': 'D-Statistic (lower is better)'}
    )
    
    ax.set_title('Back-in-Time Evaluation: Static (S) and Temporal (T) Metrics', fontsize=14)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Sampling Method', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
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
    try:
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
    except Exception as e:
        print(f"  Warning: Could not generate D-statistic curves: {e}")
    
    # 2. Property heatmaps
    print("\n2. Property heatmaps...")
    try:
        for ratio in results_df['ratio'].unique():
            plot_property_heatmap(
                results_df, ratio=ratio,
                save_path=os.path.join(output_dir, f"heatmap_ratio_{int(ratio*100)}.{FIGURE_FORMAT}")
            )
    except Exception as e:
        print(f"  Warning: Could not generate heatmaps: {e}")
    
    # 3. Method comparison bars
    print("\n3. Method comparison bar charts...")
    try:
        plot_method_comparison_bars(
            results_df, ratio=0.15,
            save_path=os.path.join(output_dir, f"method_comparison.{FIGURE_FORMAT}")
        )
    except Exception as e:
        print(f"  Warning: Could not generate method comparison: {e}")
    
    # 4. Alpha analysis for hybrid methods
    print("\n4. Alpha parameter analysis...")
    hybrid_methods = ["HYB-RN-RW", "HYB-RPN-FF", "HYB-RDN-RW"]
    for hybrid in hybrid_methods:
        try:
            if results_df['method'].str.contains(hybrid).any():
                plot_alpha_analysis(
                    results_df, hybrid_method=hybrid,
                    save_path=os.path.join(output_dir, f"alpha_{hybrid}.{FIGURE_FORMAT}")
                )
        except Exception as e:
            print(f"  Warning: Could not generate alpha analysis for {hybrid}: {e}")
    
    # 5. Temporal metrics heatmap (if available)
    print("\n5. Temporal metrics visualization...")
    try:
        if 'T_AVG' in results_df.columns or 'T1_dpl' in results_df.columns:
            plot_temporal_results_heatmap(
                results_df,
                save_path=os.path.join(output_dir, f"temporal_heatmap.{FIGURE_FORMAT}")
            )
    except Exception as e:
        print(f"  Warning: Could not generate temporal heatmap: {e}")
    
    print(f"\n✓ All figures saved to: {output_dir}")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
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
                noise = np.random.uniform(-0.1, 0.1, 9)
                
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
                    'hop_plot_wcc': base + noise[5],
                    'singular_vec': base - 0.1 + noise[6],
                    'singular_val': base - 0.1 + noise[7],
                    'clustering': base + noise[8],
                    'AVG': base + np.mean(noise)
                })
    
    results_df = pd.DataFrame(data)
    
    # Generate figures
    print("\nGenerating demo figures...")
    generate_all_figures(results_df, output_dir=FIGURES_DIR)
    
    print("\n✓ Demo completed successfully!")
