import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASETS, SAMPLING_RATIOS, NUM_RUNS, RANDOM_SEED,
    BASELINE_METHODS, HYBRID_COMBINATIONS, HYBRID_ALPHA_VALUES,
    RESULTS_DIR, DATA_DIR,
    FF_FORWARD_PROB_SCALEDOWN, FF_FORWARD_PROB_BACKTIME
)
from src.data_loader import load_dataset, get_graph_info
from src.samplers import sample_graph, get_sampler
from src.evaluator import GraphEvaluator, compute_mean_statistics


# =============================================================================
# Experiment Configuration
# =============================================================================

class ExperimentConfig:
    """
    Configuration for an experiment run.
    """
    
    def __init__(self,
                 datasets: Optional[List[str]] = None,
                 sampling_ratios: Optional[List[float]] = None,
                 num_runs: int = NUM_RUNS,
                 baseline_methods: Optional[List[str]] = None,
                 hybrid_combinations: Optional[List[Tuple[str, str]]] = None,
                 alpha_values: Optional[List[float]] = None,
                 sampling_goal: str = "scale_down",
                 random_seed: Optional[int] = RANDOM_SEED,
                 include_s6: bool = True):
        """
        Initialize experiment configuration.
        
        Args:
            datasets: List of dataset names to use
            sampling_ratios: List of sampling ratios (0 to 1)
            num_runs: Number of runs per configuration
            baseline_methods: List of baseline method names
            hybrid_combinations: List of (node_method, explore_method) tuples
            alpha_values: List of alpha values for hybrid methods
            sampling_goal: "scale_down" or "back_in_time"
            random_seed: Random seed for reproducibility
            include_s6: Whether to include S6 (hop-plot on largest WCC)
        """
        self.datasets = datasets or list(DATASETS.keys())
        self.sampling_ratios = sampling_ratios or SAMPLING_RATIOS
        self.num_runs = num_runs
        self.baseline_methods = baseline_methods or BASELINE_METHODS
        self.hybrid_combinations = hybrid_combinations or HYBRID_COMBINATIONS
        self.alpha_values = alpha_values or HYBRID_ALPHA_VALUES
        self.sampling_goal = sampling_goal
        self.random_seed = random_seed
        self.include_s6 = include_s6
        
        # Set Forest Fire probability based on goal
        if sampling_goal == "scale_down":
            self.ff_prob = FF_FORWARD_PROB_SCALEDOWN
        else:
            self.ff_prob = FF_FORWARD_PROB_BACKTIME
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "datasets": self.datasets,
            "sampling_ratios": self.sampling_ratios,
            "num_runs": self.num_runs,
            "baseline_methods": self.baseline_methods,
            "hybrid_combinations": [f"{n}-{e}" for n, e in self.hybrid_combinations],
            "alpha_values": self.alpha_values,
            "sampling_goal": self.sampling_goal,
            "random_seed": self.random_seed,
            "ff_prob": self.ff_prob,
            "include_s6": self.include_s6,
        }


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """
    Main experiment runner class.
    
    Handles the complete experimental pipeline:
    - Dataset loading
    - Sampling
    - Evaluation
    - Results aggregation
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results = []
        self.start_time = None
        
        # Set random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def _run_single_sampling(self, G: nx.Graph, method: str, 
                              n_samples: int, alpha: float = 0.5,
                              run_idx: int = 0) -> nx.Graph:
        """
        Run a single sampling operation.
        
        Args:
            G: Original graph
            method: Sampling method name
            n_samples: Number of nodes to sample
            alpha: Alpha value for hybrid methods
            run_idx: Run index (for seeding)
        
        Returns:
            Sampled subgraph
        """
        # Set seed for this specific run
        seed = (self.config.random_seed or 0) + run_idx
        
        # FIXED: Get appropriate parameters based on method type
        kwargs = {}
        
        # FIXED: Only pass alpha to hybrid methods (methods starting with HYB-)
        if method.startswith("HYB-"):
            kwargs["alpha"] = alpha
        
        # Set Forest Fire probability for FF-containing methods
        if "FF" in method:
            kwargs["forward_prob"] = self.config.ff_prob
        
        return sample_graph(G, method, n_samples, random_state=seed, **kwargs)
    
    def _evaluate_method(self, G: nx.Graph, evaluator: GraphEvaluator,
                          method: str, n_samples: int, alpha: float = 0.5,
                          desc: str = "") -> Dict[str, float]:
        """
        Evaluate a single method across multiple runs.
        
        Args:
            G: Original graph
            evaluator: GraphEvaluator instance
            method: Sampling method name
            n_samples: Number of nodes to sample
            alpha: Alpha value for hybrid methods
            desc: Description for progress bar
        
        Returns:
            Dictionary with mean KS statistics
        """
        all_stats = []
        
        # Define default stats based on whether S6 is included
        if self.config.include_s6:
            default_stats = {
                "in_degree": 1.0, "out_degree": 1.0,
                "wcc": 1.0, "scc": 1.0, "hop_plot": 1.0,
                "hop_plot_wcc": 1.0,  # S6
                "singular_vec": 1.0, "singular_val": 1.0,
                "clustering": 1.0, "AVG": 1.0
            }
        else:
            default_stats = {
                "in_degree": 1.0, "out_degree": 1.0,
                "wcc": 1.0, "scc": 1.0, "hop_plot": 1.0,
                "singular_vec": 1.0, "singular_val": 1.0,
                "clustering": 1.0, "AVG": 1.0
            }
        
        for run in range(self.config.num_runs):
            try:
                # Sample
                S = self._run_single_sampling(G, method, n_samples, alpha, run)
                
                # Evaluate (include_s6 is now configurable)
                stats = evaluator.evaluate_all(S, include_s6=self.config.include_s6)
                all_stats.append(stats)
                
            except Exception as e:
                print(f"\n    Warning: Error in {method} run {run}: {e}")
                # Add worst-case stats
                all_stats.append(default_stats.copy())
        
        # Compute mean statistics
        return compute_mean_statistics({
            prop: [s[prop] for s in all_stats]
            for prop in all_stats[0].keys()
        })
    
    def run_dataset(self, dataset_name: str) -> List[dict]:
        """
        Run experiments on a single dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            List of result dictionaries
        """
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load dataset
        G = load_dataset(dataset_name)
        info = get_graph_info(G)
        
        # Create evaluator (caches original graph properties)
        evaluator = GraphEvaluator(G, use_log_transform=True)
        
        results = []
        
        # Iterate over sampling ratios
        for ratio in self.config.sampling_ratios:
            n_samples = int(G.number_of_nodes() * ratio)
            print(f"\n  Sampling ratio: {ratio*100:.0f}% ({n_samples} nodes)")
            print(f"  {'-'*50}")
            
            # Test baseline methods
            print(f"  Testing baseline methods...")
            for method in tqdm(self.config.baseline_methods, 
                              desc="  Baselines", leave=False):
                stats = self._evaluate_method(
                    G, evaluator, method, n_samples,
                    desc=f"{method}"
                )
                
                result = {
                    "dataset": dataset_name,
                    "ratio": ratio,
                    "n_samples": n_samples,
                    "method": method,
                    "method_type": "baseline",
                    "alpha": None,
                    **stats
                }
                results.append(result)
                
                # Print result
                print(f"    {method:12s}: AVG={stats['AVG']:.4f}")
            
            # Test hybrid methods
            print(f"  Testing hybrid methods...")
            for node_method, explore_method in tqdm(self.config.hybrid_combinations,
                                                     desc="  Hybrids", leave=False):
                hybrid_name = f"HYB-{node_method}-{explore_method}"
                
                for alpha in self.config.alpha_values:
                    stats = self._evaluate_method(
                        G, evaluator, hybrid_name, n_samples, alpha,
                        desc=f"{hybrid_name}(α={alpha})"
                    )
                    
                    result = {
                        "dataset": dataset_name,
                        "ratio": ratio,
                        "n_samples": n_samples,
                        "method": f"{hybrid_name}(α={alpha})",
                        "method_type": "hybrid",
                        "alpha": alpha,
                        **stats
                    }
                    results.append(result)
                    
                    # Print best alpha only
                    if alpha == self.config.alpha_values[len(self.config.alpha_values)//2]:
                        print(f"    {hybrid_name:15s}: AVG={stats['AVG']:.4f} (α={alpha})")
        
        return results
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete experiment.
        
        Returns:
            DataFrame with all results
        """
        self.start_time = datetime.now()
        print("="*70)
        print("GRAPH SAMPLING EXPERIMENT")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Goal: {self.config.sampling_goal}")
        print(f"Include S6: {self.config.include_s6}")
        print("="*70)
        
        # Print configuration
        print(f"\nConfiguration:")
        print(f"  Datasets: {', '.join(self.config.datasets)}")
        print(f"  Sampling ratios: {self.config.sampling_ratios}")
        print(f"  Runs per config: {self.config.num_runs}")
        print(f"  Baseline methods: {len(self.config.baseline_methods)}")
        print(f"  Hybrid combinations: {len(self.config.hybrid_combinations)}")
        print(f"  Alpha values: {self.config.alpha_values}")
        
        # Run experiments
        all_results = []
        
        for dataset_name in self.config.datasets:
            try:
                results = self.run_dataset(dataset_name)
                all_results.extend(results)
            except Exception as e:
                print(f"\n  ERROR on dataset {dataset_name}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Print summary
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print(f"Duration: {duration}")
        print(f"Total results: {len(results_df)}")
        print("="*70)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, 
                      prefix: str = "experiment") -> str:
        """
        Save results to CSV file.
        
        Args:
            results_df: DataFrame with results
            prefix: Filename prefix
        
        Returns:
            Path to saved file
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.config.sampling_goal}_{timestamp}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Save
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to: {filepath}")
        
        # Also save configuration
        config_file = filepath.replace(".csv", "_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"Config saved to: {config_file}")
        
        return filepath


# =============================================================================
# Result Analysis Functions
# =============================================================================

def generate_summary_table(results_df: pd.DataFrame, 
                            ratio: float = 0.15,
                            include_s6: bool = True) -> pd.DataFrame:
    """
    Generate summary table similar to Table 1 in original paper.
    
    UPDATED: Now includes S6 (hop_plot_wcc) by default.
    
    Args:
        results_df: DataFrame with experiment results
        ratio: Sampling ratio to filter by
        include_s6: Whether to include S6 in the summary
    
    Returns:
        Summary DataFrame
    """
    # Filter by ratio
    df = results_df[results_df['ratio'] == ratio].copy()
    
    # Properties to include (now with S6)
    if include_s6 and 'hop_plot_wcc' in df.columns:
        properties = ['in_degree', 'out_degree', 'wcc', 'scc', 
                      'hop_plot', 'hop_plot_wcc',  # S5 and S6
                      'singular_val', 'singular_vec', 
                      'clustering', 'AVG']
    else:
        properties = ['in_degree', 'out_degree', 'wcc', 'scc', 
                      'hop_plot', 'singular_val', 'singular_vec', 
                      'clustering', 'AVG']
    
    # Filter to only existing columns
    properties = [p for p in properties if p in df.columns]
    
    # Group by method and compute mean across datasets
    summary = df.groupby('method')[properties].mean()
    
    # Sort by AVG
    summary = summary.sort_values('AVG')
    
    # Round to 3 decimal places
    summary = summary.round(3)
    
    return summary


def find_best_methods(results_df: pd.DataFrame) -> Dict[str, str]:
    """
    Find the best method for each dataset and ratio.
    
    Args:
        results_df: DataFrame with results
    
    Returns:
        Dictionary with best methods
    """
    best = {}
    
    for dataset in results_df['dataset'].unique():
        for ratio in results_df['ratio'].unique():
            subset = results_df[
                (results_df['dataset'] == dataset) & 
                (results_df['ratio'] == ratio)
            ]
            
            if len(subset) > 0:
                best_row = subset.loc[subset['AVG'].idxmin()]
                key = f"{dataset}_{ratio}"
                best[key] = {
                    "method": best_row['method'],
                    "AVG": best_row['AVG']
                }
    
    return best


def compare_baseline_vs_hybrid(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare baseline methods against hybrid methods.
    
    Args:
        results_df: DataFrame with results
    
    Returns:
        Comparison DataFrame
    """
    # Best baseline per dataset/ratio
    baseline_df = results_df[results_df['method_type'] == 'baseline']
    best_baseline = baseline_df.groupby(['dataset', 'ratio'])['AVG'].min()
    
    # Best hybrid per dataset/ratio
    hybrid_df = results_df[results_df['method_type'] == 'hybrid']
    best_hybrid = hybrid_df.groupby(['dataset', 'ratio'])['AVG'].min()
    
    # Create comparison
    comparison = pd.DataFrame({
        'best_baseline': best_baseline,
        'best_hybrid': best_hybrid
    })
    comparison['improvement'] = comparison['best_baseline'] - comparison['best_hybrid']
    comparison['improvement_pct'] = (comparison['improvement'] / comparison['best_baseline'] * 100)
    
    return comparison.round(4)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for running experiments.
    """
    parser = argparse.ArgumentParser(
        description="Run Graph Sampling Experiments"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with reduced parameters"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Run on a single dataset"
    )
    parser.add_argument(
        "--goal", type=str, default="scale_down",
        choices=["scale_down", "back_in_time"],
        help="Sampling goal (affects FF probability)"
    )
    parser.add_argument(
        "--runs", type=int, default=NUM_RUNS,
        help="Number of runs per configuration"
    )
    parser.add_argument(
        "--no-s6", action="store_true",
        help="Exclude S6 (hop-plot on largest WCC) from evaluation"
    )
    
    args = parser.parse_args()
    
    # Determine whether to include S6
    include_s6 = not args.no_s6
    
    # Create configuration
    if args.quick:
        # Quick test configuration
        config = ExperimentConfig(
            datasets=["astro-ph"] if args.dataset is None else [args.dataset],
            sampling_ratios=[0.15],
            num_runs=3,
            baseline_methods=["RN", "RW", "FF"],
            hybrid_combinations=[("RN", "RW"), ("RPN", "FF")],
            alpha_values=[0.5],
            sampling_goal=args.goal,
            include_s6=include_s6
        )
    else:
        # Full configuration
        config = ExperimentConfig(
            datasets=[args.dataset] if args.dataset else None,
            num_runs=args.runs,
            sampling_goal=args.goal,
            include_s6=include_s6
        )
    
    # Run experiment
    runner = ExperimentRunner(config)
    results_df = runner.run()
    
    # Save results
    runner.save_results(results_df)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (ratio=0.15)")
    print("="*70)
    summary = generate_summary_table(results_df, ratio=0.15, include_s6=include_s6)
    print(summary.to_string())
    
    # Print baseline vs hybrid comparison
    print("\n" + "="*70)
    print("BASELINE vs HYBRID COMPARISON")
    print("="*70)
    comparison = compare_baseline_vs_hybrid(results_df)
    print(comparison.to_string())
    
    return results_df


if __name__ == "__main__":
    main()