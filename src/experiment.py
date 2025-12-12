
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASETS, SAMPLING_RATIOS, NUM_RUNS, RANDOM_SEED,
    BASELINE_METHODS, HYBRID_COMBINATIONS, HYBRID_ALPHA_VALUES,
    RESULTS_DIR, FF_FORWARD_PROB_SCALEDOWN, FF_FORWARD_PROB_BACKTIME
)


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    
    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: list(DATASETS.keys()))
    
    # Sampling ratios to test
    sampling_ratios: List[float] = field(default_factory=lambda: SAMPLING_RATIOS)
    
    # Number of runs per configuration
    num_runs: int = NUM_RUNS
    
    # Baseline methods to test
    baseline_methods: List[str] = field(default_factory=lambda: BASELINE_METHODS)
    
    # Hybrid method combinations
    hybrid_combinations: List[Tuple[str, str]] = field(
        default_factory=lambda: HYBRID_COMBINATIONS
    )
    
    # Alpha values for hybrid methods
    alpha_values: List[float] = field(default_factory=lambda: HYBRID_ALPHA_VALUES)
    
    # Sampling goal: "scale_down" or "back_in_time"
    sampling_goal: str = "scale_down"
    
    # Whether to include S6 (hop-plot on largest WCC)
    include_s6: bool = True
    
    # Random seed
    random_seed: int = RANDOM_SEED
    
    def get_ff_prob(self) -> float:
        """Get Forest Fire probability based on sampling goal."""
        if self.sampling_goal == "scale_down":
            return FF_FORWARD_PROB_SCALEDOWN
        else:
            return FF_FORWARD_PROB_BACKTIME
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'datasets': self.datasets,
            'sampling_ratios': self.sampling_ratios,
            'num_runs': self.num_runs,
            'baseline_methods': self.baseline_methods,
            'hybrid_combinations': [list(c) for c in self.hybrid_combinations],
            'alpha_values': self.alpha_values,
            'sampling_goal': self.sampling_goal,
            'include_s6': self.include_s6,
            'random_seed': self.random_seed
        }


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """
    Main experiment runner class.
    
    Coordinates sampling, evaluation, and result aggregation.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results = []
        
        # Set random seed
        np.random.seed(config.random_seed)
    
    def _load_dataset(self, dataset_name: str):
        """
        Load a dataset and create evaluator.
        
        Args:
            dataset_name: Name of dataset to load
        
        Returns:
            Tuple of (graph, evaluator)
        """
        from src.data_loader import DataLoader
        from src.evaluator import GraphEvaluator
        
        loader = DataLoader()
        G = loader.load_dataset(dataset_name)
        
        print(f"  Loaded {dataset_name}: {G.number_of_nodes():,} nodes, "
              f"{G.number_of_edges():,} edges")
        
        evaluator = GraphEvaluator(G, use_log_transform=True)
        
        return G, evaluator
    
    def _run_single_sample(self, G, evaluator, method: str, n_samples: int,
                           run_id: int, **kwargs) -> Dict:
        """
        Run a single sampling and evaluation.
        
        Args:
            G: Graph to sample from
            evaluator: GraphEvaluator instance
            method: Sampling method name
            n_samples: Number of nodes to sample
            run_id: Run identifier (for random seed)
            **kwargs: Additional sampler arguments
        
        Returns:
            Dictionary with evaluation results
        """
        from src.samplers import sample_graph
        
        # Sample the graph
        random_state = self.config.random_seed + run_id
        S = sample_graph(G, method, n_samples, random_state=random_state, **kwargs)
        
        # Evaluate
        results = evaluator.evaluate_all(S, include_s6=self.config.include_s6)
        
        return results
    
    def _run_method(self, G, evaluator, dataset_name: str, method: str,
                    ratio: float, method_type: str, alpha: Optional[float] = None,
                    **kwargs) -> List[Dict]:
        """
        Run all repetitions for a single method configuration.
        
        Args:
            G: Graph
            evaluator: Evaluator
            dataset_name: Dataset name
            method: Method name
            ratio: Sampling ratio
            method_type: "baseline" or "hybrid"
            alpha: Alpha parameter for hybrid methods
            **kwargs: Additional arguments
        
        Returns:
            List of result dictionaries (one per run)
        """
        n_samples = int(G.number_of_nodes() * ratio)
        results = []
        
        # 将 alpha 添加到 kwargs 传递给采样器
        if alpha is not None:
            kwargs['alpha'] = alpha
        
        for run in range(self.config.num_runs):
            eval_results = self._run_single_sample(
                G, evaluator, method, n_samples, run, **kwargs
            )
            
            # Build result row
            row = {
                'dataset': dataset_name,
                'ratio': ratio,
                'method': method if alpha is None else f"{method}(α={alpha})",
                'method_type': method_type,
                'alpha': alpha,
                'run': run,
                'n_samples': n_samples,
            }
            row.update(eval_results)
            results.append(row)
        
        return results
        
        return results
    
    def run(self) -> pd.DataFrame:
        """
        Run the full experiment.
        
        Returns:
            DataFrame with all results
        """
        print("\n" + "="*70)
        print("STARTING EXPERIMENT")
        print("="*70)
        print(f"  Goal: {self.config.sampling_goal}")
        print(f"  Datasets: {self.config.datasets}")
        print(f"  Sampling ratios: {self.config.sampling_ratios}")
        print(f"  Runs per config: {self.config.num_runs}")
        print(f"  Include S6: {self.config.include_s6}")
        print("="*70)
        
        all_results = []
        start_time = time.time()
        
        ff_prob = self.config.get_ff_prob()
        print(f"  FF forward_prob: {ff_prob} ({self.config.sampling_goal})")
        
        for dataset_name in self.config.datasets:
            print(f"\n--- Dataset: {dataset_name} ---")
            
            # Load dataset
            try:
                G, evaluator = self._load_dataset(dataset_name)
            except Exception as e:
                print(f"  Error loading {dataset_name}: {e}")
                continue
            
            # Calculate total iterations for progress bar
            n_baselines = len(self.config.baseline_methods)
            n_hybrids = len(self.config.hybrid_combinations) * len(self.config.alpha_values)
            n_ratios = len(self.config.sampling_ratios)
            total_configs = (n_baselines + n_hybrids) * n_ratios
            
            pbar = tqdm(total=total_configs, desc=f"  {dataset_name}")
            
            for ratio in self.config.sampling_ratios:
                # Run baseline methods
                for method in self.config.baseline_methods:
                    kwargs = {}
                    if method == "FF":
                        kwargs['forward_prob'] = ff_prob
                    
                    results = self._run_method(
                        G, evaluator, dataset_name, method, ratio,
                        method_type='baseline', **kwargs
                    )
                    all_results.extend(results)
                    pbar.update(1)
                
                # Run hybrid methods
                for node_m, explore_m in self.config.hybrid_combinations:
                    for alpha in self.config.alpha_values:
                        method = f"HYB-{node_m}-{explore_m}"
                        
                        # 注意：alpha 只传递一次，不要在 kwargs 里重复
                        kwargs = {}
                        if explore_m == "FF":
                            kwargs['forward_prob'] = ff_prob
                        
                        results = self._run_method(
                            G, evaluator, dataset_name, method, ratio,
                            method_type='hybrid', alpha=alpha, **kwargs
                        )
                        all_results.extend(results)
                        pbar.update(1)
            
            pbar.close()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Experiment completed in {elapsed:.1f} seconds")
        print(f"  Total configurations: {len(all_results)}")
        
        return pd.DataFrame(all_results)
    
    def save_results(self, results_df: pd.DataFrame, 
                     prefix: str = "experiment") -> str:
        """
        Save results to CSV file.
        
        Args:
            results_df: Results DataFrame
            prefix: File name prefix
        
        Returns:
            Path to saved file
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.config.sampling_goal}_{timestamp}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        results_df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to: {filepath}")
        
        # Save config
        config_path = filepath.replace('.csv', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"  Config saved to: {config_path}")
        
        return filepath


# =============================================================================
# Analysis Functions
# =============================================================================

def generate_summary_table(results_df: pd.DataFrame, 
                           ratio: float = 0.15,
                           include_s6: bool = True) -> pd.DataFrame:
    """
    Generate summary table similar to Table 1 in original paper.
    
    Args:
        results_df: Results DataFrame
        ratio: Sampling ratio to summarize
        include_s6: Whether S6 was included
    
    Returns:
        Summary DataFrame
    """
    # Filter by ratio
    df = results_df[results_df['ratio'] == ratio].copy()
    
    # Properties to include
    properties = ['in_degree', 'out_degree', 'wcc', 'scc', 'hop_plot']
    if include_s6 and 'hop_plot_wcc' in df.columns:
        properties.append('hop_plot_wcc')
    properties.extend(['singular_vec', 'singular_val', 'clustering', 'AVG'])
    
    # Filter to existing columns
    properties = [p for p in properties if p in df.columns]
    
    # Group by method and compute mean
    summary = df.groupby('method')[properties].mean()
    
    # Sort by AVG
    summary = summary.sort_values('AVG')
    
    # Round for display
    summary = summary.round(4)
    
    return summary


def compare_baseline_vs_hybrid(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare baseline methods vs hybrid methods.
    
    Args:
        results_df: Results DataFrame
    
    Returns:
        Comparison DataFrame
    """
    # Separate baseline and hybrid
    baseline = results_df[results_df['method_type'] == 'baseline']
    hybrid = results_df[results_df['method_type'] == 'hybrid']
    
    # Best baseline per dataset/ratio
    baseline_best = baseline.groupby(['dataset', 'ratio'])['AVG'].min().reset_index()
    baseline_best.columns = ['dataset', 'ratio', 'best_baseline_AVG']
    
    # Best hybrid per dataset/ratio
    hybrid_best = hybrid.groupby(['dataset', 'ratio'])['AVG'].min().reset_index()
    hybrid_best.columns = ['dataset', 'ratio', 'best_hybrid_AVG']
    
    # Merge
    comparison = pd.merge(baseline_best, hybrid_best, on=['dataset', 'ratio'])
    
    # Compute improvement
    comparison['improvement'] = (
        comparison['best_baseline_AVG'] - comparison['best_hybrid_AVG']
    )
    comparison['improvement_pct'] = (
        comparison['improvement'] / comparison['best_baseline_AVG'] * 100
    )
    
    return comparison.round(4)


def find_best_methods(results_df: pd.DataFrame, 
                      n_best: int = 5) -> pd.DataFrame:
    """
    Find the best methods across all configurations.
    
    Args:
        results_df: Results DataFrame
        n_best: Number of top methods to return
    
    Returns:
        DataFrame with best methods
    """
    # Compute mean AVG per method
    method_avg = results_df.groupby('method')['AVG'].mean().sort_values()
    
    # Get top N
    top_methods = method_avg.head(n_best).reset_index()
    top_methods.columns = ['method', 'mean_AVG']
    
    # Add method type
    top_methods['method_type'] = top_methods['method'].apply(
        lambda x: 'hybrid' if 'HYB' in x else 'baseline'
    )
    
    return top_methods.round(4)


def analyze_alpha_effect(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the effect of alpha parameter on hybrid method performance.
    
    Args:
        results_df: Results DataFrame
    
    Returns:
        DataFrame with alpha analysis
    """
    # Filter to hybrid methods only
    hybrid = results_df[results_df['method_type'] == 'hybrid'].copy()
    
    if hybrid.empty:
        return pd.DataFrame()
    
    # Group by alpha and compute stats
    alpha_analysis = hybrid.groupby('alpha')['AVG'].agg(['mean', 'std', 'min', 'max'])
    alpha_analysis = alpha_analysis.round(4)
    
    return alpha_analysis


# =============================================================================
# Quick Test Function
# =============================================================================

def run_quick_test() -> pd.DataFrame:
    """
    Run a quick test experiment with minimal configuration.
    
    Returns:
        Results DataFrame
    """
    config = ExperimentConfig(
        datasets=["cit-HepTh"],
        sampling_ratios=[0.15],
        num_runs=3,
        baseline_methods=["RN", "RW", "FF"],
        hybrid_combinations=[("RN", "RW")],
        alpha_values=[0.5],
        sampling_goal="scale_down",
        include_s6=True
    )
    
    runner = ExperimentRunner(config)
    results_df = runner.run()
    runner.save_results(results_df, prefix="quick_test")
    
    return results_df


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Graph Sampling Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full experiment")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset")
    parser.add_argument("--goal", type=str, default="scale_down",
                       choices=["scale_down", "back_in_time"])
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick test...")
        results = run_quick_test()
        print("\nSummary:")
        print(generate_summary_table(results))
    
    elif args.full:
        config = ExperimentConfig(
            datasets=[args.dataset] if args.dataset else list(DATASETS.keys()),
            sampling_goal=args.goal
        )
        runner = ExperimentRunner(config)
        results = runner.run()
        runner.save_results(results, prefix="full")
        
        print("\nSummary at 15% ratio:")
        print(generate_summary_table(results, ratio=0.15))
        
        print("\nBaseline vs Hybrid:")
        print(compare_baseline_vs_hybrid(results))
    
    else:
        print("Use --quick for quick test or --full for full experiment")
