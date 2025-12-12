#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASETS, TEMPORAL_DATASETS, SAMPLING_RATIOS, NUM_RUNS,
    BASELINE_METHODS, HYBRID_COMBINATIONS, HYBRID_ALPHA_VALUES,
    RESULTS_DIR, FIGURES_DIR,
    FF_FORWARD_PROB_SCALEDOWN, FF_FORWARD_PROB_BACKTIME,
    is_temporal_dataset, list_all_datasets
)


def print_banner():
    """Print project banner."""
    print("\n" + "="*70)
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     EVALUATING HYBRID SAMPLING STRATEGIES FOR LARGE GRAPHS           ║
║                                                                      ║
║     Based on: Leskovec & Faloutsos, KDD 2006                         ║
║     "Sampling from Large Graphs"                                     ║
║                                                                      ║
║     FEATURES:                                                        ║
║     - Scale-down evaluation (static datasets)                        ║
║     - TRUE Back-in-time evaluation (temporal datasets)               ║
║     - S1-S9 properties with S6 (hop-plot on largest WCC)            ║
║     - Log-transform for power-law distributions                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def run_quick_test(args):
    """Run quick test with minimal configuration."""
    print("\n" + "="*70)
    print("RUNNING QUICK TEST")
    print("="*70)
    
    from src.experiment import ExperimentRunner, ExperimentConfig
    from src.experiment import generate_summary_table, compare_baseline_vs_hybrid
    
    config = ExperimentConfig(
        datasets=["cit-HepTh"],
        sampling_ratios=[0.15],
        num_runs=3,
        baseline_methods=["RN", "RW", "FF"],
        hybrid_combinations=[("RN", "RW"), ("RPN", "FF")],
        alpha_values=[0.5],
        sampling_goal=args.goal,
        include_s6=not args.no_s6
    )
    
    runner = ExperimentRunner(config)
    results_df = runner.run()
    runner.save_results(results_df, prefix="quick_test")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print("\n--- Results at 15% sampling ratio ---\n")
    summary = generate_summary_table(results_df, ratio=0.15, include_s6=not args.no_s6)
    print(summary.to_string())
    
    print("\n--- Baseline vs Hybrid Comparison ---\n")
    comparison = compare_baseline_vs_hybrid(results_df)
    print(comparison.to_string())
    
    # Print best methods
    print("\n--- Best Methods by Dataset ---\n")
    for dataset in results_df['dataset'].unique():
        subset = results_df[results_df['dataset'] == dataset]
        best = subset.loc[subset['AVG'].idxmin()]
        print(f"  {dataset:12s}: {best['method']:20s} (AVG={best['AVG']:.4f})")
    
    return results_df


def run_full_experiment(args):
    """Run full experiment on all static datasets."""
    print("\n" + "="*70)
    print(f"RUNNING FULL EXPERIMENT ({args.goal.upper()})")
    print("="*70)
    
    from src.experiment import ExperimentRunner, ExperimentConfig
    from src.experiment import generate_summary_table, compare_baseline_vs_hybrid
    
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    
    config = ExperimentConfig(
        datasets=datasets,
        sampling_ratios=SAMPLING_RATIOS,
        num_runs=args.runs,
        sampling_goal=args.goal,
        include_s6=not args.no_s6
    )
    
    runner = ExperimentRunner(config)
    results_df = runner.run()
    runner.save_results(results_df, prefix="full_experiment")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for ratio in SAMPLING_RATIOS:
        print(f"\n--- Results at {ratio*100:.0f}% sampling ratio ---\n")
        summary = generate_summary_table(results_df, ratio=ratio, include_s6=not args.no_s6)
        print(summary.to_string())
    
    print("\n--- Baseline vs Hybrid Comparison ---\n")
    comparison = compare_baseline_vs_hybrid(results_df)
    print(comparison.to_string())
    
    return results_df


def run_temporal_experiment(args):
    """
    Run TRUE back-in-time experiment with temporal datasets.
    
    This uses real timestamps to create T1-T5 time slices and evaluates
    whether a sample from T5 can represent properties of T1-T4.
    """
    print("\n" + "="*70)
    print("RUNNING TEMPORAL BACK-IN-TIME EXPERIMENT")
    print("="*70)
    
    from src.temporal_utils import (
        TemporalGraphLoader, 
        BackInTimeEvaluator,
        run_back_in_time_experiment,
        print_back_in_time_results
    )
    
    # Select temporal dataset
    if args.dataset and args.dataset in TEMPORAL_DATASETS:
        dataset = args.dataset
    else:
        dataset = "cit-HepTh"  # Default temporal dataset
        print(f"\nUsing default temporal dataset: {dataset}")
    
    print(f"\nDataset: {dataset}")
    print(f"Description: {TEMPORAL_DATASETS[dataset]['description']}")
    print(f"Time range: {TEMPORAL_DATASETS[dataset]['time_range']}")
    
    # Define methods to test
    methods = ["RN", "RW", "FF", "HYB-RN-RW", "HYB-RPN-FF"]
    
    print(f"\nMethods to test: {methods}")
    print(f"Sampling ratio: {args.ratio*100:.0f}%")
    print(f"Runs per method: {args.runs}")
    print(f"FF probability (back-in-time): {FF_FORWARD_PROB_BACKTIME}")
    
    # Run experiment
    results = run_back_in_time_experiment(
        dataset_name=dataset,
        sampling_ratio=args.ratio,
        methods=methods,
        num_runs=args.runs
    )
    
    # Print results
    print_back_in_time_results(results)
    
    # Save results
    import pandas as pd
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"temporal_{dataset}_{timestamp}.csv")
    
    # Convert to DataFrame
    rows = []
    for method, metrics in results.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    return results


def run_visualizations(results_df):
    """Generate visualization figures."""
    print("\nGenerating visualizations...")
    
    try:
        from src.visualizer import generate_all_figures
        generate_all_figures(results_df, FIGURES_DIR)
        print(f"\n✓ All figures saved to: {FIGURES_DIR}")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Graph Sampling Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                    Quick test on cit-HepTh
  python main.py --full                     Full experiment on all datasets
  python main.py --dataset hep-th           Test single dataset
  python main.py --temporal                 TRUE back-in-time with temporal data
  python main.py --temporal --dataset cit-HepPh   Back-in-time on specific dataset
  python main.py --goal back_in_time        Back-in-time (simplified, no timestamps)
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true",
                           help="Quick test with reduced parameters")
    mode_group.add_argument("--full", action="store_true",
                           help="Full experiment on all datasets")
    mode_group.add_argument("--temporal", action="store_true",
                           help="TRUE back-in-time with temporal datasets (T1-T5)")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default=None,
                       help="Run on specific dataset")
    
    # Goal selection
    parser.add_argument("--goal", type=str, default="scale_down",
                       choices=["scale_down", "back_in_time"],
                       help="Sampling goal (affects FF probability)")
    
    # Parameters
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                       help=f"Number of runs per configuration (default: {NUM_RUNS})")
    parser.add_argument("--ratio", type=float, default=0.15,
                       help="Sampling ratio for temporal experiments (default: 0.15)")
    
    # Evaluation options
    parser.add_argument("--no-s6", action="store_true",
                       help="Exclude S6 (hop-plot on largest WCC) from evaluation")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    
    # Info
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # List datasets
    if args.list_datasets:
        datasets = list_all_datasets()
        print("\nAvailable Datasets:")
        print("\n  STATIC (for scale-down evaluation):")
        for name in datasets["static"]:
            info = DATASETS[name]
            print(f"    - {name}: {info['description']}")
        print("\n  TEMPORAL (for TRUE back-in-time evaluation):")
        for name in datasets["temporal"]:
            info = TEMPORAL_DATASETS[name]
            print(f"    - {name}: {info['description']}")
            print(f"      Time range: {info['time_range']}")
        return
    
    # Run appropriate experiment
    results = None
    
    if args.temporal:
        # TRUE back-in-time with temporal datasets
        results = run_temporal_experiment(args)
    elif args.quick:
        results = run_quick_test(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
    elif args.full:
        results = run_full_experiment(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
    else:
        # Default: quick test
        results = run_quick_test(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    if results is not None:
        print(f"\nResults saved to: {RESULTS_DIR}")
        if not args.no_viz and not args.temporal:
            print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()