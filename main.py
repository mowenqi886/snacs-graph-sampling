#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

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
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     EVALUATING HYBRID SAMPLING STRATEGIES FOR LARGE GRAPHS               ║
║                                                                          ║
║     Based on: Leskovec & Faloutsos, KDD 2006                             ║
║     "Sampling from Large Graphs"                                         ║
║                                                                          ║
║     EVALUATION METRICS:                                                  ║
║     - Static (S1-S9): Degree, Components, Hop-plot, Spectral, Clustering ║
║     - Temporal (T1-T5): DPL, Diameter, CC Size, Singular, Clustering     ║
║                                                                          ║
║     SAMPLING GOALS:                                                      ║
║     - Scale-down: Match properties of full static graph                  ║
║     - Back-in-time: Match temporal evolution patterns                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_quick_test(args):
    """
    Run quick test with minimal configuration.
    
    Tests basic functionality on a single dataset.
    """
    print("\n" + "="*70)
    print("MODE: QUICK TEST")
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
        sampling_goal="scale_down",
        include_s6=not args.no_s6
    )
    
    runner = ExperimentRunner(config)
    results_df = runner.run()
    runner.save_results(results_df, prefix="quick_test")
    
    # Print summary
    print("\n" + "="*70)
    print("QUICK TEST SUMMARY")
    print("="*70)
    
    print("\n--- Results at 15% sampling ratio ---\n")
    summary = generate_summary_table(results_df, ratio=0.15, include_s6=not args.no_s6)
    print(summary.to_string())
    
    return results_df


def run_scale_down_experiment(args):
    """
    Run scale-down experiment on all static datasets.
    
    Evaluates sampling methods using static metrics S1-S9.
    """
    print("\n" + "="*70)
    print("MODE: SCALE-DOWN EXPERIMENT")
    print("="*70)
    
    from src.experiment import ExperimentRunner, ExperimentConfig
    from src.experiment import generate_summary_table, compare_baseline_vs_hybrid
    
    # Select datasets
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(DATASETS.keys())  # All static datasets
    
    print(f"\nDatasets: {datasets}")
    print(f"Sampling ratios: {SAMPLING_RATIOS}")
    print(f"Runs per config: {args.runs}")
    print(f"FF probability (scale-down): {FF_FORWARD_PROB_SCALEDOWN}")
    
    config = ExperimentConfig(
        datasets=datasets,
        sampling_ratios=SAMPLING_RATIOS,
        num_runs=args.runs,
        sampling_goal="scale_down",
        include_s6=not args.no_s6
    )
    
    runner = ExperimentRunner(config)
    results_df = runner.run()
    runner.save_results(results_df, prefix="scale_down")
    
    # Print summaries
    print("\n" + "="*70)
    print("SCALE-DOWN RESULTS SUMMARY")
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
    Run back-in-time experiment on ALL temporal datasets.
    
    Evaluates sampling methods using both:
    - Static metrics (S1-S9) against historical snapshots
    - Temporal metrics (T1-T5) for evolution patterns
    
    FIXED: Now runs on ALL temporal datasets, not just one.
    """
    print("\n" + "="*70)
    print("MODE: BACK-IN-TIME EXPERIMENT (TEMPORAL)")
    print("="*70)
    
    from src.temporal_utils import (
        run_back_in_time_experiment,
        run_all_temporal_experiments,
        print_back_in_time_results,
        TEMPORAL_DATASETS
    )
    
    # Define methods to test
    methods = BASELINE_METHODS + [
        f"HYB-{node_m}-{explore_m}" 
        for (node_m, explore_m) in HYBRID_COMBINATIONS
    ]
    
    include_temporal = not args.no_temporal_metrics
    
    print(f"\nMethods: {methods}")
    print(f"Sampling ratio: {args.ratio*100:.0f}%")
    print(f"Runs per method: {args.runs}")
    print(f"Include T1-T5 metrics: {include_temporal}")
    print(f"FF probability (back-in-time): {FF_FORWARD_PROB_BACKTIME}")
    
    # FIXED: Run on ALL temporal datasets or specified one
    if args.dataset and args.dataset in TEMPORAL_DATASETS:
        # Single dataset
        datasets_to_run = [args.dataset]
    else:
        # ALL temporal datasets
        datasets_to_run = list(TEMPORAL_DATASETS.keys())
    
    print(f"\nDatasets to evaluate: {datasets_to_run}")
    
    all_results = {}
    all_rows = []
    
    for dataset in datasets_to_run:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*70}")
        
        results = run_back_in_time_experiment(
            dataset_name=dataset,
            sampling_ratio=args.ratio,
            methods=methods,
            num_runs=args.runs,
            include_temporal=include_temporal
        )
        
        all_results[dataset] = results
        
        # Print results for this dataset
        print_back_in_time_results(results, include_temporal)
        
        # Convert to DataFrame rows
        for method, metrics in results.items():
            row = {
                'dataset': dataset,
                'method': method,
                'experiment_type': 'back_in_time',
            }
            row.update(metrics)
            all_rows.append(row)
    
    # Create combined DataFrame
    results_df = pd.DataFrame(all_rows)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"temporal_all_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to: {results_file}")
    
    return results_df


def run_fullall_experiment(args):
    """
    Run COMPLETE experiment suite:
    1. Scale-down evaluation on all static datasets (S1-S9)
    2. Back-in-time evaluation on all temporal datasets (S1-S9 + T1-T5)
    
    This is the comprehensive experiment for the paper.
    """
    print("\n" + "="*70)
    print("MODE: FULL COMPREHENSIVE EXPERIMENT")
    print("="*70)
    print("\nThis will run:")
    print("  1. Scale-down evaluation on ALL static datasets (S1-S9)")
    print("  2. Back-in-time evaluation on ALL temporal datasets (S1-S9 + T1-T5)")
    print("\n" + "="*70)
    
    all_results = []
    
    # =========================================================================
    # PART 1: Scale-Down Experiments
    # =========================================================================
    print("\n")
    print("█"*70)
    print("█  PART 1: SCALE-DOWN EXPERIMENTS (Static Datasets)")
    print("█"*70)
    
    from src.experiment import ExperimentRunner, ExperimentConfig
    from src.experiment import generate_summary_table
    
    for dataset in DATASETS.keys():
        print(f"\n--- Scale-down: {dataset} ---")
        
        config = ExperimentConfig(
            datasets=[dataset],
            sampling_ratios=SAMPLING_RATIOS,
            num_runs=args.runs,
            sampling_goal="scale_down",
            include_s6=not args.no_s6
        )
        
        runner = ExperimentRunner(config)
        results_df = runner.run()
        results_df['experiment_type'] = 'scale_down'
        all_results.append(results_df)
    
    # =========================================================================
    # PART 2: Back-in-Time Experiments
    # =========================================================================
    print("\n")
    print("█"*70)
    print("█  PART 2: BACK-IN-TIME EXPERIMENTS (Temporal Datasets)")
    print("█"*70)
    
    from src.temporal_utils import (
        run_back_in_time_experiment,
        print_back_in_time_results,
        TEMPORAL_DATASETS
    )
    
    methods = ["RN", "RPN", "RDN", "RW", "FF",
               "HYB-RN-RW", "HYB-RN-FF", "HYB-RPN-FF"]
    
    include_temporal = not args.no_temporal_metrics
    
    for dataset in TEMPORAL_DATASETS.keys():
        print(f"\n--- Back-in-time: {dataset} ---")
        
        results = run_back_in_time_experiment(
            dataset_name=dataset,
            sampling_ratio=args.ratio,
            methods=methods,
            num_runs=args.runs,
            include_temporal=include_temporal
        )
        
        # Print results
        print_back_in_time_results(results, include_temporal)
        
        # Convert to DataFrame
        rows = []
        for method, metrics in results.items():
            row = {
                'dataset': dataset,
                'method': method,
                'experiment_type': 'back_in_time',
            }
            row.update(metrics)
            rows.append(row)
        
        temporal_df = pd.DataFrame(rows)
        all_results.append(temporal_df)
    
    # =========================================================================
    # Combine and Save Results
    # =========================================================================
    print("\n")
    print("█"*70)
    print("█  SAVING COMBINED RESULTS")
    print("█"*70)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"fullall_experiment_{timestamp}.csv")
    combined_df.to_csv(results_file, index=False)
    print(f"\n✓ Combined results saved to: {results_file}")
    
    # =========================================================================
    # Print Final Summary
    # =========================================================================
    print("\n")
    print("█"*70)
    print("█  FINAL SUMMARY")
    print("█"*70)
    
    # Scale-down summary
    scale_down_df = combined_df[combined_df['experiment_type'] == 'scale_down']
    if len(scale_down_df) > 0:
        print("\n--- SCALE-DOWN BEST METHODS (by AVG) ---")
        for dataset in DATASETS.keys():
            subset = scale_down_df[scale_down_df['dataset'] == dataset]
            if len(subset) > 0 and 'AVG' in subset.columns:
                best = subset.loc[subset['AVG'].idxmin()]
                print(f"  {dataset:15s}: {best['method']:20s} (AVG={best['AVG']:.4f})")
    
    # Back-in-time summary
    backtime_df = combined_df[combined_df['experiment_type'] == 'back_in_time']
    if len(backtime_df) > 0:
        print("\n--- BACK-IN-TIME BEST METHODS (by COMBINED_AVG) ---")
        sort_col = 'COMBINED_AVG' if 'COMBINED_AVG' in backtime_df.columns else 'S_AVG_ALL'
        for dataset in TEMPORAL_DATASETS.keys():
            subset = backtime_df[backtime_df['dataset'] == dataset]
            if len(subset) > 0 and sort_col in subset.columns:
                best = subset.loc[subset[sort_col].idxmin()]
                print(f"  {dataset:15s}: {best['method']:20s} ({sort_col}={best[sort_col]:.4f})")
    
    return combined_df


def run_visualizations(results_df):
    """Generate visualization figures."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    try:
        from src.visualizer import generate_all_figures
        generate_all_figures(results_df, FIGURES_DIR)
        print(f"\n✓ All figures saved to: {FIGURES_DIR}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not generate visualizations: {e}")


def list_datasets_info():
    """Print detailed information about available datasets."""
    print("\n" + "="*70)
    print("AVAILABLE DATASETS")
    print("="*70)
    
    print("\n📁 STATIC DATASETS (for Scale-Down evaluation):")
    print("   Metrics: S1-S9 (static graph properties)")
    print("-"*60)
    for name, info in DATASETS.items():
        print(f"   {name}:")
        print(f"      Description: {info['description']}")
        print(f"      Directed: {info['directed']}")
    
    print("\n📁 TEMPORAL DATASETS (for Back-in-Time evaluation):")
    print("   Metrics: S1-S9 (static) + T1-T5 (temporal)")
    print("-"*60)
    for name, info in TEMPORAL_DATASETS.items():
        print(f"   {name}:")
        print(f"      Description: {info['description']}")
        print(f"      Time range: {info.get('time_range', 'N/A')}")
    
    print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Graph Sampling Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                       Quick test on cit-HepTh
  python main.py --full                        Scale-down on all static datasets
  python main.py --full --dataset cit-HepPh    Scale-down on specific dataset
  python main.py --temporal                    Back-in-time on ALL temporal datasets
  python main.py --temporal --dataset cit-HepTh   Back-in-time on specific dataset
  python main.py --fullall                     Complete experiment (scale-down + back-in-time)
  python main.py --list-datasets               Show available datasets

Evaluation Metrics:
  Static (S1-S9):   In/Out degree, WCC, SCC, Hop-plot, Singular, Clustering
  Temporal (T1-T5): DPL, Diameter, CC Size, Singular Value, Clustering (over time)

Forest Fire Probability:
  Scale-down:    p_f = 0.7 (larger fires to match overall properties)
  Back-in-time:  p_f = 0.2 (smaller fires to mimic temporal evolution)
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true",
                           help="Quick test with reduced parameters")
    mode_group.add_argument("--full", action="store_true",
                           help="Full scale-down experiment on all static datasets (S1-S9)")
    mode_group.add_argument("--temporal", action="store_true",
                           help="Back-in-time experiment on ALL temporal datasets (S1-S9 + T1-T5)")
    mode_group.add_argument("--fullall", action="store_true",
                           help="Complete experiment: scale-down + back-in-time on all datasets")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default=None,
                       help="Run on specific dataset (overrides default)")
    
    # Parameters
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                       help=f"Number of runs per configuration (default: {NUM_RUNS})")
    parser.add_argument("--ratio", type=float, default=0.15,
                       help="Sampling ratio for temporal experiments (default: 0.15)")
    
    # Evaluation options
    parser.add_argument("--no-s6", action="store_true",
                       help="Exclude S6 (hop-plot on largest WCC) from evaluation")
    parser.add_argument("--no-temporal-metrics", action="store_true",
                       help="Skip temporal metrics T1-T5 in back-in-time experiments")
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
        list_datasets_info()
        return
    
    # Run appropriate experiment
    results = None
    
    if args.fullall:
        # Complete experiment (scale-down + back-in-time)
        results = run_fullall_experiment(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
            
    elif args.temporal:
        # Back-in-time on ALL temporal datasets
        results = run_temporal_experiment(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
            
    elif args.full:
        # Scale-down on all static datasets
        results = run_scale_down_experiment(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
            
    elif args.quick:
        # Quick test
        results = run_quick_test(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
    else:
        # Default: quick test
        print("\nNo mode specified. Running quick test...")
        print("Use --help to see available options.\n")
        results = run_quick_test(args)
        if not args.no_viz and results is not None:
            run_visualizations(results)
    
    # Final message
    print("\n" + "="*70)
    print("✓ EXPERIMENT COMPLETE")
    print("="*70)
    
    if results is not None:
        print(f"\n  Results directory: {RESULTS_DIR}")
        if not args.no_viz:
            print(f"  Figures directory: {FIGURES_DIR}")
    
    print("\n")


if __name__ == "__main__":
    main()
