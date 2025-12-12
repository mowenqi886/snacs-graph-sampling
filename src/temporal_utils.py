
import os
import gzip
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, OrderedDict
import urllib.request

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, NUM_TIME_SNAPSHOTS, TIME_SNAPSHOT_METHOD


# =============================================================================
# Dataset URLs for Temporal Data
# =============================================================================

TEMPORAL_DATASETS = {
    "cit-HepTh": {
        "edges_url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
        "dates_url": "https://snap.stanford.edu/data/cit-HepTh-dates.txt.gz",
        "edges_file": "cit-HepTh.txt.gz",
        "dates_file": "cit-HepTh-dates.txt.gz",
        "directed": True,
        "description": "ArXiv HEP-TH citation network with timestamps (1993-2003)"
    },
    "cit-HepPh": {
        "edges_url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
        "dates_url": "https://snap.stanford.edu/data/cit-HepPh-dates.txt.gz",
        "edges_file": "cit-HepPh.txt.gz",
        "dates_file": "cit-HepPh-dates.txt.gz",
        "directed": True,
        "description": "ArXiv HEP-PH citation network with timestamps (1993-2003)"
    }
}


# =============================================================================
# Temporal Graph Loader
# =============================================================================

class TemporalGraphLoader:
    """
    Loader for temporal graph datasets with node timestamps.
    
    Handles downloading, parsing, and creating time-sliced snapshots
    of citation networks.
    """
    
    def __init__(self, dataset_name: str, data_dir: str = DATA_DIR):
        """
        Initialize temporal graph loader.
        
        Args:
            dataset_name: Name of dataset ("cit-HepTh" or "cit-HepPh")
            data_dir: Directory to store data files
        """
        if dataset_name not in TEMPORAL_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Available: {list(TEMPORAL_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.dataset_info = TEMPORAL_DATASETS[dataset_name]
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _download_file(self, url: str, filename: str) -> str:
        """Download file if not exists."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"    Saved to {filepath}")
        else:
            print(f"  File already exists: {filepath}")
        
        return filepath
    
    def _parse_dates_file(self, dates_filepath: str) -> Dict[int, datetime]:
        """
        Parse the dates file to get node timestamps.
        
        Format: node_id \t date_string
        Date format: YYYY-MM-DD (e.g., 2000-02-04)
        
        Args:
            dates_filepath: Path to dates file
        
        Returns:
            Dictionary mapping node_id -> datetime
        """
        node_times = {}
        
        open_func = gzip.open if dates_filepath.endswith('.gz') else open
        
        with open_func(dates_filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0])
                        date_str = parts[1]
                        
                        # Parse date (format: YYYY-MM-DD)
                        try:
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                        except ValueError:
                            try:
                                dt = datetime.strptime(date_str, "%Y/%m/%d")
                            except ValueError:
                                continue
                        
                        node_times[node_id] = dt
                        
                    except (ValueError, IndexError):
                        continue
        
        print(f"    Loaded timestamps for {len(node_times)} nodes")
        return node_times
    
    def _parse_edges_file(self, edges_filepath: str) -> List[Tuple[int, int]]:
        """
        Parse the edges file.
        
        Format: from_node \t to_node
        
        Args:
            edges_filepath: Path to edges file
        
        Returns:
            List of (from_node, to_node) tuples
        """
        edges = []
        
        open_func = gzip.open if edges_filepath.endswith('.gz') else open
        
        with open_func(edges_filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        from_node = int(parts[0])
                        to_node = int(parts[1])
                        edges.append((from_node, to_node))
                    except ValueError:
                        continue
        
        print(f"    Loaded {len(edges)} edges")
        return edges
    
    def load(self) -> Tuple[nx.Graph, Dict[int, datetime]]:
        """
        Load the full temporal graph with node timestamps.
        
        Returns:
            Tuple of (graph, node_times_dict)
        """
        print(f"\n{'='*60}")
        print(f"Loading temporal dataset: {self.dataset_name}")
        print(f"{'='*60}")
        
        # Download files
        edges_path = self._download_file(
            self.dataset_info["edges_url"],
            self.dataset_info["edges_file"]
        )
        dates_path = self._download_file(
            self.dataset_info["dates_url"],
            self.dataset_info["dates_file"]
        )
        
        # Parse files
        print("  Parsing edges...")
        edges = self._parse_edges_file(edges_path)
        
        print("  Parsing timestamps...")
        node_times = self._parse_dates_file(dates_path)
        
        # Create graph
        print("  Building graph...")
        if self.dataset_info["directed"]:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        G.add_edges_from(edges)
        
        # Add timestamps as node attributes
        for node, dt in node_times.items():
            if G.has_node(node):
                G.nodes[node]['timestamp'] = dt
        
        print(f"\n  Graph loaded:")
        print(f"    Nodes: {G.number_of_nodes():,}")
        print(f"    Edges: {G.number_of_edges():,}")
        print(f"    Nodes with timestamps: {len(node_times):,}")
        
        # Find time range
        if node_times:
            min_time = min(node_times.values())
            max_time = max(node_times.values())
            print(f"    Time range: {min_time.date()} to {max_time.date()}")
        
        return G, node_times
    
    def create_time_snapshots(self, G: nx.Graph, node_times: Dict[int, datetime],
                              num_snapshots: int = NUM_TIME_SNAPSHOTS,
                              snapshot_method: str = TIME_SNAPSHOT_METHOD) -> Dict[str, nx.Graph]:
        """
        Create time-sliced snapshots of the graph.
        
        Snapshot_1 is the earliest snapshot (smallest graph).
        Snapshot_N is the latest snapshot (full graph).
        
        IMPORTANT: These are called "Snapshots" (not "T1-T5") to avoid
        confusion with temporal METRICS T1-T5.
        
        Args:
            G: Full graph
            node_times: Dictionary mapping node_id -> datetime
            num_snapshots: Number of time snapshots to create
            snapshot_method: How to divide time
                - "equal_time": Equal time intervals
                - "equal_nodes": Equal number of nodes per snapshot
        
        Returns:
            OrderedDict {"Snapshot_1": G1, "Snapshot_2": G2, ..., "Snapshot_N": GN}
        """
        print(f"\n  Creating {num_snapshots} time snapshots...")
        
        # Get nodes with valid timestamps
        valid_nodes = [(n, t) for n, t in node_times.items() if G.has_node(n)]
        valid_nodes.sort(key=lambda x: x[1])
        
        if not valid_nodes:
            raise ValueError("No nodes with valid timestamps found")
        
        min_time = valid_nodes[0][1]
        max_time = valid_nodes[-1][1]
        
        print(f"    Valid nodes: {len(valid_nodes)}")
        print(f"    Time range: {min_time.date()} to {max_time.date()}")
        
        # Create time boundaries
        if snapshot_method == "equal_time":
            # Equal time intervals
            total_seconds = (max_time - min_time).total_seconds()
            slice_duration = total_seconds / num_snapshots
            
            boundaries = []
            for i in range(num_snapshots + 1):
                boundary_time = min_time.timestamp() + i * slice_duration
                boundaries.append(datetime.fromtimestamp(boundary_time))
            
        elif snapshot_method == "equal_nodes":
            # Equal number of nodes per cumulative snapshot
            nodes_per_slice = len(valid_nodes) // num_snapshots
            
            boundaries = [min_time]
            for i in range(1, num_snapshots):
                idx = min(i * nodes_per_slice, len(valid_nodes) - 1)
                boundaries.append(valid_nodes[idx][1])
            boundaries.append(max_time)
        
        else:
            raise ValueError(f"Unknown snapshot_method: {snapshot_method}")
        
        # Create snapshots (using OrderedDict to maintain order)
        time_snapshots = OrderedDict()
        
        for i in range(num_snapshots):
            # FIXED: Use "Snapshot_" prefix instead of "T" to avoid confusion
            snapshot_name = f"Snapshot_{i+1}"
            cutoff_time = boundaries[i + 1]
            
            # Get nodes up to this time
            snapshot_nodes = [n for n, t in valid_nodes if t <= cutoff_time]
            
            # Create subgraph with these nodes
            snapshot_graph = G.subgraph(snapshot_nodes).copy()
            
            time_snapshots[snapshot_name] = snapshot_graph
            
            print(f"    {snapshot_name}: {snapshot_graph.number_of_nodes():,} nodes, "
                  f"{snapshot_graph.number_of_edges():,} edges "
                  f"(up to {cutoff_time.date()})")
        
        return time_snapshots
    
    def get_monthly_snapshots(self, G: nx.Graph, node_times: Dict[int, datetime],
                               start_year: int = 1993,
                               end_year: int = 2003) -> Dict[str, nx.Graph]:
        """
        Create monthly snapshots of the graph.
        
        Args:
            G: Full graph
            node_times: Node timestamps
            start_year: Starting year
            end_year: Ending year
        
        Returns:
            OrderedDict {"YYYY-MM": graph} for each month
        """
        print(f"\n  Creating monthly snapshots ({start_year}-{end_year})...")
        
        # Group nodes by month
        monthly_nodes = defaultdict(set)
        
        for node, dt in node_times.items():
            if G.has_node(node):
                if start_year <= dt.year <= end_year:
                    month_key = f"{dt.year}-{dt.month:02d}"
                    monthly_nodes[month_key].add(node)
        
        # Create cumulative snapshots
        snapshots = OrderedDict()
        cumulative_nodes = set()
        
        for month_key in sorted(monthly_nodes.keys()):
            cumulative_nodes.update(monthly_nodes[month_key])
            snapshot = G.subgraph(cumulative_nodes).copy()
            snapshots[month_key] = snapshot
        
        print(f"    Created {len(snapshots)} monthly snapshots")
        
        return snapshots


# =============================================================================
# Back-in-Time Evaluator (UPDATED with T1-T5 support)
# =============================================================================

class BackInTimeEvaluator:
    """
    Evaluator for back-in-time sampling goal.
    
    Tests whether a sample from the final graph can represent properties of
    earlier snapshots, using both:
    - Static metrics (S1-S9): Measured on each snapshot
    - Temporal metrics (T1-T5): Measured across all snapshots
    
    Usage:
        evaluator = BackInTimeEvaluator(time_snapshots)
        results = evaluator.evaluate_sample(sampled_graph, sampled_snapshots)
    """
    
    def __init__(self, time_snapshots: Dict[str, nx.Graph],
                 use_log_transform: bool = True):
        """
        Initialize back-in-time evaluator.
        
        Args:
            time_snapshots: OrderedDict {Snapshot_1: G1, ..., Snapshot_N: GN}
            use_log_transform: Whether to use log-transform for KS tests
        """
        self.time_snapshots = OrderedDict(sorted(time_snapshots.items()))
        self.use_log_transform = use_log_transform
        self.num_snapshots = len(time_snapshots)
        
        # Get the final snapshot name (largest graph)
        self.final_snapshot_name = list(self.time_snapshots.keys())[-1]
        
        # Import evaluators
        from src.evaluator import GraphEvaluator
        from src.temporal_metrics import TemporalMetricsEvaluator
        
        # Create static evaluators (S1-S9) for each snapshot except the final one
        print("  Initializing static evaluators (S1-S9)...")
        self.static_evaluators = {}
        for snapshot_name, graph in self.time_snapshots.items():
            if snapshot_name != self.final_snapshot_name:
                self.static_evaluators[snapshot_name] = GraphEvaluator(
                    graph, use_log_transform=use_log_transform
                )
        print(f"    Created evaluators for: {list(self.static_evaluators.keys())}")
        
        # Create temporal evaluator (T1-T5)
        print("  Initializing temporal evaluator (T1-T5)...")
        self.temporal_evaluator = TemporalMetricsEvaluator(self.time_snapshots)
    
    def evaluate_static(self, sampled_graph: nx.Graph,
                        include_s6: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a sample against all earlier snapshots using static metrics (S1-S9).
        
        Args:
            sampled_graph: Graph sampled from final snapshot
            include_s6: Whether to include S6
        
        Returns:
            Dictionary {Snapshot_1: {S1: val, S2: val, ...}, ...}
        """
        results = {}
        
        for snapshot_name, evaluator in self.static_evaluators.items():
            results[snapshot_name] = evaluator.evaluate_all(
                sampled_graph, include_s6=include_s6
            )
        
        # Compute overall average across all snapshots
        all_avgs = [r["AVG"] for r in results.values()]
        results["S_AVG_ALL"] = float(np.mean(all_avgs))
        
        return results
    
    def evaluate_temporal(self, sampled_snapshots: Dict[str, nx.Graph]) -> Dict[str, float]:
        """
        Evaluate sampled snapshots using temporal metrics (T1-T5).
        
        Args:
            sampled_snapshots: Dict of {Snapshot_1: sampled_G1, ...}
        
        Returns:
            Dictionary with T1-T5 KS statistics and T_AVG
        """
        return self.temporal_evaluator.evaluate(sampled_snapshots)
    
    def evaluate_full(self, sampled_graph: nx.Graph,
                      sampled_snapshots: Optional[Dict[str, nx.Graph]] = None,
                      include_s6: bool = True,
                      include_temporal: bool = True) -> Dict[str, Union[Dict, float]]:
        """
        Full evaluation using both static (S1-S9) and temporal (T1-T5) metrics.
        
        Args:
            sampled_graph: Single sampled graph (for S1-S9 against each snapshot)
            sampled_snapshots: Sampled graphs at different sizes (for T1-T5)
                               If None, temporal metrics are skipped
            include_s6: Include S6 in static evaluation
            include_temporal: Include T1-T5 temporal metrics
        
        Returns:
            Dictionary with:
            - Static results per snapshot
            - S_AVG_ALL: Average S-metric across all snapshots
            - Temporal results (if include_temporal and sampled_snapshots provided)
            - T_AVG: Average T-metric
            - COMBINED_AVG: Overall average of S_AVG_ALL and T_AVG
        """
        results = {}
        
        # Static metrics (S1-S9) against each historical snapshot
        static_results = self.evaluate_static(sampled_graph, include_s6)
        results['static'] = static_results
        results['S_AVG_ALL'] = static_results['S_AVG_ALL']
        
        # Temporal metrics (T1-T5)
        if include_temporal and sampled_snapshots is not None:
            temporal_results = self.evaluate_temporal(sampled_snapshots)
            results['temporal'] = temporal_results
            results['T_AVG'] = temporal_results['T_AVG']
            
            # Combined average
            results['COMBINED_AVG'] = (results['S_AVG_ALL'] + results['T_AVG']) / 2
        else:
            results['temporal'] = None
            results['T_AVG'] = None
            results['COMBINED_AVG'] = results['S_AVG_ALL']
        
        return results
    
    def evaluate_method(self, G_final: nx.Graph, method: str, 
                        sampling_ratio: float, num_runs: int = 10,
                        include_temporal: bool = True,
                        **kwargs) -> Dict[str, float]:
        """
        Evaluate a sampling method with multiple runs.
        
        Creates sampled graphs at different sizes to compute temporal metrics.
        
        Args:
            G_final: Final graph (to sample from)
            method: Sampling method name
            sampling_ratio: Fraction of nodes to sample
            num_runs: Number of runs
            include_temporal: Include T1-T5 metrics
            **kwargs: Additional arguments for sampler
        
        Returns:
            Mean metrics across runs
        """
        from src.samplers import sample_graph
        
        all_results = defaultdict(list)
        
        for run in range(num_runs):
            # Sample from final graph
            n_samples = int(G_final.number_of_nodes() * sampling_ratio)
            S = sample_graph(G_final, method, n_samples, random_state=run, **kwargs)
            
            # Create sampled snapshots for temporal metrics
            sampled_snapshots = None
            if include_temporal:
                sampled_snapshots = self._create_sampled_snapshots(
                    S, n_samples, sampling_ratio
                )
            
            # Full evaluation
            run_results = self.evaluate_full(
                S, sampled_snapshots, 
                include_s6=True, 
                include_temporal=include_temporal
            )
            
            # Collect results
            all_results['S_AVG_ALL'].append(run_results['S_AVG_ALL'])
            
            if run_results['T_AVG'] is not None:
                all_results['T_AVG'].append(run_results['T_AVG'])
            
            all_results['COMBINED_AVG'].append(run_results['COMBINED_AVG'])
            
            # Also collect individual metrics
            for snapshot_name, metrics in run_results['static'].items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        all_results[f"{snapshot_name}_{metric_name}"].append(value)
            
            if run_results['temporal'] is not None:
                for metric_name, value in run_results['temporal'].items():
                    all_results[f"T_{metric_name}"].append(value)
        
        # Compute means
        mean_results = {k: float(np.mean(v)) for k, v in all_results.items()}
        
        return mean_results
    
    def _create_sampled_snapshots(self, sampled_graph: nx.Graph, 
                                   n_samples: int, 
                                   sampling_ratio: float) -> Dict[str, nx.Graph]:
        """
        Create snapshots of sampled graph at different sizes for temporal metrics.
        
        Simulates what the sampled graph would look like at earlier times.
        
        Args:
            sampled_graph: The full sampled graph
            n_samples: Number of nodes in full sample
            sampling_ratio: Original sampling ratio
        
        Returns:
            OrderedDict of sampled snapshots
        """
        sampled_snapshots = OrderedDict()
        nodes = list(sampled_graph.nodes())
        
        # Create snapshots at sizes proportional to original snapshots
        for i, (snapshot_name, orig_snapshot) in enumerate(self.time_snapshots.items()):
            # Calculate target size based on ratio
            orig_ratio = orig_snapshot.number_of_nodes() / self.time_snapshots[self.final_snapshot_name].number_of_nodes()
            target_size = max(1, int(n_samples * orig_ratio))
            target_size = min(target_size, len(nodes))
            
            # Take first target_size nodes (simulating growth)
            snapshot_nodes = nodes[:target_size]
            snapshot_graph = sampled_graph.subgraph(snapshot_nodes).copy()
            
            sampled_snapshots[snapshot_name] = snapshot_graph
        
        return sampled_snapshots


# =============================================================================
# Convenience Functions
# =============================================================================

def run_back_in_time_experiment(dataset_name: str = "cit-HepTh",
                                 sampling_ratio: float = 0.15,
                                 methods: List[str] = None,
                                 num_runs: int = 10,
                                 include_temporal: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Run complete back-in-time experiment on a temporal dataset.
    
    Evaluates sampling methods using both S1-S9 and T1-T5 metrics.
    
    Args:
        dataset_name: Name of temporal dataset
        sampling_ratio: Fraction of nodes to sample
        methods: List of methods to test (default: basic set)
        num_runs: Number of runs per method
        include_temporal: Whether to include T1-T5 metrics
    
    Returns:
        Results dictionary {method: {metric: value}}
    """
    from config import FF_FORWARD_PROB_BACKTIME
    
    if methods is None:
        methods = ["RN", "RW", "FF", "HYB-RN-RW", "HYB-RPN-FF"]
    
    print(f"\n{'='*70}")
    print(f"BACK-IN-TIME EXPERIMENT: {dataset_name}")
    print(f"{'='*70}")
    
    # Load temporal graph
    loader = TemporalGraphLoader(dataset_name)
    G, node_times = loader.load()
    
    # Create time snapshots
    time_snapshots = loader.create_time_snapshots(G, node_times)
    
    # Get final snapshot for sampling
    final_snapshot_name = list(time_snapshots.keys())[-1]
    G_final = time_snapshots[final_snapshot_name]
    n_samples = int(G_final.number_of_nodes() * sampling_ratio)
    
    print(f"\n  Sampling {n_samples} nodes ({sampling_ratio*100:.0f}%) from {final_snapshot_name}")
    print(f"  Evaluating against earlier snapshots")
    print(f"  Include temporal metrics (T1-T5): {include_temporal}")
    
    # Create evaluator
    evaluator = BackInTimeEvaluator(time_snapshots)
    
    # Run experiments
    results = {}
    
    for method in methods:
        print(f"\n  Testing {method}...")
        
        kwargs = {}
        if "FF" in method:
            kwargs["forward_prob"] = FF_FORWARD_PROB_BACKTIME  # 0.2 for back-in-time
        if method.startswith("HYB-"):
            kwargs["alpha"] = 0.5
        
        method_results = evaluator.evaluate_method(
            G_final, method, sampling_ratio, 
            num_runs=num_runs, 
            include_temporal=include_temporal,
            **kwargs
        )
        
        results[method] = method_results
        
        # Print summary
        s_avg = method_results.get("S_AVG_ALL", 0)
        t_avg = method_results.get("T_AVG", "N/A")
        combined = method_results.get("COMBINED_AVG", s_avg)
        
        print(f"    S_AVG_ALL: {s_avg:.4f}")
        if include_temporal:
            print(f"    T_AVG: {t_avg:.4f}")
            print(f"    COMBINED_AVG: {combined:.4f}")
    
    return results


def run_all_temporal_experiments(sampling_ratio: float = 0.15,
                                  methods: List[str] = None,
                                  num_runs: int = 10,
                                  include_temporal: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run back-in-time experiment on ALL temporal datasets.
    
    Args:
        sampling_ratio: Fraction of nodes to sample
        methods: List of methods to test
        num_runs: Number of runs per method
        include_temporal: Whether to include T1-T5 metrics
    
    Returns:
        Results dictionary {dataset: {method: {metric: value}}}
    """
    all_results = {}
    
    for dataset_name in TEMPORAL_DATASETS.keys():
        print(f"\n\n{'#'*70}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*70}")
        
        results = run_back_in_time_experiment(
            dataset_name=dataset_name,
            sampling_ratio=sampling_ratio,
            methods=methods,
            num_runs=num_runs,
            include_temporal=include_temporal
        )
        
        all_results[dataset_name] = results
    
    return all_results


def print_back_in_time_results(results: Dict[str, Dict[str, float]], 
                                include_temporal: bool = True) -> None:
    """
    Print back-in-time experiment results in a formatted table.
    
    Args:
        results: Results from run_back_in_time_experiment
        include_temporal: Whether temporal metrics are included
    """
    print(f"\n{'='*70}")
    print("BACK-IN-TIME RESULTS SUMMARY")
    print(f"{'='*70}")
    
    # Header
    if include_temporal:
        header = f"{'Method':<20}{'S_AVG_ALL':<12}{'T_AVG':<12}{'COMBINED':<12}"
    else:
        header = f"{'Method':<20}{'S_AVG_ALL':<12}"
    
    print(header)
    print("-" * len(header))
    
    # Sort by combined average (or S_AVG_ALL if no temporal)
    sort_key = "COMBINED_AVG" if include_temporal else "S_AVG_ALL"
    sorted_methods = sorted(results.keys(), 
                           key=lambda m: results[m].get(sort_key, 1.0))
    
    for method in sorted_methods:
        s_avg = results[method].get("S_AVG_ALL", 1.0)
        
        if include_temporal:
            t_avg = results[method].get("T_AVG", 1.0)
            combined = results[method].get("COMBINED_AVG", 1.0)
            row = f"{method:<20}{s_avg:<12.4f}{t_avg:<12.4f}{combined:<12.4f}"
        else:
            row = f"{method:<20}{s_avg:<12.4f}"
        
        print(row)
    
    print(f"{'='*70}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Back-in-Time Evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=list(TEMPORAL_DATASETS.keys()) + [None],
                       help="Temporal dataset to use (None = all)")
    parser.add_argument("--ratio", type=float, default=0.15,
                       help="Sampling ratio")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of runs per method")
    parser.add_argument("--no-temporal", action="store_true",
                       help="Skip temporal metrics (T1-T5)")
    
    args = parser.parse_args()
    
    include_temporal = not args.no_temporal
    
    if args.dataset:
        # Run on single dataset
        results = run_back_in_time_experiment(
            dataset_name=args.dataset,
            sampling_ratio=args.ratio,
            num_runs=args.runs,
            include_temporal=include_temporal
        )
        print_back_in_time_results(results, include_temporal)
    else:
        # Run on all datasets
        all_results = run_all_temporal_experiments(
            sampling_ratio=args.ratio,
            num_runs=args.runs,
            include_temporal=include_temporal
        )
        
        for dataset_name, results in all_results.items():
            print(f"\n\n{'='*70}")
            print(f"RESULTS FOR: {dataset_name}")
            print_back_in_time_results(results, include_temporal)
