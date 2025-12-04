import os
import gzip
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import urllib.request

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


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
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  Saved to {filepath}")
        else:
            print(f"File already exists: {filepath}")
        
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
                            # Try alternative format
                            try:
                                dt = datetime.strptime(date_str, "%Y/%m/%d")
                            except ValueError:
                                continue
                        
                        node_times[node_id] = dt
                        
                    except (ValueError, IndexError):
                        continue
        
        print(f"  Loaded timestamps for {len(node_times)} nodes")
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
        
        print(f"  Loaded {len(edges)} edges")
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
        print("Parsing edges...")
        edges = self._parse_edges_file(edges_path)
        
        print("Parsing timestamps...")
        node_times = self._parse_dates_file(dates_path)
        
        # Create graph
        print("Building graph...")
        if self.dataset_info["directed"]:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        G.add_edges_from(edges)
        
        # Add timestamps as node attributes
        for node, dt in node_times.items():
            if G.has_node(node):
                G.nodes[node]['timestamp'] = dt
        
        print(f"\nLoaded graph:")
        print(f"  Nodes: {G.number_of_nodes():,}")
        print(f"  Edges: {G.number_of_edges():,}")
        print(f"  Nodes with timestamps: {len(node_times):,}")
        
        # Find time range
        if node_times:
            min_time = min(node_times.values())
            max_time = max(node_times.values())
            print(f"  Time range: {min_time.date()} to {max_time.date()}")
        
        return G, node_times
    
    def create_time_slices(self, G: nx.Graph, node_times: Dict[int, datetime],
                           num_slices: int = 5,
                           slice_method: str = "equal_time") -> Dict[str, nx.Graph]:
        """
        Create time-sliced snapshots of the graph.
        
        T1 is the earliest slice, T5 is the latest (full graph).
        
        Args:
            G: Full graph
            node_times: Dictionary mapping node_id -> datetime
            num_slices: Number of time slices (default: 5 for T1-T5)
            slice_method: How to divide time
                - "equal_time": Equal time intervals
                - "equal_nodes": Equal number of nodes per slice
        
        Returns:
            Dictionary {T1: G1, T2: G2, ..., T5: G5}
        """
        print(f"\nCreating {num_slices} time slices...")
        
        # Get nodes with valid timestamps
        valid_nodes = [(n, t) for n, t in node_times.items() if G.has_node(n)]
        valid_nodes.sort(key=lambda x: x[1])
        
        if not valid_nodes:
            raise ValueError("No nodes with valid timestamps found")
        
        min_time = valid_nodes[0][1]
        max_time = valid_nodes[-1][1]
        
        print(f"  Valid nodes: {len(valid_nodes)}")
        print(f"  Time range: {min_time.date()} to {max_time.date()}")
        
        # Create time boundaries
        if slice_method == "equal_time":
            # Equal time intervals
            total_seconds = (max_time - min_time).total_seconds()
            slice_duration = total_seconds / num_slices
            
            boundaries = []
            for i in range(num_slices + 1):
                boundary_time = min_time.timestamp() + i * slice_duration
                boundaries.append(datetime.fromtimestamp(boundary_time))
            
        elif slice_method == "equal_nodes":
            # Equal number of nodes per cumulative slice
            nodes_per_slice = len(valid_nodes) // num_slices
            
            boundaries = [min_time]
            for i in range(1, num_slices):
                idx = min(i * nodes_per_slice, len(valid_nodes) - 1)
                boundaries.append(valid_nodes[idx][1])
            boundaries.append(max_time)
        
        else:
            raise ValueError(f"Unknown slice_method: {slice_method}")
        
        # Create snapshots
        time_slices = {}
        
        for i in range(num_slices):
            slice_name = f"T{i+1}"
            cutoff_time = boundaries[i + 1]
            
            # Get nodes up to this time
            slice_nodes = [n for n, t in valid_nodes if t <= cutoff_time]
            
            # Create subgraph with these nodes
            slice_graph = G.subgraph(slice_nodes).copy()
            
            time_slices[slice_name] = slice_graph
            
            print(f"  {slice_name}: {slice_graph.number_of_nodes():,} nodes, "
                  f"{slice_graph.number_of_edges():,} edges "
                  f"(up to {cutoff_time.date()})")
        
        return time_slices
    
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
            Dictionary {"YYYY-MM": graph} for each month
        """
        print(f"\nCreating monthly snapshots ({start_year}-{end_year})...")
        
        # Group nodes by month
        monthly_nodes = defaultdict(set)
        
        for node, dt in node_times.items():
            if G.has_node(node):
                if start_year <= dt.year <= end_year:
                    month_key = f"{dt.year}-{dt.month:02d}"
                    monthly_nodes[month_key].add(node)
        
        # Create cumulative snapshots
        snapshots = {}
        cumulative_nodes = set()
        
        for month_key in sorted(monthly_nodes.keys()):
            cumulative_nodes.update(monthly_nodes[month_key])
            snapshot = G.subgraph(cumulative_nodes).copy()
            snapshots[month_key] = snapshot
        
        print(f"  Created {len(snapshots)} monthly snapshots")
        
        return snapshots


# =============================================================================
# Back-in-Time Evaluator
# =============================================================================

class BackInTimeEvaluator:
    """
    Evaluator for back-in-time sampling goal.
    
    Tests whether a sample from G(T5) can represent properties of
    earlier snapshots G(T1), G(T2), G(T3), G(T4).
    """
    
    def __init__(self, time_slices: Dict[str, nx.Graph],
                 use_log_transform: bool = True):
        """
        Initialize back-in-time evaluator.
        
        Args:
            time_slices: Dictionary {T1: G1, T2: G2, ..., T5: G5}
            use_log_transform: Whether to use log-transform for KS tests
        """
        self.time_slices = time_slices
        self.use_log_transform = use_log_transform
        
        # Import evaluator
        from src.evaluator import GraphEvaluator
        
        # Create evaluators for each time slice (except T5 which we sample from)
        self.evaluators = {}
        for slice_name, graph in time_slices.items():
            if slice_name != "T5":  # Don't evaluate against T5 (we sample from it)
                self.evaluators[slice_name] = GraphEvaluator(
                    graph, use_log_transform=use_log_transform
                )
        
        print(f"Initialized evaluators for: {list(self.evaluators.keys())}")
    
    def evaluate_sample(self, sampled_graph: nx.Graph,
                        include_s6: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a sample against all earlier time slices.
        
        Args:
            sampled_graph: Graph sampled from T5
            include_s6: Whether to include S6
        
        Returns:
            Dictionary {T1: {prop: ks_stat}, T2: {...}, ...}
        """
        results = {}
        
        for slice_name, evaluator in self.evaluators.items():
            results[slice_name] = evaluator.evaluate_all(
                sampled_graph, include_s6=include_s6
            )
        
        # Compute overall average across all time slices
        all_avgs = [r["AVG"] for r in results.values()]
        results["AVG_ALL"] = np.mean(all_avgs)
        
        return results
    
    def evaluate_method(self, G_T5: nx.Graph, method: str, 
                        n_samples: int, num_runs: int = 10,
                        **kwargs) -> Dict[str, float]:
        """
        Evaluate a sampling method with multiple runs.
        
        Args:
            G_T5: Graph at time T5 (to sample from)
            method: Sampling method name
            n_samples: Number of nodes to sample
            num_runs: Number of runs
            **kwargs: Additional arguments for sampler
        
        Returns:
            Mean KS statistics across runs
        """
        from src.samplers import sample_graph
        
        all_results = defaultdict(list)
        
        for run in range(num_runs):
            # Sample from T5
            S = sample_graph(G_T5, method, n_samples, random_state=run, **kwargs)
            
            # Evaluate against T1-T4
            run_results = self.evaluate_sample(S)
            
            # Collect results
            for slice_name, stats in run_results.items():
                if isinstance(stats, dict):
                    for prop, value in stats.items():
                        all_results[f"{slice_name}_{prop}"].append(value)
                else:
                    all_results[slice_name].append(stats)
        
        # Compute means
        mean_results = {k: np.mean(v) for k, v in all_results.items()}
        
        return mean_results


# =============================================================================
# Convenience Functions
# =============================================================================

def run_back_in_time_experiment(dataset_name: str = "cit-HepTh",
                                 sampling_ratio: float = 0.15,
                                 methods: List[str] = None,
                                 num_runs: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Run complete back-in-time experiment on a temporal dataset.
    
    Args:
        dataset_name: Name of temporal dataset
        sampling_ratio: Fraction of nodes to sample
        methods: List of methods to test (default: basic set)
        num_runs: Number of runs per method
    
    Returns:
        Results dictionary {method: {metric: value}}
    """
    from config import FF_FORWARD_PROB_BACKTIME
    
    if methods is None:
        methods = ["RN", "RW", "FF", "HYB-RPN-FF"]
    
    print(f"\n{'='*70}")
    print(f"BACK-IN-TIME EXPERIMENT: {dataset_name}")
    print(f"{'='*70}")
    
    # Load temporal graph
    loader = TemporalGraphLoader(dataset_name)
    G, node_times = loader.load()
    
    # Create time slices (T1-T5)
    time_slices = loader.create_time_slices(G, node_times, num_slices=5)
    
    # Get T5 for sampling
    G_T5 = time_slices["T5"]
    n_samples = int(G_T5.number_of_nodes() * sampling_ratio)
    
    print(f"\nSampling {n_samples} nodes ({sampling_ratio*100:.0f}%) from T5")
    print(f"Evaluating against T1, T2, T3, T4")
    
    # Create evaluator
    evaluator = BackInTimeEvaluator(time_slices)
    
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
            G_T5, method, n_samples, num_runs=num_runs, **kwargs
        )
        
        results[method] = method_results
        
        # Print summary
        avg_all = method_results.get("AVG_ALL", 0)
        print(f"    AVG_ALL: {avg_all:.4f}")
    
    return results


def print_back_in_time_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print back-in-time experiment results in a formatted table.
    
    Args:
        results: Results from run_back_in_time_experiment
    """
    print(f"\n{'='*70}")
    print("BACK-IN-TIME RESULTS")
    print(f"{'='*70}")
    
    # Get time slices
    time_slices = ["T1", "T2", "T3", "T4"]
    
    # Header
    header = f"{'Method':<20}"
    for t in time_slices:
        header += f"{t+'_AVG':<12}"
    header += f"{'AVG_ALL':<12}"
    print(header)
    print("-" * len(header))
    
    # Sort by AVG_ALL
    sorted_methods = sorted(results.keys(), 
                           key=lambda m: results[m].get("AVG_ALL", 1.0))
    
    for method in sorted_methods:
        row = f"{method:<20}"
        for t in time_slices:
            key = f"{t}_AVG"
            value = results[method].get(key, 1.0)
            row += f"{value:<12.4f}"
        row += f"{results[method].get('AVG_ALL', 1.0):<12.4f}"
        print(row)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Back-in-Time Evaluation")
    parser.add_argument("--dataset", type=str, default="cit-HepTh",
                       choices=["cit-HepTh", "cit-HepPh"],
                       help="Temporal dataset to use")
    parser.add_argument("--ratio", type=float, default=0.15,
                       help="Sampling ratio")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of runs per method")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_back_in_time_experiment(
        dataset_name=args.dataset,
        sampling_ratio=args.ratio,
        num_runs=args.runs
    )
    
    # Print results
    print_back_in_time_results(results)