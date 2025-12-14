
import os
import gzip
import shutil
import urllib.request
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Dataset Configuration 
# =============================================================================
try:
    from config import DATA_DIR, DATASETS, TEMPORAL_DATASETS
except ImportError:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    DATASETS = {
        "cit-HepTh": {
            "url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
            "filename": "cit-HepTh.txt.gz",
            "directed": True,
            "description": "ArXiv HEP-TH citation network",
        },
        "cit-HepPh": {
            "url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
            "filename": "cit-HepPh.txt.gz",
            "directed": True,
            "description": "ArXiv HEP-PH citation network",
        },
        "soc-Epinions1": {
            "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
            "filename": "soc-Epinions1.txt.gz",
            "directed": True,
            "description": "Epinions trust network",
        },
        "wiki-Vote": {
            "url": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
            "filename": "wiki-Vote.txt.gz",
            "directed": True,
            "description": "Wikipedia voting network",
        },
        "p2p-Gnutella31": {
            "url": "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz",
            "filename": "p2p-Gnutella31.txt.gz",
            "directed": True,
            "description": "Gnutella P2P network",
        },
    }
    
    TEMPORAL_DATASETS = {
        "cit-HepTh": {
            "edges_url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
            "dates_url": "https://snap.stanford.edu/data/cit-HepTh-dates.txt.gz",
            "edges_file": "cit-HepTh.txt.gz",
            "dates_file": "cit-HepTh-dates.txt.gz",
            "directed": True,
            "description": "ArXiv HEP-TH with timestamps",
        },
        "cit-HepPh": {
            "edges_url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
            "dates_url": "https://snap.stanford.edu/data/cit-HepPh-dates.txt.gz",
            "edges_file": "cit-HepPh.txt.gz",
            "dates_file": "cit-HepPh-dates.txt.gz",
            "directed": True,
            "description": "ArXiv HEP-PH with timestamps",
        },
    }


# =============================================================================
# Data Loader Class
# =============================================================================

class DataLoader:
    """
    Handles loading and preprocessing of graph datasets.
    
    Supports:
    - SNAP format edge lists (.txt, .txt.gz)
    - Temporal datasets with timestamp files
    - Synthetic graph generation
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_file(self, url: str, filename: str) -> Path:
        """
        Download a file from URL if not exists.
        
        Args:
            url: URL to download from
            filename: Local filename
        
        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"  Downloading {filename}...")
            
            request = urllib.request.Request(
                url, 
                headers={"User-Agent": "Mozilla/5.0"}
            )
            
            with urllib.request.urlopen(request) as response:
                with open(filepath, 'wb') as f:
                    shutil.copyfileobj(response, f)
            
            print(f"    Saved to {filepath}")
        
        return filepath
    
    def _extract_gzip(self, gz_path: Path) -> Path:
        """
        Extract a .gz file.
        
        Args:
            gz_path: Path to .gz file
        
        Returns:
            Path to extracted file
        """
        if not gz_path.suffix == '.gz':
            return gz_path
        
        output_path = gz_path.with_suffix('')
        
        if not output_path.exists():
            print(f"  Extracting {gz_path.name}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        return output_path
    
    def _parse_edge_list(self, filepath: Path, directed: bool = True) -> nx.Graph:
        """
        Parse a SNAP-format edge list file.
        
        Format: node1 \t node2 (one edge per line)
        Lines starting with # are comments.
        
        Args:
            filepath: Path to edge list file
            directed: Whether to create directed graph
        
        Returns:
            NetworkX graph
        """
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Handle gzipped files
        if filepath.suffix == '.gz':
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        with open_func(filepath, mode, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        G.add_edge(u, v)
                    except ValueError:
                        continue
        
        return G
    
    def load_dataset(self, dataset_name: str) -> nx.Graph:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of dataset (e.g., "cit-HepTh")
        
        Returns:
            NetworkX graph
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(DATASETS.keys())}")
        
        info = DATASETS[dataset_name]
        
        print(f"\n  Loading {dataset_name}...")
        print(f"    Description: {info['description']}")
        
        # Download if needed
        gz_path = self._download_file(info['url'], info['filename'])
        
        # Extract if gzipped
        txt_path = self._extract_gzip(gz_path)
        
        # Parse edge list
        G = self._parse_edge_list(txt_path, directed=info.get('directed', True))
        
        print(f"    Nodes: {G.number_of_nodes():,}")
        print(f"    Edges: {G.number_of_edges():,}")
        
        return G
    
    def load_temporal_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[int, datetime]]:
        """
        Load a temporal dataset with node timestamps.
        
        Args:
            dataset_name: Name of temporal dataset
        
        Returns:
            Tuple of (graph, node_times_dict)
        """
        if dataset_name not in TEMPORAL_DATASETS:
            raise ValueError(f"Unknown temporal dataset: {dataset_name}. "
                           f"Available: {list(TEMPORAL_DATASETS.keys())}")
        
        info = TEMPORAL_DATASETS[dataset_name]
        
        print(f"\n  Loading temporal dataset {dataset_name}...")
        print(f"    Description: {info['description']}")
        print(f"    Time range: {info.get('time_range', 'N/A')}")
        
        # Download edges file
        edges_gz = self._download_file(info['edges_url'], info['edges_file'])
        edges_txt = self._extract_gzip(edges_gz)
        
        # Download dates file
        dates_gz = self._download_file(info['dates_url'], info['dates_file'])
        dates_txt = self._extract_gzip(dates_gz)
        
        # Parse edges
        G = self._parse_edge_list(edges_txt, directed=info.get('directed', True))
        
        # Parse timestamps
        node_times = self._parse_dates_file(dates_txt)
        
        print(f"    Nodes: {G.number_of_nodes():,}")
        print(f"    Edges: {G.number_of_edges():,}")
        print(f"    Nodes with timestamps: {len(node_times):,}")
        
        return G, node_times
    
    def _parse_dates_file(self, filepath: Path) -> Dict[int, datetime]:
        """
        Parse a dates file (node_id \t date).
        
        Args:
            filepath: Path to dates file
        
        Returns:
            Dictionary mapping node_id -> datetime
        """
        node_times = {}
        
        if filepath.suffix == '.gz':
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        with open_func(filepath, mode, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0])
                        date_str = parts[1]
                        
                        # Try different date formats
                        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"]:
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                node_times[node_id] = dt
                                break
                            except ValueError:
                                continue
                    except (ValueError, IndexError):
                        continue
        
        return node_times
    
    def create_synthetic_graph(self, graph_type: str = "ba", **kwargs) -> nx.Graph:
        """
        Create a synthetic graph for testing.
        
        Args:
            graph_type: Type of graph
                - "ba": Barabási-Albert preferential attachment
                - "er": Erdős-Rényi random graph
                - "ws": Watts-Strogatz small world
            **kwargs: Graph parameters
        
        Returns:
            NetworkX graph
        """
        if graph_type == "ba":
            n = kwargs.get('n', 1000)
            m = kwargs.get('m', 3)
            seed = kwargs.get('seed', None)
            G = nx.barabasi_albert_graph(n, m, seed=seed)
        
        elif graph_type == "er":
            n = kwargs.get('n', 1000)
            p = kwargs.get('p', 0.01)
            seed = kwargs.get('seed', None)
            G = nx.erdos_renyi_graph(n, p, seed=seed)
        
        elif graph_type == "ws":
            n = kwargs.get('n', 1000)
            k = kwargs.get('k', 4)
            p = kwargs.get('p', 0.3)
            seed = kwargs.get('seed', None)
            G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        # Convert to directed if requested
        if kwargs.get('directed', False):
            G = G.to_directed()
        
        return G
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            Dictionary with dataset information
        """
        if dataset_name in DATASETS:
            return DATASETS[dataset_name]
        elif dataset_name in TEMPORAL_DATASETS:
            return TEMPORAL_DATASETS[dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def list_datasets(self) -> Dict[str, List[str]]:
        """
        List all available datasets.
        
        Returns:
            Dictionary with 'static' and 'temporal' dataset lists
        """
        return {
            'static': list(DATASETS.keys()),
            'temporal': list(TEMPORAL_DATASETS.keys())
        }
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset has been downloaded.
        
        Args:
            dataset_name: Dataset name
        
        Returns:
            True if downloaded, False otherwise
        """
        if dataset_name in DATASETS:
            filename = DATASETS[dataset_name]['filename']
        elif dataset_name in TEMPORAL_DATASETS:
            filename = TEMPORAL_DATASETS[dataset_name]['edges_file']
        else:
            return False
        
        filepath = self.data_dir / filename
        txt_path = filepath.with_suffix('') if filepath.suffix == '.gz' else filepath
        
        return txt_path.exists()


# =============================================================================
# Convenience Functions
# =============================================================================

def load_dataset(dataset_name: str, data_dir: str = DATA_DIR) -> nx.Graph:
    """
    Convenience function to load a dataset.
    
    Args:
        dataset_name: Name of dataset
        data_dir: Data directory
    
    Returns:
        NetworkX graph
    """
    loader = DataLoader(data_dir)
    return loader.load_dataset(dataset_name)


def load_temporal_dataset(dataset_name: str, 
                          data_dir: str = DATA_DIR) -> Tuple[nx.Graph, Dict[int, datetime]]:
    """
    Convenience function to load a temporal dataset.
    
    Args:
        dataset_name: Name of temporal dataset
        data_dir: Data directory
    
    Returns:
        Tuple of (graph, node_times)
    """
    loader = DataLoader(data_dir)
    return loader.load_temporal_dataset(dataset_name)


def download_all_datasets(data_dir: str = DATA_DIR) -> None:
    """
    Download all datasets.
    
    Args:
        data_dir: Data directory
    """
    loader = DataLoader(data_dir)
    
    print("Downloading all datasets...")
    print("="*60)
    
    # Download static datasets
    for name in DATASETS.keys():
        try:
            loader.load_dataset(name)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # Download temporal datasets (just edges + dates files)
    for name in TEMPORAL_DATASETS.keys():
        try:
            loader.load_temporal_dataset(name)
            print(f"  ✓ {name} (temporal)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    print("="*60)
    print("Download complete!")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DATA LOADER MODULE DEMO")
    print("="*70)
    
    loader = DataLoader()
    
    # List available datasets
    print("\nAvailable datasets:")
    datasets = loader.list_datasets()
    print(f"  Static: {datasets['static']}")
    print(f"  Temporal: {datasets['temporal']}")
    
    # Create synthetic graph
    print("\nCreating synthetic BA graph...")
    G = loader.create_synthetic_graph("ba", n=500, m=3, seed=42)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Try loading a real dataset
    print("\nAttempting to load cit-HepTh...")
    try:
        G = loader.load_dataset("cit-HepTh")
        print(f"  Successfully loaded!")
    except Exception as e:
        print(f"  Note: Could not load (may need download): {e}")
    
    print("\n✓ Data loader demo completed!")
