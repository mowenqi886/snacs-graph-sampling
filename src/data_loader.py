import os
import gzip
import tarfile
import requests
import networkx as nx
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DATASETS


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> None:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: Source URL
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes
    """
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded to {dest_path}")


def extract_gzip(gz_path: str, output_path: str) -> None:
    """
    Extract a gzip compressed file.
    
    Args:
        gz_path: Path to .gz file
        output_path: Path for extracted content
    """
    print(f"Extracting {gz_path}...")
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    print(f"Extracted to {output_path}")


def extract_tarball(tar_path: str, output_dir: str) -> None:
    """
    Extract a tar.gz archive.
    
    Args:
        tar_path: Path to tar.gz file
        output_dir: Directory to extract contents
    """
    print(f"Extracting tarball {tar_path}...")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(output_dir)
    
    print(f"Extracted to {output_dir}")


def load_edgelist(filepath: str, directed: bool = True, 
                  comment: str = '#') -> nx.Graph:
    """
    Load a graph from edge list file.
    
    Args:
        filepath: Path to edge list file
        directed: Whether to create directed graph
        comment: Comment character in file
    
    Returns:
        NetworkX graph object
    """
    print(f"Loading graph from {filepath}...")
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Reading edges"):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith(comment):
                continue
            
            # Parse edge (handle both space and tab delimiters)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                except ValueError:
                    continue
    
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_as_graphs(data_dir: str) -> nx.Graph:
    """
    Load Autonomous Systems graph (special handling for multiple snapshots).
    Uses the latest snapshot available.
    
    Args:
        data_dir: Directory containing AS graph files
    
    Returns:
        NetworkX graph object
    """
    # Find all AS graph files
    as_dir = os.path.join(data_dir, "as-733")
    
    if not os.path.exists(as_dir):
        # Try to find extracted files
        for item in os.listdir(data_dir):
            if item.startswith("as"):
                as_dir = os.path.join(data_dir, item)
                break
    
    # Find the latest graph file
    graph_files = []
    for f in os.listdir(as_dir):
        if f.startswith("as") and f.endswith(".txt"):
            graph_files.append(f)
    
    if not graph_files:
        raise FileNotFoundError(f"No AS graph files found in {as_dir}")
    
    # Use the latest file (highest date)
    latest_file = sorted(graph_files)[-1]
    filepath = os.path.join(as_dir, latest_file)
    
    print(f"Using AS graph file: {latest_file}")
    return load_edgelist(filepath, directed=False)


def download_dataset(name: str) -> str:
    """
    Download a dataset if not already present.
    
    Args:
        name: Dataset name (key from DATASETS config)
    
    Returns:
        Path to the downloaded/extracted file
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[name]
    url = dataset_info["url"]
    
    # Determine file paths
    filename = os.path.basename(url)
    download_path = os.path.join(DATA_DIR, filename)
    
    # Download if not exists
    if not os.path.exists(download_path):
        download_file(url, download_path)
    else:
        print(f"File already exists: {download_path}")
    
    # Extract based on file type
    if filename.endswith(".tar.gz"):
        extract_tarball(download_path, DATA_DIR)
        return DATA_DIR
    elif filename.endswith(".gz"):
        extracted_path = download_path[:-3]  # Remove .gz
        if not os.path.exists(extracted_path):
            extract_gzip(download_path, extracted_path)
        return extracted_path
    
    return download_path


def load_dataset(name: str, force_download: bool = False) -> nx.Graph:
    """
    Load a dataset, downloading if necessary.
    
    Args:
        name: Dataset name
        force_download: Force re-download even if file exists
    
    Returns:
        NetworkX graph object
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {name}")
    print(f"{'='*60}")
    
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[name]
    
    # Download if needed
    file_path = download_dataset(name)
    
    # Load based on special handling
    if dataset_info.get("special_loader") == "as_graphs":
        G = load_as_graphs(DATA_DIR)
    else:
        G = load_edgelist(file_path, directed=dataset_info["directed"])
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Directed: {G.is_directed()}")
    
    if G.is_directed():
        # For directed graphs
        wcc = max(nx.weakly_connected_components(G), key=len)
        print(f"  Largest WCC: {len(wcc):,} nodes ({100*len(wcc)/G.number_of_nodes():.1f}%)")
    else:
        # For undirected graphs
        cc = max(nx.connected_components(G), key=len)
        print(f"  Largest CC: {len(cc):,} nodes ({100*len(cc)/G.number_of_nodes():.1f}%)")
    
    return G


def get_graph_info(G: nx.Graph) -> dict:
    """
    Get comprehensive information about a graph.
    
    Args:
        G: NetworkX graph object
    
    Returns:
        Dictionary with graph statistics
    """
    info = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "density": nx.density(G),
    }
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    info["avg_degree"] = np.mean(degrees)
    info["max_degree"] = max(degrees)
    info["min_degree"] = min(degrees)
    
    # Connected components
    if G.is_directed():
        wccs = list(nx.weakly_connected_components(G))
        sccs = list(nx.strongly_connected_components(G))
        info["num_wcc"] = len(wccs)
        info["num_scc"] = len(sccs)
        info["largest_wcc_size"] = max(len(c) for c in wccs)
        info["largest_scc_size"] = max(len(c) for c in sccs)
    else:
        ccs = list(nx.connected_components(G))
        info["num_cc"] = len(ccs)
        info["largest_cc_size"] = max(len(c) for c in ccs)
    
    return info


# =============================================================================
# Demo function to test data loading
# =============================================================================

def demo_load_datasets():
    """
    Demonstrate loading all datasets.
    """
    print("="*70)
    print("GRAPH SAMPLING PROJECT - DATASET LOADING DEMO")
    print("="*70)
    
    for name in DATASETS.keys():
        try:
            G = load_dataset(name)
            info = get_graph_info(G)
            print(f"\n{name} loaded successfully!")
            print(f"  Average degree: {info['avg_degree']:.2f}")
            print(f"  Density: {info['density']:.6f}")
        except Exception as e:
            print(f"\nError loading {name}: {e}")
        print("-"*70)


if __name__ == "__main__":
    demo_load_datasets()
