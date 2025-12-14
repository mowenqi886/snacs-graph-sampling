#!/usr/bin/env python3

import sys
import os
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASETS = {
    # =========================================================================
    # TEMPORAL DATASETS  - 用于 Back-in-Time
    # =========================================================================
    "cit-HepTh": {
        "description": "ArXiv HEP-TH citation network (edges + dates)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
                "filename": "cit-HepTh.txt.gz",
                "type": "edges"
            },
            {
                "url": "https://snap.stanford.edu/data/cit-HepTh-dates.txt.gz",
                "filename": "cit-HepTh-dates.txt.gz",
                "type": "dates"
            },
        ],
    },
    "cit-HepPh": {
        "description": "ArXiv HEP-PH citation network (edges + dates)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
                "filename": "cit-HepPh.txt.gz",
                "type": "edges"
            },
            {
                "url": "https://snap.stanford.edu/data/cit-HepPh-dates.txt.gz",
                "filename": "cit-HepPh-dates.txt.gz",
                "type": "dates"
            },
        ],
    },
    # =========================================================================
    # STATIC DATASETS - 只用于 Scale-Down
    # =========================================================================
    "soc-Epinions1": {
        "description": "Epinions who-trusts-whom network (static)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
                "filename": "soc-Epinions1.txt.gz",
                "type": "edges"
            }
        ],
    },
    "wiki-Vote": {
        "description": "Wikipedia adminship voting network (static)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
                "filename": "wiki-Vote.txt.gz",
                "type": "edges"
            }
        ],
    },
    "p2p-Gnutella31": {
        "description": "Gnutella P2P file sharing network (static)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz",
                "filename": "p2p-Gnutella31.txt.gz",
                "type": "edges"
            }
        ],
    },
}

# Data directory
DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, filepath: Path) -> bool:
    """
    Download a file from URL with progress indication.
    
    Args:
        url: URL to download from
        filepath: Local path to save file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Downloading from {url}")
        
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        with urllib.request.urlopen(request) as response:
            total_size = response.headers.get("content-length")
            
            if total_size:
                total_size = int(total_size)
                print(f"  File size: {total_size / 1024 / 1024:.2f} MB")
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "wb") as f:
                downloaded = 0
                block_size = 8192
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if total_size:
                        percent = downloaded * 100 / total_size
                        print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
            
            print()  # newline
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return False


def extract_gzip(gz_path: Path, output_path: Path) -> bool:
    """
    Extract a gzip file.
    
    Args:
        gz_path: Path to .gz file
        output_path: Path for extracted file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Extracting to {output_path.name}")
        
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error extracting: {e}")
        return False


def download_dataset(name: str) -> bool:
    """
    Download and extract all files for a single dataset.
    
    Args:
        name: Dataset name
    
    Returns:
        True if all files downloaded successfully
    """
    if name not in DATASETS:
        print(f"✗ Unknown dataset: {name}")
        print(f"  Available: {', '.join(DATASETS.keys())}")
        return False
    
    info = DATASETS[name]
    print(f"\n{'='*60}")
    print(f" Downloading {name}")
    print(f"   {info['description']}")
    print(f"{'='*60}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    for file_info in info["files"]:
        gz_path = DATA_DIR / file_info["filename"]
        
        # Download .gz file
        if not gz_path.exists():
            if not download_file(file_info["url"], gz_path):
                success = False
                continue
        else:
            print(f"  Already exists: {gz_path.name}")
        
        # Extract .gz to .txt
        if gz_path.suffix == ".gz":
            txt_path = gz_path.with_suffix("")
            
            if not txt_path.exists():
                if not extract_gzip(gz_path, txt_path):
                    success = False
                    continue
            else:
                print(f"  Already extracted: {txt_path.name}")
    
    if success:
        print(f"  ✓ Successfully downloaded {name}")
    else:
        print(f"  ✗ Some files failed for {name}")
    
    return success


def download_all() -> Tuple[int, int]:
    """
    Download all datasets.
    
    Returns:
        Tuple of (success_count, failed_count)
    """
    print("\n" + "="*60)
    print(" DOWNLOADING ALL DATASETS")
    print("="*60)
    
    success = 0
    failed = 0
    
    for name in DATASETS:
        if download_dataset(name):
            success += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  ✓ Successful: {success}")
    print(f"  ✗ Failed: {failed}")
    
    return success, failed


def list_datasets() -> None:
    """Print information about available datasets."""
    print("\n" + "="*60)
    print(" AVAILABLE DATASETS")
    print("="*60)
    
    for name, info in DATASETS.items():
        print(f"\n  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Files:")
        
        for file_info in info["files"]:
            txt_path = DATA_DIR / file_info["filename"].replace(".gz", "")
            status = "✓ Downloaded" if txt_path.exists() else "✗ Not downloaded"
            print(f"      - {file_info['filename']}: {status}")


def check_datasets() -> Dict[str, bool]:
    """
    Check which datasets are downloaded.
    
    Returns:
        Dictionary {dataset_name: is_downloaded}
    """
    print("\n" + "="*60)
    print(" CHECKING DOWNLOADED DATASETS")
    print("="*60)
    
    status = {}
    
    for name, info in DATASETS.items():
        all_exist = True
        
        for file_info in info["files"]:
            txt_path = DATA_DIR / file_info["filename"].replace(".gz", "")
            if not txt_path.exists():
                all_exist = False
                break
        
        status[name] = all_exist
        
        symbol = "✓" if all_exist else "✗"
        print(f"  {symbol} {name}")
    
    return status


# =============================================================================
# Statistics Function
# =============================================================================

def show_dataset_stats(name: str) -> None:
    """
    Show statistics for a downloaded dataset.
    
    Args:
        name: Dataset name
    """
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        return
    
    info = DATASETS[name]
    edges_file = None
    
    for file_info in info["files"]:
        if file_info["type"] == "edges":
            edges_file = DATA_DIR / file_info["filename"].replace(".gz", "")
            break
    
    if not edges_file or not edges_file.exists():
        print(f"Dataset {name} not downloaded")
        return
    
    print(f"\n Statistics for {name}:")
    
    # Count nodes and edges
    nodes = set()
    n_edges = 0
    
    with open(edges_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    nodes.add(u)
                    nodes.add(v)
                    n_edges += 1
                except ValueError:
                    continue
    
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Avg degree: {2 * n_edges / len(nodes):.2f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--list":
            list_datasets()
            return
        
        if arg == "--check":
            check_datasets()
            return
        
        if arg == "--stats":
            if len(sys.argv) > 2:
                show_dataset_stats(sys.argv[2])
            else:
                for name in DATASETS:
                    show_dataset_stats(name)
            return
        
        if arg in ("--help", "-h"):
            print(__doc__)
            return
        
        if arg in DATASETS:
            download_dataset(arg)
            return
        
        print(f"Unknown argument: {arg}")
        print(f"Use --help for usage information")
        return
    
    # Default: download all
    download_all()


if __name__ == "__main__":
    from typing import Tuple
    main()
