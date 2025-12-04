#!/usr/bin/env python3
"""
Dataset Download Script

Downloads the required datasets from SNAP (Stanford Network Analysis Project).

Usage:
    python download_data.py              # Download all datasets
    python download_data.py hep-th       # Download specific dataset
    python download_data.py --list       # List available datasets
"""

import os
import sys
import gzip
import shutil
import urllib.request
from pathlib import Path


# Dataset information
DATASETS = {
    'hep-th': {
        'url': 'https://snap.stanford.edu/data/cit-HepTh.txt.gz',
        'filename': 'cit-HepTh.txt.gz',
        'description': 'High Energy Physics Theory citation network'
    },
    'hep-ph': {
        'url': 'https://snap.stanford.edu/data/cit-HepPh.txt.gz',
        'filename': 'cit-HepPh.txt.gz',
        'description': 'High Energy Physics Phenomenology citation network'
    },
    'epinions': {
        'url': 'https://snap.stanford.edu/data/soc-Epinions1.txt.gz',
        'filename': 'soc-Epinions1.txt.gz',
        'description': 'Epinions who-trusts-whom network'
    },
    'astro': {
        'url': 'https://snap.stanford.edu/data/ca-AstroPh.txt.gz',
        'filename': 'ca-AstroPh.txt.gz',
        'description': 'Astro Physics collaboration network'
    }
}

# Data directory
DATA_DIR = Path(__file__).parent / 'data'


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
        
        # Create a request with headers
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        # Download with progress
        with urllib.request.urlopen(request) as response:
            total_size = response.headers.get('content-length')
            
            if total_size:
                total_size = int(total_size)
                print(f"  File size: {total_size / 1024 / 1024:.2f} MB")
            
            with open(filepath, 'wb') as f:
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
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
                
                print()  # New line after progress
        
        return True
        
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def extract_gzip(gz_path: Path, output_path: Path) -> bool:
    """
    Extract a gzip file.
    
    Args:
        gz_path: Path to .gz file
        output_path: Path for extracted file
        
    Returns:
        True if successful
    """
    try:
        print(f"  Extracting to {output_path.name}")
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return True
        
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def download_dataset(name: str) -> bool:
    """
    Download and extract a single dataset.
    
    Args:
        name: Dataset name
        
    Returns:
        True if successful
    """
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False
    
    info = DATASETS[name]
    
    print(f"\n📦 Downloading {name}: {info['description']}")
    print("-" * 50)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download
    gz_path = DATA_DIR / info['filename']
    
    if not download_file(info['url'], gz_path):
        return False
    
    # Extract
    txt_path = DATA_DIR / info['filename'].replace('.gz', '')
    
    if not extract_gzip(gz_path, txt_path):
        return False
    
    # Optionally remove .gz file
    # gz_path.unlink()
    
    print(f"  ✅ Successfully downloaded {name}")
    return True


def list_datasets():
    """Print information about available datasets."""
    print("\n📋 Available Datasets:")
    print("-" * 60)
    
    for name, info in DATASETS.items():
        print(f"\n  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    URL: {info['url']}")
        
        # Check if already downloaded
        txt_path = DATA_DIR / info['filename'].replace('.gz', '')
        if txt_path.exists():
            size_mb = txt_path.stat().st_size / 1024 / 1024
            print(f"    Status: ✅ Downloaded ({size_mb:.2f} MB)")
        else:
            print(f"    Status: ❌ Not downloaded")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == '--list':
            list_datasets()
            return
        
        if arg == '--help' or arg == '-h':
            print(__doc__)
            return
        
        # Download specific dataset
        download_dataset(arg)
    else:
        # Download all datasets
        print("📥 Downloading all datasets...")
        print("=" * 60)
        
        success = 0
        failed = 0
        
        for name in DATASETS:
            if download_dataset(name):
                success += 1
            else:
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"✅ Downloaded: {success}")
        print(f"❌ Failed: {failed}")
        
        if failed > 0:
            print("\nNote: Some datasets may have download restrictions.")
            print("You can manually download from https://snap.stanford.edu/data/")


if __name__ == "__main__":
    main()
