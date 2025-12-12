#!/usr/bin/env python3
"""
Dataset Download Script

Downloads the required datasets from SNAP (Stanford Network Analysis Project).

Usage:
    python download_data.py                  # Download all datasets
    python download_data.py cit-HepTh       # Download specific dataset
    python download_data.py --list          # List available datasets
"""

import sys
import gzip
import shutil
import urllib.request
from pathlib import Path

# -----------------------------------------------------------------------------
# Dataset information
# -----------------------------------------------------------------------------

# Each logical dataset can have multiple files (edges, dates, …)
DATASETS = {
    "cit-HepTh": {
        "description": "ArXiv HEP-TH citation network (edges + dates)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
                "filename": "cit-HepTh.txt.gz",
            },
            {
                "url": "https://snap.stanford.edu/data/cit-HepTh-dates.txt.gz",
                "filename": "cit-HepTh-dates.txt.gz",
            },
        ],
    },
    "cit-HepPh": {
        "description": "ArXiv HEP-PH citation network (edges + dates)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
                "filename": "cit-HepPh.txt.gz",
            },
            {
                "url": "https://snap.stanford.edu/data/cit-HepPh-dates.txt.gz",
                "filename": "cit-HepPh-dates.txt.gz",
            },
        ],
    },
    "epinions": {
        "description": "Epinions who-trusts-whom network (static)",
        "files": [
            {
                "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
                "filename": "soc-Epinions1.txt.gz",
            }
        ],
    },
}

# Data directory
DATA_DIR = Path(__file__).parent / "data"


def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL with progress indication."""
    try:
        print(f"  Downloading from {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(request) as response:
            total_size = response.headers.get("content-length")
            if total_size:
                total_size = int(total_size)
                print(f"  File size: {total_size / 1024 / 1024:.2f} MB")

            DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        print(f"  Error downloading: {e}")
        return False


def extract_gzip(gz_path: Path, output_path: Path) -> bool:
    """Extract a gzip file."""
    try:
        print(f"  Extracting to {output_path.name}")
        with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def download_dataset(name: str) -> bool:
    """Download and extract all files for a single logical dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[name]
    print(f"\n📦 Downloading {name}: {info['description']}")
    print("-" * 50)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for file_info in info["files"]:
        gz_path = DATA_DIR / file_info["filename"]
        if not download_file(file_info["url"], gz_path):
            return False

        # Extract .gz to .txt next to it
        if gz_path.suffix == ".gz":
            txt_path = gz_path.with_suffix("")  # remove .gz
            if not extract_gzip(gz_path, txt_path):
                return False

    print(f"  ✅ Successfully downloaded {name}")
    return True


def list_datasets():
    """Print information about available datasets."""
    print("\n📋 Available Datasets:")
    print("-" * 60)
    for name, info in DATASETS.items():
        print(f"\n  {name}:")
        print(f"    Description: {info['description']}")
        for file_info in info["files"]:
            txt_path = DATA_DIR / file_info["filename"].replace(".gz", "")
            status = "✅ Downloaded" if txt_path.exists() else "❌ Not downloaded"
            print(f"    - {file_info['filename']}: {status}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--list":
            list_datasets()
            return
        if arg in ("--help", "-h"):
            print(__doc__)
            return
        download_dataset(arg)
    else:
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


if __name__ == "__main__":
    main()