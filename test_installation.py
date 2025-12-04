#!/usr/bin/env python3
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import networkx as nx
        print("  ✓ networkx")
    except ImportError as e:
        print(f"  ✗ networkx: {e}")
        return False
    
    try:
        import scipy
        print("  ✓ scipy")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError as e:
        print(f"  ✗ seaborn: {e}")
        return False
    
    try:
        import tqdm
        print("  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    return True


def test_project_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        import config
        print("  ✓ config")
    except ImportError as e:
        print(f"  ✗ config: {e}")
        return False
    
    try:
        from src import data_loader
        print("  ✓ src.data_loader")
    except ImportError as e:
        print(f"  ✗ src.data_loader: {e}")
        return False
    
    try:
        from src import samplers
        print("  ✓ src.samplers")
    except ImportError as e:
        print(f"  ✗ src.samplers: {e}")
        return False
    
    try:
        from src import evaluator
        print("  ✓ src.evaluator")
    except ImportError as e:
        print(f"  ✗ src.evaluator: {e}")
        return False
    
    try:
        from src import experiment
        print("  ✓ src.experiment")
    except ImportError as e:
        print(f"  ✗ src.experiment: {e}")
        return False
    
    try:
        from src import visualizer
        print("  ✓ src.visualizer")
    except ImportError as e:
        print(f"  ✗ src.visualizer: {e}")
        return False
    
    return True


def test_samplers():
    """Test sampling algorithms on a small graph."""
    print("\nTesting sampling algorithms...")
    
    import networkx as nx
    import numpy as np
    from src.samplers import (
        RandomNodeSampler,
        RandomPageRankNodeSampler,
        RandomDegreeNodeSampler,
        RandomWalkSampler,
        RandomJumpSampler,
        ForestFireSampler,
        HybridSampler
    )
    
    # Create test graph
    G = nx.barabasi_albert_graph(500, 3, seed=42)
    G = G.to_directed()
    n_samples = 50
    
    samplers_to_test = [
        ("RandomNodeSampler", RandomNodeSampler(G, random_state=42)),
        ("RandomPageRankNodeSampler", RandomPageRankNodeSampler(G, random_state=42)),
        ("RandomDegreeNodeSampler", RandomDegreeNodeSampler(G, random_state=42)),
        ("RandomWalkSampler", RandomWalkSampler(G, random_state=42)),
        ("RandomJumpSampler", RandomJumpSampler(G, random_state=42)),
        ("ForestFireSampler", ForestFireSampler(G, random_state=42)),
        ("HybridSampler (RN-RW)", HybridSampler(G, random_state=42, 
                                                 node_method="RN", 
                                                 explore_method="RW",
                                                 alpha=0.5)),
    ]
    
    all_passed = True
    for name, sampler in samplers_to_test:
        try:
            S = sampler.sample(n_samples)
            if S.number_of_nodes() > 0:
                print(f"  ✓ {name}: {S.number_of_nodes()} nodes, {S.number_of_edges()} edges")
            else:
                print(f"  ✗ {name}: Empty sample")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_passed = False
    
    return all_passed


def test_evaluator():
    """Test evaluation metrics."""
    print("\nTesting evaluator...")
    
    import networkx as nx
    import numpy as np
    from src.evaluator import GraphEvaluator, compute_ks_statistic
    
    # Create test graphs
    G = nx.barabasi_albert_graph(200, 3, seed=42)
    nodes = list(G.nodes())
    sampled_nodes = np.random.choice(nodes, size=50, replace=False)
    S = G.subgraph(sampled_nodes).copy()
    
    try:
        evaluator = GraphEvaluator(G)
        print("  ✓ GraphEvaluator created")
    except Exception as e:
        print(f"  ✗ GraphEvaluator creation: {e}")
        return False
    
    try:
        results = evaluator.evaluate_all(S)
        print(f"  ✓ evaluate_all: AVG D-stat = {results['AVG']:.4f}")
    except Exception as e:
        print(f"  ✗ evaluate_all: {e}")
        return False
    
    try:
        # Test KS statistic
        d1 = np.random.normal(0, 1, 100)
        d2 = np.random.normal(0.5, 1, 100)
        ks = compute_ks_statistic(d1, d2)
        print(f"  ✓ compute_ks_statistic: D = {ks:.4f}")
    except Exception as e:
        print(f"  ✗ compute_ks_statistic: {e}")
        return False
    
    return True


def test_visualization():
    """Test visualization (without displaying)."""
    print("\nTesting visualization...")
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Create synthetic results
    data = []
    for method in ['RN', 'RW', 'FF', 'HYB-RN-RW(α=0.5)']:
        for ratio in [0.1, 0.15, 0.2]:
            method_type = 'hybrid' if 'HYB' in method else 'baseline'
            data.append({
                'dataset': 'test',
                'ratio': ratio,
                'method': method,
                'method_type': method_type,
                'alpha': 0.5 if 'HYB' in method else None,
                'in_degree': 0.3 + np.random.uniform(-0.1, 0.1),
                'out_degree': 0.3 + np.random.uniform(-0.1, 0.1),
                'wcc': 0.5 + np.random.uniform(-0.1, 0.1),
                'scc': 0.4 + np.random.uniform(-0.1, 0.1),
                'hop_plot': 0.3 + np.random.uniform(-0.1, 0.1),
                'singular_vec': 0.2 + np.random.uniform(-0.1, 0.1),
                'singular_val': 0.2 + np.random.uniform(-0.1, 0.1),
                'clustering': 0.3 + np.random.uniform(-0.1, 0.1),
                'AVG': 0.3 + np.random.uniform(-0.1, 0.1),
            })
    
    results_df = pd.DataFrame(data)
    
    try:
        from src.visualizer import plot_property_heatmap
        fig = plot_property_heatmap(results_df, ratio=0.15)
        plt.close(fig)
        print("  ✓ plot_property_heatmap")
    except Exception as e:
        print(f"  ✗ plot_property_heatmap: {e}")
        return False
    
    try:
        from src.visualizer import plot_method_comparison_bars
        fig = plot_method_comparison_bars(results_df, ratio=0.15, top_n=4)
        plt.close(fig)
        print("  ✓ plot_method_comparison_bars")
    except Exception as e:
        print(f"  ✗ plot_method_comparison_bars: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("GRAPH SAMPLING PROJECT - INSTALLATION TEST")
    print("="*70)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠️  Some dependencies are missing. Run: pip install -r requirements.txt")
        return
    
    # Test project modules
    if not test_project_modules():
        all_passed = False
        print("\n⚠️  Some project modules failed to import.")
        return
    
    # Test samplers
    if not test_samplers():
        all_passed = False
    
    # Test evaluator
    if not test_evaluator():
        all_passed = False
    
    # Test visualization
    if not test_visualization():
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe project is ready to use!")
        print("\nNext steps:")
        print("  1. Run quick test: python main.py --quick")
        print("  2. Run full experiment: python main.py --full")
        print("  3. See help: python main.py --help")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the issues above before running experiments.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
