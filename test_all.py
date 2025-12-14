import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import networkx as nx


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("Testing Imports")
    
    tests_passed = 0
    tests_failed = 0
    
    modules = [
        ("config", "Configuration"),
        ("src.utils", "Utilities"),
        ("src.data_loader", "Data Loader"),
        ("src.samplers", "Samplers Module"),
        ("src.samplers.base_sampler", "Base Sampler"),
        ("src.samplers.node_samplers", "Node Samplers"),
        ("src.samplers.exploration_samplers", "Exploration Samplers"),
        ("src.samplers.hybrid_samplers", "Hybrid Samplers"),
        ("src.evaluators", "Evaluators Module"),
        ("src.evaluators.static_metrics", "Static Metrics"),
        ("src.evaluators.temporal_metrics", "Temporal Metrics"),
        ("src.evaluators.ks_statistic", "KS Statistic"),
    ]
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print_result(description, True)
            tests_passed += 1
        except Exception as e:
            print_result(description, False, str(e))
            tests_failed += 1
    
    return tests_passed, tests_failed


def test_synthetic_graph_creation():
    """Test synthetic graph creation."""
    print_header("Testing Synthetic Graph Creation")
    
    from src.data_loader import DataLoader
    
    loader = DataLoader("data")
    tests_passed = 0
    tests_failed = 0
    
    graph_types = [
        ('ba', {'n': 500, 'm': 3}),
        ('er', {'n': 500, 'p': 0.01}),
        ('ws', {'n': 500, 'k': 4, 'p': 0.3}),
    ]
    
    for graph_type, params in graph_types:
        try:
            G = loader.create_synthetic_graph(graph_type, **params)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            
            passed = n == params['n'] and m > 0
            print_result(
                f"{graph_type.upper()} Graph", 
                passed,
                f"nodes={n}, edges={m}"
            )
            
            if passed:
                tests_passed += 1
            else:
                tests_failed += 1
                
        except Exception as e:
            print_result(f"{graph_type.upper()} Graph", False, str(e))
            tests_failed += 1
    
    return tests_passed, tests_failed


def test_baseline_samplers():
    """Test baseline sampling algorithms."""
    print_header("Testing Baseline Samplers")
    
    from src.samplers import (
        RandomNodeSampler,
        RandomPageRankNodeSampler,
        RandomDegreeNodeSampler,
        RandomWalkSampler,
        ForestFireSampler
    )
    
    # Create test graph
    G = nx.barabasi_albert_graph(500, 3)
    n_samples = 50
    
    samplers = [
        ("RN", RandomNodeSampler()),
        ("RPN", RandomPageRankNodeSampler()),
        ("RDN", RandomDegreeNodeSampler()),
        ("RW", RandomWalkSampler(restart_prob=0.15)),
        ("FF", ForestFireSampler(forward_prob=0.7)),
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for name, sampler in samplers:
        try:
            start = time.time()
            S = sampler.sample(G, n_samples)
            elapsed = time.time() - start
            
            n = S.number_of_nodes()
            m = S.number_of_edges()
            
            passed = n == n_samples
            print_result(
                f"{name} Sampler",
                passed,
                f"nodes={n}, edges={m}, time={elapsed:.3f}s"
            )
            
            if passed:
                tests_passed += 1
            else:
                tests_failed += 1
                
        except Exception as e:
            print_result(f"{name} Sampler", False, str(e))
            tests_failed += 1
    
    return tests_passed, tests_failed


def test_hybrid_samplers():
    """Test hybrid sampling algorithms."""
    print_header("Testing Hybrid Samplers")
    
    from src.samplers.hybrid_samplers import HybridSampler, get_all_hybrid_samplers
    
    # Create test graph
    G = nx.barabasi_albert_graph(500, 3)
    n_samples = 50
    
    tests_passed = 0
    tests_failed = 0
    
    hybrids = get_all_hybrid_samplers(alpha=0.5)
    
    for name, sampler in hybrids.items():
        try:
            start = time.time()
            S = sampler.sample(G, n_samples)
            elapsed = time.time() - start
            
            n = S.number_of_nodes()
            m = S.number_of_edges()
            
            passed = n == n_samples
            print_result(
                name,
                passed,
                f"nodes={n}, edges={m}, time={elapsed:.3f}s"
            )
            
            if passed:
                tests_passed += 1
            else:
                tests_failed += 1
                
        except Exception as e:
            print_result(name, False, str(e))
            tests_failed += 1
    
    return tests_passed, tests_failed


def test_static_metrics():
    """Test static graph property computation."""
    print_header("Testing Static Metrics")
    
    from src.evaluators.static_metrics import StaticMetrics
    
    # Create test graph
    G = nx.barabasi_albert_graph(300, 3)
    
    calculator = StaticMetrics()
    tests_passed = 0
    tests_failed = 0
    
    metrics = [
        ("In-degree Distribution", "s1_in_degree_distribution"),
        ("Out-degree Distribution", "s2_out_degree_distribution"),
        ("WCC Size Distribution", "s3_wcc_size_distribution"),
        ("SCC Size Distribution", "s4_scc_size_distribution"),
        ("Hop Plot", "s5_hop_plot"),
        ("Singular Vector", "s7_singular_vector_distribution"),
        ("Singular Values", "s8_singular_value_distribution"),
        ("Clustering Coefficient", "s9_clustering_coefficient_distribution"),
    ]
    
    for name, method_name in metrics:
        try:
            method = getattr(calculator, method_name)
            result = method(G)
            
            passed = len(result) > 0
            print_result(
                name,
                passed,
                f"shape={result.shape}, min={result.min():.4f}, max={result.max():.4f}"
            )
            
            if passed:
                tests_passed += 1
            else:
                tests_failed += 1
                
        except Exception as e:
            print_result(name, False, str(e))
            tests_failed += 1
    
    return tests_passed, tests_failed


def test_ks_statistic():
    """Test KS statistic computation."""
    print_header("Testing KS Statistic")
    
    from src.evaluators.ks_statistic import compute_ks_statistic, evaluate_sample
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Same distribution should have low D-statistic
    try:
        dist1 = np.random.normal(0, 1, 1000)
        dist2 = np.random.normal(0, 1, 1000)
        d = compute_ks_statistic(dist1, dist2)
        
        passed = d < 0.1
        print_result(
            "Same Distribution",
            passed,
            f"D={d:.4f} (expected < 0.1)"
        )
        
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
            
    except Exception as e:
        print_result("Same Distribution", False, str(e))
        tests_failed += 1
    
    # Test 2: Different distributions should have higher D-statistic
    try:
        dist1 = np.random.normal(0, 1, 1000)
        dist2 = np.random.normal(2, 1, 1000)
        d = compute_ks_statistic(dist1, dist2)
        
        passed = d > 0.3
        print_result(
            "Different Distributions",
            passed,
            f"D={d:.4f} (expected > 0.3)"
        )
        
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
            
    except Exception as e:
        print_result("Different Distributions", False, str(e))
        tests_failed += 1
    
    # Test 3: Full sample evaluation
    try:
        G = nx.barabasi_albert_graph(300, 3)
        nodes = list(G.nodes())
        sample_nodes = np.random.choice(nodes, size=50, replace=False)
        S = G.subgraph(sample_nodes).copy()
        
        results = evaluate_sample(G, S)
        
        passed = 'AVG' in results and 0 <= results['AVG'] <= 1
        print_result(
            "Sample Evaluation",
            passed,
            f"AVG D-statistic={results['AVG']:.4f}"
        )
        
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
            
    except Exception as e:
        print_result("Sample Evaluation", False, str(e))
        tests_failed += 1
    
    return tests_passed, tests_failed


def test_experiment_runner():
    """Test the experiment runner with synthetic data."""
    print_header("Testing Experiment Runner")
    
    from experiments.run_experiments import run_quick_test
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        print("  Running quick test (this may take a minute)...")
        results = run_quick_test()
        
        passed = len(results) > 0 and 'AVG' in results.columns
        print_result(
            "Quick Test Experiment",
            passed,
            f"rows={len(results)}, methods={results['method'].nunique()}"
        )
        
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
            
    except Exception as e:
        print_result("Quick Test Experiment", False, str(e))
        tests_failed += 1
    
    return tests_passed, tests_failed


def run_quick_tests():
    """Run only quick tests (no experiment runner)."""
    total_passed = 0
    total_failed = 0
    
    # Run tests
    for test_func in [
        test_imports,
        test_synthetic_graph_creation,
        test_baseline_samplers,
        test_hybrid_samplers,
        test_static_metrics,
        test_ks_statistic,
    ]:
        passed, failed = test_func()
        total_passed += passed
        total_failed += failed
    
    return total_passed, total_failed


def run_all_tests():
    """Run all tests including experiment runner."""
    total_passed = 0
    total_failed = 0
    
    # Run quick tests
    passed, failed = run_quick_tests()
    total_passed += passed
    total_failed += failed
    
    # Run experiment runner test
    passed, failed = test_experiment_runner()
    total_passed += passed
    total_failed += failed
    
    return total_passed, total_failed


def main():
    """Main test entry point."""
    print("\n" + "="*70)
    print(" GRAPH SAMPLING PROJECT - TEST SUITE")
    print("="*70)
    
    start_time = time.time()
    
    # Check for quick flag
    quick_mode = '--quick' in sys.argv
    
    if quick_mode:
        print("\n⚡ Running quick tests only...")
        total_passed, total_failed = run_quick_tests()
    else:
        print("\n Running all tests...")
        total_passed, total_failed = run_all_tests()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print(f"\n  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {total_passed + total_failed}")
    print(f"  Time:   {elapsed:.2f} seconds")
    
    if total_failed == 0:
        print("\n  All tests passed!")
    else:
        print(f"\n  {total_failed} test(s) failed")
    
    print("="*70 + "\n")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
