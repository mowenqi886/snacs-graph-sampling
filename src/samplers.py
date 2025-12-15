
import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Tuple, Set, Union
from abc import ABC, abstractmethod
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_WALK_RESTART_PROB,
    RANDOM_WALK_MAX_STEPS_MULTIPLIER,
    FF_FORWARD_PROB_SCALEDOWN,
    FF_FORWARD_PROB_BACKTIME,
    FF_BACKWARD_PROB
)


# =============================================================================
# Base Sampler Class
# =============================================================================

class GraphSampler(ABC):
    """
    Abstract base class for graph samplers.
    
    All samplers inherit from this class and implement the sample() method.
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None):
        """
        Initialize sampler with a graph.
        
        Args:
            G: NetworkX graph to sample from
            random_state: Random seed for reproducibility
        """
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.n_edges = G.number_of_edges()
        self.nodes = list(G.nodes())
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Lazy-loaded properties
        self._pagerank = None
        self._degrees = None
    
    @property
    def pagerank(self) -> Dict:
        """Lazy-load PageRank values."""
        if self._pagerank is None:
            self._pagerank = nx.pagerank(self.G, alpha=0.85)
        return self._pagerank
    
    @property
    def degrees(self) -> Dict:
        """Lazy-load degree values."""
        if self._degrees is None:
            if self.G.is_directed():
                # Use total degree for directed graphs
                self._degrees = {n: self.G.in_degree(n) + self.G.out_degree(n) 
                                for n in self.G.nodes()}
            else:
                self._degrees = dict(self.G.degree())
        return self._degrees
    
    @abstractmethod
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Sample nodes from the graph and return induced subgraph.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        pass
    
    def _get_induced_subgraph(self, sampled_nodes: Set[int]) -> nx.Graph:
        """
        Get the induced subgraph on sampled nodes.
        
        Args:
            sampled_nodes: Set of node IDs to include
        
        Returns:
            Induced subgraph
        """
        return self.G.subgraph(sampled_nodes).copy()


# =============================================================================
# Node Selection Samplers (RN, RPN, RDN)
# =============================================================================

class RandomNodeSampler(GraphSampler):
    """
    RN: Random Node Sampling
    
    Select n nodes uniformly at random.
    Simple but may miss important structural features.
    """
    
    def sample(self, n_samples: int) -> nx.Graph:
        """Sample n nodes uniformly at random."""
        n_samples = min(n_samples, self.n_nodes)
        sampled_nodes = set(np.random.choice(self.nodes, size=n_samples, replace=False))
        return self._get_induced_subgraph(sampled_nodes)


class RandomPageRankNodeSampler(GraphSampler):
    """
    RPN: Random PageRank Node Sampling
    
    Select nodes with probability proportional to PageRank.
    Favors important/central nodes.
    """
    
    def sample(self, n_samples: int) -> nx.Graph:
        """Sample n nodes weighted by PageRank."""
        n_samples = min(n_samples, self.n_nodes)
        
        # Get PageRank probabilities
        pr_values = np.array([self.pagerank[n] for n in self.nodes])
        pr_probs = pr_values / pr_values.sum()
        
        # Sample without replacement
        sampled_indices = np.random.choice(
            len(self.nodes), size=n_samples, replace=False, p=pr_probs
        )
        sampled_nodes = set(self.nodes[i] for i in sampled_indices)
        
        return self._get_induced_subgraph(sampled_nodes)


class RandomDegreeNodeSampler(GraphSampler):
    """
    RDN: Random Degree Node Sampling
    
    Select nodes with probability proportional to degree.
    Favors high-degree (hub) nodes.
    """
    
    def sample(self, n_samples: int) -> nx.Graph:
        """Sample n nodes weighted by degree."""
        n_samples = min(n_samples, self.n_nodes)
        
        # Get degree probabilities
        degree_values = np.array([max(1, self.degrees[n]) for n in self.nodes])
        degree_probs = degree_values / degree_values.sum()
        
        # Sample without replacement
        sampled_indices = np.random.choice(
            len(self.nodes), size=n_samples, replace=False, p=degree_probs
        )
        sampled_nodes = set(self.nodes[i] for i in sampled_indices)
        
        return self._get_induced_subgraph(sampled_nodes)


# =============================================================================
# Exploration Samplers (RW, RJ, FF)
# =============================================================================

class RandomWalkSampler(GraphSampler):
    """
    RW: Random Walk Sampling
    
    Perform random walk starting from a random node.
    With probability c, restart from the initial node.
    
    Parameters:
        restart_prob: Probability of restarting (default: 0.15)
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 restart_prob: float = RANDOM_WALK_RESTART_PROB):
        super().__init__(G, random_state)
        self.restart_prob = restart_prob
    
    def sample(self, n_samples: int) -> nx.Graph:
        """Sample n nodes via random walk."""
        n_samples = min(n_samples, self.n_nodes)
        
        sampled_nodes = set()
        max_steps = n_samples * RANDOM_WALK_MAX_STEPS_MULTIPLIER
        
        # Start from random node
        start_node = np.random.choice(self.nodes)
        current_node = start_node
        sampled_nodes.add(current_node)
        
        steps = 0
        while len(sampled_nodes) < n_samples and steps < max_steps:
            steps += 1
            
            # With probability restart_prob, restart from start node
            if np.random.random() < self.restart_prob:
                current_node = start_node
            else:
                # Get neighbors
                if self.G.is_directed():
                    neighbors = list(self.G.successors(current_node))
                    if not neighbors:
                        neighbors = list(self.G.predecessors(current_node))
                else:
                    neighbors = list(self.G.neighbors(current_node))
                
                if neighbors:
                    current_node = np.random.choice(neighbors)
                else:
                    # Stuck - restart from a new random node
                    current_node = np.random.choice(self.nodes)
            
            sampled_nodes.add(current_node)
        
        return self._get_induced_subgraph(sampled_nodes)


class RandomJumpSampler(GraphSampler):
    """
    RJ: Random Jump Sampling
    
    Similar to Random Walk, but with probability c, jump to 
    ANY random node (not just the start node).
    
    Parameters:
        jump_prob: Probability of jumping to random node (default: 0.15)
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 jump_prob: float = RANDOM_WALK_RESTART_PROB):
        super().__init__(G, random_state)
        self.jump_prob = jump_prob
    
    def sample(self, n_samples: int) -> nx.Graph:
        """Sample n nodes via random walk with random jumps."""
        n_samples = min(n_samples, self.n_nodes)
        
        sampled_nodes = set()
        max_steps = n_samples * RANDOM_WALK_MAX_STEPS_MULTIPLIER
        
        # Start from random node
        current_node = np.random.choice(self.nodes)
        sampled_nodes.add(current_node)
        
        steps = 0
        while len(sampled_nodes) < n_samples and steps < max_steps:
            steps += 1
            
            # With probability jump_prob, jump to ANY random node
            if np.random.random() < self.jump_prob:
                current_node = np.random.choice(self.nodes)
            else:
                # Get neighbors
                if self.G.is_directed():
                    neighbors = list(self.G.successors(current_node))
                    if not neighbors:
                        neighbors = list(self.G.predecessors(current_node))
                else:
                    neighbors = list(self.G.neighbors(current_node))
                
                if neighbors:
                    current_node = np.random.choice(neighbors)
                else:
                    current_node = np.random.choice(self.nodes)
            
            sampled_nodes.add(current_node)
        
        return self._get_induced_subgraph(sampled_nodes)


class ForestFireSampler(GraphSampler):
    """
    FF: Forest Fire Sampling
    
    BFS-like exploration where each node "burns" a geometric number 
    of neighbors (with mean p_f/(1-p_f) for forward, p_b/(1-p_b) for backward).
    
    Parameters:
        forward_prob: Forward burning probability p_f
        backward_prob: Backward burning probability p_b (for directed graphs)
    
    From the paper:
    - Scale-down: Use p_f = 0.7 (larger fires)
    - Back-in-time: Use p_f = 0.2 (smaller fires)
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 forward_prob: float = FF_FORWARD_PROB_SCALEDOWN,
                 backward_prob: float = FF_BACKWARD_PROB):
        super().__init__(G, random_state)
        self.forward_prob = forward_prob
        self.backward_prob = backward_prob
    
    def _geometric_sample(self, p: float) -> int:
        """
        Sample from geometric distribution with parameter p.
        Returns number of nodes to burn (mean = p/(1-p)).
        """
        if p <= 0:
            return 0
        if p >= 1:
            return 1000  # Large number
        
        # Geometric distribution: number of failures before first success
        return np.random.geometric(1 - p)
    
    def sample(self, n_samples: int) -> nx.Graph:
        """Sample n nodes via forest fire exploration."""
        n_samples = min(n_samples, self.n_nodes)
        
        sampled_nodes = set()
        burning_queue = []
        
        # Start from random node
        start_node = np.random.choice(self.nodes)
        sampled_nodes.add(start_node)
        burning_queue.append(start_node)
        
        while len(sampled_nodes) < n_samples and burning_queue:
            # Get next node to process
            current = burning_queue.pop(0)
            
            # Get neighbors to potentially burn
            if self.G.is_directed():
                out_neighbors = list(self.G.successors(current))
                in_neighbors = list(self.G.predecessors(current))
            else:
                out_neighbors = list(self.G.neighbors(current))
                in_neighbors = []
            
            # Filter out already sampled nodes
            out_neighbors = [n for n in out_neighbors if n not in sampled_nodes]
            in_neighbors = [n for n in in_neighbors if n not in sampled_nodes]
            
            # Determine how many to burn (geometric distribution)
            n_forward = min(self._geometric_sample(self.forward_prob), len(out_neighbors))
            n_backward = min(self._geometric_sample(self.backward_prob), len(in_neighbors))
            
            # Randomly select neighbors to burn
            if out_neighbors and n_forward > 0:
                burned_forward = np.random.choice(
                    out_neighbors, 
                    size=min(n_forward, len(out_neighbors)), 
                    replace=False
                )
                for node in burned_forward:
                    if len(sampled_nodes) >= n_samples:
                        break
                    sampled_nodes.add(node)
                    burning_queue.append(node)
            
            if in_neighbors and n_backward > 0:
                burned_backward = np.random.choice(
                    in_neighbors,
                    size=min(n_backward, len(in_neighbors)),
                    replace=False
                )
                for node in burned_backward:
                    if len(sampled_nodes) >= n_samples:
                        break
                    sampled_nodes.add(node)
                    burning_queue.append(node)
            
            # If queue is empty but we need more nodes, restart from new random node
            if not burning_queue and len(sampled_nodes) < n_samples:
                remaining = [n for n in self.nodes if n not in sampled_nodes]
                if remaining:
                    new_start = np.random.choice(remaining)
                    sampled_nodes.add(new_start)
                    burning_queue.append(new_start)
        
        return self._get_induced_subgraph(sampled_nodes)


# =============================================================================
# Hybrid Sampler
# =============================================================================

class HybridSampler(GraphSampler):
    """
    Hybrid Sampling: Combines node selection with exploration.
    
    Our contribution: Systematically evaluate combinations of:
    - Node selection: RN, RPN, RDN
    - Exploration: RW, FF
    
    Alpha parameter controls the mix:
    - Sample (α * n) nodes using node selection method
    - Sample ((1-α) * n) nodes using exploration method
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 node_method: str = "RN",
                 explore_method: str = "RW",
                 alpha: float = 0.5,
                 **kwargs):
        """
        Initialize hybrid sampler.
        
        Args:
            G: Graph to sample from
            random_state: Random seed
            node_method: Node selection method ("RN", "RPN", "RDN")
            explore_method: Exploration method ("RW", "FF")
            alpha: Fraction from node selection (0 to 1)
            **kwargs: Additional arguments for exploration method
        """
        super().__init__(G, random_state)
        self.node_method = node_method
        self.explore_method = explore_method
        self.alpha = alpha
        self.kwargs = kwargs
        
        # Create component samplers
        self._node_sampler = self._create_node_sampler(node_method, random_state)
        self._explore_sampler = self._create_explore_sampler(explore_method, random_state, **kwargs)
    
    def _create_node_sampler(self, method: str, random_state: int) -> GraphSampler:
        """Create node selection sampler."""
        if method == "RN":
            return RandomNodeSampler(self.G, random_state)
        elif method == "RPN":
            return RandomPageRankNodeSampler(self.G, random_state)
        elif method == "RDN":
            return RandomDegreeNodeSampler(self.G, random_state)
        else:
            raise ValueError(f"Unknown node selection method: {method}")
    
    def _create_explore_sampler(self, method: str, random_state: int, **kwargs) -> GraphSampler:
        """Create exploration sampler."""
        if method == "RW":
            restart_prob = kwargs.get('restart_prob', RANDOM_WALK_RESTART_PROB)
            return RandomWalkSampler(self.G, random_state, restart_prob=restart_prob)
        elif method == "FF":
            forward_prob = kwargs.get('forward_prob', FF_FORWARD_PROB_SCALEDOWN)
            backward_prob = kwargs.get('backward_prob', FF_BACKWARD_PROB)
            return ForestFireSampler(self.G, random_state, 
                                     forward_prob=forward_prob, 
                                     backward_prob=backward_prob)
        else:
            raise ValueError(f"Unknown exploration method: {method}")
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Sample using hybrid strategy.
        
        1. Select α*n nodes using node selection method
        2. Select (1-α)*n nodes using exploration method
        3. Combine and return induced subgraph
        """
        n_samples = min(n_samples, self.n_nodes)
        
        # Calculate split
        n_node_selection = int(n_samples * self.alpha)
        n_exploration = n_samples - n_node_selection
        
        sampled_nodes = set()
        
        # Phase 1: Node selection
        if n_node_selection > 0:
            node_sample = self._node_sampler.sample(n_node_selection)
            sampled_nodes.update(node_sample.nodes())
        
        # Phase 2: Exploration (starting from already sampled nodes if possible)
        if n_exploration > 0:
            explore_sample = self._explore_sampler.sample(n_exploration)
            sampled_nodes.update(explore_sample.nodes())
        
        # Trim to exact size if needed
        if len(sampled_nodes) > n_samples:
            sampled_nodes = set(list(sampled_nodes)[:n_samples])
        
        return self._get_induced_subgraph(sampled_nodes)


# =============================================================================
# Factory Functions
# =============================================================================

def get_sampler(G: nx.Graph, method: str, random_state: Optional[int] = None,
                **kwargs) -> GraphSampler:
    """
    Factory function to create a sampler by name.
    
    Args:
        G: Graph to sample from
        method: Method name (e.g., "RN", "RW", "HYB-RN-RW")
        random_state: Random seed
        **kwargs: Additional arguments for the sampler
    
    Returns:
        GraphSampler instance
    """
    method = method.upper()
    
    # Baseline methods
    if method == "RN":
        return RandomNodeSampler(G, random_state)
    elif method == "RPN":
        return RandomPageRankNodeSampler(G, random_state)
    elif method == "RDN":
        return RandomDegreeNodeSampler(G, random_state)
    elif method == "RW":
        return RandomWalkSampler(G, random_state, **kwargs)
    elif method == "RJ":
        return RandomJumpSampler(G, random_state, **kwargs)
    elif method == "FF":
        return ForestFireSampler(G, random_state, **kwargs)
    
    # Hybrid methods: HYB-X-Y or HYB-X-Y(α=0.5)
    elif method.startswith("HYB-"):
        parts = method.replace("HYB-", "").split("-")
        if len(parts) >= 2:
            node_method = parts[0]
            explore_method = parts[1].split("(")[0]  
            alpha = kwargs.pop('alpha', 0.5)  
            return HybridSampler(G, random_state, 
                                node_method=node_method,
                                explore_method=explore_method,
                                alpha=alpha, **kwargs)
    
    raise ValueError(f"Unknown sampling method: {method}")


def sample_graph(G: nx.Graph, method: str, n_samples: int,
                 random_state: Optional[int] = None, **kwargs) -> nx.Graph:
    """
    Convenience function to sample a graph.
    
    Args:
        G: Graph to sample from
        method: Sampling method name
        n_samples: Number of nodes to sample
        random_state: Random seed
        **kwargs: Additional arguments
    
    Returns:
        Sampled subgraph
    """
    sampler = get_sampler(G, method, random_state, **kwargs)
    return sampler.sample(n_samples)


def get_all_hybrid_samplers(G: nx.Graph = None, alpha: float = 0.5,
                             random_state: Optional[int] = None,
                             **kwargs) -> Dict[str, HybridSampler]:
    """
    Get dictionary of all hybrid sampler combinations.
    
    Args:
        G: Graph (can be None for just getting names)
        alpha: Alpha parameter
        random_state: Random seed
        **kwargs: Additional arguments
    
    Returns:
        Dictionary {name: sampler} if G provided, else {name: None}
    """
    node_methods = ["RN", "RPN", "RDN"]
    explore_methods = ["RW", "FF"]
    
    samplers = {}
    for node_m in node_methods:
        for explore_m in explore_methods:
            name = f"HYB-{node_m}-{explore_m}"
            if G is not None:
                samplers[name] = HybridSampler(
                    G, random_state, 
                    node_method=node_m, 
                    explore_method=explore_m,
                    alpha=alpha, **kwargs
                )
            else:
                samplers[name] = None
    
    return samplers


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GRAPH SAMPLING MODULE DEMO")
    print("="*70)
    
    # Create test graph
    print("\nCreating test graph (Barabási-Albert, n=500, m=3)...")
    G = nx.barabasi_albert_graph(500, 3, seed=42)
    G = G.to_directed()  # Make it directed for full testing
    
    n_samples = 50
    print(f"\nSampling {n_samples} nodes with each method...\n")
    
    # Test all baseline methods
    baseline_methods = ["RN", "RPN", "RDN", "RW", "RJ", "FF"]
    
    print("BASELINE METHODS:")
    print("-" * 50)
    for method in baseline_methods:
        S = sample_graph(G, method, n_samples, random_state=42)
        print(f"  {method:5s}: {S.number_of_nodes():4d} nodes, {S.number_of_edges():5d} edges")
    
    # Test hybrid methods
    print("\nHYBRID METHODS (α=0.5):")
    print("-" * 50)
    hybrid_samplers = get_all_hybrid_samplers(G, alpha=0.5, random_state=42)
    for name, sampler in hybrid_samplers.items():
        S = sampler.sample(n_samples)
        print(f"  {name:15s}: {S.number_of_nodes():4d} nodes, {S.number_of_edges():5d} edges")
    
    print("\n✓ Sampling module demo completed!")
