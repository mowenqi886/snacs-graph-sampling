import numpy as np
import networkx as nx
from typing import Set, List, Optional, Tuple, Union
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_WALK_RESTART_PROB,
    RANDOM_WALK_MAX_STEPS_MULTIPLIER,
    FF_FORWARD_PROB_SCALEDOWN,
    FF_BACKWARD_PROB,
)


# =============================================================================
# Base Sampling Class
# =============================================================================

class GraphSampler:
    """
    Base class for graph sampling algorithms.
    
    All sampling methods return an induced subgraph containing the sampled nodes
    and all edges between them from the original graph.
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None):
        """
        Initialize the sampler.
        
        Args:
            G: Original graph to sample from
            random_state: Random seed for reproducibility
        """
        self.G = G
        self.nodes = list(G.nodes())
        self.n_nodes = G.number_of_nodes()
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Precompute useful properties (lazily)
        self._pagerank = None
        self._degrees = None
    
    @property
    def pagerank(self) -> dict:
        """Lazily compute PageRank scores."""
        if self._pagerank is None:
            self._pagerank = nx.pagerank(self.G)
        return self._pagerank
    
    @property
    def degrees(self) -> dict:
        """Lazily compute node degrees."""
        if self._degrees is None:
            self._degrees = dict(self.G.degree())
        return self._degrees
    
    def get_neighbors(self, node: int) -> List[int]:
        """
        Get neighbors of a node (handles both directed and undirected graphs).
        
        Args:
            node: Node to get neighbors for
        
        Returns:
            List of neighbor nodes
        """
        if self.G.is_directed():
            # For directed graphs, consider out-neighbors for exploration
            return list(self.G.successors(node))
        return list(self.G.neighbors(node))
    
    def get_all_neighbors(self, node: int) -> List[int]:
        """
        Get all neighbors (both in and out for directed graphs).
        
        Args:
            node: Node to get neighbors for
        
        Returns:
            List of all neighbor nodes
        """
        if self.G.is_directed():
            out_neighbors = set(self.G.successors(node))
            in_neighbors = set(self.G.predecessors(node))
            return list(out_neighbors | in_neighbors)
        return list(self.G.neighbors(node))
    
    def create_subgraph(self, nodes: Set[int]) -> nx.Graph:
        """
        Create induced subgraph from selected nodes.
        
        Args:
            nodes: Set of nodes to include
        
        Returns:
            Induced subgraph
        """
        return self.G.subgraph(nodes).copy()
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Sample nodes from the graph. To be implemented by subclasses.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        raise NotImplementedError("Subclasses must implement sample()")


# =============================================================================
# Node Selection Methods (Baseline)
# =============================================================================

class RandomNodeSampler(GraphSampler):
    """
    Random Node (RN) Sampling.
    
    Select nodes uniformly at random. This is the simplest baseline.
    """
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform random node sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        sampled_nodes = set(np.random.choice(self.nodes, size=n_samples, replace=False))
        return self.create_subgraph(sampled_nodes)


class RandomPageRankNodeSampler(GraphSampler):
    """
    Random PageRank Node (RPN) Sampling.
    
    Select nodes with probability proportional to their PageRank score.
    Higher PageRank nodes are more likely to be selected.
    """
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform PageRank-weighted node sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        
        # Get PageRank scores as sampling probabilities
        nodes = list(self.pagerank.keys())
        probs = np.array([self.pagerank[n] for n in nodes])
        probs = probs / probs.sum()  # Normalize
        
        # Sample without replacement
        sampled_nodes = set(np.random.choice(nodes, size=n_samples, 
                                              replace=False, p=probs))
        return self.create_subgraph(sampled_nodes)


class RandomDegreeNodeSampler(GraphSampler):
    """
    Random Degree Node (RDN) Sampling.
    
    Select nodes with probability proportional to their degree.
    Higher degree nodes are more likely to be selected.
    """
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform degree-weighted node sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        
        # Get degrees as sampling probabilities
        nodes = list(self.degrees.keys())
        probs = np.array([self.degrees[n] for n in nodes], dtype=float)
        
        # Handle zero-degree nodes
        probs = probs + 1e-10
        probs = probs / probs.sum()
        
        # Sample without replacement
        sampled_nodes = set(np.random.choice(nodes, size=n_samples,
                                              replace=False, p=probs))
        return self.create_subgraph(sampled_nodes)


# =============================================================================
# Exploration Methods (Baseline)
# =============================================================================

class RandomWalkSampler(GraphSampler):
    """
    Random Walk (RW) Sampling.
    
    Start from a random node and perform random walk. With probability c,
    restart from the initial node (or a random node). Collect all visited nodes.
    
    Parameters from Leskovec & Faloutsos 2006:
    - c = 0.15 (restart probability)
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 restart_prob: float = RANDOM_WALK_RESTART_PROB):
        """
        Initialize random walk sampler.
        
        Args:
            G: Original graph
            random_state: Random seed
            restart_prob: Probability of restarting (c)
        """
        super().__init__(G, random_state)
        self.restart_prob = restart_prob
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform random walk sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        sampled_nodes = set()
        
        # Start from a random node
        start_node = np.random.choice(self.nodes)
        current_node = start_node
        sampled_nodes.add(current_node)
        
        # Maximum steps to avoid infinite loops
        max_steps = RANDOM_WALK_MAX_STEPS_MULTIPLIER * n_samples
        steps = 0
        
        while len(sampled_nodes) < n_samples and steps < max_steps:
            steps += 1
            
            # With probability c, restart
            if np.random.random() < self.restart_prob:
                current_node = start_node
            else:
                # Move to a random neighbor
                neighbors = self.get_neighbors(current_node)
                if neighbors:
                    current_node = np.random.choice(neighbors)
                else:
                    # Dead end, restart from a random node
                    current_node = np.random.choice(self.nodes)
            
            sampled_nodes.add(current_node)
        
        return self.create_subgraph(sampled_nodes)


class RandomJumpSampler(GraphSampler):
    """
    Random Jump (RJ) Sampling.
    
    Similar to random walk, but can jump to any random node (not just the start)
    with probability c.
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 jump_prob: float = RANDOM_WALK_RESTART_PROB):
        """
        Initialize random jump sampler.
        
        Args:
            G: Original graph
            random_state: Random seed
            jump_prob: Probability of random jump
        """
        super().__init__(G, random_state)
        self.jump_prob = jump_prob
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform random jump sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        sampled_nodes = set()
        
        # Start from a random node
        current_node = np.random.choice(self.nodes)
        sampled_nodes.add(current_node)
        
        max_steps = RANDOM_WALK_MAX_STEPS_MULTIPLIER * n_samples
        steps = 0
        
        while len(sampled_nodes) < n_samples and steps < max_steps:
            steps += 1
            
            # With probability c, jump to a random node
            if np.random.random() < self.jump_prob:
                current_node = np.random.choice(self.nodes)
            else:
                neighbors = self.get_neighbors(current_node)
                if neighbors:
                    current_node = np.random.choice(neighbors)
                else:
                    current_node = np.random.choice(self.nodes)
            
            sampled_nodes.add(current_node)
        
        return self.create_subgraph(sampled_nodes)


class ForestFireSampler(GraphSampler):
    """
    Forest Fire (FF) Sampling.
    
    Start from a random seed node. "Burn" a random number of outgoing edges
    (geometrically distributed with mean 1/(1-p_f)). Recursively apply to 
    burned neighbors. Continue until n nodes are collected.
    
    Parameters from Leskovec & Faloutsos 2006:
    - p_f = 0.7 for Scale-down goal
    - p_f = 0.2 for Back-in-time goal
    - p_b = 0 (no backward burning)
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 forward_prob: float = FF_FORWARD_PROB_SCALEDOWN,
                 backward_prob: float = FF_BACKWARD_PROB):
        """
        Initialize Forest Fire sampler.
        
        Args:
            G: Original graph
            random_state: Random seed
            forward_prob: Forward burning probability (p_f)
            backward_prob: Backward burning probability (p_b)
        """
        super().__init__(G, random_state)
        self.forward_prob = forward_prob
        self.backward_prob = backward_prob
    
    def _burn_neighbors(self, node: int, visited: Set[int], 
                        queue: deque, prob: float, 
                        get_neighbors_func) -> None:
        """
        Burn edges from a node.
        
        Args:
            node: Current burning node
            visited: Set of already visited nodes
            queue: Queue of nodes to process
            prob: Burning probability
            get_neighbors_func: Function to get neighbors
        """
        neighbors = get_neighbors_func(node)
        
        if not neighbors:
            return
        
        # Number of edges to burn (geometric distribution)
        if prob > 0:
            n_burn = np.random.geometric(1 - prob)
        else:
            n_burn = 0
        
        n_burn = min(n_burn, len(neighbors))
        
        if n_burn > 0:
            # Randomly select neighbors to burn
            burned = np.random.choice(neighbors, size=n_burn, replace=False)
            
            for neighbor in burned:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform forest fire sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        sampled_nodes = set()
        
        while len(sampled_nodes) < n_samples:
            # Pick a random seed node
            seed = np.random.choice(self.nodes)
            
            if seed in sampled_nodes:
                continue
            
            sampled_nodes.add(seed)
            queue = deque([seed])
            
            while queue and len(sampled_nodes) < n_samples:
                current = queue.popleft()
                
                # Forward burning (out-edges)
                if self.G.is_directed():
                    self._burn_neighbors(current, sampled_nodes, queue,
                                        self.forward_prob, 
                                        lambda n: list(self.G.successors(n)))
                    
                    # Backward burning (in-edges)
                    if self.backward_prob > 0:
                        self._burn_neighbors(current, sampled_nodes, queue,
                                            self.backward_prob,
                                            lambda n: list(self.G.predecessors(n)))
                else:
                    # Undirected graph
                    self._burn_neighbors(current, sampled_nodes, queue,
                                        self.forward_prob,
                                        lambda n: list(self.G.neighbors(n)))
        
        # Trim to exact size
        sampled_list = list(sampled_nodes)[:n_samples]
        return self.create_subgraph(set(sampled_list))


# =============================================================================
# Hybrid Sampling Methods
# =============================================================================

class HybridSampler(GraphSampler):
    """
    Hybrid Sampling Method.
    
    Combines node selection methods with exploration methods:
    1. First, select alpha*n seed nodes using a node selection method
    2. Then, explore from these seeds using an exploration method
    
    This aims to combine the global coverage of node selection with
    the local structure preservation of exploration methods.
    
    FIXED: Now properly accepts forward_prob parameter for FF exploration.
    """
    
    def __init__(self, G: nx.Graph, random_state: Optional[int] = None,
                 node_method: str = "RN", explore_method: str = "RW",
                 alpha: float = 0.5, forward_prob: float = FF_FORWARD_PROB_SCALEDOWN):
        """
        Initialize hybrid sampler.
        
        Args:
            G: Original graph
            random_state: Random seed
            node_method: Node selection method ("RN", "RPN", "RDN")
            explore_method: Exploration method ("RW", "FF")
            alpha: Fraction of nodes to select via node method (0 to 1)
            forward_prob: Forward burning probability for FF (FIXED: now properly used)
        """
        super().__init__(G, random_state)
        self.node_method = node_method
        self.explore_method = explore_method
        self.alpha = alpha
        self.forward_prob = forward_prob  # FIXED: Store forward_prob
    
    def _select_seed_nodes(self, n_seeds: int) -> Set[int]:
        """
        Select seed nodes using the specified node selection method.
        
        Args:
            n_seeds: Number of seed nodes to select
        
        Returns:
            Set of selected seed nodes
        """
        if n_seeds <= 0:
            return set()
        
        n_seeds = min(n_seeds, self.n_nodes)
        
        if self.node_method == "RN":
            # Uniform random selection
            return set(np.random.choice(self.nodes, size=n_seeds, replace=False))
        
        elif self.node_method == "RPN":
            # PageRank weighted selection
            nodes = list(self.pagerank.keys())
            probs = np.array([self.pagerank[n] for n in nodes])
            probs = probs / probs.sum()
            return set(np.random.choice(nodes, size=n_seeds, replace=False, p=probs))
        
        elif self.node_method == "RDN":
            # Degree weighted selection
            nodes = list(self.degrees.keys())
            probs = np.array([self.degrees[n] for n in nodes], dtype=float)
            probs = probs + 1e-10
            probs = probs / probs.sum()
            return set(np.random.choice(nodes, size=n_seeds, replace=False, p=probs))
        
        else:
            raise ValueError(f"Unknown node selection method: {self.node_method}")
    
    def _explore_random_walk(self, seed_nodes: Set[int], 
                              n_samples: int) -> Set[int]:
        """
        Explore from seed nodes using random walk.
        
        Args:
            seed_nodes: Starting seed nodes
            n_samples: Total number of nodes to collect
        
        Returns:
            Set of sampled nodes
        """
        sampled_nodes = seed_nodes.copy()
        seed_list = list(seed_nodes) if seed_nodes else [np.random.choice(self.nodes)]
        
        max_steps = RANDOM_WALK_MAX_STEPS_MULTIPLIER * n_samples
        current_node = np.random.choice(seed_list)
        
        steps = 0
        while len(sampled_nodes) < n_samples and steps < max_steps:
            steps += 1
            
            # With probability c, jump to a seed node
            if np.random.random() < RANDOM_WALK_RESTART_PROB:
                current_node = np.random.choice(seed_list)
            else:
                neighbors = self.get_neighbors(current_node)
                if neighbors:
                    current_node = np.random.choice(neighbors)
                else:
                    current_node = np.random.choice(seed_list)
            
            sampled_nodes.add(current_node)
        
        return sampled_nodes
    
    def _explore_forest_fire(self, seed_nodes: Set[int],
                              n_samples: int) -> Set[int]:
        """
        Explore from seed nodes using forest fire.
        
        FIXED: Now uses self.forward_prob instead of hardcoded default.
        
        Args:
            seed_nodes: Starting seed nodes
            n_samples: Total number of nodes to collect
        
        Returns:
            Set of sampled nodes
        """
        sampled_nodes = seed_nodes.copy()
        queue = deque(seed_nodes)
        
        # FIXED: Use self.forward_prob instead of hardcoded value
        forward_prob = self.forward_prob
        
        while len(sampled_nodes) < n_samples:
            if not queue:
                # All seeds exhausted, pick new random seed
                remaining = set(self.nodes) - sampled_nodes
                if not remaining:
                    break
                new_seed = np.random.choice(list(remaining))
                sampled_nodes.add(new_seed)
                queue.append(new_seed)
            
            current = queue.popleft()
            
            # Get neighbors
            if self.G.is_directed():
                neighbors = list(self.G.successors(current))
            else:
                neighbors = list(self.G.neighbors(current))
            
            if not neighbors:
                continue
            
            # Burn edges using the correct forward_prob
            n_burn = np.random.geometric(1 - forward_prob) if forward_prob > 0 else 0
            n_burn = min(n_burn, len(neighbors))
            
            if n_burn > 0:
                burned = np.random.choice(neighbors, size=n_burn, replace=False)
                for node in burned:
                    if node not in sampled_nodes:
                        sampled_nodes.add(node)
                        queue.append(node)
                        
                        if len(sampled_nodes) >= n_samples:
                            break
        
        return sampled_nodes
    
    def sample(self, n_samples: int) -> nx.Graph:
        """
        Perform hybrid sampling.
        
        Args:
            n_samples: Number of nodes to sample
        
        Returns:
            Induced subgraph on sampled nodes
        """
        n_samples = min(n_samples, self.n_nodes)
        
        # Step 1: Select seed nodes using node selection method
        n_seeds = int(n_samples * self.alpha)
        seed_nodes = self._select_seed_nodes(n_seeds)
        
        # Step 2: Explore from seeds using exploration method
        if self.explore_method == "RW":
            sampled_nodes = self._explore_random_walk(seed_nodes, n_samples)
        elif self.explore_method == "FF":
            sampled_nodes = self._explore_forest_fire(seed_nodes, n_samples)
        else:
            raise ValueError(f"Unknown exploration method: {self.explore_method}")
        
        # Trim to exact size
        sampled_list = list(sampled_nodes)[:n_samples]
        return self.create_subgraph(set(sampled_list))


# =============================================================================
# Factory Functions
# =============================================================================

def get_sampler(method: str, G: nx.Graph, random_state: Optional[int] = None,
                **kwargs) -> GraphSampler:
    """
    Factory function to get a sampler by method name.
    
    Args:
        method: Method name ("RN", "RPN", "RDN", "RW", "RJ", "FF", or "HYB-X-Y")
        G: Graph to sample from
        random_state: Random seed
        **kwargs: Additional arguments for specific samplers
            - forward_prob: Forward burning probability for FF methods
            - alpha: Mixing ratio for hybrid methods
    
    Returns:
        Appropriate sampler instance
    """
    method = method.upper()
    
    if method == "RN":
        return RandomNodeSampler(G, random_state)
    elif method == "RPN":
        return RandomPageRankNodeSampler(G, random_state)
    elif method == "RDN":
        return RandomDegreeNodeSampler(G, random_state)
    elif method == "RW":
        return RandomWalkSampler(G, random_state)
    elif method == "RJ":
        return RandomJumpSampler(G, random_state)
    elif method == "FF":
        forward_prob = kwargs.get("forward_prob", FF_FORWARD_PROB_SCALEDOWN)
        return ForestFireSampler(G, random_state, forward_prob=forward_prob)
    elif method.startswith("HYB-"):
        # Parse hybrid method name (e.g., "HYB-RN-RW")
        parts = method.split("-")
        if len(parts) != 3:
            raise ValueError(f"Invalid hybrid method format: {method}")
        node_method = parts[1]
        explore_method = parts[2]
        alpha = kwargs.get("alpha", 0.5)
        # FIXED: Pass forward_prob to HybridSampler
        forward_prob = kwargs.get("forward_prob", FF_FORWARD_PROB_SCALEDOWN)
        return HybridSampler(G, random_state, node_method, explore_method, 
                            alpha, forward_prob=forward_prob)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def sample_graph(G: nx.Graph, method: str, n_samples: int,
                 random_state: Optional[int] = None, **kwargs) -> nx.Graph:
    """
    Convenience function to sample a graph with a specified method.
    
    Args:
        G: Graph to sample from
        method: Sampling method name
        n_samples: Number of nodes to sample
        random_state: Random seed
        **kwargs: Additional arguments for the sampler
    
    Returns:
        Sampled subgraph
    """
    sampler = get_sampler(method, G, random_state, **kwargs)
    return sampler.sample(n_samples)


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_methods() -> List[str]:
    """
    Get list of all available sampling methods.
    
    Returns:
        List of method names
    """
    baseline = ["RN", "RPN", "RDN", "RW", "RJ", "FF"]
    
    node_methods = ["RN", "RPN", "RDN"]
    explore_methods = ["RW", "FF"]
    hybrid = [f"HYB-{n}-{e}" for n in node_methods for e in explore_methods]
    
    return baseline + hybrid


def describe_method(method: str) -> str:
    """
    Get a description of a sampling method.
    
    Args:
        method: Method name
    
    Returns:
        Description string
    """
    descriptions = {
        "RN": "Random Node - Uniform random node selection",
        "RPN": "Random PageRank Node - PageRank-weighted node selection",
        "RDN": "Random Degree Node - Degree-weighted node selection",
        "RW": "Random Walk - Walk-based exploration with restart",
        "RJ": "Random Jump - Random walk with random teleportation",
        "FF": "Forest Fire - BFS-like burning exploration",
    }
    
    method = method.upper()
    
    if method in descriptions:
        return descriptions[method]
    elif method.startswith("HYB-"):
        parts = method.split("-")
        if len(parts) == 3:
            node_method = parts[1]
            explore_method = parts[2]
            return f"Hybrid: {node_method} node selection + {explore_method} exploration"
    
    return "Unknown method"