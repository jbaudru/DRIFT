import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS, SIMULATION

class SourceTargetSelection:
    """
    Base class for source-target selection strategies
    """
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes)
    
    def get_source_target(self, agent_type=None, current_location=None, trip_count=0):
        """
        Get source and target nodes
        
        Args:
            agent_type: Type of agent (for activity-based selection)
            current_location: Current location of agent
            trip_count: Number of trips completed
        
        Returns:
            tuple: (source, target)
        """
        raise NotImplementedError

class RandomSelection(SourceTargetSelection):
    """
    Random source-target selection (original behavior)
    """
    def get_source_target(self, agent_type=None, current_location=None, trip_count=0):
        if current_location is None:
            source = random.choice(self.nodes)
        else:
            source = current_location
            
        target = random.choice(self.nodes)
        while target == source:
            target = random.choice(self.nodes)
        
        return source, target

class GravitySelection(SourceTargetSelection):
    """
    Gravity Model for source-target selection based on Voorhees (1956) transportation planning principles.
    
    Uses the Voorhees gravity formula:
    T_ij = K * (S_i * A_j) / d_ij^β
    
    Where:
    - T_ij: Number of trips from zone i to zone j
    - K: Constant (normalization factor)
    - S_i: Size variable of origin i (based on node degree/importance)
    - A_j: Attraction variable of destination j (based on centrality)
    - d_ij: Distance between i and j
    - β: Distance decay parameter
    
    Trip probability is:
    p_ij = T_ij / Σ(T_kl) = (S_i * A_j / d_ij^β) / Σ(S_k * A_l / d_kl^β)
    """
    
    def __init__(self, graph, config=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Voorhees gravity model parameters (configurable)
        self.beta = getattr(self.config, 'DEFAULT_GRAVITY_BETA', 2.0)   # Distance decay parameter
        self.alpha = getattr(self.config, 'DEFAULT_GRAVITY_ALPHA', 1.0)  # Attraction scaling parameter
        self.distance_cutoff = getattr(self.config, 'GRAVITY_DISTANCE_CUTOFF', None)  # Maximum distance to consider (None = no limit)
        
        # Performance optimization: cache computed probabilities
        self._probability_cache = {}
        self._cache_valid = False
        
        # Calculate size and attraction variables
        self._calculate_size_attraction_variables(graph)
        
        # Pre-compute distances efficiently
        self._precompute_distances_optimized(graph)
        
        # Pre-compute probability matrix for performance
        self._precompute_probability_matrix()
    
    def _calculate_size_attraction_variables(self, graph):
        """Calculate size variables S_i and attraction variables A_j for Voorhees formula"""
        print("Calculating size and attraction variables for Voorhees gravity model...")
        
        # Fast degree-based calculation for better performance
        print("  Computing degree centrality (fast)...")
        degree_centrality = nx.degree_centrality(graph)
        
        # Use degree as primary measure, with small base value to avoid zeros
        print("  Setting up size and attraction variables...")
        self.size_variables = {}  # S_i: origin size (generation potential)
        self.attraction_variables = {}  # A_j: destination attraction
        
        # For performance, use degree centrality as both size and attraction
        # In practice, these could be based on different node attributes
        for node in self.nodes:
            degree_score = degree_centrality.get(node, 0)
            # Add small base value to avoid zeros and ensure numerical stability
            base_value = 0.01
            
            # Size variable: generation potential (outbound trips)
            self.size_variables[node] = base_value + degree_score
            
            # Attraction variable: attractiveness for inbound trips
            # Could be different from size in real applications
            self.attraction_variables[node] = base_value + degree_score
        
    
    def _precompute_distances_optimized(self, graph):
        """Optimized distance computation with performance improvements"""
        print("Computing distances with optimizations...")
        
        # For very large graphs, use approximation immediately
        if len(self.nodes) > 1500:
            print(f"  Large graph ({len(self.nodes)} nodes). Using Euclidean approximation.")
            self.distances = self._fast_euclidean_distances(graph)
        else:
            try:
                print(f"  Computing shortest paths for {len(self.nodes)} nodes...")
                # Use generator version to reduce memory usage
                self.distances = {}
                node_count = 0
                
                with tqdm(total=len(self.nodes), desc="  Computing paths", unit="node") as pbar:
                    for source, path_dict in nx.all_pairs_shortest_path_length(graph):
                        self.distances[source] = dict(path_dict)
                        node_count += 1
                        pbar.update(1)
                        
                        # Memory check - if getting too large, switch to approximation
                        if node_count > 1000 and len(self.distances) * len(self.nodes) > 2000000:
                            print("  Memory limit reached. Switching to approximation.")
                            self.distances = self._fast_euclidean_distances(graph)
                            break
                            
            except (MemoryError, Exception) as e:
                print(f"  Error computing exact distances: {e}. Using approximation.")
                self.distances = self._fast_euclidean_distances(graph)
        
    
    def _fast_euclidean_distances(self, graph):
        """Ultra-fast Euclidean distance computation with threading and vectorization"""
        print("  Computing ultra-fast Euclidean distances...")
        
        # Extract positions efficiently
        positions = {}
        for node in self.nodes:
            if 'pos' in graph.nodes[node]:
                positions[node] = graph.nodes[node]['pos']
            else:
                # Consistent hash-based positioning
                hash_val = hash(str(node))
                positions[node] = (hash_val % 1000, (hash_val // 1000) % 1000)
        
        # Convert to numpy arrays for maximum performance
        node_list = list(positions.keys())
        pos_array = np.array([positions[node] for node in node_list])
        
        print("  Vectorized distance computation with threading...")
        
        # For very large graphs, use chunked computation with threading
        if len(node_list) > 3000:
            return self._chunked_distance_computation(node_list, pos_array)
        else:
            return self._full_vectorized_distances(node_list, pos_array)
    
    def _chunked_distance_computation(self, node_list, pos_array):
        """Compute distances in chunks using threading for memory efficiency"""
        
        # Calculate chunk size based on available memory
        chunk_size = min(1000, max(100, len(node_list) // mp.cpu_count()))
        node_chunks = [node_list[i:i + chunk_size] for i in range(0, len(node_list), chunk_size)]
        
        def compute_chunk_distances(chunk_info):
            chunk_nodes, start_idx = chunk_info
            chunk_size = len(chunk_nodes)
            chunk_pos = pos_array[start_idx:start_idx + chunk_size]
            
            # Compute distances from chunk nodes to all nodes
            chunk_distances = {}
            
            # Vectorized computation for this chunk
            for i, source_node in enumerate(chunk_nodes):
                source_pos = chunk_pos[i:i+1]  # Keep 2D shape
                diff = pos_array - source_pos
                euclidean_dists = np.sqrt(np.sum(diff**2, axis=1))
                
                # Store distances
                chunk_distances[source_node] = {}
                for j, target_node in enumerate(node_list):
                    if source_node == target_node:
                        chunk_distances[source_node][target_node] = 0.1
                    else:
                        chunk_distances[source_node][target_node] = max(euclidean_dists[j], 0.1)
            
            return chunk_distances
        
        # Prepare chunk information
        chunk_infos = [(node_chunks[i], i * len(node_chunks[i])) 
                      for i in range(len(node_chunks))]
        
        # Parallel computation
        max_workers = min(8, mp.cpu_count())
        distances = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(tqdm(
                executor.map(compute_chunk_distances, chunk_infos),
                total=len(chunk_infos),
                desc="  Distance chunks"
            ))
        
        # Combine results
        for chunk_result in chunk_results:
            distances.update(chunk_result)
        
        # Fast normalization using numpy
        self._fast_normalize_distances(distances)
        
        return distances
    
    def _full_vectorized_distances(self, node_list, pos_array):
        """Full vectorized distance computation for smaller graphs"""
        
        distances = {}
        
        # Use broadcasting for ultra-fast computation
        print("  Full vectorized computation...")
        
        # Compute all pairwise distances at once
        diff = pos_array[:, np.newaxis, :] - pos_array[np.newaxis, :, :]
        euclidean_dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Convert to dictionary format
        for i, source in enumerate(node_list):
            distances[source] = {}
            for j, target in enumerate(node_list):
                if i == j:
                    distances[source][target] = 0.1
                else:
                    distances[source][target] = max(euclidean_dists[i, j], 0.1)
        
        # Fast normalization
        self._fast_normalize_distances(distances)
        
        return distances
    
    def _fast_normalize_distances(self, distances):
        """Fast distance normalization using numpy operations"""
        
        # Collect all non-zero distances
        all_distances = []
        for source_dict in distances.values():
            all_distances.extend([d for d in source_dict.values() if d > 0])
        
        if not all_distances:
            return
        
        # Use numpy for fast min/max
        distance_array = np.array(all_distances)
        min_dist = np.min(distance_array)
        max_dist = np.max(distance_array)
        
        if max_dist > min_dist:
            # Vectorized normalization
            range_dist = max_dist - min_dist
            
            for source in distances:
                for target in distances[source]:
                    if distances[source][target] > 0:
                        # Normalize to 0.1-10 range
                        normalized = 0.1 + 9.9 * (distances[source][target] - min_dist) / range_dist
                        distances[source][target] = normalized
    
    def _precompute_probability_matrix(self):
        """Pre-compute Voorhees probability matrix with threading and optimizations"""
        print("Pre-computing Voorhees probability matrix with optimizations...")
        
        # For very large graphs, use sampling to reduce computation
        if len(self.nodes) > 5000:
            print(f"  Large graph ({len(self.nodes)} nodes). Using optimized sampling approach.")
            self._precompute_probability_matrix_optimized()
            return
        
        # Use threading for medium graphs
        self._precompute_probability_matrix_threaded()
    
    def _precompute_probability_matrix_optimized(self):
        """Optimized approach for very large graphs using sampling and approximation"""
        print("  Using sampling-based optimization...")
        
        # Sample a representative subset of nodes for full computation
        sample_size = min(2000, len(self.nodes) // 2)
        sampled_nodes = random.sample(self.nodes, sample_size)
        
        # Pre-compute for sampled nodes only
        self.trip_probabilities = {}
        self.cumulative_probabilities = {}
        
        # Convert to numpy arrays for vectorized computation
        node_indices = {node: i for i, node in enumerate(self.nodes)}
        size_array = np.array([self.size_variables[node] for node in self.nodes])
        attraction_array = np.array([self.attraction_variables[node] for node in self.nodes])
        
        print(f"  Computing probabilities for {len(sampled_nodes)} representative nodes...")
        
        # Threaded computation for sampled nodes
        def compute_source_probabilities(source):
            source_idx = node_indices[source]
            S_i = size_array[source_idx]
            
            # Get distances for this source
            source_distances = self.distances.get(source, {})
            
            # Vectorized computation where possible
            target_probs = {}
            
            for target in self.nodes:
                if source == target:
                    continue
                
                target_idx = node_indices[target]
                A_j = attraction_array[target_idx]
                d_ij = source_distances.get(target, float('inf'))
                
                if d_ij != float('inf') and d_ij > 0:
                    if not self.distance_cutoff or d_ij <= self.distance_cutoff:
                        T_ij = (S_i * A_j) / (d_ij ** self.beta)
                        target_probs[target] = T_ij
            
            # Normalize probabilities
            total = sum(target_probs.values())
            if total > 0:
                for target in target_probs:
                    target_probs[target] /= total
            
            return source, target_probs
        
        # Use threading for parallel computation
        max_workers = min(8, mp.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(compute_source_probabilities, sampled_nodes),
                total=len(sampled_nodes),
                desc="  Computing (threaded)"
            ))
        
        # Store results for sampled nodes
        for source, probs in results:
            self.trip_probabilities[source] = probs
        
        # For non-sampled nodes, use approximation based on similar nodes
        print("  Approximating probabilities for remaining nodes...")
        self._approximate_remaining_probabilities(sampled_nodes)
        
        # Pre-compute cumulative distributions
        print("  Pre-computing cumulative distributions...")
        self._precompute_cumulative_distributions_fast()
        
        self._cache_valid = True
        print("✓ Optimized probability matrix computation complete.")
    
    def _approximate_remaining_probabilities(self, sampled_nodes):
        """Approximate probabilities for non-sampled nodes based on similar nodes"""
        
        # Create a mapping from non-sampled to sampled nodes based on similarity
        sampled_sizes = {node: self.size_variables[node] for node in sampled_nodes}
        
        for node in self.nodes:
            if node in self.trip_probabilities:
                continue  # Already computed
            
            # Find the most similar sampled node based on size variable
            node_size = self.size_variables[node]
            best_match = min(sampled_nodes, 
                           key=lambda x: abs(sampled_sizes[x] - node_size))
            
            # Copy probabilities from best match with slight randomization
            if best_match in self.trip_probabilities:
                base_probs = self.trip_probabilities[best_match].copy()
                
                # Add slight randomization to avoid identical behavior
                noise_factor = 0.1
                for target in base_probs:
                    noise = random.uniform(1 - noise_factor, 1 + noise_factor)
                    base_probs[target] *= noise
                
                # Renormalize
                total = sum(base_probs.values())
                if total > 0:
                    for target in base_probs:
                        base_probs[target] /= total
                
                self.trip_probabilities[node] = base_probs
            else:
                # Fallback: uniform distribution
                self._create_uniform_distribution(node)
    
    def _create_uniform_distribution(self, source):
        """Create uniform distribution for a source node"""
        valid_targets = [n for n in self.nodes if n != source]
        if valid_targets:
            uniform_prob = 1.0 / len(valid_targets)
            self.trip_probabilities[source] = {target: uniform_prob for target in valid_targets}
        else:
            self.trip_probabilities[source] = {}
    
    def _precompute_probability_matrix_threaded(self):
        """Threaded computation for medium-sized graphs"""
        print("  Using threaded computation...")
        
        self.trip_probabilities = {}
        
        # Batch processing for better performance
        batch_size = max(50, len(self.nodes) // mp.cpu_count())
        node_batches = [self.nodes[i:i + batch_size] 
                       for i in range(0, len(self.nodes), batch_size)]
        
        def compute_batch_probabilities(node_batch):
            batch_results = {}
            
            for source in node_batch:
                S_i = self.size_variables[source]
                target_probs = {}
                
                for target in self.nodes:
                    if source == target:
                        continue
                    
                    A_j = self.attraction_variables[target]
                    d_ij = self.distances.get(source, {}).get(target, float('inf'))
                    
                    if d_ij != float('inf') and d_ij > 0:
                        if not self.distance_cutoff or d_ij <= self.distance_cutoff:
                            T_ij = (S_i * A_j) / (d_ij ** self.beta)
                            target_probs[target] = T_ij
                
                # Normalize probabilities
                total = sum(target_probs.values())
                if total > 0:
                    for target in target_probs:
                        target_probs[target] /= total
                else:
                    # Fallback to uniform distribution
                    valid_targets = [n for n in self.nodes if n != source]
                    if valid_targets:
                        uniform_prob = 1.0 / len(valid_targets)
                        target_probs = {target: uniform_prob for target in valid_targets}
                
                batch_results[source] = target_probs
            
            return batch_results
        
        # Process batches in parallel
        max_workers = min(8, mp.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(tqdm(
                executor.map(compute_batch_probabilities, node_batches),
                total=len(node_batches),
                desc="  Computing batches"
            ))
        
        # Combine results
        for batch_result in batch_results:
            self.trip_probabilities.update(batch_result)
        self._precompute_cumulative_distributions_fast()
        self._cache_valid = True
    
    def _precompute_cumulative_distributions_fast(self):
        """Fast computation of cumulative distributions using vectorization"""
        self.cumulative_probabilities = {}
        
        # Process in batches for memory efficiency
        batch_size = 1000
        node_batches = [list(self.trip_probabilities.keys())[i:i + batch_size] 
                       for i in range(0, len(self.trip_probabilities), batch_size)]
        
        for batch in tqdm(node_batches, desc="  Cumulative dist"):
            for source in batch:
                if source not in self.trip_probabilities:
                    continue
                
                probs_dict = self.trip_probabilities[source]
                
                if not probs_dict:
                    # Empty probabilities, create uniform fallback
                    valid_targets = [n for n in self.nodes if n != source]
                    if valid_targets:
                        uniform_step = 1.0 / len(valid_targets)
                        cumulative_probs = [uniform_step * (i + 1) for i in range(len(valid_targets))]
                        self.cumulative_probabilities[source] = (valid_targets, cumulative_probs)
                    continue
                
                # Sort by probability for better numerical stability
                sorted_items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                targets = [item[0] for item in sorted_items]
                probs = [item[1] for item in sorted_items]
                
                # Compute cumulative probabilities
                cumulative_probs = np.cumsum(probs).tolist()
                
                # Ensure last value is exactly 1.0
                if cumulative_probs and cumulative_probs[-1] > 0:
                    normalization_factor = 1.0 / cumulative_probs[-1]
                    cumulative_probs = [p * normalization_factor for p in cumulative_probs]
                
                self.cumulative_probabilities[source] = (targets, cumulative_probs)
    
    def get_source_target(self, agent_type=None, current_location=None, trip_count=0):
        """
        Get source and target using Voorhees gravity model with precomputed probabilities
        
        Args:
            agent_type: Type of agent (not used in basic gravity model)
            current_location: Current location of agent
            trip_count: Number of trips completed
        
        Returns:
            tuple: (source, target)
        """
        if current_location is None:
            # First trip: select source based on size variables (generation potential)
            source = self._select_weighted_source()
        else:
            source = current_location
        
        # Select target using precomputed Voorhees probabilities
        target = self._select_voorhees_target(source)
        
        return source, target
    
    def _select_weighted_source(self):
        """Select a source node weighted by its size variable (generation potential)"""
        nodes = list(self.size_variables.keys())
        weights = list(self.size_variables.values())
        return random.choices(nodes, weights=weights)[0]
    
    def _select_voorhees_target(self, source):
        """Select target using precomputed Voorhees probabilities (fast O(log n) selection)"""
        
        # Check if probabilities are cached and valid
        if not self._cache_valid or source not in self.cumulative_probabilities:
            # Fallback to slower computation if cache is invalid
            return self._compute_target_on_demand(source)
        
        targets, cumulative_probs = self.cumulative_probabilities[source]
        
        if not targets:
            # No valid targets, select randomly
            available_targets = [n for n in self.nodes if n != source]
            return random.choice(available_targets) if available_targets else source
        
        # Fast binary search selection using cumulative probabilities
        rand_val = random.random()
        
        # Binary search for target selection
        left, right = 0, len(cumulative_probs) - 1
        while left < right:
            mid = (left + right) // 2
            if cumulative_probs[mid] < rand_val:
                left = mid + 1
            else:
                right = mid
        
        return targets[left]
    
    def _compute_target_on_demand(self, source):
        """Fallback method: compute target probabilities on demand (slower)"""
        
        # Calculate Voorhees probabilities for this source
        target_probs = {}
        S_i = self.size_variables[source]
        
        for target in self.nodes:
            if target == source:
                continue
            
            A_j = self.attraction_variables[target]
            d_ij = self.distances.get(source, {}).get(target, float('inf'))
            
            if d_ij == float('inf') or d_ij <= 0:
                continue
            
            if self.distance_cutoff and d_ij > self.distance_cutoff:
                continue
            
            # Voorhees formula: T_ij = S_i * A_j / d_ij^β
            T_ij = (S_i * A_j) / (d_ij ** self.beta)
            target_probs[target] = T_ij
        
        # If no valid targets found, fallback to random selection
        if not target_probs:
            available_targets = [n for n in self.nodes if n != source]
            return random.choice(available_targets) if available_targets else source
        
        # Select target based on probabilities
        targets = list(target_probs.keys())
        probs = list(target_probs.values())
        
        return random.choices(targets, weights=probs)[0]
    
    def set_parameters(self, beta=None, distance_cutoff=None):
        """
        Set Voorhees gravity model parameters
        
        Args:
            beta: Distance decay parameter (higher = stronger distance penalty)
            distance_cutoff: Maximum distance to consider (None = no limit)
        """
        cache_invalidated = False
        
        if beta is not None and beta != self.beta:
            self.beta = beta
            cache_invalidated = True
        
        if distance_cutoff != self.distance_cutoff:
            self.distance_cutoff = distance_cutoff
            cache_invalidated = True
        
        # Recompute probability matrix if parameters changed
        if cache_invalidated:
            print("Parameters changed. Recomputing probability matrix...")
            self._precompute_probability_matrix()
    
    @property
    def node_importance(self):
        """Provide backward compatibility for visualization - use attraction variables as importance"""
        return self.attraction_variables.copy()
    
    def get_model_info(self):
        """Return information about the Voorhees gravity model parameters"""
        return {
            'model_type': 'voorhees_gravity',
            'beta': self.beta,
            'distance_cutoff': self.distance_cutoff,
            'num_nodes': len(self.nodes),
            'avg_size': np.mean(list(self.size_variables.values())),
            'avg_attraction': np.mean(list(self.attraction_variables.values())),
            'cache_valid': self._cache_valid
        }

class HubAndSpokeSelection(SourceTargetSelection):
    """
    Hub-and-Spoke Model for source-target selection.
    70% of trips involve major hubs (high centrality nodes).
    Reflects real-world traffic concentration at major intersections/centers.
    """
    
    def __init__(self, graph, config=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Hub-and-spoke parameters (configurable)
        self.hub_trip_probability = getattr(self.config, 'DEFAULT_HUB_TRIP_PROBABILITY', 0.3)  # % of trips involve hubs
        self.hub_percentage = getattr(self.config, 'DEFAULT_HUB_PERCENTAGE', 0.1)  # % of nodes considered hubs
        
        # Identify major hubs based on centrality
        self._identify_hubs(graph)
    
    def _identify_hubs(self, graph):
        """Identify major hubs based on node centrality"""
        print("Identifying major hubs for hub-and-spoke model...")
        
        # Calculate degree centrality (fast and effective for hub identification)
        print("  Computing degree centrality...")
        degree_centrality = nx.degree_centrality(graph)
        
        # For large graphs, also use betweenness centrality for better hub identification
        if len(self.nodes) <= 1000:
            print("  Computing betweenness centrality...")
            betweenness_centrality = nx.betweenness_centrality(graph)
            
            # Combine centralities: 60% degree, 40% betweenness
            combined_centrality = {}
            for node in self.nodes:
                degree_score = degree_centrality.get(node, 0)
                betweenness_score = betweenness_centrality.get(node, 0)
                combined_centrality[node] = 0.6 * degree_score + 0.4 * betweenness_score
        else:
            # For large graphs, use only degree centrality for performance
            print("  Using degree centrality only (large graph)...")
            combined_centrality = degree_centrality
        
        # Sort nodes by centrality and select top percentage as hubs
        sorted_nodes = sorted(self.nodes, key=lambda x: combined_centrality[x], reverse=True)
        num_hubs = max(1, int(len(self.nodes) * self.hub_percentage))
        
        self.hubs = sorted_nodes[:num_hubs]
        self.non_hubs = sorted_nodes[num_hubs:]
        
        # Store centrality values for reference
        self.node_centrality = combined_centrality
        
        print(f"✓ Hub identification complete. {len(self.hubs)} hubs identified from {len(self.nodes)} nodes")
        print(f"  Hub threshold: top {self.hub_percentage:.1%} of nodes")
        print(f"  Max centrality: {max(combined_centrality.values()):.3f}")
        
        # Print some example hubs
        if len(self.hubs) > 0:
            top_hubs = self.hubs[:min(5, len(self.hubs))]
            print(f"  Top hubs: {top_hubs}")
    
    def get_source_target(self, agent_type=None, current_location=None, trip_count=0):
        """
        Get source and target using hub-and-spoke model
        70% of trips involve at least one hub (source or target)
        
        Args:
            agent_type: Type of agent (not used in basic hub-and-spoke model)
            current_location: Current location of agent
            trip_count: Number of trips completed
        
        Returns:
            tuple: (source, target)
        """
        if current_location is None:
            # First trip: select source (can be hub or non-hub)
            source = random.choice(self.nodes)
        else:
            source = current_location
        
        # Decide if this trip involves a hub
        if random.random() < self.hub_trip_probability:
            # Hub trip (70%): at least one end must be a hub
            target = self._select_hub_trip_target(source)
            trip_type = "hub"
        else:
            # Non-hub trip (30%): avoid hubs
            target = self._select_non_hub_trip_target(source)
            trip_type = "non-hub"
        
        # Store trip type for this selection (can be accessed by agent)
        self.last_trip_type = trip_type
        
        return source, target
    
    def _select_hub_trip_target(self, source):
        """Select target for a hub trip (70% of trips)"""
        
        # If source is already a hub, target can be any node
        if source in self.hubs:
            # Hub-to-anywhere trip
            available_targets = [n for n in self.nodes if n != source]
            return random.choice(available_targets)
        
        # If source is not a hub, target must be a hub
        else:
            # Non-hub-to-hub trip
            available_hubs = [h for h in self.hubs if h != source]
            if available_hubs:
                return random.choice(available_hubs)
            else:
                # Fallback: any node except source
                available_targets = [n for n in self.nodes if n != source]
                return random.choice(available_targets)
    
    def _select_non_hub_trip_target(self, source):
        """Select target for a non-hub trip (30% of trips)"""
        
        # Try to avoid hubs for both source and target
        if source in self.hubs:
            # Source is a hub, try to pick non-hub target
            available_targets = [n for n in self.non_hubs if n != source]
            if available_targets:
                return random.choice(available_targets)
            else:
                # Fallback: any node except source
                available_targets = [n for n in self.nodes if n != source]
                return random.choice(available_targets)
        else:
            # Source is not a hub, pick non-hub target
            available_targets = [n for n in self.non_hubs if n != source]
            if available_targets:
                return random.choice(available_targets)
            else:
                # Fallback: any node except source
                available_targets = [n for n in self.nodes if n != source]
                return random.choice(available_targets)
    
    def set_parameters(self, hub_trip_probability=None, hub_percentage=None):
        """
        Set hub-and-spoke model parameters
        
        Args:
            hub_trip_probability: Probability that a trip involves a hub (0.0-1.0)
            hub_percentage: Percentage of nodes to consider as hubs (0.0-1.0)
        """
        if hub_trip_probability is not None:
            self.hub_trip_probability = max(0.0, min(1.0, hub_trip_probability))
        
        if hub_percentage is not None:
            self.hub_percentage = max(0.01, min(1.0, hub_percentage))
            # Recalculate hubs if percentage changed
            num_hubs = max(1, int(len(self.nodes) * self.hub_percentage))
            sorted_nodes = sorted(self.nodes, key=lambda x: self.node_centrality[x], reverse=True)
            self.hubs = sorted_nodes[:num_hubs]
            self.non_hubs = sorted_nodes[num_hubs:]
    
    def get_model_info(self):
        """Return information about the hub-and-spoke model"""
        return {
            'model_type': 'hub_and_spoke',
            'hub_trip_probability': self.hub_trip_probability,
            'hub_percentage': self.hub_percentage,
            'num_nodes': len(self.nodes),
            'num_hubs': len(self.hubs),
            'num_non_hubs': len(self.non_hubs),
            'max_centrality': max(self.node_centrality.values()),
            'min_centrality': min(self.node_centrality.values()),
            'avg_centrality': np.mean(list(self.node_centrality.values()))
        }
    
    def get_hub_nodes(self):
        """Return list of hub nodes for visualization"""
        return {
            'hubs': self.hubs,
            'non_hubs': self.non_hubs,
            'centrality': self.node_centrality
        }

class ActivityBasedSelection(SourceTargetSelection):
    """
    Activity-based source-target selection model for different agent types
    """
    
    def __init__(self, graph, config=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Store zone thresholds as configurable parameters
        self.center_threshold = getattr(self.config, 'ACTIVITY_CENTER_THRESHOLD', 0.3)  # 30% of radius from center
        self.middle_threshold = getattr(self.config, 'ACTIVITY_MIDDLE_THRESHOLD', 0.7)  # 70% of radius from center
        
        # Store node size factors as configurable parameters
        self.activity_size_factor = getattr(self.config, 'ACTIVITY_SIZE_FACTOR', 6)  # nodes//6
        self.activity_max_size = getattr(self.config, 'ACTIVITY_MAX_SIZE', 50)
        self.business_size_factor = getattr(self.config, 'BUSINESS_SIZE_FACTOR', 5)  # nodes//5  
        self.business_max_size = getattr(self.config, 'BUSINESS_MAX_SIZE', 80)
        self.work_size_factor = getattr(self.config, 'WORK_SIZE_FACTOR', 4)  # nodes//4
        self.work_max_size = getattr(self.config, 'WORK_MAX_SIZE', 100)
        self.home_size_factor = getattr(self.config, 'HOME_SIZE_FACTOR', 4)  # nodes//4
        self.home_max_size = getattr(self.config, 'HOME_MAX_SIZE', 100)
        self.distribution_size_factor = getattr(self.config, 'DISTRIBUTION_SIZE_FACTOR', 10)  # nodes//10
        self.distribution_max_size = getattr(self.config, 'DISTRIBUTION_MAX_SIZE', 20)
        
        # Store distribution ratios as configurable parameters
        self.activity_center_ratio = getattr(self.config, 'ACTIVITY_CENTER_RATIO', 0.8)  # 80% center, 20% middle
        self.business_center_ratio = getattr(self.config, 'BUSINESS_CENTER_RATIO', 0.7)  # 70% center, 30% middle
        self.work_ratios = getattr(self.config, 'WORK_RATIOS', (0.2, 0.4, 0.4))  # 20% center, 40% middle, 40% border
        self.home_ratios = getattr(self.config, 'HOME_RATIOS', (0.05, 0.35, 0.6))  # 5% center, 35% middle, 60% border
        self.distribution_ratios = getattr(self.config, 'DISTRIBUTION_RATIOS', (0.5, 0.5))  # 50% middle, 50% border
        
        # Get node positions and calculate spatial distribution
        self._distribute_nodes_spatially(graph)
    
    def _distribute_nodes_spatially(self, graph):
        """Distribute nodes based on their spatial position in the graph"""
        # Get all node positions
        node_positions = {}
        for node_id in self.nodes:
            pos = graph.nodes[node_id]['pos']
            node_positions[node_id] = pos
        
        # Calculate center and distances
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        
        # Calculate max distance from center for normalization
        max_distance = 0
        distances = {}
        for node_id, pos in node_positions.items():
            dist = ((pos[0] - center_x)**2 + (pos[1] - center_y)**2)**0.5
            distances[node_id] = dist
            max_distance = max(max_distance, dist)
        
        # Normalize distances (0 = center, 1 = edge)
        normalized_distances = {node_id: dist / max_distance for node_id, dist in distances.items()}
        
        # Sort nodes by distance from center
        nodes_by_distance = sorted(self.nodes, key=lambda x: normalized_distances[x])
        
        # Define thresholds for different zones (now configurable)
        center_nodes = [n for n in self.nodes if normalized_distances[n] <= self.center_threshold]
        middle_nodes = [n for n in self.nodes if self.center_threshold < normalized_distances[n] <= self.middle_threshold]
        border_nodes = [n for n in self.nodes if normalized_distances[n] > self.middle_threshold]
        
        # Distribute node types based on urban planning principles
        self._assign_activity_centers(center_nodes, middle_nodes, border_nodes)
        self._assign_business_nodes(center_nodes, middle_nodes, border_nodes)
        self._assign_work_nodes(center_nodes, middle_nodes, border_nodes)
        self._assign_home_nodes(center_nodes, middle_nodes, border_nodes)
        self._assign_distribution_nodes(center_nodes, middle_nodes, border_nodes)
    
    def _assign_activity_centers(self, center_nodes, middle_nodes, border_nodes):
        """Activity centers: mostly in center, some in middle"""
        activity_size = min(len(self.nodes)//self.activity_size_factor, self.activity_max_size)
        
        # Use configurable ratio
        center_count = min(int(activity_size * self.activity_center_ratio), len(center_nodes))
        middle_count = min(activity_size - center_count, len(middle_nodes))
        
        # Ensure we have at least some activity centers
        activity_centers_list = []
        if center_count > 0:
            activity_centers_list.extend(random.sample(center_nodes, center_count))
        if middle_count > 0:
            activity_centers_list.extend(random.sample(middle_nodes, middle_count))
        
        # If no activity centers assigned, fallback to any available nodes
        if not activity_centers_list:
            fallback_nodes = random.sample(self.nodes, min(3, len(self.nodes)))
            activity_centers_list = fallback_nodes
        
        self.activity_centers = activity_centers_list
    
    def _assign_business_nodes(self, center_nodes, middle_nodes, border_nodes):
        """Business nodes: mostly in center, some in middle"""
        business_size = min(len(self.nodes)//self.business_size_factor, self.business_max_size)
        
        # Use configurable ratio
        center_count = min(int(business_size * self.business_center_ratio), len(center_nodes))
        middle_count = min(business_size - center_count, len(middle_nodes))
        
        # Remove already assigned activity centers from available center nodes
        available_center = [n for n in center_nodes if n not in self.activity_centers]
        available_middle = [n for n in middle_nodes if n not in self.activity_centers]
        
        center_count = min(center_count, len(available_center))
        middle_count = min(middle_count, len(available_middle))
        
        # Ensure we have at least some business nodes
        business_nodes_list = []
        if center_count > 0:
            business_nodes_list.extend(random.sample(available_center, center_count))
        if middle_count > 0:
            business_nodes_list.extend(random.sample(available_middle, middle_count))
        
        # If no business nodes assigned, fallback to any available nodes
        if not business_nodes_list:
            fallback_nodes = [n for n in self.nodes if n not in self.activity_centers]
            if fallback_nodes:
                business_nodes_list = random.sample(fallback_nodes, min(5, len(fallback_nodes)))
            else:
                # Last resort: use any nodes
                business_nodes_list = random.sample(self.nodes, min(5, len(self.nodes)))
        
        self.business_nodes = business_nodes_list
    
    def _assign_work_nodes(self, center_nodes, middle_nodes, border_nodes):
        """Work nodes: some in center, distributed in middle and border"""
        work_size = min(len(self.nodes)//self.work_size_factor, self.work_max_size)
        
        # Use configurable ratios (center, middle, border)
        center_ratio, middle_ratio, border_ratio = self.work_ratios
        center_count = int(work_size * center_ratio)
        middle_count = int(work_size * middle_ratio)
        border_count = work_size - center_count - middle_count
        
        # Remove already assigned nodes
        assigned_nodes = set(self.activity_centers + self.business_nodes)
        available_center = [n for n in center_nodes if n not in assigned_nodes]
        available_middle = [n for n in middle_nodes if n not in assigned_nodes]
        available_border = border_nodes.copy()
        
        center_count = min(center_count, len(available_center))
        middle_count = min(middle_count, len(available_middle))
        border_count = min(border_count, len(available_border))
        
        # Build work nodes list
        work_nodes_list = []
        if center_count > 0:
            work_nodes_list.extend(random.sample(available_center, center_count))
        if middle_count > 0:
            work_nodes_list.extend(random.sample(available_middle, middle_count))
        if border_count > 0:
            work_nodes_list.extend(random.sample(available_border, border_count))
        
        # If no work nodes assigned, fallback to any available nodes
        if not work_nodes_list:
            fallback_nodes = [n for n in self.nodes if n not in assigned_nodes]
            if fallback_nodes:
                work_nodes_list = random.sample(fallback_nodes, min(5, len(fallback_nodes)))
            else:
                # Last resort: use any nodes
                work_nodes_list = random.sample(self.nodes, min(5, len(self.nodes)))
        
        self.work_nodes = work_nodes_list
    
    def _assign_home_nodes(self, center_nodes, middle_nodes, border_nodes):
        """Home nodes: rare in center, mostly in middle and border"""
        home_size = min(len(self.nodes)//self.home_size_factor, self.home_max_size)
        
        # Use configurable ratios (center, middle, border)
        center_ratio, middle_ratio, border_ratio = self.home_ratios
        center_count = int(home_size * center_ratio)
        middle_count = int(home_size * middle_ratio)
        border_count = home_size - center_count - middle_count
        
        # Remove already assigned nodes
        assigned_nodes = set(self.activity_centers + self.business_nodes + self.work_nodes)
        available_center = [n for n in center_nodes if n not in assigned_nodes]
        available_middle = [n for n in middle_nodes if n not in assigned_nodes]
        available_border = [n for n in border_nodes if n not in assigned_nodes]
        
        center_count = min(center_count, len(available_center))
        middle_count = min(middle_count, len(available_middle))
        border_count = min(border_count, len(available_border))
        
        # Build home nodes list
        home_nodes_list = []
        if center_count > 0:
            home_nodes_list.extend(random.sample(available_center, center_count))
        if middle_count > 0:
            home_nodes_list.extend(random.sample(available_middle, middle_count))
        if border_count > 0:
            home_nodes_list.extend(random.sample(available_border, border_count))
        
        # If no home nodes assigned, fallback to any available nodes
        if not home_nodes_list:
            fallback_nodes = [n for n in self.nodes if n not in assigned_nodes]
            if fallback_nodes:
                home_nodes_list = random.sample(fallback_nodes, min(5, len(fallback_nodes)))
            else:
                # Last resort: use any nodes
                home_nodes_list = random.sample(self.nodes, min(5, len(self.nodes)))
        
        self.home_nodes = home_nodes_list
    
    def _assign_distribution_nodes(self, center_nodes, middle_nodes, border_nodes):
        """Distribution nodes: 50% middle, 50% border"""
        distribution_size = min(len(self.nodes)//self.distribution_size_factor, self.distribution_max_size)
        
        # Use configurable ratios (middle, border)
        middle_ratio, border_ratio = self.distribution_ratios
        middle_count = int(distribution_size * middle_ratio)
        border_count = distribution_size - middle_count
        
        # Remove already assigned nodes
        assigned_nodes = set(self.activity_centers + self.business_nodes + 
                           self.work_nodes + self.home_nodes)
        available_middle = [n for n in middle_nodes if n not in assigned_nodes]
        available_border = [n for n in border_nodes if n not in assigned_nodes]
        
        middle_count = min(middle_count, len(available_middle))
        border_count = min(border_count, len(available_border))
        
        # Build distribution nodes list
        distribution_nodes_list = []
        if middle_count > 0:
            distribution_nodes_list.extend(random.sample(available_middle, middle_count))
        if border_count > 0:
            distribution_nodes_list.extend(random.sample(available_border, border_count))
        
        # If no distribution nodes assigned, fallback to any available nodes
        if not distribution_nodes_list:
            fallback_nodes = [n for n in self.nodes if n not in assigned_nodes]
            if fallback_nodes:
                distribution_nodes_list = random.sample(fallback_nodes, min(3, len(fallback_nodes)))
            else:
                # Last resort: use any nodes
                distribution_nodes_list = random.sample(self.nodes, min(3, len(self.nodes)))
        
        self.distribution_nodes = distribution_nodes_list
    
    def get_source_target(self, agent_type='random', current_location=None, trip_count=0):
        """
        Get source and target based on agent type and activity pattern
        
        Args:
            agent_type: 'commuter', 'delivery', 'leisure', 'business', 'random'
            current_location: current position of agent (for continuation trips)
            trip_count: number of trips completed (for pattern behavior)
        
        Returns:
            tuple: (source, target)
        """
        if agent_type == 'commuter':
            return self._commuter_pattern(current_location, trip_count)
        elif agent_type == 'delivery':
            return self._delivery_pattern(current_location)
        elif agent_type == 'leisure':
            return self._leisure_pattern(current_location)
        elif agent_type == 'business':
            return self._business_pattern(current_location)
        else:
            # Fallback to random
            return self._random_pattern(current_location)
    
    def _commuter_pattern(self, current_location, trip_count):
        """Commuters alternate between home and work"""
        if current_location is None:
            # First trip: start from home, fallback to any node if home_nodes is empty
            if self.home_nodes:
                source = random.choice(self.home_nodes)
            else:
                source = random.choice(self.nodes)
                
            if self.work_nodes:
                target = random.choice(self.work_nodes)
            else:
                target = random.choice(self.nodes)
        else:
            source = current_location
            # Alternate between home and work based on trip count
            if trip_count % 2 == 0:
                # Even trips: go to work (or stay at work area)
                if self.work_nodes:
                    target = random.choice(self.work_nodes)
                else:
                    target = random.choice(self.nodes)
            else:
                # Odd trips: go home
                if self.home_nodes:
                    target = random.choice(self.home_nodes)
                else:
                    target = random.choice(self.nodes)
        
        return source, target
    
    def _delivery_pattern(self, current_location):
        """Delivery agents prefer distribution centers and varied destinations"""
        if current_location is None:
            # Fallback to any node if distribution_nodes is empty
            if self.distribution_nodes:
                source = random.choice(self.distribution_nodes)
            else:
                source = random.choice(self.nodes)
        else:
            source = current_location
        
        # 70% chance to go to/from distribution centers, 30% to other locations
        if random.random() < 0.7 and self.distribution_nodes:
            target = random.choice(self.distribution_nodes)
        else:
            target = random.choice(self.nodes)
        
        # Ensure source != target
        attempts = 0
        while target == source and attempts < 100:  # Prevent infinite loop
            attempts += 1
            target = random.choice(self.nodes)
            
            # If we have only one node total, break the loop
            if len(self.nodes) <= 1:
                break
        
        return source, target
    
    def _leisure_pattern(self, current_location):
        """Leisure agents have varied destinations with preference for activity centers"""
        if current_location is None:
            # Fallback to any node if home_nodes is empty
            if self.home_nodes:
                source = random.choice(self.home_nodes)
            else:
                source = random.choice(self.nodes)
        else:
            source = current_location
        
        # 60% chance to go to activity centers, 40% to random locations
        if random.random() < 0.6 and self.activity_centers:
            target = random.choice(self.activity_centers)
        else:
            target = random.choice(self.nodes)
        
        # Ensure source != target
        attempts = 0
        while target == source and attempts < 100:  # Prevent infinite loop
            attempts += 1
            if random.random() < 0.6 and self.activity_centers:
                target = random.choice(self.activity_centers)
            else:
                target = random.choice(self.nodes)
            
            # If we have only one node total, break the loop
            if len(self.nodes) <= 1:
                break
        
        return source, target
    
    def _business_pattern(self, current_location):
        """Business agents focus on major business nodes"""
        if current_location is None:
            # Fallback to any node if business_nodes is empty
            if self.business_nodes:
                source = random.choice(self.business_nodes)
            else:
                source = random.choice(self.nodes)
        else:
            source = current_location
        
        # 80% chance to go to business nodes, 20% to other locations
        if random.random() < 0.8 and self.business_nodes:
            target = random.choice(self.business_nodes)
        else:
            target = random.choice(self.nodes)
        
        # Ensure source != target
        attempts = 0
        while target == source and attempts < 100:  # Prevent infinite loop
            attempts += 1
            if random.random() < 0.8 and self.business_nodes:
                target = random.choice(self.business_nodes)
            else:
                target = random.choice(self.nodes)
            
            # If we have only one node total, break the loop
            if len(self.nodes) <= 1:
                break
        
        return source, target
    
    def _random_pattern(self, current_location):
        """Random source-target selection (fallback)"""
        if current_location is None:
            source = random.choice(self.nodes)
        else:
            source = current_location
        
        target = random.choice(self.nodes)
        attempts = 0
        while target == source and attempts < 100:  # Prevent infinite loop
            attempts += 1
            target = random.choice(self.nodes)
            
            # If we have only one node total, break the loop
            if len(self.nodes) <= 1:
                break
        
        return source, target
    
    def get_agent_types_distribution(self):
        """Return suggested distribution of agent types"""
        return {
            'commuter': 0.4,    # 40% commuters
            'delivery': 0.15,   # 15% delivery
            'leisure': 0.25,    # 25% leisure
            'business': 0.2     # 20% business
        }
    
    def get_activity_nodes(self):
        """Return dictionary of activity nodes for visualization"""
        return {
            'activity_centers': getattr(self, 'activity_centers', []),
            'business_nodes': getattr(self, 'business_nodes', []),
            'work_nodes': getattr(self, 'work_nodes', []),
            'home_nodes': getattr(self, 'home_nodes', []),
            'distribution_nodes': getattr(self, 'distribution_nodes', [])
        }

class ZoneBasedSelection(SourceTargetSelection):
    """
    Zone-based source-target selection using Transportation Analysis Zones (TAZ) methodology
    Divides network into 3x3 grid with 60% intra-zone and 40% inter-zone trips
    """
    
    def __init__(self, graph, config=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Store configurable parameters
        self.intra_zone_probability = getattr(self.config, 'DEFAULT_ZONE_INTRA_PROBABILITY', 0.7)
        self.grid_size = getattr(self.config, 'ZONE_GRID_SIZE', 3)  # 3x3 grid by default
        
        # Create zone grid
        self._create_zone_grid(graph)
    
    def _create_zone_grid(self, graph):
        """Create a configurable grid of zones and assign nodes to zones"""
        # Get all node positions
        node_positions = {}
        for node_id in self.nodes:
            pos = graph.nodes[node_id]['pos']
            node_positions[node_id] = pos
        
        # Calculate bounds
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Create configurable grid boundaries
        x_step = (max_x - min_x) / self.grid_size
        y_step = (max_y - min_y) / self.grid_size
        
        # Initialize zones
        self.zones = {}
        self.node_to_zone = {}
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                zone_id = i * self.grid_size + j
                self.zones[zone_id] = []
        
        # Assign nodes to zones
        for node_id, pos in node_positions.items():
            # Determine which grid cell this node belongs to
            x_grid = min(self.grid_size - 1, int((pos[0] - min_x) / x_step))
            y_grid = min(self.grid_size - 1, int((pos[1] - min_y) / y_step))
            
            zone_id = x_grid * self.grid_size + y_grid
            self.zones[zone_id].append(node_id)
            self.node_to_zone[node_id] = zone_id
        
        # Identify major nodes in each zone (nodes with high connectivity)
        self._identify_major_nodes(graph)
    
    def _identify_major_nodes(self, graph):
        """Identify major nodes in each zone based on connectivity"""
        self.major_nodes_by_zone = {}
        
        for zone_id, zone_nodes in self.zones.items():
            if not zone_nodes:
                self.major_nodes_by_zone[zone_id] = []
                continue
            
            # Calculate degree (number of connections) for each node in the zone
            node_degrees = {}
            for node_id in zone_nodes:
                node_degrees[node_id] = graph.degree(node_id)
            
            # Select top 30% as major nodes (or at least 1 node per zone)
            num_major = max(1, len(zone_nodes) // 3)
            major_nodes = sorted(zone_nodes, key=lambda x: node_degrees[x], reverse=True)[:num_major]
            self.major_nodes_by_zone[zone_id] = major_nodes
    
    def get_source_target(self, agent_type=None, current_location=None, trip_count=0):
        """
        Get source and target based on zone-based patterns
        60% intra-zone trips, 40% inter-zone trips
        """
        if current_location is None:
            # First trip: start from random node
            source = random.choice(self.nodes)
        else:
            source = current_location
        
        source_zone = self.node_to_zone[source]
        
        # Decide if this is an intra-zone or inter-zone trip
        if random.random() < self.intra_zone_probability:
            # Intra-zone trip (configurable %)
            target = self._select_intra_zone_target(source_zone, source)
        else:
            # Inter-zone trip (remaining %)
            target = self._select_inter_zone_target(source_zone, source)
        
        return source, target
    
    def _select_intra_zone_target(self, zone_id, source):
        """Select target within the same zone"""
        zone_nodes = self.zones[zone_id]
        
        # Remove source from available targets
        available_targets = [n for n in zone_nodes if n != source]
        
        if not available_targets:
            # If no other nodes in zone, pick from adjacent zones
            return self._select_inter_zone_target(zone_id, source)
        
        return random.choice(available_targets)
    
    def _select_inter_zone_target(self, source_zone_id, source):
        """Select target from a different zone, preferring major nodes"""
        # Get all zones except source zone
        available_zones = [z for z in self.zones.keys() if z != source_zone_id]
        
        # Select destination zone
        dest_zone_id = random.choice(available_zones)
        
        # 70% chance to target major nodes, 30% any node in destination zone
        if random.random() < 0.7 and self.major_nodes_by_zone[dest_zone_id]:
            # Target major nodes in destination zone
            available_targets = [n for n in self.major_nodes_by_zone[dest_zone_id] if n != source]
        else:
            # Target any node in destination zone
            available_targets = [n for n in self.zones[dest_zone_id] if n != source]
        
        if not available_targets:
            # Fallback to any node in the destination zone
            available_targets = [n for n in self.zones[dest_zone_id] if n != source]
        
        if not available_targets:
            # Ultimate fallback: any node in the graph
            available_targets = [n for n in self.nodes if n != source]
        
        return random.choice(available_targets)
    
    def set_parameters(self, intra_zone_probability=None):
        """
        Set zone-based model parameters
        
        Args:
            intra_zone_probability: Probability of trips within the same zone (0.0-1.0)
        """
        if intra_zone_probability is not None:
            self.intra_zone_probability = max(0.0, min(1.0, intra_zone_probability))
    
    def get_zone_info(self):
        """Return information about the zones"""
        zone_info = {}
        for zone_id, nodes in self.zones.items():
            zone_info[zone_id] = {
                'total_nodes': len(nodes),
                'major_nodes': len(self.major_nodes_by_zone.get(zone_id, []))
            }
        return zone_info
    def get_zone_info(self):
        """Return information about the zones for debugging/visualization"""
        zone_info = {}
        for zone_id, zone_nodes in self.zones.items():
            zone_info[zone_id] = {
                'total_nodes': len(zone_nodes),
                'major_nodes': len(self.major_nodes_by_zone[zone_id]),
                'nodes': zone_nodes[:5]  # Show first 5 nodes as example
            }
        return zone_info
