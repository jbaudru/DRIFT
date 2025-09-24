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
    Simplified Gravity Model for source-target selection based on Voorhees (1956) transportation planning principles.
    
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
    
    def __init__(self, graph, config=None, alpha=None, beta=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Use provided parameters first, then config, then hardcoded defaults
        self.beta = beta if beta is not None else getattr(self.config, 'DEFAULT_GRAVITY_BETA', 2.0)   # Distance decay parameter
        self.alpha = alpha if alpha is not None else getattr(self.config, 'DEFAULT_GRAVITY_ALPHA', 1.0)  # Attraction scaling parameter
        self.distance_cutoff = getattr(self.config, 'GRAVITY_DISTANCE_CUTOFF', None)  # Maximum distance to consider (None = no limit)
        
        # Simplified caching
        self.size_variables = {}
        self.attraction_variables = {}
        self.distances = {}
        self._cache_valid = False
        
        # Initialize the model with a progress indicator
        self._initialize_model(graph)
    
    @classmethod
    def create_with_progress(cls, graph, config=None, alpha=None, beta=None, progress_callback=None):
        """
        Factory method to create GravitySelection with external progress callback
        Useful for showing loading spinners from external components
        
        Args:
            graph: NetworkX graph
            config: Configuration object
            alpha: Alpha parameter  
            beta: Beta parameter
            progress_callback: Function to call with progress updates (message)
        
        Returns:
            GravitySelection instance
        """
        instance = cls.__new__(cls)
        SourceTargetSelection.__init__(instance, graph)
        
        # Store configuration
        instance.config = config if config is not None else MODELS
        instance.beta = beta if beta is not None else getattr(instance.config, 'DEFAULT_GRAVITY_BETA', 2.0)
        instance.alpha = alpha if alpha is not None else getattr(instance.config, 'DEFAULT_GRAVITY_ALPHA', 1.0)
        instance.distance_cutoff = getattr(instance.config, 'GRAVITY_DISTANCE_CUTOFF', None)
        
        # Initialize containers
        instance.size_variables = {}
        instance.attraction_variables = {}
        instance.distances = {}
        instance._cache_valid = False
        
        # Initialize with progress callback
        instance._initialize_model_with_progress(graph, progress_callback)
        
        return instance
    
    def _initialize_model_with_progress(self, graph, progress_callback=None):
        """Initialize model with external progress callback"""
        if progress_callback:
            progress_callback("Initializing gravity model...")
        
        print("Initializing Voorhees Gravity Model...")
        
        try:
            # Step 1: Calculate centrality-based variables
            if progress_callback:
                progress_callback("Computing node centralities...")
            print("  Step 1/3: Computing node centralities...")
            self._calculate_size_attraction_variables(graph)
            
            # Step 2: Compute distances
            if progress_callback:
                progress_callback("Computing distances...")
            print("  Step 2/3: Computing distances...")
            self._compute_distances(graph)
            
            # Step 3: Mark as ready
            if progress_callback:
                progress_callback("Model ready!")
            print("  Step 3/3: Model ready!")
            self._cache_valid = True
            
            print("✓ Voorhees Gravity Model initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing gravity model: {e}")
            raise
    
    
    def _initialize_model(self, graph):
        """Initialize the gravity model with progress tracking"""
        print("Initializing Voorhees Gravity Model...")
        
        try:
            # Show loading spinner if we're in a GUI context
            self._show_loading("Initializing gravity model...")
            
            # Step 1: Calculate centrality-based variables
            print("  Step 1/3: Computing node centralities...")
            self._update_loading_progress("Computing node centralities...", 1, 3)
            self._calculate_size_attraction_variables(graph)
            
            # Step 2: Compute distances
            print("  Step 2/3: Computing distances...")
            self._update_loading_progress("Computing distances...", 2, 3)
            self._compute_distances(graph)
            
            # Step 3: Mark as ready
            print("  Step 3/3: Model ready!")
            self._update_loading_progress("Finalizing model...", 3, 3)
            self._cache_valid = True
            
            self._hide_loading()
            print("✓ Voorhees Gravity Model initialized successfully")
            
        except Exception as e:
            self._hide_loading()
            print(f"✗ Error initializing gravity model: {e}")
            raise
    
    def _update_loading_progress(self, message, current_step, total_steps):
        """Update loading progress if available"""
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for widget in app.topLevelWidgets():
                    if hasattr(widget, 'simulation_widget') and hasattr(widget.simulation_widget, 'parent'):
                        sim_tab = widget.simulation_widget.parent()
                        if hasattr(sim_tab, 'loading_spinner') and hasattr(sim_tab.loading_spinner, 'update_text'):
                            progress_percent = int((current_step / total_steps) * 100)
                            sim_tab.loading_spinner.update_text(f"{message} ({progress_percent}%)")
                            app.processEvents()  # Allow GUI updates
                            return
                        elif hasattr(widget, 'loading_spinner') and hasattr(widget.loading_spinner, 'update_text'):
                            progress_percent = int((current_step / total_steps) * 100)
                            widget.loading_spinner.update_text(f"{message} ({progress_percent}%)")
                            app.processEvents()  # Allow GUI updates
                            return
                
                # Fallback: search all widgets
                for widget in app.allWidgets():
                    if hasattr(widget, 'loading_spinner') and hasattr(widget.loading_spinner, 'update_text'):
                        progress_percent = int((current_step / total_steps) * 100)
                        widget.loading_spinner.update_text(f"{message} ({progress_percent}%)")
                        app.processEvents()  # Allow GUI updates
                        break
        except:
            pass
    
    def _show_loading(self, message):
        """Show loading spinner if available"""
        try:
            # Try to find the simulation widget with loading spinner
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                # Look for main window first
                main_window = None
                for widget in app.topLevelWidgets():
                    if hasattr(widget, 'simulation_widget') and hasattr(widget.simulation_widget, 'parent'):
                        # Found main window
                        main_window = widget
                        break
                
                if main_window and hasattr(main_window, 'simulation_widget'):
                    # Try to find loading spinner in the simulation widget's parent (simulation tab)
                    sim_tab = main_window.simulation_widget.parent()
                    if hasattr(sim_tab, 'loading_spinner'):
                        sim_tab.loading_spinner.show_spinner(message)
                        return
                    elif hasattr(main_window, 'loading_spinner'):
                        main_window.loading_spinner.show_spinner(message)
                        return
                
                # Fallback: search all widgets
                for widget in app.allWidgets():
                    if hasattr(widget, 'loading_spinner'):
                        widget.loading_spinner.show_spinner(message)
                        break
        except Exception as e:
            # Fallback: just print the message
            print(f"Loading: {message}")
    
    def _hide_loading(self):
        """Hide loading spinner if available"""
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                # Look for main window first
                main_window = None
                for widget in app.topLevelWidgets():
                    if hasattr(widget, 'simulation_widget') and hasattr(widget.simulation_widget, 'parent'):
                        # Found main window
                        main_window = widget
                        break
                
                if main_window and hasattr(main_window, 'simulation_widget'):
                    # Try to find loading spinner in the simulation widget's parent (simulation tab)
                    sim_tab = main_window.simulation_widget.parent()
                    if hasattr(sim_tab, 'loading_spinner'):
                        sim_tab.loading_spinner.hide_spinner()
                        return
                    elif hasattr(main_window, 'loading_spinner'):
                        main_window.loading_spinner.hide_spinner()
                        return
                
                # Fallback: search all widgets
                for widget in app.allWidgets():
                    if hasattr(widget, 'loading_spinner'):
                        widget.loading_spinner.hide_spinner()
                        break
        except:
            pass
    
    def _calculate_size_attraction_variables(self, graph):
        """Calculate size variables S_i and attraction variables A_j for Voorhees formula"""
        # Fast degree-based calculation
        degree_centrality = nx.degree_centrality(graph)
        
        # Use degree as primary measure, with small base value to avoid zeros
        for node in self.nodes:
            degree_score = degree_centrality.get(node, 0)
            # Add small base value to avoid zeros and ensure numerical stability
            base_value = 0.01
            
            # Size variable: generation potential (outbound trips)
            self.size_variables[node] = base_value + degree_score
            
            # Attraction variable: attractiveness for inbound trips
            self.attraction_variables[node] = base_value + degree_score * self.alpha
    
    def _compute_distances(self, graph):
        """Compute distances between nodes"""
        node_count = len(self.nodes)
        
        if node_count > 2000:
            # For large graphs, use Euclidean approximation
            print(f"    Large graph ({node_count} nodes), using Euclidean distances...")
            self._compute_euclidean_distances_simple(graph)
        else:
            # For smaller graphs, use shortest path
            print(f"    Computing shortest paths for {node_count} nodes...")
            try:
                # Simple approach - convert generator to dict
                path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
                self.distances = {}
                for source in path_lengths:
                    self.distances[source] = dict(path_lengths[source])
                print(f"    ✓ Computed {len(self.distances)} distance matrices")
            except (MemoryError, Exception) as e:
                print(f"    Memory error ({e}), falling back to Euclidean distances...")
                self._compute_euclidean_distances_simple(graph)
    
    def _compute_euclidean_distances_simple(self, graph):
        """Simple Euclidean distance computation without vectorization"""
        positions = {}
        
        # Extract or generate positions
        for node in self.nodes:
            if 'pos' in graph.nodes[node]:
                positions[node] = graph.nodes[node]['pos']
            else:
                # Generate consistent positions based on node hash
                hash_val = hash(str(node))
                positions[node] = (hash_val % 1000, (hash_val // 1000) % 1000)
        
        print(f"    Computing {len(self.nodes)}x{len(self.nodes)} distance matrix...")
        
        # Simple nested loop approach (more predictable than vectorization)
        self.distances = {}
        node_list = list(positions.keys())
        
        with tqdm(total=len(node_list), desc="    Distance computation", unit="node") as pbar:
            for i, source in enumerate(node_list):
                self.distances[source] = {}
                source_pos = positions[source]
                
                for j, target in enumerate(node_list):
                    if source == target:
                        self.distances[source][target] = 0.1  # Small non-zero value
                    else:
                        target_pos = positions[target]
                        dx = source_pos[0] - target_pos[0]
                        dy = source_pos[1] - target_pos[1]
                        dist = (dx*dx + dy*dy)**0.5  # Avoid np.sqrt for simplicity
                        self.distances[source][target] = max(dist, 0.1)
                
                pbar.update(1)
        
        print(f"    ✓ Distance computation complete")
    
    def get_source_target(self, agent_type=None, current_location=None, trip_count=0):
        """
        Get source and target using simplified Voorhees gravity model
        
        Args:
            agent_type: Type of agent (not used in basic gravity model)
            current_location: Current location of agent
            trip_count: Number of trips completed
        
        Returns:
            tuple: (source, target)
        """
        if current_location is None:
            # First trip: select source weighted by size variable
            source = self._select_weighted_source()
        else:
            source = current_location
        
        # Select target using Voorhees gravity formula
        target = self._select_gravity_target(source)
        
        return source, target
    
    def _select_weighted_source(self):
        """Select a source node weighted by its size variable (generation potential)"""
        if not self.size_variables:
            return random.choice(self.nodes)
        
        nodes = list(self.size_variables.keys())
        weights = list(self.size_variables.values())
        return random.choices(nodes, weights=weights)[0]
    
    def _select_gravity_target(self, source):
        """Select target using Voorhees gravity formula: T_ij = S_i * A_j / d_ij^β"""
        if not self.distances or source not in self.distances:
            # Fallback to random selection
            available_targets = [n for n in self.nodes if n != source]
            return random.choice(available_targets) if available_targets else source
        
        # Calculate gravity-based probabilities on demand
        target_probs = {}
        S_i = self.size_variables.get(source, 1.0)
        
        for target in self.nodes:
            if target == source:
                continue
                
            A_j = self.attraction_variables.get(target, 1.0)
            d_ij = self.distances[source].get(target, float('inf'))
            
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
        if beta is not None:
            self.beta = beta
        
        if distance_cutoff is not None:
            self.distance_cutoff = distance_cutoff
    
    @property
    def node_importance(self):
        """Provide backward compatibility for visualization - use attraction variables as importance"""
        return self.attraction_variables.copy()
    
    def get_model_info(self):
        """Return information about the Voorhees gravity model parameters"""
        return {
            'model_type': 'voorhees_gravity',
            'beta': self.beta,
            'alpha': self.alpha,
            'distance_cutoff': self.distance_cutoff,
            'num_nodes': len(self.nodes),
            'avg_size': np.mean(list(self.size_variables.values())) if self.size_variables else 0,
            'avg_attraction': np.mean(list(self.attraction_variables.values())) if self.attraction_variables else 0,
            'cache_valid': self._cache_valid
        }


class HubAndSpokeSelection(SourceTargetSelection):
    """
    Hub-and-Spoke Model for source-target selection.
    70% of trips involve major hubs (high centrality nodes).
    Reflects real-world traffic concentration at major intersections/centers.
    """
    
    def __init__(self, graph, config=None, hub_trip_probability=None, hub_percentage=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Use provided parameters first, then config, then hardcoded defaults
        self.hub_trip_probability = hub_trip_probability if hub_trip_probability is not None else getattr(self.config, 'DEFAULT_HUB_TRIP_PROBABILITY', 0.3)  # % of trips involve hubs
        self.hub_percentage = hub_percentage if hub_percentage is not None else getattr(self.config, 'DEFAULT_HUB_PERCENTAGE', 0.1)  # % of nodes considered hubs
        
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
    
    def __init__(self, graph, config=None, agent_type_distributions=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Use provided agent type distributions or defaults
        self.agent_type_distributions = agent_type_distributions if agent_type_distributions is not None else getattr(self.config, 'DEFAULT_AGENT_TYPE_DISTRIBUTIONS', {
            'commuter': 0.4, 'leisure': 0.3, 'business': 0.2, 'delivery': 0.1
        })
        
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
        """Return configured distribution of agent types"""
        return self.agent_type_distributions
    
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
    
    def __init__(self, graph, config=None, intra_zone_probability=None):
        super().__init__(graph)
        
        # Use provided config or fall back to default
        self.config = config if config is not None else MODELS
        
        # Use provided parameters first, then config, then hardcoded defaults
        self.intra_zone_probability = intra_zone_probability if intra_zone_probability is not None else getattr(self.config, 'DEFAULT_ZONE_INTRA_PROBABILITY', 0.7)
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
