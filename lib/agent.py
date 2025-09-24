import random
import networkx as nx
import numpy as np
from collections import deque

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIMULATION

class Agent:
    _next_id = 1  # Class variable to generate unique IDs
    
    def __init__(self, graph, agent_type='random', st_selector=None, traffic_manager=None):
        self.id = Agent._next_id  # Assign unique ID
        Agent._next_id += 1
        
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.waiting = False
        self.wait_time = 0
        self.state = 'waiting'  # Start in waiting state
        self.agent_type = agent_type
        self.st_selector = st_selector
        self.traffic_manager = traffic_manager
        self.trip_count = 0
        self.current_edge_key = None  # Track current edge for traffic management
        
        # Position interpolation for smooth movement
        self.position_history = deque(maxlen=3)  # Keep last 3 positions for smoothing
        self.last_update_time = 0
        self.velocity = np.array([0.0, 0.0])  # Current velocity vector
        
        # Set initial position to a random node
        initial_node = random.choice(self.nodes)
        self.position = np.array(self.graph.nodes[initial_node]['pos'])
        self.position_history.append((self.position.copy(), 0))  # (position, timestamp)
        self.source = initial_node
        self.target = initial_node

    def start_new_journey(self):
        current_location = getattr(self, 'target', None)
        
        if self.st_selector:
            # Use source-target selector
            self.source, self.target = self.st_selector.get_source_target(
                self.agent_type, current_location, self.trip_count
            )
            
            # For hub-and-spoke selection, update agent type based on trip type
            if hasattr(self.st_selector, 'last_trip_type') and self.st_selector.__class__.__name__ == 'HubAndSpokeSelection':
                self.agent_type = self.st_selector.last_trip_type
            
            # For zone-based selection, update agent type based on trip type
            elif hasattr(self.st_selector, 'node_to_zone') and self.st_selector.__class__.__name__ == 'ZoneBasedSelection':
                source_zone = self.st_selector.node_to_zone.get(self.source)
                target_zone = self.st_selector.node_to_zone.get(self.target)
                
                if source_zone is not None and target_zone is not None:
                    if source_zone == target_zone:
                        self.agent_type = 'intra_zone'  # Same zone trip
                    else:
                        self.agent_type = 'inter_zone'  # Different zone trip
                # If zones can't be determined, keep original agent_type
        else:
            # Fallback to original random selection
            if not hasattr(self, 'source'):
                # First journey, start from random point
                self.source = random.choice(self.nodes)
            else:
                # Start from last target
                self.source = self.target
                
            self.target = random.choice(self.nodes)
            while self.target == self.source:  # Ensure different target
                self.target = random.choice(self.nodes)
        
        self.trip_count += 1
            
        try:
            self.path = nx.shortest_path(self.graph, self.source, self.target, weight='travel_time')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.start_new_journey()
            return

        self.path_index = 0
        self.position = np.array(self.graph.nodes[self.source]['pos'])
        self.distance_on_edge = 0.0
        self.setup_current_edge()

    def setup_current_edge(self):
        if self.path_index >= len(self.path) - 1:
            # Unregister from current edge before stopping
            if self.current_edge_key and self.traffic_manager:
                u, v, key = self.current_edge_key
                self.traffic_manager.unregister_agent_from_edge(self, u, v, key)
                self.current_edge_key = None
            
            self.state = 'waiting'
            self.wait_time = 0
            return

        u = self.path[self.path_index]
        v = self.path[self.path_index + 1]
        
        # Find the best edge (e.g., with the lowest travel_time)
        edge_data = self.graph.get_edge_data(u, v)
        best_edge = min(edge_data.values(), key=lambda x: x.get('travel_time', float('inf')))
        
        # Get the key for this specific edge
        edge_key = 0  # Default key
        for key, data in edge_data.items():
            if data == best_edge:
                edge_key = key
                break
        
        # Unregister from previous edge
        if self.current_edge_key and self.traffic_manager:
            old_u, old_v, old_key = self.current_edge_key
            self.traffic_manager.unregister_agent_from_edge(self, old_u, old_v, old_key)
        
        # Register on new edge
        if self.traffic_manager:
            self.traffic_manager.register_agent_on_edge(self, u, v, edge_key)
            self.current_edge_key = (u, v, edge_key)
        
        self.current_edge = best_edge
        
        # Handle potential string values in edge data
        try:
            self.edge_length = float(best_edge.get('length', SIMULATION.DEFAULT_EDGE_LENGTH))
            base_speed_kph = float(best_edge.get('speed_kph', SIMULATION.DEFAULT_SPEED_KPH))
        except (TypeError, ValueError):
            # Fallback to defaults if conversion fails
            self.edge_length = SIMULATION.DEFAULT_EDGE_LENGTH
            base_speed_kph = SIMULATION.DEFAULT_SPEED_KPH
        
        # Apply traffic congestion to speed
        if self.traffic_manager:
            congestion_factor = self.traffic_manager.get_congested_speed(self, u, v, edge_key)
            effective_speed_kph = base_speed_kph * congestion_factor
        else:
            effective_speed_kph = base_speed_kph
        
        self.speed = effective_speed_kph * SIMULATION.KPH_TO_MPS_FACTOR  # m/s

        start_pos = np.array(self.graph.nodes[u]['pos'])
        end_pos = np.array(self.graph.nodes[v]['pos'])
        self.edge_vector = end_pos - start_pos
        # Normalize edge_vector, handle zero length case
        if self.edge_length > 0:
            self.edge_direction = self.edge_vector / self.edge_length
        else:
            self.edge_direction = np.array([0,0])

    def update(self, dt, simulation_time=0):
        if self.state == 'waiting':
            self.wait_time += dt
            
            # Check if agent should start a new trip based on time-dependent probability
            travel_prob = self.get_travel_probability(simulation_time)
            
            # Use probability to decide whether to start traveling
            # Higher probability during peak hours, lower at night
            if random.random() < travel_prob * dt / SIMULATION.SECONDS_PER_HOUR:  # Scale by time step
                self.state = 'moving'
                self.start_new_journey()
                return
            
            # Minimum wait time before considering another trip
            if self.wait_time < SIMULATION.MIN_WAIT_TIME:
                return
                
        elif self.state == 'moving':
            if not hasattr(self, 'speed'):
                self.start_new_journey()
                return
            
            # Store previous position for smooth interpolation
            prev_position = self.position.copy()
            
            distance_to_move = self.speed * dt
            self.distance_on_edge += distance_to_move

            if self.distance_on_edge >= self.edge_length:
                self.path_index += 1
                self.distance_on_edge = 0
                self.setup_current_edge()

            if hasattr(self, 'path') and self.path_index < len(self.path):
                u = self.path[self.path_index]
                start_pos = np.array(self.graph.nodes[u]['pos'])
                new_position = start_pos + self.edge_direction * self.distance_on_edge
                
                # Calculate velocity for interpolation
                if dt > 0:
                    self.velocity = (new_position - prev_position) / dt
                else:
                    self.velocity = np.array([0.0, 0.0])
                
                # Update position with smooth interpolation
                self.position = new_position
                
                # Store position in history for potential future interpolation
                self.position_history.append((self.position.copy(), simulation_time))
                self.last_update_time = simulation_time
    
    def get_interpolated_position(self, current_time=None):
        """Get smoothly interpolated position based on velocity and time"""
        if current_time is None or len(self.position_history) == 0:
            return self.position
        
        # If we're waiting or just started, return current position
        if self.state == 'waiting' or len(self.position_history) < 2:
            return self.position
        
        # For moving agents, use velocity-based interpolation
        time_since_update = current_time - self.last_update_time
        if time_since_update > 0 and hasattr(self, 'velocity'):
            # Extrapolate position based on velocity (with reasonable limits)
            max_extrapolation_time = 0.5  # Don't extrapolate more than 0.5 seconds
            safe_time = min(time_since_update, max_extrapolation_time)
            interpolated_pos = self.position + self.velocity * safe_time
            return interpolated_pos
        
        return self.position

    def get_travel_probability(self, simulation_time):
        """
        Calculate probability of starting a trip based on time of day
        Returns probability between 0 and 1
        Based on real departure time distribution
        """
        # Convert simulation time to display time (add hours to start at configured time)
        # This matches the time shown in the simulation window
        display_time = simulation_time + SIMULATION.SIMULATION_START_OFFSET  # seconds offset
        hour_of_day = (display_time / SIMULATION.SECONDS_PER_HOUR) % 24
        
        # Define travel probability curve based on the provided distribution
        # Significantly increased all probabilities to make agents much more active
        if hour_of_day < 1:
            return 0.12  # Increased from 0.04
        elif 1 <= hour_of_day < 2:
            return 0.08  # Increased from 0.02
        elif 2 <= hour_of_day < 3:
            return 0.08  # Increased from 0.02
        elif 3 <= hour_of_day < 4:
            return 0.08  # Increased from 0.02
        elif 4 <= hour_of_day < 5:
            return 0.12  # Increased from 0.03
        elif 5 <= hour_of_day < 6:
            return 0.25  # Increased from 0.08
        elif 6 <= hour_of_day < 7:
            return 0.45  # Increased from 0.22
        elif 7 <= hour_of_day < 8:
            return 0.80  # Increased from 0.50 - Morning peak start
        elif 8 <= hour_of_day < 9:
            return 0.95  # Increased from 0.65 - Morning peak
        elif 9 <= hour_of_day < 10:
            return 0.65  # Increased from 0.35
        elif 10 <= hour_of_day < 11:
            return 0.60  # Increased from 0.30
        elif 11 <= hour_of_day < 12:
            return 0.65  # Increased from 0.35
        elif 12 <= hour_of_day < 13:
            return 0.75  # Increased from 0.42 - Lunch peak
        elif 13 <= hour_of_day < 14:
            return 0.65  # Increased from 0.35
        elif 14 <= hour_of_day < 15:
            return 0.62  # Increased from 0.32
        elif 15 <= hour_of_day < 16:
            return 0.65  # Increased from 0.35
        elif 16 <= hour_of_day < 17:
            return 0.75  # Increased from 0.42
        elif 17 <= hour_of_day < 18:
            return 0.90  # Increased from 0.55 - Evening peak
        elif 18 <= hour_of_day < 19:
            return 0.85  # Increased from 0.50 - Evening peak continuation
        elif 19 <= hour_of_day < 20:
            return 0.65  # Increased from 0.35
        elif 20 <= hour_of_day < 21:
            return 0.50  # Increased from 0.22
        elif 21 <= hour_of_day < 22:
            return 0.35  # Increased from 0.15
        elif 22 <= hour_of_day < 23:
            return 0.25  # Increased from 0.10
        elif 23 <= hour_of_day < 24:
            return 0.18  # Increased from 0.06
        else:
            return 0.08  # Default fallback
