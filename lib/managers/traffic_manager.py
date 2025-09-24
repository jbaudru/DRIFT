from collections import defaultdict

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NETWORK, SIMULATION


class TrafficManager:
    """
    Manages traffic congestion and speed calculations for all agents on the network
    Uses fundamental diagram of traffic flow and BPR (Bureau of Public Roads) function
    """
    
    def __init__(self, graph):
        self.graph = graph
        # Track agents on each edge: {(u, v, key): [agent1, agent2, ...]}
        self.edge_agents = defaultdict(list)
        
        # Calculate edge capacities based on road attributes
        self._calculate_edge_capacities()
        
        # BPR function parameters (can be modified via settings)
        self.bpr_alpha = NETWORK.DEFAULT_BPR_ALPHA
        self.bpr_beta = NETWORK.DEFAULT_BPR_BETA
    
    def set_bpr_parameters(self, alpha, beta):
        """Set Bureau of Public Roads function parameters"""
        self.bpr_alpha = alpha
        self.bpr_beta = beta
    
    def _calculate_edge_capacities(self):
        """Calculate theoretical capacity for each edge based on road characteristics"""
        self.edge_capacities = {}
        
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            # Base capacity calculation (vehicles per hour)
            # Default values if not specified in data, with type conversion
            try:
                lanes = float(data.get('lanes', 1))
                speed_kph = float(data.get('speed_kph', SIMULATION.DEFAULT_SPEED_KPH))
                length_m = float(data.get('length', SIMULATION.DEFAULT_EDGE_LENGTH))
            except (TypeError, ValueError):
                # Fallback to defaults if conversion fails
                lanes = 1
                speed_kph = SIMULATION.DEFAULT_SPEED_KPH
                length_m = SIMULATION.DEFAULT_EDGE_LENGTH
            
            # Theoretical capacity using highway capacity manual principles
            # Base capacity per lane: ~2000 vehicles/hour for urban roads
            base_capacity_per_lane = 2000 if speed_kph >= 50 else 1500
            theoretical_capacity = int(lanes * base_capacity_per_lane)
            
            # Adjust for road length (longer roads can hold more vehicles)
            # Assume average vehicle length + following distance = 7.5m
            vehicle_space = 7.5
            max_vehicles_on_road = max(1, int(length_m / vehicle_space))
            
            # Final capacity is minimum of flow capacity and storage capacity
            self.edge_capacities[(u, v, key)] = min(theoretical_capacity, max_vehicles_on_road * 4)
    
    def register_agent_on_edge(self, agent, u, v, key=0):
        """Register an agent as traveling on a specific edge (only if moving)"""
        edge_key = (u, v, key)
        # Only register moving agents for congestion calculations
        if agent.state == 'moving' and agent not in self.edge_agents[edge_key]:
            self.edge_agents[edge_key].append(agent)
    
    def unregister_agent_from_edge(self, agent, u, v, key=0):
        """Remove an agent from an edge when they finish traversing it"""
        edge_key = (u, v, key)
        if agent in self.edge_agents[edge_key]:
            self.edge_agents[edge_key].remove(agent)
    
    def get_congested_speed(self, agent, u, v, key=0):
        """
        Calculate speed reduction due to congestion using BPR function
        Only counts moving agents for congestion calculations
        Returns multiplier (0-1) to apply to free-flow speed
        """
        edge_key = (u, v, key)
        
        # Count only moving agents on this edge
        moving_agents_count = sum(1 for a in self.edge_agents[edge_key] if a.state == 'moving')
        capacity = self.edge_capacities.get(edge_key, 10)
        
        if moving_agents_count == 0:
            return 1.0  # No congestion
        
        # Volume-to-capacity ratio based on moving agents only
        v_c_ratio = moving_agents_count / capacity
        
        # BPR (Bureau of Public Roads) function for travel time
        # t = t0 * (1 + α * (V/C)^β)
        # Use configurable values instead of hardcoded ones
        alpha = self.bpr_alpha
        beta = self.bpr_beta
        
        # Travel time multiplier
        time_multiplier = 1 + alpha * (v_c_ratio ** beta)
        
        # Speed is inversely proportional to travel time
        speed_multiplier = 1 / time_multiplier
        
        # Ensure speed doesn't go below 10% of free-flow speed
        return max(0.1, speed_multiplier)
    
    def get_edge_congestion_level(self, u, v, key=0):
        """Get congestion level for visualization (0-1, where 1 is maximum congestion)
        Only counts moving agents"""
        edge_key = (u, v, key)
        moving_agents_count = sum(1 for a in self.edge_agents[edge_key] if a.state == 'moving')
        capacity = self.edge_capacities.get(edge_key, 10)
        
        return min(1.0, moving_agents_count / capacity)
    
    def clear_edge_agents(self):
        """Clear all edge agent tracking (call this each update if needed)"""
        self.edge_agents.clear()
    
    def get_network_statistics(self):
        """Get overall network congestion statistics (only counting moving agents)"""
        # Count only moving agents for accurate statistics
        total_moving_agents = sum(
            sum(1 for agent in agents if agent.state == 'moving') 
            for agents in self.edge_agents.values()
        )
        total_capacity = sum(self.edge_capacities.values())
        
        # Count edges with moving agents
        congested_edges = sum(
            1 for agents in self.edge_agents.values() 
            if any(agent.state == 'moving' for agent in agents)
        )
        total_edges = len(self.edge_capacities)
        
        return {
            'total_agents_on_roads': total_moving_agents,
            'total_network_capacity': total_capacity,
            'network_utilization': total_moving_agents / total_capacity if total_capacity > 0 else 0,
            'congested_edges': congested_edges,
            'total_edges': total_edges,
            'congestion_ratio': congested_edges / total_edges if total_edges > 0 else 0
        }
