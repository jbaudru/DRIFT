"""
Status Manager for the Traffic Simulation Application

Handles all status-related functionality including:
- Status bar updates
- Performance monitoring
- UI status label management
- Real-time status tracking
"""

import datetime
from config import UI, COLORS


class StatusManager:
    """Manager for handling status updates and performance monitoring"""
    
    def __init__(self, main_window):
        """Initialize the status manager
        
        Args:
            main_window: Reference to the main window instance
        """
        self.main_window = main_window
        self.simulation_start_time = None
        
    def set_simulation_start_time(self, start_time=None):
        """Set the simulation start time
        
        Args:
            start_time: datetime object, if None uses current time
        """
        if start_time is None:
            self.simulation_start_time = datetime.datetime.now()
        else:
            self.simulation_start_time = start_time
    
    def clear_simulation_start_time(self):
        """Clear the simulation start time"""
        self.simulation_start_time = None
    
    def reset_status_labels(self):
        """Reset all status labels to their default values"""
        if hasattr(self.main_window, 'sim_time_label'):
            self.main_window.sim_time_label.setText(UI.SIM_TIME_TEMPLATE.format(UI.DEFAULT_SIM_TIME))
        if hasattr(self.main_window, 'real_time_label'):
            self.main_window.real_time_label.setText(UI.REAL_TIME_TEMPLATE.format(UI.DEFAULT_REAL_TIME))
        if hasattr(self.main_window, 'moving_agents_label'):
            self.main_window.moving_agents_label.setText(UI.MOVING_AGENTS_TEMPLATE.format(UI.DEFAULT_MOVING_AGENTS))
        if hasattr(self.main_window, 'network_util_label'):
            self.main_window.network_util_label.setText(UI.NETWORK_UTIL_TEMPLATE.format(UI.DEFAULT_NETWORK_UTIL))
    
    def update_status(self, status_info):
        """Update the status bar with simulation information
        
        Args:
            status_info: Dictionary containing status information
        """
        # Update trip manager's current simulation time
        if hasattr(self.main_window, 'trip_manager'):
            self.main_window.trip_manager.update_current_simulation_time(status_info['simulation_time'])
        
        # Update simulation time (starting at 6 AM)
        self._update_simulation_time(status_info['simulation_time'])
        
        # Update real time elapsed
        self._update_real_time()
        
        # Update moving agents
        self._update_moving_agents(status_info['moving_agents'], status_info['total_agents'])
        
        # Update network utilization
        self._update_network_utilization(status_info['network_utilization'])
        
        # Update performance status
        self.update_performance_status()
        
        # Update statistics for real-time plots
        if hasattr(self.main_window, 'statistics_manager'):
            self.main_window.statistics_manager.update_statistics(status_info)
    
    def _update_simulation_time(self, simulation_time):
        """Update the simulation time display
        
        Args:
            simulation_time: Current simulation time in seconds
        """
        if hasattr(self.main_window, 'sim_time_label'):
            sim_time_seconds = simulation_time + 21600  # Add 6 hours (start at 6 AM)
            sim_hours = int(sim_time_seconds / 3600) % 24
            sim_minutes = int((sim_time_seconds % 3600) / 60)
            sim_seconds = int(sim_time_seconds % 60)
            self.main_window.sim_time_label.setText(f"Simulation Time: {sim_hours:02d}:{sim_minutes:02d}:{sim_seconds:02d}")
    
    def _update_real_time(self):
        """Update the real time elapsed display"""
        if hasattr(self.main_window, 'real_time_label') and self.simulation_start_time:
            elapsed = datetime.datetime.now() - self.simulation_start_time
            elapsed_seconds = elapsed.total_seconds()
            real_hours = int(elapsed_seconds / 3600)
            real_minutes = int((elapsed_seconds % 3600) / 60)
            real_secs = int(elapsed_seconds % 60)
            self.main_window.real_time_label.setText(f"Real Time: {real_hours:02d}:{real_minutes:02d}:{real_secs:02d}")
    
    def _update_moving_agents(self, moving_agents, total_agents):
        """Update the moving agents display
        
        Args:
            moving_agents: Number of currently moving agents
            total_agents: Total number of agents
        """
        if hasattr(self.main_window, 'moving_agents_label'):
            self.main_window.moving_agents_label.setText(f"Moving Agents: {moving_agents}/{total_agents}")
    
    def _update_network_utilization(self, network_utilization):
        """Update the network utilization display
        
        Args:
            network_utilization: Network utilization as a float (0.0 to 1.0)
        """
        if hasattr(self.main_window, 'network_util_label'):
            utilization = network_utilization * 100
            self.main_window.network_util_label.setText(f"Network Utilization: {utilization:.3f}%")
    
    def update_performance_status(self):
        """Update performance optimization status"""
        if not hasattr(self.main_window, 'performance_label'):
            return
            
        if hasattr(self.main_window, 'simulation_widget') and hasattr(self.main_window.simulation_widget, 'get_performance_info'):
            perf_info = self.main_window.simulation_widget.get_performance_info()
            
            if "status" in perf_info and perf_info["status"] == "No graph loaded":
                self.main_window.performance_label.setText("Performance: No Graph")
                return
            
            # Determine performance status with thread optimization indicator
            total_nodes = perf_info.get("total_nodes", 0)
            use_lod = perf_info.get("use_lod", False)
            zoom = perf_info.get("zoom_factor", 1.0)
            
            # Check if simulation is running in optimized thread mode
            thread_optimized = (hasattr(self.main_window, 'simulation_thread') and 
                              self.main_window.simulation_thread and 
                              self.main_window.simulation_thread.running)
            
            # Determine status and color based on node count and LOD usage
            if total_nodes > 10000:
                status = "LOD+" if use_lod else "Heavy"
                color = COLORS.PERFORMANCE_COLORS['fair'] if use_lod else COLORS.PERFORMANCE_COLORS['critical']
            elif total_nodes > 5000:
                status = "LOD" if use_lod else "Medium"
                color = COLORS.PERFORMANCE_COLORS['good'] if use_lod else COLORS.PERFORMANCE_COLORS['poor']
            else:
                status = "Full"
                color = COLORS.PERFORMANCE_COLORS['excellent']
            
            # Add thread optimization indicator
            thread_indicator = " [T]" if thread_optimized else ""
            
            # Update the performance label
            self.main_window.performance_label.setText(f"Performance: {status}{thread_indicator} ({total_nodes:,} nodes, {zoom:.1f}x)")
            self.main_window.performance_label.setStyleSheet(f"color: {color};")
    
    def update_final_status(self):
        """Update status labels to show final state after simulation completion"""
        if hasattr(self.main_window, 'moving_agents_label'):
            self.main_window.moving_agents_label.setText("Moving Agents: 0")