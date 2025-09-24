"""
Trip Manager Module
Handles trip data management, table updates, and trip-related operations
"""

from PyQt5.QtWidgets import QTableWidgetItem


class TripManager:
    """Handles trip data and table management"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.completed_trips = []
        self.current_simulation_time = 0.0
        self.last_plot_update_trip_count = 0
        
    def clear_trips(self):
        """Clear all trip data"""
        self.completed_trips = []
        self.current_simulation_time = 0.0
        self.last_plot_update_trip_count = 0
        if hasattr(self.main_window, 'trips_table'):
            self.main_window.trips_table.setRowCount(0)
    
    def add_completed_trip(self, trip_info):
        """Add a completed trip to the trips table"""
        # Add completion time based on current simulation time
        # Get current simulation time from the status_info if available, otherwise estimate
        if hasattr(self, 'current_simulation_time'):
            trip_info['completion_time'] = self.current_simulation_time
        else:
            # Fallback: estimate from start_time and duration
            try:
                start_time_parts = trip_info.get('start_time', '06:00:00').split(':')
                start_seconds = int(start_time_parts[0]) * 3600 + int(start_time_parts[1]) * 60 + int(start_time_parts[2])
                duration_seconds = trip_info.get('duration', 0)
                trip_info['completion_time'] = start_seconds + duration_seconds
            except:
                trip_info['completion_time'] = 0
        
        self.completed_trips.append(trip_info)
        self._add_trip_to_table(trip_info)
    
    def _add_trip_to_table(self, trip_info):
        """Add a trip to the trips table widget"""
        if not hasattr(self.main_window, 'trips_table'):
            return
            
        row_count = self.main_window.trips_table.rowCount()
        self.main_window.trips_table.insertRow(row_count)
        
        # Format start time - it's already a timestamp string (HH:MM:SS)
        try:
            start_time_str = str(trip_info['start_time'])
            # Validate it's a proper time format, otherwise use default
            if ':' not in start_time_str:
                start_time_str = "06:00:00"
        except (ValueError, TypeError):
            start_time_str = "06:00:00"  # Default start time
        
        # Format numeric values safely
        avg_speed = self._safe_float(trip_info.get('avg_speed', 0)) * 3.6  # m/s to km/h
        duration = self._safe_float(trip_info.get('duration', 0))
        distance = self._safe_float(trip_info.get('distance', 0))
        
        # Add data to table (10 columns now)
        self.main_window.trips_table.setItem(row_count, 0, QTableWidgetItem(str(trip_info['trip_id'])))
        self.main_window.trips_table.setItem(row_count, 1, QTableWidgetItem(str(trip_info.get('agent_id', 'N/A'))))
        self.main_window.trips_table.setItem(row_count, 2, QTableWidgetItem(str(trip_info['agent_type'])))
        self.main_window.trips_table.setItem(row_count, 3, QTableWidgetItem(str(trip_info.get('start_node', 'Unknown'))))
        self.main_window.trips_table.setItem(row_count, 4, QTableWidgetItem(str(trip_info.get('end_node', 'Unknown'))))
        self.main_window.trips_table.setItem(row_count, 5, QTableWidgetItem(start_time_str))
        self.main_window.trips_table.setItem(row_count, 6, QTableWidgetItem(f"{duration:.1f}"))
        self.main_window.trips_table.setItem(row_count, 7, QTableWidgetItem(f"{distance:.1f}"))
        self.main_window.trips_table.setItem(row_count, 8, QTableWidgetItem(f"{avg_speed:.1f}"))
        
        # Format path nodes as a string
        path_nodes = trip_info.get('path_nodes', [])
        if isinstance(path_nodes, list):
            path_str = " -> ".join(str(node) for node in path_nodes) if path_nodes else "No path"
        else:
            path_str = str(path_nodes)
        self.main_window.trips_table.setItem(row_count, 9, QTableWidgetItem(path_str))
        
        # Auto-scroll to bottom
        self.main_window.trips_table.scrollToBottom()
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            return float(value) if value else 0
        except (ValueError, TypeError):
            return 0
    
    def get_completed_trips(self):
        """Get list of completed trips"""
        return self.completed_trips
    
    def get_trips_count(self):
        """Get the number of completed trips"""
        return len(self.completed_trips)
    
    def update_current_simulation_time(self, simulation_time):
        """Update the current simulation time for trip completion tracking"""
        self.current_simulation_time = simulation_time
    
    def get_current_simulation_time(self):
        """Get the current simulation time"""
        return self.current_simulation_time
    
    def update_last_plot_trip_count(self, count):
        """Update the last plot update trip count"""
        self.last_plot_update_trip_count = count
    
    def get_last_plot_trip_count(self):
        """Get the last plot update trip count"""
        return self.last_plot_update_trip_count
    
    def has_new_trips_since_last_plot(self):
        """Check if there are new trips since the last plot update"""
        return len(self.completed_trips) > self.last_plot_update_trip_count