from PyQt5.QtWidgets import QFileDialog
from matplotlib.figure import Figure
from matplotlib import ticker
from collections import defaultdict, deque
import datetime
import os
from config import STATISTICS, COLORS


class StatisticsManager:
    """Handles statistics collection and plotting"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.stats_history = self._initialize_stats_history()
        self.last_plot_update_trip_count = 0
        # Rolling window size for more stable statistics (number of completed trips to average)
        self.rolling_window_size = 50  # Average over last 50 completed trips
        
    def _initialize_stats_history(self):
        """Initialize the statistics history data structure"""
        return {
            'time': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'moving_agents': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'utilization': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'avg_speed': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'avg_trip_distance': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'avg_trip_duration': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'avg_nodes_per_trip': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'agent_types': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN),
            'trip_count': deque(maxlen=STATISTICS.STATS_HISTORY_MAXLEN)
        }
    
    def format_time_axis(self, axis, times):
        """Format the time axis with appropriate labels and finer granularity"""
        if not times:
            return
        
        # Calculate time range for adaptive tick intervals
        time_range = max(times) - min(times) if len(times) > 1 else 1
        
        # Set the ticks and labels with finer granularity
        axis.set_xlabel('Time (hours)', fontsize=8)
        
        # Format tick labels with adaptive precision based on time range
        
        # Choose tick interval based on time range for better granularity
        if time_range <= STATISTICS.TIME_RANGE_THRESHOLD_30MIN:
            # For very short simulations, show 5-minute intervals
            tick_interval = STATISTICS.TICK_INTERVAL_5MIN
            formatter = ticker.FuncFormatter(lambda x, p: f'{x:.2f}')
        elif time_range <= STATISTICS.TIME_RANGE_THRESHOLD_2H:
            # For short simulations, show 15-minute intervals  
            tick_interval = STATISTICS.TICK_INTERVAL_15MIN
            formatter = ticker.FuncFormatter(lambda x, p: f'{x:.2f}')
        elif time_range <= STATISTICS.TIME_RANGE_THRESHOLD_4H:
            # For medium simulations, show 30-minute intervals
            tick_interval = STATISTICS.TICK_INTERVAL_30MIN
            formatter = ticker.FuncFormatter(lambda x, p: f'{x:.1f}')
        elif time_range <= STATISTICS.TIME_RANGE_THRESHOLD_12H:
            # For longer simulations, show 1-hour intervals
            tick_interval = STATISTICS.TICK_INTERVAL_1HOUR
            formatter = ticker.FuncFormatter(lambda x, p: f'{x:.1f}')
        else:
            # For very long simulations, show 2-hour intervals
            tick_interval = STATISTICS.TICK_INTERVAL_2HOUR
            formatter = ticker.FuncFormatter(lambda x, p: f'{x:.0f}')
        
        # Set major ticks at regular intervals
        axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        axis.xaxis.set_major_formatter(formatter)
        
        # Set minor ticks for finer granularity
        if tick_interval >= 1.0:
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(tick_interval / 4))
        else:
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(tick_interval / 2))
    
    def update_statistics(self, status_info):
        """Update statistics data and refresh plots"""
        # Store current simulation time for trip completion tracking
        self.main_window.current_simulation_time = status_info['simulation_time']

        # Convert simulation time to hours for plotting (starting at 6 AM)
        sim_time_hours = (status_info['simulation_time'] + 21600) / 3600.0

        # Add current data point
        self.stats_history['time'].append(sim_time_hours)
        self.stats_history['moving_agents'].append(status_info['moving_agents'])
        self.stats_history['utilization'].append(status_info['network_utilization'] * 100)

        # Calculate real-time average speed from currently moving agents
        avg_speed = self._calculate_current_average_speed()
        
        # Calculate trip-based statistics from completed trips (since last update)
        avg_distance, avg_duration, avg_nodes_per_trip, total_trip_count = self._calculate_trip_statistics_since_last_update()

        # --- NEW: Count agent types from actual moving agents ---
        agent_type_counts = {}
        agent_list = []
        # Try to get the agent list from the main window or simulation thread
        if hasattr(self.main_window, 'simulation_thread') and hasattr(self.main_window.simulation_thread, 'agents'):
            agent_list = self.main_window.simulation_thread.agents
        elif hasattr(self.main_window, 'agents'):
            agent_list = self.main_window.agents

        for agent in agent_list:
            if getattr(agent, 'state', None) == 'moving':
                agent_type = getattr(agent, 'agent_type', 'unknown')
                agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        # Debug: Log agent type counts and moving agent sum occasionally
        if len(self.stats_history['time']) % 20 == 0:
            self.main_window.add_log_message(f"Debug: Agent type counts: {agent_type_counts} (sum={sum(agent_type_counts.values())}, moving_agents={status_info['moving_agents']})")

        self.stats_history['avg_speed'].append(avg_speed)
        self.stats_history['avg_trip_distance'].append(avg_distance)
        self.stats_history['avg_trip_duration'].append(avg_duration)
        self.stats_history['avg_nodes_per_trip'].append(avg_nodes_per_trip)
        self.stats_history['agent_types'].append(agent_type_counts)
        self.stats_history['trip_count'].append(total_trip_count)

        # Update plots (only if we have data and plots are initialized)
        if hasattr(self.main_window, 'stats_axes') and len(self.stats_history['time']) > 0:
            self.update_plots()
    
    def _calculate_current_average_speed(self):
        """Calculate the current average speed from all moving agents
        This provides real-time speed reflecting current congestion"""
        
        # Get the list of agents from the simulation thread
        agent_list = []
        if hasattr(self.main_window, 'simulation_thread') and hasattr(self.main_window.simulation_thread, 'agents'):
            agent_list = self.main_window.simulation_thread.agents
        elif hasattr(self.main_window, 'agents'):
            agent_list = self.main_window.agents
        
        if not agent_list:
            return 0
        
        # Collect speeds from all currently moving agents
        moving_speeds = []
        for agent in agent_list:
            if getattr(agent, 'state', None) == 'moving':
                speed_ms = getattr(agent, 'speed', 0)  # Speed in m/s
                if speed_ms > 0:
                    moving_speeds.append(speed_ms * 3.6)  # Convert m/s to km/h
        
        # Calculate average
        if moving_speeds:
            avg_speed = sum(moving_speeds) / len(moving_speeds)
            return avg_speed
        else:
            return 0
    
    def _calculate_trip_statistics_rolling_window(self):
        """Calculate statistics from a rolling window of recently completed trips
        This provides more stable metrics compared to 'since last update' approach"""
        
        total_trip_count = len(self.main_window.completed_trips)
        
        if total_trip_count == 0:
            return 0, 0, 0, 0, 0
        
        # Use rolling window: take last N trips (or all trips if fewer than N)
        window_size = min(self.rolling_window_size, total_trip_count)
        recent_trips = self.main_window.completed_trips[-window_size:]
        
        if not recent_trips:
            return 0, 0, 0, 0, total_trip_count
        
        # Calculate averages from recent trips in the window with validation
        speeds = []
        distances = []
        
        # Debug: Log first few trips occasionally to diagnose the issue
        if total_trip_count <= 5 or (total_trip_count % 100 == 0 and total_trip_count <= 500):
            self.main_window.add_log_message(f"DEBUG Trip Stats - Total trips: {total_trip_count}, Window size: {window_size}")
            for i, trip in enumerate(recent_trips[:3]):  # Log first 3 trips in window
                self.main_window.add_log_message(
                    f"  Trip {i}: speed={trip.get('avg_speed', 0):.2f} m/s, "
                    f"distance={trip.get('distance', 0):.1f} m, "
                    f"recorded_duration={trip.get('duration', 0):.1f} s, "
                    f"nodes={len(trip.get('path_nodes', []))}"
                )
        
        for trip in recent_trips:
            # Validate and collect speed data
            speed = trip.get('avg_speed', 0)
            if speed > 0 and speed < 100:  # Filter unrealistic speeds (> 360 km/h)
                speeds.append(speed * 3.6)  # Convert m/s to km/h
            
            # Validate and collect distance data
            distance = trip.get('distance', 0)
            if distance > 0 and distance < 1000000:  # Filter unrealistic distances (> 1000 km)
                distances.append(distance / 1000)  # Convert m to km
        
        # Calculate average nodes per trip
        nodes_per_trip = []
        for trip in recent_trips:
            path_nodes = trip.get('path_nodes', [])
            if isinstance(path_nodes, list) and len(path_nodes) > 0:
                nodes_per_trip.append(len(path_nodes))
        
        # Calculate averages for speed and distance first
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        avg_distance = sum(distances) / len(distances) if distances else 0
        avg_nodes_per_trip = sum(nodes_per_trip) / len(nodes_per_trip) if nodes_per_trip else 0
        
        # CORRECTED: Calculate duration from the computed average speed and distance
        # This ensures perfect consistency: duration = distance / speed
        if avg_speed > 0 and avg_distance > 0:
            # Convert avg_speed from km/h to m/s, avg_distance from km to m
            avg_speed_ms = avg_speed / 3.6  # km/h to m/s
            avg_distance_m = avg_distance * 1000  # km to m
            
            # Calculate duration: time = distance / speed
            avg_duration_s = avg_distance_m / avg_speed_ms  # seconds
            avg_duration = avg_duration_s / 60  # Convert to minutes
            
            # Debug: Show the calculation
            if total_trip_count <= 10 or (total_trip_count % 100 == 0 and total_trip_count <= 500):
                self.main_window.add_log_message(
                    f"  Computed duration: {avg_duration:.2f} min from "
                    f"avg_distance={avg_distance:.3f} km, avg_speed={avg_speed:.2f} km/h"
                )
        else:
            avg_duration = 0
        
        # Debug: Log computed averages occasionally
        if total_trip_count % 100 == 0 and total_trip_count <= 500:
            self.main_window.add_log_message(
                f"DEBUG Averages - Speed: {avg_speed:.1f} km/h, "
                f"Distance: {avg_distance:.2f} km, "
                f"Duration: {avg_duration:.1f} min (computed from speed & distance), "
                f"Valid samples: speed={len(speeds)}, dist={len(distances)}"
            )
        
        return avg_speed, avg_distance, avg_duration, avg_nodes_per_trip, total_trip_count
    
    def _calculate_trip_statistics_since_last_update(self):
        """Calculate statistics from trips completed since the last plot update
        Returns distance, duration, and nodes per trip (speed is calculated separately from moving agents)"""
        # Get trips that have been completed since last update
        current_trip_count = len(self.main_window.completed_trips)
        
        if current_trip_count <= self.last_plot_update_trip_count:
            # No new trips since last update
            return 0, 0, 0, current_trip_count
        
        # Get only the new trips since last update
        new_trips = self.main_window.completed_trips[self.last_plot_update_trip_count:]
        
        if not new_trips:
            return 0, 0, 0, current_trip_count
        
        # Calculate averages from new trips only
        distances = [trip.get('distance', 0) / 1000 for trip in new_trips if trip.get('distance', 0) > 0]  # Convert m to km
        
        # Calculate average nodes per trip
        nodes_per_trip = []
        for trip in new_trips:
            path_nodes = trip.get('path_nodes', [])
            if isinstance(path_nodes, list) and len(path_nodes) > 0:
                nodes_per_trip.append(len(path_nodes))
        
        # Calculate averages
        avg_distance = sum(distances) / len(distances) if distances else 0
        avg_nodes_per_trip = sum(nodes_per_trip) / len(nodes_per_trip) if nodes_per_trip else 0
        
        # Calculate duration from distance using the current average speed from moving agents
        # This uses real-time speed data instead of historical trip data
        avg_duration = 0
        if avg_distance > 0:
            # Get current average speed from moving agents
            current_avg_speed = self._calculate_current_average_speed()
            
            if current_avg_speed > 0:
                # Calculate duration: time = distance / speed
                avg_speed_ms = current_avg_speed / 3.6  # km/h to m/s
                avg_distance_m = avg_distance * 1000  # km to m
                avg_duration_s = avg_distance_m / avg_speed_ms  # seconds
                avg_duration = avg_duration_s / 60  # Convert to minutes
        
        # Update the counter for next time
        self.last_plot_update_trip_count = current_trip_count
        
        return avg_distance, avg_duration, avg_nodes_per_trip, current_trip_count

    def _calculate_trend_line(self, data, window_size=10):
        """Calculate a simple moving average trend line"""
        if len(data) < window_size:
            return list(data)
        
        trend = []
        for i in range(len(data)):
            if i < window_size - 1:
                # For the beginning, use available data
                trend.append(sum(data[:i+1]) / (i+1))
            else:
                # Rolling average
                trend.append(sum(data[i-window_size+1:i+1]) / window_size)
        return trend
    
    def update_plots(self):
        """Update all real-time plots"""
        try:
            times = list(self.stats_history['time'])
            
            if len(times) < 1:  # Need at least 1 point to plot
                return
            
            # Plot 1: Moving agents vs time
            self.main_window.stats_axes[0].clear()
            moving_agents_data = list(self.stats_history['moving_agents'])
            moving_agents_trend = self._calculate_trend_line(moving_agents_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[0].plot(times, moving_agents_data, 'b:', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[0].plot(times, moving_agents_trend, 'b-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[0].set_title('Moving Agents vs Time', fontsize=10)
            self.main_window.stats_axes[0].set_ylabel('Number of Moving Agents', fontsize=8)
            self.main_window.stats_axes[0].grid(True, alpha=0.3)
            self.main_window.stats_axes[0].tick_params(labelsize=8)
            self.main_window.stats_axes[0].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[0], times)
            
            # Plot 2: Network utilization vs time
            self.main_window.stats_axes[1].clear()
            utilization_data = list(self.stats_history['utilization'])
            utilization_trend = self._calculate_trend_line(utilization_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[1].plot(times, utilization_data, 'r:', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[1].plot(times, utilization_trend, 'r-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[1].set_title('Network Utilization vs Time', fontsize=10)
            self.main_window.stats_axes[1].set_ylabel('Network Utilization (%)', fontsize=8)
            self.main_window.stats_axes[1].grid(True, alpha=0.3)
            self.main_window.stats_axes[1].tick_params(labelsize=8)
            self.main_window.stats_axes[1].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[1], times)
            
            # Plot 3: Average speed vs time
            self.main_window.stats_axes[2].clear()
            speed_data = list(self.stats_history['avg_speed'])
            speed_trend = self._calculate_trend_line(speed_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[2].plot(times, speed_data, 'g:', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[2].plot(times, speed_trend, 'g-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[2].set_title('Average Speed of Moving Agents vs Time', fontsize=10)
            self.main_window.stats_axes[2].set_ylabel('Average Speed (km/h)', fontsize=8)
            self.main_window.stats_axes[2].grid(True, alpha=0.3)
            self.main_window.stats_axes[2].tick_params(labelsize=8)
            self.main_window.stats_axes[2].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[2], times)
            
            # Plot 4: Average trip distance vs time
            self.main_window.stats_axes[3].clear()
            distance_data = list(self.stats_history['avg_trip_distance'])
            distance_trend = self._calculate_trend_line(distance_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[3].plot(times, distance_data, 'm:', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[3].plot(times, distance_trend, 'm-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[3].set_title('Average Trip Distance vs Time', fontsize=10)
            self.main_window.stats_axes[3].set_ylabel('Average Trip Distance (km)', fontsize=8)
            self.main_window.stats_axes[3].grid(True, alpha=0.3)
            self.main_window.stats_axes[3].tick_params(labelsize=8)
            self.main_window.stats_axes[3].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[3], times)
            
            # Plot 5: Average trip duration vs time
            self.main_window.stats_axes[4].clear()
            duration_data = list(self.stats_history['avg_trip_duration'])
            duration_trend = self._calculate_trend_line(duration_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[4].plot(times, duration_data, 'c:', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[4].plot(times, duration_trend, 'c-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[4].set_title('Average Trip Duration vs Time', fontsize=10)
            self.main_window.stats_axes[4].set_ylabel('Average Trip Duration (min)', fontsize=8)
            self.main_window.stats_axes[4].grid(True, alpha=0.3)
            self.main_window.stats_axes[4].tick_params(labelsize=8)
            self.main_window.stats_axes[4].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[4], times)
            
            # Plot 6: Average nodes per trip vs time
            self.main_window.stats_axes[5].clear()
            nodes_data = list(self.stats_history['avg_nodes_per_trip'])
            nodes_trend = self._calculate_trend_line(nodes_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[5].plot(times, nodes_data, color='orange', linestyle=':', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[5].plot(times, nodes_trend, color='orange', linestyle='-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[5].set_title('Average Nodes per Trip vs Time', fontsize=10)
            self.main_window.stats_axes[5].set_ylabel('Average Number of Nodes', fontsize=8)
            self.main_window.stats_axes[5].grid(True, alpha=0.3)
            self.main_window.stats_axes[5].tick_params(labelsize=8)
            self.main_window.stats_axes[5].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[5], times)
            
            # Plot 7: Agent type distribution vs time
            self.main_window.stats_axes[6].clear()
            if self.stats_history['agent_types']:
                # Get all unique agent types across all time points
                all_agent_types = set()
                for type_dict in self.stats_history['agent_types']:
                    if isinstance(type_dict, dict):
                        all_agent_types.update(type_dict.keys())
                
                # Debug: Log agent types found (only occasionally)
                if all_agent_types and len(self.stats_history['time']) % 50 == 0:  # Log every 50th update
                    self.main_window.add_log_message(f"Debug: Agent types in history: {all_agent_types}")
                
                # Plot each agent type as a separate line
                legend_labels = []
                legend_lines = []
                
                for i, agent_type in enumerate(sorted(all_agent_types)):
                    counts = []
                    for type_dict in self.stats_history['agent_types']:
                        if isinstance(type_dict, dict):
                            counts.append(type_dict.get(agent_type, 0))
                        else:
                            counts.append(0)
                    
                    # Plot if there's meaningful data (not all zeros, or if it's the only type)
                    if any(c > 0 for c in counts) or len(all_agent_types) == 1:
                        color = COLORS.AGENT_TYPE_PLOT_COLORS[i % len(COLORS.AGENT_TYPE_PLOT_COLORS)]
                        
                        # Calculate trend line for this agent type
                        counts_trend = self._calculate_trend_line(counts)
                        
                        # Instant data: dotted line with less opacity
                        self.main_window.stats_axes[6].plot(times, counts, color=color, linestyle=':', linewidth=1.0, 
                                                           alpha=0.4, marker='o', markersize=1.5)
                        # Trend line: solid line
                        line = self.main_window.stats_axes[6].plot(times, counts_trend, color=color, linestyle='-', 
                                                                   linewidth=2.0, alpha=1.0, label=agent_type)[0]
                        legend_labels.append(agent_type)
                        legend_lines.append(line)
                
                # Add legend if we have plotted data
                if legend_labels:
                    try:
                        self.main_window.stats_axes[6].legend(legend_lines, legend_labels, fontsize=7, 
                                                            loc='upper right', framealpha=0.8)
                    except Exception as e:
                        # If legend fails, just continue without it
                        self.main_window.add_log_message(f"Debug: Legend failed: {e}")
                        pass
                else:
                    # No meaningful data to plot
                    self.main_window.stats_axes[6].text(0.5, 0.5, 'No agent type data available', 
                                                       transform=self.main_window.stats_axes[6].transAxes,
                                                       ha='center', va='center', fontsize=8)
            else:
                # No data at all
                self.main_window.stats_axes[6].text(0.5, 0.5, 'Waiting for agent type data...', 
                                                   transform=self.main_window.stats_axes[6].transAxes,
                                                   ha='center', va='center', fontsize=8)
            
            self.main_window.stats_axes[6].set_title('Active Agent Type Count vs Time', fontsize=10)
            self.main_window.stats_axes[6].set_ylabel('Agent Type Count', fontsize=8)
            self.main_window.stats_axes[6].grid(True, alpha=0.3)
            self.main_window.stats_axes[6].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[6], times)
            
            # Plot 8: Total trip count vs time
            self.main_window.stats_axes[7].clear()
            trip_count_data = list(self.stats_history['trip_count'])
            trip_count_trend = self._calculate_trend_line(trip_count_data)
            # Instant data: dotted line with less opacity
            self.main_window.stats_axes[7].plot(times, trip_count_data, color='purple', linestyle=':', linewidth=1.0, alpha=0.4, marker='o', markersize=1.5)
            # Trend line: solid line
            self.main_window.stats_axes[7].plot(times, trip_count_trend, color='purple', linestyle='-', linewidth=2.0, alpha=1.0, label='Trend')
            self.main_window.stats_axes[7].set_title('Total Trip Count vs Time', fontsize=10)
            self.main_window.stats_axes[7].set_ylabel('Total Number of Trips', fontsize=8)
            self.main_window.stats_axes[7].grid(True, alpha=0.3)
            self.main_window.stats_axes[7].tick_params(labelsize=8)
            self.main_window.stats_axes[7].legend(fontsize=7, loc='best')
            self.format_time_axis(self.main_window.stats_axes[7], times)
            
            # Refresh all canvases
            for canvas in self.main_window.stats_canvases:
                canvas.draw_idle()
            
        except Exception as e:
            self.main_window.add_log_message(f"Error updating plots: {str(e)}")

    def clear_statistics(self):
        """Clear all statistics data and plots"""
        for key in self.stats_history:
            self.stats_history[key].clear()
        
        # Reset trip counter for next simulation
        self.last_plot_update_trip_count = 0
        
        if hasattr(self.main_window, 'stats_axes'):
            for ax in self.main_window.stats_axes:
                ax.clear()
            for canvas in self.main_window.stats_canvases:
                canvas.draw_idle()

    def export_plots(self):
        """Export all statistics plots as images"""
        if not hasattr(self.main_window, 'stats_figures') or not self.main_window.stats_figures:
            self.main_window.add_log_message("❌ No plots available to export.")
            return
        
        # Open directory selection dialog
        directory = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Directory to Save Plots",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not directory:
            return  # User cancelled
        
        try:
            # Define plot names
            plot_names = [
                "moving_agents_vs_time",
                "network_utilization_vs_time", 
                "average_speed_vs_time",
                "average_trip_distance_vs_time",
                "average_trip_duration_vs_time",
                "average_nodes_per_trip_vs_time",
                "active_agent_type_count_vs_time",
                "total_trip_count_vs_time"
            ]
            
            exported_count = 0
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export each plot
            for i, (figure, plot_name) in enumerate(zip(self.main_window.stats_figures, plot_names)):
                if figure:
                    filename = f"{plot_name}_{timestamp}.png"
                    file_path = os.path.join(directory, filename)
                    
                    # Save the figure with high DPI and tight layout
                    figure.savefig(
                        file_path,
                        dpi=300,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        format='png',
                        pad_inches=0.2
                    )
                    exported_count += 1
                    self.main_window.add_log_message(f"Exported: {filename}")
            
            self.main_window.add_log_message(f"Successfully exported {exported_count} plots to: {directory}")
            
            # Also create a summary plot with all subplots
            if len(self.main_window.stats_figures) >= 7:  # Ensure we have enough plots
                self.create_summary_plot(directory, timestamp)
                
        except Exception as e:
            self.main_window.add_log_message(f"❌ Export plots failed: {str(e)}")
    
    def create_summary_plot(self, directory, timestamp):
        """Create a summary plot with all statistics in one image"""
        try:
            # Create a large figure with subplots
            summary_fig = Figure(figsize=(16, 12), dpi=100)
            summary_fig.suptitle('Traffic Simulation Statistics Summary', fontsize=16, fontweight='bold')
            
            # Create a 3x3 grid (we have 8 plots, so we'll use 3x3)
            for i in range(min(8, len(self.main_window.stats_axes))):
                ax_summary = summary_fig.add_subplot(3, 3, i + 1)
                
                # Copy the data from the original axis
                original_ax = self.main_window.stats_axes[i]
                
                # Get the line data from the original plot
                lines = original_ax.get_lines()
                if lines:
                    for line in lines:
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        label = line.get_label()
                        color = line.get_color()
                        
                        if len(xdata) > 0 and len(ydata) > 0:
                            ax_summary.plot(xdata, ydata, color=color, linewidth=1.2, label=label if label and not label.startswith('_') else None)
                
                # Copy the title and labels
                ax_summary.set_title(original_ax.get_title(), fontsize=10)
                ax_summary.set_xlabel(original_ax.get_xlabel(), fontsize=8)
                ax_summary.set_ylabel(original_ax.get_ylabel(), fontsize=8)
                ax_summary.grid(True, alpha=0.3)
                ax_summary.tick_params(labelsize=7)
                
                # Add legend if the original plot had one
                if original_ax.get_legend():
                    ax_summary.legend(fontsize=6)
            
            # Adjust layout
            summary_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the summary plot
            summary_filename = f"simulation_summary_{timestamp}.png"
            summary_path = os.path.join(directory, summary_filename)
            summary_fig.savefig(
                summary_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                pad_inches=0.2
            )
            
            self.main_window.add_log_message(f"Created summary plot: {summary_filename}")
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Failed to create summary plot: {str(e)}")
