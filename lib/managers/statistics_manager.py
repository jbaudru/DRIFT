from PyQt5.QtWidgets import QFileDialog
from matplotlib.figure import Figure
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
        import matplotlib.ticker as ticker
        
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
        
        # Calculate trip-based statistics from completed trips (since last update)
        avg_speed, avg_distance, avg_duration, avg_nodes_per_trip, total_trip_count = self._calculate_trip_statistics_since_last_update()
        
        # Get active agent type distribution from status_info
        active_agent_type_percentages = status_info.get('active_agent_types', {})
        
        self.stats_history['avg_speed'].append(avg_speed)
        self.stats_history['avg_trip_distance'].append(avg_distance)
        self.stats_history['avg_trip_duration'].append(avg_duration)
        self.stats_history['avg_nodes_per_trip'].append(avg_nodes_per_trip)
        self.stats_history['agent_types'].append(active_agent_type_percentages)
        self.stats_history['trip_count'].append(total_trip_count)
        
        # Update plots (only if we have data and plots are initialized)
        if hasattr(self.main_window, 'stats_axes') and len(self.stats_history['time']) > 0:
            self.update_plots()
    
    def _calculate_trip_statistics_since_last_update(self):
        """Calculate statistics from trips completed since the last plot update"""
        # Get trips that have been completed since last update
        current_trip_count = len(self.main_window.completed_trips)
        
        if current_trip_count <= self.last_plot_update_trip_count:
            # No new trips since last update
            return 0, 0, 0, 0, current_trip_count
        
        # Get only the new trips since last update
        new_trips = self.main_window.completed_trips[self.last_plot_update_trip_count:]
        
        if not new_trips:
            return 0, 0, 0, 0, current_trip_count
        
        # Calculate averages from new trips only
        speeds = [trip.get('avg_speed', 0) * 3.6 for trip in new_trips if trip.get('avg_speed', 0) > 0]  # Convert m/s to km/h
        distances = [trip.get('distance', 0) / 1000 for trip in new_trips if trip.get('distance', 0) > 0]  # Convert m to km
        durations = [trip.get('duration', 0) / 60 for trip in new_trips if trip.get('duration', 0) > 0]  # Convert s to min
        
        # Calculate average nodes per trip
        nodes_per_trip = []
        for trip in new_trips:
            path_nodes = trip.get('path_nodes', [])
            if isinstance(path_nodes, list) and len(path_nodes) > 0:
                nodes_per_trip.append(len(path_nodes))
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        avg_distance = sum(distances) / len(distances) if distances else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_nodes_per_trip = sum(nodes_per_trip) / len(nodes_per_trip) if nodes_per_trip else 0
        
        # Update the counter for next time
        self.last_plot_update_trip_count = current_trip_count
        
        return avg_speed, avg_distance, avg_duration, avg_nodes_per_trip, current_trip_count

    def update_plots(self):
        """Update all real-time plots"""
        try:
            times = list(self.stats_history['time'])
            
            if len(times) < 2:  # Need at least 2 points to plot
                return
            
            # Plot 1: Moving agents vs time
            self.main_window.stats_axes[0].clear()
            self.main_window.stats_axes[0].plot(times, list(self.stats_history['moving_agents']), 'b-', linewidth=1.5)
            self.main_window.stats_axes[0].set_title('Moving Agents vs Time', fontsize=10)
            self.main_window.stats_axes[0].set_ylabel('Number of Moving Agents', fontsize=8)
            self.main_window.stats_axes[0].grid(True, alpha=0.3)
            self.main_window.stats_axes[0].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[0], times)
            
            # Plot 2: Network utilization vs time
            self.main_window.stats_axes[1].clear()
            self.main_window.stats_axes[1].plot(times, list(self.stats_history['utilization']), 'r-', linewidth=1.5)
            self.main_window.stats_axes[1].set_title('Network Utilization vs Time', fontsize=10)
            self.main_window.stats_axes[1].set_ylabel('Network Utilization (%)', fontsize=8)
            self.main_window.stats_axes[1].grid(True, alpha=0.3)
            self.main_window.stats_axes[1].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[1], times)
            
            # Plot 3: Average speed vs time
            self.main_window.stats_axes[2].clear()
            self.main_window.stats_axes[2].plot(times, list(self.stats_history['avg_speed']), 'g-', linewidth=1.5)
            self.main_window.stats_axes[2].set_title('Average Speed vs Time (Since Last Update)', fontsize=10)
            self.main_window.stats_axes[2].set_ylabel('Average Speed (km/h)', fontsize=8)
            self.main_window.stats_axes[2].grid(True, alpha=0.3)
            self.main_window.stats_axes[2].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[2], times)
            
            # Plot 4: Average trip distance vs time
            self.main_window.stats_axes[3].clear()
            self.main_window.stats_axes[3].plot(times, list(self.stats_history['avg_trip_distance']), 'm-', linewidth=1.5)
            self.main_window.stats_axes[3].set_title('Average Trip Distance vs Time (Since Last Update)', fontsize=10)
            self.main_window.stats_axes[3].set_ylabel('Average Trip Distance (km)', fontsize=8)
            self.main_window.stats_axes[3].grid(True, alpha=0.3)
            self.main_window.stats_axes[3].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[3], times)
            
            # Plot 5: Average trip duration vs time
            self.main_window.stats_axes[4].clear()
            self.main_window.stats_axes[4].plot(times, list(self.stats_history['avg_trip_duration']), 'c-', linewidth=1.5)
            self.main_window.stats_axes[4].set_title('Average Trip Duration vs Time (Since Last Update)', fontsize=10)
            self.main_window.stats_axes[4].set_ylabel('Average Trip Duration (min)', fontsize=8)
            self.main_window.stats_axes[4].grid(True, alpha=0.3)
            self.main_window.stats_axes[4].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[4], times)
            
            # Plot 6: Average nodes per trip vs time
            self.main_window.stats_axes[5].clear()
            self.main_window.stats_axes[5].plot(times, list(self.stats_history['avg_nodes_per_trip']), 'orange', linewidth=1.5)
            self.main_window.stats_axes[5].set_title('Average Nodes per Trip vs Time (Since Last Update)', fontsize=10)
            self.main_window.stats_axes[5].set_ylabel('Average Number of Nodes', fontsize=8)
            self.main_window.stats_axes[5].grid(True, alpha=0.3)
            self.main_window.stats_axes[5].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[5], times)
            
            # Plot 7: Agent type distribution vs time
            self.main_window.stats_axes[6].clear()
            if self.stats_history['agent_types']:
                # Get all unique agent types
                all_agent_types = set()
                for type_dict in self.stats_history['agent_types']:
                    all_agent_types.update(type_dict.keys())
                
                # Plot each agent type as a separate line
                for i, agent_type in enumerate(sorted(all_agent_types)):
                    percentages = []
                    for type_dict in self.stats_history['agent_types']:
                        percentages.append(type_dict.get(agent_type, 0))
                    
                    if any(p > 0 for p in percentages):  # Only plot if there's data
                        color = COLORS.AGENT_TYPE_PLOT_COLORS[i % len(COLORS.AGENT_TYPE_PLOT_COLORS)]
                        self.main_window.stats_axes[6].plot(times, percentages, color=color, linewidth=1.5, label=agent_type)
                
                self.main_window.stats_axes[6].legend(fontsize=6, loc='upper right')
            
            self.main_window.stats_axes[6].set_title('Active Agent Type Distribution vs Time', fontsize=10)
            self.main_window.stats_axes[6].set_ylabel('Agent Type Percentage (%)', fontsize=8)
            self.main_window.stats_axes[6].grid(True, alpha=0.3)
            self.main_window.stats_axes[6].tick_params(labelsize=8)
            self.format_time_axis(self.main_window.stats_axes[6], times)
            
            # Plot 8: Total trip count vs time
            self.main_window.stats_axes[7].clear()
            self.main_window.stats_axes[7].plot(times, list(self.stats_history['trip_count']), 'purple', linewidth=1.5)
            self.main_window.stats_axes[7].set_title('Total Trip Count vs Time', fontsize=10)
            self.main_window.stats_axes[7].set_ylabel('Total Number of Trips', fontsize=8)
            self.main_window.stats_axes[7].grid(True, alpha=0.3)
            self.main_window.stats_axes[7].tick_params(labelsize=8)
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
                "active_agent_type_distribution_vs_time",
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
