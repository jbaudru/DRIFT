from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

# Added standard library imports
import datetime
from lib.st_selection import RandomSelection, ActivityBasedSelection, ZoneBasedSelection, GravitySelection, HubAndSpokeSelection
from lib.simulation_controller import SimulationController
from lib.managers.statistics_manager import StatisticsManager
from lib.managers.export_manager import ExportManager
from lib.managers.settings_manager import SettingsManager
from lib.managers.ui_manager import UIManager
from lib.managers.network_manager import NetworkManager
from lib.managers.trip_manager import TripManager
from lib.managers.status_manager import StatusManager

# Configuration imports
from config import (UI, FILES)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize simulation controller
        self.simulation_controller = SimulationController(self)
        
        # Initialize statistics manager
        self.statistics_manager = StatisticsManager(self)
        
        # Initialize export manager
        self.export_manager = ExportManager(self)
        
        # Initialize settings manager
        self.settings_manager = SettingsManager(self)
        
        # Initialize UI manager
        self.ui_manager = UIManager(self)
        
        # Initialize network manager
        self.network_manager = NetworkManager(self)
        
        # Initialize trip manager
        self.trip_manager = TripManager(self)
        
        # Initialize status manager
        self.status_manager = StatusManager(self)
        
        # Simulation tracking variables (moved to managers, but keeping for compatibility)
        self.simulation_time = 0.0
        
        # Initialize the UI
        self.ui_manager.init_ui()
        self.ui_manager.setup_widget_references()
        
        # Add initial log message after UI is fully set up
        self.add_log_message("GUI initialized. Load a graph file to begin.")
        
        # Initialize export controls
        self.update_export_controls()
        
        # Load settings and initialize UI
        self.settings_manager.load_settings()
        self.settings_manager.load_settings_into_ui()
    
    @property 
    def simulation_thread(self):
        """Property for backward compatibility"""
        return self.simulation_controller.simulation_thread
        
    @property
    def simulation_timer(self):
        """Property for backward compatibility"""
        return self.simulation_controller.simulation_timer
    
    @property
    def export_location(self):
        """Property for backward compatibility"""
        return self.export_manager.export_location
    
    @export_location.setter
    def export_location(self, value):
        """Property setter for backward compatibility"""
        self.export_manager.export_location = value
        
    @property
    def is_paused(self):
        """Property for backward compatibility"""
        return self.simulation_controller.is_paused
        
    @property
    def stats_history(self):
        """Property for backward compatibility"""
        return self.statistics_manager.stats_history
    
    @property
    def graph(self):
        """Property for backward compatibility"""
        return self.network_manager.graph
    
    @graph.setter
    def graph(self, value):
        """Property setter for backward compatibility"""
        self.network_manager.graph = value
    
    @property
    def completed_trips(self):
        """Property for backward compatibility"""
        return self.trip_manager.completed_trips
    
    @completed_trips.setter
    def completed_trips(self, value):
        """Property setter for backward compatibility"""
        self.trip_manager.completed_trips = value
    
    @property
    def current_simulation_time(self):
        """Property for backward compatibility"""
        return self.trip_manager.current_simulation_time
    
    @current_simulation_time.setter
    def current_simulation_time(self, value):
        """Property setter for backward compatibility"""
        self.trip_manager.current_simulation_time = value
    
    @property
    def last_plot_update_trip_count(self):
        """Property for backward compatibility"""
        return self.trip_manager.last_plot_update_trip_count
    
    @last_plot_update_trip_count.setter
    def last_plot_update_trip_count(self, value):
        """Property setter for backward compatibility"""
        self.trip_manager.last_plot_update_trip_count = value
    
    @property
    def simulation_start_time(self):
        """Property for backward compatibility"""
        return self.status_manager.simulation_start_time
    
    @simulation_start_time.setter
    def simulation_start_time(self, value):
        """Property setter for backward compatibility"""
        self.status_manager.simulation_start_time = value
    
    def reset_settings_to_defaults(self):
        """Reset all settings to their default values - delegates to settings manager"""
        self.settings_manager.reset_settings_to_defaults()
    
    def on_reset_button_clicked(self):
        """Handle reset button click - delegates to settings manager"""
        self.settings_manager.on_reset_button_clicked()
    
    def on_apply_button_clicked(self):
        """Handle apply button click - delegates to settings manager"""
        self.settings_manager.on_apply_button_clicked()
    
    def load_settings_into_ui(self):
        """Load current settings values into the UI - delegates to settings manager"""
        self.settings_manager.load_settings_into_ui()
    
    def apply_settings(self):
        """Apply the current settings - delegates to settings manager"""
        self.settings_manager.apply_settings()
    
    def update_widget_st_selector_parameters(self):
        """Update the simulation widget's ST selector parameters - delegates to settings manager"""
        self.settings_manager.update_widget_st_selector_parameters()
    
    def get_current_settings(self):
        """Get current settings as a dictionary - delegates to settings manager"""
        return self.settings_manager.get_current_settings()
    
    def load_graph(self):
        """Load a graph file - delegates to network manager"""
        self.network_manager.load_graph()
    
    def update_network_statistics(self):
        """Update network statistics - delegates to network manager"""
        self.network_manager.update_network_statistics()
    
    def _is_simulation_active(self):
        """Return True if a simulation is started (running or paused)."""
        return self.simulation_controller.is_simulation_active()
    
    def stop_and_clear_simulation(self):
        """Fully stop the simulation and clear temporary UI/data while keeping the loaded graph."""
        # Stop timer and pause state
        self.simulation_timer.stop()
        self.is_paused = False
        self.pause_button.setVisible(False)
        self.pause_button.setIcon(QIcon(UI.PAUSE_ICON_PATH))
        self.pause_button.setToolTip("Pause")
        
        # Disconnect speed change signal
        if hasattr(self.speed_spinbox, 'valueChanged'):
            try:
                self.speed_spinbox.valueChanged.disconnect(self.update_simulation_speed)
            except:
                pass
        
        # Stop thread and disconnect signals
        if self.simulation_thread:
            try:
                self.simulation_thread.log_message.disconnect(self.add_log_message)
            except:
                pass
            try:
                self.simulation_thread.agents_updated.disconnect(self.update_agents)
            except:
                pass
            try:
                self.simulation_thread.simulation_finished.disconnect(self.simulation_finished)
            except:
                pass
            try:
                self.simulation_thread.trip_completed.disconnect(self.add_completed_trip)
            except:
                pass
            try:
                self.simulation_thread.status_updated.disconnect(self.update_status)
            except:
                pass
            
            # Ensure underlying sim loop stops
            try:
                self.simulation_thread.stop()
                # Also reset pause state when stopping
                if hasattr(self.simulation_thread, 'paused'):
                    self.simulation_thread.paused = False
            except:
                pass
            
            self.simulation_thread = None
        
        # Clear agents in the visualization
        self.simulation_widget.set_agents([])
        
        # Return to full rendering mode
        self.simulation_widget.stop_simulation_rendering()
        
        # Clear trips and temporary state using trip manager
        self.trip_manager.clear_trips()
        self.status_manager.clear_simulation_start_time()
        
        # Clear statistics data and plots
        self.statistics_manager.clear_statistics()
        
        # Reset UI labels using status manager
        self.status_manager.reset_status_labels()
        
        # Re-enable controls
        self.start_button.setIcon(QIcon(UI.PLAY_ICON_PATH))
        self.start_button.setToolTip("Start Simulation")
        self.load_button.setEnabled(True)
        self.selection_combo.setEnabled(True)
        self.agents_spinbox.setEnabled(True)
        self.duration_spinbox.setEnabled(True)
        
        self.add_log_message("Simulation stopped. Cleared agents and temporary data.")
    
    def toggle_simulation(self):
        """Start or stop the simulation"""
        self.simulation_controller.toggle_simulation()

    def toggle_pause(self):
        """Pause or resume the simulation without finishing it."""
        self.simulation_controller.toggle_pause()
    
    def update_simulation(self):
        """Monitor simulation thread - optimized approach"""
        self.simulation_controller.update_simulation()

    def update_simulation_speed(self, new_speed):
        """Update simulation speed dynamically during runtime"""
        self.simulation_controller.update_simulation_speed(new_speed)
    
    def on_selection_mode_changed(self, mode):
        """Handle selection mode change in the combo box"""
        if self.graph:
            # Temporarily disable the combo box to prevent rapid changes
            self.selection_combo.setEnabled(False)
            try:
                self.update_selection_mode()
                # Force a repaint to show the new visualization style
                self.simulation_widget.update()
            finally:
                # Re-enable the combo box
                self.selection_combo.setEnabled(True)
    
    def update_selection_mode(self):
        """Update the selection mode for visualization"""
        if not self.graph:
            return
            
        selection_mode = self.selection_combo.currentText()
        
        try:
            if selection_mode == 'activity':
                st_selector = ActivityBasedSelection(self.graph)
            elif selection_mode == 'zone':
                st_selector = ZoneBasedSelection(self.graph)
                # Apply current zone settings
                if hasattr(st_selector, 'set_parameters'):
                    st_selector.set_parameters(intra_zone_probability=self.zone_intra_prob_spinbox.value())
            elif selection_mode == 'gravity':
                self.add_log_message("Initializing gravity model (this may take a moment for large networks)...")
                
                # Use the improved factory method with progress callback
                def progress_callback(message):
                    # Try to show loading spinner if available
                    if hasattr(self, 'simulation_controller') and hasattr(self.simulation_controller, 'loading_spinner'):
                        self.simulation_controller.loading_spinner.update_text(message)
                    # Also add to log
                    self.add_log_message(f"  {message}")
                
                # Create gravity model with progress tracking
                try:
                    st_selector = GravitySelection.create_with_progress(
                        self.graph,
                        progress_callback=progress_callback,
                        alpha=self.gravity_alpha_spinbox.value(),
                        beta=self.gravity_beta_spinbox.value()
                    )
                    # Apply current gravity settings (parameters were already set in factory method)
                    self.add_log_message("Gravity model initialized successfully.")
                except Exception as e:
                    self.add_log_message(f"Error initializing gravity model: {e}")
                    # Fallback to regular initialization
                    st_selector = GravitySelection(
                        self.graph,
                        alpha=self.gravity_alpha_spinbox.value(),
                        beta=self.gravity_beta_spinbox.value()
                    )
            elif selection_mode == 'hub':
                self.add_log_message("Initializing hub-and-spoke model (this may take a moment for large networks)...")
                st_selector = HubAndSpokeSelection(self.graph)
                # Apply current hub settings
                if hasattr(st_selector, 'set_parameters'):
                    st_selector.set_parameters(
                        hub_trip_probability=self.hub_trip_prob_spinbox.value(),
                        hub_percentage=self.hub_percentage_spinbox.value()
                    )
                self.add_log_message("Hub-and-spoke model initialized successfully.")
            else:
                st_selector = RandomSelection(self.graph)
            
            self.simulation_widget.set_selection_mode(selection_mode, st_selector)
            
        except Exception as e:
            self.add_log_message(f"Error initializing {selection_mode} selection for preview: {str(e)}")
            # Do not override user's choice; keep current selection.
            # Visualization will be updated when the simulation thread emits st_selector_ready.
            self.add_log_message("Keeping selected mode; preview will update when simulation starts.")
    
    def update_agents(self, agents):
        """Update the agents in the visualization"""
        self.simulation_widget.set_agents(agents)
    
    def simulation_finished(self, stats):
        """Handle simulation completion"""
        # Note: Timer management is now handled by SimulationController
        
        self.start_button.setIcon(QIcon(UI.PLAY_ICON_PATH))
        self.start_button.setToolTip("Start Simulation")
        
        # Re-enable controls
        self.load_button.setEnabled(True)
        self.selection_combo.setEnabled(True)
        self.agents_spinbox.setEnabled(True)
        self.duration_spinbox.setEnabled(True)
        
        # Hide and reset pause state
        self.pause_button.setIcon(QIcon(UI.PAUSE_ICON_PATH))
        self.pause_button.setToolTip("Pause")
        self.pause_button.setVisible(False)
        
        # Reset simulation tracking
        self.status_manager.clear_simulation_start_time()
        
        # Update status labels to show final state
        self.status_manager.update_final_status()
        
        # Return to full rendering mode
        self.simulation_widget.stop_simulation_rendering()
        
        if stats:
            self.add_log_message("=== Simulation Statistics ===")
            self.add_log_message(f"Total trips: {stats.get('total_trips', 0)}")
            self.add_log_message(f"Total distance: {stats.get('total_distance_km', 0):.1f} km")
            self.add_log_message(f"Average speed: {stats.get('average_speed_kmh', 0):.1f} km/h")
            self.add_log_message(f"Completed trips recorded: {len(self.completed_trips)}")
        
        # Update export controls
        self.update_export_controls()
    
    def add_log_message(self, message):
        """Add a message to the log"""
        timestamp = datetime.datetime.now().strftime(FILES.LOG_TIMESTAMP_FORMAT)
        formatted_message = FILES.LOG_MESSAGE_TEMPLATE.format(timestamp, message)
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
    
    def add_completed_trip(self, trip_info):
        """Add a completed trip - delegates to trip manager"""
        self.trip_manager.add_completed_trip(trip_info)
    
    def update_status(self, status_info):
        """Update the status bar with simulation information - delegates to status manager"""
        self.status_manager.update_status(status_info)
    
    def handle_st_selector_ready(self, st_selector, selection_mode):
        """Handle ST selector ready signal from simulation thread"""
        print(f"DEBUG: Main window received st_selector_ready: mode='{selection_mode}', selector={type(st_selector).__name__}")
        
        # Update the simulation widget with the correct selection mode and selector
        if hasattr(self, 'simulation_widget'):
            self.simulation_widget.set_selection_mode(selection_mode, st_selector)
        
        # Also update any other components that need the ST selector
        print(f"DEBUG: Updated simulation widget with mode '{selection_mode}'")
    
    def update_performance_status(self):
        """Update performance optimization status - delegates to status manager"""
        self.status_manager.update_performance_status()

    def browse_export_location(self):
        """Open file dialog to select export location - delegates to export manager"""
        self.export_manager.browse_export_location()

    def export_current_data(self):
        """Export current trip data - delegates to export manager"""
        self.export_manager.export_trip_data()

    def export_plots(self):
        """Export all statistics plots as images - delegates to export manager"""
        self.export_manager.export_plots()

    def convert_data_format(self):
        """Convert data between CSV and JSON formats - delegates to export manager"""
        self.export_manager.convert_data_format()

    def update_export_controls(self):
        """Update export controls based on simulation state - delegates to export manager"""
        self.export_manager.update_export_controls()


def main():
    """Main function"""
    import sys
    import os
    
    app = QApplication(sys.argv)
    
    # Windows-specific taskbar icon fix
    if os.name == 'nt':  # Windows
        try:
            # Set a unique app ID to separate from Python's default
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('traffic.simulation.app')
        except:
            pass  # Ignore if this fails on older Windows versions
    
    # Set application icon for taskbar display
    app_icon = QIcon()
    for size in UI.ICON_SIZES:
        app_icon.addFile(f'{UI.ICON_BASE_PATH}/{size[0]}x{size[1]}.ico', QSize(size[0], size[1]))
    app.setWindowIcon(app_icon)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
