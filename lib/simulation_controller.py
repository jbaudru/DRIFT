"""
Simulation Controller Module
Handles all simulation-related logic and lifecycle management

This module extracts simulation management logic from the main window to provide
better separation of concerns and cleaner code organization.

Key responsibilities:
- Starting and stopping simulations
- Managing simulation threads and timers
- Handling pause/resume functionality
- Coordinating with UI updates
- Managing simulation state transitions
"""

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from .simulation_thread import SimulationThread
from .threaded_loaders import ModelComputationThread
from .loading_spinner import LoadingSpinner
from config import UI, SIMULATION, PERFORMANCE


class SimulationController:
    """Handles all simulation-related logic"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.simulation_thread = None
        self.simulation_timer = QTimer()
        self.is_paused = False
        self.current_simulation_time = 0.0
        
        # Model computation threading
        self.model_computation_thread = None
        self.loading_spinner = None
        self.pending_simulation_params = None
        
        # Connect timer to update function
        self.simulation_timer.timeout.connect(self.update_simulation)
        
    def start_simulation(self):
        """Start the simulation with current parameters, including threaded model computation"""
        if not self.main_window.graph:
            self.main_window.add_log_message("❌ No graph loaded. Please load a graph first.")
            return
            
        # Get simulation parameters
        num_agents = self.main_window.agents_spinbox.value()
        duration_hours = self.main_window.duration_spinbox.value()
        speed_multiplier = self.main_window.speed_spinbox.value()
        selection_mode = self.main_window.selection_combo.currentText()
        
        # Store parameters for use after model computation
        self.pending_simulation_params = {
            'num_agents': num_agents,
            'duration_hours': duration_hours,
            'speed_multiplier': speed_multiplier,
            'selection_mode': selection_mode
        }
        
        # Check if we already have a computed selector in the simulation widget
        simulation_widget = getattr(self.main_window, 'simulation_widget', None)
        has_existing_selector = (simulation_widget and 
                               simulation_widget.st_selector is not None and 
                               simulation_widget.selection_mode == selection_mode.lower())
        
        if has_existing_selector:
            # Use the existing selector - no need to recompute
            self.main_window.add_log_message(f"Using existing {selection_mode} model...")
            self._start_simulation_with_existing_model(simulation_widget.st_selector)
        else:
            # Check if model computation is needed (for complex models)
            needs_model_computation = selection_mode in ["Gravity", "Zone-Based", "Hub and Spoke"]
            
            if needs_model_computation:
                self._start_model_computation()
            else:
                # Simple models can start directly
                self._start_simulation_with_parameters()
    
    def _start_model_computation(self):
        """Start model computation in a separate thread with loading spinner"""
        try:
            # Get the simulation widget to show spinner on visualization area
            simulation_widget = getattr(self.main_window, 'simulation_widget', None)
            if simulation_widget is None:
                # Fallback to central widget if simulation widget not found
                spinner_parent = self.main_window.centralWidget()
            else:
                spinner_parent = simulation_widget
            
            # Create and show loading spinner (blue, no text)
            self.loading_spinner = LoadingSpinner(
                parent=spinner_parent,
                spinner_size=60,
                show_text=False
            )
            self.loading_spinner.show_spinner()
            
            # Disable UI during computation
            self._disable_ui_during_computation()
            
            # Start model computation thread
            self.model_computation_thread = ModelComputationThread(
                self.main_window.graph,
                self.pending_simulation_params['selection_mode'],
                self.main_window
            )
            
            # Connect signals
            self.model_computation_thread.progress_updated.connect(self._on_model_progress)
            self.model_computation_thread.computation_finished.connect(self._on_model_ready)
            self.model_computation_thread.error_occurred.connect(self._on_model_error)
            self.model_computation_thread.log_message.connect(self.main_window.add_log_message)
            self.model_computation_thread.finished.connect(self._on_model_computation_finished)
            
            # Start computation
            self.model_computation_thread.start()
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error starting model computation: {str(e)}")
            if self.loading_spinner:
                self.loading_spinner.hide_spinner()
            self._enable_ui_after_computation()
    
    def _on_model_progress(self, progress, status):
        """Handle model computation progress updates"""
        # Simple spinner doesn't show progress, just keep spinning
        pass
    
    def _on_model_ready(self, st_selector):
        """Handle successful model computation"""
        if st_selector is not None:
            # Start simulation with the computed model
            self._start_simulation_with_model(st_selector)
        else:
            self.main_window.add_log_message("❌ Model computation failed")
            self._enable_ui_after_computation()
    
    def _on_model_error(self, error_message):
        """Handle model computation errors"""
        self.main_window.add_log_message(f"❌ Model computation error: {error_message}")
        self._enable_ui_after_computation()
    
    def _on_model_computation_finished(self):
        """Clean up after model computation is finished"""
        if self.loading_spinner:
            self.loading_spinner.hide_spinner()
            self.loading_spinner = None
        
        if self.model_computation_thread:
            self.model_computation_thread.deleteLater()
            self.model_computation_thread = None
    
    def _disable_ui_during_computation(self):
        """Disable UI elements during model computation"""
        self.main_window.start_button.setEnabled(False)
        self.main_window.load_button.setEnabled(False)
        self.main_window.selection_combo.setEnabled(False)
        self.main_window.agents_spinbox.setEnabled(False)
        self.main_window.duration_spinbox.setEnabled(False)
    
    def _enable_ui_after_computation(self):
        """Re-enable UI elements after model computation"""
        self.main_window.start_button.setEnabled(True)
        self.main_window.load_button.setEnabled(True)
        self.main_window.selection_combo.setEnabled(True)
        self.main_window.agents_spinbox.setEnabled(True)
        self.main_window.duration_spinbox.setEnabled(True)
    
    def _start_simulation_with_parameters(self):
        """Start simulation directly without model computation"""
        params = self.pending_simulation_params
        self.main_window.add_log_message("Starting simulation...")
        
        # Reset simulation state
        self.current_simulation_time = 0.0
        self.main_window.status_manager.set_simulation_start_time()
        self.is_paused = False
        
        # Clear previous data
        self.main_window.completed_trips = []
        self.main_window.last_plot_update_trip_count = 0
        self.main_window.trips_table.setRowCount(0)
        self.main_window.statistics_manager.clear_statistics()
        
        # Create and start simulation thread
        self.simulation_thread = SimulationThread(
            graph=self.main_window.graph,
            num_agents=params['num_agents'],
            selection_mode=params['selection_mode'],
            duration_hours=params['duration_hours']
        )
        
        self._setup_and_start_simulation(params)
    
    def _start_simulation_with_model(self, st_selector):
        """Start simulation with a pre-computed model"""
        params = self.pending_simulation_params
        self.main_window.add_log_message("Starting simulation with computed model...")
        
        # Reset simulation state
        self.current_simulation_time = 0.0
        self.main_window.status_manager.set_simulation_start_time()
        self.is_paused = False
        
        # Clear previous data
        self.main_window.completed_trips = []
        self.main_window.last_plot_update_trip_count = 0
        self.main_window.trips_table.setRowCount(0)
        self.main_window.statistics_manager.clear_statistics()
        
        # Create simulation thread with pre-computed model
        self.simulation_thread = SimulationThread(
            graph=self.main_window.graph,
            num_agents=params['num_agents'],
            selection_mode=params['selection_mode'],
            duration_hours=params['duration_hours']
        )
        
        # Set the pre-computed model
        self.simulation_thread.st_selector = st_selector
        
        self._setup_and_start_simulation(params)
    
    def _start_simulation_with_existing_model(self, st_selector):
        """Start simulation with an existing pre-computed model (avoids recomputation)"""
        params = self.pending_simulation_params
        self.main_window.add_log_message("Starting simulation with existing model...")
        
        # Reset simulation state
        self.current_simulation_time = 0.0
        self.main_window.status_manager.set_simulation_start_time()
        self.is_paused = False
        
        # Clear previous data
        self.main_window.completed_trips = []
        self.main_window.last_plot_update_trip_count = 0
        self.main_window.trips_table.setRowCount(0)
        self.main_window.statistics_manager.clear_statistics()
        
        # Create simulation thread (without st_selector parameter)
        self.simulation_thread = SimulationThread(
            graph=self.main_window.graph,
            num_agents=params['num_agents'],
            selection_mode=params['selection_mode'],
            duration_hours=params['duration_hours']
        )
        
        # Set the existing pre-computed model
        self.simulation_thread.st_selector = st_selector
        
        self._setup_and_start_simulation(params)
    
    def _setup_and_start_simulation(self, params):
        """Common setup and start logic for both direct and model-based simulation"""
        # Set the speed multiplier after creation
        self.simulation_thread.time_acceleration = params['speed_multiplier']
        
        # Apply current settings to the simulation
        current_settings = self.main_window.get_current_settings()
        self.simulation_thread.apply_settings(current_settings)
        
        # Connect signals
        self.simulation_thread.log_message.connect(self.main_window.add_log_message)
        self.simulation_thread.agents_updated.connect(self.main_window.update_agents)
        self.simulation_thread.trip_completed.connect(self.main_window.add_completed_trip)
        self.simulation_thread.status_updated.connect(self.main_window.update_status)
        self.simulation_thread.simulation_finished.connect(self.main_window.simulation_finished)
        self.simulation_thread.st_selector_ready.connect(self.main_window.handle_st_selector_ready)
        
        # Update UI state
        self._update_ui_for_simulation_start()
        
        # Start simulation and timer
        self.simulation_thread.start()
        self.simulation_timer.start(PERFORMANCE.GUI_UPDATE_INTERVAL)
        
        # Switch to performance rendering mode
        self.main_window.simulation_widget.start_simulation_rendering()
        
        self.main_window.add_log_message(f"Simulation started with {params['num_agents']} agents for {params['duration_hours']} hours")
        
        # Note: Don't call _enable_ui_after_computation() here as we want controls to remain disabled during simulation
        
    def stop_simulation(self):
        """Stop and cleanup simulation"""
        self.main_window.add_log_message("Stopping simulation...")
        
        # Stop timer and reset pause state
        self.simulation_timer.stop()
        self.is_paused = False
        self.main_window.pause_button.setVisible(False)
        self.main_window.pause_button.setIcon(QIcon(UI.PAUSE_ICON_PATH))
        self.main_window.pause_button.setToolTip("Pause")
        
        # Disconnect speed change signal if connected
        if hasattr(self.main_window.speed_spinbox, 'valueChanged'):
            try:
                self.main_window.speed_spinbox.valueChanged.disconnect(self.update_simulation_speed)
            except TypeError:
                pass  # Signal wasn't connected
        
        # Stop thread and disconnect signals
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()
            self.simulation_thread.deleteLater()
            self.simulation_thread = None
        
        # Clear agents in visualization
        self.main_window.simulation_widget.set_agents([])
        
        # Return to full rendering mode
        self.main_window.simulation_widget.stop_simulation_rendering()
        
        # Clear trips and temporary state
        self.main_window.completed_trips = []
        self.current_simulation_time = 0.0
        self.main_window.last_plot_update_trip_count = 0
        self.main_window.trips_table.setRowCount(0)
        self.main_window.status_manager.clear_simulation_start_time()
        
        # Clear statistics data and plots
        self.main_window.statistics_manager.clear_statistics()
        
        # Reset UI labels
        self.main_window.sim_time_label.setText(UI.SIM_TIME_TEMPLATE.format(UI.DEFAULT_SIM_TIME))
        self.main_window.real_time_label.setText(UI.REAL_TIME_TEMPLATE.format(UI.DEFAULT_REAL_TIME))
        self.main_window.moving_agents_label.setText(UI.MOVING_AGENTS_TEMPLATE.format(UI.DEFAULT_MOVING_AGENTS))
        self.main_window.network_util_label.setText(UI.NETWORK_UTIL_TEMPLATE.format(UI.DEFAULT_NETWORK_UTIL))
        
        # Re-enable controls
        self.main_window.start_button.setIcon(QIcon(UI.PLAY_ICON_PATH))
        self.main_window.start_button.setToolTip("Start Simulation")
        self.main_window.load_button.setEnabled(True)
        self.main_window.selection_combo.setEnabled(True)
        self.main_window.agents_spinbox.setEnabled(True)
        self.main_window.duration_spinbox.setEnabled(True)
        
        # Update export controls
        self.main_window.update_export_controls()
        
        self.main_window.add_log_message("Simulation stopped and cleaned up")
        
    def toggle_pause(self):
        """Pause/resume simulation"""
        if not self.simulation_thread or (not self.simulation_thread.running and not self.is_paused):
            self.main_window.add_log_message("No simulation to pause/resume")
            return
        
        if not self.is_paused:
            # Pause simulation
            self.simulation_timer.stop()
            self.simulation_thread.pause()
            self.is_paused = True
            self.main_window.pause_button.setIcon(QIcon(UI.PLAY_ICON_PATH))
            self.main_window.pause_button.setToolTip("Resume")
            # Keep st-model selection disabled when paused
            self.main_window.selection_combo.setEnabled(False)
            self.main_window.add_log_message("Simulation paused")
        else:
            # Resume simulation
            self.simulation_thread.resume()
            self.simulation_timer.start(PERFORMANCE.GUI_UPDATE_INTERVAL)
            self.is_paused = False
            self.main_window.pause_button.setIcon(QIcon(UI.PAUSE_ICON_PATH))
            self.main_window.pause_button.setToolTip("Pause")
            # Keep st-model selection disabled when running
            self.main_window.selection_combo.setEnabled(False)
            self.main_window.add_log_message("Simulation resumed")
        
        # Update export controls
        self.main_window.update_export_controls()
        
    def toggle_simulation(self):
        """Start or stop the simulation"""
        if not self.is_simulation_active():
            self.start_simulation()
        else:
            self.stop_simulation()
            
    def is_simulation_active(self):
        """Return True if a simulation is started (running or paused)"""
        return (self.simulation_thread is not None) and (self.simulation_thread.running or self.is_paused)
        
    def update_simulation(self):
        """Monitor simulation thread - optimized approach"""
        if self.simulation_thread and self.simulation_thread.running:
            # Update performance monitoring
            self.main_window.status_manager.update_performance_status()
        else:
            # Simulation may have finished
            self.simulation_timer.stop()
            
    def update_simulation_speed(self, new_speed):
        """Update simulation speed dynamically during runtime"""
        if self.simulation_thread and self.simulation_thread.running:
            self.simulation_thread.set_speed_multiplier(new_speed)
            self.main_window.add_log_message(f"Simulation speed updated to {new_speed}x")
            
    def _update_ui_for_simulation_start(self):
        """Update UI elements when simulation starts"""
        # Update button states
        self.main_window.start_button.setIcon(QIcon(UI.STOP_ICON_PATH))
        self.main_window.start_button.setToolTip("Stop Simulation")
        self.main_window.pause_button.setVisible(True)
        self.main_window.pause_button.setIcon(QIcon(UI.PAUSE_ICON_PATH))
        self.main_window.pause_button.setToolTip("Pause")
        
        # Disable controls during simulation
        self.main_window.load_button.setEnabled(False)
        self.main_window.selection_combo.setEnabled(False)
        self.main_window.agents_spinbox.setEnabled(False)
        self.main_window.duration_spinbox.setEnabled(False)
        
        # Connect speed change signal for dynamic updates
        self.main_window.speed_spinbox.valueChanged.connect(self.update_simulation_speed)
        
        # Update export controls
        self.main_window.update_export_controls()
