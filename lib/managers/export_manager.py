"""
Export Manager Module
Handles data export functionality including trip data and plot exports
"""

import os
import datetime
from PyQt5.QtWidgets import QFileDialog


class ExportManager:
    """Handles data export functionality"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.export_location = None
    
    def browse_export_location(self):
        """Open file dialog to select export location"""
        current_format = self.main_window.export_format_combo.currentText().lower()
        if current_format == 'csv':
            file_filter = "CSV Files (*.csv)"
            default_name = f"simulation_trips_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            file_filter = "JSON Files (*.json)"
            default_name = f"simulation_trips_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, 
            "Choose Export Location", 
            default_name,
            file_filter
        )
        
        if file_path:
            self.export_location = file_path
            # Show just the filename in the label for space
            filename = os.path.basename(file_path)
            self.main_window.export_location_label.setText(filename)
            self.main_window.export_location_label.setToolTip(file_path)
    
    def export_trip_data(self, format_type=None):
        """Export trip data in specified format"""
        if not hasattr(self.main_window, 'simulation_thread') or not self.main_window.simulation_thread:
            self.main_window.add_log_message("No simulation data available to export.")
            return
        
        if not hasattr(self.main_window.simulation_thread, 'data_logger') or not self.main_window.simulation_thread.data_logger:
            self.main_window.add_log_message("No data logger available.")
            return
        
        data_logger = self.main_window.simulation_thread.data_logger
        if not data_logger.trips_data:
            self.main_window.add_log_message("No trip data available to export.")
            return
        
        try:
            # Use provided format or get from combo box
            current_format = format_type if format_type else self.main_window.export_format_combo.currentText().lower()
            
            if self.export_location:
                # Use user-specified location
                file_path = self.export_location
                # Ensure the file extension matches the selected format
                base_path = os.path.splitext(file_path)[0]
                if current_format == 'csv':
                    file_path = base_path + '.csv'
                else:
                    file_path = base_path + '.json'
            else:
                # Use default location
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                if current_format == 'csv':
                    file_path = os.path.join('data', f'simulation_trips_{timestamp}.csv')
                else:
                    file_path = os.path.join('data', f'simulation_trips_{timestamp}.json')
            
            # Export the data
            result_path = data_logger.export_to_format(file_path, current_format)
            self.main_window.add_log_message(f"Data exported successfully to: {result_path}")
            self.main_window.add_log_message(f"Total trips exported: {len(data_logger.trips_data)}")
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Export failed: {str(e)}")
    
    def export_plots(self):
        """Export all plots to files - delegates to statistics manager"""
        self.main_window.statistics_manager.export_plots()
    
    def convert_data_format(self):
        """Convert data between CSV and JSON formats"""
        # First, select the source file
        source_formats = "All Supported (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json)"
        source_file, _ = QFileDialog.getOpenFileName(
            self.main_window, 
            "Select Source File to Convert", 
            "",
            source_formats
        )
        
        if not source_file:
            return
        
        # Determine source format
        source_ext = os.path.splitext(source_file)[1].lower()
        if source_ext == '.csv':
            target_format = 'json'
            target_ext = '.json'
            file_filter = "JSON Files (*.json)"
        elif source_ext == '.json':
            target_format = 'csv'
            target_ext = '.csv'
            file_filter = "CSV Files (*.csv)"
        else:
            self.main_window.add_log_message("❌ Unsupported file format. Please select a CSV or JSON file.")
            return
        
        # Select target file
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        default_target = f"{base_name}_converted{target_ext}"
        
        target_file, _ = QFileDialog.getSaveFileName(
            self.main_window, 
            f"Save as {target_format.upper()}", 
            default_target,
            file_filter
        )
        
        if not target_file:
            return
        
        try:
            # Use data logger for conversion
            if hasattr(self.main_window.simulation_controller, 'simulation_thread') and self.main_window.simulation_controller.simulation_thread and hasattr(self.main_window.simulation_controller.simulation_thread, 'data_logger'):
                data_logger = self.main_window.simulation_controller.simulation_thread.data_logger
            else:
                # Create temporary data logger for conversion
                from lib.data_logger import SimulationDataLogger
                data_logger = SimulationDataLogger()
            
            if target_format == 'json':
                result = data_logger.convert_csv_to_json(source_file, target_file)
            else:
                result = data_logger.convert_json_to_csv(source_file, target_file)
            
            if result:
                self.main_window.add_log_message(f"Successfully converted {source_file} to {target_file}")
            else:
                self.main_window.add_log_message("Conversion failed. Please check the log for details.")
                
        except Exception as e:
            self.main_window.add_log_message(f"Conversion failed: {str(e)}")
    
    def update_export_controls(self):
        """Update export controls based on simulation state"""
        # Enable export button when simulation is paused or finished
        simulation_active = (hasattr(self.main_window, 'simulation_thread') and 
                           self.main_window.simulation_thread and 
                           self.main_window.simulation_thread.running and 
                           not self.main_window.is_paused)
        
        self.main_window.export_current_button.setEnabled(not simulation_active)
        
        # Update export plots button (now in statistics tab)
        if hasattr(self.main_window, 'export_plots_button'):
            self.main_window.export_plots_button.setEnabled(not simulation_active and hasattr(self.main_window, 'stats_figures') and len(self.main_window.stats_figures) > 0)
        
        # Update button text based on state
        if self.main_window.is_paused:
            self.main_window.export_current_button.setText("Export Current Data (Paused)")
            if hasattr(self.main_window, 'export_plots_button'):
                self.main_window.export_plots_button.setText("Export (Paused)")
        elif hasattr(self.main_window, 'simulation_thread') and self.main_window.simulation_thread and not self.main_window.simulation_thread.running:
            self.main_window.export_current_button.setText("Export Final Data")
            if hasattr(self.main_window, 'export_plots_button'):
                self.main_window.export_plots_button.setText("Export (Final)")
        else:
            self.main_window.export_current_button.setText("Export Current Data")
            if hasattr(self.main_window, 'export_plots_button'):
                self.main_window.export_plots_button.setText("Export")
