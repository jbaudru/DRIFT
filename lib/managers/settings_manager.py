"""
Settings Manager Module
Handles application settings and configuration management
"""

import json
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer


class SimulationConfig:
    """Configuration class for simulation settings"""
    
    def __init__(self):
        # Import config defaults
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from config import MODELS, NETWORK
        
        # Default BPR parameters
        self.bpr_alpha = NETWORK.DEFAULT_BPR_ALPHA
        self.bpr_beta = NETWORK.DEFAULT_BPR_BETA
        
        # Use default hourly probabilities from config
        self.hourly_probabilities = MODELS.DEFAULT_HOURLY_PROBABILITIES.copy()
        
        # Zone model settings
        self.zone_intra_probability = MODELS.DEFAULT_ZONE_INTRA_PROBABILITY
        
        # Activity model settings
        self.agent_type_distributions = MODELS.DEFAULT_AGENT_TYPE_DISTRIBUTIONS.copy()
        
        # Gravity model settings
        self.gravity_alpha = MODELS.DEFAULT_GRAVITY_ALPHA
        self.gravity_beta = MODELS.DEFAULT_GRAVITY_BETA
        
        # Hub model settings
        self.hub_trip_probability = MODELS.DEFAULT_HUB_TRIP_PROBABILITY
        self.hub_percentage = MODELS.DEFAULT_HUB_PERCENTAGE


class SettingsManager:
    """Handles application settings and configuration"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.config = SimulationConfig()
        self.settings_file = "simulation_settings.json"
    
    def load_settings(self):
        """Load settings from file or defaults"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                
                # Update config with loaded data
                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                self.main_window.add_log_message(f"Settings loaded from {self.settings_file}")
            else:
                self.main_window.add_log_message("Using default settings (no saved settings file found)")
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error loading settings: {str(e)}")
            self.main_window.add_log_message("Using default settings")
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            # Get current settings from UI
            current_settings = self.get_current_settings()
            
            # Update config with current settings
            for key, value in current_settings.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Save to file
            settings_data = {
                'bpr_alpha': self.config.bpr_alpha,
                'bpr_beta': self.config.bpr_beta,
                'hourly_probabilities': self.config.hourly_probabilities,
                'zone_intra_probability': self.config.zone_intra_probability,
                'agent_type_distributions': self.config.agent_type_distributions,
                'gravity_alpha': self.config.gravity_alpha,
                'gravity_beta': self.config.gravity_beta,
                'hub_trip_probability': self.config.hub_trip_probability,
                'hub_percentage': self.config.hub_percentage
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_data, f, indent=4)
            
            self.main_window.add_log_message(f"Settings saved to {self.settings_file}")
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error saving settings: {str(e)}")
    
    def reset_settings_to_defaults(self):
        """Reset all settings to their default values"""
        self.main_window.add_log_message("Resetting all settings to default values...")
        
        try:
            # Reset config to defaults
            self.config = SimulationConfig()
            
            # Update UI with default values
            self.load_settings_into_ui()
            
            self.main_window.add_log_message("All settings successfully reset to default values!")
            self.main_window.add_log_message("Click 'Apply Settings' to use these values in your simulation")
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error resetting settings: {str(e)}")
            import traceback
            self.main_window.add_log_message(f"Details: {traceback.format_exc()}")
    
    def on_reset_button_clicked(self):
        """Handle reset button click with visual feedback"""
        # Change button appearance to show it's been clicked
        original_style = self.main_window.reset_button.styleSheet()
        self.main_window.reset_button.setStyleSheet("QPushButton { background-color: #28A745; color: white; padding: 8px 16px; border-radius: 4px; }")
        self.main_window.reset_button.setText("Resetting...")
        self.main_window.reset_button.setEnabled(False)
        
        # Process the reset
        QApplication.processEvents()  # Update UI immediately
        
        try:
            self.reset_settings_to_defaults()
        finally:
            # Restore button appearance after a short delay
            QTimer.singleShot(1000, lambda: self.restore_reset_button(original_style))
    
    def restore_reset_button(self, original_style):
        """Restore reset button to original state"""
        self.main_window.reset_button.setStyleSheet(original_style)
        self.main_window.reset_button.setText("Reset to Defaults")
        self.main_window.reset_button.setEnabled(True)
    
    def on_apply_button_clicked(self):
        """Handle apply button click with visual feedback"""
        # Change button appearance to show it's been clicked
        original_style = self.main_window.apply_button.styleSheet()
        self.main_window.apply_button.setStyleSheet("QPushButton { background-color: #28A745; color: white; padding: 8px 16px; border-radius: 4px; }")
        self.main_window.apply_button.setText("Applying...")
        self.main_window.apply_button.setEnabled(False)
        
        # Process the application
        QApplication.processEvents()  # Update UI immediately
        
        try:
            self.apply_settings()
        finally:
            # Restore button appearance after a short delay
            QTimer.singleShot(1500, lambda: self.restore_apply_button(original_style))
    
    def restore_apply_button(self, original_style):
        """Restore apply button to original state"""
        self.main_window.apply_button.setStyleSheet(original_style)
        self.main_window.apply_button.setText("Apply Settings")
        self.main_window.apply_button.setEnabled(True)
    
    def load_settings_into_ui(self):
        """Load current settings values into the UI spinboxes"""
        # This method is called during initialization to populate spinboxes with default values
        
        # BPR parameters
        self.main_window.bpr_alpha_spinbox.setValue(self.config.bpr_alpha)
        self.main_window.bpr_beta_spinbox.setValue(self.config.bpr_beta)
        
        # Hourly probabilities
        for hour, prob in self.config.hourly_probabilities.items():
            if hour in self.main_window.hour_probability_spinboxes:
                self.main_window.hour_probability_spinboxes[hour].setValue(prob)
        
        # Zone model
        self.main_window.zone_intra_prob_spinbox.setValue(self.config.zone_intra_probability)
        
        # Activity model
        for agent_type, prob in self.config.agent_type_distributions.items():
            if agent_type in self.main_window.agent_type_spinboxes:
                self.main_window.agent_type_spinboxes[agent_type].setValue(prob)
        
        # Gravity model
        self.main_window.gravity_alpha_spinbox.setValue(self.config.gravity_alpha)
        self.main_window.gravity_beta_spinbox.setValue(self.config.gravity_beta)
        
        # Hub model
        self.main_window.hub_trip_prob_spinbox.setValue(self.config.hub_trip_probability)
        self.main_window.hub_percentage_spinbox.setValue(self.config.hub_percentage)
        
        self.main_window.add_log_message("Settings UI initialized with current values")
    
    def apply_settings(self):
        """Apply the current settings to the simulation models"""
        self.main_window.add_log_message("Applying settings...")
        
        if not self.main_window.graph:
            self.main_window.add_log_message("❌ No graph loaded. Settings will be applied when a simulation starts.")
            return
        
        try:
            # Normalize agent type probabilities
            agent_type_sum = sum(spinbox.value() for spinbox in self.main_window.agent_type_spinboxes.values())
            if agent_type_sum > 0:
                self.main_window.add_log_message(f"Normalizing agent type probabilities (sum was {agent_type_sum:.3f})")
                for spinbox in self.main_window.agent_type_spinboxes.values():
                    spinbox.setValue(spinbox.value() / agent_type_sum)
            
            # Normalize hourly probabilities
            hour_sum = sum(spinbox.value() for spinbox in self.main_window.hour_probability_spinboxes.values())
            if hour_sum > 0:
                self.main_window.add_log_message(f"Normalizing hourly probabilities (sum was {hour_sum:.3f})")
                for spinbox in self.main_window.hour_probability_spinboxes.values():
                    spinbox.setValue(spinbox.value() / hour_sum)
            
            # Apply settings to running simulation if active
            applied_to_simulation = False
            if self.main_window.simulation_thread and self.main_window.simulation_thread.isRunning():
                current_settings = self.get_current_settings()
                self.main_window.simulation_thread.update_runtime_settings(current_settings)
                self.main_window.add_log_message("Settings applied to running simulation")
                applied_to_simulation = True
            
            # Also update the simulation widget's ST selector if it exists
            applied_to_widget = False
            if hasattr(self.main_window.simulation_widget, 'st_selector') and self.main_window.simulation_widget.st_selector:
                self.update_widget_st_selector_parameters()
                self.main_window.add_log_message("Settings applied to visualization widget")
                applied_to_widget = True
            
            # Provide comprehensive feedback
            self.main_window.add_log_message("Settings applied successfully!")
            self.main_window.add_log_message(f"BPR Parameters: α={self.main_window.bpr_alpha_spinbox.value():.3f}, β={self.main_window.bpr_beta_spinbox.value():.1f}")
            self.main_window.add_log_message(f"Zone intra-trip probability: {self.main_window.zone_intra_prob_spinbox.value():.2f}")
            self.main_window.add_log_message(f"Gravity model: α={self.main_window.gravity_alpha_spinbox.value():.1f}, β={self.main_window.gravity_beta_spinbox.value():.1f}")
            self.main_window.add_log_message(f"Hub model: trip_prob={self.main_window.hub_trip_prob_spinbox.value():.2f}, hub_pct={self.main_window.hub_percentage_spinbox.value():.2f}")
            
            # Summary of where settings were applied
            if applied_to_simulation and applied_to_widget:
                self.main_window.add_log_message("Applied to: Running simulation + Visualization widget")
            elif applied_to_simulation:
                self.main_window.add_log_message("Applied to: Running simulation only")
            elif applied_to_widget:
                self.main_window.add_log_message("Applied to: Visualization widget only")
            else:
                self.main_window.add_log_message("Settings saved - will be applied to next simulation")
            
            # Save settings after successful application
            self.save_settings()
                
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error applying settings: {str(e)}")
            import traceback
            self.main_window.add_log_message(f"Details: {traceback.format_exc()}")
    
    def update_widget_st_selector_parameters(self):
        """Update the simulation widget's ST selector parameters"""
        current_mode = self.main_window.selection_combo.currentText()
        st_selector = self.main_window.simulation_widget.st_selector
        
        if current_mode == 'zone' and hasattr(st_selector, 'set_parameters'):
            st_selector.set_parameters(intra_zone_probability=self.main_window.zone_intra_prob_spinbox.value())
        elif current_mode == 'gravity' and hasattr(st_selector, 'set_parameters'):
            st_selector.set_parameters(
                alpha=self.main_window.gravity_alpha_spinbox.value(),
                beta=self.main_window.gravity_beta_spinbox.value()
            )
        elif current_mode == 'hub' and hasattr(st_selector, 'set_parameters'):
            st_selector.set_parameters(
                hub_trip_probability=self.main_window.hub_trip_prob_spinbox.value(),
                hub_percentage=self.main_window.hub_percentage_spinbox.value()
            )
    
    def get_current_settings(self):
        """Get current settings as a dictionary"""
        settings = {
            'bpr_alpha': self.main_window.bpr_alpha_spinbox.value(),
            'bpr_beta': self.main_window.bpr_beta_spinbox.value(),
            'hourly_probabilities': {hour: spinbox.value() for hour, spinbox in self.main_window.hour_probability_spinboxes.items()},
            'zone_intra_probability': self.main_window.zone_intra_prob_spinbox.value(),
            'agent_type_distributions': {agent_type: spinbox.value() for agent_type, spinbox in self.main_window.agent_type_spinboxes.items()},
            'gravity_alpha': self.main_window.gravity_alpha_spinbox.value(),
            'gravity_beta': self.main_window.gravity_beta_spinbox.value(),
            'hub_trip_probability': self.main_window.hub_trip_prob_spinbox.value(),
            'hub_percentage': self.main_window.hub_percentage_spinbox.value()
        }
        return settings
