from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
from ..simulation_widget import SimulationWidget

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import UI


class SimulationTab(QWidget):
    """Simulation tab containing the main simulation view and status information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setup_ui()
    
    def setup_ui(self):
        """Create the simulation tab with full screen view and status info at bottom"""
        layout = QVBoxLayout(self)

        # Create main content layout with simulation widget
        main_layout = QVBoxLayout()
        
        # Create simulation widget container
        sim_container = QWidget()
        sim_layout = QVBoxLayout(sim_container)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create simulation widget (takes most of the space)
        self.simulation_widget = SimulationWidget()
        
        # Connect simulation widget log messages to main window if available
        if self.parent_window and hasattr(self.parent_window, 'add_log_message'):
            self.simulation_widget.add_log_message = self.parent_window.add_log_message
        
        sim_layout.addWidget(self.simulation_widget)
        
        # Add screenshot button overlay in bottom-left corner
        self.screenshot_button = QPushButton()
        self.screenshot_button.setIcon(QIcon("assets/screenshot.png"))
        self.screenshot_button.setIconSize(QSize(15, 15))
        self.screenshot_button.setFixedSize(25, 25)  # Match recenter button size
        self.screenshot_button.setToolTip("Export Screenshot")
        self.screenshot_button.setFlat(True)
        self.screenshot_button.setStyleSheet(
            "QPushButton { "
            "    border: 1px solid #888; "
            "    background: rgba(255, 255, 255, 200); "
            "    border-radius: 4px; "
            "} "
            "QPushButton:hover { "
            "    background: rgba(255, 255, 255, 255); "
            "    border: 1px solid #555; "
            "}"
        )
        self.screenshot_button.clicked.connect(self.export_screenshot)
        self.screenshot_button.setVisible(False)  # Initially hidden until network is loaded
        
        # Position the button in bottom-left corner
        self.screenshot_button.setParent(self.simulation_widget)
        self.screenshot_button.move(10, self.simulation_widget.height() - 45)
        
        main_layout.addWidget(sim_container)
        layout.addLayout(main_layout)

        # Add status bar at the bottom of simulation tab
        self.status_frame = self.create_status_frame()
        layout.addWidget(self.status_frame)
    
    def create_status_frame(self):
        """Create the status information frame at the bottom"""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_frame.setStyleSheet("border: 0px;")
        status_frame.setMaximumHeight(40)
        status_layout = QHBoxLayout(status_frame)

        # Simulation time
        self.sim_time_label = QLabel(UI.SIM_TIME_DEFAULT)
        self.sim_time_label.setStyleSheet("color: #000000;")
        status_layout.addWidget(self.sim_time_label)

        # Real time elapsed
        self.real_time_label = QLabel(UI.RUNNING_TIME_DEFAULT)
        self.real_time_label.setStyleSheet("color: #000000;")
        status_layout.addWidget(self.real_time_label)

        # Moving agents
        self.moving_agents_label = QLabel(UI.MOVING_AGENTS_DEFAULT)
        self.moving_agents_label.setStyleSheet("color: #000000;")
        status_layout.addWidget(self.moving_agents_label)

        # Network utilization
        self.network_util_label = QLabel(UI.NETWORK_UTIL_DEFAULT)
        self.network_util_label.setStyleSheet("color: #000000;")
        status_layout.addWidget(self.network_util_label)
        
        # Performance status with optimization info
        self.performance_label = QLabel(UI.PERFORMANCE_DEFAULT)
        self.performance_label.setStyleSheet("color: #000000;")
        self.performance_label.setToolTip(UI.PERFORMANCE_TOOLTIP)
        status_layout.addWidget(self.performance_label)

        status_layout.addStretch()
        return status_frame
    
    def export_screenshot(self):
        """Export a screenshot of the current simulation state"""
        # Open file dialog to select save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Screenshot",
            "simulation_screenshot.png",
            "PNG Images (*.png);;All Files (*)"
        )
        
        if file_path:
            # Temporarily hide the screenshot button to exclude it from the capture
            button_was_visible = self.screenshot_button.isVisible()
            self.screenshot_button.setVisible(False)
            
            # Temporarily hide the recenter button in simulation widget
            recenter_was_visible = False
            if hasattr(self.simulation_widget, 'show_reset_button'):
                recenter_was_visible = self.simulation_widget.show_reset_button
                self.simulation_widget.show_reset_button = False
            
            # Allow the UI to update before capturing
            self.screenshot_button.repaint()
            self.simulation_widget.update()  # Force repaint of simulation widget
            
            # Capture the simulation widget as an image
            pixmap = self.simulation_widget.grab()
            
            # Restore the button visibility states
            self.screenshot_button.setVisible(button_was_visible)
            if hasattr(self.simulation_widget, 'show_reset_button'):
                self.simulation_widget.show_reset_button = recenter_was_visible
                self.simulation_widget.update()  # Restore recenter button
            
            # Save the image
            if pixmap.save(file_path, "PNG"):
                if self.parent_window and hasattr(self.parent_window, 'add_log_message'):
                    self.parent_window.add_log_message(f"Screenshot saved to: {file_path}")
            else:
                if self.parent_window and hasattr(self.parent_window, 'add_log_message'):
                    self.parent_window.add_log_message(f"Failed to save screenshot to: {file_path}")
    
    def resizeEvent(self, event):
        """Handle resize event to reposition screenshot button"""
        super().resizeEvent(event)
        if hasattr(self, 'screenshot_button') and hasattr(self, 'simulation_widget'):
            # Reposition the screenshot button in bottom-left corner
            button_y = self.simulation_widget.height() - self.screenshot_button.height() - 10
            self.screenshot_button.move(10, button_y)
    
    def set_network_loaded(self, loaded):
        """Show/hide screenshot button based on network load state"""
        if hasattr(self, 'screenshot_button'):
            self.screenshot_button.setVisible(loaded)
