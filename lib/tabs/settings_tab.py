from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
                             QGridLayout, QPushButton, QScrollArea, QDoubleSpinBox)
from PyQt5.QtCore import Qt

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import UI, MODELS, NETWORK


class SettingsTab(QWidget):
    """Settings tab for configuring simulation parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setup_ui()
    
    def setup_ui(self):
        """Create the comprehensive settings tab"""
        self.setStyleSheet("QWidget { background-color: transparent; }")
        main_layout = QVBoxLayout(self)
        
        # Create scroll area for the settings
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("QScrollArea { border: 0px;  }")
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # === SIMULATION PARAMETERS SECTION ===
        sim_group = self.create_simulation_parameters_section()
        scroll_layout.addWidget(sim_group)
        
        # === SOURCE-TARGET MODEL PARAMETERS SECTION ===
        models_group = self.create_model_parameters_section()
        scroll_layout.addWidget(models_group)
        
        # === BUTTONS SECTION ===
        buttons_layout = self.create_buttons_section()
        scroll_layout.addLayout(buttons_layout)
        
        # Set up scroll area
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        main_layout.addWidget(scroll_area)
    
    def create_simulation_parameters_section(self):
        """Create the simulation parameters section"""
        sim_group = QFrame()
        sim_group.setFrameStyle(QFrame.StyledPanel)
        sim_group.setStyleSheet("QFrame { background-color: #F8F9FA; border: 0px solid #E9ECEF; border-radius: 5px; }")
        sim_layout = QVBoxLayout(sim_group)
        
        sim_title = QLabel(UI.SIMULATION_PARAMETERS_LABEL)
        sim_title.setStyleSheet("font-size: 14px; margin-bottom: 8px; border: 0px;")
        sim_layout.addWidget(sim_title)
        
        # Bureau of Public Roads (BPR) Parameters
        bpr_frame = self.create_bpr_parameters()
        sim_layout.addWidget(bpr_frame)
        
        # Trip Generation by Hour
        trip_hour_frame = self.create_hourly_trip_parameters()
        sim_layout.addWidget(trip_hour_frame)
        
        return sim_group
    
    def create_bpr_parameters(self):
        """Create BPR (Bureau of Public Roads) parameters section"""
        bpr_frame = QFrame()
        bpr_layout = QGridLayout(bpr_frame)
        
        bpr_label = QLabel(UI.TRAFFIC_FLOW_LABEL)
        bpr_label.setStyleSheet("font-weight: bold; border: 0px;")
        bpr_layout.addWidget(bpr_label, 0, 0, 1, 2)
        
        # Alpha parameter
        bpr_layout.addWidget(QLabel(UI.ALPHA_CONGESTION_LABEL), 1, 0)
        self.bpr_alpha_spinbox = QDoubleSpinBox()
        self.bpr_alpha_spinbox.setRange(0.01, 2.0)
        self.bpr_alpha_spinbox.setValue(0.15)
        self.bpr_alpha_spinbox.setSingleStep(0.01)
        self.bpr_alpha_spinbox.setDecimals(3)
        self.bpr_alpha_spinbox.setToolTip(UI.BPR_ALPHA_TOOLTIP)
        bpr_layout.addWidget(self.bpr_alpha_spinbox, 1, 1)
        
        # Beta parameter
        bpr_layout.addWidget(QLabel(UI.BETA_DISTANCE_LABEL), 2, 0)
        self.bpr_beta_spinbox = QDoubleSpinBox()
        self.bpr_beta_spinbox.setRange(1.0, 8.0)
        self.bpr_beta_spinbox.setValue(4.0)
        self.bpr_beta_spinbox.setSingleStep(0.1)
        self.bpr_beta_spinbox.setDecimals(1)
        self.bpr_beta_spinbox.setToolTip(UI.BPR_BETA_TOOLTIP)
        bpr_layout.addWidget(self.bpr_beta_spinbox, 2, 1)
        
        return bpr_frame
    
    def create_hourly_trip_parameters(self):
        """Create hourly trip generation parameters section"""
        trip_hour_frame = QFrame()
        trip_hour_layout = QVBoxLayout(trip_hour_frame)
        
        trip_hour_label = QLabel(UI.TRIP_GENERATION_LABEL)
        trip_hour_label.setStyleSheet("font-weight: bold; margin-top: 10px; border: 0px;")
        trip_hour_layout.addWidget(trip_hour_label)
        
        # Create a grid for hourly probabilities (4 rows x 6 columns)
        hours_grid = QGridLayout()
        self.hour_probability_spinboxes = {}
        
        # Use default probabilities from config
        from config import MODELS
        default_hourly_probs = MODELS.DEFAULT_HOURLY_PROBABILITIES
        
        for hour in range(24):
            row = hour // 6
            col = hour % 6
            
            hour_label = QLabel(f"{hour:02d}:00")
            hour_label.setStyleSheet("font-size: 10px;")
            hours_grid.addWidget(hour_label, row * 2, col)
            
            hour_spinbox = QDoubleSpinBox()
            hour_spinbox.setRange(0.0, 1.0)
            hour_spinbox.setValue(default_hourly_probs[hour])
            hour_spinbox.setSingleStep(0.01)
            hour_spinbox.setDecimals(3)
            hour_spinbox.setMaximumWidth(80)
            hour_spinbox.setToolTip(UI.TRIP_GENERATION_HOUR_TOOLTIP_TEMPLATE.format(hour))
            hours_grid.addWidget(hour_spinbox, row * 2 + 1, col)
            
            self.hour_probability_spinboxes[hour] = hour_spinbox
        
        trip_hour_layout.addLayout(hours_grid)
        return trip_hour_frame
    
    def create_model_parameters_section(self):
        """Create the source-target model parameters section"""
        models_group = QFrame()
        models_group.setFrameStyle(QFrame.StyledPanel)
        models_group.setStyleSheet("QFrame { background-color: #F8F9FA; border: 0px solid #E9ECEF; border-radius: 5px; }")
        models_layout = QVBoxLayout(models_group)
        
        models_title = QLabel(UI.ST_MODEL_PARAMETERS_LABEL)
        models_title.setStyleSheet("font-size: 14px; margin-bottom: 8px; border: 0px;")
        models_layout.addWidget(models_title)
        
        # Zone-Based Model Parameters
        zone_frame = self.create_zone_parameters()
        models_layout.addWidget(zone_frame)
        
        # Activity-Based Model Parameters
        activity_frame = self.create_activity_parameters()
        models_layout.addWidget(activity_frame)
        
        # Gravity Model Parameters
        gravity_frame = self.create_gravity_parameters()
        models_layout.addWidget(gravity_frame)
        
        # Hub-and-Spoke Model Parameters
        hub_frame = self.create_hub_parameters()
        models_layout.addWidget(hub_frame)
        
        return models_group
    
    def create_zone_parameters(self):
        """Create zone-based model parameters"""
        zone_frame = QFrame()
        zone_layout = QGridLayout(zone_frame)
        
        zone_label = QLabel(UI.ZONE_BASED_MODEL_LABEL)
        zone_label.setStyleSheet("font-weight: bold; border: 0px;")
        zone_layout.addWidget(zone_label, 0, 0, 1, 2)
        
        zone_layout.addWidget(QLabel(UI.INTRA_ZONE_PROB_LABEL), 1, 0)
        self.zone_intra_prob_spinbox = QDoubleSpinBox()
        self.zone_intra_prob_spinbox.setRange(0.0, 1.0)
        self.zone_intra_prob_spinbox.setValue(0.6)
        self.zone_intra_prob_spinbox.setSingleStep(0.05)
        self.zone_intra_prob_spinbox.setDecimals(2)
        self.zone_intra_prob_spinbox.setToolTip(UI.ZONE_INTRA_TOOLTIP)
        zone_layout.addWidget(self.zone_intra_prob_spinbox, 1, 1)
        
        return zone_frame
    
    def create_activity_parameters(self):
        """Create activity-based model parameters"""
        activity_frame = QFrame()
        activity_layout = QGridLayout(activity_frame)
        
        activity_label = QLabel(UI.ACTIVITY_BASED_MODEL_LABEL)
        activity_label.setStyleSheet("font-weight: bold; margin-top: 10px; border: 0px;")
        activity_layout.addWidget(activity_label, 0, 0, 1, 2)
        
        # Agent type distributions
        agent_types = ['commuter', 'delivery', 'leisure', 'business']
        default_distributions = {'commuter': 0.4, 'delivery': 0.2, 'leisure': 0.25, 'business': 0.15}
        self.agent_type_spinboxes = {}
        
        for i, agent_type in enumerate(agent_types):
            activity_layout.addWidget(QLabel(f"{agent_type.capitalize()} probability:"), i + 1, 0)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 1.0)
            spinbox.setValue(default_distributions[agent_type])
            spinbox.setSingleStep(0.05)
            spinbox.setDecimals(2)
            spinbox.setToolTip(UI.AGENT_TYPE_PROBABILITY_TOOLTIP_TEMPLATE.format(agent_type))
            activity_layout.addWidget(spinbox, i + 1, 1)
            self.agent_type_spinboxes[agent_type] = spinbox
        
        return activity_frame
    
    def create_gravity_parameters(self):
        """Create gravity model parameters"""
        gravity_frame = QFrame()
        gravity_layout = QGridLayout(gravity_frame)
        
        gravity_label = QLabel(UI.GRAVITY_MODEL_LABEL)
        gravity_label.setStyleSheet("font-weight: bold; margin-top: 10px; border: 0px;")
        gravity_layout.addWidget(gravity_label, 0, 0, 1, 2)
        
        gravity_layout.addWidget(QLabel(UI.ATTRACTION_FACTOR_LABEL), 1, 0)
        self.gravity_alpha_spinbox = QDoubleSpinBox()
        self.gravity_alpha_spinbox.setRange(0.1, 5.0)
        self.gravity_alpha_spinbox.setValue(1.0)
        self.gravity_alpha_spinbox.setSingleStep(0.1)
        self.gravity_alpha_spinbox.setDecimals(1)
        self.gravity_alpha_spinbox.setToolTip(UI.GRAVITY_ALPHA_TOOLTIP)
        gravity_layout.addWidget(self.gravity_alpha_spinbox, 1, 1)
        
        gravity_layout.addWidget(QLabel(UI.DISTANCE_DECAY_LABEL), 2, 0)
        self.gravity_beta_spinbox = QDoubleSpinBox()
        self.gravity_beta_spinbox.setRange(0.5, 8.0)
        self.gravity_beta_spinbox.setValue(2.0)
        self.gravity_beta_spinbox.setSingleStep(0.1)
        self.gravity_beta_spinbox.setDecimals(1)
        self.gravity_beta_spinbox.setToolTip(UI.GRAVITY_BETA_TOOLTIP)
        gravity_layout.addWidget(self.gravity_beta_spinbox, 2, 1)
        
        return gravity_frame
    
    def create_hub_parameters(self):
        """Create hub-and-spoke model parameters"""
        hub_frame = QFrame()
        hub_layout = QGridLayout(hub_frame)
        
        hub_label = QLabel(UI.HUB_SPOKE_MODEL_LABEL)
        hub_label.setStyleSheet("font-weight: bold; margin-top: 10px; border: 0px;")
        hub_layout.addWidget(hub_label, 0, 0, 1, 2)
        
        hub_layout.addWidget(QLabel(UI.HUB_TRIP_PROB_LABEL), 1, 0)
        self.hub_trip_prob_spinbox = QDoubleSpinBox()
        self.hub_trip_prob_spinbox.setRange(0.0, 1.0)
        self.hub_trip_prob_spinbox.setValue(0.7)
        self.hub_trip_prob_spinbox.setSingleStep(0.05)
        self.hub_trip_prob_spinbox.setDecimals(2)
        self.hub_trip_prob_spinbox.setToolTip(UI.HUB_TRIP_TOOLTIP)
        hub_layout.addWidget(self.hub_trip_prob_spinbox, 1, 1)
        
        hub_layout.addWidget(QLabel(UI.HUB_PERCENTAGE_LABEL), 2, 0)
        self.hub_percentage_spinbox = QDoubleSpinBox()
        self.hub_percentage_spinbox.setRange(0.05, 0.5)
        self.hub_percentage_spinbox.setValue(0.30)
        self.hub_percentage_spinbox.setSingleStep(0.01)
        self.hub_percentage_spinbox.setDecimals(2)
        self.hub_percentage_spinbox.setToolTip(UI.HUB_PERCENTAGE_TOOLTIP)
        hub_layout.addWidget(self.hub_percentage_spinbox, 2, 1)
        
        return hub_frame
    
    def create_buttons_section(self):
        """Create the buttons section"""
        buttons_layout = QHBoxLayout()
        
        # Reset to defaults button
        self.reset_button = QPushButton(UI.RESET_BUTTON)
        self.reset_button.setStyleSheet("QPushButton { background-color: #6C757D; color: white; padding: 8px 16px; border-radius: 4px; }")
        if self.parent_window and hasattr(self.parent_window, 'settings_manager'):
            self.reset_button.clicked.connect(self.parent_window.settings_manager.on_reset_button_clicked)
        buttons_layout.addWidget(self.reset_button)
        
        buttons_layout.addStretch()
        
        # Apply settings button
        self.apply_button = QPushButton(UI.APPLY_BUTTON)
        self.apply_button.setStyleSheet("QPushButton { background-color: #007BFF; color: white; padding: 8px 16px; border-radius: 4px; }")
        if self.parent_window and hasattr(self.parent_window, 'settings_manager'):
            self.apply_button.clicked.connect(self.parent_window.settings_manager.on_apply_button_clicked)
        buttons_layout.addWidget(self.apply_button)
        
        return buttons_layout
