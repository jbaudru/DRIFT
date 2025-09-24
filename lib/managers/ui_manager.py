"""
UI Manager Module
Handles user interface initialization and management for the main application window
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QSpinBox, QLabel, QTabWidget)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
from lib.tabs import SimulationTab, LogTab, TripsTab, StatisticsTab, SettingsTab
from config import UI, SIMULATION


class UIManager:
    """Handles UI initialization and management for the main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.tabs = {}
        self.toolbar_widgets = {}
        
    def init_ui(self):
        """Initialize the user interface"""
        self.main_window.setWindowTitle(UI.APP_NAME)
        self.main_window.setGeometry(UI.WINDOW_X, UI.WINDOW_Y, UI.WINDOW_WIDTH, UI.WINDOW_HEIGHT)

        # Set application icon
        self._set_window_icon()

        # Create central widget
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Create toolbar
        toolbar_layout = self._create_toolbar()
        main_layout.addLayout(toolbar_layout)

        # Create main tab widget
        self.main_tab_widget = self._create_tabs()
        main_layout.addWidget(self.main_tab_widget)

    def _set_window_icon(self):
        """Set the application window icon"""
        app_icon = QIcon()
        for size in UI.ICON_SIZES:
            app_icon.addFile(f'{UI.ICON_BASE_PATH}/{size[0]}x{size[1]}.ico', QSize(size[0], size[1]))
        self.main_window.setWindowIcon(app_icon)
    
    def _create_toolbar(self):
        """Create and configure the toolbar"""
        toolbar_layout = QHBoxLayout()

        # Load Graph button
        self.toolbar_widgets['load_button'] = QPushButton(UI.LOAD_GRAPH_BUTTON)
        self.toolbar_widgets['load_button'].clicked.connect(self.main_window.load_graph)
        toolbar_layout.addWidget(self.toolbar_widgets['load_button'])

        # Selection mode combo box
        toolbar_layout.addWidget(QLabel(UI.ST_MODEL_LABEL))
        self.toolbar_widgets['selection_combo'] = QComboBox()
        self.toolbar_widgets['selection_combo'].addItems(SIMULATION.SELECTION_MODES)
        self.toolbar_widgets['selection_combo'].setCurrentText(SIMULATION.DEFAULT_SELECTION_MODE)
        self.toolbar_widgets['selection_combo'].currentTextChanged.connect(self.main_window.on_selection_mode_changed)
        # Initially disabled since no graph is loaded
        self.toolbar_widgets['selection_combo'].setEnabled(False)
        toolbar_layout.addWidget(self.toolbar_widgets['selection_combo'])

        # Number of agents spin box
        toolbar_layout.addWidget(QLabel(UI.AGENTS_LABEL))
        self.toolbar_widgets['agents_spinbox'] = QSpinBox()
        self.toolbar_widgets['agents_spinbox'].setRange(SIMULATION.MIN_AGENTS, SIMULATION.MAX_AGENTS)
        self.toolbar_widgets['agents_spinbox'].setValue(SIMULATION.DEFAULT_NUM_AGENTS)
        toolbar_layout.addWidget(self.toolbar_widgets['agents_spinbox'])

        # Duration spin box
        toolbar_layout.addWidget(QLabel(UI.HOURS_LABEL))
        self.toolbar_widgets['duration_spinbox'] = QSpinBox()
        self.toolbar_widgets['duration_spinbox'].setRange(SIMULATION.MIN_DURATION, SIMULATION.MAX_DURATION)
        self.toolbar_widgets['duration_spinbox'].setValue(SIMULATION.DEFAULT_DURATION_HOURS)
        toolbar_layout.addWidget(self.toolbar_widgets['duration_spinbox'])

        # Speed multiplier spin box
        toolbar_layout.addWidget(QLabel(UI.SPEED_LABEL))
        self.toolbar_widgets['speed_spinbox'] = QSpinBox()
        self.toolbar_widgets['speed_spinbox'].setRange(SIMULATION.MIN_SPEED, SIMULATION.MAX_SPEED)
        self.toolbar_widgets['speed_spinbox'].setValue(SIMULATION.DEFAULT_SPEED_MULTIPLIER)
        self.toolbar_widgets['speed_spinbox'].setSuffix(SIMULATION.SPEED_SUFFIX)
        self.toolbar_widgets['speed_spinbox'].setToolTip(SIMULATION.SPEED_TOOLTIP)
        toolbar_layout.addWidget(self.toolbar_widgets['speed_spinbox'])

        # Start/Stop simulation button
        self.toolbar_widgets['start_button'] = QPushButton()
        self.toolbar_widgets['start_button'].setIcon(QIcon(UI.PLAY_ICON_PATH))
        self.toolbar_widgets['start_button'].setIconSize(QSize(20, 20))
        self.toolbar_widgets['start_button'].setFixedSize(30, 30)
        self.toolbar_widgets['start_button'].setToolTip("Start Simulation")
        self.toolbar_widgets['start_button'].setFlat(True)
        self.toolbar_widgets['start_button'].setStyleSheet("QPushButton { border: none; background: transparent; }")
        self.toolbar_widgets['start_button'].clicked.connect(self.main_window.simulation_controller.toggle_simulation)
        self.toolbar_widgets['start_button'].setEnabled(False)
        toolbar_layout.addWidget(self.toolbar_widgets['start_button'])

        # Pause/Resume button (visible only when running)
        self.toolbar_widgets['pause_button'] = QPushButton()
        self.toolbar_widgets['pause_button'].setIcon(QIcon(UI.PAUSE_ICON_PATH))
        self.toolbar_widgets['pause_button'].setIconSize(QSize(20, 20))
        self.toolbar_widgets['pause_button'].setFixedSize(30, 30)
        self.toolbar_widgets['pause_button'].setToolTip("Pause")
        self.toolbar_widgets['pause_button'].setFlat(True)
        self.toolbar_widgets['pause_button'].setStyleSheet("QPushButton { border: none; background: transparent; }")
        self.toolbar_widgets['pause_button'].clicked.connect(self.main_window.simulation_controller.toggle_pause)
        self.toolbar_widgets['pause_button'].setVisible(False)
        toolbar_layout.addWidget(self.toolbar_widgets['pause_button'])

        toolbar_layout.addStretch()
        return toolbar_layout

    def _create_tabs(self):
        """Create and configure the main tab widget"""
        main_tab_widget = QTabWidget()

        # Tab 1: Simulation
        self.tabs['simulation'] = SimulationTab(self.main_window)
        main_tab_widget.addTab(self.tabs['simulation'], UI.SIMULATION_TAB)

        # Tab 2: Trip Data
        self.tabs['trips'] = TripsTab(self.main_window)
        main_tab_widget.addTab(self.tabs['trips'], UI.TRIP_DATA_TAB)

        # Tab 3: Statistics
        self.tabs['statistics'] = StatisticsTab(self.main_window)
        main_tab_widget.addTab(self.tabs['statistics'], UI.STATISTICS_TAB)

        # Tab 4: Settings
        self.tabs['settings'] = SettingsTab(self.main_window)
        main_tab_widget.addTab(self.tabs['settings'], UI.SETTINGS_TAB)

        # Tab 5: Log (moved to last position)
        self.tabs['log'] = LogTab(self.main_window)
        main_tab_widget.addTab(self.tabs['log'], UI.LOG_TAB)

        return main_tab_widget

    def get_toolbar_widget(self, widget_name):
        """Get a toolbar widget by name for backward compatibility"""
        return self.toolbar_widgets.get(widget_name)
    
    def get_tab(self, tab_name):
        """Get a tab by name"""
        return self.tabs.get(tab_name)
    
    def setup_widget_references(self):
        """Setup widget references on the main window for backward compatibility"""
        # Toolbar widget references
        for widget_name, widget in self.toolbar_widgets.items():
            setattr(self.main_window, widget_name, widget)
        
        # Main tab widget reference
        self.main_window.main_tab_widget = self.main_tab_widget
        
        # Tab references
        for tab_name, tab in self.tabs.items():
            setattr(self.main_window, f"{tab_name}_tab", tab)
        
        # Store references to simulation tab widgets for backward compatibility
        if 'simulation' in self.tabs:
            sim_tab = self.tabs['simulation']
            self.main_window.simulation_widget = sim_tab.simulation_widget
            self.main_window.sim_time_label = sim_tab.sim_time_label
            self.main_window.real_time_label = sim_tab.real_time_label
            self.main_window.moving_agents_label = sim_tab.moving_agents_label
            self.main_window.network_util_label = sim_tab.network_util_label
            self.main_window.performance_label = sim_tab.performance_label

        # Store reference to log text widget for backward compatibility
        if 'log' in self.tabs:
            self.main_window.log_text = self.tabs['log'].log_text

        # Store references to trips tab widgets for backward compatibility
        if 'trips' in self.tabs:
            trips_tab = self.tabs['trips']
            self.main_window.export_format_combo = trips_tab.export_format_combo
            self.main_window.browse_location_button = trips_tab.browse_location_button
            self.main_window.convert_data_button = trips_tab.convert_data_button
            self.main_window.export_current_button = trips_tab.export_current_button
            self.main_window.trips_table = trips_tab.trips_table
            self.main_window.export_location_label = trips_tab.export_location_label

        # Store references to statistics tab widgets for backward compatibility
        if 'statistics' in self.tabs:
            stats_tab = self.tabs['statistics']
            self.main_window.network_stats_grid = stats_tab.network_stats_grid
            self.main_window.network_stats_labels = stats_tab.network_stats_labels
            self.main_window.export_plots_button = stats_tab.export_plots_button
            self.main_window.stats_figures = stats_tab.stats_figures
            self.main_window.stats_canvases = stats_tab.stats_canvases
            self.main_window.stats_axes = stats_tab.stats_axes

        # Store references to settings tab widgets for backward compatibility
        if 'settings' in self.tabs:
            settings_tab = self.tabs['settings']
            self.main_window.bpr_alpha_spinbox = settings_tab.bpr_alpha_spinbox
            self.main_window.bpr_beta_spinbox = settings_tab.bpr_beta_spinbox
            self.main_window.hour_probability_spinboxes = settings_tab.hour_probability_spinboxes
            self.main_window.zone_intra_prob_spinbox = settings_tab.zone_intra_prob_spinbox
            self.main_window.agent_type_spinboxes = settings_tab.agent_type_spinboxes
            self.main_window.gravity_alpha_spinbox = settings_tab.gravity_alpha_spinbox
            self.main_window.gravity_beta_spinbox = settings_tab.gravity_beta_spinbox
            self.main_window.hub_trip_prob_spinbox = settings_tab.hub_trip_prob_spinbox
            self.main_window.hub_percentage_spinbox = settings_tab.hub_percentage_spinbox
            self.main_window.reset_button = settings_tab.reset_button
            self.main_window.apply_button = settings_tab.apply_button
    
    def update_window_title(self, filename=None):
        """Update the window title with optional filename"""
        if filename:
            self.main_window.setWindowTitle(f"DRIFT - {filename}")
        else:
            self.main_window.setWindowTitle(UI.APP_NAME)