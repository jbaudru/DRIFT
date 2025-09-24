from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
                             QGridLayout, QPushButton, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import COLORS, UI


class StatisticsTab(QWidget):
    """Statistics tab with network statistics and real-time plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setup_ui()
    
    def setup_ui(self):
        """Create the statistics tab with network statistics and real-time plots"""
        self.setStyleSheet("QWidget { background-color: transparent; }")
        layout = QVBoxLayout(self)
        
        # Create a scroll area to allow scrolling when content is too large
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: 0px;  }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # === NETWORK GENERAL STATISTICS SECTION ===
        network_stats_frame = self.create_network_stats_section()
        scroll_layout.addWidget(network_stats_frame)
        
        # Add export button section
        export_frame = self.create_export_section()
        scroll_layout.addWidget(export_frame)
        
        # === REAL-TIME PLOTS SECTION ===
        plots_widget = self.create_plots_section()
        scroll_layout.addWidget(plots_widget)
        
        # Set up scroll area
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(scroll_area)
    
    def create_network_stats_section(self):
        """Create the network statistics section"""
        network_stats_frame = QFrame()
        network_stats_frame.setFrameStyle(QFrame.StyledPanel)
        network_stats_frame.setStyleSheet("QFrame { border: 0px; border-radius: 5px; margin: 5px; }")
        
        network_stats_layout = QVBoxLayout(network_stats_frame)
        
        # Create a grid layout for network statistics
        self.network_stats_grid = QGridLayout()
        
        # Network statistics labels (will be populated when graph is loaded)
        self.network_stats_labels = {}
        stats_names = [
            ('nodes', UI.NODES_LABEL),
            ('edges', UI.EDGES_LABEL), 
            ('avg_degree', UI.AVG_DEGREE_LABEL),
            ('density', UI.DENSITY_LABEL),
            ('components', UI.COMPONENTS_LABEL),
            ('diameter', UI.DIAMETER_LABEL),
            ('clustering', UI.CLUSTERING_LABEL)
        ]
        
        for i, (key, label_text) in enumerate(stats_names):
            row, col = i // 4, (i % 4) * 2
            label = QLabel(label_text)
            label.setStyleSheet("border: 0px;")
            value_label = QLabel("N/A")
            value_label.setStyleSheet("border: 0px;")
            self.network_stats_labels[key] = value_label
            self.network_stats_grid.addWidget(label, row, col)
            self.network_stats_grid.addWidget(value_label, row, col + 1)
        
        network_stats_layout.addLayout(self.network_stats_grid)
        return network_stats_frame
    
    def create_export_section(self):
        """Create the export button section"""
        export_frame = QFrame()
        export_frame.setFrameStyle(QFrame.StyledPanel)
        export_frame.setStyleSheet("QFrame { border: 0x solid #E9ECEF; }")
        export_layout = QHBoxLayout(export_frame)
        
        export_spacer = QWidget()
        export_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        export_layout.addWidget(export_spacer)
        
        self.export_plots_button = QPushButton(UI.EXPORT_ALL_PLOTS_BUTTON)
        if self.parent_window and hasattr(self.parent_window, 'export_plots'):
            self.export_plots_button.clicked.connect(self.parent_window.export_plots)
        self.export_plots_button.setEnabled(False)  # Enabled when simulation is paused/finished
        self.export_plots_button.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.EXPORT_BUTTON_COLOR}; "
            f"color: {COLORS.EXPORT_BUTTON_TEXT_COLOR}; "
            f"padding: {COLORS.BUTTON_PADDING}; "
            f"border-radius: {COLORS.BUTTON_BORDER_RADIUS}; }}"
        )
        export_layout.addWidget(self.export_plots_button)
        
        export_layout.addWidget(QWidget())  # Right spacer
        
        return export_frame
    
    def create_plots_section(self):
        """Create the real-time plots section"""
        # Create a grid layout for the plots (2 columns x 4 rows now)
        plots_widget = QWidget()
        plots_widget.setMinimumHeight(1200)  # Increased height for 4 rows
        plots_layout = QGridLayout(plots_widget)
        
        # Set matplotlib style for better appearance
        mplstyle.use('fast')
        plt.style.use('default')
        
        # Create the 7 subplots with matplotlib
        self.stats_figures = []
        self.stats_canvases = []
        self.stats_axes = []
        
        # Plot configurations
        plot_configs = [
            (0, 0, UI.MOVING_AGENTS_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.MOVING_AGENTS_AXIS_LABEL),
            (0, 1, UI.NETWORK_UTIL_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.NETWORK_UTIL_AXIS_LABEL),
            (1, 0, UI.AVG_SPEED_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.AVG_SPEED_AXIS_LABEL),
            (1, 1, UI.AVG_DISTANCE_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.AVG_DISTANCE_AXIS_LABEL),
            (2, 0, UI.AVG_DURATION_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.AVG_DURATION_AXIS_LABEL),
            (2, 1, UI.AVG_NODES_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.AVG_NODES_AXIS_LABEL),
            (3, 0, UI.AGENT_TYPE_DIST_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.AGENT_TYPE_PERCENTAGE_AXIS_LABEL),
            (3, 1, UI.TRIP_COUNT_PLOT_TITLE, UI.TIME_AXIS_LABEL, UI.TRIP_COUNT_AXIS_LABEL)
        ]
        
        for i, (row, col, title, xlabel, ylabel) in enumerate(plot_configs):
            fig = Figure(figsize=(6, 7), dpi=80)
            fig.subplots_adjust(left=0.12, bottom=0.25, right=0.95, top=0.8)
            fig.set_facecolor('none')
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color:transparent;")
            canvas.setMinimumHeight(250)  # Set minimum height for each canvas
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            ax = fig.add_subplot(111)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            self.stats_figures.append(fig)
            self.stats_canvases.append(canvas)
            self.stats_axes.append(ax)
            plots_layout.addWidget(canvas, row, col)
        
        # Adjust spacing between plots for better visibility
        plots_layout.setRowStretch(0, 1)  # Allow rows to expand
        plots_layout.setRowStretch(1, 1)
        plots_layout.setRowStretch(2, 1)
        plots_layout.setRowStretch(3, 1)
        plots_layout.setColumnStretch(0, 1)  # Allow columns to expand
        plots_layout.setColumnStretch(1, 1)
        
        return plots_widget
