from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
                             QComboBox, QPushButton, QTableWidget)

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import UI, FILES, STATISTICS


class TripsTab(QWidget):
    """Trip data tab for displaying completed trips and export controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.export_location = None
        self.setup_ui()
    
    def setup_ui(self):
        """Create the trips tab"""
        layout = QVBoxLayout(self)
        
        # Export controls section
        export_frame = self.create_export_controls()
        layout.addWidget(export_frame)
        
        # Trips table section
        layout.addWidget(QLabel(UI.COMPLETED_TRIPS_LABEL))

        # Create trips table
        self.trips_table = self.create_trips_table()
        layout.addWidget(self.trips_table)
    
    def create_export_controls(self):
        """Create the export controls frame"""
        export_frame = QFrame()
        export_frame.setFrameStyle(QFrame.StyledPanel)
        export_frame.setStyleSheet("QFrame { border: 0x; }")
        export_layout = QVBoxLayout(export_frame)
        
        # First row: format selection and location
        export_row1 = QHBoxLayout()
        
        export_row1.addWidget(QLabel(UI.FORMAT_LABEL))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(FILES.EXPORT_FORMATS[:2])  # CSV and JSON only
        export_row1.addWidget(self.export_format_combo)
        
        export_row1.addWidget(QLabel(UI.LOCATION_LABEL))
        
        self.export_location_label = QLabel(UI.DEFAULT_LOCATION_LABEL)
        self.export_location_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        export_row1.addWidget(self.export_location_label)
        
        self.browse_location_button = QPushButton(UI.BROWSE_BUTTON)
        if self.parent_window and hasattr(self.parent_window, 'browse_export_location'):
            self.browse_location_button.clicked.connect(self.parent_window.browse_export_location)
        export_row1.addWidget(self.browse_location_button)
        
        export_layout.addLayout(export_row1)
        
        # Second row: export buttons
        export_row2 = QHBoxLayout()
        
        self.convert_data_button = QPushButton(UI.CONVERT_FORMAT_BUTTON)
        if self.parent_window and hasattr(self.parent_window, 'convert_data_format'):
            self.convert_data_button.clicked.connect(self.parent_window.convert_data_format)
        export_row2.addWidget(self.convert_data_button)
        
        self.export_current_button = QPushButton(UI.EXPORT_CURRENT_DATA_BUTTON)
        self.export_current_button.setStyleSheet("QPushButton { background-color: #007BFF; color: white; padding: 4px 4px; border-radius: 4px; }")
        if self.parent_window and hasattr(self.parent_window, 'export_current_data'):
            self.export_current_button.clicked.connect(self.parent_window.export_current_data)
        self.export_current_button.setEnabled(False)  # Enabled when simulation is paused/finished
        export_row2.addWidget(self.export_current_button)
        
        export_layout.addLayout(export_row2)
        
        return export_frame
    
    def create_trips_table(self):
        """Create the trips table widget"""
        trips_table = QTableWidget()
        trips_table.setColumnCount(10)
        trips_table.setHorizontalHeaderLabels(STATISTICS.TRIP_TABLE_COLUMNS)
        trips_table.horizontalHeader().setStretchLastSection(True)
        trips_table.setAlternatingRowColors(True)
        trips_table.setSelectionBehavior(QTableWidget.SelectRows)

        # Set column widths
        trips_table.setColumnWidth(0, 80)    # Trip ID
        trips_table.setColumnWidth(1, 80)    # Agent ID  
        trips_table.setColumnWidth(2, 100)   # Agent Type
        trips_table.setColumnWidth(3, 100)   # Start Node
        trips_table.setColumnWidth(4, 100)   # End Node
        trips_table.setColumnWidth(5, 100)   # Start Time
        trips_table.setColumnWidth(6, 100)   # Duration
        trips_table.setColumnWidth(7, 100)   # Distance
        trips_table.setColumnWidth(8, 100)   # Avg Speed

        return trips_table
