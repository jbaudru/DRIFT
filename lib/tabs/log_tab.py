from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit


class LogTab(QWidget):
    """Log tab for displaying application messages and events"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setup_ui()
    
    def setup_ui(self):
        """Create the log tab"""
        layout = QVBoxLayout(self)
        
        # Store reference to log text widget for backward compatibility
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
