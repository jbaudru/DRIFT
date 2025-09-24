"""
Loading Spinner Widget Module
A reusable loading spinner that can be overlaid on top of other widgets during long operations.
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush
import math


class LoadingSpinner(QWidget):
    """
    A loading spinner widget that can be overlaid on top of other widgets.
    Features a semi-transparent background with an animated spinner and optional text.
    """
    
    def __init__(self, parent=None, text="Loading...", spinner_size=50, show_text=True):
        super().__init__(parent)
        
        self.text = text
        self.spinner_size = spinner_size
        self.angle = 0
        self.dot_count = 8
        self.animation_speed = 100  # milliseconds
        self.show_text = show_text
        
        # Setup widget properties
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100);")  # Semi-transparent background
        
        # Setup layout and label (only if text should be shown)
        if self.show_text:
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignCenter)
            
            self.text_label = QLabel(self.text)
            self.text_label.setAlignment(Qt.AlignCenter)
            self.text_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    background-color: transparent;
                    padding: 10px;
                }
            """)
            
            layout.addWidget(self.text_label)
            self.setLayout(layout)
        else:
            self.text_label = None
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        
        # Initially hidden
        self.hide()
    
    def update_animation(self):
        """Update the spinner animation"""
        self.angle = (self.angle + 45) % 360
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw the spinner"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate center position
        center_x = self.width() // 2
        center_y = self.height() // 2
        if self.show_text:
            center_y -= 30  # Offset up to account for text below
        
        # Draw spinner dots
        for i in range(self.dot_count):
            angle_deg = (self.angle + i * 45) % 360
            angle_rad = math.radians(angle_deg)
            
            # Calculate dot position
            radius = self.spinner_size // 2
            x = center_x + radius * math.cos(angle_rad)
            y = center_y + radius * math.sin(angle_rad)
            
            # Calculate opacity based on position (trailing effect)
            opacity = 0.3 + 0.7 * (i / self.dot_count)
            
            # Set color with opacity - blue color instead of white
            color = QColor(0, 120, 255, int(255 * opacity))  # Blue color
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            
            # Draw dot
            dot_size = 8  # Slightly larger dots for better visibility
            painter.drawEllipse(int(x - dot_size//2), int(y - dot_size//2), dot_size, dot_size)
    
    def show_spinner(self, text=None):
        """Show the spinner with optional custom text"""
        if text and self.show_text and self.text_label:
            self.text = text
            self.text_label.setText(text)
        
        # Resize to parent if available and position correctly
        if self.parent():
            self.resize(self.parent().size())
            # Position at (0,0) relative to parent to cover entire area
            self.move(0, 0)
        
        self.show()
        self.raise_()  # Bring to front
        self.timer.start(self.animation_speed)
        
        # Ensure the spinner is visible by updating the parent
        if self.parent():
            self.parent().update()
    
    def hide_spinner(self):
        """Hide the spinner and stop animation"""
        self.timer.stop()
        self.hide()
    
    def update_text(self, text):
        """Update the spinner text while it's showing"""
        if self.show_text and self.text_label:
            self.text = text
            self.text_label.setText(text)
    
    def resizeEvent(self, event):
        """Handle widget resize to maintain overlay coverage"""
        super().resizeEvent(event)
        if self.parent():
            self.resize(self.parent().size())
            # Ensure proper positioning
            self.move(0, 0)


class ProgressSpinner(LoadingSpinner):
    """
    Enhanced loading spinner with progress indication
    """
    
    def __init__(self, parent=None, text="Loading...", spinner_size=50):
        super().__init__(parent, text, spinner_size)
        self.progress = 0  # 0-100
        self.show_progress = False
    
    def set_progress(self, progress, text=None):
        """Set progress percentage (0-100) and optional text"""
        self.progress = max(0, min(100, progress))
        self.show_progress = True
        
        if text:
            self.update_text(f"{text} ({self.progress}%)")
        else:
            self.update_text(f"{self.text.split(' (')[0]} ({self.progress}%)")
    
    def paintEvent(self, event):
        """Custom paint event to draw spinner with progress"""
        super().paintEvent(event)
        
        if not self.show_progress:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw progress arc
        center_x = self.width() // 2
        center_y = self.height() // 2 - 30
        arc_radius = self.spinner_size // 2 + 15
        
        # Background arc
        painter.setPen(QPen(QColor(255, 255, 255, 50), 3))
        painter.drawArc(
            center_x - arc_radius, center_y - arc_radius,
            arc_radius * 2, arc_radius * 2,
            0, 360 * 16  # Full circle
        )
        
        # Progress arc
        painter.setPen(QPen(QColor(0, 150, 255, 200), 3))
        span_angle = int(360 * 16 * (self.progress / 100))
        painter.drawArc(
            center_x - arc_radius, center_y - arc_radius,
            arc_radius * 2, arc_radius * 2,
            90 * 16, -span_angle  # Start from top, go clockwise
        )