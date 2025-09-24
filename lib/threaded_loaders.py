"""
Threaded Graph Loader Module
Handles graph loading operations in separate threads to prevent UI blocking.
"""

from PyQt5.QtCore import QThread, pyqtSignal
from .graph_loader import GraphLoader
import traceback


class GraphLoaderThread(QThread):
    """
    Thread for loading graphs without blocking the UI.
    Emits signals for progress updates and completion.
    """
    
    # Signals
    progress_updated = pyqtSignal(int, str)  # progress percentage, status text
    graph_loaded = pyqtSignal(object)  # loaded graph (None if failed)
    error_occurred = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # log message
    
    def __init__(self, file_path, main_window=None):
        super().__init__()
        self.file_path = file_path
        self.main_window = main_window
        self.graph_loader = GraphLoader(main_window)
        
        # Connect internal logging to signal
        self.graph_loader._log_message = self._emit_log_message
    
    def _emit_log_message(self, message):
        """Emit log message signal"""
        self.log_message.emit(message)
    
    def run(self):
        """Run the graph loading operation in the thread"""
        try:
            self.progress_updated.emit(0, "Starting graph loading...")
            
            # Load the graph
            graph = self.graph_loader.load_graph(self.file_path)
            
            if graph is not None:
                self.progress_updated.emit(100, "Graph loaded successfully")
                self.graph_loaded.emit(graph)
            else:
                self.error_occurred.emit("Failed to load graph - unsupported format or invalid file")
                self.graph_loaded.emit(None)
                
        except Exception as e:
            error_msg = f"Error loading graph: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.log_message.emit(f"Details: {traceback.format_exc()}")
            self.graph_loaded.emit(None)


class ModelComputationThread(QThread):
    """
    Thread for ST selection model computation to prevent UI blocking.
    """
    
    # Signals
    progress_updated = pyqtSignal(int, str)  # progress percentage, status text
    computation_finished = pyqtSignal(object)  # computed st_selector object
    error_occurred = pyqtSignal(str)  # error message
    log_message = pyqtSignal(str)  # log message
    
    def __init__(self, graph, selection_mode, main_window=None):
        super().__init__()
        self.graph = graph
        self.selection_mode = selection_mode
        self.main_window = main_window
        self.st_selector = None
    
    def run(self):
        """Run the model computation in the thread"""
        try:
            self.progress_updated.emit(0, "Initializing model computation...")
            
            # Import ST selection classes
            from .st_selection import (
                RandomSelection, ActivityBasedSelection, ZoneBasedSelection, 
                GravitySelection, HubAndSpokeSelection
            )
            
            self.progress_updated.emit(10, f"Setting up {self.selection_mode} model...")
            
            # Create the appropriate selector
            if self.selection_mode == "Random":
                self.st_selector = RandomSelection(self.graph)
                self.progress_updated.emit(100, "Random model ready")
            elif self.selection_mode == "Activity-Based":
                self.st_selector = ActivityBasedSelection(self.graph)
                self.progress_updated.emit(100, "Activity-based model ready")
            elif self.selection_mode == "Zone-Based":
                self.st_selector = ZoneBasedSelection(self.graph)
                self.progress_updated.emit(100, "Zone-based model ready")
            elif self.selection_mode == "Gravity":
                self.progress_updated.emit(20, "Computing gravity model distances...")
                
                # Use the progress-aware wrapper for better progress tracking
                progress_aware = ProgressAwareGravitySelection(
                    self.graph, 
                    progress_callback=lambda p, s: self.progress_updated.emit(p, s)
                )
                self.st_selector = progress_aware.compute_with_progress()
                
                if self.st_selector:
                    self.progress_updated.emit(100, "Gravity model ready")
                else:
                    self.error_occurred.emit("Failed to compute gravity model")
                    return
            elif self.selection_mode == "Hub and Spoke":
                self.st_selector = HubAndSpokeSelection(self.graph)
                self.progress_updated.emit(100, "Hub and spoke model ready")
            else:
                # Default to random
                self.st_selector = RandomSelection(self.graph)
                self.progress_updated.emit(100, "Default random model ready")
            
            self.log_message.emit(f"{self.selection_mode} model computation completed")
            self.computation_finished.emit(self.st_selector)
            
        except Exception as e:
            error_msg = f"Error during model computation: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.log_message.emit(f"Details: {traceback.format_exc()}")
            self.computation_finished.emit(None)


class ProgressAwareGravitySelection:
    """
    A wrapper for GravitySelection that emits progress signals during computation.
    This can be used by ModelComputationThread for better progress tracking.
    """
    
    def __init__(self, graph, progress_callback=None):
        self.graph = graph
        self.progress_callback = progress_callback
        self.gravity_selector = None
        
    def compute_with_progress(self):
        """Compute the gravity model with progress callbacks"""
        try:
            if self.progress_callback:
                self.progress_callback(20, "Analyzing graph structure...")
            
            # Import the actual GravitySelection
            from .st_selection import GravitySelection
            
            if self.progress_callback:
                self.progress_callback(40, "Computing node distances...")
            
            # Create the gravity selection (this does the heavy computation)
            self.gravity_selector = GravitySelection(self.graph)
            
            if self.progress_callback:
                self.progress_callback(80, "Finalizing gravity model...")
            
            if self.progress_callback:
                self.progress_callback(100, "Gravity model ready")
            
            return self.gravity_selector
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(0, f"Error: {str(e)}")
            raise