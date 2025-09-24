"""
Network Manager Module
Handles graph loading, network statistics, and performance optimization
"""

import os
import random
import traceback
import networkx as nx
from PyQt5.QtWidgets import QFileDialog
from lib.graph_loader import GraphLoader
from lib.threaded_loaders import GraphLoaderThread
from lib.loading_spinner import LoadingSpinner
from config import FILES


class NetworkManager:
    """Handles network/graph operations and statistics"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.graph = None
        self.graph_loader = GraphLoader(main_window=main_window)
        self.graph_loader_thread = None
        self.loading_spinner = None
        
    def load_graph(self):
        """Load a graph file using the threaded GraphLoader with loading spinner"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load Graph File", "", FILES.GRAPH_FILE_FILTER)
        
        if file_path:
            self._load_graph_with_spinner(file_path)
    
    def _load_graph_with_spinner(self, file_path):
        """Load graph using threaded loader with progress spinner"""
        try:
            # Get the simulation widget to show spinner on visualization area
            simulation_widget = getattr(self.main_window, 'simulation_widget', None)
            if simulation_widget is None:
                # Fallback to central widget if simulation widget not found
                spinner_parent = self.main_window.centralWidget()
            else:
                spinner_parent = simulation_widget
            
            # Create and show loading spinner (blue, no text)
            self.loading_spinner = LoadingSpinner(
                parent=spinner_parent,
                spinner_size=60,
                show_text=False
            )
            self.loading_spinner.show_spinner()
            
            # Create and setup graph loader thread
            self.graph_loader_thread = GraphLoaderThread(file_path, self.main_window)
            
            # Connect signals
            self.graph_loader_thread.progress_updated.connect(self._on_loading_progress)
            self.graph_loader_thread.graph_loaded.connect(self._on_graph_loaded)
            self.graph_loader_thread.error_occurred.connect(self._on_loading_error)
            self.graph_loader_thread.log_message.connect(self.main_window.add_log_message)
            self.graph_loader_thread.finished.connect(self._on_loading_finished)
            
            # Start loading
            self.graph_loader_thread.start()
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error starting graph loading: {str(e)}")
            if self.loading_spinner:
                self.loading_spinner.hide_spinner()
    
    def _on_loading_progress(self, progress, status):
        """Handle loading progress updates"""
        # Simple spinner doesn't show progress, just keep spinning
        pass
    
    def _on_graph_loaded(self, G):
        """Handle successful graph loading"""
        if G is not None:
            self.graph = G
            self.main_window.simulation_widget.set_graph(G)
            
            # Configure performance settings based on network size
            self._configure_performance_settings(G)
            
            self.main_window.start_button.setEnabled(True)
            
            # Enable st-model selection combo now that graph is loaded
            self.main_window.selection_combo.setEnabled(True)
            
            # Show screenshot button now that network is loaded
            if hasattr(self.main_window, 'simulation_tab') and hasattr(self.main_window.simulation_tab, 'set_network_loaded'):
                self.main_window.simulation_tab.set_network_loaded(True)
            
            # Update window title with filename
            if self.graph_loader_thread:
                filename = os.path.basename(self.graph_loader_thread.file_path)
                self.main_window.add_log_message(f"File: {filename}")
                self.main_window.ui_manager.update_window_title(filename)
            
            # Update initial performance status
            self.main_window.status_manager.update_performance_status()
            
            # Update network statistics display
            self.update_network_statistics()
        else:
            self.main_window.add_log_message("❌ Failed to load graph - unsupported format or invalid file")
    
    def _on_loading_error(self, error_message):
        """Handle loading errors"""
        self.main_window.add_log_message(f"❌ {error_message}")
    
    def _on_loading_finished(self):
        """Clean up after loading is finished"""
        if self.loading_spinner:
            self.loading_spinner.hide_spinner()
            self.loading_spinner = None
        
        if self.graph_loader_thread:
            self.graph_loader_thread.deleteLater()
            self.graph_loader_thread = None
    
    def _configure_performance_settings(self, G):
        """Configure performance settings based on network size"""
        node_count = len(G.nodes)
        
        if node_count > 50000:  # Very large network
            self.main_window.simulation_widget.set_performance_settings(
                max_nodes=None, max_edges=None, max_agents=None, 
                lod_threshold=5000, viewport_culling=True)
            self.main_window.add_log_message("Very large network detected - applying optimizations (unlimited nodes/edges/agents)")
        elif node_count > 20000:  # Large network
            self.main_window.simulation_widget.set_performance_settings(
                max_nodes=None, max_edges=None, max_agents=None, 
                lod_threshold=2000, viewport_culling=True)
            self.main_window.add_log_message("Large network detected - applying optimizations (unlimited nodes/edges/agents)")
        elif node_count > 5000:  # Medium network
            self.main_window.simulation_widget.set_performance_settings(
                max_nodes=None, max_edges=None, max_agents=None, 
                lod_threshold=1000, viewport_culling=True)
            self.main_window.add_log_message("Medium network - enabling viewport culling (unlimited nodes/edges/agents)")
        else:  # Small network
            self.main_window.simulation_widget.set_performance_settings(
                max_nodes=None, max_edges=None, max_agents=None, 
                lod_threshold=10000, viewport_culling=False)
    
    def update_network_statistics(self):
        """Calculate and update network general statistics display"""
        if not self.graph or not hasattr(self.main_window, 'network_stats_labels'):
            return
        
        try:
            G = self.graph
            
            # Basic statistics
            num_nodes = len(G.nodes())
            num_edges = len(G.edges())
            
            self.main_window.network_stats_labels['nodes'].setText(f"{num_nodes:,}")
            self.main_window.network_stats_labels['edges'].setText(f"{num_edges:,}")
            
            # Average degree
            if num_nodes > 0:
                total_degree = sum(dict(G.degree()).values())
                avg_degree = total_degree / num_nodes
                self.main_window.network_stats_labels['avg_degree'].setText(f"{avg_degree:.2f}")
            else:
                self.main_window.network_stats_labels['avg_degree'].setText("0.00")
            
            # Density
            if num_nodes > 1:
                max_edges = num_nodes * (num_nodes - 1)
                if not G.is_directed():
                    max_edges = max_edges // 2
                density = num_edges / max_edges
                self.main_window.network_stats_labels['density'].setText(f"{density:.4f}")
            else:
                self.main_window.network_stats_labels['density'].setText("N/A")
            
            # Connected components (for large graphs, we'll estimate)
            self._calculate_components(G, num_nodes)
            
            # Network diameter
            self._calculate_diameter(G, num_nodes)
            
            # Average clustering coefficient
            self._calculate_clustering(G, num_nodes)
            
            self.main_window.add_log_message("Network statistics updated")
            
        except Exception as e:
            self.main_window.add_log_message(f"❌ Error calculating network statistics: {str(e)}")
            # Set all labels to error state
            for label in self.main_window.network_stats_labels.values():
                label.setText("Error")
    
    def _calculate_components(self, G, num_nodes):
        """Calculate connected components"""
        if num_nodes <= 10000:  # Only calculate for smaller graphs
            if G.is_directed():
                num_components = nx.number_weakly_connected_components(G)
            else:
                num_components = nx.number_connected_components(G)
            self.main_window.network_stats_labels['components'].setText(str(num_components))
        else:
            self.main_window.network_stats_labels['components'].setText("Large graph")
    
    def _calculate_diameter(self, G, num_nodes):
        """Calculate network diameter"""
        try:
            if G.is_directed():
                if nx.is_weakly_connected(G):
                    # For directed graphs, use the underlying undirected graph
                    undirected_G = G.to_undirected()
                    if num_nodes <= 5000:
                        diameter = nx.diameter(undirected_G)
                    else:
                        # For large graphs, estimate diameter using eccentricity of random nodes
                        diameter = self._estimate_diameter(undirected_G, num_nodes)
                else:
                    diameter = "∞ (Disconnected)"
            else:
                if nx.is_connected(G):
                    if num_nodes <= 5000:
                        diameter = nx.diameter(G)
                    else:
                        # For large graphs, estimate diameter
                        diameter = self._estimate_diameter(G, num_nodes)
                else:
                    diameter = "∞ (Disconnected)"
            self.main_window.network_stats_labels['diameter'].setText(str(diameter))
        except Exception:
            self.main_window.network_stats_labels['diameter'].setText("Error")
    
    def _estimate_diameter(self, G, num_nodes):
        """Estimate diameter for large graphs using random sampling"""
        sample_nodes = random.sample(list(G.nodes()), min(50, num_nodes))
        eccentricities = []
        for node in sample_nodes:
            try:
                ecc = nx.eccentricity(G, node)
                eccentricities.append(ecc)
            except:
                continue
        return max(eccentricities) if eccentricities else "Unknown"
    
    def _calculate_clustering(self, G, num_nodes):
        """Calculate clustering coefficient"""
        try:
            if num_nodes <= 2:
                clustering = 0.0
            else:
                # Handle multigraphs by converting to simple graph first
                if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                    # Convert multigraph to simple graph
                    if G.is_directed():
                        simple_G = nx.DiGraph(G)
                    else:
                        simple_G = nx.Graph(G)
                else:
                    simple_G = G.copy()
                
                if simple_G.is_directed():
                    # Convert to undirected for clustering calculation as directed clustering can be problematic
                    undirected_G = simple_G.to_undirected()
                    # Remove self-loops which can cause issues
                    undirected_G.remove_edges_from(nx.selfloop_edges(undirected_G))
                    clustering = nx.average_clustering(undirected_G)
                else:
                    # Remove self-loops which can cause issues
                    simple_G.remove_edges_from(nx.selfloop_edges(simple_G))
                    clustering = nx.average_clustering(simple_G)
            
            self.main_window.network_stats_labels['clustering'].setText(f"{clustering:.4f}")
        except Exception as e:
            self.main_window.add_log_message(f"❌ Clustering calculation error: {str(e)}")
            self.main_window.network_stats_labels['clustering'].setText("N/A")
    
    def has_graph(self):
        """Check if a graph is loaded"""
        return self.graph is not None
    
    def get_graph(self):
        """Get the current graph"""
        return self.graph
    
    def clear_graph(self):
        """Clear the current graph"""
        self.graph = None
        
        # Hide screenshot button when no network is loaded
        if hasattr(self.main_window, 'simulation_tab') and hasattr(self.main_window.simulation_tab, 'set_network_loaded'):
            self.main_window.simulation_tab.set_network_loaded(False)