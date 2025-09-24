from PyQt5.QtCore import Qt, QUrl, QRect, QPointF
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap, QDragEnterEvent, QDragMoveEvent, QDropEvent, QPolygonF
import networkx as nx
import os
import math
import numpy as np

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, PERFORMANCE, SIMULATION, hex_to_qcolor_tuple

class SimulationWidget(QWidget):
    """Custom widget for displaying the traffic simulation"""
    
    def __init__(self):
        super().__init__()
        self.graph = None
        self.agents = []
        self.selection_mode = SIMULATION.DEFAULT_SELECTION_MODE
        self.st_selector = None
        
        # Visualization parameters
        self.node_size = 1  # Smaller nodes for better network visibility
        self.agent_size = 3  # Slightly larger agents for better visibility
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Performance optimization for large networks
        self.max_visible_nodes = 5000  # Maximum nodes to draw
        self.max_visible_edges = None  # Maximum edges to draw (None = unlimited, since edges are cached)
        self.max_visible_agents = None  # Maximum agents to draw (None = unlimited)
        self.level_of_detail_threshold = 1000  # Node count threshold for LOD
        self.use_level_of_detail = False
        self.viewport_culling = True  # Enable viewport culling
        
        # Cached data for performance
        self.cached_node_positions = None
        self.cached_edges = None
        self.cached_background_elements = None
        self.cache_dirty = True
        
        # Background rendering optimization
        self.background_pixmap = None  # Cached background image
        self.background_needs_redraw = True
        self.agents_only_mode = False  # When True, only redraw agents
        
        # Color schemes
        self.setup_colors()
        
        # Set minimum size for better visualization
        self.setMinimumSize(800, 600)
        
        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Enable mouse tracking for pan and zoom
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        self.is_panning = False
        
        # Enable drag and drop for network files
        self.setAcceptDrops(True)
        self.drag_hover = False
        
        # Legend settings
        self.show_legend = True  # Toggle for legend visibility
        
        # Reset button overlay settings
        self.show_reset_button = True
        self.reset_button_rect = None
        self.reset_button_hovered = False
        
        # Edge labels toggle button settings
        self.show_edge_labels = False  # Initially hidden
        self.show_edge_labels_button = True  # Show button when network is loaded
        self.edge_labels_button_rect = None
        self.edge_labels_button_hovered = False
        self.edge_labels_button_size = 25
        self.edge_labels_button_margin = 10
        
        # Agent selection and info display
        self.selected_agent = None
        self.show_agent_info = False
        self.agent_info_rect = None
        self.show_reset_button = False  # Will be enabled when graph is loaded
        self.reset_button_size = 25
        self.reset_button_margin = 10
        self.reset_button_hovered = False
        self.reset_button_rect = None
        
        # Supported file extensions for drag and drop
        self.supported_extensions = {'.json', '.graphml', '.osm', '.pbf', '.osm.pbf', '.mtx', '.csv'}
        
        # Initialize reset button rect
        self._update_reset_button_rect()
        
        # Initialize edge labels button rect
        self._update_edge_labels_button_rect()
    
    def add_log_message(self, message):
        """Add a log message - stub method for when no parent logger is available"""
        # This is a fallback method when the widget doesn't have access to main window logger
        print(f"SimulationWidget: {message}")
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        # Get zoom delta
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 1.0 / 1.1
        
        # Apply zoom
        old_zoom = self.zoom_factor
        self.zoom_factor *= zoom_factor
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))  # Limit zoom range
        
        # Update display if zoom changed
        if abs(old_zoom - self.zoom_factor) > 0.01:
            # Mark background for redraw when zoom changes
            self.background_needs_redraw = True
            self.background_pixmap = None
            self.update()
            
            # Update performance info when zoom changes significantly
            if self.graph:
                perf_info = self.get_performance_info()
                if hasattr(self, 'add_log_message'):
                    status = "LOD" if perf_info["use_lod"] else "Full Detail"
                    self.add_log_message(f"Zoom: {self.zoom_factor:.1f}x - Mode: {status}")
    
    def mousePressEvent(self, event):
        """Handle mouse press for panning, reset button, and agent selection"""
        if event.button() == Qt.LeftButton:
            # Check if click is on reset button
            if self.show_reset_button and self._is_reset_button_clicked(event.pos()):
                self.reset_view()
                return
            
            # Check if click is on edge labels button
            if self.show_edge_labels_button and self._is_edge_labels_button_clicked(event.pos()):
                self.toggle_edge_labels()
                return
            
            # Check if click is on an agent
            clicked_agent = self.get_agent_at_position(event.pos())
            if clicked_agent:
                self.selected_agent = clicked_agent
                self.show_agent_info = True
                self.update()  # Redraw to show selection
                return
            
            # If clicked elsewhere, deselect agent
            if self.selected_agent:
                self.selected_agent = None
                self.show_agent_info = False
                self.update()
            
            # Otherwise handle panning
            self.is_panning = True
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning and reset button hover"""
        # Check reset button hover state
        if self.show_reset_button and hasattr(self, 'reset_button_rect') and self.reset_button_rect:
            hover = self._is_reset_button_clicked(event.pos())
            if hover != self.reset_button_hovered:
                self.reset_button_hovered = hover
                self.update()  # Redraw for hover effect
        
        # Check edge labels button hover state
        if self.show_edge_labels_button and hasattr(self, 'edge_labels_button_rect') and self.edge_labels_button_rect:
            hover = self._is_edge_labels_button_clicked(event.pos())
            if hover != self.edge_labels_button_hovered:
                self.edge_labels_button_hovered = hover
                self.update()  # Redraw for hover effect
        
        # Handle panning
        if self.is_panning and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_mouse_pos = event.pos()
            
            # Mark background for redraw when panning
            self.background_needs_redraw = True
            self.background_pixmap = None
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.last_mouse_pos = None
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        # Mark background for redraw when widget is resized
        if self.graph:
            self.background_needs_redraw = True
            self.background_pixmap = None
        
        # Update button positions on resize
        self._update_reset_button_rect()
        self._update_edge_labels_button_rect()
        
    def setup_colors(self):
        """Define color schemes for different agent types and zones using configuration"""
        # Convert hex colors to QColor objects for agent types
        self.type_colors = {}
        for agent_type, hex_color in COLORS.AGENT_TYPE_COLORS.items():
            rgb = hex_to_qcolor_tuple(hex_color)
            self.type_colors[agent_type] = QColor(*rgb)
        
        # Add special zone-based colors
        self.type_colors['intra_zone'] = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['zone']['intra_zone']))
        self.type_colors['inter_zone'] = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['zone']['inter_zone']))
        
        # Activity colors for nodes
        self.activity_colors = {}
        for activity_type, hex_color in COLORS.ACTIVITY_NODE_COLORS.items():
            rgb = hex_to_qcolor_tuple(hex_color)
            self.activity_colors[activity_type] = QColor(*rgb)
        
        # Zone colors (cycling list)
        self.zone_colors = []
        for hex_color in COLORS.ZONE_COLORS:
            rgb = hex_to_qcolor_tuple(hex_color)
            self.zone_colors.append(QColor(*rgb))
    
    def set_graph(self, graph):
        """Set the graph to display"""
        self.graph = graph
        if graph:
            self.calculate_view_bounds()
            # Determine if we should use level of detail optimizations
            node_count = len(graph.nodes)
            edge_count = len(graph.edges)
            self.use_level_of_detail = node_count > self.level_of_detail_threshold
            
            # Cache network structure for performance
            self.cache_network_structure()
            
            # Mark background for redraw
            self.background_needs_redraw = True
            self.background_pixmap = None
            
            # Enable reset button and update its position
            self.show_reset_button = True
            self.show_edge_labels_button = True
            self._update_reset_button_rect()
            self._update_edge_labels_button_rect()
            
            self.add_log_message(f"Network loaded: {node_count} nodes, {edge_count} edges")
            if self.use_level_of_detail:
                self.add_log_message("Large network detected - enabling performance optimizations")
            
            # Enable button overlays
            self.show_reset_button = True
            self.show_edge_labels_button = True
        else:
            # Disable button overlays
            self.show_reset_button = False
            self.show_edge_labels_button = False
        
        self.update()
    
    def _notify_graph_loaded(self, loaded):
        """Notify parent widget about graph load state"""
        # Try to find the simulation tab and enable/disable reset button
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'reset_view_button'):
                parent.reset_view_button.setEnabled(loaded)
                break
            parent = parent.parent()
    
    def set_performance_settings(self, max_nodes=5000, max_edges=None, max_agents=None, 
                                lod_threshold=1000, viewport_culling=True):
        """Configure performance optimization settings"""
        self.max_visible_nodes = max_nodes
        self.max_visible_edges = max_edges
        self.max_visible_agents = max_agents  # None means unlimited
        self.level_of_detail_threshold = lod_threshold
        self.viewport_culling = viewport_culling
        
        # Re-evaluate LOD settings if graph is loaded
        if self.graph:
            node_count = len(self.graph.nodes)
            self.use_level_of_detail = node_count > self.level_of_detail_threshold
            
        self.update()
    
    def get_performance_info(self):
        """Get current performance optimization information"""
        if not self.graph:
            return {"status": "No graph loaded"}
        
        node_count = len(self.graph.nodes)
        edge_count = len(self.graph.edges)
        agent_count = len(self.agents)
        
        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "total_agents": agent_count,
            "max_visible_nodes": self.max_visible_nodes,
            "max_visible_edges": self.max_visible_edges,
            "max_visible_agents": self.max_visible_agents,
            "use_lod": self.use_level_of_detail,
            "viewport_culling": self.viewport_culling,
            "zoom_factor": self.zoom_factor
        }
    
    def start_simulation_rendering(self):
        """Switch to agents-only rendering mode for simulation"""
        if self.graph and self.background_pixmap is None:
            # Create the background pixmap once
            self.create_background_pixmap()
        self.agents_only_mode = True
        self.add_log_message("Switched to agents-only rendering mode for better performance")
    
    def stop_simulation_rendering(self):
        """Return to full rendering mode"""
        self.agents_only_mode = False
        self.add_log_message("Returned to full rendering mode")
    
    def force_background_redraw(self):
        """Force the background to be redrawn on next paint event"""
        self.background_needs_redraw = True
        self.background_pixmap = None
        self.update()
    
    def prepare_for_simulation(self):
        """Prepare the widget for simulation by pre-rendering the background"""
        if self.graph and not self.background_pixmap:
            self.create_background_pixmap()
            self.add_log_message("Pre-rendered network background for simulation")
    
    def create_background_pixmap(self):
        """Create a cached pixmap of the network background"""
        if not self.graph or not self.cached_node_positions:
            return
        
        # Ensure widget has valid size
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return
        
        # Create pixmap with widget size
        self.background_pixmap = QPixmap(widget_size)
        self.background_pixmap.fill(QColor(240, 240, 240))  # Background color
        
        # Paint the network background to the pixmap
        painter = QPainter(self.background_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            # Draw background elements
            if self.cached_background_elements:
                self.draw_cached_background(painter)
            
            # Always draw edges in background cache for visibility
            # Since edges are cached once in background pixmap, showing all edges 
            # doesn't impact real-time performance during simulation
            self.draw_edges_optimized(painter)
            
            # Draw network nodes with level-of-detail
            if self.should_draw_detailed():
                self.draw_nodes_optimized(painter)
            else:
                # For simplified mode, only draw simplified nodes (edges already drawn above)
                self.draw_simplified_nodes(painter)
            
            self.background_needs_redraw = False
            
        except Exception as e:
            self.add_log_message(f"❌ Error creating background pixmap: {str(e)}")
        finally:
            painter.end()
    
    def cache_network_structure(self):
        """Cache network structure for better performance"""
        if not self.graph:
            return
            
        # Cache node positions
        self.cached_node_positions = nx.get_node_attributes(self.graph, 'pos')
        
        # Cache edges with their positions
        self.cached_edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in self.cached_node_positions and v in self.cached_node_positions:
                self.cached_edges.append((u, v, data))
        
        # Cache background elements based on selection mode
        self.cache_background_elements()
        
        self.cache_dirty = False
    
    def cache_background_elements(self):
        """Cache background visualization elements"""
        if not self.graph or not self.cached_node_positions:
            return
            
        self.cached_background_elements = {
            'zones': [],
            'activity_circles': [],
            'gravity_circles': [],
            'hub_circles': []
        }
        
        # Pre-calculate zone rectangles
        if self.selection_mode == 'zone':
            self.cached_background_elements['zones'] = self.calculate_zone_rectangles()
        
        # Pre-calculate activity circles
        elif self.selection_mode == 'activity' and self.st_selector:
            self.cached_background_elements['activity_circles'] = self.calculate_activity_circles()
        
        # Pre-calculate gravity circles
        elif self.selection_mode == 'gravity' and self.st_selector:
            print(f"DEBUG: Caching gravity circles for selector {type(self.st_selector).__name__}")
            gravity_circles = self.calculate_gravity_circles()
            print(f"DEBUG: Got {len(gravity_circles)} gravity circles to cache")
            self.cached_background_elements['gravity_circles'] = gravity_circles
        
        # Pre-calculate hub circles
        elif self.selection_mode == 'hub' and self.st_selector:
            self.cached_background_elements['hub_circles'] = self.calculate_hub_circles()
    
    def calculate_zone_rectangles(self):
        """Pre-calculate zone rectangle data"""
        x_coords = [pos[0] for pos in self.cached_node_positions.values()]
        y_coords = [pos[1] for pos in self.cached_node_positions.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        x_step = (max_x - min_x) / 3
        y_step = (max_y - min_y) / 3
        
        zones = []
        for i in range(3):
            for j in range(3):
                zone_id = i * 3 + j
                color = self.zone_colors[zone_id % len(self.zone_colors)]
                
                x_start = min_x + i * x_step
                y_start = min_y + j * y_step
                x_end = min_x + (i + 1) * x_step
                y_end = min_y + (j + 1) * y_step
                
                zones.append({
                    'bounds': (x_start, y_start, x_end, y_end),
                    'color': color
                })
        
        return zones
    
    def calculate_activity_circles(self):
        """Pre-calculate activity circle data"""
        if not hasattr(self.st_selector, 'get_activity_nodes'):
            return []
        
        activity_nodes = self.st_selector.get_activity_nodes()
        x_coords = [pos[0] for pos in self.cached_node_positions.values()]
        y_coords = [pos[1] for pos in self.cached_node_positions.values()]
        network_width = max(x_coords) - min(x_coords)
        network_height = max(y_coords) - min(y_coords)
        base_radius = min(network_width, network_height) * 0.02
        
        circles = []
        for activity_type, nodes in activity_nodes.items():
            if not nodes:
                continue
            
            color = self.activity_colors.get(activity_type, QColor(128, 128, 128))
            for node_id in nodes:
                if node_id in self.cached_node_positions:
                    pos = self.cached_node_positions[node_id]
                    circles.append({
                        'pos': pos,
                        'radius': base_radius,
                        'color': color
                    })
        
        return circles
    
    def calculate_gravity_circles(self):
        """Pre-calculate gravity circle data for Voorhees gravity model"""
        # Check for new Voorhees gravity model attributes
        if hasattr(self.st_selector, 'size_variables') and hasattr(self.st_selector, 'attraction_variables'):
            return self._calculate_voorhees_gravity_circles()
        # Fallback for old gravity model
        elif hasattr(self.st_selector, 'node_importance'):
            return self._calculate_legacy_gravity_circles()
        else:
            return []
    
    def _calculate_voorhees_gravity_circles(self):
        """Calculate gravity circles for the new Voorhees model"""
        print("DEBUG: Calculating Voorhees gravity circles...")
        
        size_variables = self.st_selector.size_variables
        attraction_variables = self.st_selector.attraction_variables
        
        print(f"DEBUG: Found {len(size_variables)} size variables, {len(attraction_variables)} attraction variables")
        
        x_coords = [pos[0] for pos in self.cached_node_positions.values()]
        y_coords = [pos[1] for pos in self.cached_node_positions.values()]
        network_width = max(x_coords) - min(x_coords)
        network_height = max(y_coords) - min(y_coords)
        base_radius = min(network_width, network_height) * 0.005  # Reduced size for better visibility
        
        print(f"DEBUG: Network size: {network_width:.2f} x {network_height:.2f}, base_radius: {base_radius:.2f}")
        
        # Use attraction variables for circle visualization (destination attractiveness)
        attraction_values = list(attraction_variables.values())
        if not attraction_values:
            return []
            
        max_attraction = max(attraction_values)
        min_attraction = min(attraction_values)
        attraction_range = max_attraction - min_attraction if max_attraction > min_attraction else 1
        
        circles = []
        # Show top 30% most attractive nodes for better visibility
        sorted_attractions = sorted(attraction_variables.items(), key=lambda x: x[1], reverse=True)
        top_count = max(15, len(sorted_attractions) // 3)  # Show at least 15 nodes, top 1/3
        
        print(f"DEBUG: Will create {top_count} circles from {len(sorted_attractions)} total nodes")
        
        for i, (node_id, attraction) in enumerate(sorted_attractions[:top_count]):
            if node_id in self.cached_node_positions:
                # Use attraction for circle size (destination attractiveness) - smaller scaling
                attraction_ratio = (attraction - min_attraction) / attraction_range
                radius = base_radius * (0.8 + attraction_ratio * 1.2)  # Smaller size scaling
                
                # More distinct colors based on attraction level
                if attraction_ratio > 0.8:
                    color = QColor(255, 0, 0)      # Bright red for highest attraction
                elif attraction_ratio > 0.6:
                    color = QColor(255, 100, 0)    # Orange for high attraction
                elif attraction_ratio > 0.4:
                    color = QColor(255, 150, 0)    # Orange-yellow for medium attraction
                else:
                    color = QColor(255, 200, 0)    # Yellow for lower attraction
                
                circles.append({
                    'pos': self.cached_node_positions[node_id],
                    'radius': radius,
                    'color': color
                })
                
                print(f"DEBUG: Circle {i+1}: node {node_id}, attraction {attraction:.4f}, radius {radius:.2f}")
        
        # Debug message
        print(f"DEBUG: Created {len(circles)} gravity circles total")
        if hasattr(self, 'add_log_message'):
            self.add_log_message(f"Gravity visualization: showing {len(circles)} attraction circles")
        
        return circles
    
    def _calculate_legacy_gravity_circles(self):
        """Calculate gravity circles for the legacy gravity model (fallback)"""
        node_importance = self.st_selector.node_importance
        x_coords = [pos[0] for pos in self.cached_node_positions.values()]
        y_coords = [pos[1] for pos in self.cached_node_positions.values()]
        network_width = max(x_coords) - min(x_coords)
        network_height = max(y_coords) - min(y_coords)
        base_radius = min(network_width, network_height) * 0.008
        
        importance_values = list(node_importance.values())
        max_importance = max(importance_values)
        min_importance = min(importance_values)
        importance_range = max_importance - min_importance if max_importance > min_importance else 1
        
        circles = []
        for node_id, importance in node_importance.items():
            if node_id in self.cached_node_positions:
                importance_ratio = (importance - min_importance) / importance_range
                radius = base_radius * (0.3 + importance_ratio * 1.0)
                
                circles.append({
                    'pos': self.cached_node_positions[node_id],
                    'radius': radius,
                    'color': QColor(255, 0, 0)  # Red
                })
        
        return circles
    
    def calculate_hub_circles(self):
        """Pre-calculate hub circle data"""
        if not hasattr(self.st_selector, 'get_hub_nodes'):
            return []
        
        try:
            hub_data = self.st_selector.get_hub_nodes()
            if isinstance(hub_data, dict) and 'hubs' in hub_data:
                hub_nodes = hub_data['hubs']
            else:
                hub_nodes = hub_data
        except:
            return []
        
        x_coords = [pos[0] for pos in self.cached_node_positions.values()]
        y_coords = [pos[1] for pos in self.cached_node_positions.values()]
        network_width = max(x_coords) - min(x_coords)
        network_height = max(y_coords) - min(y_coords)
        base_radius = min(network_width, network_height) * 0.005
        
        circles = []
        for hub_id in hub_nodes:
            if hub_id in self.cached_node_positions:
                circles.append({
                    'pos': self.cached_node_positions[hub_id],
                    'radius': base_radius,
                    'color': QColor(255, 165, 0)  # Orange
                })
        
        return circles
    
    def set_agents(self, agents):
        """Set the agents to display"""
        # Preserve selected agent and trip counts if they exist
        selected_agent_id = None
        if self.selected_agent and hasattr(self.selected_agent, 'id'):
            selected_agent_id = self.selected_agent.id
            
        # Preserve trip counts by agent ID
        existing_trip_counts = {}
        if hasattr(self, 'agents') and self.agents:
            for agent in self.agents:
                if hasattr(agent, 'id') and hasattr(agent, 'trip_count'):
                    existing_trip_counts[agent.id] = agent.trip_count
        
        self.agents = agents
        
        # Ensure all agents have IDs and required attributes (for backward compatibility)
        if self.agents:
            # Check if Agent class has _next_id
            if hasattr(self.agents[0].__class__, '_next_id'):
                next_id = self.agents[0].__class__._next_id
            else:
                # Initialize _next_id if it doesn't exist
                self.agents[0].__class__._next_id = 1
                next_id = 1
                
            for agent in self.agents:
                # Ensure agent has ID
                if not hasattr(agent, 'id'):
                    agent.id = next_id
                    next_id += 1
                
                # Restore or initialize trip_count - prioritize incoming agent data
                if not hasattr(agent, 'trip_count'):
                    # Agent doesn't have trip_count, try to restore from existing data
                    if hasattr(agent, 'id') and agent.id in existing_trip_counts:
                        agent.trip_count = existing_trip_counts[agent.id]
                    else:
                        # Initialize new agent with trip_count 0
                        agent.trip_count = 0
                # If agent already has trip_count, keep it (it's the most up-to-date from simulation)
                
                # Ensure agent has wait_time
                if not hasattr(agent, 'wait_time'):
                    agent.wait_time = 0.0
                    
            # Update the class counter
            self.agents[0].__class__._next_id = max(next_id, self.agents[0].__class__._next_id)
            
            # Restore selected agent by ID AFTER trip counts have been properly restored
            if selected_agent_id is not None:
                self.selected_agent = None  # Reset first
                for agent in self.agents:
                    if hasattr(agent, 'id') and agent.id == selected_agent_id:
                        self.selected_agent = agent
                        break
                
                # If we couldn't find the selected agent, clear the selection
                if self.selected_agent is None:
                    self.show_agent_info = False
        
        # When agents are updated during simulation, only redraw agents (not background)
        if self.background_pixmap is not None:
            self.agents_only_mode = True
        self.update()
    
    def set_selection_mode(self, mode, st_selector):
        """Set the selection mode and selector"""
        print(f"DEBUG: SimulationWidget.set_selection_mode called with mode='{mode}', selector={type(st_selector).__name__ if st_selector else None}")
        
        self.selection_mode = mode
        self.st_selector = st_selector
        
        # Recache background elements when selection mode changes
        if self.graph:
            print(f"DEBUG: Recaching background elements for mode {mode}")
            self.cache_background_elements()
            # Mark background for redraw when selection mode changes
            self.background_needs_redraw = True
            self.background_pixmap = None
            
            # Force immediate redraw for gravity mode
            if mode == 'gravity':
                print("DEBUG: Forcing immediate redraw for gravity mode")
                print(f"DEBUG: ST selector type: {type(st_selector).__name__}")
                print(f"DEBUG: ST selector has size_variables: {hasattr(st_selector, 'size_variables')}")
                print(f"DEBUG: ST selector has attraction_variables: {hasattr(st_selector, 'attraction_variables')}")
                print(f"DEBUG: ST selector has node_importance: {hasattr(st_selector, 'node_importance')}")
                self.force_background_redraw()
        
        self.update()
    
    def calculate_view_bounds(self):
        """Calculate the bounds of the network for proper scaling"""
        if not self.graph:
            return
            
        node_positions = nx.get_node_attributes(self.graph, 'pos')
        if not node_positions:
            return
            
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        
        self.min_x, self.max_x = min(x_coords), max(x_coords)
        self.min_y, self.max_y = min(y_coords), max(y_coords)
        
        # Add some padding
        padding_x = (self.max_x - self.min_x) * 0.05
        padding_y = (self.max_y - self.min_y) * 0.05
        
        self.min_x -= padding_x
        self.max_x += padding_x
        self.min_y -= padding_y
        self.max_y += padding_y
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        if not self.graph:
            return 0, 0
            
        # Calculate scale to fit the widget
        widget_width = self.width()
        widget_height = self.height()
        
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        
        scale_x = widget_width / world_width
        scale_y = widget_height / world_height
        scale = min(scale_x, scale_y) * self.zoom_factor
        
        # Convert to screen coordinates
        screen_x = (world_x - self.min_x) * scale + self.pan_x
        screen_y = widget_height - (world_y - self.min_y) * scale + self.pan_y
        
        return int(screen_x), int(screen_y)
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        if not self.graph:
            return 0, 0
            
        widget_width = self.width()
        widget_height = self.height()
        
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        
        scale_x = widget_width / world_width
        scale_y = widget_height / world_height
        scale = min(scale_x, scale_y) * self.zoom_factor
        
        # Convert to world coordinates
        world_x = (screen_x - self.pan_x) / scale + self.min_x
        world_y = self.min_y + (widget_height - (screen_y - self.pan_y)) / scale
        
        return world_x, world_y
    
    def get_agent_at_position(self, screen_pos):
        """Find the agent at the given screen position, return None if no agent found"""
        if not self.agents:
            return None
        
        # Convert screen position to world coordinates
        world_x, world_y = self.screen_to_world(screen_pos.x(), screen_pos.y())
        
        # Calculate click tolerance based on agent size and zoom
        agent_size = max(1, int(self.agent_size * min(self.zoom_factor, 2.0)))
        tolerance = max(5, agent_size * 2)  # Minimum 5 pixel tolerance
        
        # Check all agents (prioritize moving agents)
        moving_agents = [agent for agent in self.agents if agent.state == 'moving']
        waiting_agents = [agent for agent in self.agents if agent.state == 'waiting']
        
        # Check moving agents first, then waiting agents
        for agent in moving_agents + waiting_agents:
            # Get agent position (interpolated if moving)
            position = self.get_interpolated_agent_position(agent)
            if position is not None:
                agent_x, agent_y = position
                
                # Calculate distance between click and agent
                dx = world_x - agent_x
                dy = world_y - agent_y
                distance_world = (dx*dx + dy*dy)**0.5
                
                # Convert distance to screen space for comparison
                screen_agent_x, screen_agent_y = self.world_to_screen(agent_x, agent_y)
                screen_distance = ((screen_pos.x() - screen_agent_x)**2 + (screen_pos.y() - screen_agent_y)**2)**0.5
                
                if screen_distance <= tolerance:
                    return agent
        
        return None
    
    def is_in_viewport(self, world_x, world_y, margin=0):
        """Check if a world coordinate is visible in the current viewport"""
        if not self.viewport_culling:
            return True
            
        screen_x, screen_y = self.world_to_screen(world_x, world_y)
        
        # Add margin for objects that might be partially visible
        return (-margin <= screen_x <= self.width() + margin and 
                -margin <= screen_y <= self.height() + margin)
    
    def get_viewport_bounds(self):
        """Get the world coordinate bounds of the current viewport"""
        if not self.graph:
            return None
            
        # Convert screen corners to world coordinates
        widget_width = self.width()
        widget_height = self.height()
        
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        
        scale_x = widget_width / world_width
        scale_y = widget_height / world_height
        scale = min(scale_x, scale_y) * self.zoom_factor
        
        # Calculate visible world bounds
        visible_world_width = widget_width / scale
        visible_world_height = widget_height / scale
        
        center_x = self.min_x + world_width / 2 - self.pan_x / scale
        center_y = self.min_y + world_height / 2 + self.pan_y / scale
        
        return {
            'min_x': center_x - visible_world_width / 2,
            'max_x': center_x + visible_world_width / 2,
            'min_y': center_y - visible_world_height / 2,
            'max_y': center_y + visible_world_height / 2
        }
    
    def should_draw_detailed(self):
        """Determine if detailed drawing should be used based on zoom level and network size"""
        if not self.use_level_of_detail:
            return True
            
        # Use detailed drawing when zoomed in or network is small
        node_count = len(self.graph.nodes) if self.graph else 0
        zoom_threshold = 2.0  # Threshold for detailed drawing
        
        return (self.zoom_factor >= zoom_threshold or 
                node_count <= self.level_of_detail_threshold)
    
    def paintEvent(self, event):
        """Paint the simulation with optimized background caching"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if not self.graph:
            # Draw drag and drop zone when no graph is loaded
            bg_color = QColor(245, 245, 245)
            if self.drag_hover:
                bg_color = QColor(235, 245, 255)  # Light blue when hovering
            
            painter.fillRect(self.rect(), bg_color)
            
            # Draw border
            border_color = QColor(200, 200, 200)
            if self.drag_hover:
                border_color = QColor(100, 150, 255)  # Blue border when hovering
            
            painter.setPen(QPen(border_color, 2, Qt.DashLine))
            painter.setBrush(QBrush())
            margin = 20
            painter.drawRect(margin, margin, self.width() - 2*margin, self.height() - 2*margin)
            
            # Draw text
            text_color = QColor(200, 200, 200)
            if self.drag_hover:
                text_color = QColor(60, 100, 180)
            
            painter.setPen(text_color)
            painter.setFont(QFont("Arial", 18, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.AlignTop, "Drop Network File Here\n\n")
            
            painter.setFont(QFont("Arial", 12))
            supported_formats = "Supported formats: JSON, GraphML, OSM, PBF, MTX, CSV"
            painter.drawText(self.rect(), Qt.AlignCenter, f"\n\n\n\n{supported_formats}\n\nOr use the 'Load Graph' button")
            return
        
        # Check if we need to create or recreate the background pixmap
        if (self.background_needs_redraw or 
            self.background_pixmap is None or 
            self.background_pixmap.size() != self.size()):
            self.create_background_pixmap()
        
        # In agents-only mode, use cached background
        if self.agents_only_mode and self.background_pixmap is not None:
            # Draw cached background
            painter.drawPixmap(0, 0, self.background_pixmap)
            
            # Only draw agents on top
            self.draw_agents(painter)
        else:
            # Full rendering mode - draw everything
            painter.fillRect(self.rect(), QColor(240, 240, 240))
            self.draw_background(painter)
            self.draw_agents(painter)
        
        # Always draw the legend on top (only when graph is loaded)
        if self.show_legend:
            self.draw_legend(painter)
        
        # Draw button overlays (only when graph is loaded)
        if self.show_reset_button:
            self.draw_reset_button(painter)
        if self.show_edge_labels_button:
            self.draw_edge_labels_button(painter)
        
        # Draw agent info panel if an agent is selected
        if self.show_agent_info and self.selected_agent:
            self.draw_agent_info_panel(painter)
    
    def draw_background(self, painter):
        """Draw the network background with optimizations"""
        if not self.cached_node_positions:
            return
        
        # Use cached background elements
        if self.cached_background_elements:
            self.draw_cached_background(painter)
        
        # Always draw edges for visibility, use LOD only for nodes
        self.draw_edges_optimized(painter)
        
        # Draw network nodes with level-of-detail
        if self.should_draw_detailed():
            self.draw_nodes_optimized(painter)
        else:
            self.draw_simplified_network(painter)
    
    def draw_cached_background(self, painter):
        """Draw cached background elements (zones, circles, etc.)"""
        background = self.cached_background_elements
        
        # Draw zone rectangles
        for zone_data in background.get('zones', []):
            self.draw_zone_rectangle(painter, zone_data)
        
        # Draw activity circles
        for circle_data in background.get('activity_circles', []):
            self.draw_circle(painter, circle_data)
        
        # Draw gravity circles  
        for circle_data in background.get('gravity_circles', []):
            self.draw_circle(painter, circle_data)
        
        # Draw hub circles
        for circle_data in background.get('hub_circles', []):
            self.draw_circle(painter, circle_data)
    
    def draw_zone_rectangle(self, painter, zone_data):
        """Draw a single zone rectangle"""
        bounds = zone_data['bounds']
        color = zone_data['color']
        
        # Check if zone intersects viewport
        x_start, y_start, x_end, y_end = bounds
        if (self.viewport_culling and 
            not (self.is_in_viewport(x_start, y_start) or 
                 self.is_in_viewport(x_end, y_end) or
                 self.is_in_viewport(x_start, y_end) or
                 self.is_in_viewport(x_end, y_start))):
            return
        
        screen_x1, screen_y1 = self.world_to_screen(x_start, y_start)
        screen_x2, screen_y2 = self.world_to_screen(x_end, y_end)
        
        color.setAlpha(20)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color, 1))
        painter.drawRect(screen_x1, screen_y2, screen_x2 - screen_x1, screen_y1 - screen_y2)
    
    def draw_circle(self, painter, circle_data):
        """Draw a single circle with viewport culling"""
        pos = circle_data['pos']
        radius = circle_data['radius']
        color = circle_data['color']
        
        # Viewport culling - check if circle is visible
        if self.viewport_culling and not self.is_in_viewport(pos[0], pos[1], radius):
            return
        
        # Convert to screen coordinates
        cx, cy = self.world_to_screen(pos[0], pos[1])
        rx, _ = self.world_to_screen(pos[0] + radius, pos[1])
        screen_radius = abs(rx - cx)
        
        # Make circles more visible - increase minimum size and alpha
        screen_radius = max(screen_radius, 3)  # Minimum 3 pixel radius
        
        # Set more visible alpha and drawing properties for gravity circles
        fill_color = QColor(color)
        border_color = QColor(color)
        
        # Make gravity circles much more visible
        if self.selection_mode == 'gravity':
            fill_color.setAlpha(80)    # Higher alpha for gravity circles
            border_color.setAlpha(150) # Even higher alpha for border
            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(border_color, 1))  # Thicker border
        # Make hub circles more visible
        elif self.selection_mode == 'hub':
            fill_color.setAlpha(100)   # Higher alpha for hub circles
            border_color.setAlpha(180) # Even higher alpha for border
            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(border_color, 1))  # Thicker border
        else:
            fill_color.setAlpha(26)  # Low alpha for other background elements
            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(Qt.NoPen))
        
        painter.drawEllipse(cx - screen_radius, cy - screen_radius,
                          screen_radius * 2, screen_radius * 2)
    
    def draw_activity_circles(self, painter, node_positions):
        """Legacy method - now uses cached data"""
        if self.cached_background_elements:
            for circle_data in self.cached_background_elements.get('activity_circles', []):
                self.draw_circle(painter, circle_data)
    
    def draw_hub_circles(self, painter, node_positions):
        """Legacy method - now uses cached data"""
        if self.cached_background_elements:
            for circle_data in self.cached_background_elements.get('hub_circles', []):
                self.draw_circle(painter, circle_data)
    
    def draw_zone_rectangles(self, painter, node_positions):
        """Legacy method - now uses cached data"""
        if self.cached_background_elements:
            for zone_data in self.cached_background_elements.get('zones', []):
                self.draw_zone_rectangle(painter, zone_data)
    
    def draw_gravity_circles(self, painter, node_positions):
        """Legacy method - now uses cached data"""
        if self.cached_background_elements:
            for circle_data in self.cached_background_elements.get('gravity_circles', []):
                self.draw_circle(painter, circle_data)
    
    def draw_edges_optimized(self, painter):
        """Draw network edges with viewport culling and performance optimizations
        
        For large networks, edges are cached once in background pixmap so drawing all edges
        doesn't impact real-time simulation performance. Edge limits can be optionally applied
        if max_visible_edges is set to a specific value (not None).
        """
        # Use configured edge color and transparency
        edge_color = QColor(COLORS.NETWORK_VISUALIZATION_COLORS['edge'])
        edge_color.setAlpha(COLORS.EDGE_ALPHA)  # Apply transparency
        
        if not self.cached_edges:
            return
        
        # Get edges to draw (apply limit only if specified)
        edges_to_draw = self.cached_edges
        if self.max_visible_edges is not None and len(edges_to_draw) > self.max_visible_edges:
            # Use spatial or importance-based filtering here if needed
            edges_to_draw = edges_to_draw[:self.max_visible_edges]
        
        drawn_edges = 0
        for u, v, data in edges_to_draw:
            # Apply limit only if max_visible_edges is set
            if self.max_visible_edges is not None and drawn_edges >= self.max_visible_edges:
                break
                
            # Get positions
            pos_u = self.cached_node_positions[u]
            pos_v = self.cached_node_positions[v]
            
            # Viewport culling - skip edges that are completely outside viewport
            if (self.viewport_culling and 
                not (self.is_in_viewport(pos_u[0], pos_u[1]) or 
                     self.is_in_viewport(pos_v[0], pos_v[1]))):
                continue
            
            screen_x1, screen_y1 = self.world_to_screen(pos_u[0], pos_u[1])
            screen_x2, screen_y2 = self.world_to_screen(pos_v[0], pos_v[1])
            
            # Use thinner edges with transparency
            painter.setPen(QPen(edge_color, COLORS.EDGE_WIDTH))
            painter.drawLine(screen_x1, screen_y1, screen_x2, screen_y2)
            
            # Draw edge label if enabled (this uses a different pen/color)
            if self.show_edge_labels:
                # Save current painter state
                current_pen = painter.pen()
                current_font = painter.font()
                
                # Draw label (this method sets its own pen and font)
                self.draw_edge_label(painter, u, v, data, screen_x1, screen_y1, screen_x2, screen_y2)
                
                # Restore painter state for edge drawing
                painter.setPen(current_pen)
                painter.setFont(current_font)
            
            drawn_edges += 1
    
    def draw_nodes_optimized(self, painter):
        """Draw network nodes with viewport culling and performance optimizations
        
        Note: Node limits removed since nodes are cached once and drawn efficiently.
        This allows full network visualization without significant performance impact.
        """
        if not self.cached_node_positions:
            return
        
        painter.setBrush(QBrush(QColor(150, 150, 150)))  # Lighter gray
        painter.setPen(QPen(QColor(120, 120, 120), 1))   # Even lighter border
        
        # Draw all nodes (or limit if max_visible_nodes is set)
        nodes_to_draw = list(self.cached_node_positions.items())
        if self.max_visible_nodes is not None and len(nodes_to_draw) > self.max_visible_nodes:
            nodes_to_draw = nodes_to_draw[:self.max_visible_nodes]
        
        drawn_nodes = 0
        for node, pos in nodes_to_draw:
            if self.max_visible_nodes is not None and drawn_nodes >= self.max_visible_nodes:
                break
                
            # Viewport culling
            if self.viewport_culling and not self.is_in_viewport(pos[0], pos[1]):
                continue
            
            screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
            
            # Adaptive node size based on zoom
            node_size = max(1, int(self.node_size * self.zoom_factor))
            
            painter.drawEllipse(screen_x - node_size, screen_y - node_size, 
                              node_size * 2, node_size * 2)
            drawn_nodes += 1
    
    def draw_simplified_network(self, painter):
        """Draw a simplified version of the network for better performance"""
        if not self.cached_node_positions:
            return
        
        # For very large networks when zoomed out, just draw a density map or key nodes
        viewport_bounds = self.get_viewport_bounds()
        if not viewport_bounds:
            return
        
        # Use configured colors with transparency for simplified view
        edge_color = QColor(COLORS.NETWORK_VISUALIZATION_COLORS['edge'])
        edge_color.setAlpha(COLORS.EDGE_ALPHA)
        node_color = QColor(COLORS.NETWORK_VISUALIZATION_COLORS['node'])
        node_color.setAlpha(COLORS.NODE_OUTLINE_ALPHA)
        
        # Draw only a subset of nodes and edges in viewport
        painter.setPen(QPen(edge_color, COLORS.EDGE_WIDTH))
        painter.setBrush(QBrush(node_color))
        
        # Sample nodes within viewport
        visible_nodes = []
        for node, pos in self.cached_node_positions.items():
            if (viewport_bounds['min_x'] <= pos[0] <= viewport_bounds['max_x'] and
                viewport_bounds['min_y'] <= pos[1] <= viewport_bounds['max_y']):
                visible_nodes.append((node, pos))
        
        # Subsample for performance
        step = max(1, len(visible_nodes) // 500)  # Draw at most 500 nodes
        for i in range(0, len(visible_nodes), step):
            node, pos = visible_nodes[i]
            screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
            painter.drawEllipse(screen_x - 1, screen_y - 1, 2, 2)
    
    def draw_simplified_nodes(self, painter):
        """Draw only simplified nodes (without edges) for background cache"""
        if not self.cached_node_positions:
            return
        
        viewport_bounds = self.get_viewport_bounds()
        if not viewport_bounds:
            return
        
        painter.setBrush(QBrush(QColor(150, 150, 150)))
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        
        # Sample nodes within viewport
        visible_nodes = []
        for node, pos in self.cached_node_positions.items():
            if (viewport_bounds['min_x'] <= pos[0] <= viewport_bounds['max_x'] and
                viewport_bounds['min_y'] <= pos[1] <= viewport_bounds['max_y']):
                visible_nodes.append((node, pos))
        
        # Draw all visible nodes (no subsampling for simplified view since nodes are cached)
        for node, pos in visible_nodes:
            screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
            painter.drawEllipse(screen_x - 1, screen_y - 1, 2, 2)
    
    def draw_edges(self, painter, node_positions):
        """Legacy method - kept for compatibility"""
        self.draw_edges_optimized(painter)
    
    def draw_nodes(self, painter, node_positions):
        """Legacy method - kept for compatibility"""
        self.draw_nodes_optimized(painter)
    
    def draw_agent_triangle(self, painter, screen_x, screen_y, agent_size, direction_vector):
        """Draw a triangular agent pointing in the direction of movement
        
        Args:
            painter: QPainter instance
            screen_x: X coordinate on screen
            screen_y: Y coordinate on screen
            agent_size: Size of the agent triangle
            direction_vector: 2D numpy array or tuple representing direction (will be normalized)
        """
        # Normalize direction vector
        if direction_vector is None or np.linalg.norm(direction_vector) == 0:
            # Default direction (pointing right) when no direction is available
            direction_vector = np.array([1.0, 0.0])
        else:
            direction_vector = np.array(direction_vector)
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # Calculate angle from direction vector
        # Note: Screen coordinates may have inverted Y axis, so we might need to flip it
        # Add 180 degrees (π radians) to flip all triangles
        angle = math.atan2(-direction_vector[1], direction_vector[0]) + math.pi  # Flip Y for screen coordinates + 180° rotation
        
        # Define triangle vertices relative to center (pointing right by default)
        # The triangle's base (where two longer sides meet) points in the positive X direction (0 radians)
        triangle_points = [
            (-agent_size, 0),          # Back tip (sharp point at the rear)
            (agent_size * 0.6, -agent_size * 0.8),   # Front left (base corner)
            (agent_size * 0.6, agent_size * 0.8)     # Front right (base corner)
        ]
        
        # Rotate and translate triangle points
        rotated_points = []
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        for x, y in triangle_points:
            # Rotate point
            rotated_x = x * cos_angle - y * sin_angle
            rotated_y = x * sin_angle + y * cos_angle
            
            # Translate to screen position
            final_x = screen_x + rotated_x
            final_y = screen_y + rotated_y
            rotated_points.append(QPointF(final_x, final_y))
        
        # Create polygon and draw it
        triangle_polygon = QPolygonF(rotated_points)
        painter.drawPolygon(triangle_polygon)
    
    def draw_agents(self, painter):
        """Draw agents with smooth interpolated positions and performance optimizations"""
        if not self.agents:
            return
        
        # Limit number of agents drawn for performance (None means unlimited)
        agents_to_draw = self.agents
        if self.max_visible_agents is not None and len(agents_to_draw) > self.max_visible_agents:
            # Prioritize moving agents
            moving_agents = [agent for agent in agents_to_draw if agent.state == 'moving']
            waiting_agents = [agent for agent in agents_to_draw if agent.state == 'waiting']
            
            # Draw all moving agents first, then fill with waiting agents
            agents_to_draw = moving_agents[:self.max_visible_agents]
            remaining_slots = self.max_visible_agents - len(agents_to_draw)
            if remaining_slots > 0:
                agents_to_draw.extend(waiting_agents[:remaining_slots])
        
        drawn_agents = 0
        for agent in agents_to_draw:
            if self.max_visible_agents is not None and drawn_agents >= self.max_visible_agents:
                break
                
            # Use interpolated position if available
            position = self.get_interpolated_agent_position(agent)
            
            if position is not None:
                x, y = position
                
                # Viewport culling for agents
                if self.viewport_culling and not self.is_in_viewport(x, y):
                    continue
                
                screen_x, screen_y = self.world_to_screen(x, y)
                
                # Check if this is the selected agent
                is_selected = (self.selected_agent is not None and agent is self.selected_agent)
                
                # Get agent color based on type and state
                if agent.state == 'waiting':
                    color = QColor(0, 0, 0)  # Black for waiting
                else:
                    color = self.get_agent_color(agent)
                
                # Adaptive agent size based on zoom
                agent_size = max(1, int(self.agent_size * min(self.zoom_factor, 2.0)))
                
                # Determine direction for triangle orientation based on edge geometry
                direction_vector = None
                
                if agent.state == 'moving' and hasattr(agent, 'path') and hasattr(agent, 'path_index'):
                    # Calculate direction from current edge geometry
                    if agent.path_index < len(agent.path) - 1:
                        # Get current edge nodes
                        current_node = agent.path[agent.path_index]
                        next_node = agent.path[agent.path_index + 1]
                        
                        # Get node positions from graph
                        if (current_node in self.graph.nodes and next_node in self.graph.nodes):
                            current_pos = np.array(self.graph.nodes[current_node]['pos'])
                            next_pos = np.array(self.graph.nodes[next_node]['pos'])
                            
                            # Calculate edge direction vector
                            edge_vector = next_pos - current_pos
                            if np.linalg.norm(edge_vector) > 0:
                                direction_vector = edge_vector / np.linalg.norm(edge_vector)
                
                # Fallback to agent's stored direction if edge calculation fails
                if direction_vector is None:
                    if hasattr(agent, 'edge_direction') and agent.edge_direction is not None:
                        direction_vector = agent.edge_direction
                    elif hasattr(agent, 'velocity') and agent.velocity is not None:
                        velocity_magnitude = np.linalg.norm(agent.velocity)
                        if velocity_magnitude > 0.01:
                            direction_vector = agent.velocity
                
                
                
                # Draw the agent triangle
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color, 1))
                self.draw_agent_triangle(painter, screen_x, screen_y, agent_size, direction_vector)
                
                # Draw selection highlight on top if selected
                if is_selected:
                    # Draw bright yellow selection outline for the triangle
                    painter.setBrush(QBrush())  # No fill for the outline
                    painter.setPen(QPen(QColor(255, 255, 0), 3))  # Bright yellow, thick border
                    highlight_size = agent_size + 10
                    self.draw_agent_triangle(painter, screen_x, screen_y, highlight_size, direction_vector)
                    
                    # Draw inner yellow triangle for extra visibility
                    inner_size = agent_size + 8
                    painter.setBrush(QBrush(QColor(255, 255, 0, 100)))  # Semi-transparent yellow fill
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    self.draw_agent_triangle(painter, screen_x, screen_y, inner_size, direction_vector)
                drawn_agents += 1
    
    def get_interpolated_agent_position(self, agent):
        """Calculate smooth interpolated position for an agent using enhanced interpolation"""
        # If agent is waiting or has no position, use basic position
        if agent.state == 'waiting' or not hasattr(agent, 'position') or agent.position is None:
            return getattr(agent, 'position', None)
        
        # For moving agents, use the agent's enhanced interpolation if available
        if agent.state == 'moving' and hasattr(agent, 'get_interpolated_position'):
            try:
                # Use agent's own interpolation method with current time
                import time
                current_time = time.time()  # Use wall clock time for smooth rendering
                interpolated_pos = agent.get_interpolated_position(current_time)
                
                # Convert numpy array to tuple if needed
                if hasattr(interpolated_pos, 'tolist'):
                    return tuple(interpolated_pos.tolist())
                elif isinstance(interpolated_pos, (list, tuple)) and len(interpolated_pos) >= 2:
                    return (float(interpolated_pos[0]), float(interpolated_pos[1]))
                else:
                    return interpolated_pos
                    
            except (AttributeError, IndexError, TypeError):
                # Fall back to basic position if interpolation fails
                pass
        
        # Fallback: use the agent's current position
        if hasattr(agent, 'position'):
            try:
                # Convert numpy array to tuple if needed
                if hasattr(agent.position, 'tolist'):
                    return tuple(agent.position.tolist())
                elif isinstance(agent.position, (list, tuple)) and len(agent.position) >= 2:
                    return (float(agent.position[0]), float(agent.position[1]))
                else:
                    return agent.position
                    
            except (AttributeError, IndexError, TypeError):
                # Fall back to basic position if conversion fails
                pass
        
        # Default fallback
        return getattr(agent, 'position', None)
    
    def get_agent_color(self, agent):
        """Get the color for an agent based on its type and selection mode using configuration"""
        # Waiting agents use configured waiting color
        if agent.state == 'waiting':
            rgb = hex_to_qcolor_tuple(COLORS.WAITING_AGENT_COLOR)
            return QColor(*rgb)

        if self.selection_mode == 'random':
            # For random mode, use configured random color (red)
            rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['random']['default'])
            return QColor(*rgb)

        elif self.selection_mode == 'activity':
            # Color by agent type - use configured colors
            return self.type_colors.get(agent.agent_type, self._get_fallback_color())
        
        elif self.selection_mode == 'hub':
            # For hub mode, check if agent has target and whether it's a hub destination
            if hasattr(agent, 'target') and agent.target is not None:
                hub_data = self.st_selector.get_hub_nodes()
                hub_nodes = hub_data['hubs']
                if agent.target in hub_nodes:
                    # Hub destination
                    rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['hub_destination'])
                    return QColor(*rgb)
                else:
                    # Non-hub destination
                    rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['non_hub_destination'])
                    return QColor(*rgb)
            else:
                # Fallback: use agent_type if available (HubAndSpokeSelection sets this)
                if hasattr(agent, 'agent_type'):
                    if agent.agent_type == 'hub':
                        rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['hub_agent'])
                        return QColor(*rgb)
                    elif agent.agent_type == 'non-hub':
                        rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['non_hub_agent'])
                        return QColor(*rgb)
                # Default fallback for agents without target or agent_type info
                rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['fallback'])
                return QColor(*rgb)

        elif self.selection_mode == 'zone':
            # For zone-based selection
            if agent.agent_type == 'intra_zone':
                return self.type_colors.get('intra_zone', self._get_fallback_color())
            elif agent.agent_type == 'inter_zone':
                return self.type_colors.get('inter_zone', self._get_fallback_color())
        
        elif self.selection_mode == 'gravity':
            # Use configured gravity color
            rgb = hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['gravity']['default'])
            return QColor(*rgb)
            
        # Default fallback using agent type from configuration
        return self.type_colors.get(agent.agent_type, self._get_fallback_color())
    
    def _get_fallback_color(self):
        """Get fallback color from configuration"""
        rgb = hex_to_qcolor_tuple(COLORS.DEFAULT_AGENT_COLOR)
        return QColor(*rgb)
    
    def draw_legend(self, painter):
        """Draw legend showing colors for the current ST model"""
        if not self.graph:
            return
        
        # Legend configuration
        legend_margin = 15
        legend_width = 180
        legend_padding = 12
        item_height = 18
        color_box_size = 14
        
        # Get legend items based on current selection mode
        legend_items = self._get_legend_items()
        
        if not legend_items:
            return
        
        # Calculate legend dimensions
        legend_height = len(legend_items) * item_height + 2 * legend_padding + 45  # +45 for title and help text
        
        # Position legend in top-right corner with boundary check
        legend_x = max(10, self.width() - legend_width - legend_margin)  # Ensure it doesn't go off-screen
        legend_y = legend_margin
        
        # Draw legend background - fill first, then draw border
        painter.fillRect(legend_x, legend_y, legend_width, legend_height, QColor(250, 250, 250, 200))  # Light semi-transparent white
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.setBrush(QBrush())  # No brush for border
        painter.drawRect(legend_x, legend_y, legend_width, legend_height)
        
        # Draw legend title
        painter.setPen(QColor(40, 40, 40))  # Darker for better contrast
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        title_text = f"{self.selection_mode.title()} Model"
        title_rect = painter.boundingRect(legend_x + legend_padding, legend_y + legend_padding, 
                                        legend_width - 2 * legend_padding, 20, Qt.AlignCenter, title_text)
        painter.drawText(title_rect, Qt.AlignCenter, title_text)
        
        # Draw help text
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(80, 80, 80))  # Medium gray for help text
        help_text = "L:legend R:reset"
        help_rect = painter.boundingRect(legend_x + legend_padding, legend_y + legend_padding + 20, 
                                       legend_width - 2 * legend_padding, 15, Qt.AlignCenter, help_text)
        painter.drawText(help_rect, Qt.AlignCenter, help_text)
        
        # Draw legend items
        painter.setFont(QFont("Arial", 9))
        current_y = legend_y + legend_padding + 40  # Start below title and help text
        
        for label, color in legend_items:
            # Draw color box with proper border
            color_rect_x = legend_x + legend_padding
            color_rect_y = current_y + (item_height - color_box_size) // 2
            painter.setBrush(QBrush(color))  # Set brush for fill
            painter.fillRect(color_rect_x, color_rect_y, color_box_size, color_box_size, color)
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.setBrush(QBrush())  # Clear brush for border
            painter.drawRect(color_rect_x, color_rect_y, color_box_size, color_box_size)
            
            # Draw label text with good contrast
            painter.setPen(QColor(40, 40, 40))  # Dark text for readability
            text_x = color_rect_x + color_box_size + 8
            text_rect = painter.boundingRect(text_x, current_y, 
                                           legend_width - text_x + legend_x - legend_padding, item_height, 
                                           Qt.AlignVCenter | Qt.AlignLeft, label)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, label)
            
            current_y += item_height
    
    def _get_legend_items(self):
        """Get legend items (label, color) based on current selection mode"""
        legend_items = []
        
        if self.selection_mode == 'random':
            # Random model - single color for all agents
            color = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['random']['default']))
            legend_items.append(("Moving Agents", color))
            
        elif self.selection_mode == 'activity':
            # Activity model - different colors for each agent type
            # Show agent types in a specific order for better readability
            activity_types = ['commuter', 'delivery', 'leisure', 'business']
            for agent_type in activity_types:
                if agent_type in COLORS.AGENT_TYPE_COLORS:
                    hex_color = COLORS.AGENT_TYPE_COLORS[agent_type]
                    color = QColor(*hex_to_qcolor_tuple(hex_color))
                    legend_items.append((agent_type.title(), color))
                    
        elif self.selection_mode == 'zone':
            # Zone model - intra-zone vs inter-zone
            intra_color = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['zone']['intra_zone']))
            inter_color = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['zone']['inter_zone']))
            legend_items.append(("Intra-zone Trips", intra_color))
            legend_items.append(("Inter-zone Trips", inter_color))
            
        elif self.selection_mode == 'gravity':
            # Gravity model - single color (could be enhanced to show attraction levels)
            color = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['gravity']['default']))
            legend_items.append(("Moving Agents", color))
            
        elif self.selection_mode == 'hub':
            # Hub model - hub vs non-hub destinations
            hub_dest_color = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['hub_destination']))
            non_hub_dest_color = QColor(*hex_to_qcolor_tuple(COLORS.ST_MODEL_COLORS['hub']['non_hub_destination']))
            legend_items.append(("To Hub Nodes", hub_dest_color))
            legend_items.append(("To Non-Hub Nodes", non_hub_dest_color))
        
        # Add waiting agents color (common to all models)
        waiting_color = QColor(*hex_to_qcolor_tuple(COLORS.WAITING_AGENT_COLOR))
        legend_items.append(("Waiting Agents", waiting_color))
        
        return legend_items
    
    def reset_view(self):
        """Reset zoom and pan to default values"""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Force background redraw after view reset
        self.background_needs_redraw = True
        self.background_pixmap = None
        
        # Trigger repaint
        self.update()
        
        # Log the reset action
        self.add_log_message("View reset to default zoom and position")
    
    def toggle_legend(self):
        """Toggle legend visibility"""
        self.show_legend = not self.show_legend
        self.update()  # Trigger repaint
        return self.show_legend
    
    def keyPressEvent(self, event):
        """Handle key press events for legend toggle and other shortcuts"""
        if event.key() == Qt.Key_L:  # 'L' key to toggle legend
            visible = self.toggle_legend()
            self.add_log_message(f"Legend {'shown' if visible else 'hidden'} (Press 'L' to toggle)")
        elif event.key() == Qt.Key_R:  # 'R' key to reset view
            self.reset_view()
        else:
            super().keyPressEvent(event)
    

    
    def draw_reset_button(self, painter):
        """Draw the reset view button overlay"""
        if not hasattr(self, 'reset_button_rect') or not self.reset_button_rect:
            return
            
        # Set button background color (more opaque when hovered)
        if self.reset_button_hovered:
            painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
        else:
            painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
            
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRoundedRect(self.reset_button_rect, 3, 3)
        
        # Draw corner square icon
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        center_x = self.reset_button_rect.center().x()
        center_y = self.reset_button_rect.center().y()
        size = 6
        corner_length = 3
        
        # Top-left corner
        painter.drawLine(center_x - size, center_y - size, center_x - size + corner_length, center_y - size)  # horizontal
        painter.drawLine(center_x - size, center_y - size, center_x - size, center_y - size + corner_length)  # vertical
        
        # Top-right corner
        painter.drawLine(center_x + size, center_y - size, center_x + size - corner_length, center_y - size)  # horizontal
        painter.drawLine(center_x + size, center_y - size, center_x + size, center_y - size + corner_length)  # vertical
        
        # Bottom-left corner
        painter.drawLine(center_x - size, center_y + size, center_x - size + corner_length, center_y + size)  # horizontal
        painter.drawLine(center_x - size, center_y + size, center_x - size, center_y + size - corner_length)  # vertical
        
        # Bottom-right corner
        painter.drawLine(center_x + size, center_y + size, center_x + size - corner_length, center_y + size)  # horizontal
        painter.drawLine(center_x + size, center_y + size, center_x + size, center_y + size - corner_length)  # vertical
    
    def _is_reset_button_clicked(self, pos):
        """Check if the given position is within the reset button"""
        return self.reset_button_rect and self.reset_button_rect.contains(pos)
    
    def _update_reset_button_rect(self):
        """Update the reset button rectangle position based on current widget size"""
        if self.width() > 0 and self.height() > 0:
            x = self.width() - self.reset_button_size - self.reset_button_margin
            y = self.height() - self.reset_button_size - self.reset_button_margin
            self.reset_button_rect = QRect(x, y, self.reset_button_size, self.reset_button_size)
        else:
            self.reset_button_rect = None
    
    def draw_edge_labels_button(self, painter):
        """Draw the edge labels toggle button overlay"""
        if not hasattr(self, 'edge_labels_button_rect') or not self.edge_labels_button_rect:
            return
            
        # Set button background color (more opaque when hovered)
        if self.edge_labels_button_hovered:
            painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
        else:
            painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
            
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRoundedRect(self.edge_labels_button_rect, 3, 3)
        
        # Draw "T" icon to represent text/labels
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        center_x = self.edge_labels_button_rect.center().x()
        center_y = self.edge_labels_button_rect.center().y()
        
        # Draw T shape
        font_size = 8
        # Top horizontal line of T
        painter.drawLine(center_x - font_size//2, center_y - font_size//2, 
                        center_x + font_size//2, center_y - font_size//2)
        # Vertical line of T
        painter.drawLine(center_x, center_y - font_size//2, center_x, center_y + font_size//2)
        
        # Add visual indicator if labels are currently shown
        if self.show_edge_labels:
            painter.setBrush(QBrush(QColor(0, 200, 0, 150)))
            painter.setPen(QPen(QColor(0, 150, 0), 1))
            indicator_size = 4
            painter.drawEllipse(self.edge_labels_button_rect.right() - indicator_size - 2,
                              self.edge_labels_button_rect.top() + 2,
                              indicator_size, indicator_size)
    
    def _is_edge_labels_button_clicked(self, pos):
        """Check if the given position is within the edge labels button"""
        return self.edge_labels_button_rect and self.edge_labels_button_rect.contains(pos)
    
    def _update_edge_labels_button_rect(self):
        """Update the edge labels button rectangle position based on current widget size"""
        if self.width() > 0 and self.height() > 0:
            x = self.width() - self.edge_labels_button_size - self.edge_labels_button_margin
            # Position above the reset button
            y = self.height() - (2 * self.edge_labels_button_size) - (2 * self.edge_labels_button_margin)
            self.edge_labels_button_rect = QRect(x, y, self.edge_labels_button_size, self.edge_labels_button_size)
        else:
            self.edge_labels_button_rect = None
    
    def toggle_edge_labels(self):
        """Toggle the display of edge labels"""
        self.show_edge_labels = not self.show_edge_labels
        
        # Force a full redraw when toggling labels since they affect the background
        self.background_needs_redraw = True
        self.background_pixmap = None
        self.update()
        
        # Log the state change
        state = "enabled" if self.show_edge_labels else "disabled"
        self.add_log_message(f"Edge labels {state}")
    
    def draw_edge_label(self, painter, u, v, data, screen_x1, screen_y1, screen_x2, screen_y2):
        """Draw label for an edge based on zoom level and edge data"""
        # Only draw labels if zoom is sufficient (performance optimization)
        if self.zoom_factor < 0.5:
            return
        
        # Calculate midpoint of edge
        mid_x = (screen_x1 + screen_x2) / 2
        mid_y = (screen_y1 + screen_y2) / 2
        
                # Determine what to display as label - use second element, then first, then u-v
        label_text = ""
        if hasattr(data, 'get') and isinstance(data, dict):
            if data:  # Check if data dictionary is not empty
                # Try to get the second element first
                data_items = list(data.items())
                
                if len(data_items) >= 2:
                    # Use second element if available
                    second_key, second_value = data_items[1]
                    
                    # Format the second attribute value appropriately
                    if isinstance(second_value, (int, float)):
                        if second_key == 'length' and second_value > 1000:
                            # Special formatting for length values
                            length_km = second_value / 1000
                            label_text = f"{length_km:.1f}km"
                        else:
                            label_text = f"{second_value:.1f}"
                    else:
                        # For string or other types, truncate if too long
                        label_text = str(second_value)[:25] + "..."
                
                elif len(data_items) >= 1:
                    # Fall back to first element if only one element exists
                    first_key, first_value = data_items[0]
                    
                    # Format the first attribute value appropriately
                    if isinstance(first_value, (int, float)):
                        if first_key == 'length' and first_value > 1000:
                            # Special formatting for length values
                            length_km = first_value / 1000
                            label_text = f"{length_km:.1f}km"
                        else:
                            label_text = f"{first_value:.1f}"
                    else:
                        # For string or other types, truncate if too long
                        label_text = str(first_value)[:25] + "..."
        
        # Fallback to edge endpoints if no data available
        if not label_text:
            label_text = f"{u}-{v}"
        
        # Scale font size based on zoom level
        base_font_size = 2
        font_size = max(2, min(8, int(base_font_size * self.zoom_factor)))
        
        # Set up font and color for label
        font = QFont("Arial", font_size)
        painter.setFont(font)
        painter.setPen(QPen(QColor(50, 50, 150), 1))  # Dark blue color for labels
        
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(label_text)
        text_height = metrics.height()
        
        # Background rectangle
        #bg_rect = QRect(int(mid_x - text_width/2 - 2), int(mid_y - text_height/2 - 1),text_width + 4, text_height + 2)
        #painter.fillRect(bg_rect, QColor(255, 255, 255, 180))
        
        # Draw the label text
        painter.drawText(int(mid_x - text_width/2), int(mid_y + text_height/4), label_text)
    
    def draw_agent_info_panel(self, painter):
        """Draw the agent information panel for the selected agent"""
        if not self.selected_agent:
            return
        
        agent = self.selected_agent
        
        # Prepare agent information text
        info_lines = []
        
        # Handle agents that may not have ID (for backward compatibility)
        agent_id = getattr(agent, 'id', 'N/A')
        info_lines.append(f"Agent ID: {agent_id}")
        
        info_lines.append(f"Type: {agent.agent_type}")
        info_lines.append(f"State: {agent.state}")
        
        if agent.state == 'moving':
            if hasattr(agent, 'speed'):
                speed_kmh = agent.speed * 3.6  # Convert m/s to km/h
                info_lines.append(f"Speed: {speed_kmh:.1f} km/h")
            
            if hasattr(agent, 'source') and hasattr(agent, 'target'):
                info_lines.append(f"From: Node {agent.source}")
                info_lines.append(f"To: Node {agent.target}")
            
            if hasattr(agent, 'path') and hasattr(agent, 'path_index'):
                progress = (agent.path_index / max(1, len(agent.path) - 1)) * 100
                info_lines.append(f"Progress: {progress:.1f}%")
        
        # Handle trip_count attribute safely
        trip_count = getattr(agent, 'trip_count', 'N/A')
        info_lines.append(f"Trips: {trip_count}")
        
        # Calculate panel size based on text
        font = QFont("Arial", 10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        
        max_width = 0
        line_height = metrics.height()
        for line in info_lines:
            width = metrics.width(line)
            max_width = max(max_width, width)
        
        # Panel dimensions with padding
        padding = 10
        panel_width = max_width + 2 * padding
        panel_height = len(info_lines) * line_height + 2 * padding
        
        # Position panel at top-left corner
        panel_x = 10
        panel_y = 10
        
        # Draw panel background
        panel_rect = QRect(panel_x, panel_y, panel_width, panel_height)
        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))  # Semi-transparent white
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRoundedRect(panel_rect, 0, 0)
        
        # Draw text lines
        painter.setPen(QPen(QColor(20, 20, 20)))
        text_x = panel_x + padding
        text_y = panel_y + padding + metrics.ascent()
        
        for line in info_lines:
            painter.drawText(text_x, text_y, line)
            text_y += line_height
        
        # Store panel rect for potential future use (e.g., click detection)
        self.agent_info_rect = panel_rect
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event for file drops"""
        # Only accept drops if no graph is loaded
        if self.graph is not None:
            event.ignore()
            return
            
        if event.mimeData().hasUrls():
            # Check if any of the dragged files have supported extensions
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    # Handle .osm.pbf files specifically
                    if file_path.lower().endswith('.osm.pbf'):
                        file_ext = '.osm.pbf'
                    
                    if file_ext in self.supported_extensions:
                        event.acceptProposedAction()
                        self.drag_hover = True
                        self.update()
                        return
        
        event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        """Handle drag move event"""
        # Only accept if no graph is loaded and we have valid files
        if self.graph is not None:
            event.ignore()
            return
            
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self.drag_hover = False
        self.update()
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drop event"""
        # Only accept drops if no graph is loaded
        if self.graph is not None:
            event.ignore()
            return
            
        self.drag_hover = False
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            
            # Find the first supported file
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    # Handle .osm.pbf files specifically
                    if file_path.lower().endswith('.osm.pbf'):
                        file_ext = '.osm.pbf'
                    
                    if file_ext in self.supported_extensions:
                        # Try to load the graph through the network manager
                        self._load_dropped_file(file_path)
                        event.acceptProposedAction()
                        self.update()
                        return
        
        event.ignore()
        self.update()
    
    def _load_dropped_file(self, file_path):
        """Load a dropped file through the network manager with loading spinner"""
        try:
            # We need to access the main window through the parent chain
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'network_manager'):
                    # Found the main window with network manager
                    main_window = parent
                    break
                parent = parent.parent()
            
            if parent is None:
                # Fallback: try to find main window through simulation tab
                parent = self.parent()
                if hasattr(parent, 'parent_window') and parent.parent_window:
                    main_window = parent.parent_window
                else:
                    self.add_log_message("❌ Could not access network manager for drag and drop")
                    return
            
            # Use the threaded loading with spinner from network manager
            if hasattr(main_window, 'network_manager') and hasattr(main_window.network_manager, '_load_graph_with_spinner'):
                # Use the network manager's threaded loading method
                main_window.network_manager._load_graph_with_spinner(file_path)
            else:
                # Fallback to direct loading (synchronous)
                self._load_dropped_file_sync(file_path, main_window)
                
        except Exception as e:
            self.add_log_message(f"❌ Error during drag and drop: {str(e)}")
    
    def _load_dropped_file_sync(self, file_path, main_window):
        """Synchronous fallback for loading dropped files"""
        try:
            # Load the graph using the network manager
            if hasattr(main_window, 'graph_loader'):
                # Use the graph loader directly
                G = main_window.graph_loader.load_graph(file_path)
            elif hasattr(main_window, 'network_manager') and hasattr(main_window.network_manager, 'graph_loader'):
                # Use through network manager
                G = main_window.network_manager.graph_loader.load_graph(file_path)
            else:
                self.add_log_message("❌ Could not access graph loader for drag and drop")
                return
            
            if G is None:
                self.add_log_message("❌ Failed to load dropped file - unsupported format or invalid file")
                return
            
            # Set the graph in the network manager
            if hasattr(main_window, 'network_manager'):
                main_window.network_manager.graph = G
                main_window.simulation_widget.set_graph(G)
                
                # Configure performance settings
                main_window.network_manager._configure_performance_settings(G)
                
                # Enable start button and update UI
                main_window.start_button.setEnabled(True)
                main_window.add_log_message(f"Loaded: {os.path.basename(file_path)} (via drag & drop)")
                main_window.ui_manager.update_window_title(os.path.basename(file_path))
                
                # Update initial performance status
                main_window.status_manager.update_performance_status()
                
                # Update network statistics display
                main_window.network_manager.update_network_statistics()
            else:
                self.add_log_message("❌ Could not access network manager to set graph")
                
        except Exception as e:
            self.add_log_message(f"❌ Error loading dropped file: {str(e)}")
            import traceback
            self.add_log_message(f"Details: {traceback.format_exc()}")
            self.add_log_message(f"Details: {traceback.format_exc()}")

