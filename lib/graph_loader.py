"""
Graph loading module for handling various graph file formats.

This module provides a centralized class for loading and processing graph data
from different file formats including JSON, GraphML, OSM XML, OSM PBF, MTX, and CSV.
"""

import os
import json
import xml.etree.ElementTree as ET
import networkx as nx
import random
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any, Callable


class GraphLoader:
    """
    A class to handle loading and processing of graph data from various file formats.
    
    Supports:
    - JSON format
    - GraphML format
    - OSM XML format
    - OSM PBF format (requires osmium)
    - MTX format (Matrix Market)
    - CSV format
    - Auto-detection of format
    """
    
    def __init__(self, main_window=None):
        """
        Initialize the GraphLoader.
        
        Args:
            main_window: Main window instance for logging
        """
        self.main_window = main_window
        
    def _log_message(self, message: str) -> None:
        """Log message using main window's logging system."""
        if self.main_window and hasattr(self.main_window, 'add_log_message'):
            self.main_window.add_log_message(message)
        else:
            print(f"[GraphLoader] {message}")
        
    def load_graph(self, file_path: str) -> Optional[nx.MultiDiGraph]:
        """
        Load a graph from various file formats with auto-detection.
        
        Args:
            file_path: Path to the graph file
            
        Returns:
            NetworkX MultiDiGraph or None if loading failed
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            # Handle .osm.pbf files specifically
            if file_path.lower().endswith('.osm.pbf'):
                file_ext = '.osm.pbf'
                
            self._log_message(f"Loading graph from {os.path.basename(file_path)} ({file_ext} format)...")
            
            G = None
            
            if file_ext == '.json':
                G = self._load_json_graph(file_path)
            elif file_ext == '.graphml':
                G = self._load_graphml_graph(file_path)
            elif file_ext == '.osm':
                G = self._load_osm_graph(file_path)
            elif file_ext in ['.pbf', '.osm.pbf']:
                G = self._load_osm_pbf_graph(file_path)
            elif file_ext == '.mtx':
                G = self._load_mtx_graph(file_path)
            elif file_ext == '.csv':
                G = self._load_csv_graph(file_path)
            else:
                # Try to auto-detect format
                self._log_message("Unknown file extension, attempting auto-detection...")
                G = self._load_graph_auto_detect(file_path)
            
            if G is None:
                self._log_message("❌ Failed to load graph - unsupported format or invalid file")
                return None
            
            # Ensure graph is a MultiDiGraph for simulation compatibility
            if not isinstance(G, nx.MultiDiGraph):
                self._log_message("Converting graph to MultiDiGraph for simulation compatibility...")
                G = nx.MultiDiGraph(G)
            
            # Analyze position data comprehensively
            pos_analysis = self._analyze_position_attributes(G)
            
            # Report position analysis results
            if pos_analysis['nodes_with_detectable_pos'] > 0:
                coverage = pos_analysis['coverage_percentage']
                self._log_message(f"Position coverage: {pos_analysis['nodes_with_detectable_pos']}/{pos_analysis['total_nodes']} nodes ({coverage:.1f}%)")
                
                if pos_analysis['attribute_counts']:
                    attrs_summary = ", ".join([f"{attr}({count})" for attr, count in pos_analysis['attribute_counts'].items()])
                    self._log_message(f"Found position attributes: {attrs_summary}")
                
                if pos_analysis['position_ranges']:
                    ranges = pos_analysis['position_ranges']
                    self._log_message(f"Coordinate ranges: X({ranges['x_range'][0]:.3f}, {ranges['x_range'][1]:.3f}), Y({ranges['y_range'][0]:.3f}, {ranges['y_range'][1]:.3f})")
                    
                    # Detect coordinate system type
                    x_span, y_span = ranges['x_span'], ranges['y_span']
                    if (-180 <= ranges['x_range'][0] <= 180) and (-90 <= ranges['y_range'][0] <= 90):
                        self._log_message("Detected coordinate system: Geographic (lat/lon)")
                    elif x_span > 1000 or y_span > 1000:
                        self._log_message("Detected coordinate system: Projected/Cartesian (large scale)")
                    else:
                        self._log_message("Detected coordinate system: Local/Cartesian (small scale)")
            else:
                self._log_message("No position data found - will generate layout")
                G = self._generate_graph_layout(G)
            
            # Ensure edge weights and validate structure
            self._ensure_edge_weights(G)
            self._validate_graph_for_simulation(G)
            
            # Final graph summary
            num_nodes = len(G.nodes())
            num_edges = len(G.edges())
            self._log_message(f"Graph loaded successfully: {num_nodes} nodes, {num_edges} edges")
            
            return G
            
        except Exception as e:
            self._log_message(f"❌ Error loading graph: {str(e)}")
            return None

    def _detect_and_extract_position(self, node_data: Dict[str, Any]) -> Optional[Tuple[float, float, List[str]]]:
        """
        Detect and extract position data from various attribute formats.
        
        Args:
            node_data: Dictionary containing node attributes
            
        Returns:
            tuple: (x, y, found_attrs) where found_attrs is a list of attribute names used
            None: if no valid position data found
        """
        x, y = None, None
        found_attrs = []
        
        # Common attribute names for X coordinate (longitude-like)
        x_attrs = ['x', 'lon', 'longitude', 'lng', 'long']
        # Common attribute names for Y coordinate (latitude-like) 
        y_attrs = ['y', 'lat', 'latitude']
        
        # Try to find X coordinate
        for x_attr in x_attrs:
            if x_attr in node_data:
                try:
                    x = float(node_data[x_attr])
                    found_attrs.append(f"x:{x_attr}")
                    break
                except (ValueError, TypeError):
                    continue
        
        # Try to find Y coordinate
        for y_attr in y_attrs:
            if y_attr in node_data:
                try:
                    y = float(node_data[y_attr])
                    found_attrs.append(f"y:{y_attr}")
                    break
                except (ValueError, TypeError):
                    continue
        
        # Check if we have a complete position
        if x is not None and y is not None:
            return (x, y, found_attrs)
        
        # Special case: check for 'pos' attribute as tuple/list
        if 'pos' in node_data:
            try:
                pos = node_data['pos']
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    x, y = float(pos[0]), float(pos[1])
                    found_attrs.append("pos:tuple")
                    return (x, y, found_attrs)
            except (ValueError, TypeError, IndexError):
                pass
        
        return None

    def _analyze_position_attributes(self, G: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Analyze what position attributes are available in the graph.
        Returns detailed information about position data coverage.
        
        Args:
            G: NetworkX graph to analyze
            
        Returns:
            Dictionary with position analysis results
        """
        total_nodes = len(G.nodes())
        if total_nodes == 0:
            return {"total_nodes": 0, "analysis": "Empty graph"}
        
        nodes_with_detectable_pos = 0
        attribute_counts = defaultdict(int)
        x_coords, y_coords = [], []
        
        for node_id, node_data in G.nodes(data=True):
            # Check if node already has 'pos' attribute
            if 'pos' in node_data and isinstance(node_data['pos'], (tuple, list)) and len(node_data['pos']) >= 2:
                try:
                    x, y = float(node_data['pos'][0]), float(node_data['pos'][1])
                    nodes_with_detectable_pos += 1
                    attribute_counts['pos'] += 1
                    x_coords.append(x)
                    y_coords.append(y)
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Try to detect position from other attributes
            pos_result = self._detect_and_extract_position(node_data)
            if pos_result:
                x, y, found_attrs = pos_result
                nodes_with_detectable_pos += 1
                for attr in found_attrs:
                    attribute_counts[attr] += 1
                x_coords.append(x)
                y_coords.append(y)
        
        # Calculate ranges if we have coordinates
        position_ranges = None
        if x_coords and y_coords:
            x_range = (min(x_coords), max(x_coords))
            y_range = (min(y_coords), max(y_coords))
            x_span = x_range[1] - x_range[0]
            y_span = y_range[1] - y_range[0]
            
            position_ranges = {
                'x_range': x_range,
                'y_range': y_range,
                'x_span': x_span,
                'y_span': y_span
            }
        
        coverage_percentage = (nodes_with_detectable_pos / total_nodes) * 100 if total_nodes > 0 else 0
        
        return {
            'total_nodes': total_nodes,
            'nodes_with_detectable_pos': nodes_with_detectable_pos,
            'coverage_percentage': coverage_percentage,
            'attribute_counts': dict(attribute_counts),
            'position_ranges': position_ranges,
            'coordinate_systems': [
                'Geographic (lat/lon in degrees)',
                'Projected/Cartesian (any units)',
                'Local coordinates (any scale)'
            ]
        }

    def _load_json_graph(self, file_path: str) -> nx.MultiDiGraph:
        """Load graph from JSON format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        G = nx.MultiDiGraph()
        
        # Track position detection statistics
        nodes_with_pos = 0
        pos_attrs_found = set()
        
        # Add nodes
        for node in data['nodes']:
            node_id = node['id']
            node_attrs = {k: v for k, v in node.items() if k != 'id'}
            
            # Try to detect and extract position data
            pos_result = self._detect_and_extract_position(node)
            if pos_result:
                x, y, found_attrs = pos_result
                node_attrs['pos'] = (x, y)
                nodes_with_pos += 1
                pos_attrs_found.update(found_attrs)
                
                # Remove original position attributes to avoid duplication
                for attr in ['x', 'y', 'lon', 'lat', 'longitude', 'latitude', 'lng', 'long']:
                    node_attrs.pop(attr, None)
            
            G.add_node(node_id, **node_attrs)
        
        # Add edges
        for link in data['links']:
            u, v = link['source'], link['target']
            if G.has_node(u) and G.has_node(v):
                link_data = link.copy()
                key = link_data.pop('key', 0)
                G.add_edge(u, v, key=key, **link_data)
        
        # Report position detection results
        total_nodes = len(G.nodes())
        if pos_attrs_found:
            self._log_message(f"Position attributes found: {', '.join(sorted(pos_attrs_found))}")
            self._log_message(f"{nodes_with_pos}/{total_nodes} nodes have position data")
        
        # Generate layout for nodes without position data if needed
        if nodes_with_pos < total_nodes:
            missing_pos = total_nodes - nodes_with_pos
            self._log_message(f"Generating layout for {missing_pos} nodes without position data...")
            G = self._generate_graph_layout(G)
        else:
            # Ensure edge weights and validate structure
            self._ensure_edge_weights(G)
            self._validate_graph_for_simulation(G)
        
        return G

    def _load_graphml_graph(self, file_path: str) -> nx.MultiDiGraph:
        """Load graph from GraphML format."""
        try:
            # First try NetworkX's built-in reader
            G = nx.read_graphml(file_path)
            
            # Convert to MultiDiGraph if needed
            if not isinstance(G, nx.MultiDiGraph):
                G = nx.MultiDiGraph(G)
            
            # Extract and process position data
            nodes_with_pos = 0
            pos_attrs_found = set()
            
            for node_id, node_data in G.nodes(data=True):
                pos_result = self._detect_and_extract_position(node_data)
                if pos_result:
                    x, y, found_attrs = pos_result
                    node_data['pos'] = (x, y)
                    nodes_with_pos += 1
                    pos_attrs_found.update(found_attrs)
                    
                    # Clean up redundant position attributes
                    for attr in ['x', 'y', 'lon', 'lat', 'longitude', 'latitude', 'lng', 'long']:
                        node_data.pop(attr, None)
            
            total_nodes = len(G.nodes())
            if pos_attrs_found:
                self._log_message(f"Position attributes found: {', '.join(sorted(pos_attrs_found))}")
                self._log_message(f"{nodes_with_pos}/{total_nodes} nodes have position data")
            
            # Generate layout for missing positions
            if nodes_with_pos < total_nodes:
                missing_pos = total_nodes - nodes_with_pos
                self._log_message(f"Generating layout for {missing_pos} nodes without position data...")
                G = self._generate_graph_layout(G)
            
            return G
            
        except Exception as e:
            self._log_message(f"❌ Error reading GraphML file: {str(e)}")
            return None

    def _load_osm_graph(self, file_path: str) -> nx.MultiDiGraph:
        """Load graph from OSM XML format."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            G = nx.MultiDiGraph()
            
            # Parse nodes
            nodes = {}
            for node in root.findall('node'):
                node_id = node.get('id')
                lat = float(node.get('lat'))
                lon = float(node.get('lon'))
                
                # Store node data
                nodes[node_id] = {'pos': (lon, lat)}  # Note: OSM uses (lon, lat) order
                
                # Parse tags
                for tag in node.findall('tag'):
                    key = tag.get('k')
                    value = tag.get('v')
                    nodes[node_id][key] = value
                
                G.add_node(node_id, **nodes[node_id])
            
            # Parse ways (edges)
            edge_count = 0
            for way in root.findall('way'):
                way_id = way.get('id')
                
                # Get node references
                nd_refs = [nd.get('ref') for nd in way.findall('nd')]
                
                # Parse way tags
                way_attrs = {'way_id': way_id}
                for tag in way.findall('tag'):
                    key = tag.get('k')
                    value = tag.get('v')
                    way_attrs[key] = value
                
                # Create edges between consecutive nodes
                for i in range(len(nd_refs) - 1):
                    u, v = nd_refs[i], nd_refs[i + 1]
                    if u in nodes and v in nodes:
                        G.add_edge(u, v, **way_attrs)
                        edge_count += 1
            
            self._log_message(f"Loaded {len(nodes)} nodes and {edge_count} edges from OSM XML")
            
            return G
            
        except Exception as e:
            self._log_message(f"❌ Error reading OSM XML file: {str(e)}")
            return None

    def _load_osm_pbf_graph(self, file_path: str) -> nx.MultiDiGraph:
        """Load graph from OSM PBF format using osmium."""
        try:
            import osmium
            
            class OSMHandler(osmium.SimpleHandler):
                def __init__(self):
                    osmium.SimpleHandler.__init__(self)
                    self.nodes = {}
                    self.ways = []
                
                def node(self, n):
                    self.nodes[n.id] = {
                        'pos': (n.location.lon, n.location.lat),
                        'tags': dict(n.tags)
                    }
                
                def way(self, w):
                    if len(w.nodes) >= 2:  # Only process ways with at least 2 nodes
                        node_refs = [n.ref for n in w.nodes]
                        tags = dict(w.tags)
                        self.ways.append({
                            'id': w.id,
                            'nodes': node_refs,
                            'tags': tags
                        })
            
            # Parse PBF file
            handler = OSMHandler()
            handler.apply_file(file_path)
            
            # Build NetworkX graph
            G = nx.MultiDiGraph()
            
            # Add nodes
            for node_id, node_data in handler.nodes.items():
                G.add_node(node_id, **node_data, **node_data['tags'])
            
            # Add edges from ways
            edge_count = 0
            for way in handler.ways:
                way_attrs = way['tags'].copy()
                way_attrs['way_id'] = way['id']
                
                # Create edges between consecutive nodes
                for i in range(len(way['nodes']) - 1):
                    u, v = way['nodes'][i], way['nodes'][i + 1]
                    if G.has_node(u) and G.has_node(v):
                        G.add_edge(u, v, **way_attrs)
                        edge_count += 1
            
            self._log_message(f"Loaded {len(handler.nodes)} nodes and {edge_count} edges from OSM PBF")
            
            return G
            
        except ImportError:
            self._log_message("❌ osmium library not available. Install with: pip install osmium")
            return None
        except Exception as e:
            self._log_message(f"❌ Error reading OSM PBF file: {str(e)}")
            return None

    def _load_csv_graph(self, file_path: str) -> nx.MultiDiGraph:
        """Load graph from CSV format."""
        import csv
        
        try:
            G = nx.MultiDiGraph()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect CSV format
                sample = f.read(1024)
                f.seek(0)
                
                # Auto-detect delimiter
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                headers = reader.fieldnames
                
                # Determine if this is an edge list or node list
                has_source_target = 'source' in headers and 'target' in headers
                has_from_to = 'from' in headers and 'to' in headers
                has_u_v = 'u' in headers and 'v' in headers
                
                if has_source_target or has_from_to or has_u_v:
                    # This is an edge list
                    self._log_message("Detected edge list format")
                    
                    # Determine column names
                    if has_source_target:
                        src_col, tgt_col = 'source', 'target'
                    elif has_from_to:
                        src_col, tgt_col = 'from', 'to'
                    else:
                        src_col, tgt_col = 'u', 'v'
                    
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=delimiter)
                    
                    for row in reader:
                        u, v = row[src_col], row[tgt_col]
                        
                        # Add nodes if they don't exist
                        if not G.has_node(u):
                            G.add_node(u)
                        if not G.has_node(v):
                            G.add_node(v)
                        
                        # Add edge with all attributes
                        edge_attrs = {k: v for k, v in row.items() if k not in [src_col, tgt_col]}
                        G.add_edge(u, v, **edge_attrs)
                
                else:
                    # This might be a node list
                    self._log_message("Detected node list format")
                    
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=delimiter)
                    
                    for row in reader:
                        # Try to find node ID
                        node_id = None
                        for id_col in ['id', 'node_id', 'node', 'name']:
                            if id_col in row:
                                node_id = row[id_col]
                                break
                        
                        if node_id is None:
                            # Use row number as ID
                            node_id = reader.line_num - 1
                        
                        # Add node with all attributes
                        node_attrs = {k: v for k, v in row.items() if k != 'id'}
                        
                        # Try to detect position data
                        pos_result = self._detect_and_extract_position(row)
                        if pos_result:
                            x, y, found_attrs = pos_result
                            node_attrs['pos'] = (x, y)
                        
                        G.add_node(node_id, **node_attrs)
            
            # Generate layout if needed
            nodes_with_pos = sum(1 for _, data in G.nodes(data=True) if 'pos' in data)
            if nodes_with_pos == 0:
                self._log_message("No position data found, generating layout...")
                G = self._generate_graph_layout(G)
            
            return G
            
        except Exception as e:
            self._log_message(f"❌ Error reading CSV file: {str(e)}")
            return None

    def _load_mtx_graph(self, file_path: str) -> nx.MultiDiGraph:
        """Load graph from Matrix Market (MTX) format."""
        try:
            from scipy.io import mmread
            
            # Read matrix
            matrix = mmread(file_path)
            
            # Convert to NetworkX graph
            G = nx.from_scipy_sparse_array(matrix, create_using=nx.MultiDiGraph)
            
            # Generate layout since MTX format doesn't include position data
            self._log_message("MTX format detected, generating layout...")
            G = self._generate_graph_layout(G)
            
            return G
            
        except ImportError:
            self._log_message("❌ scipy library not available. Install with: pip install scipy")
            return None
        except Exception as e:
            self._log_message(f"❌ Error reading MTX file: {str(e)}")
            return None

    def _load_graph_auto_detect(self, file_path: str) -> nx.MultiDiGraph:
        """Attempt to auto-detect and load graph format."""
        # Try different formats in order of likelihood
        formats_to_try = [
            ('JSON', self._load_json_graph),
            ('GraphML', self._load_graphml_graph),
            ('CSV', self._load_csv_graph),
            ('OSM XML', self._load_osm_graph),
        ]
        
        for format_name, loader_func in formats_to_try:
            try:
                self._log_message(f"Trying {format_name} format...")
                G = loader_func(file_path)
                if G is not None and len(G.nodes()) > 0:
                    self._log_message(f"Successfully loaded as {format_name}")
                    return G
            except Exception as e:
                self._log_message(f"❌ {format_name} failed: {str(e)}")
                continue
        
        self._log_message("❌ Auto-detection failed for all known formats")
        return None

    def _generate_graph_layout(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Generate layout for nodes without position data."""
        if len(G.nodes()) == 0:
            return G
        
        # Find nodes that need positions
        nodes_needing_pos = []
        nodes_with_pos = {}
        
        for node_id, node_data in G.nodes(data=True):
            if 'pos' in node_data and isinstance(node_data['pos'], (tuple, list)) and len(node_data['pos']) >= 2:
                try:
                    x, y = float(node_data['pos'][0]), float(node_data['pos'][1])
                    nodes_with_pos[node_id] = (x, y)
                except (ValueError, TypeError):
                    nodes_needing_pos.append(node_id)
            else:
                nodes_needing_pos.append(node_id)
        
        if not nodes_needing_pos:
            return G  # All nodes already have positions
        
        # Choose layout algorithm based on graph size
        num_nodes = len(nodes_needing_pos)
        
        if num_nodes <= 100:
            # Use spring layout for small graphs
            layout_func = nx.spring_layout
            layout_name = "spring"
        elif num_nodes <= 1000:
            # Use circular layout for medium graphs
            layout_func = nx.circular_layout
            layout_name = "circular"
        else:
            # Use random layout for large graphs
            layout_func = nx.random_layout
            layout_name = "random"
        
        self._log_message(f"Generating {layout_name} layout for {num_nodes} nodes...")
        
        # Create subgraph with nodes needing positions
        subgraph = G.subgraph(nodes_needing_pos)
        
        # Generate layout
        try:
            if layout_func == nx.spring_layout and num_nodes > 50:
                # Use fewer iterations for large spring layouts
                pos = layout_func(subgraph, iterations=20)
            else:
                pos = layout_func(subgraph)
            
            # Scale layout if we have existing positioned nodes
            if nodes_with_pos:
                existing_coords = list(nodes_with_pos.values())
                x_coords = [coord[0] for coord in existing_coords]
                y_coords = [coord[1] for coord in existing_coords]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Scale new positions to match existing coordinate system
                scale_x = (x_max - x_min) if x_max != x_min else 1.0
                scale_y = (y_max - y_min) if y_max != y_min else 1.0
                
                for node_id in pos:
                    x, y = pos[node_id]
                    scaled_x = x_min + x * scale_x
                    scaled_y = y_min + y * scale_y
                    pos[node_id] = (scaled_x, scaled_y)
            
            # Apply positions to graph
            for node_id, (x, y) in pos.items():
                G.nodes[node_id]['pos'] = (x, y)
                
        except Exception as e:
            self._log_message(f"❌ Layout generation failed: {str(e)}, using random positions")
            
            # Fallback to simple random positions
            for node_id in nodes_needing_pos:
                x = random.uniform(0, 1)
                y = random.uniform(0, 1)
                G.nodes[node_id]['pos'] = (x, y)
        
        return G

    def _ensure_edge_weights(self, G: nx.MultiDiGraph) -> None:
        """Ensure all edges have weight attributes."""
        for u, v, key, data in G.edges(keys=True, data=True):
            if 'weight' not in data:
                # Try to calculate weight from position data
                if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
                    try:
                        x1, y1 = G.nodes[u]['pos']
                        x2, y2 = G.nodes[v]['pos']
                        weight = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        data['weight'] = weight
                    except (ValueError, TypeError):
                        data['weight'] = 1.0
                else:
                    data['weight'] = 1.0

    def _validate_graph_for_simulation(self, G: nx.MultiDiGraph) -> None:
        """Validate that the graph is suitable for simulation."""
        if len(G.nodes()) == 0:
            self._log_message("Graph has no nodes")
            return
        
        if len(G.edges()) == 0:
            self._log_message("Graph has no edges")
            return
        
        # Check connectivity
        if nx.is_strongly_connected(G):
            self._log_message("Graph is strongly connected")
        elif nx.is_weakly_connected(G):
            self._log_message("Graph is weakly connected (may affect simulation)")
        else:
            components = list(nx.weakly_connected_components(G))
            self._log_message(f"Graph has {len(components)} disconnected components")
        
        # Check for self-loops
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            self._log_message(f"Graph contains {len(self_loops)} self-loops")
        
        # Check node degrees
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            self._log_message(f"Graph has {len(isolated_nodes)} isolated nodes")

    def _normalize_node_ids(self, G):
        """Normalize node IDs to strings for consistency"""
        # Check if node IDs need normalization
        non_string_nodes = [n for n in G.nodes() if not isinstance(n, str)]
        
        if non_string_nodes:
            self._log_message(f"Normalizing {len(non_string_nodes)} non-string node IDs to strings")
            
            # Create mapping from old IDs to new string IDs
            mapping = {}
            for node in G.nodes():
                if isinstance(node, str):
                    mapping[node] = node  # Keep strings as-is
                else:
                    mapping[node] = str(node)  # Convert others to strings
            
            # Relabel nodes
            G = nx.relabel_nodes(G, mapping)
            self._log_message("Node IDs normalized to strings")

        return G
