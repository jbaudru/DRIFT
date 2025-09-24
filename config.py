"""
Configuration module for the DRIFT simulation application.

This module centralizes all constants and configuration settings used throughout
the application to improve maintainability and reduce magic numbers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class UIConfig:
    """User Interface configuration constants."""
    
    # Application metadata
    APP_NAME: str = "DRIFT"
    
    # Window settings
    WINDOW_X: int = 100
    WINDOW_Y: int = 100
    WINDOW_WIDTH: int = 1400
    WINDOW_HEIGHT: int = 900
    
    # Icon sizes and paths
    ICON_BASE_PATH: str = "assets/icons"
    ICON_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [
        (16, 16), (24, 24), (32, 32), (48, 48), (256, 256)
    ])
    
    # Button icon paths
    PLAY_ICON_PATH: str = "assets/play.png"
    PAUSE_ICON_PATH: str = "assets/pause.png"
    STOP_ICON_PATH: str = "assets/stop.png"
    
    # UI Text labels
    LOAD_GRAPH_BUTTON: str = "Load Graph"
    START_SIMULATION_BUTTON: str = "Start Simulation"
    PAUSE_BUTTON: str = "Pause"
    RESUME_BUTTON: str = "Resume"
    ST_MODEL_LABEL: str = "ST Model:"
    AGENTS_LABEL: str = "Agents:"
    HOURS_LABEL: str = "Hours:"
    SPEED_LABEL: str = "Speed:"
    RESET_BUTTON: str = "Reset to Defaults"
    APPLY_BUTTON: str = "Apply Settings"
    
    # Tab names
    SIMULATION_TAB: str = "Simulation"
    LOG_TAB: str = "Log"
    TRIP_DATA_TAB: str = "Trip Data"
    STATISTICS_TAB: str = "Statistics"
    SETTINGS_TAB: str = "Settings"
    
    # Trips Tab labels
    COMPLETED_TRIPS_LABEL: str = "Completed Trips:"
    FORMAT_LABEL: str = "Format:"
    LOCATION_LABEL: str = "Location:"
    DEFAULT_LOCATION_LABEL: str = "Default location"
    BROWSE_BUTTON: str = "Browse"
    CONVERT_FORMAT_BUTTON: str = "Convert Format"
    EXPORT_CURRENT_DATA_BUTTON: str = "Export Current Data"
    
    # Settings Tab labels
    SIMULATION_PARAMETERS_LABEL: str = "Simulation Parameters"
    TRAFFIC_FLOW_LABEL: str = "Traffic Flow (BPR Function):"
    ALPHA_CONGESTION_LABEL: str = "α (Alpha - Congestion factor):"
    BETA_DISTANCE_LABEL: str = "β (Beta - Distance decay):"
    TRIP_GENERATION_LABEL: str = "Trip Generation Probability by Hour:"
    ST_MODEL_PARAMETERS_LABEL: str = "Source-Target Model Parameters"
    ZONE_BASED_MODEL_LABEL: str = "Zone-Based Model:"
    INTRA_ZONE_PROB_LABEL: str = "Intra-zone trip probability:"
    ACTIVITY_BASED_MODEL_LABEL: str = "Activity-Based Model:"
    COMMUTER_PROB_LABEL: str = "Commuter probability:"
    DELIVERY_PROB_LABEL: str = "Delivery probability:"
    LEISURE_PROB_LABEL: str = "Leisure probability:"
    BUSINESS_PROB_LABEL: str = "Business probability:"
    GRAVITY_MODEL_LABEL: str = "Gravity Model:"
    ATTRACTION_FACTOR_LABEL: str = "α (Attraction factor):"
    DISTANCE_DECAY_LABEL: str = "β (Distance decay):"
    HUB_SPOKE_MODEL_LABEL: str = "Hub-and-Spoke Model:"
    HUB_TRIP_PROB_LABEL: str = "Hub trip probability:"
    HUB_PERCENTAGE_LABEL: str = "Hub percentage:"
    
    # Simulation Tab status labels
    SIM_TIME_DEFAULT: str = "Simulation Time: 06:00:00"
    RUNNING_TIME_DEFAULT: str = "Running Time: 00:00:00"
    MOVING_AGENTS_DEFAULT: str = "Moving Agents: 0"
    NETWORK_UTIL_DEFAULT: str = "Network Utilization: 0.000%"
    PERFORMANCE_DEFAULT: str = "Performance: OK"
    
    # Statistics Tab labels
    NODES_LABEL: str = "Nodes:"
    EDGES_LABEL: str = "Edges:"
    AVG_DEGREE_LABEL: str = "Average Degree:"
    DENSITY_LABEL: str = "Density:"
    COMPONENTS_LABEL: str = "Connected Components:"
    DIAMETER_LABEL: str = "Network Diameter:"
    CLUSTERING_LABEL: str = "Average Clustering:"
    EXPORT_ALL_PLOTS_BUTTON: str = "📊 Export All Plots"
    
    # Statistics Tab plot titles
    MOVING_AGENTS_PLOT_TITLE: str = "Moving Agents vs Time"
    NETWORK_UTIL_PLOT_TITLE: str = "Network Utilization vs Time"
    AVG_SPEED_PLOT_TITLE: str = "Average Speed vs Time (Since Last Update)"
    AVG_DISTANCE_PLOT_TITLE: str = "Average Trip Distance vs Time (Since Last Update)"
    AVG_DURATION_PLOT_TITLE: str = "Average Trip Duration vs Time (Since Last Update)"
    AVG_NODES_PLOT_TITLE: str = "Average Nodes per Trip vs Time (Since Last Update)"
    AGENT_TYPE_DIST_PLOT_TITLE: str = "Active Agent Type Distribution vs Time"
    TRIP_COUNT_PLOT_TITLE: str = "Total Trip Count vs Time"
    
    # Plot axis labels
    TIME_AXIS_LABEL: str = "Time (24h format)"
    MOVING_AGENTS_AXIS_LABEL: str = "Number of Moving Agents"
    NETWORK_UTIL_AXIS_LABEL: str = "Network Utilization (%)"
    AVG_SPEED_AXIS_LABEL: str = "Average Speed (km/h)"
    AVG_DISTANCE_AXIS_LABEL: str = "Average Trip Distance (km)"
    AVG_DURATION_AXIS_LABEL: str = "Average Trip Duration (min)"
    AVG_NODES_AXIS_LABEL: str = "Average Number of Nodes"
    AGENT_TYPE_PERCENTAGE_AXIS_LABEL: str = "Agent Type Percentage (%)"
    TRIP_COUNT_AXIS_LABEL: str = "Total Number of Trips"
    
    # Tooltip texts
    BPR_ALPHA_TOOLTIP: str = "Standard BPR alpha parameter (typical: 0.15)"
    BPR_BETA_TOOLTIP: str = "Standard BPR beta parameter (typical: 4.0)"
    ZONE_INTRA_TOOLTIP: str = "Probability of trips within the same zone (typical: 0.6)"
    GRAVITY_ALPHA_TOOLTIP: str = "Centrality weight in gravity model"
    GRAVITY_BETA_TOOLTIP: str = "Distance penalty in gravity model"
    HUB_TRIP_TOOLTIP: str = "Probability that trips involve hub nodes"
    HUB_PERCENTAGE_TOOLTIP: str = "Percentage of nodes considered as hubs"
    AGENT_TYPE_PROBABILITY_TOOLTIP_TEMPLATE: str = "Probability of {} agent type"
    TRIP_GENERATION_HOUR_TOOLTIP_TEMPLATE: str = "Probability of trip generation at {:02d}:00"
    PERFORMANCE_TOOLTIP: str = ("Performance optimization status:\n"
                               "• Full: All elements rendered (small networks)\n"
                               "• LOD: Level-of-detail optimizations active\n"
                               "• LOD+: Aggressive optimizations for very large networks\n"
                               "• Thread: Simulation runs in optimized background thread\n"
                               "• Zoom with mouse wheel, pan with left click+drag\n"
                               "• Viewport culling hides off-screen elements")
    
    # Status label templates
    SIM_TIME_TEMPLATE: str = "Simulation Time: {}"
    REAL_TIME_TEMPLATE: str = "Real Time: {}"
    MOVING_AGENTS_TEMPLATE: str = "Moving Agents: {}"
    NETWORK_UTIL_TEMPLATE: str = "Network Utilization: {:.1f}%"
    
    # Default status values
    DEFAULT_SIM_TIME: str = "06:00:00"
    DEFAULT_REAL_TIME: str = "00:00:00"
    DEFAULT_MOVING_AGENTS: int = 0
    DEFAULT_NETWORK_UTIL: float = 0.0


@dataclass
class SimulationConfig:
    """Simulation parameter configuration constants."""
    
    # Default simulation parameters
    DEFAULT_NUM_AGENTS: int = 300
    MIN_AGENTS: int = 10
    MAX_AGENTS: int = 10000
    
    DEFAULT_DURATION_HOURS: int = 24
    MIN_DURATION: int = 1
    MAX_DURATION: int = 48
    
    DEFAULT_SPEED_MULTIPLIER: int = 50
    MIN_SPEED: int = 1
    MAX_SPEED: int = 1000
    SPEED_SUFFIX: str = "x"
    SPEED_TOOLTIP: str = "Simulation speed multiplier"
    
    # Selection modes
    SELECTION_MODES: List[str] = field(default_factory=lambda: ['random', 'activity', 'zone', 'hub', 'gravity'])
    DEFAULT_SELECTION_MODE: str = 'random'
    
    # Time constants
    SECONDS_PER_HOUR: int = 3600
    SIMULATION_START_HOUR: int = 6  # 6:00 AM
    SIMULATION_START_OFFSET: int = 21600  # 6 * 3600 seconds
    
    # Agent behavior constants
    MIN_WAIT_TIME: float = 60.0  # seconds
    DEFAULT_SPEED_KPH: float = 30.0  # 30 km/h default speed when not specified
    DEFAULT_EDGE_LENGTH: float = 1000.0  # 1000 meters default distance when not specified
    KPH_TO_MPS_FACTOR: float = 1000 / 3600  # Convert km/h to m/s
    
    # Update frequencies and performance
    SIMULATION_DT: float = 0.1  # Time step in seconds (reduced for smoother movement)
    DEFAULT_TIME_ACCELERATION: int = 10
    UPDATE_FREQUENCY: int = 1  # Update visualization every N steps (increased for smoother visuals)
    BATCH_SIZE: int = 50  # Process agents in batches (reduced for better responsiveness)


@dataclass
class StatisticsConfig:
    """Statistics and data tracking configuration constants."""
    
    # Data history limits
    STATS_HISTORY_MAXLEN: int = 1000  # Keep last 1000 data points
    AGENT_UPDATE_BUFFER_MAXLEN: int = 500  # Reduced for more responsive updates
    
    # Statistics update frequency - optimized for better performance
    STATS_UPDATE_STEP_FREQUENCY: int = 100  # Update statistics every N simulation steps (reduced frequency for better performance)
    
    # Plot formatting
    TIME_AXIS_FONTSIZE: int = 8
    TIME_LABEL_ROTATION: int = 45
    TIME_LABEL_ALIGNMENT: str = "right"
    
    # Time axis tick intervals (hours)
    TICK_INTERVAL_5MIN: float = 5/60   # 5 minutes
    TICK_INTERVAL_15MIN: float = 0.25  # 15 minutes  
    TICK_INTERVAL_30MIN: float = 0.5
    TICK_INTERVAL_1HOUR: float = 1.0
    TICK_INTERVAL_2HOUR: float = 2.0
    TICK_INTERVAL_4HOUR: float = 4.0
    
    # Time range thresholds for tick intervals
    TIME_RANGE_THRESHOLD_30MIN: float = 0.5  # For 5-min intervals
    TIME_RANGE_THRESHOLD_2H: int = 2         # For 15-min intervals
    TIME_RANGE_THRESHOLD_4H: int = 4
    TIME_RANGE_THRESHOLD_12H: int = 12
    TIME_RANGE_THRESHOLD_24H: int = 24
    
    # Trip table columns
    TRIP_TABLE_COLUMNS: List[str] = field(default_factory=lambda: [
        "Trip ID", "Agent ID", "Agent Type", "Start Node", "End Node", 
        "Start Time", "Duration (s)", "Distance (m)", "Avg Speed (km/h)", "Path Nodes"
    ])


@dataclass
class FileConfig:
    """File handling and export configuration constants."""
    
    # File dialog filters
    GRAPH_FILE_FILTER: str = (
        "All supported formats (*.json *.graphml *.osm *.pbf *.osm.pbf *.mtx *.csv);;"
        "JSON files (*.json);;"
        "GraphML files (*.graphml);;"
        "OSM XML files (*.osm);;"
        "OSM PBF files (*.pbf *.osm.pbf);;"
        "MTX files (*.mtx);;"
        "CSV files (*.csv);;"
        "All files (*)"
    )
    
    # Export formats
    EXPORT_FORMATS: List[str] = field(default_factory=lambda: ['CSV', 'JSON', 'XML'])
    DEFAULT_EXPORT_FORMAT: str = 'CSV'
    
    # Log formatting
    LOG_TIMESTAMP_FORMAT: str = "%H:%M:%S"
    LOG_MESSAGE_TEMPLATE: str = "[{}] {}"


@dataclass
class NetworkConfig:
    """Network analysis and validation configuration constants."""
    
    # Graph validation thresholds
    MAX_NODES_WARNING: int = 1000
    MAX_EDGES_PER_NODE_WARNING: int = 5
    
    # Graph sampling for performance tests
    PATHFINDING_TEST_SAMPLES: int = 10
    
    # Default network attributes
    DEFAULT_EDGE_WEIGHT: str = "weight"
    DEFAULT_EDGE_LENGTH: str = "length"
    DEFAULT_SPEED_ATTRIBUTE: str = "speed_kph"
    
    # BPR function default parameters
    DEFAULT_BPR_ALPHA: float = 0.15
    DEFAULT_BPR_BETA: float = 4.0


@dataclass
class ModelConfig:
    """Configuration for different transportation models."""
    
    # Zone-based model
    DEFAULT_ZONE_INTRA_PROBABILITY: float = 0.7
    
    # Activity-based model
    DEFAULT_AGENT_TYPE_DISTRIBUTIONS: Dict[str, float] = field(default_factory=dict)
    
    # Gravity model
    DEFAULT_GRAVITY_ALPHA: float = 1.0
    DEFAULT_GRAVITY_BETA: float = 2.0
    
    # Hub-and-spoke model
    DEFAULT_HUB_TRIP_PROBABILITY: float = 0.3
    DEFAULT_HUB_PERCENTAGE: float = 0.1
    
    # Hourly probability distribution (24 hours)
    DEFAULT_HOURLY_PROBABILITIES: Dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default distributions after dataclass creation."""
        if not self.DEFAULT_AGENT_TYPE_DISTRIBUTIONS:
            self.DEFAULT_AGENT_TYPE_DISTRIBUTIONS.update({
                'commuter': 0.4,
                'leisure': 0.3,
                'business': 0.2,
                'delivery': 0.1
            })
        
        if not self.DEFAULT_HOURLY_PROBABILITIES:
            # Peak hours at 8 AM and 6 PM, lower probabilities during night
            self.DEFAULT_HOURLY_PROBABILITIES.update({
                0: 0.05, 1: 0.02, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.05,
                6: 0.15, 7: 0.25, 8: 0.35, 9: 0.20, 10: 0.15, 11: 0.18,
                12: 0.22, 13: 0.18, 14: 0.16, 15: 0.20, 16: 0.25, 17: 0.30,
                18: 0.35, 19: 0.25, 20: 0.15, 21: 0.12, 22: 0.08, 23: 0.06
            })


@dataclass
class ColorConfig:
    """Color scheme configuration for UI elements."""
    
    # Button colors
    SUCCESS_BUTTON_COLOR: str = "#28A745"
    SUCCESS_BUTTON_TEXT_COLOR: str = "white"
    BUTTON_PADDING: str = "8px 16px"
    BUTTON_BORDER_RADIUS: str = "4px"
    
    # Export button colors
    EXPORT_BUTTON_COLOR: str = "#007BFF"
    EXPORT_BUTTON_TEXT_COLOR: str = "white"
    
    # ========== AGENT VISUALIZATION COLORS ==========
    
    # Base agent state colors
    WAITING_AGENT_COLOR: str = "#000000"  # Black for waiting agents
    DEFAULT_AGENT_COLOR: str = "#FF0000"  # Red fallback
    UNKNOWN_AGENT_COLOR: str = "#808080"  # Gray for unknown state
    
    # Agent type colors (activity-based model)
    AGENT_TYPE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'commuter': "#0000FF",    # Blue
        'delivery': "#FFA500",    # Orange  
        'leisure': "#800080",     # Purple
        'business': "#A52A2A",    # Brown
        'random': "#FF0000",      # Red
    })
    
    # ST Model-specific agent colors
    ST_MODEL_COLORS: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'random': {
            'default': "#FF0000",  # Red for all random agents
        },
        'activity': {
            # Uses AGENT_TYPE_COLORS above
            'fallback': "#FF0000"  # Red fallback
        },
        'zone': {
            'intra_zone': "#FF0000",    # Red for intra-zone trips
            'inter_zone': "#0000FF",    # Blue for inter-zone trips
            'zone_0': "#FF0000",        # Red - Zone colors cycle through
            'zone_1': "#0000FF",        # Blue
            'zone_2': "#00FF00",        # Green
            'zone_3': "#FFA500",        # Orange
            'zone_4': "#800080",        # Purple
            'zone_5': "#A52A2A",        # Brown
            'zone_6': "#FFC0CB",        # Pink
            'zone_7': "#808080",        # Gray
            'zone_8': "#808000",        # Olive
        },
        'gravity': {
            'default': "#FF0000",       # Red for gravity model
            'high_attraction': "#FF0000",   # Red for high attraction
            'medium_attraction': "#FFA500", # Orange for medium
            'low_attraction': "#FFFF00",    # Yellow for low
        },
        'hub': {
            'hub_destination': "#FF0000",     # Red for hub destinations
            'non_hub_destination': "#0000FF", # Blue for non-hub destinations
            'hub_agent': "#FF0000",           # Red for hub-related agents
            'non_hub_agent': "#0000FF",       # Blue for non-hub agents
            'fallback': "#808080",            # Gray for unknown
        }
    })
    
    # Activity node colors (for activity-based visualization)
    ACTIVITY_NODE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'activity_centers': "#800080",  # Purple
        'business_nodes': "#A52A2A",    # Brown
        'work_nodes': "#0000FF",        # Blue
        'home_nodes': "#008000",        # Green (standard green)
        'distribution_nodes': "#FFA500" # Orange
    })
    
    # Zone colors (cycling list for zone visualization)
    ZONE_COLORS: List[str] = field(default_factory=lambda: [
        "#FF0000",   # Red
        "#0000FF",   # Blue
        "#00FF00",   # Green
        "#FFA500",   # Orange
        "#800080",   # Purple
        "#A52A2A",   # Brown
        "#FFC0CB",   # Pink
        "#808080",   # Gray
        "#808000",   # Olive
        "#00FFFF",   # Cyan
        "#FF69B4",   # Hot Pink
        "#32CD32",   # Lime Green
    ])
    
    # ========== STATISTICS PLOT COLORS ==========
    
    # Plot background and styling
    PLOT_BACKGROUND_COLOR: str = "none"  # Transparent
    PLOT_FACE_COLOR: str = "white"
    PLOT_EDGE_COLOR: str = "none"
    PLOT_GRID_COLOR: str = "#E0E0E0"
    PLOT_GRID_ALPHA: float = 0.3
    
    # Statistics line plot colors (cycling through different metrics)
    STATS_LINE_COLORS: List[str] = field(default_factory=lambda: [
        "#1f77b4",   # Blue
        "#ff7f0e",   # Orange
        "#2ca02c",   # Green
        "#d62728",   # Red
        "#9467bd",   # Purple
        "#8c564b",   # Brown
        "#e377c2",   # Pink
        "#7f7f7f",   # Gray
        "#bcbd22",   # Olive
        "#17becf",   # Cyan
    ])
    
    # Agent type distribution plot colors (for statistics tab)
    AGENT_TYPE_PLOT_COLORS: List[str] = field(default_factory=lambda: [
        "#d62728",   # Red
        "#1f77b4",   # Blue
        "#2ca02c",   # Green
        "#ff7f0e",   # Orange
        "#9467bd",   # Purple
        "#8c564b",   # Brown
        "#e377c2",   # Pink
        "#7f7f7f",   # Gray
        "#bcbd22",   # Olive
    ])
    
    # Network utilization color scheme (traffic levels)
    NETWORK_UTILIZATION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'low': "#00CC00",      # Green (< 30% utilization)
        'medium': "#FFCC00",   # Yellow (30-70% utilization)  
        'high': "#FF9900",     # Orange (70-90% utilization)
        'critical': "#FF0000", # Red (> 90% utilization)
    })
    
    # Network visualization colors
    NETWORK_VISUALIZATION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'edge': "#505050",        # Dark gray for edges
        'node': "#2E2E2E",        # Darker gray for nodes
        'background': "#F0F0F0",  # Light gray background
    })
    
    # Network edge rendering settings
    EDGE_WIDTH: float = 0.5         # Thinner edges
    EDGE_ALPHA: int = 120           # Transparency (0-255, 120 is ~47% opacity)
    NODE_OUTLINE_ALPHA: int = 180   # Node outline transparency
    
    # Performance indicator colors
    PERFORMANCE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'excellent': "#00CC00",  # Green
        'good': "#99CC00",       # Light green
        'fair': "#FFCC00",       # Yellow
        'poor': "#FF9900",       # Orange
        'critical': "#FF0000",   # Red
    })
    
    # ========== UI ELEMENT COLORS ==========
    
    # Status indicator colors
    STATUS_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'running': "#28A745",    # Green
        'paused': "#FFC107",     # Yellow
        'stopped': "#DC3545",    # Red
        'loading': "#17A2B8",    # Blue
        'error': "#DC3545",      # Red
    })
    
    # Tab and panel colors
    PANEL_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'background': "#F8F9FA",
        'border': "#E9ECEF",
        'transparent': "transparent",
    })


# Color utility functions
def get_agent_color_by_type(agent_type: str) -> str:
    """Get color for a specific agent type."""
    return COLORS.AGENT_TYPE_COLORS.get(agent_type, COLORS.DEFAULT_AGENT_COLOR)


def get_agent_color_by_st_model(st_model: str, agent_type: str = None, **kwargs) -> str:
    """
    Get agent color based on ST model and additional parameters.
    
    Args:
        st_model: The source-target model ('random', 'activity', 'zone', 'gravity', 'hub')
        agent_type: The agent type (for activity model)
        **kwargs: Additional parameters like 'zone_id', 'is_hub_destination', etc.
    """
    model_colors = COLORS.ST_MODEL_COLORS.get(st_model, {})
    
    if st_model == 'activity' and agent_type:
        return get_agent_color_by_type(agent_type)
    elif st_model == 'zone':
        if agent_type in ['intra_zone', 'inter_zone']:
            return model_colors.get(agent_type, COLORS.DEFAULT_AGENT_COLOR)
        elif 'zone_id' in kwargs:
            zone_id = kwargs['zone_id']
            zone_key = f'zone_{zone_id % len(COLORS.ZONE_COLORS)}'
            return model_colors.get(zone_key, COLORS.ZONE_COLORS[zone_id % len(COLORS.ZONE_COLORS)])
    elif st_model == 'hub':
        if kwargs.get('is_hub_destination'):
            return model_colors.get('hub_destination', COLORS.DEFAULT_AGENT_COLOR)
        elif kwargs.get('is_non_hub_destination'):
            return model_colors.get('non_hub_destination', COLORS.DEFAULT_AGENT_COLOR)
        elif agent_type == 'hub':
            return model_colors.get('hub_agent', COLORS.DEFAULT_AGENT_COLOR)
        elif agent_type == 'non-hub':
            return model_colors.get('non_hub_agent', COLORS.DEFAULT_AGENT_COLOR)
        else:
            return model_colors.get('fallback', COLORS.UNKNOWN_AGENT_COLOR)
    elif st_model == 'gravity':
        attraction_level = kwargs.get('attraction_level', 'default')
        return model_colors.get(attraction_level, model_colors.get('default', COLORS.DEFAULT_AGENT_COLOR))
    
    return model_colors.get('default', COLORS.DEFAULT_AGENT_COLOR)


def get_zone_color(zone_id: int) -> str:
    """Get color for a specific zone ID."""
    return COLORS.ZONE_COLORS[zone_id % len(COLORS.ZONE_COLORS)]


def get_stats_line_color(line_index: int) -> str:
    """Get color for a statistics line plot."""
    return COLORS.STATS_LINE_COLORS[line_index % len(COLORS.STATS_LINE_COLORS)]


def get_agent_type_plot_color(type_index: int) -> str:
    """Get color for agent type distribution plots."""
    return COLORS.AGENT_TYPE_PLOT_COLORS[type_index % len(COLORS.AGENT_TYPE_PLOT_COLORS)]


def get_utilization_color(utilization_percent: float) -> str:
    """Get color based on network utilization percentage."""
    if utilization_percent < 30:
        return COLORS.NETWORK_UTILIZATION_COLORS['low']
    elif utilization_percent < 70:
        return COLORS.NETWORK_UTILIZATION_COLORS['medium']
    elif utilization_percent < 90:
        return COLORS.NETWORK_UTILIZATION_COLORS['high']
    else:
        return COLORS.NETWORK_UTILIZATION_COLORS['critical']


def hex_to_qcolor_tuple(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple for QColor."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


@dataclass
class PerformanceConfig:
    """Performance optimization configuration constants."""
    
    # Rendering optimization
    ENABLE_FAST_RENDERING: bool = True
    MAX_VISIBLE_AGENTS: int = 5000
    
    # Update intervals (milliseconds)
    GUI_UPDATE_INTERVAL: int = 50
    STATS_UPDATE_INTERVAL: int = 250  # Reduced for finer plot granularity
    
    # Memory management
    MAX_LOG_ENTRIES: int = 1000
    GARBAGE_COLLECT_INTERVAL: int = 100  # steps


# Global configuration instances
UI = UIConfig()
SIMULATION = SimulationConfig()
STATISTICS = StatisticsConfig()
FILES = FileConfig()
NETWORK = NetworkConfig()
MODELS = ModelConfig()
COLORS = ColorConfig()
PERFORMANCE = PerformanceConfig()


# Utility functions for configuration access
def get_icon_path(size: Tuple[int, int]) -> str:
    """Get the path for an icon of the specified size."""
    return f"{UI.ICON_BASE_PATH}/{size[0]}x{size[1]}.ico"


def get_all_icon_paths() -> List[str]:
    """Get all icon paths for different sizes."""
    return [get_icon_path(size) for size in UI.ICON_SIZES]


def format_time_from_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_display_time_from_simulation_time(simulation_time: float) -> float:
    """Convert simulation time to display time (starting from 6:00 AM)."""
    return simulation_time + SIMULATION.SIMULATION_START_OFFSET


def validate_configuration():
    """Validate configuration values and log warnings for potential issues."""
    warnings = []
    
    # Check for reasonable ranges
    if SIMULATION.MAX_AGENTS > 10000:
        warnings.append(f"MAX_AGENTS ({SIMULATION.MAX_AGENTS}) is very high and may cause performance issues")
    
    if STATISTICS.STATS_HISTORY_MAXLEN > 10000:
        warnings.append(f"STATS_HISTORY_MAXLEN ({STATISTICS.STATS_HISTORY_MAXLEN}) may consume significant memory")
    
    # Check model probability sums
    if abs(sum(MODELS.DEFAULT_AGENT_TYPE_DISTRIBUTIONS.values()) - 1.0) > 0.01:
        warnings.append("Agent type distributions do not sum to 1.0")
    
    return warnings
