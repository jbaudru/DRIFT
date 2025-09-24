from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QTimer

import random
import ast
import datetime
import time
import threading
import queue
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.agent import Agent
from lib.st_selection import RandomSelection, ActivityBasedSelection, ZoneBasedSelection, GravitySelection, HubAndSpokeSelection
from lib.managers.traffic_manager import TrafficManager
from lib.data_logger import SimulationDataLogger

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import SIMULATION, STATISTICS
    
    def _queue_visualization_update(self):
        """Queue visualization update for async processing"""
        self.signal_emission_queue.put(('visualization', self.simulation_time, self.step))
    
    def _queue_status_update(self):
        """Queue status update for async processing"""
        self.signal_emission_queue.put(('status', self.simulation_time, self.step))
    
    def _queue_progress_report(self):
        """Queue progress report for async processing"""
        self.signal_emission_queue.put(('progress', self.simulation_time, self.step))
    
    def _process_queued_signals(self):
        """Process queued signals asynchronously"""
        # Process pending trip completions first
        while not self.pending_trip_completions.empty():
            try:
                completion_data = self.pending_trip_completions.get_nowait()
                if completion_data:
                    trip_info = self._format_trip_completion_data(completion_data)
                    self.trip_completed.emit(trip_info)
            except queue.Empty:
                break
        
        # Process other queued signals
        signals_processed = 0
        max_signals_per_cycle = 3  # Limit to prevent UI blocking
        
        while signals_processed < max_signals_per_cycle and not self.signal_emission_queue.empty():
            try:
                signal_type, sim_time, step = self.signal_emission_queue.get_nowait()
                
                if signal_type == 'visualization':
                    with QMutexLocker(self.update_mutex):
                        agent_data = self._prepare_agent_data()
                        for agent in agent_data:
                            agent.simulation_time = sim_time
                        self.agents_updated.emit(agent_data)
                        self.last_visual_update = step
                        
                elif signal_type == 'status':
                    self._emit_status_update()
                    
                elif signal_type == 'progress':
                    self._emit_progress_report()
                
                signals_processed += 1
                
            except queue.Empty:
                break
    
    def _prepare_trip_completion_data(self, agent):
        """Prepare trip completion data for thread-safe processing"""
        if hasattr(self.data_logger, 'trips_data') and self.data_logger.trips_data:
            last_trip = self.data_logger.trips_data[-1]
            return {
                'agent': agent,
                'trip_data': last_trip.copy(),  # Make a copy for thread safety
                'trips_count': len(self.data_logger.trips_data)
            }
        return None
    
    def _format_trip_completion_data(self, completion_data):
        """Format trip completion data for emission"""
        agent = completion_data['agent']
        last_trip = completion_data['trip_data']
        
        # Parse route_taken back to list if it's a string
        route_taken = last_trip.get('route_taken', '[]')
        if isinstance(route_taken, str):
            try:
                route_taken = ast.literal_eval(route_taken)
            except:
                route_taken = []
        
        return {
            'trip_id': completion_data['trips_count'],
            'agent_id': getattr(agent, 'id', 'N/A'),
            'agent_type': agent.agent_type,
            'start_node': last_trip.get('origin_node', 'Unknown'),
            'end_node': last_trip.get('destination_node', 'Unknown'),
            'start_time': last_trip.get('start_time', '00:00:00'),
            'duration': last_trip.get('trip_duration_min', 0) * 60,
            'distance': last_trip.get('trip_distance_km', 0) * 1000,
            'avg_speed': last_trip.get('average_speed_kmh', 0) / 3.6,
            'path_nodes': route_taken
        }
    
    def _cleanup_threading_resources(self):
        """Clean up threading resources"""
        if hasattr(self, 'async_timer') and self.async_timer:
            self.async_timer.stop()
        
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        # Clear queues
        if hasattr(self, 'signal_emission_queue'):
            while not self.signal_emission_queue.empty():
                try:
                    self.signal_emission_queue.get_nowait()
                except queue.Empty:
                    break
        
        if hasattr(self, 'pending_trip_completions'):
            while not self.pending_trip_completions.empty():
                try:
                    self.pending_trip_completions.get_nowait()
                except queue.Empty:
                    breakanager import TrafficManager
from lib.data_logger import SimulationDataLogger

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIMULATION, STATISTICS, PERFORMANCE

class SimulationThread(QThread):

    log_message = pyqtSignal(str)
    agents_updated = pyqtSignal(list)
    simulation_finished = pyqtSignal(dict)
    trip_completed = pyqtSignal(dict)  # New signal for completed trips
    status_updated = pyqtSignal(dict)  # New signal for status updates
    st_selector_ready = pyqtSignal(object, str)
    
    def __init__(self, graph, num_agents, selection_mode, duration_hours=24):
        super().__init__()
        self.graph = graph
        self.num_agents = num_agents
        self.selection_mode = selection_mode
        self.duration_hours = duration_hours
        self.running = False
        self.paused = False  # Add pause state
        
        # Simulation state variables
        self.agents = []
        self.traffic_manager = None
        self.data_logger = None
        self.simulation_time = 0.0
        self.dt = SIMULATION.SIMULATION_DT
        self.time_acceleration = SIMULATION.DEFAULT_TIME_ACCELERATION
        self.max_simulation_time = duration_hours * SIMULATION.SECONDS_PER_HOUR
        self.agent_trip_states = {}
        self.last_report_time = 0
        self.step = 0
        
        # Performance optimizations
        self.agent_update_buffer = deque(maxlen=STATISTICS.AGENT_UPDATE_BUFFER_MAXLEN)
        self.update_mutex = QMutex()  # Thread safety
        self.batch_size = SIMULATION.BATCH_SIZE
        self.update_frequency = SIMULATION.UPDATE_FREQUENCY
        self.last_visual_update = 0
        
        # Pre-allocated arrays for better performance
        self.agent_positions = []
        self.agent_states = []
        
        # Settings storage
        self.settings = None
        
        # Pre-computed ST selector (can be set externally to skip computation)
        self.st_selector = None
        
        # Threading infrastructure for concurrent processing
        self.thread_pool = None
        self.max_worker_threads = min(4, (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4))
        self.batch_results_queue = queue.Queue()
        self.signal_emission_queue = queue.Queue()
        self.async_timer = QTimer()
        self.async_timer.timeout.connect(self._process_queued_signals)
        self.async_timer.setSingleShot(False)
        self.async_timer.setInterval(16)  # ~60 FPS for signal processing
        
        # Thread-safe data structures
        self.pending_trip_completions = queue.Queue()
        self.batch_processing_lock = threading.Lock()
        
    def apply_settings(self, settings):
        """Apply settings to the simulation thread"""
        self.settings = settings
        
        # Apply BPR parameters to traffic manager when it's created
        if hasattr(self, 'traffic_manager') and self.traffic_manager:
            if hasattr(self.traffic_manager, 'set_bpr_parameters'):
                self.traffic_manager.set_bpr_parameters(
                    settings['bpr_alpha'], 
                    settings['bpr_beta']
                )
    
    def update_runtime_settings(self, settings):
        """Update settings during runtime for active simulation"""
        self.settings = settings
        
        # Update BPR parameters in traffic manager
        if hasattr(self, 'traffic_manager') and self.traffic_manager:
            if hasattr(self.traffic_manager, 'set_bpr_parameters'):
                self.traffic_manager.set_bpr_parameters(
                    settings['bpr_alpha'], 
                    settings['bpr_beta']
                )
        
        # Update ST selector parameters if available
        if hasattr(self, 'st_selector') and self.st_selector:
            self.update_st_selector_parameters(settings)
            self.log_message.emit("Runtime settings updated successfully")
    
    def update_st_selector_parameters(self, settings):
        """Update ST selector parameters based on the current selection mode"""
        try:
            if self.selection_mode == 'zone' and hasattr(self.st_selector, 'set_parameters'):
                self.st_selector.set_parameters(intra_zone_probability=settings['zone_intra_probability'])
                self.log_message.emit(f"Zone model updated: intra-zone probability = {settings['zone_intra_probability']:.2f}")
            
            elif self.selection_mode == 'gravity' and hasattr(self.st_selector, 'set_parameters'):
                self.st_selector.set_parameters(
                    alpha=settings['gravity_alpha'],
                    beta=settings['gravity_beta']
                )
                self.log_message.emit(f"Gravity model updated: α={settings['gravity_alpha']:.1f}, β={settings['gravity_beta']:.1f}")
            
            elif self.selection_mode == 'hub' and hasattr(self.st_selector, 'set_parameters'):
                self.st_selector.set_parameters(
                    hub_trip_probability=settings['hub_trip_probability'],
                    hub_percentage=settings['hub_percentage']
                )
                self.log_message.emit(f"Hub model updated: trip_prob={settings['hub_trip_probability']:.2f}, hub_pct={settings['hub_percentage']:.2f}")
            
            elif self.selection_mode == 'activity':
                # For activity-based model, we can't change the node assignments during runtime,
                # but we could potentially update the agent type distributions for new agents
                self.log_message.emit("Activity model: parameters will apply to new agents in next simulation")
            
            else:
                # Random selection has no parameters to update
                self.log_message.emit("Random selection: no parameters to update")
                
        except Exception as e:
            self.log_message.emit(f"Error updating ST selector parameters: {str(e)}")
        
    def initialize_simulation(self):
        """Initialize simulation components"""
        try:
            # Initialize components
            self.traffic_manager = TrafficManager(self.graph)
            
            # Apply BPR settings if available
            if self.settings:
                self.traffic_manager.set_bpr_parameters(
                    self.settings['bpr_alpha'], 
                    self.settings['bpr_beta']
                )
            
            # TODO: Change start date based on today
            simulation_start_datetime = datetime.datetime(2025, 7, 10, 6, 0, 0)
            self.data_logger = SimulationDataLogger(output_dir='data', simulation_start_time=simulation_start_datetime)
            
            # Create agents based on selection mode
            self.log_message.emit(f"Initializing {self.num_agents} agents with {self.selection_mode} selection...")
            
            # Check if we have a pre-computed ST selector
            if self.st_selector is not None:
                self.log_message.emit("Using pre-computed ST selector model")
                st_selector = self.st_selector
                
                # Create agents with the pre-computed selector
                if self.selection_mode == 'activity':
                    # Handle activity-based agent type distribution
                    if self.settings and 'agent_type_distributions' in self.settings:
                        type_distribution = self.settings['agent_type_distributions']
                    else:
                        type_distribution = st_selector.get_agent_types_distribution()
                    
                    agent_types = list(type_distribution.keys())
                    type_weights = list(type_distribution.values())
                    
                    self.agents = []
                    for _ in range(self.num_agents):
                        agent_type = random.choices(agent_types, weights=type_weights)[0]
                        agent = Agent(self.graph, agent_type=agent_type, st_selector=st_selector, traffic_manager=self.traffic_manager)
                        self.agents.append(agent)
                elif self.selection_mode == 'gravity':
                    self.agents = [Agent(self.graph, agent_type='gravity', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                    model_info = st_selector.get_model_info()
                    self.log_message.emit(f"Pre-computed gravity model ready (α={model_info['alpha']}, β={model_info['beta']})")
                elif self.selection_mode == 'zone':
                    self.agents = [Agent(self.graph, agent_type='zone', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                    zone_info = st_selector.get_zone_info()
                    self.log_message.emit(f"Pre-computed zone model ready with {len(zone_info)} zones")
                elif self.selection_mode == 'hub':
                    self.agents = [Agent(self.graph, agent_type='hub', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                    self.log_message.emit("Pre-computed hub and spoke model ready")
                else:
                    # Default to random
                    self.agents = [Agent(self.graph, agent_type='random', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                
            elif self.selection_mode == 'activity':
                st_selector = ActivityBasedSelection(self.graph)
                
                # Apply activity-based settings if available
                if self.settings and 'agent_type_distributions' in self.settings:
                    # Use custom distribution from settings
                    type_distribution = self.settings['agent_type_distributions']
                else:
                    # Use default distribution
                    type_distribution = st_selector.get_agent_types_distribution()
                
                agent_types = list(type_distribution.keys())
                type_weights = list(type_distribution.values())
                
                self.agents = []
                for _ in range(self.num_agents):
                    agent_type = random.choices(agent_types, weights=type_weights)[0]
                    agent = Agent(self.graph, agent_type=agent_type, st_selector=st_selector, traffic_manager=self.traffic_manager)
                    self.agents.append(agent)
            
            elif self.selection_mode == 'zone':
                st_selector = ZoneBasedSelection(self.graph)
                
                # Apply zone-based settings if available
                if self.settings and 'zone_intra_probability' in self.settings:
                    if hasattr(st_selector, 'set_parameters'):
                        st_selector.set_parameters(intra_zone_probability=self.settings['zone_intra_probability'])
                
                self.agents = [Agent(self.graph, agent_type='zone', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                zone_info = st_selector.get_zone_info()
                self.log_message.emit(f"Zone-based selection initialized with {len(zone_info)} zones")
            
            elif self.selection_mode == 'gravity':
                st_selector = GravitySelection(self.graph)
                
                # Apply gravity model settings if available
                if self.settings:
                    alpha = self.settings.get('gravity_alpha', 1.0)
                    beta = self.settings.get('gravity_beta', 2.0)
                    st_selector.set_parameters(alpha=alpha, beta=beta)
                
                self.agents = [Agent(self.graph, agent_type='gravity', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                model_info = st_selector.get_model_info()
                self.log_message.emit(f"Gravity model initialized (α={model_info['alpha']}, β={model_info['beta']})")
                
                # Ensure the selection mode is properly stored
                self.log_message.emit(f"Selection mode confirmed: {self.selection_mode}")
            
            elif self.selection_mode == 'hub':
                st_selector = HubAndSpokeSelection(self.graph)
                
                # Apply hub-and-spoke settings if available
                if self.settings:
                    hub_trip_prob = self.settings.get('hub_trip_probability', 0.7)
                    hub_percentage = self.settings.get('hub_percentage', 0.15)
                    st_selector.set_parameters(hub_trip_probability=hub_trip_prob, hub_percentage=hub_percentage)
                
                self.agents = [Agent(self.graph, agent_type='hub', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
                model_info = st_selector.get_model_info()
                self.log_message.emit(f"Hub-and-spoke model initialized ({model_info['num_hubs']} hubs)")
            
            else:  # random
                st_selector = RandomSelection(self.graph)
                self.agents = [Agent(self.graph, agent_type='random', st_selector=st_selector, traffic_manager=self.traffic_manager) for _ in range(self.num_agents)]
            
            # Store selector for color mapping - CRITICAL: Ensure this is set correctly
            self.st_selector = st_selector
            
            print(f"DEBUG: SimulationThread - About to emit st_selector_ready with mode='{self.selection_mode}', selector={type(st_selector).__name__}")
            
            # Emit the ST selector and selection mode to the main window
            self.st_selector_ready.emit(st_selector, self.selection_mode)
            
            print(f"DEBUG: SimulationThread - st_selector_ready signal emitted successfully")
            
            # Initialize trips
            for agent in self.agents:
                self.data_logger.start_trip(agent, self.simulation_time)
                self.agent_trip_states[id(agent)] = True
            
            self.log_message.emit(f"Starting {self.duration_hours}-hour simulation...")
            return True
            
        except Exception as e:
            self.log_message.emit(f"Simulation initialization error: {str(e)}")
            return False
    
    ### ============================================================================== ###
    ## BOTTLENECKS ###
    
    def update_simulation_step(self):
        """Update one simulation step - optimized with concurrent processing"""
        if not self.running or self.simulation_time >= self.max_simulation_time:
            return False
            
        # Use accelerated time for simulation logic
        accelerated_dt = self.dt * self.time_acceleration
        self.simulation_time += accelerated_dt
        self.step += 1
        
        # Initialize thread pool if not exists
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_worker_threads)
            self.async_timer.start()
        
        # Process agents concurrently in batches
        self._process_agents_concurrent(accelerated_dt)
        
        # Update visualization only every N steps for smoother performance
        if self.step % self.update_frequency == 0:
            self._queue_visualization_update()
        
        # Update status more frequently for finer plot granularity 
        if self.step % STATISTICS.STATS_UPDATE_STEP_FREQUENCY == 0:
            self._queue_status_update()

        # Progress report every 30 minutes (adjusted for acceleration)
        if self.simulation_time - self.last_report_time >= 1800:  # Every 30 minutes
            self._queue_progress_report()
        
        return True
    
    def _process_agents_concurrent(self, accelerated_dt):
        """Process agents concurrently using thread pool"""
        if len(self.agents) == 0:
            return
            
        # Create batches for concurrent processing
        batches = []
        batch_start = 0
        while batch_start < len(self.agents):
            batch_end = min(batch_start + self.batch_size, len(self.agents))
            batches.append(self.agents[batch_start:batch_end])
            batch_start = batch_end
        
        # Process batches concurrently
        if len(batches) == 1:
            # Single batch - process directly to avoid threading overhead
            self._update_agent_batch(batches[0], accelerated_dt)
        else:
            # Multiple batches - use concurrent processing
            futures = []
            for batch in batches:
                future = self.thread_pool.submit(self._update_agent_batch_threadsafe, batch, accelerated_dt)
                futures.append(future)
            
            # Wait for all batches to complete with timeout
            for future in as_completed(futures, timeout=0.1):  # 100ms timeout
                try:
                    future.result()
                except Exception as e:
                    self.log_message.emit(f"Agent batch processing error: {str(e)}")
    
    def _update_agent_batch(self, agent_batch, accelerated_dt):
        """Update a batch of agents efficiently (original method)"""
        for agent in agent_batch:
            agent_id = id(agent)
            old_state = agent.state
            
            # Update agent with accelerated time step for faster simulation
            agent.update(accelerated_dt, self.simulation_time)
            
            # Track trip changes efficiently
            self._handle_trip_state_change(agent, agent_id, old_state)
            
            # Update trip data for moving agents
            if agent.state == 'moving' and self.agent_trip_states.get(agent_id, False):
                self._update_trip_data(agent)
    
    def _update_agent_batch_threadsafe(self, agent_batch, accelerated_dt):
        """Thread-safe version of agent batch update"""
        trip_completions = []
        
        for agent in agent_batch:
            agent_id = id(agent)
            old_state = agent.state
            
            # Update agent with accelerated time step for faster simulation
            agent.update(accelerated_dt, self.simulation_time)
            
            # Track trip changes efficiently (thread-safe version)
            completion_data = self._handle_trip_state_change_threadsafe(agent, agent_id, old_state)
            if completion_data:
                trip_completions.append(completion_data)
            
            # Update trip data for moving agents
            if agent.state == 'moving' and self.agent_trip_states.get(agent_id, False):
                self._update_trip_data(agent)
        
        # Queue trip completions for main thread processing
        for completion in trip_completions:
            self.pending_trip_completions.put(completion)
    
    def _handle_trip_state_change(self, agent, agent_id, old_state):
        """Handle trip state changes efficiently"""
        if old_state == 'waiting' and agent.state == 'moving':
            if not self.agent_trip_states.get(agent_id, False):
                self.data_logger.start_trip(agent, self.simulation_time)
                self.agent_trip_states[agent_id] = True
                
        elif old_state == 'moving' and agent.state == 'waiting':
            if self.agent_trip_states.get(agent_id, False):
                self.data_logger.end_trip(agent, self.simulation_time, self.traffic_manager)
                self.agent_trip_states[agent_id] = False
                
                # Queue trip completion data for batch emission
                self._queue_trip_completion(agent)
    
    def _handle_trip_state_change_threadsafe(self, agent, agent_id, old_state):
        """Thread-safe version of trip state change handling"""
        with self.batch_processing_lock:
            if old_state == 'waiting' and agent.state == 'moving':
                if not self.agent_trip_states.get(agent_id, False):
                    self.data_logger.start_trip(agent, self.simulation_time)
                    self.agent_trip_states[agent_id] = True
                    
            elif old_state == 'moving' and agent.state == 'waiting':
                if self.agent_trip_states.get(agent_id, False):
                    self.data_logger.end_trip(agent, self.simulation_time, self.traffic_manager)
                    self.agent_trip_states[agent_id] = False
                    
                    # Return trip completion data for queuing
                    return self._prepare_trip_completion_data(agent)
        return None
    
    def _queue_trip_completion(self, agent):
        """Queue trip completion data for efficient emission"""
        if hasattr(self.data_logger, 'trips_data') and self.data_logger.trips_data:
            last_trip = self.data_logger.trips_data[-1]
            
            # Parse route_taken back to list if it's a string
            route_taken = last_trip.get('route_taken', '[]')
            if isinstance(route_taken, str):
                try:
                    route_taken = ast.literal_eval(route_taken)
                except:
                    route_taken = []
            
            trip_info = {
                'trip_id': len(self.data_logger.trips_data),
                'agent_id': getattr(agent, 'id', 'N/A'),  # Add agent ID safely
                'agent_type': agent.agent_type,
                'start_node': last_trip.get('origin_node', 'Unknown'),
                'end_node': last_trip.get('destination_node', 'Unknown'),
                'start_time': last_trip.get('start_time', '00:00:00'),
                'duration': last_trip.get('trip_duration_min', 0) * 60,  # Convert minutes to seconds
                'distance': last_trip.get('trip_distance_km', 0) * 1000,  # Convert km to meters
                'avg_speed': last_trip.get('average_speed_kmh', 0) / 3.6,  # Convert km/h to m/s
                'path_nodes': route_taken
            }
            self.trip_completed.emit(trip_info)
    
    def _update_trip_data(self, agent):
        """Update trip data efficiently"""
        current_node = None
        if hasattr(agent, 'path') and hasattr(agent, 'path_index'):
            if agent.path_index < len(agent.path):
                current_node = agent.path[agent.path_index]
        self.data_logger.update_trip(agent, self.simulation_time, current_node)
    
    ### ============================================================================== ###
    
    def _prepare_agent_data(self):
        """Prepare lightweight agent data for visualization"""
        # Create a shallow copy with only essential data for rendering
        agent_data = []
        for agent in self.agents:
            # Only include data needed for visualization
            agent_copy = type('Agent', (), {
                'id': getattr(agent, 'id', None),  # Include agent ID for selection persistence
                'position': getattr(agent, 'position', (0, 0)),
                'state': agent.state,
                'agent_type': agent.agent_type,
                'trip_count': getattr(agent, 'trip_count', 0),  # Include trip count for info display
                'speed': getattr(agent, 'speed', 0),  # Include speed for info display
                'source': getattr(agent, 'source', None),  # Include source for info display
                'target': getattr(agent, 'target', None),  # Include target for info display
                'path': getattr(agent, 'path', []),  # Include path for progress calculation
                'path_index': getattr(agent, 'path_index', 0),  # Include path index for progress
                'current_node': getattr(agent, 'current_node', None),
                'target_node': getattr(agent, 'target_node', None),
                'progress': getattr(agent, 'progress', 0.0)
            })()
            agent_data.append(agent_copy)
        return agent_data

    ### ============================================================================== ###
    ### BOTTLENECKS ###
    
    def _emit_status_update(self):
        """Emit status update efficiently"""
        # Count moving agents efficiently and track their types
        moving_agents = [agent for agent in self.agents if agent.state == 'moving']
        moving_count = len(moving_agents)
        
        # Calculate agent type distribution for moving agents
        agent_type_counts = {}
        for agent in moving_agents:
            agent_type = getattr(agent, 'agent_type', 'unknown')
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1
        
        # Convert to percentages
        agent_type_percentages = {}
        if moving_count > 0:
            for agent_type, count in agent_type_counts.items():
                agent_type_percentages[agent_type] = (count / moving_count) * 100
        
        # Get network statistics (cache if possible)
        stats = self.traffic_manager.get_network_statistics()
        
        # Emit status update
        status_info = {
            'simulation_time': self.simulation_time,
            'moving_agents': moving_count,
            'total_agents': len(self.agents),
            'network_utilization': stats.get('network_utilization', 0.0),
            'active_agent_types': agent_type_percentages
        }
        self.status_updated.emit(status_info)
    
    def _emit_progress_report(self):
        """Emit progress report efficiently"""
        hours_elapsed = self.simulation_time / 3600
        trips_completed = len(self.data_logger.trips_data)
        simulation_start_datetime = datetime.datetime(2025, 7, 10, 6, 0, 0)
        current_time = simulation_start_datetime + datetime.timedelta(seconds=self.simulation_time)
        stats = self.traffic_manager.get_network_statistics()
        
        self.log_message.emit(f"Time: {current_time.strftime('%H:%M')} | "
                            f"Progress: {hours_elapsed:.1f}/{self.duration_hours}h | "
                            f"Trips: {trips_completed} | "
                            f"On roads: {stats['total_agents_on_roads']} | "
                            f"Utilization: {stats['network_utilization']:.1%} | "
                            f"Speed: {self.time_acceleration}x")
        self.last_report_time = self.simulation_time

    ### ============================================================================== ###
    
    
    def finish_simulation(self):
        """Finish the simulation and emit results"""
        # End all trips
        for agent in self.agents:
            if self.agent_trip_states.get(id(agent), False):
                self.data_logger.end_trip(agent, self.simulation_time, self.traffic_manager)
        
        # Save data and get statistics
        csv_file = self.data_logger.save_to_csv()
        stats = self.data_logger.get_statistics()
        stats['csv_file'] = csv_file
        
        self.log_message.emit(f"Simulation completed! Total trips: {stats.get('total_trips', 0)}")
        self.log_message.emit(f"Data saved to: {csv_file}")
        
        # Mark not running and emit statistics
        self.running = False
        self.simulation_finished.emit(stats)
    
    def set_performance_mode(self, mode):
        """Set performance optimization mode"""
        if mode == "ultra_smooth":
            self.update_frequency = 1  # Update every step
            self.batch_size = 10  # Smaller batches
        elif mode == "balanced":
            self.update_frequency = 1  # Update every step (improved for smoothness)
            self.batch_size = 50  # Medium batches
        elif mode == "performance":
            self.update_frequency = 1  # Update every step (still smooth but optimized)
            self.batch_size = 75  # Larger batches (reduced from 100)
        
        self.log_message.emit(f"Performance mode set to: {mode}")
    
    def set_speed_multiplier(self, multiplier):
        """Set the simulation speed multiplier"""
        self.time_acceleration = multiplier
        self.log_message.emit(f"Speed multiplier set to: {multiplier}x")
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        return {
            'update_frequency': self.update_frequency,
            'batch_size': self.batch_size,
            'agent_count': len(self.agents),
            'step': self.step,
            'simulation_time': self.simulation_time
        }
        
    def run(self):
        """Run the simulation in its own thread"""
        self.running = True
        
        # Initialize simulation
        if not self.initialize_simulation():
            return
        
        # Auto-adjust performance based on agent count
        agent_count = len(self.agents)
        if agent_count < 200:
            self.set_performance_mode("ultra_smooth")
        elif agent_count < 800:
            self.set_performance_mode("balanced")
        else:
            self.set_performance_mode("performance")
        
        # Run simulation loop in thread
        while self.running and self.simulation_time < self.max_simulation_time:
            start_time = time.time()
            
            # Check if paused - if so, just wait and continue
            if self.paused:
                self.msleep(5)  # Sleep for 50ms when paused
                continue
            
            # Update simulation step
            self.update_simulation_step()
            
            # Adaptive sleep to maintain consistent timing
            elapsed = time.time() - start_time
            target_frame_time = 0.033  # Target ~30 FPS equivalent for smoother animation
            sleep_time = max(0, target_frame_time - elapsed)
            
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))
        
        # Finish simulation
        if self.running:
            self.finish_simulation()
        
    def stop(self):
        """Stop the simulation"""
        self.running = False
    
    def pause(self):
        """Pause the simulation"""
        self.paused = True
        
    def resume(self):
        """Resume the simulation"""
        self.paused = False
