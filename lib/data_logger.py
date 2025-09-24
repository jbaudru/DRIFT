import csv
import os
import datetime
import json


class SimulationDataLogger:
    def __init__(self, output_dir='data', simulation_start_time=None):
        self.output_dir = output_dir
        self.trips_data = []
        self.trip_counter = 0
        
        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided start time or current time
        self.simulation_start_time = simulation_start_time or datetime.datetime.now()
        
    def start_trip(self, agent, simulation_time):
        """Record the start of a new trip"""
        self.trip_counter += 1
        
        trip_data = {
            'trip_id': f'T{self.trip_counter:06d}',
            'agent_id': f'A{id(agent):06d}',  # Use object id as unique identifier
            'start_time': self._get_timestamp(simulation_time),
            'end_time': None,  # Will be filled when trip ends
            'origin_node': agent.source,
            'destination_node': agent.target,
            'route_taken': [agent.source],  # Will be updated as agent moves
            'trip_distance_km': 0.0,
            'trip_duration_min': 0.0,
            'average_speed_kmh': 0.0,
            'simulation_start_time': simulation_time,
            'path_nodes': agent.path.copy() if hasattr(agent, 'path') else []
        }
        
        # Store reference to current trip in agent
        agent.current_trip_data = trip_data
        
        return trip_data
    
    def update_trip(self, agent, simulation_time, current_node=None):
        """Update trip data as agent moves"""
        if not hasattr(agent, 'current_trip_data') or agent.current_trip_data is None:
            return
            
        trip_data = agent.current_trip_data
        
        # Update route if agent has moved to a new node
        if current_node and current_node not in trip_data['route_taken']:
            trip_data['route_taken'].append(current_node)
    
    def end_trip(self, agent, simulation_time, traffic_manager=None):
        """Record the end of a trip and add to dataset"""
        if not hasattr(agent, 'current_trip_data') or agent.current_trip_data is None:
            return
            
        trip_data = agent.current_trip_data
        
        # Calculate trip metrics
        trip_duration_seconds = simulation_time - trip_data['simulation_start_time']
        trip_data['trip_duration_min'] = trip_duration_seconds / 60.0
        trip_data['end_time'] = self._get_timestamp(simulation_time)
        
        # Calculate distance from path
        if hasattr(agent, 'path') and len(agent.path) > 1:
            total_distance = self._calculate_path_distance(agent.path, agent.graph)
            trip_data['trip_distance_km'] = total_distance / 1000.0  # Convert to km
        else:
            trip_data['trip_distance_km'] = 0.0
        
        # Calculate average speed
        if trip_data['trip_duration_min'] > 0:
            trip_data['average_speed_kmh'] = (trip_data['trip_distance_km'] / trip_data['trip_duration_min']) * 60
        else:
            trip_data['average_speed_kmh'] = 0.0
        
        # Ensure destination is in route
        if agent.target not in trip_data['route_taken']:
            trip_data['route_taken'].append(agent.target)
        
        # Clean up trip data for CSV export
        clean_trip_data = trip_data.copy()
        clean_trip_data['route_taken'] = str(clean_trip_data['route_taken'])  # Convert list to string
        del clean_trip_data['simulation_start_time']  # Remove internal field
        del clean_trip_data['path_nodes']  # Remove internal field
        
        # Add to trips dataset
        self.trips_data.append(clean_trip_data)
        
        # Clear agent's current trip
        agent.current_trip_data = None
        
        return clean_trip_data
    
    def _get_timestamp(self, simulation_time):
        """Convert simulation time to readable timestamp (HH:MM:SS format)"""
        real_datetime = self.simulation_start_time + datetime.timedelta(seconds=simulation_time)
        return real_datetime.strftime('%H:%M:%S')
    
    def _calculate_path_distance(self, path, graph):
        """Calculate total distance of a path through the network"""
        total_distance = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            try:
                # Get edge data and find the best edge
                edge_data = graph.get_edge_data(u, v)
                if edge_data:
                    best_edge = min(edge_data.values(), key=lambda x: x.get('length', float('inf')))
                    distance = best_edge.get('length', 0)
                    total_distance += distance
            except:
                # If edge not found, use estimated distance
                total_distance += 100  # 100m estimate
        
        return total_distance
    
    def save_to_csv(self, filename=None):
        """Save collected trip data to CSV file"""
        if not filename:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'simulation_trips_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not self.trips_data:
            print("No trip data to save.")
            return filepath
        
        # Define CSV columns in the simplified format
        fieldnames = [
            'trip_id', 'agent_id', 'start_time', 'end_time', 'origin_node', 
            'destination_node', 'route_taken', 'trip_distance_km', 'trip_duration_min',
            'average_speed_kmh'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for trip in self.trips_data:
                # Only write fields that exist in fieldnames
                filtered_trip = {k: v for k, v in trip.items() if k in fieldnames}
                writer.writerow(filtered_trip)
        
        print(f"Trip data saved to: {filepath}")
        print(f"Total trips recorded: {len(self.trips_data)}")
        return filepath
    
    def get_statistics(self):
        """Get summary statistics of collected data"""
        if not self.trips_data:
            return {}
        
        total_trips = len(self.trips_data)
        total_distance = sum(trip['trip_distance_km'] for trip in self.trips_data)
        total_duration = sum(trip['trip_duration_min'] for trip in self.trips_data)
        avg_speed = sum(trip['average_speed_kmh'] for trip in self.trips_data) / total_trips
        
        return {
            'total_trips': total_trips,
            'total_distance_km': total_distance,
            'total_duration_hours': total_duration / 60,
            'average_speed_kmh': avg_speed,
            'average_trip_distance_km': total_distance / total_trips,
            'average_trip_duration_min': total_duration / total_trips
        }
    
    def save_to_json(self, filename=None):
        """Save collected trip data to JSON file"""
        if not filename:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'simulation_trips_{timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not self.trips_data:
            print("No trip data to save.")
            return filepath
        
        # Prepare data for JSON export
        json_data = {
            'metadata': {
                'export_timestamp': datetime.datetime.now().isoformat(),
                'simulation_start_time': self.simulation_start_time.isoformat(),
                'total_trips': len(self.trips_data)
            },
            'trips': []
        }
        
        for trip in self.trips_data:
            # Convert route_taken from string back to list if needed
            route_taken = trip.get('route_taken', [])
            if isinstance(route_taken, str):
                try:
                    import ast
                    route_taken = ast.literal_eval(route_taken)
                except:
                    route_taken = []
            
            json_trip = {
                'trip_id': trip.get('trip_id'),
                'agent_id': trip.get('agent_id'),
                'start_time': trip.get('start_time'),
                'end_time': trip.get('end_time'),
                'origin_node': trip.get('origin_node'),
                'destination_node': trip.get('destination_node'),
                'route_taken': route_taken,
                'trip_distance_km': trip.get('trip_distance_km', 0.0),
                'trip_duration_min': trip.get('trip_duration_min', 0.0),
                'average_speed_kmh': trip.get('average_speed_kmh', 0.0)
            }
            json_data['trips'].append(json_trip)
        
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Trip data saved to: {filepath}")
        print(f"Total trips recorded: {len(self.trips_data)}")
        return filepath
    
    def export_to_format(self, filepath, format_type='csv'):
        """Export data to specified format and location"""
        if format_type.lower() == 'csv':
            return self.export_to_csv(filepath)
        elif format_type.lower() == 'json':
            return self.export_to_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def export_to_csv(self, filepath):
        """Export collected trip data to specified CSV file path"""
        if not self.trips_data:
            print("No trip data to export.")
            return filepath
        
        # Define CSV columns
        fieldnames = [
            'trip_id', 'agent_id', 'start_time', 'end_time', 'origin_node', 
            'destination_node', 'route_taken', 'trip_distance_km', 'trip_duration_min',
            'average_speed_kmh'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for trip in self.trips_data:
                # Only write fields that exist in fieldnames
                filtered_trip = {k: v for k, v in trip.items() if k in fieldnames}
                writer.writerow(filtered_trip)
        
        print(f"Trip data exported to: {filepath}")
        print(f"Total trips exported: {len(self.trips_data)}")
        return filepath
    
    def export_to_json(self, filepath):
        """Export collected trip data to specified JSON file path"""
        if not self.trips_data:
            print("No trip data to export.")
            return filepath
        
        # Prepare data for JSON export
        json_data = {
            'metadata': {
                'export_timestamp': datetime.datetime.now().isoformat(),
                'simulation_start_time': self.simulation_start_time.isoformat(),
                'total_trips': len(self.trips_data)
            },
            'trips': []
        }
        
        for trip in self.trips_data:
            # Convert route_taken from string back to list if needed
            route_taken = trip.get('route_taken', [])
            if isinstance(route_taken, str):
                try:
                    import ast
                    route_taken = ast.literal_eval(route_taken)
                except:
                    route_taken = []
            
            json_trip = {
                'trip_id': trip.get('trip_id'),
                'agent_id': trip.get('agent_id'),
                'start_time': trip.get('start_time'),
                'end_time': trip.get('end_time'),
                'origin_node': trip.get('origin_node'),
                'destination_node': trip.get('destination_node'),
                'route_taken': route_taken,
                'trip_distance_km': trip.get('trip_distance_km', 0.0),
                'trip_duration_min': trip.get('trip_duration_min', 0.0),
                'average_speed_kmh': trip.get('average_speed_kmh', 0.0)
            }
            json_data['trips'].append(json_trip)
        
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Trip data exported to: {filepath}")
        print(f"Total trips exported: {len(self.trips_data)}")
        return filepath
    
    def convert_csv_to_json(self, csv_filepath, json_filepath):
        """Convert CSV file to JSON format"""
        try:
            trips = []
            with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert route_taken from string to list
                    route_taken = row.get('route_taken', '[]')
                    try:
                        import ast
                        route_taken = ast.literal_eval(route_taken)
                    except:
                        route_taken = []
                    
                    trip = {
                        'trip_id': row.get('trip_id'),
                        'agent_id': row.get('agent_id'),
                        'start_time': row.get('start_time'),
                        'end_time': row.get('end_time'),
                        'origin_node': int(row.get('origin_node', 0)) if row.get('origin_node', '').isdigit() else row.get('origin_node'),
                        'destination_node': int(row.get('destination_node', 0)) if row.get('destination_node', '').isdigit() else row.get('destination_node'),
                        'route_taken': route_taken,
                        'trip_distance_km': float(row.get('trip_distance_km', 0.0)),
                        'trip_duration_min': float(row.get('trip_duration_min', 0.0)),
                        'average_speed_kmh': float(row.get('average_speed_kmh', 0.0))
                    }
                    trips.append(trip)
            
            json_data = {
                'metadata': {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'converted_from': csv_filepath,
                    'total_trips': len(trips)
                },
                'trips': trips
            }
            
            with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
            
            print(f"Converted {csv_filepath} to {json_filepath}")
            return json_filepath
            
        except Exception as e:
            print(f"Error converting CSV to JSON: {e}")
            return None
    
    def convert_json_to_csv(self, json_filepath, csv_filepath):
        """Convert JSON file to CSV format"""
        try:
            with open(json_filepath, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
            
            trips = data.get('trips', [])
            if not trips:
                print("No trips found in JSON file.")
                return csv_filepath
            
            fieldnames = [
                'trip_id', 'agent_id', 'start_time', 'end_time', 'origin_node', 
                'destination_node', 'route_taken', 'trip_distance_km', 'trip_duration_min',
                'average_speed_kmh'
            ]
            
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for trip in trips:
                    # Convert route_taken list to string for CSV
                    csv_trip = trip.copy()
                    csv_trip['route_taken'] = str(trip.get('route_taken', []))
                    writer.writerow(csv_trip)
            
            print(f"Converted {json_filepath} to {csv_filepath}")
            return csv_filepath
            
        except Exception as e:
            print(f"Error converting JSON to CSV: {e}")
            return None
