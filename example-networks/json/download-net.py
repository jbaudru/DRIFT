import osmnx as ox
import json

def download_road_network(city_name, country=None, network_type='drive', output_file=None):
    """
    Download road network from OpenStreetMap using OSMnx and save as JSON.
    
    Parameters:
    -----------
    city_name : str
        Name of the city (e.g., "Barcelona", "Paris")
    country : str, optional
        Country name for disambiguation (e.g., "Spain", "France")
    network_type : str, default='drive'
        Type of street network to get. Options: 'drive', 'walk', 'bike', 'all'
    output_file : str, optional
        Output JSON filename. If None, uses city_name.json
    
    Returns:
    --------
    str : Path to the saved JSON file
    """
    
    # Construct place query
    place_query = f"{city_name}, {country}" if country else city_name
    
    print(f"Downloading road network for: {place_query}")
    
    # Download the street network
    G = ox.graph_from_place(place_query, network_type=network_type)
    
    # Add edge speeds and travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    print(f"Downloaded {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Convert to JSON format
    print("Converting to JSON format...")
    graph_data = ox.graph_to_gdfs(G, nodes=True, edges=True)
    
    # Prepare the JSON structure
    json_data = {
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph(),
        "graph": {
            "created_with": f"OSMnx {ox.__version__}",
            "crs": "epsg:4326",
            "simplified": True,
            "place": place_query,
            "network_type": network_type
        },
        "nodes": [],
        "links": []
    }
    
    # Add nodes
    nodes_gdf = graph_data[0]
    for node_id, node_data in nodes_gdf.iterrows():
        node_dict = {
            "id": int(node_id),
            "y": float(node_data.geometry.y),
            "x": float(node_data.geometry.x)
        }
        # Add additional node attributes if available
        if 'street_count' in node_data:
            node_dict['street_count'] = int(node_data['street_count'])
        
        json_data["nodes"].append(node_dict)
    
    # Add edges (links)
    edges_gdf = graph_data[1]
    for (u, v, key), edge_data in edges_gdf.iterrows():
        edge_dict = {
            "source": int(u),
            "target": int(v),
            "key": int(key)
        }
        
        # Add common edge attributes
        for attr in ['osmid', 'name', 'highway', 'oneway', 'reversed', 
                     'length', 'lanes', 'maxspeed', 'speed_kph', 'travel_time']:
            if attr in edge_data:
                value = edge_data[attr]
                
                # Skip NaN and None values
                if value is None:
                    continue
                
                # Check for NaN in numeric types
                if isinstance(value, float):
                    import math
                    if math.isnan(value):
                        continue
                    edge_dict[attr] = value
                elif isinstance(value, (int, bool, str)):
                    edge_dict[attr] = value
                elif isinstance(value, list):
                    # Filter out NaN values from lists
                    import math
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, float) and math.isnan(item):
                            continue
                        cleaned_list.append(item)
                    if cleaned_list:  # Only add if list is not empty after filtering
                        edge_dict[attr] = cleaned_list
                else:
                    edge_dict[attr] = str(value)
        
        json_data["links"].append(edge_dict)
    
    # Determine output filename
    if output_file is None:
        output_file = f"{city_name.replace(' ', '_').replace(',', '')}.json"
    
    # Save to JSON file
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)
    
    print(f"Successfully saved road network to {output_file}")
    return output_file


# Example usage
if __name__ == "__main__":
    # Example 1: Download Barcelona road network
    download_road_network("Nancy", country="France", network_type='drive')
    
    # Example 2: Download another city
    # download_road_network("Paris", country="France", network_type='drive')
    
    # Example 3: With custom output filename
    # download_road_network("Brussels", country="Belgium", 
    #                      network_type='drive', output_file='Brussels_network.json')
