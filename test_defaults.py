#!/usr/bin/env python3
"""
Test script to verify that default values are properly applied to network edges
when speed_kph and length attributes are missing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from lib.graph_loader import GraphLoader
from config import SIMULATION

def test_default_values():
    """Test that default values are applied when edge attributes are missing."""
    print("Testing default value assignment for network edges...")
    print(f"Expected defaults: Speed = {SIMULATION.DEFAULT_SPEED_KPH} km/h, Length = {SIMULATION.DEFAULT_EDGE_LENGTH} m")
    print()
    
    # Create a simple test graph with missing attributes
    G = nx.MultiDiGraph()
    
    # Add nodes with positions
    G.add_node('A', pos=(0, 0))
    G.add_node('B', pos=(100, 0))
    G.add_node('C', pos=(200, 100))
    
    # Add edges with various missing attributes
    G.add_edge('A', 'B', key=0)  # No speed or length
    G.add_edge('B', 'C', key=0, speed_kph=None, length=None)  # Explicitly None values
    G.add_edge('C', 'A', key=0, speed_kph="invalid", length="invalid")  # Invalid string values
    G.add_edge('A', 'C', key=0, speed_kph=50, length=500)  # Valid values should be preserved
    
    print("Original edge data:")
    for u, v, key, data in G.edges(keys=True, data=True):
        print(f"  {u}->{v}: {data}")
    print()
    
    # Create a GraphLoader instance and process the graph
    loader = GraphLoader()
    
    # Apply the edge weight processing
    loader._ensure_edge_weights(G)
    
    print("After applying defaults:")
    for u, v, key, data in G.edges(keys=True, data=True):
        print(f"  {u}->{v}: {data}")
    print()
    
    # Verify that defaults were applied correctly
    test_passed = True
    expected_results = [
        ('A', 'B', SIMULATION.DEFAULT_SPEED_KPH, SIMULATION.DEFAULT_EDGE_LENGTH),
        ('B', 'C', SIMULATION.DEFAULT_SPEED_KPH, SIMULATION.DEFAULT_EDGE_LENGTH),
        ('C', 'A', SIMULATION.DEFAULT_SPEED_KPH, SIMULATION.DEFAULT_EDGE_LENGTH),
        ('A', 'C', 50, 500)  # Should preserve original valid values
    ]
    
    for i, (u, v, expected_speed, expected_length) in enumerate(expected_results):
        edge_data = G.get_edge_data(u, v, key=0)
        actual_speed = edge_data.get('speed_kph')
        actual_length = edge_data.get('length')
        
        if actual_speed != expected_speed:
            print(f"❌ FAIL: Edge {u}->{v} speed. Expected: {expected_speed}, Got: {actual_speed}")
            test_passed = False
        
        if actual_length != expected_length:
            print(f"❌ FAIL: Edge {u}->{v} length. Expected: {expected_length}, Got: {actual_length}")
            test_passed = False
    
    # Check that travel_time was calculated
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'travel_time' not in data:
            print(f"❌ FAIL: Edge {u}->{v} missing travel_time")
            test_passed = False
        elif data['travel_time'] <= 0:
            print(f"❌ FAIL: Edge {u}->{v} has invalid travel_time: {data['travel_time']}")
            test_passed = False
    
    if test_passed:
        print("✅ All tests passed! Default values are correctly applied.")
    else:
        print("❌ Some tests failed.")
    
    return test_passed

if __name__ == "__main__":
    success = test_default_values()
    sys.exit(0 if success else 1)
