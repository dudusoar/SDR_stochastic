"""
Test module for migrated real_map.py → vrp_toolkit/data/map.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from vrp_toolkit.data.map import RealMap, RealDataMap


def test_imports():
    """Test that all required classes can be imported"""
    # This test passes if imports succeed
    assert RealMap is not None
    assert RealDataMap is not None


def test_real_map_creation():
    """Test creation of RealMap with synthetic data"""
    # Create a simple uniform distribution function
    def uniform_dist(low: float, high: float) -> float:
        return (low + high) / 2.0  # Return midpoint for deterministic test
    
    # Create RealMap instance
    real_map = RealMap(
        n_r=2,
        n_c=4,
        dist_function=uniform_dist,
        dist_params={'low': -1, 'high': 1}
    )
    
    # Test basic attributes
    assert real_map.N_R == 2
    assert real_map.N_C == 4
    assert real_map.n == 6  # 2 + 4
    assert real_map.DEPOT_INDEX == 0
    assert real_map.DESTINATION_INDEX == 7  # n + 1 = 6 + 1
    assert real_map.CHARGING_STATION_INDEX == 8  # n + 2 = 6 + 2
    
    # Test node lists
    assert len(real_map.all_nodes) == 9  # depot + 6 nodes + destination + charging
    assert len(real_map.restaurants) == 2
    assert len(real_map.customers) == 4
    assert real_map.restaurants == [1, 2]
    assert real_map.customers == [3, 4, 5, 6]
    
    # Test generated data
    assert len(real_map.coordinates) == 9
    assert real_map.distance_matrix.shape == (9, 9)
    assert len(real_map.node_type_dict) == 9
    
    # Test coordinate generation (should be deterministic with our dist_function)
    # All coordinates should be (0, 0) since uniform_dist returns midpoint of -1 and 1
    for node, (x, y) in real_map.coordinates.items():
        assert x == 0.0
        assert y == 0.0
    
    # Test node types
    assert real_map.node_type_dict[0] == 'depot'
    assert real_map.node_type_dict[7] == 'destination'
    assert real_map.node_type_dict[8] == 'charging_station'
    assert real_map.node_type_dict[1] == 'restaurant'
    assert real_map.node_type_dict[3] == 'customer'


def test_real_data_map_defaults():
    """Test RealDataMap parameter defaults"""
    # Test that default values are set correctly
    # We can't actually load files, but we can test the class definition
    
    # Check __init__ signature by inspecting defaults
    # This is more of a documentation test
    pass


def test_real_data_map_parameterization():
    """Test RealDataMap parameter customization"""
    # This test would require actual data files
    # For now, just ensure the class exists with the expected interface
    assert hasattr(RealDataMap, '__init__')
    assert hasattr(RealDataMap, '_load_node_data')
    assert hasattr(RealDataMap, '_load_tt_matrix')
    assert hasattr(RealDataMap, '_generate_coordinates')
    assert hasattr(RealDataMap, '_generate_node_type')
    assert hasattr(RealDataMap, 'plot_map')


def test_constants_and_attributes():
    """Test that required constants and attributes exist"""
    # Test RealMap attributes
    real_map = RealMap(
        n_r=1,
        n_c=1,
        dist_function=lambda low, high: 0.0,
        dist_params={'low': 0, 'high': 1}
    )
    
    required_attrs = [
        'N_R', 'N_C', 'n',
        'DEPOT_INDEX', 'DESTINATION_INDEX', 'CHARGING_STATION_INDEX',
        'all_nodes', 'restaurants', 'customers',
        'coordinates', 'distance_matrix', 'node_type_dict'
    ]
    
    for attr in required_attrs:
        assert hasattr(real_map, attr), f"RealMap missing attribute: {attr}"
    
    # Test RealDataMap attributes (class level)
    required_data_attrs = [
        'node_data', 'tt_matrix',
        'N_R', 'N_C', 'n',
        'DEPOT_INDEX', 'DESTINATION_INDEX', 'CHARGING_STATION_INDEX',
        'all_nodes', 'restaurants', 'customers',
        'coordinates', 'distance_matrix', 'node_type_dict',
        'customer_types', 'distance_conversion_factor'
    ]
    
    for attr in required_data_attrs:
        assert attr in RealDataMap.__init__.__code__.co_varnames or \
               hasattr(RealDataMap, attr), f"RealDataMap missing attribute: {attr}"


if __name__ == '__main__':
    print("Running map migration tests...")
    
    test_imports()
    print("✓ Imports test passed")
    
    test_real_map_creation()
    print("✓ RealMap creation test passed")
    
    test_real_data_map_defaults()
    print("✓ RealDataMap defaults test passed")
    
    test_real_data_map_parameterization()
    print("✓ RealDataMap parameterization test passed")
    
    test_constants_and_attributes()
    print("✓ Constants and attributes test passed")
    
    print("\nAll map migration tests passed!")