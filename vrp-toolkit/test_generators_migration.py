"""Test module for migrated OrderGenerator and DemandGenerator."""

import sys
import os

# Add the vrp_toolkit package to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_import_order_generator():
    """Test that OrderGenerator can be imported."""
    from vrp_toolkit.data.generators import OrderGenerator
    assert OrderGenerator is not None


def test_import_demand_generator():
    """Test that DemandGenerator can be imported."""
    from vrp_toolkit.data.generators import DemandGenerator
    assert DemandGenerator is not None


def test_constants_available():
    """Test that required constants are available."""
    from vrp_toolkit.data.generators import (
        COL_ID, COL_TYPE, COL_X, COL_Y, COL_DEMAND,
        COL_START_TIME, COL_END_TIME, COL_SERVICE_TIME,
        COL_PARTNER_ID, COL_REAL_INDEX, COL_REAL_TYPE
    )
    
    # Check that constants have expected values
    assert COL_ID == 'ID'
    assert COL_TYPE == 'Type'
    assert COL_X == 'X'
    assert COL_Y == 'Y'
    assert COL_DEMAND == 'Demand'
    assert COL_START_TIME == 'StartTime'
    assert COL_END_TIME == 'EndTime'
    assert COL_SERVICE_TIME == 'ServiceTime'
    assert COL_PARTNER_ID == 'PartnerID'
    assert COL_REAL_INDEX == 'RealIndex'
    assert COL_REAL_TYPE == 'RealType'


def test_node_type_constants():
    """Test that node type constants are available."""
    from vrp_toolkit.data.generators import (
        NODE_TYPE_DEPOT,
        NODE_TYPE_PICKUP,
        NODE_TYPE_DELIVERY,
        NODE_TYPE_CHARGING,
        NODE_TYPE_DESTINATION
    )
    
    # Check that constants have expected values
    assert NODE_TYPE_DEPOT == 'depot'
    assert NODE_TYPE_PICKUP == 'cp'
    assert NODE_TYPE_DELIVERY == 'cd'
    assert NODE_TYPE_CHARGING == 'charging'
    assert NODE_TYPE_DESTINATION == 'destination'


def test_order_generator_class_attributes():
    """Test OrderGenerator class attributes."""
    from vrp_toolkit.data.generators import OrderGenerator
    
    # Check that DEFAULT_COLUMNS exists and has correct length
    assert hasattr(OrderGenerator, 'DEFAULT_COLUMNS')
    assert len(OrderGenerator.DEFAULT_COLUMNS) == 11
    
    # Check that node type constants are defined as class attributes
    assert hasattr(OrderGenerator, 'NODE_TYPE_PICKUP')
    assert hasattr(OrderGenerator, 'NODE_TYPE_DELIVERY')
    assert hasattr(OrderGenerator, 'NODE_TYPE_DEPOT')
    assert hasattr(OrderGenerator, 'NODE_TYPE_DESTINATION')
    assert hasattr(OrderGenerator, 'NODE_TYPE_CHARGING')


def test_demand_generator_class_attributes():
    """Test DemandGenerator class attributes."""
    from vrp_toolkit.data.generators import DemandGenerator
    
    # Check that required methods exist
    assert hasattr(DemandGenerator, '_generate_time_intervals')
    assert hasattr(DemandGenerator, '_generate_pairs')
    assert hasattr(DemandGenerator, '_generate_demand_table')
    assert hasattr(DemandGenerator, 'get_demand_table')
    assert hasattr(DemandGenerator, 'plot_demand_heatmap')


if __name__ == "__main__":
    # Run tests
    test_import_order_generator()
    print("✓ test_import_order_generator passed")
    
    test_import_demand_generator()
    print("✓ test_import_demand_generator passed")
    
    test_constants_available()
    print("✓ test_constants_available passed")
    
    test_node_type_constants()
    print("✓ test_node_type_constants passed")
    
    test_order_generator_class_attributes()
    print("✓ test_order_generator_class_attributes passed")
    
    test_demand_generator_class_attributes()
    print("✓ test_demand_generator_class_attributes passed")
    
    print("\nAll tests passed!")