"""
Test script for OrderGenerator
Tests the order generation from map and demand data
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from real_map import RealMap
from demands import DemandGenerator
from order_info import OrderGenerator

def create_test_data():
    """Create test data for order generation"""
    print("Creating test data...")

    # Create RealMap
    n_r = 2
    n_c = 4
    dist_function = np.random.uniform
    dist_params = {'low': 0, 'high': 100}
    real_map = RealMap(n_r, n_c, dist_function, dist_params)

    # Create DemandGenerator
    time_range = 30
    time_step = 10
    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 2, 'high': 4}
        },
        'demand_dist': {
            'function': np.random.poisson,
            'params': {'lam': 1}
        }
    }
    demand_gen = DemandGenerator(
        time_range=time_range,
        time_step=time_step,
        restaurants=real_map.restaurants,
        customers=real_map.customers,
        random_params=random_params
    )

    print(f"[OK] Test data created: {real_map.N_R} restaurants, {real_map.N_C} customers")
    print()

    return real_map, demand_gen


def test_order_generator(real_map, demand_gen):
    """Test OrderGenerator"""
    print("=" * 60)
    print("Testing OrderGenerator...")
    print("=" * 60)

    # Time parameters
    time_params = {
        'time_window_length': 30,  # 30 minutes window
        'service_time': 2,          # 2 minutes service time
        'extra_time': 5,            # 5 minutes extra time
        'big_time': 10000           # Large number to replace 'inf'
    }

    # Robot speed
    robot_speed = 0.5  # distance units per minute

    # Create OrderGenerator
    order_gen = OrderGenerator(
        real_map=real_map,
        demand_table=demand_gen.demand_table,
        time_params=time_params,
        robot_speed=robot_speed
    )

    # Verify basic properties
    assert hasattr(order_gen, 'order_table'), "Missing order_table"
    assert hasattr(order_gen, 'distance_matrix'), "Missing distance_matrix"
    assert hasattr(order_gen, 'time_matrix'), "Missing time_matrix"
    assert hasattr(order_gen, 'total_number_orders'), "Missing total_number_orders"

    # Verify order table structure
    order_table = order_gen.order_table
    required_columns = ['ID', 'Type', 'Demand', 'StartTime', 'EndTime', 'ServiceTime', 'PartnerID']
    for col in required_columns:
        assert col in order_table.columns, f"Missing column: {col}"

    # Verify total orders
    total_demand = demand_gen.demand_table.iloc[:, 2:].sum().sum()
    assert order_gen.total_number_orders == total_demand, "Total orders mismatch"

    # Verify time matrix
    expected_time_matrix_shape = real_map.distance_matrix.shape
    assert order_gen.time_matrix.shape == expected_time_matrix_shape, "Time matrix shape mismatch"

    print(f"[OK] OrderGenerator created successfully")
    print(f"  - Total orders: {order_gen.total_number_orders}")
    print(f"  - Robot speed: {robot_speed} units/min")
    print(f"  - Time window length: {time_params['time_window_length']} min")
    print(f"  - Service time: {time_params['service_time']} min")
    print()
    print("Order table preview:")
    print(order_table.head(10))
    print()
    print(f"Order table shape: {order_table.shape}")
    print(f"Time matrix shape: {order_gen.time_matrix.shape}")
    print()

    return order_gen


def main():
    """Run order generation test"""
    print("\n" + "=" * 60)
    print("PAPER-CODE ORDER GENERATION TEST")
    print("=" * 60)
    print()

    try:
        # Create test data
        real_map, demand_gen = create_test_data()

        # Test OrderGenerator
        order_gen = test_order_generator(real_map, demand_gen)

        # Summary
        print("=" * 60)
        print("ORDER GENERATION TEST PASSED [OK]")
        print("=" * 60)
        print("Summary:")
        print(f"  - Total orders generated: {order_gen.total_number_orders}")
        print(f"  - Order table shape: {order_gen.order_table.shape}")
        print(f"  - Distance matrix: {order_gen.distance_matrix.shape}")
        print(f"  - Time matrix: {order_gen.time_matrix.shape}")
        print()

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED [ERROR]")
        print("=" * 60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
