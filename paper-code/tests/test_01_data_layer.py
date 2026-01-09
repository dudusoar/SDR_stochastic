"""
Test script for data layer: RealMap and DemandGenerator
Tests the basic infrastructure for generating map and demand data
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from real_map import RealMap
from demands import DemandGenerator

def test_real_map():
    """Test RealMap generation"""
    print("=" * 60)
    print("Testing RealMap...")
    print("=" * 60)

    # Create a small map with 2 restaurants and 4 customers
    n_r = 2
    n_c = 4
    dist_function = np.random.uniform
    dist_params = {'low': 0, 'high': 100}

    real_map = RealMap(n_r, n_c, dist_function, dist_params)

    # Verify basic properties
    assert real_map.N_R == n_r, "Number of restaurants mismatch"
    assert real_map.N_C == n_c, "Number of customers mismatch"
    assert real_map.n == n_r + n_c, "Total nodes mismatch"

    # Verify coordinates
    assert len(real_map.coordinates) == real_map.n + 3, "Coordinates count mismatch"
    assert real_map.DEPOT_INDEX == 0, "Depot index should be 0"

    # Verify distance matrix shape
    expected_size = real_map.n + 3
    assert real_map.distance_matrix.shape == (expected_size, expected_size), "Distance matrix shape mismatch"

    # Verify node types
    assert len(real_map.node_type_dict) == real_map.n + 3, "Node type dict size mismatch"

    print(f"[OK] RealMap created successfully")
    print(f"  - Restaurants: {n_r}")
    print(f"  - Customers: {n_c}")
    print(f"  - Total nodes: {real_map.n}")
    print(f"  - Distance matrix shape: {real_map.distance_matrix.shape}")
    print()

    return real_map


def test_demand_generator(real_map):
    """Test DemandGenerator"""
    print("=" * 60)
    print("Testing DemandGenerator...")
    print("=" * 60)

    # Time parameters
    time_range = 30  # 30 minutes
    time_step = 10   # 10-minute intervals

    # Random parameters for demand generation
    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 2, 'high': 5}  # 2-4 samples per interval
        },
        'demand_dist': {
            'function': np.random.poisson,
            'params': {'lam': 1}  # Average 1 unit per order
        }
    }

    demand_gen = DemandGenerator(
        time_range=time_range,
        time_step=time_step,
        restaurants=real_map.restaurants,
        customers=real_map.customers,
        random_params=random_params
    )

    demand_table = demand_gen.get_demand_table()

    # Verify demand table structure
    assert 'Pickup' in demand_table.columns, "Missing 'Pickup' column"
    assert 'Delivery' in demand_table.columns, "Missing 'Delivery' column"

    # Expected number of restaurant-customer pairs
    expected_pairs = len(real_map.restaurants) * len(real_map.customers)
    assert len(demand_table) == expected_pairs, f"Expected {expected_pairs} pairs, got {len(demand_table)}"

    # Expected time intervals
    expected_intervals = len(demand_gen.time_intervals)
    time_columns = [col for col in demand_table.columns if col not in ['Pickup', 'Delivery']]
    assert len(time_columns) == expected_intervals, "Time intervals count mismatch"

    print(f"[OK] DemandGenerator created successfully")
    print(f"  - Time range: {time_range} minutes")
    print(f"  - Time step: {time_step} minutes")
    print(f"  - Number of pairs: {expected_pairs}")
    print(f"  - Time intervals: {expected_intervals}")
    print(f"  - Total demand: {demand_table.iloc[:, 2:].sum().sum()}")
    print()
    print("Demand table preview:")
    print(demand_table.head())
    print()

    return demand_gen


def main():
    """Run all data layer tests"""
    print("\n" + "=" * 60)
    print("PAPER-CODE DATA LAYER TEST")
    print("=" * 60)
    print()

    try:
        # Test RealMap
        real_map = test_real_map()

        # Test DemandGenerator
        demand_gen = test_demand_generator(real_map)

        # Summary
        print("=" * 60)
        print("ALL DATA LAYER TESTS PASSED [OK]")
        print("=" * 60)
        print("Summary:")
        print(f"  - RealMap: {real_map.N_R} restaurants, {real_map.N_C} customers")
        print(f"  - DemandGenerator: {len(demand_gen.time_intervals)} time intervals")
        print(f"  - Total demand: {demand_gen.demand_table.iloc[:, 2:].sum().sum()} orders")
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
