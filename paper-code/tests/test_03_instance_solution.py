"""
Test script for PDPTWInstance and PDPTWSolution
Tests the problem instance and solution representation
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from real_map import RealMap
from demands import DemandGenerator
from order_info import OrderGenerator
from instance import PDPTWInstance
from solution import PDPTWSolution

def create_test_instance():
    """Create a test PDPTW instance"""
    print("Creating test PDPTW instance...")

    # Create RealMap
    n_r = 2
    n_c = 3
    dist_function = np.random.uniform
    dist_params = {'low': 0, 'high': 50}
    real_map = RealMap(n_r, n_c, dist_function, dist_params)

    # Create DemandGenerator with smaller demand
    time_range = 20
    time_step = 10
    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 1, 'high': 3}  # 1-2 samples per interval
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

    # Create OrderGenerator
    time_params = {
        'time_window_length': 30,
        'service_time': 2,
        'extra_time': 5,
        'big_time': 10000
    }
    robot_speed = 0.5

    order_gen = OrderGenerator(
        real_map=real_map,
        demand_table=demand_gen.demand_table,
        time_params=time_params,
        robot_speed=robot_speed
    )

    # Create PDPTWInstance
    instance = PDPTWInstance(order_gen)

    print(f"[OK] Test instance created")
    print(f"  - Number of orders: {instance.n}")
    print()

    return instance


def test_pdptw_instance(instance):
    """Test PDPTWInstance"""
    print("=" * 60)
    print("Testing PDPTWInstance...")
    print("=" * 60)

    # Verify basic properties
    assert hasattr(instance, 'n'), "Missing 'n' attribute"
    assert hasattr(instance, 'demands'), "Missing 'demands' attribute"
    assert hasattr(instance, 'time_windows'), "Missing 'time_windows' attribute"
    assert hasattr(instance, 'service_times'), "Missing 'service_times' attribute"
    assert hasattr(instance, 'distance_matrix'), "Missing 'distance_matrix' attribute"
    assert hasattr(instance, 'time_matrix'), "Missing 'time_matrix' attribute"

    # Verify data structures
    assert len(instance.demands) > 0, "Empty demands"
    assert len(instance.time_windows) > 0, "Empty time_windows"
    assert len(instance.service_times) > 0, "Empty service_times"

    # Verify matrices
    assert instance.distance_matrix.shape[0] == instance.distance_matrix.shape[1], "Distance matrix not square"
    assert instance.time_matrix.shape[0] == instance.time_matrix.shape[1], "Time matrix not square"

    print(f"[OK] PDPTWInstance verified")
    print(f"  - Number of orders (n): {instance.n}")
    print(f"  - Distance matrix shape: {instance.distance_matrix.shape}")
    print(f"  - Time matrix shape: {instance.time_matrix.shape}")
    print(f"  - Demands length: {len(instance.demands)}")
    print(f"  - Time windows length: {len(instance.time_windows)}")
    print()


def test_pdptw_solution(instance):
    """Test PDPTWSolution"""
    print("=" * 60)
    print("Testing PDPTWSolution...")
    print("=" * 60)

    # Create a simple solution with one vehicle
    # Route: depot -> pickup 1 -> delivery 1 -> depot
    routes = [[0, 1, instance.n + 1, 0]]

    # Solution parameters
    vehicle_capacity = 10
    battery_capacity = 1000
    battery_consume_rate = 0.1
    penalty_unvisit = 1000
    penalty_delay = 10

    solution = PDPTWSolution(
        instance=instance,
        vehicle_capacity=vehicle_capacity,
        battery_capacity=battery_capacity,
        battery_consume_rate=battery_consume_rate,
        routes=routes,
        penalty_unvisit=penalty_unvisit,
        penalty_delay=penalty_delay
    )

    # Verify solution attributes
    assert hasattr(solution, 'routes'), "Missing 'routes' attribute"
    assert hasattr(solution, 'num_vehicles'), "Missing 'num_vehicles' attribute"
    assert hasattr(solution, 'instance'), "Missing 'instance' attribute"

    # Test solution methods
    assert hasattr(solution, 'objective_function'), "Missing 'objective_function' method"
    assert hasattr(solution, 'is_feasible'), "Missing 'is_feasible' method"

    # Calculate objective
    obj_value = solution.objective_function()
    print(f"[OK] PDPTWSolution created")
    print(f"  - Number of vehicles: {solution.num_vehicles}")
    print(f"  - Routes: {routes}")
    print(f"  - Objective value: {obj_value:.2f}")

    # Check feasibility
    is_feasible = solution.is_feasible()
    print(f"  - Is feasible: {is_feasible}")
    print()

    return solution


def test_multi_vehicle_solution(instance):
    """Test solution with multiple vehicles"""
    print("=" * 60)
    print("Testing Multi-Vehicle Solution...")
    print("=" * 60)

    # Create solution with 2 vehicles
    # Each vehicle handles one order
    if instance.n >= 2:
        routes = [
            [0, 1, instance.n + 1, 0],
            [0, 2, instance.n + 2, 0]
        ]
    else:
        routes = [[0, 1, instance.n + 1, 0]]

    vehicle_capacity = 10
    battery_capacity = 1000
    battery_consume_rate = 0.1
    penalty_unvisit = 1000
    penalty_delay = 10

    solution = PDPTWSolution(
        instance=instance,
        vehicle_capacity=vehicle_capacity,
        battery_capacity=battery_capacity,
        battery_consume_rate=battery_consume_rate,
        routes=routes,
        penalty_unvisit=penalty_unvisit,
        penalty_delay=penalty_delay
    )

    obj_value = solution.objective_function()
    is_feasible = solution.is_feasible()

    print(f"[OK] Multi-vehicle solution created")
    print(f"  - Number of vehicles: {solution.num_vehicles}")
    print(f"  - Objective value: {obj_value:.2f}")
    print(f"  - Is feasible: {is_feasible}")
    print()


def main():
    """Run instance and solution tests"""
    print("\n" + "=" * 60)
    print("PAPER-CODE INSTANCE & SOLUTION TEST")
    print("=" * 60)
    print()

    try:
        # Create test instance
        instance = create_test_instance()

        # Test PDPTWInstance
        test_pdptw_instance(instance)

        # Test PDPTWSolution
        solution = test_pdptw_solution(instance)

        # Test multi-vehicle solution
        test_multi_vehicle_solution(instance)

        # Summary
        print("=" * 60)
        print("INSTANCE & SOLUTION TESTS PASSED [OK]")
        print("=" * 60)
        print("Summary:")
        print(f"  - Instance with {instance.n} orders created")
        print(f"  - Solution objectives computed successfully")
        print(f"  - Feasibility checks working")
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
