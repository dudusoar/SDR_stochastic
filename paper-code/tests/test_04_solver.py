"""
Test script for ALNS solver
Tests initial solution generation and ALNS algorithm (short run with 1-2 segments)
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from real_map import RealMap
from demands import DemandGenerator
from order_info import OrderGenerator
from instance import PDPTWInstance
from solvers import greedy_insertion_init, ALNS

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def create_small_test_instance():
    """Create a small test instance for solver testing"""
    print("Creating small test instance for solver...")

    set_random_seed(42)

    # Create small RealMap
    n_r = 2  # 2 restaurants
    n_c = 3  # 3 customers
    dist_function = np.random.uniform
    dist_params = {'low': 0, 'high': 30}  # Smaller area
    real_map = RealMap(n_r, n_c, dist_function, dist_params)

    # Create DemandGenerator with minimal demand
    time_range = 20
    time_step = 10
    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 2, 'high': 4}  # 2-3 samples per interval
        },
        'demand_dist': {
            'function': np.random.poisson,
            'params': {'lam': 1.5}  # Low demand
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
        'time_window_length': 40,
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

    print(f"[OK] Small instance created: {instance.n} orders")
    print()

    return instance, order_gen


def test_initial_solution(instance):
    """Test greedy insertion initial solution"""
    print("=" * 60)
    print("Testing Initial Solution Generation...")
    print("=" * 60)

    # Parameters
    num_vehicles = 3
    vehicle_capacity = 10
    battery_capacity = 1000
    battery_consume_rate = 0.1
    penalty_unvisit = 1000
    penalty_delay = 10

    # Generate initial solution
    initial_solution = greedy_insertion_init(
        instance=instance,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        battery_capacity=battery_capacity,
        battery_consume_rate=battery_consume_rate,
        penalty_unvisit=penalty_unvisit,
        penalty_delay=penalty_delay
    )

    # Verify solution
    assert initial_solution is not None, "Initial solution is None"
    assert hasattr(initial_solution, 'routes'), "Missing routes"
    assert len(initial_solution.routes) > 0, "Empty routes"

    # Calculate objective
    obj_value = initial_solution.objective_function()
    is_feasible = initial_solution.is_feasible()

    print(f"[OK] Initial solution generated")
    print(f"  - Number of vehicles: {initial_solution.num_vehicles}")
    print(f"  - Number of routes: {len(initial_solution.routes)}")
    print(f"  - Objective value: {obj_value:.2f}")
    print(f"  - Is feasible: {is_feasible}")
    print()
    print("Routes:")
    for i, route in enumerate(initial_solution.routes):
        print(f"  Vehicle {i}: {route}")
    print()

    return initial_solution


def test_alns_solver_short(instance, initial_solution, order_gen):
    """Test ALNS solver with short run (1-2 segments)"""
    print("=" * 60)
    print("Testing ALNS Solver (Short Run: 2 segments)...")
    print("=" * 60)

    # Adjust num_removal based on instance size
    # num_removal should be at most the number of orders
    num_removal = min(2, max(1, instance.n // 2))

    # ALNS parameters - minimal for testing
    params_operators = {
        'num_removal': num_removal,  # Adjusted based on instance size
        'p': 6,
        'k': 3,
        'L_max': 5,
        'avg_remove_order': min(2, instance.n),
        'd_matrix': instance.distance_matrix
    }

    # ALNS control parameters - minimal for testing
    max_no_improve = 10  # Small for testing
    segment_length = 5   # Only 5 iterations per segment
    num_segments = 2     # Only 2 segments for testing
    r = 0.1
    sigma = [33, 9, 13]
    start_temp = 100
    cooling_rate = 0.95

    # Battery parameter
    battery = 50

    print(f"ALNS Parameters:")
    print(f"  - Segments: {num_segments}")
    print(f"  - Iterations per segment: {segment_length}")
    print(f"  - Total iterations: {num_segments * segment_length}")
    print(f"  - Number of removals: {params_operators['num_removal']}")
    print()

    # Create ALNS solver
    alns = ALNS(
        initial_solution=initial_solution,
        params_operators=params_operators,
        dist_matrix=instance.distance_matrix,
        battery=battery,
        max_no_improve=max_no_improve,
        segment_length=segment_length,
        num_segments=num_segments,
        r=r,
        sigma=sigma,
        start_temp=start_temp,
        cooling_rate=cooling_rate
    )

    # Verify ALNS initialization
    assert alns.current_solution is not None, "Current solution is None"
    assert alns.best_solution is not None, "Best solution is None"

    initial_obj = alns.current_solution.objective_function()
    print(f"Initial objective: {initial_obj:.2f}")
    print()
    print("Running ALNS (this will take a moment)...")
    print()

    # Run ALNS
    try:
        alns.run()
        print()
        print("[OK] ALNS completed successfully")
    except Exception as e:
        print(f"\n[ERROR] ALNS run failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    # Get final solution
    final_obj = alns.best_solution.objective_function()
    improvement = initial_obj - final_obj

    print()
    print("ALNS Results:")
    print(f"  - Initial objective: {initial_obj:.2f}")
    print(f"  - Final objective: {final_obj:.2f}")
    print(f"  - Improvement: {improvement:.2f} ({improvement/initial_obj*100:.1f}%)")
    print(f"  - Number of vehicles: {alns.best_solution.num_vehicles}")
    print()
    print("Best solution routes:")
    for i, route in enumerate(alns.best_solution.routes):
        if len(route) > 2:  # Only show non-empty routes
            print(f"  Vehicle {i}: {route}")
    print()

    return alns


def main():
    """Run solver tests"""
    print("\n" + "=" * 60)
    print("PAPER-CODE SOLVER TEST")
    print("=" * 60)
    print()

    try:
        # Create test instance
        instance, order_gen = create_small_test_instance()

        # Test initial solution
        initial_solution = test_initial_solution(instance)

        # Test ALNS solver (short run)
        alns = test_alns_solver_short(instance, initial_solution, order_gen)

        # Summary
        print("=" * 60)
        print("SOLVER TESTS PASSED [OK]")
        print("=" * 60)
        print("Summary:")
        print(f"  - Instance: {instance.n} orders")
        print(f"  - Initial solution generated successfully")
        print(f"  - ALNS completed {alns.num_segments} segments")
        print(f"  - Final objective: {alns.best_solution.objective_function():.2f}")
        print()
        print("Note: This was a short test run (2 segments, 5 iterations each).")
        print("Full optimization would require more segments and iterations.")
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
