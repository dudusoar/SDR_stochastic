"""Test migration of solvers.py to vrp_toolkit/algorithms/alns/solver.py"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from vrp_toolkit.problems.pdptw import PDPTWInstance, PDPTWSolution
from vrp_toolkit.algorithms.alns import (
    ALNS,
    ALNSConfig,
    greedy_insertion_initial_solution,
    RemovalOperators,
    RepairOperators,
)


def create_test_instance() -> PDPTWInstance:
    """Create a minimal test instance."""
    data = {
        'ID': [0, 1, 2, 3, 4],
        'Type': ['depot', 'cp', 'cp', 'cd', 'cd'],
        'X': [0.0, 1.0, 2.0, 1.0, 2.0],
        'Y': [0.0, 1.0, 2.0, 1.0, 2.0],
        'Demand': [0.0, 1.0, 1.0, -1.0, -1.0],
        'StartTime': [0.0, 0.0, 0.0, 0.0, 0.0],
        'EndTime': [100.0, 100.0, 100.0, 100.0, 100.0],
        'ServiceTime': [0.0, 5.0, 5.0, 5.0, 5.0],
        'PartnerID': [0, 3, 4, 1, 2],
        'RealIndex': [0, 1, 2, 3, 4],
        'RealType': ['depot', 'cp', 'cp', 'cd', 'cd']
    }
    
    order_table = pd.DataFrame(data)
    n_nodes = 5
    distance_matrix = np.random.rand(n_nodes, n_nodes) * 10
    time_matrix = distance_matrix / 2.0
    robot_speed = 2.0
    
    instance = PDPTWInstance(
        order_table=order_table,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        robot_speed=robot_speed
    )
    return instance


def test_imports():
    """Test that all required classes can be imported."""
    print("Testing imports...")
    
    # Test that classes exist
    assert ALNS is not None
    assert ALNSConfig is not None
    assert greedy_insertion_initial_solution is not None
    assert RemovalOperators is not None
    assert RepairOperators is not None
    
    print("✓ All imports successful")


def test_config_creation():
    """Test ALNSConfig creation."""
    print("Testing ALNSConfig creation...")
    
    config = ALNSConfig()
    
    # Check default values
    assert config.num_removal == 5
    assert config.p == 4.0
    assert config.k == 3
    assert config.L_max == 5
    assert config.avg_remove_order == 2.0
    assert config.d_matrix is None
    assert config.max_no_improve == 100
    assert config.segment_length == 100
    assert config.num_segments == 10
    assert config.r == 0.1
    assert config.sigma == (33.0, 9.0, 13.0)
    assert config.start_temp == 10000.0
    assert config.cooling_rate == 0.99
    assert config.cost_ci_obj_diff_threshold == 0.1
    assert config.cost_ci_window_size == 25
    assert config.removal_indices == [0, 2, 3]
    assert config.repair_indices == [0, 1]
    assert config.charging_station_index is None
    
    print("✓ ALNSConfig creation test passed")
    return config


def test_greedy_insertion_initial_solution():
    """Test greedy insertion initial solution creation."""
    print("Testing greedy_insertion_initial_solution...")
    
    instance = create_test_instance()
    
    solution = greedy_insertion_initial_solution(
        instance=instance,
        num_vehicles=2,
        vehicle_capacity=10.0,
        battery_capacity=100.0,
        battery_consume_rate=1.0,
        penalty_unvisit=1000.0,
        penalty_delay=100.0
    )
    
    assert isinstance(solution, PDPTWSolution)
    assert len(solution.routes) == 2
    assert solution.num_vehicles == 2
    
    print("✓ greedy_insertion_initial_solution test passed")
    return solution


def test_alns_initialization():
    """Test ALNS class initialization."""
    print("Testing ALNS initialization...")
    
    instance = create_test_instance()
    config = ALNSConfig()
    
    # Create initial solution using greedy insertion
    initial_solution = greedy_insertion_initial_solution(
        instance=instance,
        num_vehicles=2,
        vehicle_capacity=10.0,
        battery_capacity=100.0,
        battery_consume_rate=1.0,
        penalty_unvisit=1000.0,
        penalty_delay=100.0
    )
    
    # Create distance matrix for charging insertion (simple 5x5)
    dist_matrix = np.random.rand(5, 5) * 10
    
    # Initialize ALNS
    alns = ALNS(
        initial_solution=initial_solution,
        config=config,
        dist_matrix=dist_matrix,
        battery_capacity=100.0
    )
    
    # Check attributes
    assert alns.current_solution is not None
    assert alns.best_solution is not None
    assert alns.charging_solution is not None
    assert alns.num_removal == config.num_removal
    assert alns.max_no_improve == config.max_no_improve
    assert alns.segment_length == config.segment_length
    assert alns.num_segments == config.num_segments
    
    print("✓ ALNS initialization test passed")
    return alns


def test_operators():
    """Test RemovalOperators and RepairOperators."""
    print("Testing operators...")
    
    instance = create_test_instance()
    
    # Create a simple solution
    routes = [[0, 1, 3, 0], [0, 2, 4, 0]]
    solution = PDPTWSolution(
        instance=instance,
        vehicle_capacity=10.0,
        battery_capacity=100.0,
        battery_consume_rate=1.0,
        routes=routes,
        penalty_unvisit=1000.0,
        penalty_delay=100.0
    )
    
    # Test RemovalOperators
    removal_ops = RemovalOperators(solution)
    assert removal_ops.solution is solution
    assert removal_ops.instance is instance
    
    # Test RepairOperators
    repair_ops = RepairOperators(solution)
    assert repair_ops.solution is not None
    assert repair_ops.instance is instance
    
    print("✓ Operators test passed")


def main():
    """Run all tests."""
    print("Testing migration of solvers.py to ALNS package")
    print("=" * 60)
    
    try:
        test_imports()
        config = test_config_creation()
        solution = test_greedy_insertion_initial_solution()
        alns = test_alns_initialization()
        test_operators()
        
        print("\n" + "=" * 60)
        print("All tests passed! Migration successful.")
        print(f"ALNSConfig: {config}")
        print(f"Initial solution objective: {solution.objective_function()}")
        print(f"ALNS instance created: {alns}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()