"""Test migration of instance.py and solution.py to pdptw.py"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from vrp_toolkit.problems.pdptw import PDPTWInstance, PDPTWSolution

def test_instance_creation():
    """Test creation of PDPTWInstance with minimal data"""
    # Create a simple order table
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
    
    # Create simple distance and time matrices (5x5)
    n_nodes = 5
    distance_matrix = np.random.rand(n_nodes, n_nodes) * 10
    time_matrix = distance_matrix / 2.0  # Assume speed of 2 units per minute
    robot_speed = 2.0
    
    # Create instance
    instance = PDPTWInstance(
        order_table=order_table,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        robot_speed=robot_speed
    )
    
    # Test basic attributes
    assert instance.n == 2  # 2 orders (pickup-delivery pairs)
    assert len(instance.indices) == 5
    assert len(instance.demands) == 5
    assert len(instance.time_windows) == 5
    assert len(instance.service_times) == 5
    assert instance.distance_matrix.shape == (5, 5)
    assert instance.time_matrix.shape == (5, 5)
    assert instance.robot_speed == robot_speed
    
    print("✓ PDPTWInstance creation test passed")
    return instance

def test_solution_creation():
    """Test creation of PDPTWSolution with simple routes"""
    instance = test_instance_creation()
    
    # Create simple routes: one vehicle serving both orders
    # Route: depot -> pickup1 -> pickup2 -> delivery1 -> delivery2 -> depot
    routes = [[0, 1, 2, 3, 4, 0]]
    
    # Create solution
    solution = PDPTWSolution(
        instance=instance,
        vehicle_capacity=10.0,
        battery_capacity=100.0,
        battery_consume_rate=1.0,
        routes=routes,
        penalty_unvisit=1000.0,
        penalty_delay=100.0
    )
    
    # Test basic attributes
    assert solution.num_vehicles == 1
    assert len(solution.routes) == 1
    assert solution.routes[0] == [0, 1, 2, 3, 4, 0]
    assert solution.instance is instance
    
    # Test calculated attributes
    assert hasattr(solution, 'route_battery_levels')
    assert hasattr(solution, 'route_capacity_levels')
    assert hasattr(solution, 'route_arrival_times')
    assert hasattr(solution, 'route_leave_times')
    assert hasattr(solution, 'route_wait_times')
    
    # Test objective function can be calculated
    obj_value = solution.objective_function()
    assert isinstance(obj_value, float)
    
    # Test feasibility check
    is_feasible = solution.is_feasible()
    assert isinstance(is_feasible, bool)
    
    print("✓ PDPTWSolution creation test passed")
    return solution

def test_solution_methods():
    """Test solution methods"""
    solution = test_solution_creation()
    
    # Test get_selected_vehicles
    selected = solution.get_selected_vehicles()
    assert isinstance(selected, list)
    assert len(selected) == 1
    assert selected[0] == 0
    
    # Test check_capacity_constraint
    capacity_ok = solution.check_capacity_constraint(selected)
    assert isinstance(capacity_ok, bool)
    
    # Test check_battery_constraint
    battery_ok = solution.check_battery_constraint(selected)
    assert isinstance(battery_ok, bool)
    
    # Test check_pickup_delivery_order
    order_ok = solution.check_pickup_delivery_order(selected)
    assert isinstance(order_ok, bool)
    
    # Test visited/unvisited records
    assert hasattr(solution, 'visited_requests')
    assert hasattr(solution, 'unvisited_requests')
    assert hasattr(solution, 'visited_pairs')
    assert hasattr(solution, 'unvisited_pairs')
    
    print("✓ PDPTWSolution methods test passed")

if __name__ == "__main__":
    print("Testing migration of instance.py and solution.py to pdptw.py")
    print("=" * 60)
    
    try:
        instance = test_instance_creation()
        solution = test_solution_creation()
        test_solution_methods()
        
        print("\n" + "=" * 60)
        print("All tests passed! Migration successful.")
        print(f"Instance: {instance}")
        print(f"Solution objective: {solution.objective_function()}")
        print(f"Solution feasible: {solution.is_feasible()}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)