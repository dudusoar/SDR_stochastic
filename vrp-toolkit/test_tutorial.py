#!/usr/bin/env python3
"""Test the quickstart tutorial execution."""

import sys
import numpy as np

def test_tutorial_steps():
    """Test the main steps from the quickstart tutorial."""
    print("Testing quickstart tutorial steps...")

    try:
        # Step 1: Import modules
        from vrp_toolkit.data.map import RealMap
        from vrp_toolkit.data.generators import DemandGenerator, OrderGenerator
        from vrp_toolkit.problems.pdptw import PDPTWInstance
        from vrp_toolkit.algorithms.alns.solver import ALNS, ALNSConfig, greedy_insertion_initial_solution

        print("[OK] All imports successful")

        # Step 2: Create synthetic map
        import random
        realMap = RealMap(
            n_r=2,
            n_c=10,
            dist_function=random.uniform,
            dist_params={'a': 0.0, 'b': 100.0}
        )
        print(f"[OK] Created RealMap with {len(realMap.restaurants)} restaurants and {len(realMap.customers)} customers")

        # Step 3: Generate demand data
        random_params = {
            'sample_dist': {
                'function': np.random.randint,
                'params': {'low': 7, 'high': 9}
            },
            'demand_dist': {
                'function': np.random.poisson,
                'params': {'lam': 2.0}
            }
        }

        demands = DemandGenerator(
            time_range=30,
            time_step=10,
            restaurants=realMap.restaurants,
            customers=realMap.customers,
            random_params=random_params
        )
        print("[OK] Created DemandGenerator")

        # Step 4: Generate orders
        time_params = {
            'time_window_length': 30,
            'service_time': 5,
            'extra_time': 10,
            'big_time': 1000
        }

        order_generator = OrderGenerator(
            realMap,
            demands.demand_table,
            time_params,
            4.0
        )
        print("[OK] Created OrderGenerator")

        # Step 5: Create PDPTW instance
        instance = PDPTWInstance(
            order_table=order_generator.order_table,
            distance_matrix=realMap.distance_matrix,
            time_matrix=realMap.distance_matrix,  # Using distance as time for simplicity
            battery_capacity=100,
            vehicle_capacity=50,
            depot_index=realMap.depot_index
        )
        print(f"[OK] Created PDPTWInstance with {instance.num_nodes} nodes")

        # Step 6: Create initial solution
        initial_solution = greedy_insertion_initial_solution(instance)
        print(f"[OK] Created initial solution with {len(initial_solution.routes)} routes")

        # Step 7: Configure ALNS
        config = ALNSConfig(
            num_removal=5,
            p=4.0,
            removal_weights=[0.25, 0.25, 0.25, 0.25],
            insertion_weights=[0.5, 0.5],
            segment_length=100,
            num_segments=5,
            r=0.1,
            sigma=0.1,
            start_temp=100.0,
            cooling_rate=0.95,
            max_no_improve=50
        )
        print("[OK] Created ALNSConfig")

        # Step 8: Initialize ALNS solver
        alns = ALNS(
            initial_solution=initial_solution,
            config=config,
            dist_matrix=realMap.distance_matrix,
            battery_capacity=100
        )
        print("[OK] Created ALNS solver")

        # Step 9: Run a few iterations (not full solve to save time)
        print("Running 5 ALNS iterations for test...")
        for i in range(5):
            alns.iteration()
            if i % 1 == 0:
                print(f"  Iteration {i+1}: current cost = {alns.current_solution.total_cost:.2f}")

        print("\n✅ All tutorial steps completed successfully!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Tutorial test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tutorial_steps()
    sys.exit(0 if success else 1)