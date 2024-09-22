import time
import pandas as pd
import numpy as np
import random
from datetime import datetime
from real_map import RealDataMap
from demands import DemandGenerator
from order_info import OrderGenerator
from instance import PDPTWInstance
from solvers import greedy_insertion_init, ALNS

def run_sensitivity_analysis(variable_to_change, values_to_test, fixed_params, num_scenarios):
    results = []
    start_time = time.time()

    for value in values_to_test:
        print(f"\nTesting {variable_to_change} = {value}")
        
        for scenario in range(num_scenarios):
            scenario_start_time = time.time()
            print(f"\nScenario {scenario + 1}/{num_scenarios}")
            
            # Set random seed
            seed_value = fixed_params['base_seed'] + scenario
            np.random.seed(seed_value)
            random.seed(seed_value)

            # Update the variable we're testing
            current_params = fixed_params.copy()
            current_params[variable_to_change] = value

            # Run the simulation
            scenario_results = run_single_scenario(current_params)
            
            # Add the current variable and its value to the results
            scenario_results[variable_to_change] = value
            scenario_results['scenario'] = scenario + 1
            results.append(scenario_results)

            scenario_end_time = time.time()
            print(f"Scenario completed in {scenario_end_time - scenario_start_time:.2f} seconds")

    end_time = time.time()
    print(f"\nSensitivity analysis completed in {end_time - start_time:.2f} seconds")

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sensitivity_analysis_{variable_to_change}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def run_single_scenario(params):
    # Extract parameters
    average_order = params['average_order']
    num_vehicles = params['num_vehicles']
    battery = params['battery']
    
    # Real data map
    real_data_map = RealDataMap(params['node_info_file'], params['tt_matrix'])

    # Demands
    n_r, n_c = real_data_map.N_R, real_data_map.N_C
    n_pairs = n_r * n_c
    n_time_intervals = params['time_range'] / params['time_step']
    lam_poisson = average_order / (n_pairs * n_time_intervals)

    random_params = {
        'sample_dist': {'function': np.random.randint, 'params': {'low': n_pairs-0.5, 'high': n_pairs}},
        'demand_dist': {'function': np.random.poisson, 'params': {'lam': lam_poisson}}
    }

    demands = DemandGenerator(params['time_range'], params['time_step'], 
                              real_data_map.restaurants, real_data_map.customers, 
                              random_params=random_params)

    # Order
    pdptw_order = OrderGenerator(real_data_map, demands.demand_table, params['time_params'], params['robot_speed'])

    # Instance
    pdptw_instance = PDPTWInstance(pdptw_order)

    # Solution
    dist_matrix = pdptw_instance.distance_matrix
    robot_speed = pdptw_instance.robot_speed
    battery_capacity = battery_relaxation(battery, dist_matrix, robot_speed, params['if_battery_relaxation'])

    initial_solution = greedy_insertion_init(pdptw_instance, num_vehicles, params['vehicle_capacity'],
                                             battery_capacity, params['battery_consume_rate'],
                                             params['penalty_unvisited'], params['penalty_delayed'])

    print(f"Initial solution objective value: {initial_solution.objective_function():.2f}")

    d_matrix = generate_d_matrix(pdptw_instance)
    params['params_operators']['d_matrix'] = d_matrix
    
    alns = ALNS(
        initial_solution=initial_solution,
        params_operators=params['params_operators'],
        dist_matrix=dist_matrix,
        battery=battery,
        max_no_improve=params['max_no_improve'],
        segment_length=params['segment_length'],
        num_segments=params['num_segments'],
        r=params['r'],
        sigma=params['sigma'],
        start_temp=params['start_temp'],
        cooling_rate=params['cooling_rate']
    )

    best_solution, best_charging_solution = alns.run()

    # Collect results
    results = {
        'order_count': best_solution.instance.n,
        'distance_uncharge': np.sum(best_solution.total_travel_times) / 60 * params['robot_speed'],
        'distance_charge': np.sum(best_charging_solution.total_travel_times) / 60 * params['robot_speed'],
        'max_dist': max(best_solution.total_travel_times) / 60 * params['robot_speed'],
        'obj_uncharge': best_solution.objective_function(),
        'obj_charge': best_charging_solution.objective_function(),
        'num_veh': len([sublist for sublist in best_charging_solution.routes if sublist != [0,0]]),
        'battery_swapping': sum(2*(best_charging_solution.instance.n)+1 in route for route in best_charging_solution.routes)
    }

    return results

# Helper functions
def battery_relaxation(battery, dist_matrix, robot_speed, indicator):
    if indicator:
        battery_capacity = (battery - np.mean(dist_matrix[0][1:-1]))*2/robot_speed*60
    else:
        battery_capacity = battery/robot_speed*60
    return battery_capacity

def generate_d_matrix(instance):
    n = instance.n
    robot_speed = instance.robot_speed
    dist_matrix = instance.distance_matrix
    start_time = np.array([instance.time_windows[i][0] for i in range(1, n+1)])
    end_time = np.array([instance.time_windows[i][1] for i in range(1, n+1)])
        
    d_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # pickup-pickup
            d_1 = dist_matrix[i+1][j+1]
            t_1 = abs(start_time[i] - start_time[j]) / 60 * robot_speed 
            dt_1 = d_1 + t_1 * 0.3

            # pickup-dropoff
            d_2 = dist_matrix[i+1][j+n+1]
            t_2 = abs(start_time[i] - end_time[j]) / 60 * robot_speed 
            dt_2 = d_2 + t_2 * 0.3

            # dropoff-pickup
            d_3 = dist_matrix[i+n+1][j+1]
            t_3 = abs(start_time[j] - end_time[i]) / 60 * robot_speed 
            dt_3 = d_3 + t_3 * 0.3

            # dropoff-dropoff
            d_4 = dist_matrix[i+n+1][j+n+1]
            t_4 = abs(start_time[j] - end_time[i]) / 60 * robot_speed 
            dt_4 = d_4 + t_4 * 0.3

            d_matrix[i][j] = min(dt_1, dt_2, dt_3, dt_4)
    
    return d_matrix
                
# Main execution
if __name__ == "__main__":
    # Fixed parameters
    fixed_params = {
        'base_seed': 42,
        'node_info_file': 'data/purdue_node_info.csv',
        'tt_matrix': 'data/tt_matrix.csv',
        'time_range': 120,
        'time_step': 8,
        'time_params': {'time_window_length': 60, 'service_time': 2, 'extra_time': 0, 'big_time': 1000},
        'robot_speed': 4,
        'vehicle_capacity': 6,
        'battery_consume_rate': 1,
        'penalty_unvisited': 100,
        'penalty_delayed': 5,
        'if_battery_relaxation': 1,
        'params_operators': {
            'num_removal': 3,  # Assuming 30% of average orders (60)
            'p': 3,
            'k': 3,
            'L_max': 6,
            'avg_remove_order': 6
        },
        'max_no_improve': 25,
        'segment_length': 10,
        'num_segments': 15,
        'r': 0.2,
        'sigma': [10, 5, 1],
        'start_temp': 100,
        'cooling_rate': 0.99,
        
        # Default values for the variables we'll be changing
        'average_order': 40,
        'num_vehicles': 7,
        'battery': 8
    }

    # # Sensitivity analysis for battery capacity
    # run_sensitivity_analysis('battery', [6, 7, 8, 9, 10], fixed_params, num_scenarios=5)
    #
    # # Sensitivity analysis for number of vehicles
    # run_sensitivity_analysis('num_vehicles', [7, 8, 9, 10, 11], fixed_params, num_scenarios=5)

    # Sensitivity analysis for average order
    run_sensitivity_analysis('average_order', [40], fixed_params, num_scenarios=5)



