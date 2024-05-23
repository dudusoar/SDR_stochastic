import numpy as np
import matplotlib.pyplot as plt

def generate_time_intervals(time_range:int, time_step:int)-> tuple:
    '''
    产生时间区间
    '''
    intervals = tuple((i, min(i+time_step,time_range)) for i in range(0,time_range,time_step))
    return intervals

def print_realmap_attributes(realmap):
    print("RealMap Attributes:")
    print("- n_r (Number of Restaurants):", realmap.n_r)
    print("- n_c (Number of Customers):", realmap.n_c)
    print("- n (Total Number of Nodes):", realmap.n)
    print("- all_nodes (List of All Nodes):", realmap.all_nodes)
    print("- coordinates (Dictionary of Node Coordinates):", realmap.coordinates)
    print("- all_nodes_names (Dictionary of Node Names):", realmap.all_nodes_names)
    print("- distance_matrix (Distance Matrix between Nodes):", realmap.distance_matrix)
    print("- pairs (List of Restaurant-Customer Pairs):", realmap.pairs)

def print_pdpgraph_attributes(pdpgraph):
    print("PDPGraph Attributes:")
    print("- real_map (Reference to RealMap Object):", pdpgraph.real_map)
    print("- order_pairs_table (Table of Order Pairs):", pdpgraph.order_pairs_table)
    print("- n (Number of Order Pairs):", pdpgraph.n)
    print("- depot (List of Depot Nodes):", pdpgraph.depot)
    print("- dest (List of Destination Nodes):", pdpgraph.dest)
    print("- charging_station (List of Charging Station Nodes):", pdpgraph.charging_station)
    print("- C (List of Customer Nodes):", pdpgraph.C)
    print("- P (List of Pickup Nodes):", pdpgraph.P)
    print("- D (List of Delivery Nodes):", pdpgraph.D)
    print("- N (List of All PDP Nodes):", pdpgraph.N)
    print("- N_depot (List of PDP Nodes for Depot):", pdpgraph.N_depot)
    print("- N_dest (List of PDP Nodes for Destination):", pdpgraph.N_dest)
    print("- p_i (Dictionary of Demands for Each Node):", pdpgraph.p_i)
    print("- pdp_to_real (Dictionary Mapping PDP Nodes to Real Nodes):", pdpgraph.pdp_to_real)
    print("- real_to_pdp (Dictionary Mapping Real Nodes to PDP Nodes):", pdpgraph.real_to_pdp)
    print("- Distance_matrix (Distance Matrix between PDP Nodes):", pdpgraph.distance_matrix)

def print_params_contents(params):
    print("Parameters Contents:")
    print("- realMap (RealMap Object):", params['realMap'])
    print("- time_intervals (List of Time Intervals):", params['time_intervals'])
    print("- demand_table (Demand Table):", params['demand_table'])
    print("- order_pairs_table (Table of Order Pairs):", params['order_pairs_table'])
    print("- v (Number of Vehicles):", params['v'])
    print("- B (Battery Capacity):", params['B'])
    print("- b (Battery Consumption Rate):", params['b'])
    print("- Q (Maximum Capacity of Vehicles):", params['Q'])
    print("- M (Big M Constant):", params['M'])
    print("- n (Number of Order Pairs):", params['n'])
    print("- charging (List of Charging Station Nodes):", params['charging'])
    print("- P (List of Pickup Nodes):", params['P'])
    print("- D (List of Delivery Nodes):", params['D'])
    print("- C (List of Customer Nodes):", params['C'])
    print("- N (List of All PDP Nodes):", params['N'])
    print("- N_depot (List of PDP Nodes for Depot):", params['N_depot'])
    print("- N_dest (List of PDP Nodes for Destination):", params['N_dest'])
    print("- N_0 (List of PDP Nodes Excluding Charging Station):", params['N_0'])
    print("- N_2n1 (List of PDP Nodes Excluding Charging Station):", params['N_2n1'])
    print("- V (List of Vehicle Indices):", params['V'])
    print("- distance_matrix (Distance Matrix between PDP Nodes):", params['distance_matrix'])
    print("- p_i (Dictionary of Demands for Each Node):", params['p_i'])
    print("- e_i (Earliest Time for Each Node):", params['e_i'])
    print("- l_i (Latest Time for Each Node):", params['l_i'])



