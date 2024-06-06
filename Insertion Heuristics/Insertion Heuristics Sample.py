import pandas as pd
import numpy as np

orders = pd.read_csv('.../order_samples.csv')

num_vehicles = 5
vehicle_capacity = 6  # Example capacity
total_orders = int(orders.shape[0]/2-1)
penalty = 2
speed = 4
service = 2
battery = 40

# Extract necessary information from the DataFrame
pickup_delivery_pairs = orders[['pair_index', 'x', 'y']].values
earliest = orders[['earliest']].values
latest = orders[['latest']].values
demand = orders[['demand']].values
x = orders[['x']].values
y = orders[['y']].values

# Initialize the routes, start and end with depot
routes = [[0, len(pickup_delivery_pairs)-1] for _ in range(num_vehicles)]

# Initialize the capacities and time for each vehicle
capacity_dict = {0: 0}
time_dict = {0: 0}

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Insertion heuristic algorithm
def insert_request(routes, pickup_delivery_pairs, earliest, latest, demand, capacity_dict, time_dict):
    for request in pickup_delivery_pairs[1:(total_orders+1)]:  #Enumerate number of orders
        best_vehicle = None
        best_pickup_position = None
        best_dropoff_position = None
        best_cost = float('inf')

        for vehicle_idx, route in enumerate(routes):
            for i in range(1, len(route)):
                new_route = route[:i] + [int(request[0])] + route[i:]
                for j in range(i+1, len(new_route)):
                  new_route = new_route[:j] + [int(request[0]+total_orders)] + new_route[j:]
                  new_time_dict = time_dict.copy()
                  new_capacity_dict = capacity_dict.copy()
                  new_time_dict, new_capacity_dict = update_node(new_route, earliest, new_time_dict, new_capacity_dict)
                  dist = total_distance(new_route)
                  if any(value > vehicle_capacity for value in new_capacity_dict.values()) or (dist > battery):
                      continue
                  else:
                      new_routes = routes.copy()
                      new_routes[vehicle_idx] = new_route
                      cost = calculate_route_cost(new_routes, new_time_dict)
                      if cost < best_cost:
                          best_vehicle = vehicle_idx
                          best_pickup_position = i
                          best_dropoff_position = j
                          best_cost = cost

        if best_vehicle is not None:
            routes[best_vehicle].insert(best_pickup_position, int(request[0]))
            routes[best_vehicle].insert(best_dropoff_position, int(request[0]+total_orders))

            time_dict, capacity_dict = update_node(routes[best_vehicle], earliest, time_dict, capacity_dict)
        else:
            print('Need more vehicles!')

def calculate_route_cost(routes, time_dict):
    cost = 0
    for vehicle_idx, route in enumerate(routes):
      for i in range(len(route) - 1):
          cost += euclidean_distance([x[route[i]], y[route[i]]],[x[route[i+1]], y[route[i+1]]])
    for i in time_dict:
      if time_dict[i] > latest[i]:
        cost += penalty  #penalty cost
    return cost

def update_node(route, earliest, time_dict, capacity_dict):
    for idx, node in enumerate(route[:-1]):
      if idx == 0:
        continue
      time_dict[node] = max(earliest[node],
                            time_dict[route[idx-1]] + service + euclidean_distance([x[route[idx-1]], y[route[idx-1]]],[x[route[idx]], y[route[idx]]])/speed*60)[0] #speed is 4
      capacity_dict[node] = (capacity_dict[route[idx-1]] + demand[node])[0]
    return time_dict, capacity_dict

def total_distance(route):
   dist = 0
   for vehicle_idx, route in enumerate(routes):
      for i in range(len(route) - 1):
          dist += euclidean_distance([x[route[i]], y[route[i]]],[x[route[i+1]], y[route[i+1]]])
   return dist

# Execute the insertion heuristic
insert_request(routes, pickup_delivery_pairs, earliest, latest, demand, capacity_dict, time_dict)

print(routes)