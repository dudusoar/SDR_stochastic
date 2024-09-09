import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class PDPTWSolution:
    def __init__(self, instance, vehicle_capacity, battery_capacity, battery_consume_rate, routes, penalty_unvisited, penalty_delayed):
        self.instance = instance
        self.vehicle_capacity = vehicle_capacity
        self.battery_capacity = battery_capacity
        self.battery_consume_rate = battery_consume_rate
        self.routes = routes
        self.penalty_unvisited= penalty_unvisited # for unvisited request
        self.penalty_delayed = penalty_delayed # for delayed orders
        self.num_vehicles = len(routes)

        self.df = self.instance.generate_whole_table()
        self.time_matrix = self.instance.time_matrix
        self.distance_matrix = self.instance.distance_matrix

        self.route_battery_levels = [np.zeros(len(route)) for route in self.routes]
        self.route_capacity_levels = [np.zeros(len(route)) for route in self.routes]
        self.route_arrival_times = [np.zeros(len(route)) for route in self.routes]
        self.route_leave_times = [np.zeros(len(route)) for route in self.routes]
        self.route_wait_times = [np.zeros(len(route)) for route in self.routes]

        self.total_travel_times = np.zeros(self.num_vehicles)
        self.total_delay_times = np.zeros(self.num_vehicles)
        self.total_wait_times = np.zeros(self.num_vehicles)
        self.total_delay_orders = np.zeros(self.num_vehicles)

        self.visited_requests = set()
        self.unvisited_requests = set()

        self.update_all()

    # ================ status update ============
    def update_all(self):
        self.calculate_all()
        self.update_visit_record()

    def calculate_all(self):
        for vehicle_id in range(self.num_vehicles):
            self.calculate_battery_capacity_levels(vehicle_id)
            self.calculate_arrival_leave_wait_times(vehicle_id)
            self.calculate_travel_delay_wait_times(vehicle_id)
    def update_visit_record(self):
        self.visited_requests = set()
        all_requests = set(self.df[self.df['Type'].isin(['cp', 'cd'])]['ID'].values)

        for route in self.routes:
            self.visited_requests.update(set(route) & all_requests)

        self.unvisited_requests = all_requests - self.visited_requests
    # ============================

    def calculate_battery_capacity_levels(self, vehicle_id):
        route = self.routes[vehicle_id]
        battery_levels = self.route_battery_levels[vehicle_id]
        capacity_levels = self.route_capacity_levels[vehicle_id]

        battery_levels[0] = self.battery_capacity
        capacity_levels[0] = 0

        for i in range(1, len(route)):
            prev_id, curr_id = route[i - 1], route[i]
            prev_real_index = self.df.loc[self.df['ID'] == prev_id, 'RealIndex'].values[0]
            curr_real_index = self.df.loc[self.df['ID'] == curr_id, 'RealIndex'].values[0]

            travel_time = self.time_matrix[prev_real_index][curr_real_index]

            if self.df.loc[self.df['ID'] == curr_id, 'Type'].values[0] == 'charging':
                battery_levels[i] = self.battery_capacity
            else:
                battery_levels[i] = battery_levels[i - 1] - travel_time * self.battery_consume_rate

            capacity_levels[i] = capacity_levels[i - 1] + self.df.loc[self.df['ID'] == curr_id, 'Demand'].values[0]

    def calculate_arrival_leave_wait_times(self, vehicle_id):
        route = self.routes[vehicle_id]
        arrival_times = self.route_arrival_times[vehicle_id]
        leave_times = self.route_leave_times[vehicle_id]
        wait_times = self.route_wait_times[vehicle_id]

        arrival_times[0] = 0
        wait_times[0] = 0
        leave_times[0] = self.df.loc[self.df['ID'] == route[0], 'ServiceTime'].values[0]

        for i in range(1, len(route)):
            prev_id, curr_id = route[i - 1], route[i]
            prev_real_index = self.df.loc[self.df['ID'] == prev_id, 'RealIndex'].values[0]
            curr_real_index = self.df.loc[self.df['ID'] == curr_id, 'RealIndex'].values[0]

            travel_time = self.time_matrix[prev_real_index][curr_real_index]
            arrival_times[i] = leave_times[i - 1] + travel_time

            start_time = self.df.loc[self.df['ID'] == curr_id, 'StartTime'].values[0]
            wait_times[i] = max(0, start_time - arrival_times[i])

            service_time = self.df.loc[self.df['ID'] == curr_id, 'ServiceTime'].values[0]
            leave_times[i] = arrival_times[i] + wait_times[i] + service_time

    def calculate_travel_delay_wait_times(self, vehicle_id):
        route = self.routes[vehicle_id]
        arrival_times = self.route_arrival_times[vehicle_id]
        leave_times = self.route_leave_times[vehicle_id]
        wait_times = self.route_wait_times[vehicle_id]

        travel_time = 0
        wait_time = sum(wait_times)
        delay_time = 0
        delay_count = 0

        for i in range(len(route) - 1):
            curr_id, next_id = route[i], route[i + 1]
            curr_real_index = self.df.loc[self.df['ID'] == curr_id, 'RealIndex'].values[0]
            next_real_index = self.df.loc[self.df['ID'] == next_id, 'RealIndex'].values[0]

            travel_time += self.time_matrix[curr_real_index][next_real_index]

        for i, node_id in enumerate(route):
            end_time = self.df.loc[self.df['ID'] == node_id, 'EndTime'].values[0]
            if end_time != float('inf') and arrival_times[i] > end_time:
                delay_count += 1
                delay_time += arrival_times[i] - end_time

        self.total_travel_times[vehicle_id] = travel_time
        self.total_delay_times[vehicle_id] = delay_time
        self.total_wait_times[vehicle_id] = wait_time
        self.total_delay_orders[vehicle_id] = delay_count

    def objective_function(self):
        travel_cost = sum(self.total_travel_times) / 60 * self.instance.robot_speed
        unvisited_penalty = len(self.unvisited_requests) * self.penalty_unvisited
        late_penalty = sum(self.total_delay_orders) * self.penalty_delayed
        return travel_cost + unvisited_penalty + late_penalty

    def is_feasible(self):
        return (self.check_capacity_constraint() and
                self.check_battery_constraint() and
                self.check_pickup_delivery_order())

    def check_capacity_constraint(self):
        return all(np.all(capacity <= self.vehicle_capacity) for capacity in self.route_capacity_levels)

    def check_battery_constraint(self):
        return all(np.all(battery >= 0) for battery in self.route_battery_levels)

    def check_pickup_delivery_order(self):
        for route in self.routes:
            visited_pickups = set()
            for node_id in route:
                node_type = self.df.loc[self.df['ID'] == node_id, 'Type'].values[0]
                if node_type == 'cp':
                    visited_pickups.add(node_id)
                elif node_type == 'cd':
                    partner_id = self.df.loc[self.df['ID'] == node_id, 'PartnerID'].values[0]
                    if partner_id not in visited_pickups:
                        return False
        return True

    def plot_solution(self):
        plt.figure(figsize=(12, 10))

        # Create a dictionary to track all orders pointing to the same location
        location_orders = defaultdict(list)

        # Plot all nodes and collect overlapping indices
        for node_type in ['depot', 'charging', 'cp', 'cd']:
            nodes = self.df[self.df['Type'] == node_type]
            color = {'depot': 'red', 'charging': 'purple', 'cp': 'blue', 'cd': 'green'}[node_type]
            marker = {'depot': 's', 'charging': '^', 'cp': 'o', 'cd': 'd'}[node_type]
            label = {'depot': 'Depot', 'charging': 'Charging Station', 'cp': 'Pickup', 'cd': 'Delivery'}[node_type]

            for _, node in nodes.iterrows():
                plt.scatter(node['X'], node['Y'], c=color, marker=marker, s=100)
                location = (node['X'], node['Y'])
                location_orders[location].append(str(node['ID']))

        # Annotate overlapping points
        for location, orders in location_orders.items():
            if len(orders) > 1:
                plt.annotate(f"[{', '.join(orders)}]", location, xytext=(5, 5), textcoords='offset points', fontsize=8,
                             ha='left', va='bottom')
            else:
                plt.annotate(orders[0], location, xytext=(5, 5), textcoords='offset points', fontsize=8, ha='left',
                             va='bottom')

        # Plot routes
        color_map = plt.cm.get_cmap('viridis', self.num_vehicles)
        for vehicle_id, route in enumerate(self.routes):
            if len(route) > 2:  # Only plot non-empty routes
                color = color_map(vehicle_id)
                route_x = self.df.loc[self.df['ID'].isin(route), 'X'].tolist()
                route_y = self.df.loc[self.df['ID'].isin(route), 'Y'].tolist()
                plt.plot(route_x, route_y, color=color, linestyle='-', linewidth=1.5, alpha=0.8,
                         label=f'Vehicle {vehicle_id + 1}')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('PDPTW Solution')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_solution_summary(self):
        return {
            "Total Travel Time": sum(self.total_travel_times),
            "Total Delay Time": sum(self.total_delay_times),
            "Total Wait Time": sum(self.total_wait_times),
            "Total Delayed Orders": sum(self.total_delay_orders),
            "Unvisited Requests": len(self.unvisited_requests),
            "Objective Value": self.objective_function(),
            "Is Feasible": self.is_feasible()
        }