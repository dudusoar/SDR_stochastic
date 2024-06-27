import numpy as np
import matplotlib.pyplot as plt

class PDPTWSolution:
    '''
    structure
    1. update_all:
       - initialize_routes
       - calculate_all
         - calculate_battery_capacity_levels
         - calculate_arrival_leave_wait_times
         - calculate_travel_delay_wait_times
       - update_visit_record
    2. objective_function
    3. is_feasible:
       - get_selected_vehicles
       - check_capacity_constraint
       - check_battery_constraint
       - check_pickup_delivery_order
    '''
    def __init__(self, instance, vehicle_capacity, battery_capacity, battery_consume_rate, routes):
        """
        初始化 PDPTWSolution 对象
        :param instance: PDPTWInstance 对象
        :param vehicle_capacity: 车辆容量
        :param battery_capacity: 电池容量
        :param battery_consume_rate: 电池消耗率
        :param routes: 给定的路径列表
        """
        # unchanged parameters within the same instance
        self.instance = instance
        self.gamma = instance.gamma
        self.vehicle_capacity = vehicle_capacity
        self.battery_capacity = battery_capacity
        self.battery_consume_rate = battery_consume_rate
        self.routes = routes
        self.num_vehicles = len(routes)

        # need to be initialized based on the new routes
        # battery and capacity
        self.route_battery_levels = np.empty((self.num_vehicles,), dtype=object)
        self.route_capacity_levels = np.empty((self.num_vehicles,), dtype=object)
        # time
        self.route_arrival_times = np.empty((self.num_vehicles,), dtype=object)
        self.route_leave_times = np.empty((self.num_vehicles,), dtype=object)
        self.route_wait_times = np.empty((self.num_vehicles,), dtype=object)
        # evaluation
        self.total_travel_times = np.zeros((self.num_vehicles,))
        self.total_delay_times = np.zeros((self.num_vehicles,))
        self.total_wait_times = np.zeros((self.num_vehicles,))
        # visited record
        self.visited_requests = set()
        self.unvisited_requests = set()
        self.unvisited_pairs = []
        self.visited_pairs = []

        self.initialize_routes()
        self.calculate_all()
        self.update_visit_record()

    def update_all(self):
        """
        更新所有相关属性
        :param new_routes: 新的路径列表
        """
        self.initialize_routes()
        self.calculate_all()
        self.update_visit_record()

    def initialize_routes(self):
        """
        初始化与路径相关的变量
        """
        for vehicle_id in range(self.num_vehicles):
            route = self.routes[vehicle_id]
            self.route_battery_levels[vehicle_id] = np.zeros((len(route),))
            self.route_capacity_levels[vehicle_id] = np.zeros((len(route),))
            self.route_arrival_times[vehicle_id] = np.zeros((len(route),))
            self.route_leave_times[vehicle_id] = np.zeros((len(route),))
            self.route_wait_times[vehicle_id] = np.zeros((len(route),))

    def calculate_all(self):
        """
        计算所有状态变量和目标函数值
        """
        for vehicle_id in range(self.num_vehicles):
            self.calculate_battery_capacity_levels(vehicle_id)
            self.calculate_arrival_leave_wait_times(vehicle_id)
            self.calculate_travel_delay_wait_times(vehicle_id)
    
    def update_visit_record(self):
        """
        更新未被服务的请求
        """
        visited_requests = set()
        for route in self.routes:
            for node in route:
                if node != 0:  # 忽略depot
                    if node <= self.instance.n:
                        visited_requests.add(node)
                    else:
                        visited_requests.add(node - self.instance.n)
        all_requests = set(range(1, self.instance.n + 1))    
        unvisited_requests = all_requests - visited_requests
        # update
        self.visited_requests = visited_requests
        self.unvisited_requests = unvisited_requests 
        self.visited_pairs = [(node, node + self.instance.n) for node in visited_requests]
        self.unvisited_pairs = [(node, node + self.instance.n) for node in unvisited_requests]

    def calculate_battery_capacity_levels(self, vehicle_id):
        """
        计算指定车辆在路径上每个节点的电池电量和容量水平
        :param vehicle_id: 车辆 ID
        """
        route = self.routes[vehicle_id]
        battery_levels = self.route_battery_levels[vehicle_id]
        capacity_levels = self.route_capacity_levels[vehicle_id]
        time_matrix = self.instance.time_matrix

        battery_levels[0] = self.battery_capacity
        capacity_levels[0] = 0

        for i in range(1, len(route)):
            prev_node = route[i - 1]
            curr_node = route[i]
            # update the battery
            time = time_matrix[prev_node][curr_node]
            battery_levels[i] = battery_levels[i - 1] - time * self.battery_consume_rate
            # update the capacity
            capacity_levels[i] = capacity_levels[i - 1] + self.instance.demands[curr_node]

    def calculate_arrival_leave_wait_times(self, vehicle_id):
        """
        计算指定车辆在路径上每个节点的到达时间、离开时间和等待时间
        :param vehicle_id: 车辆 ID
        """
        route = self.routes[vehicle_id]
        arrival_times = self.route_arrival_times[vehicle_id]
        leave_times = self.route_leave_times[vehicle_id]
        wait_times = self.route_wait_times[vehicle_id]
        time_matrix = self.instance.time_matrix

        arrival_times[0] = 0
        wait_times[0] = 0
        leave_times[0] = self.instance.service_times[route[0]]

        for i in range(1, len(route)):
            prev_node = route[i - 1]
            curr_node = route[i]
            arrival_times[i] = leave_times[i - 1] + time_matrix[prev_node][curr_node]
            wait_times[i] = max(0, self.instance.time_windows[curr_node][0] - arrival_times[i])
            if arrival_times[i] > self.instance.time_windows[curr_node][0]:
                wait_times[i] = 0
            leave_times[i] = arrival_times[i] + wait_times[i] + self.instance.service_times[curr_node]

    def calculate_travel_delay_wait_times(self, vehicle_id):
        """
        计算指定车辆的总行驶时间、总延误时间和总等待时间
        :param vehicle_id: 车辆 ID
        """
        route = self.routes[vehicle_id]
        leave_times = self.route_leave_times[vehicle_id]
        wait_times = self.route_wait_times[vehicle_id]
        time_matrix = self.instance.time_matrix

        travel_time = 0
        delay_time = 0
        wait_time = 0

        for i in range(len(route) - 1):
            curr_node = route[i]
            next_node = route[i + 1]
            travel_time += time_matrix[curr_node][next_node]
            
            if self.instance.time_windows[next_node][1] != float('inf'):
                delay_time += max(0, leave_times[i] - self.instance.time_windows[next_node][1])
            
            wait_time += wait_times[i]

        self.total_travel_times[vehicle_id] = travel_time
        self.total_delay_times[vehicle_id] = delay_time
        self.total_wait_times[vehicle_id] = wait_time

    def objective_function(self):
        """
        计算目标函数值
        :return: 目标函数值（总行驶时间 + 总延误时间）
        """
        unvisited_penalty = len(self.unvisited_requests) * self.gamma
        return np.sum(self.total_travel_times) + np.sum(self.total_delay_times) + unvisited_penalty

    # ================================ feasibility check ================================
    def is_feasible(self):
        """
        检查解的可行性
        """
        selected_vehicles = self.get_selected_vehicles()
        if not selected_vehicles:
            return False
        if not self.check_capacity_constraint(selected_vehicles):
            return False
        if not self.check_battery_constraint(selected_vehicles):
            return False
        if not self.check_pickup_delivery_order(selected_vehicles):
            return False
        return True

    def get_selected_vehicles(self):
        """
        获取被选择的车辆列表
        :return: 被选择的车辆 ID 列表
        没有被选择的车辆的路径被初始化[0,0]
        """
        selected_vehicles = []
        for vehicle_id in range(self.num_vehicles):
            if len(self.routes[vehicle_id]) > 2:
                selected_vehicles.append(vehicle_id)
        return selected_vehicles

    def check_capacity_constraint(self, selected_vehicles):
        """
        检查容量约束
        :param selected_vehicles: 被选择的车辆 ID 列表
        """
        for vehicle_id in selected_vehicles:
            if np.any(self.route_capacity_levels[vehicle_id] > self.vehicle_capacity):
                return False
        return True

    def check_battery_constraint(self, selected_vehicles):
        """
        检查电池约束
        :param selected_vehicles: 被选择的车辆 ID 列表
        """
        for vehicle_id in selected_vehicles:
            if np.any(self.route_battery_levels[vehicle_id] < 0):
                return False
        return True
    
    def check_pickup_delivery_order(self, selected_vehicles):
        """
        检查取货-送货顺序约束
        :param selected_vehicles: 被选择的车辆 ID 列表
        :return: True 如果取货-送货顺序约束满足，False 否则
        """
        for vehicle_id in selected_vehicles:
            route = self.routes[vehicle_id]
            visited_pickups = set()
            for node in route:
                if 1 <= node <= self.instance.n:
                    visited_pickups.add(node)
                elif self.instance.n < node <= 2 * self.instance.n:
                    if node - self.instance.n not in visited_pickups:
                        return False
        return True

    # ================================ plot ================================
    def plot_solution(self, vehicle_ids=None):
        """
        绘制解的路线图
        :param vehicle_ids: 要绘制路径的车辆 ID 列表（可选），默认为 None，表示绘制所有被选择车辆的路径
        """
        selected_vehicles = self.get_selected_vehicles()
        if not selected_vehicles:
            print("No vehicle is selected in the solution.")
            return
    
        if vehicle_ids is None:
            vehicle_ids = selected_vehicles
        else:
            vehicle_ids = [vehicle_id for vehicle_id in vehicle_ids if vehicle_id in selected_vehicles]
    
        if not vehicle_ids:
            print("No valid vehicle IDs provided.")
            return
    
        plt.figure(figsize=(8, 8))
        plt.scatter(self.instance.depot[0], self.instance.depot[1], c='red', marker='s', s=100, label='Depot')
        plt.scatter([p[0] for p in self.instance.pickup_points], [p[1] for p in self.instance.pickup_points], c='blue', marker='o', label='Pickup')
        plt.scatter([d[0] for d in self.instance.delivery_points], [d[1] for d in self.instance.delivery_points], c='green', marker='d', label='Delivery')
    
        plt.annotate('Depot', (self.instance.depot[0], self.instance.depot[1]), textcoords='offset points', xytext=(0, 10), ha='center')
        for i, pickup_point in enumerate(self.instance.pickup_points, start=1):
            plt.annotate(f'P{i}', (pickup_point[0], pickup_point[1]), textcoords='offset points', xytext=(0, 5), ha='center')
        for i, delivery_point in enumerate(self.instance.delivery_points, start=1):
            plt.annotate(f'D{i}', (delivery_point[0], delivery_point[1]), textcoords='offset points', xytext=(0, 5), ha='center')
    
        color_map = plt.cm.get_cmap('viridis', len(vehicle_ids))
        for i, vehicle_id in enumerate(vehicle_ids):
            route = self.routes[vehicle_id]
            color = color_map(i)
            
            route_points = [self.instance.depot] + [self.instance.pickup_points[node-1] if node <= self.instance.n else self.instance.delivery_points[node-self.instance.n-1] for node in route[1:-1]] + [self.instance.depot]
            route_x = [point[0] for point in route_points]
            route_y = [point[1] for point in route_points]
            
            plt.plot(route_x, route_y, color=color, linestyle='-', linewidth=1.5, alpha=0.8, label=f'Vehicle {vehicle_id+1}')
    
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PDPTW Solution')
        plt.legend()
        plt.grid(True)
        plt.show()