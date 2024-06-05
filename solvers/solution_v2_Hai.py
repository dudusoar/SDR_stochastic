import numpy as np
class Solution:
    """
    PDPTW problem solver.

    Attributes:
        params (dict): A dictionary containing problem parameters.
        n (int): Number of orders.
        v (int): Number of vehicles.
        routes (list): A list of routes for each vehicle.
        arrival_times (ndarray): Arrival times at each node for each vehicle.
        battery_levels (ndarray): Battery levels at each node for each vehicle.
        capacities (ndarray): Capacities at each node for each vehicle.
        total_time (float): Total travel time.
        total_delay (float): Total delay time.
        feasible (bool): Indicates if the solution is feasible.
    """

    def __init__(self, params):
        """
        Initialize the PDPTW solver.

        Args:
            params (dict): A dictionary containing problem parameters.
        """
        self.params = params
        self.n = params['n']  # number of orders
        self.v = params['v']  # number of vehicles

        # data structure for the solution
        self.routes = [[] for _ in range(self.v)]  # each route list for each vehicle
        self.arrival_times = np.zeros((self.v, 2 * self.n + 3))
        self.battery_levels = np.full((self.v, 2 * self.n + 3), self.params['B'])
        self.capacities = np.zeros((self.v, 2 * self.n + 3))

        # evaluation metrics
        self.total_time = 0  # total driving time
        self.total_delay = 0  # total delay time
        self.feasible = True  # feasibility of the solution

        # 错误处理
        if self.params['v'] < 1 or self.params['n'] < 1:
            raise ValueError("Invalid number of vehicles or orders.")
        if self.params['B'] <= 0 or self.params['b'] <= 0 or self.params['Q'] <= 0:
            raise ValueError("Invalid vehicle parameters.")

    def update_solution_state(self):
        """
        Update the arrival times, battery levels, and capacities based on the current vehicle routes.
        Also, calculate the total travel time and total delay time.
        """
        self.total_time = 0
        self.total_delay = 0
        self.arrival_times = np.zeros((self.v, 2 * self.n + 3))
        self.battery_levels = np.full((self.v, 2 * self.n + 3), self.params['B'])
        self.capacities = np.zeros((self.v, 2 * self.n + 3))

        for k in range(self.v):
            route = self.routes[k]
            if len(route) > 0:  # 车辆k被选择
                self.battery_levels[k, 0] = self.params['B']  # 起始点电池电量初始化为满
                self.capacities[k, 0] = 0  # 起始点载荷初始化为 0
                for i in range(1, len(route)):
                    prev_node = route[i - 1]
                    node = route[i]
                    self.arrival_times[k, i] = self.arrival_times[k, i - 1] + self.params['service_time'] + self.params['time_matrix'][prev_node][
                        node] #改用time_matrix计算时间,同时算上每个站点的服务时长
                    if node in self.params['D'] and self.arrival_times[k, i] > self.params['l_i'][node]:
                        self.total_delay += max(self.arrival_times[k, i] - self.params['l_i'][node],0)
                    self.total_time += self.params['time_matrix'][prev_node][node]
                    self.battery_levels[k, i] = self.battery_levels[k, i - 1] - self.params['b'] * \
                                                self.params['distance_matrix'][prev_node][node]
                    self.capacities[k, i] = self.capacities[k, i - 1] + self.params['p_i'][node]

    def evaluate(self):
        """
        Evaluate the current solution and return the objective function value.
        """
        self.update_solution_state()
        obj_value = self.total_time + self.total_delay
        return obj_value

    def check_feasibility(self):
        """
        Check if the current solution satisfies all the constraints.
        """
        # if be served only once
        self.constr_served_only_once()
        if not self.feasible:
            return
        # vehicle behavior constraints
        self.constr_vehicle_behavior()
        if not self.feasible:
            return
        # time constraints
        # self.constr_time()
        # if not self.feasible:
        #     return
        # capacity constraints
        self.constr_capacity()
        if not self.feasible:
            return
        # battery level constraints
        self.constr_battery()
        if not self.feasible:
            return

    def constr_served_only_once(self):
        """
        Check constraint 1: Each node in params['C'] can only be served once.
        """
        visited = {node: 0 for node in self.params['C']}
        for route in self.routes:
            for node in route:
                if node in self.params['C']:
                    visited[node] += 1
                    if visited[node] > 1:
                        self.feasible = False
                        return
        if any(value == 0 for value in visited.values()):
            self.feasible = False
            return
        self.feasible = True


    def constr_vehicle_behavior(self):
        """
        Check constraints 2-5: Vehicle behavior constraints.
        """
        self.constr_vehicle_behavior_1()
        if not self.feasible:
            return
        self.constr_vehicle_behavior_2()
        if not self.feasible:
            return

    def constr_vehicle_behavior_1(self):
        """
        Constraint 2: If a vehicle is selected, it must depart from the depot and return to the destination.
        """
        for k in range(self.v):
            route = self.routes[k]
            if len(route) > 0:  # 车辆k被选择
                if route[0] != 0 or route[-1] != 2 * self.n + 1:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_vehicle_behavior_2(self):
        """
        Constraint 3: If a vehicle serves a node, it must leave that node.
        Also check constraints 4 and 5 simultaneously.

        可以删掉constraint 3相关的check, 因为一个route的node序列一定包含了constraint 3
        """
        for k in range(self.v):
            route = self.routes[k]
            # visited = {node: 0 for node in self.params['N']}
            for i in range(len(route)):
                node = route[i]
                # visited[node] += 1
                # if i < len(route) - 1:  # 不是路线的最后一个点
                #     next_node = route[i + 1]
                #     if node != next_node:  # 下一个点不是当前点
                #         visited[node] -= 1
                #         # 同时检查约束4和约束5
                if node in self.params['P']:
                    delivery_node = node + self.n
                    if delivery_node not in route or route.index(delivery_node) < route.index(node):
                        self.feasible = False
                        return
                elif node in self.params['D']:
                    pickup_node = node - self.n
                    if pickup_node not in route or route.index(pickup_node) > route.index(node):
                        self.feasible = False
                        return
            # if any(visited[node] != 0 for node in self.params['N']):
            #     self.feasible = False
            #     return
        self.feasible = True


    def constr_capacity(self):
        """
        Check constraints 6-8: Capacity constraints.
        """
        self.constr_capacity_1()
        if not self.feasible:
            return
        self.constr_capacity_2()
        if not self.feasible:
            return
        self.constr_capacity_3()

    def constr_capacity_1(self):
        """
        Constraint 6: The capacity level at the depot and destination must be 0.
        """
        for k in range(self.v):
            route = self.routes[k]
            if len(route) > 0:  # 车辆k被选择
                if self.capacities[k, 0] != 0 or self.capacities[k, -1] != 0:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_capacity_2(self):
        """
        Constraint 7: The capacity level at each node must be less than or equal to the maximum capacity params['Q'] and greater than or equal to 0.
        """
        for k in range(self.v):
            route = self.routes[k]
            for i in range(len(route)):
                node = route[i]
                if self.capacities[k, i] > self.params['Q'] or self.capacities[k, i] < 0:
                    self.feasible = False
                    return
        self.feasible = True

    # def constr_capacity_3(self):
    #     """
    #     Constraint 8: When passing a pickup node, the capacity level should decrease by the demand at that node;
    #     when passing a delivery node, the capacity level should increase by the demand at that node (note that the demand at a delivery node is negative).
    #     这个条件可以删掉,前面的function已经根据route的站点序列更新了载货量,这个条件自然成立
    #     """
    #     for k in range(self.v):
    #         route = self.routes[k]
    #         for i in range(1, len(route)):
    #             node = route[i]
    #             prev_node = route[i - 1]
    #             if node in self.params['P']:  # pickup node
    #                 if self.capacities[k, i] != self.capacities[k, i - 1] - self.params['p_i'][node]:
    #                     self.feasible = False
    #                     return
    #             elif node in self.params['D']:  # delivery node
    #                 if self.capacities[k, i] != self.capacities[k, i - 1] - self.params['p_i'][node]:
    #                     self.feasible = False
    #                     return
    #             else:  # other nodes (depot, destination, charging station)
    #                 if self.capacities[k, i] != self.capacities[k, i - 1]:
    #                     self.feasible = False
    #                     return
    #     self.feasible = True

    # def constr_time(self):
    #     """
    #     Time constraints. The arrival time at each node must be later than the earliest time e_i
    #     """
    #     for k in range(self.v):
    #         route = self.routes[k]
    #         for i in range(len(route)):
    #             node = route[i]
    #             if self.arrival_times[k, i] > self.params['e_i'][node]:
    #                 self.feasible = False
    #                 return
    #     self.feasible = True

    def constr_battery(self):
        """
        The battery level at each node must be between 0 and params['B'].
        """
        for k in range(self.v):
            route = self.routes[k]
            for i in range(len(route)):
                if self.battery_levels[k, i] < 0 or self.battery_levels[k, i] > self.params['B']:
                    self.feasible = False
                    return
        self.feasible = True
