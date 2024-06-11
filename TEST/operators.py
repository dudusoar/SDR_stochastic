# Removal operators

import random
import numpy as np
from copy import deepcopy

class RemovalOperators:
    def __init__(self, solution):
        self.solution = solution
        self.instance = solution.instance 

    def shaw_removal(self, num_remove, p):
        removed_requests = []
        remaining_requests = list(range(1, self.instance.n+1))

        # 随机选择一个起点请求
        initial_request =  random.choice(remaining_requests)
        removed_requests.append(initial_request)
        remaining_requests.remove(initial_request)

        # 距离和时间的归一化因子
        max_distance = np.max(self.instance.distance_matrix)
        max_arrive_time = np.max(self.solution.route_arrival_times)

        while len(removed_requests) < num_remove:
            last_removed = random.choice(removed_requests)
            L = [req for req in remaining_requests]
            L.sort(key = lambda req: self.calculate_similarity(last_removed, req, max_distance,max_arrive_time))

            y = random.random()
            selected_request = L[int(y**p*len(L))]
            removed_requests.append(selected_request)
            remaining_requests.remove(selected_request)
        
        return self.remove_requests(removed_requests)
    
    def calculate_similarity(self,req1,req2,max_distance,max_arrive_time):
        '''for shaw_removal'''
        pickup1, delivery1 = req1, req1 + self.instance.n
        pickup2, delivery2 = req2, req2 + self.instance.n

        dist_pickup = self.instance.distance_matrix[pickup1][pickup2] / max_distance
        dist_delivery = self.instance.distance_matrix[delivery1][delivery2] / max_distance

        arrival_time_pickup = (self.get_arrival_time(pickup1) - self.get_arrival_time(pickup2))/ max_arrive_time
        arrival_time_delivery = (self.get_arrival_time(delivery1) - self.get_arrival_time(delivery2)) / max_arrive_time

        return  dist_pickup + dist_delivery + arrival_time_pickup +  arrival_time_delivery
    
    def get_arrival_time(self, request):
        '''for shaw_removal'''
        for vehicle_id, route in enumerate(self.solution.routes):
            if request in route:
                return self.solution.route_arrival_times[vehicle_id][route.index(request)]
        return None
    
    def random_removal(self, num_remove):
        removed_requests = random.sample(range(1, self.instance.n + 1), num_remove)
        return self.remove_requests(removed_requests)
    
    def worst_removal(self, num_remove):
        contributions = [(req, self.calculate_contribution(req)) for req in range(1, self.instance.n + 1)]
        contributions.sort(key=lambda x: x[1], reverse=True)
        removed_requests = [req for req, _ in contributions[:num_remove]]
        return self.remove_requests(removed_requests)

    def calculate_contribution(self, req):
        '''for  worst_removal'''
        temp_solution = deepcopy(self.solution)
        pickup, delivery = req, req + self.instance.n

        # 移除请求的pickup和delivery点
        for route in temp_solution.routes:
            if pickup in route:
                route.remove(pickup)
                route.remove(delivery)

        # 重新计算所有状态变量和目标函数值
        temp_solution.calculate_all()
        original_objective = self.solution.objective_function()
        new_objective = temp_solution.objective_function()

        # 贡献度定义为目标函数值的变化量
        contribution = original_objective - new_objective
        return contribution

    def remove_requests(self,requests):
        new_solution = deepcopy(self.solution)
        removed_pairs= []
        
        for request in requests:
            pickup_node, delivery_node = request, request + self.instance.n
            for route in new_solution:
                if pickup_node in route:
                    route.remove(pickup_node)
                    route.remove(delivery_node)

            removed_pairs.append((pickup_node,delivery_node))
        
        return new_solution, removed_pairs

class RepairOperators:
    def __init__(self, solution):
        self.solution = solution
        self.instance = solution.instance

    def basic_greedy_heuristic(self, removed_pairs):
        for pickup, delivery in removed_pairs:
            best_cost = float('inf')
            best_route = None
            best_insert_pos = None

            for vehicle_id, route in enumerate(self.solution.routes):
                for i in range(1, len(route)):
                    for j in range(i, len(route) + 1):
                        temp_route = route[:i] + [pickup] + route[i:j] + [delivery] + route[j:]
                        temp_solution = deepcopy(self.solution)
                        temp_solution.routes[vehicle_id] = temp_route
                        temp_solution.calculate_all()

                        cost = temp_solution.objective_function()
                        if cost < best_cost and temp_solution.is_feasible():
                            best_cost = cost
                            best_route = vehicle_id
                            best_insert_pos = (i, j)

            if best_route is not None and best_insert_pos is not None:
                i, j = best_insert_pos
                self.solution.routes[best_route] = self.solution.routes[best_route][:i] + [pickup] + self.solution.routes[best_route][i:j] + [delivery] + self.solution.routes[best_route][j:]
                self.solution.calculate_all()

    def regret_heuristic(self, removed_pairs, k):
        while removed_pairs:
            max_regret = float('-inf')
            best_request = None
            best_route = None
            best_insert_pos = None

            for pickup, delivery in removed_pairs:
                insertion_costs = []
                for vehicle_id, route in enumerate(self.solution.routes):
                    for i in range(1, len(route)):
                        for j in range(i, len(route) + 1):
                            temp_route = route[:i] + [pickup] + route[i:j] + [delivery] + route[j:]
                            temp_solution = deepcopy(self.solution)
                            temp_solution.routes[vehicle_id] = temp_route
                            temp_solution.calculate_all()

                            cost = temp_solution.objective_function()
                            if temp_solution.is_feasible():
                                insertion_costs.append((cost, vehicle_id, i, j))

                insertion_costs.sort(key=lambda x: x[0])
                if len(insertion_costs) >= k:
                    regret = insertion_costs[k-1][0] - insertion_costs[0][0]
                else:
                    regret = insertion_costs[-1][0] - insertion_costs[0][0]

                if regret > max_regret:
                    max_regret = regret
                    best_request = (pickup, delivery)
                    best_route = insertion_costs[0][1]
                    best_insert_pos = (insertion_costs[0][2], insertion_costs[0][3])

            if best_request is not None and best_route is not None and best_insert_pos is not None:
                removed_pairs.remove(best_request)
                pickup, delivery = best_request
                i, j = best_insert_pos
                self.solution.routes[best_route] = self.solution.routes[best_route][:i] + [pickup] + self.solution.routes[best_route][i:j] + [delivery] + self.solution.routes[best_route][j:]
                self.solution.calculate_all()
