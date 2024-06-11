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

