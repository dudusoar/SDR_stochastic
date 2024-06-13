# Removal operators

import random
import numpy as np
from copy import deepcopy

class RemovalOperators:
    def __init__(self, solution):
        self.solution = solution
        self.instance = solution.instance 
    
    # first removal mothod
    def shaw_removal(self, num_remove, p):
        removed_requests = []
        remaining_requests = list(range(1, self.instance.n+1))

        # 随机选择一个起点请求
        initial_request =  random.choice(remaining_requests)
        removed_requests.append(initial_request)
        remaining_requests.remove(initial_request)

        # 距离和时间的归一化因子
        max_distance = np.max(self.instance.distance_matrix)
        max_arrive_time = np.max([np.max(arrival_time) for arrival_time in self.solution.route_arrival_times])

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
    
    def get_arrival_time(self, node):
        '''
        for shaw_removal
        get the arrival time of the node
        '''
        for vehicle_id, route in enumerate(self.solution.routes):
            if node in route:
                return self.solution.route_arrival_times[vehicle_id][route.index(node)]
        return None
    
    # second removal method
    def random_removal(self, num_remove):
        removed_requests = random.sample(range(1, self.instance.n + 1), num_remove)
        return self.remove_requests(removed_requests)
    
    # third removal method
    def worst_removal(self, num_remove):
        contributions = [(req, self.calculate_contribution(req)) for req in range(1, self.instance.n + 1)]
        contributions.sort(key=lambda x: x[1], reverse=True)
        removed_requests = [req for req, _ in contributions[:num_remove]]
        #print(contributions)
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
            for route in new_solution.routes:
                if pickup_node in route:
                    route.remove(pickup_node)
                    route.remove(delivery_node)

            removed_pairs.append((pickup_node,delivery_node))
            new_solution.update_all()
        
        return new_solution, removed_pairs

class RepairOperators:
    def __init__(self, solution):
        self.solution = solution
        self.instance = solution.instance
        self.insertion_log = []  # 用于记录插入日志

    # first insertion method
    def greedy_insertion(self, removed_pairs):
        for pickup, delivery in removed_pairs:
            best_cost = float('inf')
            best_route = None
            best_insert_position = None

            for vehicle_id, route in enumerate(self.solution.routes):
                for i in range(1, len(route)):
                    for j in range(i, len(route) + 1):
                        temp_route = route[:i] + [pickup] + route[i:j] + [delivery] + route[j:]

                        temp_solution = deepcopy(self.solution)
                        temp_solution.routes[vehicle_id] = temp_route
                        temp_solution.update_all()

                        if temp_solution.is_feasible():
                            cost = temp_solution.objective_function()
                            if cost < best_cost:
                                best_cost = cost
                                best_route = vehicle_id
                                best_insert_position = (i, j)

            if best_route is not None and best_insert_position is not None:
                i, j = best_insert_position
                self.solution.routes[best_route] = self.solution.routes[best_route][:i] + [pickup] + self.solution.routes[best_route][i:j] + [delivery] + self.solution.routes[best_route][j:]
                self.solution.update_all()
                self.record_insertion(best_route, pickup, delivery, best_insert_position)  # 记录插入位置

    # second insertion algorithm
    def regret_insertion(self, removed_pairs, k):
        while removed_pairs:
            max_regret = float('-inf')
            best_request = None
            best_route = None
            best_insert_position = None

            for pickup, delivery in removed_pairs:
                insertion_costs = []
                for vehicle_id, route in enumerate(self.solution.routes):
                    for i in range(1, len(route)):
                        for j in range(i, len(route) + 1):
                            temp_route = route[:i] + [pickup] + route[i:j] + [delivery] + route[j:]

                            temp_solution = deepcopy(self.solution)
                            temp_solution.routes[vehicle_id] = temp_route
                            temp_solution.update_all()

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
                    best_insert_position = (insertion_costs[0][2], insertion_costs[0][3])

            if best_request is not None and best_route is not None and best_insert_position is not None:
                removed_pairs.remove(best_request)
                pickup, delivery = best_request
                i, j = best_insert_position
                self.solution.routes[best_route] = self.solution.routes[best_route][:i] + [pickup] + self.solution.routes[best_route][i:j] + [delivery] + self.solution.routes[best_route][j:]
                self.solution.update_all()
                self.record_insertion(best_route, pickup, delivery, best_insert_position)  # 记录插入位置

    def record_insertion(self, vehicle_id, pickup, delivery, position):
        """
        记录插入位置
        :param vehicle_id: 车辆ID
        :param pickup: 取货点
        :param delivery: 送货点
        :param position: 插入位置 (i, j)
        """
        self.insertion_log.append({
            'vehicle_id': vehicle_id,
            'pickup': pickup,
            'delivery': delivery,
            'position': position
        })

    def get_insertion_log(self):
        """
        获取插入日志
        :return: 插入日志
        """
        return self.insertion_log
    
class SwapOperators:
    def __init__(self, solution):
        self.solution = solution
        self.instance = solution.instance

    def two_opt(self, max_iterations):
        """
        运行 2-opt 算法求解 PDPTW 问题
        :return: 最优解（PDPTWSolution 对象）
        """
        best_solution = deepcopy(self.solution)
        current_solution = deepcopy(self.solution)

        for _ in range(max_iterations):
            improved = False

            for vehicle_id in range(current_solution.num_vehicles):
                route = current_solution.routes[vehicle_id]
                if len(route) <= 4:
                    continue
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        new_route = self.two_opt_swap(route, i, j)
                        if new_route is None:
                            continue

                        new_routes = deepcopy(current_solution.routes)
                        new_routes[vehicle_id] = new_route

                        # 不创建新的实例，直接更新当前解
                        current_solution.routes = new_routes
                        current_solution.calculate_all()

                        if current_solution.is_feasible() and current_solution.objective_function() < best_solution.objective_function():
                            best_solution = deepcopy(current_solution)
                            improved = True
                            break

                    if improved:
                        break

            if not improved:
                break

        return best_solution

    def two_opt_swap(route, i, j):
        """
        执行 2-opt 交换
        :param route: 路径列表
        :param i: 第一个要交换的节点的索引
        :param j: 第二个要交换的节点的索引
        :return: 交换后的新路径列表，如果交换无效则返回 None
        """
        if i == 0 or j == len(route) - 1:
            return None

        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
        return new_route

    def swap_star(self, max_iterations):
        """
        运行 SWAP* 算法求解 PDPTW 问题
        :param max_iterations: 最大迭代次数
        :return: 最优解（PDPTWSolution 对象）
        """
        best_solution = deepcopy(self.solution)
        current_solution = deepcopy(self.solution)

        for _ in range(max_iterations):
            improved = False

            for vehicle_id in range(current_solution.num_vehicles):
                route = current_solution.routes[vehicle_id]
                if len(route) <= 2:
                    continue

                for i in range(1, len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = self.swap_star_move(route, i, j)
                        if new_route is None:
                            continue

                        new_routes = deepcopy(current_solution.routes)
                        new_routes[vehicle_id] = new_route

                        # 不创建新的实例，直接更新当前解
                        current_solution.routes = new_routes
                        current_solution.calculate_all()

                        if current_solution.is_feasible() and current_solution.objective_function() < best_solution.objective_function():
                            best_solution = deepcopy(current_solution)
                            improved = True
                            break

                    if improved:
                        break

            if not improved:
                break

        return best_solution

    def swap_star_move(self, route, i, j):
        """
        执行 SWAP* 交换
        :param route: 路径列表
        :param i: 第一个要交换的节点的索引
        :param j: 第二个要交换的节点的索引
        :return: 交换后的新路径列表，如果交换无效则返回 None
        """
        if i == 0 or j == len(route) - 1:
            return None

        # 获取客户v和v'
        v = route[i]
        v_prime = route[j]

        # 计算新插入位置
        insertion_positions_v = self.get_best_insertion_positions(route, v, j)
        insertion_positions_v_prime = self.get_best_insertion_positions(route, v_prime, i)

        # 找到最佳插入位置
        best_insertion_v = min(insertion_positions_v, key=lambda pos: pos[1])
        best_insertion_v_prime = min(insertion_positions_v_prime, key=lambda pos: pos[1])

        # 执行交换并返回新路线
        new_route = route[:best_insertion_v[0]] + [v] + route[best_insertion_v[0]:best_insertion_v[1]] + [v_prime] + route[best_insertion_v[1]:]
        return new_route

    def get_best_insertion_positions(self, route, node, node_index):
        """
        获取节点的三个最佳插入位置
        :param route: 路径列表
        :param node: 节点
        :param node_index: 节点索引
        :return: 最佳插入位置列表
        """
        insertion_positions = []
        for i in range(1, len(route) - 1):
            if i != node_index:
                cost = self.calculate_insertion_cost(route, node, i)
                insertion_positions.append((i, cost))
        insertion_positions.sort(key=lambda x: x[1])
        return insertion_positions[:3]

    def calculate_insertion_cost(self, route, node, position):
        """
        计算节点插入到指定位置的成本
        :param route: 路径列表
        :param node: 节点
        :param position: 位置
        :return: 插入成本
        """
        prev_node = route[position - 1]
        next_node = route[position]
        insertion_cost = self.instance.distance_matrix[prev_node][node] + self.instance.distance_matrix[node][next_node] - self.instance.distance_matrix[prev_node][next_node]
        return insertion_cost
