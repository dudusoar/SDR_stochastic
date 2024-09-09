import numpy as np
import random
from copy import deepcopy

# 自定义异常类，用于处理找不到节点的情况
class NodeNotFoundError(Exception):
    def __init__(self, node):
        super().__init__(f"Node {node} not found in any route.")

# 移除操作类
class RemovalOperators:
    def __init__(self, solution):
        self.solution = solution
        self.instance = solution.instance
        self.df = self.instance.generate_whole_table()

    # ==== SISR (Sequential Insertion String Removal) 移除方法 ====
    def SISR_removal(self, L_max, avg_remove_order, d_matrix):
        """
        实现SISR移除算法
        :param L_max: 最大字符串长度
        :param avg_remove_order: 平均移除订单数
        :param d_matrix: 距离矩阵
        :return: 移除节点后的新解
        """
        remaining_orders = self.solution.unvisited_requests
        routes_copy = deepcopy(self.solution.routes)
        n_orders = len(self.df[self.df['Type'] == 'cp'])
        removed_list = []
        deconstructed_route_list = []
        k_s, l_s_max = self.number_of_strings(L_max, avg_remove_order, routes_copy, n_orders)
        k_s = min(k_s, len(routes_copy))

        for i in range(int(k_s)):
            if i == 0:
                start_order = self.order_to_start(n_orders, remaining_orders)
                route = self.find_route_containing_element(routes_copy, start_order)
                l_t = self.number_of_orders(l_s_max, route)
                routes_copy, removed_list, deconstructed_route_list, primal_sorted_indices = self.primal_string_removal(
                    d_matrix, routes_copy, route, l_t, start_order, n_orders, removed_list, deconstructed_route_list)
            elif not primal_sorted_indices:
                break
            else:
                route, next_order = self.find_next_route(primal_sorted_indices, routes_copy, remaining_orders)
                if not route:
                    break
                l_t = self.number_of_orders(l_s_max, route)
                routes_copy, removed_list, deconstructed_route_list, primal_sorted_indices = self.other_string_removal(
                    d_matrix, routes_copy, route, l_t, next_order, n_orders, removed_list, deconstructed_route_list,
                    primal_sorted_indices)

        remaining_routes = deconstructed_route_list + routes_copy
        return self.remove_requests(removed_list)

    # ==== 计算要移除的字符串数量 ====
    def number_of_strings(self, L_max, avg_remove_order, routes, n_orders):
        """
        计算要移除的字符串数量
        :return: 字符串数量和最大字符串长度
        """
        T_avg = np.floor(n_orders / len(routes))
        l_s_max = min(T_avg, L_max)
        k_s_max = 4 * avg_remove_order / (1 + l_s_max) - 1
        k_s = np.floor(np.random.uniform(1, k_s_max + 1))
        return k_s, l_s_max

    # ==== 计算要移除的订单数量 ====
    def number_of_orders(self, l_s_max, route):
        """
        计算要从路径中移除的订单数量
        :return: 要移除的订单数量
        """
        l_t_max = min(l_s_max, (len(route) - 2) / 2)
        l_t = np.floor(np.random.uniform(1, l_t_max + 1))
        return l_t

    # ==== 选择起始订单 ====
    def order_to_start(self, n_orders, remaining_orders):
        """
        随机选择一个起始订单
        :return: 起始订单的ID
        """
        pickup_ids = self.df[self.df['Type'] == 'cp']['ID'].tolist()
        while True:
            reference_order = random.choice(pickup_ids)
            if reference_order not in remaining_orders:
                return reference_order

    # ==== 查找包含特定元素的路径 ====
    def find_route_containing_element(self, routes, order):
        """
        在路径列表中查找包含特定订单的路径
        :return: 包含订单的路径，如果没找到则返回None
        """
        for route in routes:
            if order in route:
                return route
        return None

    # ==== 主要字符串移除 ====
    def primal_string_removal(self, d_matrix, routes, route, l_t, reference_order, n_orders, removed_list,
                              deconstructed_route_list):
        """
        执行主要的字符串移除操作
        :return: 更新后的路径、移除列表、解构路径列表和排序索引
        """
        reference_real_index = self.df.loc[self.df['ID'] == reference_order, 'RealIndex'].values[0]
        distances = d_matrix[reference_real_index]
        sorted_indices = np.argsort(distances).tolist()

        route_copy = deepcopy(route)
        a = 0
        for real_index in sorted_indices:
            if a >= l_t:
                break
            node_id = self.df.loc[self.df['RealIndex'] == real_index, 'ID'].values[0]
            if node_id in route_copy:
                route_copy.remove(node_id)
                removed_list.append(node_id)
                partner_id = self.df.loc[self.df['ID'] == node_id, 'PartnerID'].values[0]
                route_copy.remove(partner_id)
                removed_list.append(partner_id)
                a += 1

        routes.remove(route)
        deconstructed_route_list.append(route_copy)

        sorted_indices = [idx for idx in sorted_indices if
                          self.df.loc[self.df['RealIndex'] == idx, 'ID'].values[0] in [node for r in routes for node in
                                                                                       r]]

        return routes, removed_list, deconstructed_route_list, sorted_indices

    # ==== 查找下一个路径 ====
    def find_next_route(self, primal_sorted_indices, routes, remaining_orders):
        """
        查找下一个要处理的路径和订单
        :return: 下一个路径和订单，如果没找到则返回None, None
        """
        for real_index in primal_sorted_indices:
            next_order = self.df.loc[self.df['RealIndex'] == real_index, 'ID'].values[0]
            if next_order not in remaining_orders:
                for route in routes:
                    if next_order in route:
                        return route, next_order
        return None, None

    # ==== 其他字符串移除 ====
    def other_string_removal(self, d_matrix, routes, route, l_t, reference_order, n_orders, removed_list,
                             deconstructed_route_list, primal_sorted_indices):
        """
        执行其他的字符串移除操作
        :return: 更新后的路径、移除列表、解构路径列表和排序索引
        """
        reference_real_index = self.df.loc[self.df['ID'] == reference_order, 'RealIndex'].values[0]
        distances = d_matrix[reference_real_index]
        sorted_indices = np.argsort(distances).tolist()

        route_copy = deepcopy(route)
        a = 0
        for real_index in sorted_indices:
            if a >= l_t:
                break
            node_id = self.df.loc[self.df['RealIndex'] == real_index, 'ID'].values[0]
            if node_id in route_copy:
                route_copy.remove(node_id)
                removed_list.append(node_id)
                partner_id = self.df.loc[self.df['ID'] == node_id, 'PartnerID'].values[0]
                route_copy.remove(partner_id)
                removed_list.append(partner_id)
                a += 1

        routes.remove(route)
        deconstructed_route_list.append(route_copy)

        primal_sorted_indices = [idx for idx in primal_sorted_indices if
                                 self.df.loc[self.df['RealIndex'] == idx, 'ID'].values[0] in [node for r in routes for
                                                                                              node in r]]

        return routes, removed_list, deconstructed_route_list, primal_sorted_indices

    # ==== Shaw移除方法 ====
    def shaw_removal(self, num_remove, p):
        """
        实现Shaw移除算法
        :param num_remove: 要移除的请求数量
        :param p: Shaw移除的参数
        :return: 移除节点后的新解
        """
        removed_requests = []
        remaining_requests = list(self.solution.visited_requests)

        initial_request = random.choice(remaining_requests)
        removed_requests.append(initial_request)
        remaining_requests.remove(initial_request)

        max_distance = np.max(self.instance.distance_matrix)
        max_arrive_time = np.max([np.max(arrival_time) for arrival_time in self.solution.route_arrival_times])

        while len(removed_requests) < num_remove:
            last_removed = random.choice(removed_requests)
            L = [req for req in remaining_requests]
            L.sort(key=lambda req: self.calculate_similarity(last_removed, req, max_distance, max_arrive_time))

            y = random.random()
            selected_request = L[int(y ** p * len(L))]
            removed_requests.append(selected_request)
            remaining_requests.remove(selected_request)

        return self.remove_requests(removed_requests)

    # ==== 计算相似度 ====
    def calculate_similarity(self, req1, req2, max_distance, max_arrive_time):
        """
        计算两个请求之间的相似度
        :return: 相似度得分
        """
        pickup1, delivery1 = req1, self.df.loc[self.df['ID'] == req1, 'PartnerID'].values[0]
        pickup2, delivery2 = req2, self.df.loc[self.df['ID'] == req2, 'PartnerID'].values[0]

        pickup1_real_index = self.df.loc[self.df['ID'] == pickup1, 'RealIndex'].values[0]
        pickup2_real_index = self.df.loc[self.df['ID'] == pickup2, 'RealIndex'].values[0]
        delivery1_real_index = self.df.loc[self.df['ID'] == delivery1, 'RealIndex'].values[0]
        delivery2_real_index = self.df.loc[self.df['ID'] == delivery2, 'RealIndex'].values[0]

        dist_pickup = self.instance.distance_matrix[pickup1_real_index][pickup2_real_index] / max_distance
        dist_delivery = self.instance.distance_matrix[delivery1_real_index][delivery2_real_index] / max_distance

        arrival_time_pickup = (self.get_arrival_time(pickup1) - self.get_arrival_time(pickup2)) / max_arrive_time
        arrival_time_delivery = (self.get_arrival_time(delivery1) - self.get_arrival_time(delivery2)) / max_arrive_time

        return dist_pickup + dist_delivery + arrival_time_pickup + arrival_time_delivery

    # ==== 获取到达时间 ====
    def get_arrival_time(self, node):
        """
        获取节点的到达时间
        :return: 节点的到达时间
        """
        for vehicle_id, route in enumerate(self.solution.routes):
            if node in route:
                return self.solution.route_arrival_times[vehicle_id][route.index(node)]
        raise NodeNotFoundError(node)

    # ==== 随机移除方法 ====
    def random_removal(self, num_remove):
        """
        随机移除指定数量的请求
        :param num_remove: 要移除的请求数量
        :return: 移除节点后的新解
        """
        remaining_requests = list(self.solution.visited_requests)
        removed_requests = random.sample(remaining_requests, num_remove)
        return self.remove_requests(removed_requests)

    # ==== 最差移除方法 ====
    def worst_removal(self, num_remove):
        """
        移除对目标函数贡献最差的请求
        :param num_remove: 要移除的请求数量
        :return: 移除节点后的新解
        """
        remaining_requests = list(self.solution.visited_requests)
        contributions = [(req, self.calculate_contribution(req)) for req in remaining_requests]
        contributions.sort(key=lambda x: x[1], reverse=True)
        removed_requests = [req for req, _ in contributions[:num_remove]]
        return self.remove_requests(removed_requests)

    # ==== 计算请求对目标函数的贡献 ====
    def calculate_contribution(self, req):
        """
        计算单个请求对目标函数的贡献
        :return: 请求的贡献值
        """
        temp_solution = deepcopy(self.solution)
        pickup = req
        delivery = self.df.loc[self.df['ID'] == req, 'PartnerID'].values[0]

        for route in temp_solution.routes:
            if pickup in route:
                route.remove(pickup)
                route.remove(delivery)

        temp_solution.update_all()
        original_objective = self.solution.objective_function()
        new_objective = temp_solution.objective_function()

        return original_objective - new_objective

    # ==== 移除请求 ====
    def remove_requests(self, requests):
        """
        从解中移除指定的请求
        :param requests: 要移除的请求列表
        :return: 移除请求后的新解
        """
        new_solution = deepcopy(self.solution)

        for request in requests:
            pickup_node = request
            delivery_node = self.df.loc[self.df['ID'] == request, 'PartnerID'].values[0]
            for route in new_solution.routes:
                if pickup_node in route:
                    route.remove(pickup_node)
                    route.remove(delivery_node)

        new_solution.update_all()
        return new_solution


# 修复操作类
class RepairOperators:
    def __init__(self, solution):
        self.solution = deepcopy(solution)
        self.instance = solution.instance
        self.df = self.instance.generate_whole_table()

    # ==== 贪心插入方法 ====
    def greedy_insertion(self, removed_pairs):
        """
        使用贪心策略将移除的请求重新插入到解中
        :param removed_pairs: 被移除的请求列表
        :return: 修复后的解
        """
        for pickup in removed_pairs:
            delivery = self.df.loc[self.df['ID'] == pickup, 'PartnerID'].values[0]
            best_cost = float('inf')
            best_route = None
            best_insert_position = None
            temp_solution = deepcopy(self.solution)
            for vehicle_id, route in enumerate(self.solution.routes):
                for i in range(1, len(route)):
                    for j in range(i, len(route)):
                        temp_route = route[:i] + [pickup] + route[i:j] + [delivery] + route[j:]
                        temp_solution.routes[vehicle_id] = temp_route
                        temp_solution.update_all()

                        if temp_solution.is_feasible():
                            cost = temp_solution.objective_function()
                            if cost < best_cost:
                                best_cost = cost
                                best_route = vehicle_id
                                best_insert_position = (i, j)

            if best_route is not None and best_insert_position is not None:
                self.insert_single_request(pickup, delivery, best_route, best_insert_position)

        return self.solution

    # ==== 遗憾插入方法 ====
    def regret_insertion(self, removed_pairs, k):
        """
        使用遗憾值策略将移除的请求重新插入到解中
        :param removed_pairs: 被移除的请求列表
        :param k: 遗憾值计算中考虑的最佳插入位置数量
        :return: 修复后的解
        """
        while removed_pairs:
            insertion_costs = []
            for pickup in removed_pairs:
                delivery = self.df.loc[self.df['ID'] == pickup, 'PartnerID'].values[0]
                costs = []
                for vehicle_id, route in enumerate(self.solution.routes):
                    min_cost = float('inf')
                    temp_solution = deepcopy(self.solution)
                    for i in range(1, len(route)):
                        for j in range(i, len(route)):
                            temp_route = route[:i] + [pickup] + route[i:j] + [delivery] + route[j:]
                            temp_solution.routes[vehicle_id] = temp_route
                            temp_solution.update_all()

                            if temp_solution.is_feasible():
                                cost = temp_solution.objective_function()
                                if cost < min_cost:
                                    min_cost = cost
                                    best_i, best_j = i, j

                    if min_cost < float('inf'):
                        costs.append((min_cost, vehicle_id, best_i, best_j))
                costs.sort(key=lambda x: x[0])
                insertion_costs.append((pickup, costs))

            best_request = None
            best_route = None
            best_insert_position = None
            max_regret = float('-inf')

            for pickup, costs in insertion_costs:
                if len(costs) == 0:
                    removed_pairs.remove(pickup)
                    continue
                elif len(costs) < k:
                    best_request = pickup
                    best_route = costs[0][1]
                    best_insert_position = (costs[0][2], costs[0][3])
                    break
                else:
                    regret = sum(cost[0] for cost in costs[1:k]) - costs[0][0]
                    if regret > max_regret:
                        max_regret = regret
                        best_request = pickup
                        best_route = costs[0][1]
                        best_insert_position = (costs[0][2], costs[0][3])

            if best_request is not None and best_route is not None and best_insert_position is not None:
                removed_pairs.remove(best_request)
                delivery = self.df.loc[self.df['ID'] == best_request, 'PartnerID'].values[0]
                self.insert_single_request(best_request, delivery, best_route, best_insert_position)

        return self.solution

    # ==== 插入单个请求 ====
    def insert_single_request(self, pickup, delivery, vehicle_id, insert_position):
        """
        将单个请求插入到指定路径的指定位置
        :param pickup: 取货点ID
        :param delivery: 送货点ID
        :param vehicle_id: 车辆ID
        :param insert_position: 插入位置元组 (pickup_pos, delivery_pos)
        """
        i, j = insert_position
        self.solution.routes[vehicle_id] = (
                self.solution.routes[vehicle_id][:i] +
                [pickup] +
                self.solution.routes[vehicle_id][i:j] +
                [delivery] +
                self.solution.routes[vehicle_id][j:]
        )
        self.solution.update_all()