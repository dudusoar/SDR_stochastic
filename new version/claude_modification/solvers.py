import numpy as np
import random
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from solution import PDPTWSolution
from operators import RemovalOperators, RepairOperators

def greedy_insertion_init(instance, num_vehicles, vehicle_capacity, battery_capacity, battery_consume_rate, penalty_unvisited, penalty_delayed):
    """
    使用贪心插入法构建初始解
    """
    routes = []
    df = instance.generate_whole_table()

    # 获取depot的ID
    depot_id = df[df['Type'] == 'depot']['ID'].values[0]

    pickup_nodes = df[df['Type'] == 'cp']['ID'].tolist()
    pickup_nodes.sort(key=lambda x: df.loc[df['ID'] == x, 'StartTime'].values[0])

    for vehicle_id in range(num_vehicles):
        route = [depot_id, depot_id]

        while pickup_nodes:
            best_pickup_node = None
            best_insertion_index = None
            best_objective_value = float('inf')

            for pickup_node in pickup_nodes:
                delivery_node = df.loc[df['ID'] == pickup_node, 'PartnerID'].values[0]
                for insertion_index in range(1, len(route)):
                    new_route = route[:insertion_index] + [pickup_node, delivery_node] + route[insertion_index:]
                    temp_solution = PDPTWSolution(instance, vehicle_capacity, battery_capacity, battery_consume_rate,
                                                  [new_route], penalty_unvisited, penalty_delayed)

                    if temp_solution.is_feasible():
                        objective_value = temp_solution.objective_function()
                        if objective_value < best_objective_value:
                            best_pickup_node = pickup_node
                            best_insertion_index = insertion_index
                            best_objective_value = objective_value

            if best_pickup_node is not None:
                pickup_nodes.remove(best_pickup_node)
                delivery_node = df.loc[df['ID'] == best_pickup_node, 'PartnerID'].values[0]
                route = route[:best_insertion_index] + [best_pickup_node, delivery_node] + route[best_insertion_index:]
            else:
                break

        routes.append(route)

    solution = PDPTWSolution(instance, vehicle_capacity, battery_capacity, battery_consume_rate, routes, penalty_unvisited, penalty_delayed)
    return solution

class ALNS:
    def __init__(self, initial_solution,
                 params_operators, d_matrix, dist_matrix, battery,
                 max_no_improve, segment_length, num_segments, r, sigma,
                 start_temp, cooling_rate):
        """
        初始化ALNS类
        :param initial_solution: 初始解
        :param params_operators: 操作符参数
        :param d_matrix: 距离矩阵
        :param dist_matrix: 距离矩阵（可能与d_matrix相同）
        :param battery: 电池容量
        :param max_no_improve: 最大不改进次数
        :param segment_length: 每个段的长度
        :param num_segments: 段的数量
        :param r: 权重调整参数
        :param sigma: 得分参数
        :param start_temp: 初始温度
        :param cooling_rate: 冷却率
        """
        # ==== 初始化解 ====
        self.current_solution = deepcopy(initial_solution)
        self.best_solution = deepcopy(initial_solution)
        self.charging_solution = deepcopy(initial_solution)
        self.charging_solution.battery_capacity = battery * 2 / self.current_solution.instance.robot_speed * 60

        self.best_charging_solution = None

        # ==== 初始化操作符参数 ====
        self.num_removal = params_operators['num_removal']
        self.p = params_operators['p']
        self.k = params_operators['k']
        self.L_max = params_operators['L_max']
        self.avg_remove_order = params_operators['avg_remove_order']

        self.d_matrix = d_matrix
        self.dist_matrix = dist_matrix
        self.battery = battery

        # ==== 初始化ALNS参数 ====
        self.max_no_improve = max_no_improve
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.r = r
        self.sigma1, self.sigma2, self.sigma3 = sigma

        # ==== 初始化模拟退火参数 ====
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.current_temp = start_temp

        # ==== 初始化操作符列表 ====
        self.removal_list = [0, 2, 3]  # 0: Shaw, 2: Worst, 3: SISR
        self.repair_list = [0, 1]  # 0: Greedy, 1: Regret

        # ==== 初始化权重、得分和使用次数 ====
        self.removal_weights = np.ones((num_segments, len(self.removal_list))) / len(self.removal_list)
        self.repair_weights = np.ones((num_segments, len(self.repair_list))) / len(self.repair_list)
        self.removal_scores = np.zeros((num_segments, len(self.removal_list)))
        self.repair_scores = np.zeros((num_segments, len(self.repair_list)))
        self.removal_theta = np.zeros((num_segments, len(self.removal_list)))
        self.repair_theta = np.zeros((num_segments, len(self.repair_list)))

    # ==== 选择操作符 ====
    def select_operator(self, weights):
        """
        使用轮盘赌选择原则选择启发式算法
        :param weights: 不同操作符的权重列表
        :return: 选中的操作符索引
        """
        return np.random.choice(len(weights), p=weights / np.sum(weights))

    # ==== 计算路径总距离 ====
    def total_distance(self, route):
        """
        计算给定路径的总距离
        :param route: 路径
        :return: 总距离
        """
        return sum(self.dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    # ==== ALNS主要运行逻辑 ====
    def run(self, seeding):
        random.seed(seeding)
        np.random.seed(seeding)
        num_no_improve = 0
        segment = 0
        r = self.r
        start_time = time.time()
        best_obj_diff = 100
        best_obj_list = []
        insert_index = len(self.dist_matrix) - 1

        cost_ci_best = float('inf')
        cost_ci_obj_diff = 100
        cost_ci_best_list = []

        while segment < self.num_segments and cost_ci_obj_diff > 0.00001:
            segment_start_time = time.time()
            print(f"Segment {segment + 1} / {self.num_segments}")

            if segment > 0:
                for i in range(len(self.removal_list)):
                    self.removal_weights[segment][i] = self.removal_weights[segment - 1][i] * (1 - r) \
                                                       + r * self.removal_scores[segment - 1][i] / max(1,
                                                                                                       self.removal_theta[
                                                                                                           segment - 1][
                                                                                                           i])
                for i in range(len(self.repair_list)):
                    self.repair_weights[segment][i] = self.repair_weights[segment - 1][i] * (1 - r) \
                                                      + r * self.repair_scores[segment - 1][i] / max(1,
                                                                                                     self.repair_theta[
                                                                                                         segment - 1][
                                                                                                         i])

            for iteration in range(self.segment_length):
                removal_operators = RemovalOperators(self.current_solution)
                removal_idx = self.select_operator(self.removal_weights[segment])

                # 初始化 removed_solution
                removed_solution = None

                try:
                    if removal_idx == 0:
                        removed_solution = removal_operators.shaw_removal(self.num_removal, self.p)
                    elif removal_idx == 1:
                        removed_solution = removal_operators.random_removal(self.num_removal)
                    elif removal_idx == 2:
                        removed_solution = removal_operators.worst_removal(self.num_removal)
                    elif removal_idx == 3:
                        removed_solution = removal_operators.SISR_removal(self.L_max, self.avg_remove_order,
                                                                          self.d_matrix)

                    if removed_solution is None:
                        print(f"Warning: No valid removal operation performed for index {removal_idx}")
                        continue

                    removed_solution.update_all()
                    unvisited_pairs = removed_solution.unvisited_pairs

                    repair_operators = RepairOperators(removed_solution)
                    repair_idx = self.select_operator(self.repair_weights[segment])

                    if repair_idx == 0:
                        repair_solution = repair_operators.greedy_insertion(unvisited_pairs)
                    elif repair_idx == 1:
                        repair_solution = repair_operators.regret_insertion(unvisited_pairs, self.k)
                    else:
                        print(f"Warning: Invalid repair index {repair_idx}")
                        continue

                    self.removal_theta[segment][removal_idx] += 1
                    self.repair_theta[segment][repair_idx] += 1

                    repair_solution.update_all()
                    new_objective = repair_solution.objective_function()

                    current_objective = self.current_solution.objective_function()
                    best_objective = self.best_solution.objective_function()

                    if new_objective < best_objective:
                        self.best_solution = deepcopy(repair_solution)
                        self.current_solution = deepcopy(repair_solution)
                        num_no_improve = 0
                        self.removal_scores[segment][removal_idx] += self.sigma1
                        self.repair_scores[segment][repair_idx] += self.sigma1
                    elif new_objective < current_objective:
                        self.current_solution = deepcopy(repair_solution)
                        num_no_improve = 0
                        self.removal_scores[segment][removal_idx] += self.sigma2
                        self.repair_scores[segment][repair_idx] += self.sigma2
                    else:
                        acceptance_probability = np.exp(-(new_objective - current_objective) / self.current_temp)
                        if random.random() < acceptance_probability:
                            self.current_solution = deepcopy(repair_solution)
                            self.removal_scores[segment][removal_idx] += self.sigma3
                            self.repair_scores[segment][repair_idx] += self.sigma3
                        num_no_improve += 1

                    # Add charging insertion
                    battery = self.battery
                    if segment > 0:
                        z = 0
                        routes_charge = deepcopy(repair_solution.routes)
                        for route_id, route_1 in enumerate(routes_charge):
                            route_best = []
                            if self.total_distance(route_1) <= battery:
                                z += 1
                                continue
                            c_best = float('inf')
                            for i in range(2, len(route_1) - 1):
                                route_copy = route_1[:i] + [insert_index] + route_1[i:]

                                if self.total_distance(route_copy) > 2 * battery:
                                    continue
                                else:
                                    self.charging_solution.routes[route_id] = route_copy
                                    self.charging_solution.update_all()
                                    cr_best = self.charging_solution.objective_function()
                                    if cr_best < c_best:
                                        second_index = route_copy.index(insert_index)
                                        subroute_1 = route_copy[:second_index + 1]
                                        subroute_2 = route_copy[second_index:]
                                        dist_1 = self.total_distance(subroute_1)
                                        dist_2 = self.total_distance(subroute_2)
                                        if (dist_1 <= battery) & (dist_2 <= battery):
                                            c_best = cr_best
                                            route_best = deepcopy(route_copy)
                                            routes_charge[route_id] = route_best
                            if len(route_best) > 0:
                                z += 1

                        if z == len(routes_charge):
                            self.charging_solution.routes = routes_charge
                            self.charging_solution.update_all()
                            cost_ci = self.charging_solution.objective_function()
                            if cost_ci < cost_ci_best:
                                cost_ci_best = cost_ci
                                self.best_charging_solution = deepcopy(self.charging_solution)
                        cost_ci_best_list.append(cost_ci_best)

                    if len(cost_ci_best_list) >= 25:
                        cost_ci_best_list = cost_ci_best_list[-25:]
                        cost_ci_obj_diff = np.mean(cost_ci_best_list) - cost_ci_best
                    else:
                        cost_ci_obj_diff = 100

                except Exception as e:
                    print(f"Error occurred during iteration: {e}")
                    continue

            print(best_objective, best_obj_diff, cost_ci_best, cost_ci_obj_diff)

            segment_end_time = time.time()
            segment_duration = segment_end_time - segment_start_time
            print(f"Segment {segment + 1} completed in {segment_duration:.2f} seconds")

            segment += 1
            self.current_temp *= self.cooling_rate

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"ALNS run completed in {total_duration:.2f} seconds")

        return self.best_solution, self.best_charging_solution
    # ==== 更新权重 ====
    def update_weights(self, segment):
        """
        更新移除和修复操作符的权重
        :param segment: 当前段
        """
        for i in range(len(self.removal_list)):
            self.removal_weights[segment][i] = self.removal_weights[segment - 1][i] * (1 - self.r) + self.r * \
                                               self.removal_scores[segment - 1][i] / max(1, self.removal_theta[segment - 1][i])
        for i in range(len(self.repair_list)):
            self.repair_weights[segment][i] = self.repair_weights[segment - 1][i] * (1 - self.r) + self.r * \
                                              self.repair_scores[segment - 1][i] / max(1,self.repair_theta[segment - 1][i])

    # ==== 充电站插入 ====
    def charging_insertion(self, solution):
        """
        在路径中插入充电站
        :param solution: 当前解
        :return: 插入充电站后的目标函数值和更新后的路径
        """
        routes_charge = deepcopy(solution.routes)
        insert_index = len(self.dist_matrix) - 1
        battery = self.battery

        for route_id, route in enumerate(routes_charge):
            if self.total_distance(route) <= battery:
                continue

            best_route = []
            best_cost = float('inf')
            for i in range(2, len(route) - 1):
                route_copy = route[:i] + [insert_index] + route[i:]
                if self.total_distance(route_copy) > 2 * battery:
                    continue

                self.charging_solution.routes[route_id] = route_copy
                self.charging_solution.update_all()
                current_cost = self.charging_solution.objective_function()

                if current_cost < best_cost:
                    second_index = route_copy.index(insert_index)
                    subroute_1 = route_copy[:second_index + 1]
                    subroute_2 = route_copy[second_index:]
                    if self.total_distance(subroute_1) <= battery and self.total_distance(subroute_2) <= battery:
                        best_cost = current_cost
                        best_route = route_copy

            if best_route:
                routes_charge[route_id] = best_route

        self.charging_solution.routes = routes_charge
        self.charging_solution.update_all()
        return self.charging_solution.objective_function(), routes_charge

    # ==== 绘制得分 ====
    def plot_scores(self):
        """
        绘制各操作符的得分
        """
        plt.figure(figsize=(12, 6))
        segments = range(self.removal_scores.shape[0])

        for i, name in enumerate(['Shaw Removal', 'Worst Removal', 'SISR Removal']):
            plt.plot(segments, self.removal_scores[:, i], label=name)

        for i, name in enumerate(['Greedy Insertion', 'Regret Insertion']):
            plt.plot(segments, self.repair_scores[:, i], label=name)

        plt.xlabel('Segment')
        plt.ylabel('Scores')
        plt.title('Scores of Operators')
        plt.xticks(segments)
        plt.legend()
        plt.grid(True)
        plt.show()

    # ==== 绘制使用次数 ====
    def plot_theta(self):
        """
        绘制各操作符的使用次数
        """
        plt.figure(figsize=(12, 6))
        segments = range(self.removal_theta.shape[0])

        for i, name in enumerate(['Shaw Removal', 'Worst Removal', 'SISR Removal']):
            plt.plot(segments, self.removal_theta[:, i], label=name)

        for i, name in enumerate(['Greedy Insertion', 'Regret Insertion']):
            plt.plot(segments, self.repair_theta[:, i], label=name)

        plt.xlabel('Segment')
        plt.ylabel('Theta (Usage Count)')
        plt.title('Usage Count of Operators')
        plt.xticks(segments)
        plt.legend()
        plt.grid(True)
        plt.show()