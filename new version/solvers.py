from operators import RemovalOperators, RepairOperators
import matplotlib.pyplot as plt

def greedy_insertion_init(instance, num_vehicles, vehicle_capacity, battery_capacity, battery_consume_rate, penalty_unvisit, penalty_delay):
    """
    使用贪心插入法构建初始解
    :param instance: PDPTWInstance 对象
    :param num_vehicles: 车辆数量
    :param vehicle_capacity: 车辆容量
    :param battery_capacity: 电池容量
    :param battery_consume_rate: 电池消耗率
    :return: 初始解
    """
    routes = []
    pickup_nodes = list(range(1, instance.n + 1))
    pickup_nodes.sort(key=lambda x: instance.time_windows[x][0])  # 按照 pickup 点的开始时间排序

    for vehicle_id in range(num_vehicles):
        route = [0, 0]

        while pickup_nodes:
            best_pickup_node = None
            best_insertion_index = None
            best_objective_value = float('inf')

            for pickup_node in pickup_nodes:
                delivery_node = pickup_node + instance.n
                for insertion_index in range(1, len(route)):
                    new_route = route
                    new_route = route[:insertion_index] + [pickup_node, delivery_node] + route[insertion_index:]
                    temp_solution = PDPTWSolution(instance, vehicle_capacity, battery_capacity, battery_consume_rate,
                                                  [new_route],penalty_unvisit, penalty_delay)

                    if temp_solution.is_feasible():
                        objective_value = temp_solution.objective_function()
                        if objective_value < best_objective_value:
                            best_pickup_node = pickup_node
                            best_insertion_index = insertion_index
                            best_objective_value = objective_value

            if best_pickup_node is not None:
                pickup_nodes.remove(best_pickup_node)
                route = route[:best_insertion_index] + [best_pickup_node, best_pickup_node + instance.n] + route[best_insertion_index:]
            else:
                break

        routes.append(route)

    solution = PDPTWSolution(instance, vehicle_capacity, battery_capacity, battery_consume_rate, routes, penalty_unvisit, penalty_delay)

    return solution


from solution import PDPTWSolution
import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
import time


class ALNS:
    def __init__(self, initial_solution,
                 params_operators, d_matrix, dist_matrix, battery,
                 max_no_improve, segment_length, num_segments, r, sigma,
                 start_temp, cooling_rate):

        # Solution
        self.current_solution = deepcopy(initial_solution)
        self.best_solution = deepcopy(initial_solution)
        self.charging_solution = deepcopy(initial_solution)
        self.charging_solution.battery_capacity = battery * 2 / self.current_solution.instance.speed * 60

        self.best_charging_route = []

        # Parameters for Operators
        self.num_removal = params_operators['num_removal']
        self.p = params_operators['p']
        self.k = params_operators['k']
        self.L_max = params_operators['L_max']
        self.avg_remove_order = params_operators['avg_remove_order']

        self.d_matrix = d_matrix
        self.dist_matrix = dist_matrix
        self.battery = battery

        # Parameters for ALNS
        self.max_no_improve = max_no_improve
        self.segment_length = segment_length
        self.num_segments = num_segments
        # self.max_iterations = segment_length * num_segments
        self.r = r
        self.sigma1 = sigma[0]
        self.sigma2 = sigma[1]
        self.sigma3 = sigma[2]

        # Acceptance criteria
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.current_temp = start_temp

        # ======== Initialization============
        # Methods list
        # self.removal_list = [0, 1, 2, 3]
        self.removal_list = [0, 2, 3]
        self.repair_list = [0, 1]

        # Weights
        self.removal_weights = np.zeros((num_segments, len(self.removal_list)))
        self.repair_weights = np.zeros((num_segments, len(self.repair_list)))
        self.removal_weights[0] = np.ones(len(self.removal_list)) / len(self.removal_list)
        self.repair_weights[0] = np.ones(len(self.repair_list)) / len(self.repair_list)

        # Scores
        self.removal_scores = np.zeros((num_segments, len(self.removal_list)))
        self.repair_scores = np.zeros((num_segments, len(self.repair_list)))

        # Theta: the number of times we have attempted to use heuristic i in every segment
        self.removal_theta = np.zeros((num_segments, len(self.removal_list)))
        self.repair_theta = np.zeros((num_segments, len(self.repair_list)))

    def select_operator(self, weights):
        '''
        Select the heuristic algorithms using roulette wheel selection principle
        The insertion heuristic is selected independently of the removal heuristic

        :param weights: weight lists for different operators
        :return: index for the selected operator
        '''
        total_weight = np.sum(weights)
        probabilities = weights / total_weight
        cumulative_probabilities = np.cumsum(probabilities)
        random_number = random.random()
        for i, probability in enumerate(cumulative_probabilities):
            if random_number < probability:
                return i
        return len(weights) - 1  # select the last one

    def total_distance(self, route):
        dist = 0
        for i in range(len(route) - 1):
            dist += self.dist_matrix[route[i]][route[i + 1]]
        return dist

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

        cost_ci_best = float('inf')  # recording cost after chargning insertion
        cost_ci_obj_diff = 100
        cost_ci_best_list = []

        # while segment < self.num_segments and num_no_improve < self.max_no_improve:
        # while segment < self.num_segments and best_obj_diff > 0.00001 and cost_ci_obj_diff > 0.00001:
        while segment < self.num_segments and cost_ci_obj_diff > 0.00001:
            # (time and information)
            segment_start_time = time.time()
            print(f"Segment {segment + 1} / {self.num_segments}")

            # =========== A new segment begins ===========
            # Update the weights for the current segment
            if segment > 0:
                for i in range(len(self.removal_list)):
                    self.removal_weights[segment][i] = self.removal_weights[segment - 1][i] * (1 - r) \
                                                       + r * self.removal_scores[segment - 1][i] / max(1,self.removal_theta[segment - 1][i])
                for i in range(len(self.repair_list)):
                    self.repair_weights[segment][i] = self.repair_weights[segment - 1][i] * (1 - r) \
                                                      + r * self.repair_scores[segment - 1][i] / max(1,self.repair_theta[segment - 1][i])

            for iteration in range(self.segment_length):
                # ====== select the operators ======
                # removal
                removal_operators = RemovalOperators(self.current_solution)
                removal_idx = self.select_operator(self.removal_weights[segment])

                if removal_idx == 0:
                    removed_solution = removal_operators.shaw_removal(self.num_removal, self.p)
                elif removal_idx == 1:
                    removed_solution = removal_operators.random_removal(self.num_removal)
                elif removal_idx == 2:
                    removed_solution = removal_operators.worst_removal(self.num_removal)
                elif removal_idx == 3:
                    removed_solution = removal_operators.SISR_removal(self.L_max, self.avg_remove_order, self.d_matrix)

                removed_solution.update_all()
                unvisited_pairs = removed_solution.unvisited_pairs
                # print(unvisited_pairs)

                # repair
                repair_operators = RepairOperators(removed_solution)
                repair_idx = self.select_operator(self.repair_weights[segment])
                if repair_idx == 0:
                    repair_solution = repair_operators.greedy_insertion(unvisited_pairs)
                elif repair_idx == 1:
                    repair_solution = repair_operators.regret_insertion(unvisited_pairs, self.k)

                # print('repair method',repair_idx)
                # print('remove_routes',removed_solution.routes)
                # print('repair_routes',repair_solution.routes)

                # update the count
                self.removal_theta[segment][removal_idx] += 1
                self.repair_theta[segment][repair_idx] += 1

                # ====== update the scores ======
                repair_solution.update_all()
                new_objective = repair_solution.objective_function()

                current_objective = self.current_solution.objective_function()
                best_objective = self.best_solution.objective_function()

                if new_objective < best_objective:  # sigma1
                    self.best_solution = deepcopy(repair_solution)
                    self.current_solution = deepcopy(repair_solution)
                    num_no_improve = 0
                    self.removal_scores[segment][removal_idx] += self.sigma1
                    self.repair_scores[segment][repair_idx] += self.sigma1
                elif new_objective < current_objective:  # sigma2
                    self.current_solution = deepcopy(repair_solution)
                    num_no_improve = 0
                    self.removal_scores[segment][removal_idx] += self.sigma2
                    self.repair_scores[segment][repair_idx] += self.sigma2
                else:  # sigma3
                    acceptance_probability = np.exp(-(new_objective - current_objective) / self.current_temp)
                    if random.random() < acceptance_probability:
                        self.current_solution = deepcopy(repair_solution)
                        self.removal_scores[segment][removal_idx] += self.sigma3
                        self.repair_scores[segment][repair_idx] += self.sigma3
                    num_no_improve += 1
                # best_obj_list.append(best_objective)
                # if len(best_obj_list) >= 125:
                #     best_obj_list = best_obj_list[-125:]
                #     best_obj_diff = np.mean(best_obj_list) - best_objective
                # else:
                #     best_obj_diff = 100

                # Add charging insertion
                battery = self.battery
                if segment > 0:
                    z = 0
                    routes_charge = deepcopy(repair_solution.routes)
                    # routes_temp = deepcopy(self.current_solution.routes)
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
                                # routes_temp[route_id] = route_copy
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

            print(best_objective, best_obj_diff, cost_ci_best, cost_ci_obj_diff)

            # (time spent on this segment)
            segment_end_time = time.time()
            segment_duration = segment_end_time - segment_start_time
            print(f"Segment {segment + 1} completed in {segment_duration:.2f} seconds")

            # update the segment, temperature
            segment += 1
            self.current_temp *= self.cooling_rate

            # === End of the segment

        # (time spend on the whole process)
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"ALNS run completed in {total_duration:.2f} seconds")

        return self.best_solution, self.best_charging_solution

    def plot_scores(self):
        plt.figure(figsize=(12, 6))
        segments = range(self.removal_scores.shape[0])

        # removal scores
        for i in range(len(self.removal_list)):
            plt.plot(segments, self.removal_scores[:, i],
                     label=f'Shaw Removal' if i == 0 else (f'Random Removal' if i == 1 else 'Worst Removal'))

        # repair scores
        for i in range(len(self.repair_list)):
            plt.plot(segments, self.repair_scores[:, i], label=f'Greedy Insertion' if i == 0 else 'Regret Insertion')

        plt.xlabel('Segment')
        plt.ylabel('Scores')
        plt.title('Scores of Operators')
        plt.xticks(segments)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_theta(self):
        plt.figure(figsize=(12, 6))
        segments = range(self.removal_theta.shape[0])

        # removal theta
        for i in range(len(self.removal_list)):
            plt.plot(segments, self.removal_theta[:, i], label=f'Shaw Removal' if i == 0 else (
                f'Worst Removal' if i == 1 else f'SISR Removal' if i == 2 else 'SISR Removal'))

        # repair theta
        for i in range(len(self.repair_list)):
            plt.plot(segments, self.repair_theta[:, i], label=f'Greedy Insertion' if i == 0 else 'Regret Insertion')

        plt.xlabel('Segment')
        plt.ylabel('Theta (Usage Count)')
        plt.title('Usage Count of Operators')
        plt.xticks(segments)
        plt.legend()
        plt.grid(True)
        plt.show()
