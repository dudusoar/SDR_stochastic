from solution import PDPTWSolution
import random 
import numpy as np
from copy import deepcopy
from collections import defaultdict
from operators import RemovalOperators, RepairOperators, ReverseOperators

# 初始解
def greedy_insertion_init(instance, num_vehicles, vehicle_capacity, battery_capacity, battery_consume_rate):
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
                                                  [new_route])

                    if temp_solution.is_feasible():
                        objective_value = temp_solution.objective_function()
                        if objective_value < best_objective_value:
                            best_pickup_node = pickup_node
                            best_insertion_index = insertion_index
                            best_objective_value = objective_value

            if best_pickup_node is not None:
                pickup_nodes.remove(best_pickup_node)
                route = route[:best_insertion_index] + [best_pickup_node, best_pickup_node + instance.n] + route[
                                                                                                           best_insertion_index:]
            else:
                break

        routes.append(route)

    solution = PDPTWSolution(instance, vehicle_capacity, battery_capacity, battery_consume_rate, routes)

    return solution


# ALNS
class ALNS:
    def __init__(self, initial_solution, 
                 removal_operators, repair_operators, 
                 max_iterations, max_no_improve, segment_length, 
                 r, weight_update_interval, sigma,
                 start_temp, cooling_rate,):
        # solution
        self.current_solution = initial_solution  
        self.best_solution = deepcopy(initial_solution) 
        # Removal and Repair operators
        self.removal_operators = removal_operators 
        self.repair_operators = repair_operators
        # Parameters for ALNS
        self.max_iterations = max_iterations 
        self.max_no_improve = max_no_improve 
        self.weight_update_interval = weight_update_interval  
        self.segment_length = segment_length  
        self.r = r
        # Scores
        self.sigma1 = sigma[0]
        self.sigma2 = sigma[1] 
        self.sigma3 = sigma[2] 
        # Acceptance criteria
        self.start_temp = start_temp 
        self.cooling_rate = cooling_rate  
        self.current_temp = start_temp  
        # Visited solutions
        self.visited_solutions = set() 

        # initialization
        self.removal_weights = np.ones(len(removal_operators))
        self.repair_weights = np.ones(len(repair_operators))
        self.removal_scores = np.zeros(len(removal_operators))
        self.repair_scores = np.zeros(len(repair_operators))
    
    def select_operators(self,weights):
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
        return len(weights) - 1 # select the last one
    
    def update_weight(self):
        '''
        At the end of each segement
        '''
        pass

    def acceptance_criterion(self, new_solution):
        pass


    def run(self):
        '''
        main function
        :return: the best solution
        '''
        pass




        