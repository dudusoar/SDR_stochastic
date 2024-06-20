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





        