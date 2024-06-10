from solution import PDPTWSolution

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


# 其他算法
# 2-opt
class TwoOptSolver:
    def __init__(self, initial_solution, max_iterations=100):
        """
        初始化 TwoOptSolver 对象
        :param initial_solution: 初始解（PDPTWSolution 对象）
        :param max_iterations: 最大迭代次数
        """
        self.initial_solution = initial_solution
        self.max_iterations = max_iterations
        self.best_solution = None

    def solve(self):
        """
        运行 2-opt 算法求解 PDPTW 问题
        :return: 最优解（PDPTWSolution 对象）
        """
        self.best_solution = self.initial_solution
        current_solution = self.initial_solution

        for _ in range(self.max_iterations):
            improved = False

            for vehicle_id in range(current_solution.num_vehicles):
                route = current_solution.routes[vehicle_id]
                if len(route) <= 4:
                    continue
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        new_route = self.two_opt_swap(route, i, j)
                        new_routes = current_solution.routes.copy()
                        new_routes[vehicle_id] = new_route
    
                        new_solution = PDPTWSolution(current_solution.instance, current_solution.vehicle_capacity,
                                                     current_solution.battery_capacity, current_solution.battery_consume_rate,
                                                     new_routes)
        
                        if new_solution.is_feasible() and new_solution.objective_function() < self.best_solution.objective_function():
                            self.best_solution = new_solution
                            current_solution = new_solution
                            improved = True
                            break

                    if improved:
                        break

            if not improved:
                break

        return self.best_solution

    @staticmethod
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
