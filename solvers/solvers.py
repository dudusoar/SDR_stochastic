from solution_v2 import Solution

class GreedyInsertion:
    def __init__(self, params):
        self.params = params
        self.solution = Solution(params)

    def solve(self):
        """
        Solve the PDPTW problem using the greedy insertion algorithm.

        Returns:
            list: The routes for each vehicle.
        """
        # 按最早服务时间对订单进行排序
        sorted_orders = sorted(range(1, self.params['n'] + 1), key=lambda x: self.params['e_i'][x])

        for order in sorted_orders:
            pickup, delivery = order, order + self.params['n']

            # 尝试插入订单
            assigned = False
            for k in range(self.solution.v):
                for i in range(len(self.solution.routes[k]) + 1):
                    if self.can_insert(k, i, pickup, delivery):
                        self.insert(k, i, pickup, delivery)
                        assigned = True
                        break
                if assigned:
                    break

            # 如果无法插入,尝试延后交货时间
            if not assigned:
                for k in range(self.solution.v):
                    for i in range(len(self.solution.routes[k]) + 1):
                        if self.can_insert_late(k, i, pickup, delivery):
                            self.insert(k, i, pickup, delivery)
                            assigned = True
                            break
                    if assigned:
                        break

            # 如果仍无法插入,则将该订单标记为未分配
            if not assigned:
                print(f"Order {order} cannot be assigned to any vehicle.")

        return self.solution.routes

    def can_insert(self, vehicle_idx, insert_idx, pickup, delivery):
        """
        Check if an order can be inserted at the given location on the given vehicle.
        """
        return self.solution.check_vehicle_feasibility(vehicle_idx, pickup, delivery)

    def can_insert_late(self, vehicle_idx, insert_idx, pickup, delivery):
        """
        Check if an order can be inserted by delaying the delivery time.
        """
        route = self.solution.routes[vehicle_idx]

        # 检查取货点是否满足时间窗约束
        earliest_pickup_time = max(self.params['e_i'][pickup],
                                   self.solution.arrival_times[vehicle_idx][insert_idx - 1] +
                                   self.params['distance_matrix'][route[insert_idx - 1]][pickup])
        if earliest_pickup_time > self.params['l_i'][pickup]:
            return False

        # 检查在取货点之后的节点是否满足时间窗约束
        earliest_delivery_time = earliest_pickup_time + self.params['distance_matrix'][pickup][delivery]
        for j in range(insert_idx, len(route)):
            node = route[j]
            earliest_time = max(self.params['e_i'][node],
                                earliest_delivery_time + self.params['distance_matrix'][delivery][node])
            if earliest_time > self.params['l_i'][node]:
                return False
            earliest_delivery_time = earliest_time

        # 检查电量和载重约束
        battery = self.solution.battery_levels[vehicle_idx][insert_idx - 1] - self.params['b'] * \
                  self.params['distance_matrix'][route[insert_idx - 1]][pickup]
        capacity = self.solution.capacities[vehicle_idx][insert_idx - 1] + self.params['p_i'][pickup]
        if battery < self.params['b'] * self.params['distance_matrix'][pickup][delivery] or capacity > self.params['Q']:
            return False
        battery -= self.params['b'] * self.params['distance_matrix'][pickup][delivery]
        capacity += self.params['p_i'][delivery]
        for j in range(insert_idx, len(route)):
            node = route[j]
            if battery < self.params['b'] * self.params['distance_matrix'][delivery][node] or capacity > self.params[
                'Q']:
                return False
            battery -= self.params['b'] * self.params['distance_matrix'][delivery][node]
            capacity += self.params['p_i'][node]

        return True

    def insert(self, vehicle_idx, insert_idx, pickup, delivery):
        """
        Insert an order at the given location on the given vehicle.
        """
        self.solution.routes[vehicle_idx].insert(insert_idx, pickup)
        self.solution.routes[vehicle_idx].insert(insert_idx + 1, delivery)

        self.solution.update_solution_state()