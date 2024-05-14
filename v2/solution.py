class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.routes = [[instance.depot]] + [[] for _ in range(instance.num_vehicles)] + [[instance.destination]]
        self.initialize_solution()

    def initialize_solution(self):
        # 随机生成初始解
        pickup_points = self.instance.pickup_points.copy()
        delivery_points = self.instance.delivery_points.copy()
        random.shuffle(pickup_points)
        random.shuffle(delivery_points)
        for i in range(len(pickup_points)):
            vehicle_id = random.randint(1, self.instance.num_vehicles)
            self.routes[vehicle_id].append(pickup_points[i])
            self.routes[vehicle_id].append(delivery_points[i])

    def evaluate(self):
        # 计算解的目标函数值
        total_distance = 0
        for route in self.routes:
            route_distance = 0
            current_time = 0
            current_load = 0
            for i in range(len(route)):
                node = route[i]
                if i == 0:
                    route_distance += self.instance.distance_matrix[0][node]
                    current_time += self.instance.time_windows[node - 1][0]
                    current_load += self.instance.demands[node - 1]
                elif i == len(route) - 1:
                    route_distance += self.instance.distance_matrix[node][self.instance.destination]
                else:
                    route_distance += self.instance.distance_matrix[route[i - 1]][node]
                    if node in self.instance.pickup_points:
                        current_time += self.instance.time_windows[node - 1][0]
                        current_load += self.instance.demands[node - 1]
                    else:
                        current_time += self.instance.time_windows[node - 1][0]
                        current_load -= self.instance.demands[node - self.instance.num_pairs - 1]
                    if current_time < self.instance.time_windows[node - 1][0]:
                        current_time = self.instance.time_windows[node - 1][0]
                    elif current_time > self.instance.time_windows[node - 1][1]:
                        return float('inf')
            total_distance += route_distance
        return total_distance