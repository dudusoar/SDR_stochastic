class Solution:
    def __init__(self, params):
        self.params = params
        self.n = params['n']  # 订单数量
        self.v = params['v']  # 车辆数量

        # 初始化解决方案的数据结构
        self.routes = [[] for _ in range(self.v)]  # 每辆车的路线,为一个嵌套列表
        self.arrival_times = [[0] * (2 * self.n + 3) for _ in range(self.v)]  # 每辆车在每个点的到达时间
        self.battery_levels = [[params['B']] * (2 * self.n + 3) for _ in range(self.v)]  # 每辆车在每个点的电量
        self.capacities = [[0] * (2 * self.n + 3) for _ in range(self.v)]  # 每辆车在每个点的载货量

        # 评估指标
        self.total_time = 0  # 总行驶时间
        self.total_delay = 0  # 总延迟时间
        self.feasible = True  # 解的可行性


    def update_solution_state(self):
        """根据当前的车辆路径更新arrival_times, battery_levels和capacities"""
        self.arrival_times = [[0] * (2 * self.n + 3) for _ in range(self.v)]
        self.battery_levels = [[self.params['B']] * (2 * self.n + 3) for _ in range(self.v)]
        self.capacities = [[0] * (2 * self.n + 3) for _ in range(self.v)]

        for k in range(self.v):
            route = self.routes[k]
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                node = route[i]

                # 更新到达时间
                self.arrival_times[k][i] = max(self.params['e_i'][node],
                                               self.arrival_times[k][i - 1] + self.params['time_matrix'][prev_node][
                                                   node])

                # 更新电量水平
                self.battery_levels[k][i] = self.battery_levels[k][i - 1] - self.params['b'] * \
                                            self.params['time_matrix'][prev_node][node]

                # 更新载货量
                self.capacities[k][i] = self.capacities[k][i - 1] + self.params['p_i'][node]
                
    def evaluate(self):
        """评估当前解的目标函数值"""
        # 计算总行驶时间
        self.total_time = 0
        for k in range(self.v):
            route = self.routes[k]
            for i in range(len(route) - 1):
                self.total_time += self.params['time_matrix'][route[i]][route[i + 1]]

        # 计算总延迟时间
        self.total_delay = 0
        for k in range(self.v):
            for i in self.params['D']:
                if i in self.routes[k]:
                    idx = self.routes[k].index(i)
                    if self.arrival_times[k][idx] > self.params['l_i'][i]:
                        self.total_delay += self.arrival_times[k][idx] - self.params['l_i'][i]

        # 检查解的可行性
        self.check_feasibility()

        # 计算目标函数值
        obj_value = self.total_time + self.total_delay
        return obj_value

    def check_feasibility(self):
        """检查当前解是否满足所有约束条件"""
        # 是否只被服务一次
        self.constr_served_only_once()
        if not self.feasible:
            return
        # 车辆行为约束
        self.constr_vehicle_behavior()
        if not self.feasible:
            return
        # 时间约束
        self.constr_time()
        if not self.feasible:
            return
        # 容量约束
        self.constr_capacity()
        if not self.feasible:
            return

        # 电池容量约束
        self.constr_battery()
        if not self.feasible:
            return

    # ======================== served only once  ========================
    def constr_served_only_once(self):
        """检查约束1:每个在params['C']中的点只能被服务一次"""
        visited = {node: 0 for node in self.params['C']}

        for route in self.routes:
            for node in route:
                if node in self.params['C']:
                    visited[node] += 1
                    if visited[node] > 1:
                        self.feasible = False
                        return

        self.feasible = True

    # ======================== vehicle behavior  ========================
    def constr_vehicle_behavior(self):
        """检查约束2-5:车辆行为"""
        self.constr_vehicle_behavior_1()
        if not self.feasible:
            return
        self.constr_vehicle_behavior_2()
        if not self.feasible:
            return
        self.constr_vehicle_behavior_3()
        if not self.feasible:
            return
        self.constr_vehicle_behavior_4()

    def constr_vehicle_behavior_1(self):
        """约束2:车辆如果被选择,必须从depot出发,并返回destination"""
        for k in range(self.v):
            route = self.routes[k]
            if len(route) > 0:  # 车辆k被选择
                if route[0] != 0 or route[-1] != 2 * self.n + 1:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_vehicle_behavior_2(self):
        """约束3:车辆服务完一个点,必须从该点离开"""
        for k in range(self.v):
            route = self.routes[k]
            visited = {node: 0 for node in self.params['N']}
            for i in range(len(route)):
                node = route[i]
                visited[node] += 1
                if i < len(route) - 1:  # 不是路线的最后一个点
                    next_node = route[i + 1]
                    if node != next_node:  # 下一个点不是当前点
                        visited[node] -= 1
            if any(visited[node] != 0 for node in self.params['N']):
                self.feasible = False
                return
        self.feasible = True

    def constr_vehicle_behavior_3(self):
        """约束4:如果车从i点pickup了,那么它一定要去对应的点(i+n)配送"""
        for k in range(self.v):
            route = self.routes[k]
            pickup_nodes = [node for node in route if node in self.params['P']]
            delivery_nodes = [node for node in route if node in self.params['D']]
            if set(pickup_nodes) != {node - self.n for node in delivery_nodes}:
                self.feasible = False
                return
        self.feasible = True

    def constr_vehicle_behavior_4(self):
        """约束5:一辆车不能先去delivery点再去pick up点"""
        for k in range(self.v):
            route = self.routes[k]
            pickup_nodes = [node for node in route if node in self.params['P']]
            delivery_nodes = [node for node in route if node in self.params['D']]
            if len(pickup_nodes) != len(delivery_nodes):
                self.feasible = False
                return
            for i in range(len(pickup_nodes)):
                if route.index(pickup_nodes[i]) > route.index(delivery_nodes[i]):
                    self.feasible = False
                    return
        self.feasible = True
    # ======================== capacity constraints  ========================
    def constr_capacity(self):
        """约束6-8:容量约束"""
        self.constr_capacity_1()
        if not self.feasible:
            return
        self.constr_capacity_2()
        if not self.feasible:
            return
        self.constr_capacity_3()

    def constr_capacity_1(self):
        """约束6:depot和destination的capacity level一定是params['Q']"""
        for k in range(self.v):
            route = self.routes[k]
            if len(route) > 0:  # 车辆k被选择
                if self.capacities[k][0] != self.params['Q'] or self.capacities[k][-1] != self.params['Q']:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_capacity_2(self):
        """约束7:每个点的capacity level一定小于或等于最大容量params['Q']且大于等于0"""
        for k in range(self.v):
            route = self.routes[k]
            for i in range(len(route)):
                node = route[i]
                if self.capacities[k][i] > self.params['Q'] or self.capacities[k][i] < 0:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_capacity_3(self):
        """
        约束8:每经过一个pick up的点,capacity level减少params['p_i']中对应点的需求量,
        每经过一个delivery的点,capacity level应该相应的增加
        注意delivery point的params['p_i'][node]是负数
        """
        for k in range(self.v):
            route = self.routes[k]
            for i in range(1, len(route)):
                node = route[i]
                prev_node = route[i - 1]
                if node in self.params['P']:  # pick up点
                    if self.capacities[k][i] != self.capacities[k][i - 1] - self.params['p_i'][node]:
                        self.feasible = False
                        return
                elif node in self.params['D']:  # delivery点
                    if self.capacities[k][i] != self.capacities[k][i - 1] - self.params['p_i'][node]:
                        # 注意这里params['p_i'][node]是负数
                        self.feasible = False
                        return
                else:  # 其他点(depot, destination, charging station)
                    if self.capacities[k][i] != self.capacities[k][i - 1]:
                        self.feasible = False
                        return
        self.feasible = True

    # ======================== time constraints  ========================
    def constr_time(self):
        """约束9-10:时间约束"""
        self.constr_time_1()
        if not self.feasible:
            return
        self.constr_time_2()

    def constr_time_1(self):
        """约束9:车辆的达到时间必须晚于e_i"""
        for k in range(self.v):
            route = self.routes[k]
            for i in range(len(route)):
                node = route[i]
                if self.arrival_times[k][i] > self.params['e_i'][node]:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_time_2(self):
        """约束10:路径中前一个点的达到时间肯定早于路径中后一个点的到达时间"""
        for k in range(self.v):
            route = self.routes[k]
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                node = route[i]
                if self.arrival_times[k][i - 1] + self.params['time_matrix'][prev_node][node] <= self.arrival_times[k][i]:
                    self.feasible = False
                    return
        self.feasible = True


    # ======================== battery constraints  ========================
    def constr_battery(self):
        """约束11-13:电量约束"""
        self.constr_battery_1()
        if not self.feasible:
            return
        self.constr_battery_2()
        if not self.feasible:
            return
        self.constr_battery_3()

    def constr_battery_1(self):
        """约束11:车在depot的电量肯定是满的,即params['B']"""
        for k in range(self.v):
            route = self.routes[k]
            if len(route) > 0:  # 车辆k被选择
                if self.battery_levels[k][0] != self.params['B']:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_battery_2(self):
        """约束12:车辆在路径中上一个点的Battery level一定大于等于在下一个点的Battery level + 车辆行驶途中的耗电量"""
        for k in range(self.v):
            route = self.routes[k]
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                node = route[i]
                if self.battery_levels[k][i - 1] < self.battery_levels[k][i] + self.params['b'] * \
                        self.params['time_matrix'][prev_node][node]:
                    self.feasible = False
                    return
        self.feasible = True

    def constr_battery_3(self):
        """约束13:每个点的battery level都应该在0和params['B']之间"""
        for k in range(self.v):
            route = self.routes[k]
            for i in range(len(route)):
                if self.battery_levels[k][i] < 0 or self.battery_levels[k][i] > self.params['B']:
                    self.feasible = False
                    return
        self.feasible = True


