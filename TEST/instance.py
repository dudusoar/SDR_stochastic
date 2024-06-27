import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PDPTWInstance:
    def __init__(self, n, map_size, speed, extra_time, gamma, seed=None):
        """
        初始化 PDPTW 实例
        :param n: pickup点的数量
        :param map_size: 地图大小
        :param speed: 车辆速度
        :param extra_time: delivery 点时间窗口起始时间的额外时间
        :param gamma: 未被服务请求的惩罚系数
        :param seed: 随机数种子
        """
        self.n = n
        self.map_size = map_size
        self.speed = speed
        self.extra_time = extra_time
        self.gamma = gamma
        # coordinates
        self.depot = (0, 0)  # depot 位于原点
        self.pickup_points = []  # pickup 点的坐标
        self.delivery_points = []  # delivery 点的坐标
        # time
        self.time_windows = []  # 时间窗口列表
        self.service_times = []  # 服务时间列表
        # demand
        self.demands = []  # 需求量列表

        # 设置随机数种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 生成所有点的索引列表
        self.indices = [0] + list(range(1, self.n + 1)) + list(range(self.n + 1, 2 * self.n + 1))

        self.generate_points()  # 生成 pickup 和 delivery 点
        self.distance_matrix = self.calculate_distance_matrix() # 距离矩阵
        self.time_matrix = self.calculate_time_matrix() # 时间矩阵
        self.generate_time_windows()  # 生成时间窗口和服务时间
        self.generate_demands()  # 生成需求量

    def generate_points(self):
        """
        生成 pickup 和 delivery 点
        """
        for _ in range(self.n):
            # 在地图范围内随机生成 pickup 点坐标
            pickup_x = random.uniform(-self.map_size, self.map_size)
            pickup_y = random.uniform(-self.map_size, self.map_size)
            self.pickup_points.append((pickup_x, pickup_y))

            # 在地图范围内随机生成 delivery 点坐标
            delivery_x = random.uniform(-self.map_size, self.map_size)
            delivery_y = random.uniform(-self.map_size, self.map_size)
            self.delivery_points.append((delivery_x, delivery_y))

    def plot_instance(self):
        """
        绘制 PDPTW 实例图
        """
        plt.figure(figsize=(8, 8))
        # 绘制 depot
        plt.scatter(self.depot[0], self.depot[1], c='red', marker='s', s=100, label='Depot')
        # 绘制 pickup 点
        plt.scatter([p[0] for p in self.pickup_points], [p[1] for p in self.pickup_points], c='blue', marker='o',
                    label='Pickup')
        # 绘制 delivery 点
        plt.scatter([d[0] for d in self.delivery_points], [d[1] for d in self.delivery_points], c='green', marker='d',
                    label='Delivery')

        # 为每个点添加标签
        for i in range(self.n):
            plt.annotate(f'P{i + 1}', (self.pickup_points[i][0], self.pickup_points[i][1]), textcoords='offset points',
                         xytext=(0, 5), ha='center')
            plt.annotate(f'D{i + 1}', (self.delivery_points[i][0], self.delivery_points[i][1]),
                         textcoords='offset points', xytext=(0, 5), ha='center')

        # 设置坐标轴范围
        plt.xlim(-self.map_size - 1, self.map_size + 1)
        plt.ylim(-self.map_size - 1, self.map_size + 1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PDPTW Instance')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_distance_matrix(self):
        """
        计算距离矩阵
        :return: 距离矩阵
        """
        points = [self.depot] + self.pickup_points + self.delivery_points
        num_points = len(points)
        distance_matrix = np.zeros((num_points, num_points))

        # 计算每对点之间的欧几里得距离
        for i in range(num_points):
            for j in range(num_points):
                distance_matrix[i][j] = np.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)

        return distance_matrix

    def calculate_time_matrix(self):
        """
        计算时间矩阵
        :return: 时间矩阵
        """
        time_matrix = (self.distance_matrix / self.speed) * 60
        return time_matrix

    def generate_time_windows(self):
        """
        生成时间窗口和服务时间
        """
        time_matrix = self.calculate_time_matrix()

        # Depot 的时间窗口设置为 [0, inf]
        self.time_windows.append((0, float('inf')))

        for i in range(self.n):
            # Pickup 点的起始时间在 [0, 120] 范围内随机生成，结束时间设置为 inf
            pickup_start_time = random.randint(0, 120)
            self.time_windows.append((pickup_start_time, float('inf')))

            # 每个 pickup 点和 delivery 点的服务时间在 [5, 10] 范围内随机生成一个整数
            self.service_times.append(random.randint(5, 10))

        for i in range(1, self.n + 1):
            pickup_start_time = self.time_windows[i][0]
            # Delivery 点的起始时间为对应 pickup 点的起始时间 + 时间矩阵中的时间 + 额外时间，向上取整
            # 结束时间为起始时间 + 30 分钟，向下取整
            delivery_start_time = math.ceil(pickup_start_time + time_matrix[i][i + self.n] + self.extra_time)
            delivery_end_time = math.floor(delivery_start_time + 60)
            self.time_windows.append((delivery_start_time, delivery_end_time))

            # 每个 pickup 点和 delivery 点的服务时间在 [5, 10] 范围内随机生成一个整数
            self.service_times.append(random.randint(5, 10))

        # Depot 的服务时间设置为 0
        self.service_times.append(0)

    def generate_demands(self):
        """
        生成需求量
        """
        # Depot 的需求量为 0
        self.demands.append(0)

        for _ in range(self.n):
            # Pickup 点的需求量为 1
            self.demands.append(1)
        for _ in range(self.n):
            # Delivery 点的需求量为 -1
            self.demands.append(-1)

    def to_dataframe(self):
        """
        将 PDPTW 实例转换为 pandas表
        :return: pandas表
        """
        data = []

        for i in range(len(self.indices)):
            if i == 0:
                point_type = 'd'
                x, y = self.depot
                partner_id = 0
            elif i <= self.n:
                point_type = 'cp'
                x, y = self.pickup_points[i - 1]
                partner_id = i + self.n
            else:
                point_type = 'cd'
                x, y = self.delivery_points[i - self.n - 1]
                partner_id = i - self.n

            data.append([
                i,
                point_type,
                x,
                y,
                self.demands[i],
                self.time_windows[i][0],
                self.time_windows[i][1],
                self.service_times[i],
                partner_id
            ])

        columns = ['ID', 'Type', 'X', 'Y', 'Demand', 'ReadyTime', 'DueDate', 'ServiceTime', 'PartnerID']
        df = pd.DataFrame(data, columns=columns)

        return df