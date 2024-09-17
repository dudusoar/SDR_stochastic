import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PDPTWInstance:
    def __init__(self, order_info):
        # from order
        self.order_table = order_info.order_table.copy()
        self.robot_speed = order_info.robot_speed
        self.n = order_info.total_number_orders

        # 从order中提取index信息
        self.indices = self.order_table['ID'].tolist()

        # data
        self.demands = self._extract_demands()
        self.time_windows = self._extract_time_windows()
        self.service_times = self._extract_service_times()

        # matrix
        self.distance_matrix = self._create_distance_matrix(order_info.distance_matrix)
        self.time_matrix = self._create_time_matrix(order_info.time_matrix)

        # coordinates
        self.depot = self._extract_depot_coordinates()
        self.pickup_points = self._extract_pickup_coordinates()
        self.delivery_points = self._extract_delivery_coordinates()
        self.charging = self._extract_charging_coordinates()

    def _extract_demands(self):
        demands = [0] * (max(self.indices) + 1)
        for _, row in self.order_table.iterrows():
            demands[row['ID']] = row['Demand']
        return demands

    def _extract_time_windows(self):
        time_windows = [[0, float('inf')] for _ in range(max(self.indices) + 1)]
        for _, row in self.order_table.iterrows():
            time_windows[row['ID']] = [row['StartTime'], row['EndTime']]
        return time_windows

    def _extract_service_times(self):
        service_times = [0] * (max(self.indices) + 1)
        for _, row in self.order_table.iterrows():
            service_times[row['ID']] = row['ServiceTime']
        return service_times

    def _create_distance_matrix(self, real_distance_matrix):
        n = max(self.indices) + 1
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                real_i = self.order_table.loc[self.order_table['ID'] == i, 'RealIndex'].values[0]
                real_j = self.order_table.loc[self.order_table['ID'] == j, 'RealIndex'].values[0]
                distance_matrix[i][j] = real_distance_matrix[real_i][real_j]
        return distance_matrix

    def _create_time_matrix(self, real_time_matrix):
        n = max(self.indices) + 1
        time_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                real_i = self.order_table.loc[self.order_table['ID'] == i, 'RealIndex'].values[0]
                real_j = self.order_table.loc[self.order_table['ID'] == j, 'RealIndex'].values[0]
                time_matrix[i][j] = real_time_matrix[real_i][real_j]
        return time_matrix

    def _extract_depot_coordinates(self):
        depot_row = self.order_table[self.order_table['Type'] == 'depot'].iloc[0]
        return (depot_row['X'], depot_row['Y'])

    def _extract_pickup_coordinates(self):
        pickup_rows = self.order_table[self.order_table['Type'] == 'cp']
        return [(row['X'], row['Y']) for _, row in pickup_rows.iterrows()]

    def _extract_delivery_coordinates(self):
        delivery_rows = self.order_table[self.order_table['Type'] == 'cd']
        return [(row['X'], row['Y']) for _, row in delivery_rows.iterrows()]

    def _extract_charging_coordinates(self):
        charging_row = self.order_table[self.order_table['Type'] == 'charging'].iloc[0]
        return (charging_row['X'], charging_row['Y'])
    

