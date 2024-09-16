import math
import pandas as pd
import numpy as np

class PDPTWInstance:
    def __init__(self, real_map, demand_table, time_params, robot_speed):
        """
        Initialize the PDPTWInstance.
        :param real_map: the instance of RealMap class
        :param time_params: Dictionary containing time-related parameters (time_window_length, service_time, extra_time).
        :param demand_table: DataFrame with demand information per time interval.
        :param robot_speed: Speed of the robot in distance per minute.
        """
        self.real_coordinates = real_map.coordinates
        self.distance_matrix = real_map.distance_matrix
        self.node_type_dict = real_map.node_type_dict
        self.demand_table = demand_table
        self.time_window_length = time_params['time_window_length']
        self.service_time = time_params['service_time']
        self.extra_time = time_params['extra_time']

        # Calculate total number of orders
        self.total_number_orders = self.demand_table.iloc[:, 2:].sum().sum()

        # time_matrix
        self.robot_speed = robot_speed
        self.time_matrix = self.generate_time_matrix()

        # data


    def generate_time_matrix(self):
        '''Assume time in minutes'''
        time_matrix = (self.distance_matrix / self.robot_speed) * 60
        return time_matrix

    def generate_whole_table(self):
        """
        Generate the complete table with all required information.
        :return: Pandas DataFrame containing all necessary details.
        """
        data = []
        count = 0

        for j in range(2, self.demand_table.shape[1]):  # Columns excluding 'Pickup' and 'Delivery'
            time_start = int(self.demand_table.columns[j].split('-')[0])  # Start time of the interval
            for i in range(self.demand_table.shape[0]):
                orders_count = self.demand_table.iloc[i, j]
                if orders_count > 0:
                    pickup_real_index = self.demand_table.iloc[i, 0]
                    delivery_real_index = self.demand_table.iloc[i, 1]

                    for _ in range(orders_count):
                        count += 1
                        # Calculate times for delivery
                        travel_time = self.time_matrix[pickup_real_index][delivery_real_index]
                        delivery_start_time = time_start + travel_time + self.service_time + self.extra_time

                        # Pickup point entry
                        data.append([
                            count,  # ID
                            'cp',  # Type for pickup
                            self.real_coordinates[pickup_real_index][0],  # X
                            self.real_coordinates[pickup_real_index][1],  # Y
                            1,  # Demand
                            time_start,  # StartTime
                            'inf',  # EndTime
                            self.service_time,  # ServiceTime
                            count + self.total_number_orders,  # PartnerID
                            pickup_real_index,  # RealIndex
                            self.node_type_dict[pickup_real_index]  # RealType
                        ])

                        # Delivery point entry
                        data.append([
                            count + self.total_number_orders,  # ID
                            'cd',  # Type for delivery
                            self.real_coordinates[delivery_real_index][0],  # X
                            self.real_coordinates[delivery_real_index][1],  # Y
                            -1,  # Demand
                            delivery_start_time,  # StartTime
                            delivery_start_time + self.time_window_length,  # EndTime
                            self.service_time,  # ServiceTime
                            count,  # PartnerID
                            delivery_real_index,  # RealIndex
                            self.node_type_dict[delivery_real_index]  # RealType
                        ])

        # Add depot information to the table
        depot_real_index = 0
        depot_coordinates = self.real_coordinates[depot_real_index]
        data.append([
            0,  # ID
            'depot',  # Type
            depot_coordinates[0],  # X
            depot_coordinates[1],  # Y
            0,  # Demand
            0,  # StartTime
            float('inf'),  # EndTime
            0,  # ServiceTime
            0,  # PartnerID
            depot_real_index,  # RealIndex
            self.node_type_dict[depot_real_index]  # RealType
        ])

        # Add charging station information to the table
        charging_station_real_index = len(self.real_coordinates) - 1
        charging_station_coordinates = self.real_coordinates[charging_station_real_index]
        data.append([
            self.total_number_orders*2 + 1,  # ID
            'charging',  # Type
            charging_station_coordinates[0],  # X
            charging_station_coordinates[1],  # Y
            0,  # Demand
            0,  # StartTime
            float('inf'),  # EndTime
            0,  # ServiceTime
            self.total_number_orders*2 + 1,  # PartnerID
            charging_station_real_index,  # RealIndex
            self.node_type_dict[charging_station_real_index]  # RealType
        ])

        columns = ['ID', 'Type', 'X', 'Y', 'Demand', 'StartTime', 'EndTime', 'ServiceTime', 'PartnerID', 'RealIndex', 'RealType']
        df = pd.DataFrame(data, columns=columns)
        df = df.sort_values(by='ID').reset_index(drop=True)
        return df

# Example usage
if __name__ == "__main__":
    from real_map import RealMap
    from demands import DemandGenerator
    import random
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)

    # real map
    realMap = RealMap(n_r=2, n_c=4, dist_function=np.random.uniform, dist_params={'low': -1, 'high': 1})
    # demands
    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 1, 'high': 3}
        },
        'demand_dist': {
            'function': np.random.poisson,
            'params': {'lam': 2}
        }
    }
    time_range = 30
    time_step = 10
    demands= DemandGenerator(time_range, time_step, realMap.restaurants, realMap.customers, random_params)
    print(demands.demand_table)
    # inistance
    time_params = {
        'time_window_length': 30,
        'service_time': 5,
        'extra_time': 10
    }
    pdptw_instance = PDPTWInstance(realMap, demands.demand_table, time_params, robot_speed=4)
    df = pdptw_instance.generate_whole_table()
    print('total number of orders', pdptw_instance.total_number_orders)
    pd.set_option('display.max_columns', None)  # 显示所有列
    print(df)
    # plot
    from utils import plot_instance
    plot_instance(df)