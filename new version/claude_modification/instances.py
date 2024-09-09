import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

class PDPTWInstance:
    def __init__(self, real_map, demand_table: pd.DataFrame, time_params: Dict[str, int], robot_speed: float):
        """
        Initialize the PDPTWInstance.

        Args:
            real_map: Instance of RealMap class
            demand_table (pd.DataFrame): DataFrame with demand information per time interval.
            time_params (Dict[str, int]): Dictionary containing time-related parameters.
            robot_speed (float): Speed of the robot in distance per minute.
        """
        self.real_coordinates: Dict[int, Tuple[float, float]] = real_map.coordinates
        self.distance_matrix: np.ndarray = real_map.distance_matrix
        self.node_type_dict: Dict[int, str] = real_map.node_type_dict
        self.demand_table: pd.DataFrame = demand_table
        self.time_window_length: int = time_params['time_window_length']
        self.service_time: int = time_params['service_time']
        self.extra_time: int = time_params['extra_time']
        self.robot_speed: float = robot_speed

        self.total_number_orders: int = self.demand_table.iloc[:, 2:].sum().sum().astype(int)
        self.time_matrix: np.ndarray = self._generate_time_matrix()

    def _generate_time_matrix(self) -> np.ndarray:
        """Generate time matrix in minutes."""
        return (self.distance_matrix / self.robot_speed) * 60

    def generate_whole_table(self) -> pd.DataFrame:
        """Generate the complete table with all required information."""
        data: List[List] = []
        count: int = 0

        for j in range(2, self.demand_table.shape[1]):
            time_start: int = int(self.demand_table.columns[j].split('-')[0])
            for i in range(self.demand_table.shape[0]):
                orders_count: int = self.demand_table.iloc[i, j]
                if orders_count > 0:
                    pickup_real_index: int = self.demand_table.iloc[i, 0]
                    delivery_real_index: int = self.demand_table.iloc[i, 1]

                    for _ in range(orders_count):
                        count += 1
                        travel_time: float = self.time_matrix[pickup_real_index][delivery_real_index]
                        delivery_start_time: float = time_start + travel_time + self.service_time + self.extra_time

                        data.extend([
                            self._create_pickup_entry(count, pickup_real_index, time_start),
                            self._create_delivery_entry(count, delivery_real_index, delivery_start_time)
                        ])

        data.extend([
            self._create_depot_entry(),
            self._create_charging_station_entry()
        ])

        columns: List[str] = ['ID', 'Type', 'X', 'Y', 'Demand', 'StartTime', 'EndTime', 'ServiceTime', 'PartnerID',
                              'RealIndex', 'RealType']
        df: pd.DataFrame = pd.DataFrame(data, columns=columns)
        return df.sort_values(by='ID').reset_index(drop=True)

    def _create_pickup_entry(self, count: int, pickup_real_index: int, time_start: int) -> List:
        return [
            count,
            'cp',
            self.real_coordinates[pickup_real_index][0],
            self.real_coordinates[pickup_real_index][1],
            1,
            time_start,
            float('inf'),
            self.service_time,
            count + self.total_number_orders,
            pickup_real_index,
            self.node_type_dict[pickup_real_index]
        ]

    def _create_delivery_entry(self, count: int, delivery_real_index: int, delivery_start_time: float) -> List:
        return [
            count + self.total_number_orders,
            'cd',
            self.real_coordinates[delivery_real_index][0],
            self.real_coordinates[delivery_real_index][1],
            -1,
            delivery_start_time,
            delivery_start_time + self.time_window_length,
            self.service_time,
            count,
            delivery_real_index,
            self.node_type_dict[delivery_real_index]
        ]

    def _create_depot_entry(self) -> List:
        depot_real_index: int = 0
        depot_coordinates: Tuple[float, float] = self.real_coordinates[depot_real_index]
        return [
            0,
            'depot',
            depot_coordinates[0],
            depot_coordinates[1],
            0,
            0,
            float('inf'),
            0,
            0,
            depot_real_index,
            self.node_type_dict[depot_real_index]
        ]

    def _create_charging_station_entry(self) -> List:
        charging_station_real_index: int = len(self.real_coordinates) - 1
        charging_station_coordinates: Tuple[float, float] = self.real_coordinates[charging_station_real_index]
        return [
            self.total_number_orders * 2 + 1,
            'charging',
            charging_station_coordinates[0],
            charging_station_coordinates[1],
            0,
            0,
            float('inf'),
            0,
            self.total_number_orders * 2 + 1,
            charging_station_real_index,
            self.node_type_dict[charging_station_real_index]
        ]

    def plot_instance(self):
        """
        Plot the PDPTW instance orders, highlighting overlapping locations with order IDs.
        """
        plt.figure(figsize=(12, 10))

        location_orders = defaultdict(list)
        labels_added = set()

        df = self.generate_whole_table()

        for _, row in df.iterrows():
            location = (row['X'], row['Y'])
            location_orders[location].append(row['ID'])

            if row['Type'] == 'cp':
                label = 'Pickup' if 'Pickup' not in labels_added else ''
                plt.scatter(row['X'], row['Y'], c='blue', marker='o', s=100, label=label)
                labels_added.add('Pickup')
            elif row['Type'] == 'cd':
                label = 'Delivery' if 'Delivery' not in labels_added else ''
                plt.scatter(row['X'], row['Y'], c='green', marker='d', s=100, label=label)
                labels_added.add('Delivery')
            elif row['Type'] == 'depot':
                label = 'Depot' if 'Depot' not in labels_added else ''
                plt.scatter(row['X'], row['Y'], c='red', marker='s', s=100, label=label)
                labels_added.add('Depot')
            elif row['Type'] == 'charging':
                label = 'Charging Station' if 'Charging Station' not in labels_added else ''
                plt.scatter(row['X'], row['Y'], c='purple', marker='^', s=100, label=label)
                labels_added.add('Charging Station')

        for location, orders in location_orders.items():
            if len(orders) > 1:
                plt.text(location[0], location[1] + 0.02, f"[{', '.join(map(str, orders))}]", fontsize=8, ha='center')
            else:
                plt.text(location[0], location[1] + 0.02, f"{orders[0]}", fontsize=8, ha='center')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('PDPTW Orders Plot')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    from real_map import RealMap
    from demands import DemandGenerator
    import random

    # Set random seed
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Create RealMap instance
    real_map = RealMap(n_r=2, n_c=4, dist_function=np.random.uniform, dist_params={'low': -1, 'high': 1})

    # Generate demands
    random_params = {
        'sample_dist': {'function': np.random.randint, 'params': {'low': 1, 'high': 3}},
        'demand_dist': {'function': np.random.poisson, 'params': {'lam': 2}}
    }
    demands = DemandGenerator(time_range=30, time_step=10, restaurants=real_map.restaurants,
                              customers=real_map.customers, random_params=random_params)

    # Create PDPTWInstance
    time_params = {'time_window_length': 30, 'service_time': 5, 'extra_time': 10}
    pdptw_instance = PDPTWInstance(real_map, demands.demand_table, time_params, robot_speed=4)

    # Generate and display the whole table
    df = pdptw_instance.generate_whole_table()
    print('Total number of orders:', pdptw_instance.total_number_orders)
    pd.set_option('display.max_columns', None)
    print(df)

    # Plot the instance with improved visualization
    pdptw_instance.plot_instance()