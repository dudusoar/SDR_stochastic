import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class RealMap:
    """
    Represents a real map with restaurants, customers, and other nodes.
    """
    def __init__(self, n_r: int, n_c: int, dist_function: Callable, dist_params: Dict):
        """
        Initialize the RealMap instance.

        Args:
            n_r (int): Number of restaurants
            n_c (int): Number of customers
            dist_function (Callable): Distribution function for generating coordinates
            dist_params (Dict): Parameters for the distribution function
        """
        # number counts
        self.N_R = n_r
        self.N_C = n_c
        self.n = self.N_R + self.N_C

        # index list
        self.DEPOT_INDEX = 0
        self.DESTINATION_INDEX = self.n + 1
        self.CHARGING_STATION_INDEX = self.n + 2
        self.all_nodes: List[int] = [self.DEPOT_INDEX] + list(range(1, self.n + 1)) + [self.DESTINATION_INDEX, self.CHARGING_STATION_INDEX]
        self.restaurants: List[int] = list(range(1, self.N_R + 1))
        self.customers: List[int] = list(range(self.N_R + 1, self.n + 1))

        # other properties
        self.coordinates: Dict[int, Tuple[float, float]] = self._generate_coordinates(dist_function, dist_params)
        self.distance_matrix: np.ndarray = self._generate_distance_matrix()
        self.node_type_dict: Dict[int, str] = self._generate_node_type()

    def _generate_coordinates(self, dist_function: Callable, dist_params: Dict) -> Dict[int, Tuple[float, float]]:
        """
        Generate random coordinates for all nodes.

        Args:
            dist_function (Callable): Distribution function for generating coordinates
            dist_params (Dict): Parameters for the distribution function

        Returns:
            Dict[int, Tuple[float, float]]: Dictionary of node indices to coordinates
        """
        coordinates = {node: (dist_function(**dist_params), dist_function(**dist_params)) for node in self.all_nodes[:self.n + 1]}
        coordinates[self.DESTINATION_INDEX] = coordinates[self.DEPOT_INDEX]
        coordinates[self.CHARGING_STATION_INDEX] = coordinates[self.DEPOT_INDEX]
        return coordinates

    def _generate_distance_matrix(self) -> np.ndarray:
        """
        Calculate the distance matrix based on Euclidean distance between nodes.

        Returns:
            np.ndarray: Distance matrix
        """
        coords = np.array([self.coordinates[i] for i in self.all_nodes])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def _generate_node_type(self) -> Dict[int, str]:
        """
        Generate a dictionary mapping node indices to their types.

        Returns:
            Dict[int, str]: Dictionary of node indices to node types
        """
        return {
            self.DEPOT_INDEX: 'depot',
            self.DESTINATION_INDEX: 'destination',
            self.CHARGING_STATION_INDEX: 'charging_station',
            **{i: 'restaurant' for i in self.restaurants},
            **{i: 'customer' for i in self.customers}
        }

    def plot_map(self, show_index: bool = True):
        """
        Plot the map with all nodes.

        Args:
            show_index (bool): Whether to show node indices on the plot
        """
        plt.figure(figsize=(10, 10))
        colors = {'depot': 'red', 'destination': 'red', 'charging_station': 'green',
                  'restaurant': 'blue', 'customer': 'orange'}

        for node, (x, y) in self.coordinates.items():
            node_type = self.node_type_dict[node]
            plt.scatter(x, y, c=colors[node_type], s=100)
            if show_index:
                plt.annotate(str(node), (x, y), xytext=(5, 5), textcoords='offset points')

        plt.title("Real Map")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend(colors.keys())
        plt.grid(True)
        plt.show()

class RealDataMap:
    def __init__(self, node_file: str, tt_matrix_file: str):
        """
        初始化 RealDataMap 实例。
        :param node_file: 节点信息文件路径
        :param tt_matrix_file: 距离矩阵文件路径
        """
        self.node_data = self._load_node_data(node_file)
        self.tt_matrix = self._load_tt_matrix(tt_matrix_file)
        
        self.N_R = len(self.node_data[self.node_data['type'] == 'restaurant'])
        self.N_C = len(self.node_data[self.node_data['type'] == 'apartment'])
        self.n = self.N_R + self.N_C

        self.DEPOT_INDEX = self.node_data[self.node_data['type'] == 'restaurant'].index[0]
        self.DESTINATION_INDEX = self.DEPOT_INDEX  # 假设目的地与仓库相同
        self.CHARGING_STATION_INDEX = self.DEPOT_INDEX  # 假设充电站与仓库相同

        self.all_nodes = list(range(len(self.node_data)))
        self.restaurants = list(self.node_data[self.node_data['type'] == 'restaurant'].index)
        self.customers = list(self.node_data[self.node_data['type'] == 'apartment'].index)

        self.coordinates = self._generate_coordinates()
        self.distance_matrix = self.tt_matrix
        self.node_type_dict = self._generate_node_type()

    def _load_node_data(self, file_path: str) -> pd.DataFrame:
        """加载节点数据"""
        df = pd.read_csv(file_path)
        return df.set_index('index')

    def _load_tt_matrix(self, file_path: str) -> np.ndarray:
        """加载距离矩阵"""
        return np.loadtxt(file_path, delimiter=',')

    def _generate_coordinates(self) -> Dict[int, Tuple[float, float]]:
        """生成坐标字典"""
        return {index: (row['longitude'], row['latitude']) for index, row in self.node_data.iterrows()}

    def _generate_node_type(self) -> Dict[int, str]:
        """生成节点类型字典"""
        return dict(zip(self.node_data.index, self.node_data['type']))

    def plot_map(self, show_index: bool = True):
        """
        绘制地图，显示所有节点
        """
        plt.figure(figsize=(15, 15))
        colors = {'restaurant': 'red', 'apartment': 'blue', 'university building': 'green'}

        for node_type, group in self.node_data.groupby('type'):
            plt.scatter(group['longitude'], group['latitude'], c=colors.get(node_type, 'gray'), 
                        label=node_type, s=50, alpha=0.7)

        if show_index:
            for idx, row in self.node_data.iterrows():
                plt.annotate(str(idx), (row['longitude'], row['latitude']), xytext=(3, 3), 
                             textcoords='offset points', fontsize=8)

        plt.title("Real Data Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        plt.show()




if __name__ == '__main__':
    real_map = RealMap(n_r=2, n_c=4, dist_function=np.random.uniform, dist_params={'low': -1, 'high': 1})
    print(f"All nodes: {real_map.all_nodes}")
    print(f"Number of coordinates: {len(real_map.coordinates)}")
    real_map.plot_map(show_index=True)