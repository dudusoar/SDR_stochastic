import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable

class RealMap:
    """
    Represents a real map with restaurants, customers, and other nodes.

    Attributes:
        N_R (int): Number of restaurants
        N_C (int): Number of customers
        DEPOT_INDEX (int): Index of the depot
        DESTINATION_INDEX (int): Index of the destination
        CHARGING_STATION_INDEX (int): Index of the charging station
    """

    N_R: int
    N_C: int
    DEPOT_INDEX: int = 0
    DESTINATION_INDEX: int
    CHARGING_STATION_INDEX: int

    def __init__(self, n_r: int, n_c: int, dist_function: Callable, dist_params: Dict):
        """
        Initialize the RealMap instance.

        Args:
            n_r (int): Number of restaurants
            n_c (int): Number of customers
            dist_function (Callable): Distribution function for generating coordinates
            dist_params (Dict): Parameters for the distribution function
        """
        self.N_R = n_r
        self.N_C = n_c
        self.n = self.N_R + self.N_C
        self.DESTINATION_INDEX = self.n + 1
        self.CHARGING_STATION_INDEX = self.n + 2

        self.all_nodes: List[int] = [self.DEPOT_INDEX] + list(range(1, self.n + 1)) + [self.DESTINATION_INDEX, self.CHARGING_STATION_INDEX]
        self.restaurants: List[int] = list(range(1, self.N_R + 1))
        self.customers: List[int] = list(range(self.N_R + 1, self.n + 1))

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

if __name__ == '__main__':
    real_map = RealMap(n_r=2, n_c=4, dist_function=np.random.uniform, dist_params={'low': -1, 'high': 1})
    print(f"All nodes: {real_map.all_nodes}")
    print(f"Number of coordinates: {len(real_map.coordinates)}")
    real_map.plot_map(show_index=True)