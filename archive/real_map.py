import numpy as np
import matplotlib.pyplot as plt


# For generating the instances
class RealMap:
    '''
    Include all of the information from the real map:
    - Number of restaurants and customers
    - Index lists for all nodes (for restaurants, customers)
    - Features: coordinates, distances matrix, node-to-type dict
    '''
    def __init__(self, n_r, n_c, dist_function, dist_params):
        # Number
        self.n_r = n_r  # Number of Restaurants
        self.n_c = n_c  # Number of Customers
        self.n = self.n_r + self.n_c # total number of nodes in the map

        # Index lists
        # 0：depot, self.n + 1: destination, self.n + 2: charging station
        self.all_nodes = [0] + list(range(1, self.n + 1)) + [self.n + 1, self.n + 2]
        self.restaurants = list(range(1, self.n_r + 1))
        self.customers = list(range(self.n_r + 1, self.n + 1))

        # Features
        self.coordinates = self.generate_coordinates(dist_function, dist_params)
        self.distance_matrix = self.generate_distance_matrix() # real time matrix
        self.node_type_dict= self.generate_node_type()  # Save the type-to-index mapping

    def generate_coordinates(self, dist_function, dist_params)-> dict:
        '''
        Generate the random coordinates
        :param dist_function: random function
        :param dist_params: corresponding parameters
        '''
        # Dynamically adjust the range based on the number of nodes
        coordinates = {}

        # Generate coordinates for pickup and delivery nodes
        for node in self.all_nodes[:self.n + 1]:
            x = dist_function(**dist_params)
            y = dist_function(**dist_params)
            coordinates[node] = (x, y)

        # Depot, destination, and charging station are placed at the same location
        coordinates[self.all_nodes[self.n + 1]] = coordinates[0]
        coordinates[self.all_nodes[self.n + 2]] = coordinates[0]

        return coordinates

    def generate_distance_matrix(self) -> np.ndarray:
        """
        Calculate the time matrix based on Euclidean distance between nodes.
        """
        num_nodes = len(self.all_nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))

        for i in self.all_nodes:
            for j in self.all_nodes:
                if i != j:
                    dist = np.sqrt((self.coordinates[i][0] - self.coordinates[j][0])**2 +
                                   (self.coordinates[i][1] - self.coordinates[j][1])**2)
                    distance_matrix[i][j] = dist
                else:
                    distance_matrix[i][j] = 0  # Distance to self is zero

        return distance_matrix

    def generate_node_type(self):
        '''
        Generate a dictionary mapping node types to their indices.
        Key: index, Value: type
        Types: depot, destination, charging station, restaurant, customer
        '''
        node_names = {}
        node_names[0] = 'depot'
        node_names[self.n + 1] = 'destination'
        node_names[self.n + 2] = 'charging_station'
        for i in range(1, self.n_r + 1):
            node_names[i] = 'restaurant'
        for j in range(self.n_r + 1, self.n + 1):
            node_names[j] = 'customer'
        return node_names

# Data loader

if __name__ == '__main__':
    # create RealMap instance，n_r restaurants and n_r customers
    real_map = RealMap(n_r=2, n_c=4, dist_function = np.random.uniform, dist_params = {'low': -1, 'high': 1})
    print(real_map.all_nodes)
    print(len(real_map.coordinates))
    # plot
    from utils import plot_real_map
    plot_real_map(real_map, show_index='True')