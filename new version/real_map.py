import numpy as np
import matplotlib.pyplot as plt

class RealMap:
    def __init__(self, n_r, n_c):
        self.n_r = n_r  # Number of Restaurants
        self.n_c = n_c  # Number of Customers
        self.n = self.n_r + self.n_c

        # list
        # 0：start(depot), self.n + 1: end(depot), self.n + 2: charging station
        self.all_nodes = [0] + list(range(1, self.n + 1)) + [self.n + 1, self.n + 2]

    def generate_coordinates(self, dist_function, dist_params)-> dict:
        '''
        Generate the random coordinates.
        :param dist_function: random function
        :param dist_params: corresponding parameters
        '''
        self.coordinates = {}

        # Generate coordinates for pickup and delivery nodes
        for node in self.all_nodes[:self.n + 1]:
            x = dist_function(**dist_params)
            y = dist_function(**dist_params)
            self.coordinates[node] = (x, y)

        # Depot, destination, and charging station are placed at the same location
        self.coordinates[self.all_nodes[self.n + 1]] = self.coordinates[0]
        self.coordinates[self.all_nodes[self.n + 2]] = self.coordinates[0]

        return self.coordinates

    def generate_node_dict(self):
        ''' Generate a dictionary mapping node names to their indices. '''
        node_names = {}
        node_names[0] = 'depot'
        node_names[self.n + 1] = 'destination'
        node_names[self.n + 2] = 'charging_station'
        for i in range(1, self.n_r + 1):
            node_names[i] = f'r_{i}'
        for j in range(self.n_r + 1, self.n + 1):
            node_names[j] = f'c_{j - self.n_r}'
        return node_names

    def generate_pairs(self):
        ''' Generate all restaurant-customer pairs. '''
        pairs = []
        for r in range(1, self.n_r + 1):
            for c in range(self.n_r + 1, self.n + 1):
                pairs.append((r,c))
        return pairs

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

    def plot_map(self, show_index=None, highlight_nodes=None):
        '''
        Plot all nodes on a map, optionally showing indices or names.
        :param show_index: If set to 'number', shows node numbers; if set to 'name', shows node names.
        :param highlight_nodes: List of node indices to highlight.
        '''
        fig, ax = plt.subplots()
        added_labels = set()  # To track added labels for the legend
        x_coords = [coord[0] for coord in self.coordinates.values()]
        y_coords = [coord[1] for coord in self.coordinates.values()]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # Plotting nodes with different styles
        for node, (x, y) in self.coordinates.items():
            if node == 0 or node == self.n + 1 or node == self.n + 2:  # depot, destination, charging station
                marker, color, label = 'o', 'black', 'Depot/Destination/Charging Station'
            elif node <= self.n_r:  # Restaurants
                marker, color, label = 's', 'red', 'Restaurant'
            else:  # Customers
                marker, color, label = '^', 'blue', 'Customer'

            if label not in added_labels:
                ax.plot(x, y, marker, markersize=10, markerfacecolor='none', markeredgecolor=color, label=label)
                added_labels.add(label)
            else:
                ax.plot(x, y, marker, markersize=10, markerfacecolor='none', markeredgecolor=color)

            if show_index:
                label_text = str(node) if show_index == 'number' else self.all_nodes_names[node]
                ax.text(x + 0.02 * x_range, y + 0.02 * y_range, label_text, fontsize=8, ha='right')

            # Optionally highlight specified nodes
            if highlight_nodes and node in highlight_nodes:
                ax.plot(x, y, marker, markersize=15, markerfacecolor='yellow', markeredgecolor='gold')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    # create RealMap instance，n_r restaurants and n_r customers
    realMap = RealMap(n_r=2, n_c=4, dist_function = np.random.uniform, dist_params = {'low': -1, 'high': 1})
    # plot
    realMap.plot_map(show_index='number')
    # print the contents
    from utils import print_realmap_attributes
    print_realmap_attributes(realMap)