import numpy as np
import matplotlib.pyplot as plt

class RealMap:
    def __init__(self, n_r, n_c):
        self.n_r = n_r  # Number of Restaurants
        self.n_c = n_c  # Number of Customers
        self.n = self.n_r + self.n_c  # Total number of nodes
        self.all_nodes = [0] + list(range(1, self.n + 1)) + [self.n + 1, self.n + 2]
        self.node_names = self.generate_node_dict()  # Save the name-to-index mapping

    def generate_coordinates(self, dist_function, dist_params)-> dict:
        '''
        Generate the random coordinates.
        :param dist_function: random function
        :param dist_params: corresponding parameters
        '''
        # Dynamically adjust the range based on the number of nodes
        range_multiplier = np.sqrt(self.n)
        self.coordinates = {}

        # Generate coordinates for pickup and delivery nodes
        for node in self.all_nodes[:self.n + 1]:
            x = dist_function(**dist_params) * range_multiplier
            y = dist_function(**dist_params) * range_multiplier
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

    def generate_time_matrix(self) -> np.ndarray:
        """
        Calculate the time matrix based on Euclidean distance between nodes.
        """
        num_nodes = len(self.all_nodes)
        self.time_matrix = np.zeros((num_nodes, num_nodes))

        for i in self.all_nodes:
            for j in self.all_nodes:
                if i != j:
                    dist = np.sqrt((self.coordinates[i][0] - self.coordinates[j][0])**2 +
                                   (self.coordinates[i][1] - self.coordinates[j][1])**2)
                    self.time_matrix[i][j] = dist
                    self.time_matrix[j][i] = dist  # Ensure the matrix is symmetric
                else:
                    self.time_matrix[i][j] = 0  # Distance to self is zero

        return self.time_matrix

    def plot_map(self, show_index=None):
        '''
        Plot all nodes on a map, optionally showing indices or names.
        '''
        fig, ax = plt.subplots()
        # Plotting nodes with different styles
        for node, (x, y) in self.coordinates.items():
            if node == 0 or node == self.n + 1 or node == self.n + 2:  # Special nodes
                ax.plot(x, y, 'o', markersize=10, markerfacecolor='none', markeredgecolor='black', label=self.node_names[node] if self.node_names[node] not in ax.get_legend_handles_labels()[1] else "")
            elif node <= self.n_r:  # Restaurants
                ax.plot(x, y, 's', color='red', label='Restaurant' if 'Restaurant' not in ax.get_legend_handles_labels()[1] else "")
            else:  # Customers
                ax.plot(x, y, '^', color='blue', label='Customer' if 'Customer' not in ax.get_legend_handles_labels()[1] else "")
            if show_index:
                label = node if show_index == 'number' else self.node_names[node]
                ax.text(x + 0.5, y + 0.5, f'{label}', fontsize=8, ha='right')  # Adjusted label position for clarity
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    # 创建 RealMap 实例，设定有3个餐厅和6个顾客
    real_map = RealMap(n_r=3, n_c=6)

    # 生成坐标，使用 numpy 的 uniform 分布
    coordinates = real_map.generate_coordinates(np.random.uniform, {'low': 0, 'high': 10})
    print("Coordinates:")
    for node, coord in coordinates.items():
        print(f"Node {node}: ({coord[0]:.2f}, {coord[1]:.2f})")

    # 生成时间矩阵
    time_matrix = real_map.generate_time_matrix()
    print("\nTime Matrix:")
    # 打印矩阵，保持格式整齐
    for row in time_matrix:
        print(" ".join(f"{elem:.2f}" for elem in row))

    # 生成node对应名字的字典
    print("nodes_dict", real_map.generate_node_dict())
    # 生成所有餐厅和顾客的pair
    print("pairs", real_map.generate_pairs())

    # 画图
    real_map.plot_map(show_index='number')