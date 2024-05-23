import numpy as np
import matplotlib.pyplot as plt
class PDPGraph:
    def __init__(self, real_map, order_pairs_table):
        '''
        :param real_time_matrix: generated by the class RealMap
        :param order_pairs: generated by the function generate_order_pairs
        '''
        self.real_map = real_map
        self.order_pairs_table = order_pairs_table

        self.n = len(self.order_pairs_table) // 2 # each pair has two nodes (pickup + delivery)
        # === Special points===
        self.depot = [0]
        self.dest = [2 * self.n + 1]
        self.charging_station = [2 * self.n + 2]
        # === Sets: list ===
        self.C = sorted(self.order_pairs_table['pair_index'].tolist())
        self.P = self.C[:self.n]
        self.D = self.C[self.n:]
        self.N = sorted(self.depot + self.C + self.dest + self.charging_station)
        self.N_depot = sorted(self.depot + self.C + self.charging_station)
        self.N_dest = sorted(self.C + self.dest + self.charging_station)

        # features
        self.p_i = self.generate_demand_dict()
        self.pdp_to_real, self.real_to_pdp = self.create_mapping()
        self.distance_matrix = self.generate_distance_matrix()

    def generate_demand_dict(self):
        '''
        Generate demand for each pair, positive for pickups and negative for deliveries.
        '''
        p_i = {node: 0 for node in self.N}  # Initialize all node demands to zero
        for i in range(1, self.n + 1):
            p_i[i] = 1  # Positive demand for pickup
            p_i[i + self.n] = -1  # Negative demand for delivery
        return p_i

    def create_mapping(self):
        '''
        Create mappings between PDP nodes and RealMap nodes.
        Returns:
            - pdp_to_real: Dictionary mapping PDP nodes to RealMap nodes.
            - real_to_pdp: Dictionary mapping RealMap nodes to lists of PDP nodes.
        '''
        pdp_to_real = {}
        real_to_pdp = {}

        # special nodes
        pdp_to_real[0] = self.real_map.all_nodes[0]
        pdp_to_real[2 * self.n + 1] = self.real_map.all_nodes[-2]
        pdp_to_real[2 * self.n + 2] = self.real_map.all_nodes[-1]

        real_to_pdp[self.real_map.all_nodes[0]] = [0]
        real_to_pdp[self.real_map.all_nodes[-2]] = [2 * self.n + 1]
        real_to_pdp[self.real_map.all_nodes[-1]] = [2 * self.n + 2]

        # pickup and delivery nodes
        for i in self.C:
            real_index = self.order_pairs_table.loc[self.order_pairs_table['pair_index'] == i, 'real_index'].iloc[0]
            pdp_to_real[i] = real_index

            if real_index not in real_to_pdp:
                real_to_pdp[real_index] = []
            real_to_pdp[real_index].append(i)

        return pdp_to_real, real_to_pdp

    def generate_distance_matrix(self):
        '''
        build the time matrix for pdp based on the real time matrix
        '''
        num_nodes = len(self.N)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        real_distance_matrix = self.real_map.distance_matrix

        # pick up and delivery nodes
        for i in self.N:
            for j in self.N:
                if i != j:
                    distance_matrix[i][j] = real_distance_matrix[self.pdp_to_real[i]][self.pdp_to_real[j]]
                    distance_matrix[j][i] = distance_matrix[i][j]
                else:
                    distance_matrix[i][j] = 0

        return distance_matrix

    def plot_graph(self, show_index=True, highlight_nodes=None):
        """
        Plot the PDP graph, optionally showing indices or highlighting nodes.
        :param show_index: If True, shows node indices.
        :param highlight_nodes: List of node indices to highlight.
        """
        fig, ax = plt.subplots()

        # Plot nodes with different styles based on their type
        for real_node, pdp_nodes in self.real_to_pdp.items():
            x, y = self.real_map.coordinates[real_node]

            node_type = ''
            if pdp_nodes[0] in self.depot:
                node_type = 'Depot'
            elif pdp_nodes[0] in self.charging_station:
                node_type = 'Charging Station'
            elif pdp_nodes[0] in self.P:
                node_type = 'Pickup'
            else:  # Delivery node
                node_type = 'Delivery'

            marker, color = {'Depot': ('o', 'black'),
                             'Charging Station': ('s', 'green'),
                             'Pickup': ('^', 'blue'),
                             'Delivery': ('v', 'red')}[node_type]

            ax.plot(x, y, marker, markersize=10, markerfacecolor=color, markeredgecolor='black', label=node_type)

            if show_index:
                label_text = ', '.join(str(i) for i in pdp_nodes)
                ax.text(x, y + 0.2, f"[{label_text}]", fontsize=8, ha='center', va='bottom')

            # Optionally highlight specified nodes
            if highlight_nodes and any(node in highlight_nodes for node in pdp_nodes):
                ax.plot(x, y, marker, markersize=15, markerfacecolor='none', markeredgecolor='yellow',
                        markeredgewidth=2)

        # Plot edges between pickup and delivery nodes
        for pickup, delivery in zip(self.P, self.D):
            pickup_coords = self.real_map.coordinates[self.pdp_to_real[pickup]]
            delivery_coords = self.real_map.coordinates[self.pdp_to_real[delivery]]
            ax.plot([pickup_coords[0], delivery_coords[0]], [pickup_coords[1], delivery_coords[1]], 'k--',
                    linewidth=0.8)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('PDP Graph')
        #ax.grid(True)

        # Create a legend with unique labels
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        ax.legend(unique_handles, unique_labels)

        plt.show()



# === examples ===
if __name__ == "__main__":
    from real_map import RealMap
    from demands import generate_demand_table, generate_order_pairs
    from utils import print_pdpgraph_attributes

    # real map
    real_map = RealMap(n_r=3, n_c=6, dist_function = np.random.uniform, dist_params = {'low': -3.2, 'high': 3.2})
    # demand table
    time_intervals = ((0, 10), (10, 20), (20, 30))
    demand_table = generate_demand_table(
        real_map.pairs, time_intervals,
        sample_dist=np.random.randint, sample_params={'low': 1, 'high': 3},
        demand_dist=np.random.poisson, demand_params={'lam': 2})
    # order pairs table
    order_pairs_table = generate_order_pairs(demand_table, real_map.coordinates)

    # pdp graph instances
    pdpGraph = PDPGraph(real_map, order_pairs_table)

    # plot
    pdpGraph.plot_graph(show_index=True)

    # print
    print_pdpgraph_attributes(pdpGraph)





