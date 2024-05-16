import numpy as np

class PDPGraph:
    def __init__(self, real_map, order_pairs_table):
        '''
        :param real_time_matrix: 由类RealMap产生
        :param order_pairs: 由函数generate_order_pairs产生
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
        self.pdp_to_real = self.create_mapping() # dict {pair index: real index}
        self.time_matrix = self.generate_time_matrix()

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
        Create a mapping from PDP nodes to RealMap nodes.
        '''
        mapping = {}

        # special nodes
        mapping[0] = self.real_map.all_nodes[0]
        mapping[2 * self.n + 1] = self.real_map.all_nodes[-2]
        mapping[2 * self.n + 2] = self.real_map.all_nodes[-1]

        # pickup and delivery nodes
        for i in self.C:
            real_index = self.order_pairs_table.loc[self.order_pairs_table['pair_index'] == i, 'real_index'].iloc[0]
            mapping[i] = real_index
        sorted_mapping = {key: mapping[key] for key in sorted(mapping)}
        return sorted_mapping

    def generate_time_matrix(self):
        '''
        build the time matrix for pdp based on the real time matrix
        '''
        num_nodes = len(self.N)
        time_matrix = np.zeros((num_nodes, num_nodes))
        real_time_matrix = self.real_map.time_matrix

        # pick up and delivery nodes
        for i in self.N:
            for j in self.N:
                if i != j:
                    time_matrix[i][j] = real_time_matrix[self.pdp_to_real[i]][self.pdp_to_real[j]]
                    time_matrix[j][i] = time_matrix[i][j]
                else:
                    time_matrix[i][j] = 0

        return time_matrix



# === examples ===
if __name__ == "__main__":
    from real_map import RealMap
    from demands import generate_demand_table, generate_order_pairs
    # real map
    real_map = RealMap(n_r=1, n_c=2, dist_function = np.random.uniform, dist_params = {'low': 0, 'high': 10})
    # demand table
    time_intervals = ((0, 10), (10, 20), (20, 30))
    demand_table = generate_demand_table(
        real_map.pairs, time_intervals,
        sample_dist=np.random.randint, sample_params={'low': 1, 'high': 3},
        demand_dist=np.random.poisson, demand_params={'lam': 2})
    # order pairs table
    order_pairs_table = generate_order_pairs(demand_table, real_map.coordinates)

    pdpGraph = PDPGraph(real_map, order_pairs_table)
    print('Number of pairs', pdpGraph.n)
    print('Customer sets', pdpGraph.C)
    print('Pickup sets', pdpGraph.P)
    print('Delivery sets', pdpGraph.D)
    print('All nodes', pdpGraph.N)
    print('N_depot',pdpGraph.N_depot)
    print('N_dest',pdpGraph.N_dest)
    print('p_i', pdpGraph.p_i)
    print('pdp graph mapping to real map', pdpGraph.pdp_to_real)

    # print("\nTime Matrix:")
    # # 打印矩阵，保持格式整齐
    # for row in pdpGraph.time_matrix:
    #     print(" ".join(f"{elem:.2f}" for elem in row))
    # print(pdpGraph.time_matrix.shape)





