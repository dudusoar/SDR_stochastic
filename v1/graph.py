import numpy as np
import matplotlib.pyplot as plt

class NodesGenerator:
    '''
    Generate the graph
    '''
    def __init__(self,n):

      self.n = n # number of the pickup nodes

      # === Special points: tuple ===
      self.depot = (0,)
      self.dest = (2*self.n + 1,)
      self.charging_station = (2*self.n + 2,)

      # === Sets: tuple ===
      self.P = tuple(range(1, self.n + 1))  # pick up nodes
      self.D = tuple(range(self.n + 1, 2 * self.n + 1)) # delivery nodes

      self.C = self.P + self.D  # all customer nodes
      self.N_depot = self.depot + self.C + self.charging_station
      self.N_dest = self.C + self.dest + self.charging_station
      self.N = self.depot + self.C + self.dest + self.charging_station  # all nodes

    def generate_pairs(self)-> tuple:
      '''
      generate all pickup-delivery pairs
      not include the depot, destination and charging station
      '''
      self.pairs = tuple((p, p + self.n) for p in self.P)
      return self.pairs

    def generate_coordinates(self, dist_function, dist_params)-> dict:
      '''
      generate the random coordinates
      :params dist_function: random function
      :params dist_params: corresponding parameters
      '''

      # Dynamically adjust the range based on the number of nodes
      range_multiplier = np.sqrt(self.n)
      self.coordinates:dict[int,tuple] = {}

      # === pickup and delivery nodes ===
      for node in self.N_depot:
        x = dist_function(**dist_params) * range_multiplier
        y = dist_function(**dist_params) * range_multiplier
        self.coordinates[node] = (x, y)

      # === depot, destination, charging station ===
      self.coordinates[self.dest[0]] = self.coordinates[self.depot[0]]
      self.coordinates[self.charging_station[0]] = self.coordinates[self.depot[0]]

      return self.coordinates

    def generate_time_matrix(self) -> np.ndarray:
        """
        calculate the time matrix
        """
        num_nodes = len(self.N)
        self.time_matrix = np.zeros((num_nodes, num_nodes))

        for i in self.N:
          for j in self.N:
            if i != j:
              dist = np.sqrt((self.coordinates[self.N[i]][0] - self.coordinates[self.N[j]][0])**2 +
                      (self.coordinates[self.N[i]][1] - self.coordinates[self.N[j]][1])**2)
              self.time_matrix[i][j] = dist
              self.time_matrix[j][i] = dist #symmetric matrix
            else:
              self.time_matrix[i][j] = 0

        return self.time_matrix

# === examples ===
if __name__ == "__main__":
  generator = NodesGenerator(n=5)
  coordinates = generator.generate_coordinates(dist_function=np.random.uniform, dist_params={'low': 0, 'high': 10})
  pairs = generator.generate_pairs()
  time_matrix = generator.generate_time_matrix()

  # 画图
  # from utils import plot_nodes
  # plot_nodes(coordinates)

  print("Generated Pairs:", pairs)
  print("Coordinates:", generator.coordinates)
  print("Time Matrix:\n", time_matrix)dwdaw