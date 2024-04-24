import numpy as np
from demands import generate_demand_table
from demands import generate_order_details
from real_map import RealMap
from utils import generate_time_intervals


def ParametersGenerator(n,v,time_range,time_stepsize,B,b,Q, M, stochasitc_setting)-> dict:
  '''
  n: number of pairs
  v: number of vehicles
  time_range: the whole service time
  time_stepsize: size of each time step
  B: maximum battery level
  b: battery consume rate
  Q: maximum capacity
  stochasitc_setting: random function for generating the coordinates, sampling the pairs and generating the demands

  return: a dictionary for the parameters
  '''
  # initialization for the parameters
  params = {}
  params['n'] = n
  params['v'] = v
  params['B'] = B
  params['b'] = b
  params['Q'] = Q
  params['M'] = M

  # === generate graph ===
  graph = NodesGenerator(n)
  ## sets
  params['P']:tuple = graph.P
  params['D']:tuple = graph.D
  params['C']:tuple = graph.C
  params['charging_station']:tuple = graph.charging_station
  params['N']:tuple = graph.N
  params['N_depot ']:tuple = graph.N_depot
  params['N_dest']:tuple = graph.N_dest

  # === generate other sets ===
  params['V'] = tuple(range(v))

  ## other information related the graph
  params['coordinates'] = graph.generate_coordinates(
      stochasitc_setting['coordinates'],
      stochasitc_setting['coordinates_params'])
  params['time_matrix'] = graph.generate_time_matrix()
  params['pairs'] = graph.generate_pairs()


  # === generate time intervals ===
  params['time_intervals'] = generate_time_intervals(time_range,time_stepsize)

  # === generate demans table ===
  params['demand_table'] = generate_demand_table(
      params['pairs'],
      params['time_intervals'],
      stochasitc_setting['sample'],stochasitc_setting['sample_params'],
      stochasitc_setting['demand'],stochasitc_setting['demand_params'])

  return params

# === examples ===
if __name__ == "__main__":

    # Generate the map
    real_map = RealMap(n_r=3, n_c=6)

    n,v = 3,1
    time_range = 120
    time_stepsize = 10
    B,b = 100,0.5
    Q = 30
    M = 1000 # big M
    stochasitc_setting = {
        'coordinates':np.random.uniform,
        'coordinates_params':{'low': 0, 'high': 10},
        'sample':np.random.randint,
        'sample_params': {'low': 0, 'high': 10},
        'demand':np.random.poisson,
        'demand_params': {'lam': 5},
    }
    params = ParametersGenerator(n,v,time_range,time_stepsize,B,b,Q,M, stochasitc_setting)


