import numpy as np
from demands import generate_demand_table, generate_order_pairs
from real_map import RealMap
from utils import generate_time_intervals
from graph import PDPGraph


def ParametersGenerator(n_r,n_c,stochasitc_setting,time_range,time_stepsize,v,B,b,Q,M)-> dict:
  '''
  v: number of vehicles
  B: maximum battery level
  b: battery consume rate
  Q: maximum capacity
  M:
  n_r: number of restaurants
  n_c: number of customers
  stochasitc_setting: random function for generating the coordinates, sampling the pairs and generating the demands
  time_range: the whole service time
  time_stepsize: size of each time step

  return: a dictionary for the parameters
  '''
  params = {}
  # information from real data
  realMap= RealMap(
      n_r, n_c,
      dist_function = stochasitc_setting['coordinates'],
      dist_params = stochasitc_setting['coordinates_params'])
  time_intervals = generate_time_intervals(time_range, time_stepsize)
  demand_table = generate_demand_table(
      realMap.pairs, time_intervals,
      sample_dist=stochasitc_setting['sample'], sample_params = stochasitc_setting['sample_params'],
      demand_dist=stochasitc_setting['demand'], demand_params = stochasitc_setting['demand_params'])
  order_pairs_table = generate_order_pairs(demand_table, realMap.coordinates)

  params['realMap'] = realMap
  params['time_intervals'] = time_intervals
  params['demand_table'] = demand_table
  params['order_pairs_table'] = order_pairs_table

  # direct params
  params['v'] = v
  params['B'] = B
  params['b'] = b
  params['Q'] = Q
  params['M'] = M # big M

  # pdp graph
  pdp = PDPGraph(realMap, order_pairs_table)
  params['n'] = pdp.n # number of pairs
  ## sets
  params['P'] = pdp.P
  params['D'] = pdp.D
  params['C'] = pdp.C
  params['N'] = pdp.N
  params['N_depot '] = pdp.N_depot
  params['N_dest'] = pdp.N_dest
  params['V'] = list(range(v)) # vehicle sets

  params['time_matrix'] = pdp.time_matrix
  params['p_i'] = pdp.p_i

  return params

# === examples ===
if __name__ == "__main__":
    n_r,n_c = 1,2
    stochasitc_setting = {
        'coordinates':np.random.uniform,
        'coordinates_params':{'low': 0, 'high': 10},
        'sample':np.random.randint,
        'sample_params': {'low': 0, 'high': 10},
        'demand':np.random.poisson,
        'demand_params': {'lam': 5},
    }
    time_range = 120
    time_stepsize = 10

    v = 1
    B,b = 100,0.5
    Q = 30
    M = 1000 # big M

    params = ParametersGenerator(n_r,n_c,stochasitc_setting,time_range,time_stepsize,v,B,b,Q,M)


