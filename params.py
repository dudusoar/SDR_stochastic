import numpy as np
from demands import generate_demand_table, generate_order_pairs
from real_map import RealMap
from utils import generate_time_intervals
from graph import PDPGraph



def ParametersGenerator(n_r,n_c,stochasitc_setting,time_range,time_stepsize, time_delay, v,B,b,Q,M)-> dict:
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
  time_delay: latest delay time

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
  params['charging'] = pdp.charging_station
  params['P'] = pdp.P
  params['D'] = pdp.D
  params['C'] = pdp.C
  params['N'] = pdp.N
  params['N_depot'] = pdp.N_depot
  params['N_dest'] = pdp.N_dest
  params['N_0'] = [item for item in pdp.N_depot if item not in pdp.charging_station]
  params['N_2n1'] = [item for item in pdp.N_dest if item not in pdp.charging_station]
  params['V'] = list(range(v)) # vehicle sets


  params['time_matrix'] = pdp.time_matrix
  params['p_i'] = pdp.p_i

  # time windows
  e_i_demands = order_pairs_table[['pair_index', 'earliest']].sort_values(by='pair_index')['earliest'].values
  e_i = np.concatenate(([0], e_i_demands, [0, 0]))
  params['e_i'] = e_i

  l_i = [1000] * len(e_i)
  l_i[pdp.n + 1:2 * pdp.n + 1] = [x + time_delay for x in e_i[pdp.n + 1:2 * pdp.n + 1]]
  params['l_i'] = l_i

  return params

# === examples ===
if __name__ == "__main__":
    n_r,n_c = 1,2
    stochasitc_setting = {
        'coordinates':np.random.uniform,
        'coordinates_params':{'low': 0, 'high': 10},
        'sample':np.random.randint,
        'sample_params': {'low': 0, 'high': 3},
        'demand':np.random.poisson,
        'demand_params': {'lam': 2},
    }
    time_range = 30
    time_stepsize = 10
    time_delay = 60

    v = 1
    B,b = 100,0.5
    Q = 30
    M = 1000 # big M

    params = ParametersGenerator(n_r,n_c,stochasitc_setting,time_range,time_stepsize,time_delay, v,B,b,Q,M)

    print(params['time_matrix'].shape)
    print(params['n'])
    print(params['demand_table'])
    print(params['order_pairs_table'])

    print(params['e_i'])
    print(len(params['e_i']))
    print(params['l_i'])
    print(len(params['l_i']))



