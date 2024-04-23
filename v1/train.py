import numpy as np
from params import ParametersGenerator
from gurobipy import Model, GRB
import gurobipy as gp
from gurobipy import quicksum

# ================ 处理参数 ================
n, v = 3, 1
time_range = 120
time_stepsize = 10
B, b = 100, 0.5
Q = 30
M = 1000 # big M
stochasitc_setting = \
    {
        'coordinates': np.random.uniform,
        'coordinates_params': {'low': 0, 'high': 10},
        'sample': np.random.randint,
        'sample_params': {'low': 0, 'high': 10},
        'demand': np.random.poisson,
        'demand_params': {'lam': 5},
    }
params = ParametersGenerator(n, v, time_range, time_stepsize, B, b, Q, M, stochasitc_setting)


# ================ model ================
params_gp = {
            "WLSACCESSID": 'daa4d2b5-ee06-49b9-834b-15010a0c0cfe',
            "WLSSECRET": 'e2dc2c9f-4503-4035-ae0f-28e4b3aa9668',
            "LICENSEID": 2442699,
         }
env = gp.Env(params=params_gp)
m = gp.Model(env=env)

## ------------------------- decision variables -------------------------
x = m.addVars(params['N'],params['N'],params['V'], vtype=gp.GRB.BINARY, name='x') # x_ijk, link and vehicle selection
y = m.addVars(params['N'],params['V'], vtype=gp.GRB.BINARY, name='y') # y_ik, ensuring all stops are visited once
r = m.addVars(params['N'],params['V'], vtype=gp.GRB.BINARY, name='r') # r_ik, charging behavior
tau = m.addVars(params['N'],params['V'], ub=GRB.INFINITY, name="tau") # tau_ik, arrive time at node i for vehicle k
E = m.addVars(params['N'], params['V'], ub=params['B'], name="E")  # E_ik, current battery level at node i for vehicle k
C = m.addVars(params['N'], params['V'], ub=params['Q'], name="C") # C_ik, remaining capacity at node i for vehicle k

## ------------------------- objective function -------------------------
term_1 = quicksum(params['time_matrix'][i, j] * x[i, j, k] for i in params['N'] for j in params['N'] for k in params['V'])
term_2 = quicksum((quicksum(tau[i, k] for k in params['V']) - tilde_tau_i[i]) for i in params['P'])
objective = term_1 + term_2
m.setObjective(objective, gp.GRB.MINIMIZE)

## ------------------------- constraints -------------------------
# 车辆行为约束
m.addConstrs((quicksum(y[i, k] for k in params['V']) == 1)  for i in params['C'] ) # 每个点只能被一辆车服务
m.addConstrs((y[i, k] <= 1) for i in params['charging_station'] for k in params['V'])   # 每辆车最多充一次电
m.addConstrs((quicksum(x[i, j, k] for j in params['N'] if j != i) == y[i, k] for i in params['N_depot'] for k in params['V']), "pickup_once") # 如果车辆k被分配到起始点i，那么它必须从i出发到任意不同的点j
m.addConstrs((quicksum(x[i, 2*params['n']+1, k] for i in params['N_depot']) == y[2*params['n']+1, k]) for k in params['V']) # 一辆车如果被使用，必须最后回到的dest
m.addConstrs((quicksum(x[i, j, k] for j in params['N'] if j != i) - quicksum(x[j, i, k] for j in params['N'] if j != i) == 0) \
             for i in (params['C'] + params['charging_station']) for k in params['V']) # 一个点的flow
m.addConstrs((quicksum(x[0, j, k] for j in params['N_dest'] ) - quicksum(x[j, 0, k] for j in params['N_dest'] ) == 1) for k in params['V'])

# 时间窗口约束
m.addConstrs(tau[0, k] == 0 for k in params['V']) # depot的到达时间都是0
m.addConstrs((tau[i, k] + params['time_matrix'][i, j] * x[i, j, k] - M * (1 - x[i, j, k]) <= tau[j, k] for i in params['N_depot'] for j in params['N_dest'] for k in params['V']), "time_window_1")
m.addConstrs((tau[i, k] + params['time_matrix'][i, i+params['n']] * x[i, i+params['n'], k] <= tau[i+params['n'], k] for i in params['P'] for k in params['V']), "pickup_before_delivery")
m.addConstrs((tau[i, k] >= e_i[i] for i in params['P'] for k in params['V']), "earliest_start_time")

# # 电池约束
m.addConstrs((E[0, k] == params['B'] for k in params['V']), "initial_battery") # 每辆车的初始电量都是满的
m.addConstrs((E[j, k] <= E[i, k] - params['b'] * params['time_matrix'][i, j] * x[i, j, k] + params['B'] *(1-x[i, j, k]) for i in params['N_depot'] for j in params['N_dest'] for k in params['V'] ), "battery_usage")
m.addConstrs((E[j, k] <= params['B']  - params['b'] * params['time_matrix'][i, j] * x[i, j, k] for i in params['charging_station'] for j in params['N_dest'] for k in params['V']))

# # 载重约束
m.addConstrs((C[i, k] + p_i[i] * x[i, j, k] - Q * (1 - x[i, j, k]) <= C[j, k] for i in params['N_depot'] for j in params['N_dest'] for k in params['V']), "capacity_transfer")
m.addConstrs(C[0, k] == 0 for k in params['V'])
m.addConstrs(C[2*params['n']+1, k] == 0 for k in params['V'])

# 确保决策变量的定义域
m.addConstrs((x[i, j, k] >= 0 for i in params['N'] for j in params['N'] for k in params['V']), "domain_x")
m.addConstrs((tau[i, k] >= 0 for i in params['N'] for k in params['V']), "domain_tau")
m.addConstrs(y[i, k] == y[i+n, k] for i in params['P'] for k in params['V'])
m.addConstrs(E[i, k] >= 0 for i in params['N'] for k in params['V'])


## 求解
m.optimize()
if m.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for k in V:
      for i in N:
        for j in N:
          if (tau[i,k].X > 0) & (x[i, j, k].X > 0.5):
            print(f"Vehicle {k} travels from {i} to {j}")
else:
    print("Optimal solution not found.")