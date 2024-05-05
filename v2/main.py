import argparse
from params import ParametersGenerator
from gurobipy import Model, GRB
import gurobipy as gp

# input from terminal
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument("-v", type=int, default=2, help="Number of vehicles")
parser.add_argument("-B", type=float, default=20, help="Battery capacity")
parser.add_argument("-b", type=float, default=0.5, help="Battery consumption rate ")
parser.add_argument("-Q", type=int, default=6, help="Maximum capacity")
parser.add_argument("-M", type=int, default=1000, help="Big M")

parser.add_argument("-n_r", type=int, default=2, help="Number of restaurant")
parser.add_argument("-n_c", type=int, default=6, help="Number of customers")
parser.add_argument("--time_range", type=int, default=120, help="Whole service time")
parser.add_argument("--time_stepsize", type=int, default=10, help="Size of each time step")

args = parser.parse_args()

# 随机函数的参数设置
stochasitc_setting = {
        'coordinates':np.random.uniform,
        'coordinates_params':{'low': 0, 'high': 10},
        'sample':np.random.randint,
        'sample_params': {'low': 0, 'high': 10},
        'demand':np.random.poisson,
        'demand_params': {'lam': 5},
    }

# 从命令行中获取值
n_r = args.n_r
n_c = args.n_c
time_range = args.time_range
time_stepsize = args.time_stepsize
v = args.v
B = args.B
b = args.b
Q = args.Q
M = args.M

# 包含所有参数的dict
params = ParametersGenerator(n_r,n_c,stochasitc_setting,time_range,time_stepsize,v,B,b,Q,M)



# 模型
params_gp = {
"WLSACCESSID": 'daa4d2b5-ee06-49b9-834b-15010a0c0cfe',
"WLSSECRET": 'e2dc2c9f-4503-4035-ae0f-28e4b3aa9668',
"LICENSEID": 2442699,
}
env = gp.Env(params=params_gp)
m = gp.Model(env=env)

x = m.addVars(params['N'], params['N'], params['V'], vtype=GRB.BINARY, name="x")
y = m.addVars(N_E, V, vtype=GRB.BINARY, name="y")
r = m.addVars(D, vtype=GRB.BINARY, name="r") # late penalty decision
tau = m.addVars(N_E, V, ub=GRB.INFINITY, name="tau")
E = m.addVars(N_E, V, ub=B, name="E")
C = m.addVars(N_E, V, ub=Q, name="C")