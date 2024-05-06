import argparse
import numpy as np
from params import ParametersGenerator
from gurobipy import Model, GRB
import gurobipy as gp


def create_model(params):
    # create model
    params_gp = {
        "WLSACCESSID": 'daa4d2b5-ee06-49b9-834b-15010a0c0cfe',
        "WLSSECRET": 'e2dc2c9f-4503-4035-ae0f-28e4b3aa9668',
        "LICENSEID": 2442699,
    }
    env = gp.Env(params=params_gp)
    m = gp.Model(env=env)

    # decision variables
    x = m.addVars(params['N'], params['N'], params['V'], vtype=gp.GRB.BINARY,name='x')  # x_ijk, link and vehicle selection
    y = m.addVars(params['N'], params['V'], vtype=gp.GRB.BINARY, name='y')  # y_ik, ensuring all stops are visited once
    r = m.addVars(params['D'], vtype=gp.GRB.BINARY, name='r')  # r_ik, late penalty
    tau = m.addVars(params['N'], params['V'], ub=GRB.INFINITY,name="tau")  # tau_ik, arrive time at node i for vehicle k
    E = m.addVars(params['N'], params['V'], ub=params['B'],name="E")  # E_ik, current battery level at node i for vehicle k
    C = m.addVars(params['N'], params['V'], ub=params['Q'],name="C")  # C_ik, remaining capacity at node i for vehicle k

    # constraints
    ## no selection between the same nodes
    for i in params['N']:
        for j in params['N']:
            for k in params['V']:
                if (i == j) | ((i == j + params['n']) & (j > 0)):
                    x[i, j, k].ub = 0

    # 每个点被服务一次
    m.addConstrs((sum(y[i, k] for k in params['V']) == 1) for i in params['C'])  # 每个点只被一辆车服务一次
    m.addConstrs((y[i, k] <= 1) for i in params['charging'] for k in params['V'])  # 每个车最多充一次电
    m.addConstrs((sum(x[i, 2 * params['n'] + 1, k] for i in params['N_depot']) == y[2 * params['n'] + 1, k]) for k in
                 params['V'])

    # 车辆行为约束 ———— 遍历每一个vehicle
    ## 对于每一辆车，如果yik为0，表明该点没有被车辆k选择，那么与i点相连的每一个边都不会被选择，如果为1，那么表示其中一条边被选择
    m.addConstrs((sum(x[i, j, k] for j in params['N'] if (j != i)) == y[i, k] for i in params['N_depot'] for k in params['V']))  # 从i出发
    # m.addConstrs((sum(x[j, i, k] for j in params['N'] if (j != i)) == y[i, k] for i in params['N_depot'] for k in params['V'])) # 从其他点到i

    m.addConstrs((sum(x[i, j, k] for j in params['N'] if j != i) - sum(x[j, i, k] for j in params['N'] if j != i) == 0) for i in (params['C'] + params['charging']) for k in params['V'])
    m.addConstrs((sum(x[0, j, k] for j in params['N_dest']) - sum(x[j, 0, k] for j in params['N_dest']) == 1) for k in params['V'])  # 从depot出发

    # 时间窗口约束
    m.addConstrs(tau[0, k] == 0 for k in params['V'])
    m.addConstrs(
        (tau[i, k] + 2 + params['time_matrix'][i, j] * x[i, j, k] - params['M'] * (1 - x[i, j, k]) <= tau[j, k] for i in
         params['N_depot'] for j in params['N_dest'] if (j != i) for k in params['V']), "time_window_1")
    m.addConstrs(
        (tau[i, k] + params['time_matrix'][i, i + params['n']] - params['M'] * (1 - y[i, k]) <= tau[i + params['n'], k]
         for i in params['P'] for k in params['V']), "pickup_before_delivery")
    m.addConstrs((tau[i, k] >= params['e_i'][i] * y[i, k] for i in params['P'] for k in params['V']),
                 "earliest_start_time")
    ## 延误时间
    m.addConstrs((params['l_i'][i] - tau[i, k] <= params['M'] * (1 - r[i]) for i in params['D'] for k in params['V']), "no late no penalty")
    m.addConstrs((tau[i, k] - params['l_i'][i] <= params['M'] * r[i] for i in params['D'] for k in params['V']), "no late no penalty")

    # 电池约束
    m.addConstrs((E[0, k] == params['B'] for k in params['V']), "initial_battery")
    # 运行途中
    m.addConstrs(
        (E[j, k] <= E[i, k] - params['b'] * params['time_matrix'][i, j] * x[i, j, k] + params['B'] * (1 - x[i, j, k])
         for i in params['N_0'] for j in params['N_dest'] if (j != i) for k in params['V']),
        "battery_usage")

    m.addConstrs(
        (E[j, k] >= E[i, k] - params['b'] * params['time_matrix'][i, j] * x[i, j, k] - params['B'] * (1 - x[i, j, k])
         for i in params['N_0'] for j in (params['C'] + params['charging']) if (j != i) for k
         in params['V']), "battery_usage")
    # 到了充电站
    m.addConstrs((E[j, k] <= params['B'] - params['b'] * params['time_matrix'][i, j] * x[i, j, k] + params['M'] * (
                1 - x[i, j, k]) for i in params['charging'] for j in params['N_2n1'] if (j != i)
                  for k in params['V']))
    m.addConstrs((E[j, k] >= params['B'] - params['b'] * params['time_matrix'][i, j] * x[i, j, k] - params['M'] * (
                1 - x[i, j, k]) for i in params['charging'] for j in params['N_2n1'] if (j != i)
                  for k in params['V']))

    # # 载重约束
    m.addConstrs(
        (C[i, k] + params['p_i'][i] * x[i, j, k] - params['Q'] * (1 - x[i, j, k]) <= C[j, k] for i in params['N_depot']
         for j in params['N_dest'] if (j != i) for k in params['V']), "capacity_transfer")
    m.addConstrs(C[0, k] == 0 for k in params['V'])
    m.addConstrs(C[2 * params['n'] + 1, k] == 0 for k in params['V'])

    # 确保决策变量的定义域
    m.addConstrs((x[i, j, k] >= 0 for i in params['N_depot'] for j in params['N_depot'] for k in params['V']),
                 "domain_x")
    m.addConstrs((tau[i, k] >= 0 for i in params['N_depot'] for k in params['V']), "domain_tau")
    m.addConstrs(y[i, k] == y[i + params['n'], k] for i in params['P'] for k in params['V'])
    m.addConstrs(E[i, k] >= 0 for i in params['N_depot'] for k in params['V'])

    return m

def print_params(params):
    # print("=== Model Parameters ===")
    # print(f"Number of restaurants: {params['n_r']}")
    # print(f"Number of customers: {params['n_c']}")
    # print(f"Number of vehicles: {params['v']}")
    # print(f"Battery capacity: {params['B']}")
    # print(f"Battery consumption rate: {params['b']}")
    # print(f"Maximum capacity: {params['Q']}")
    # print(f"Big M: {params['M']}")
    # print(f"Service time range: {params['time_range']}")
    # print(f"Time step size: {params['time_stepsize']}")
    # print(f"Latest delay time: {params['time_delay']}")
    print(f"ALl pdp nodes: {params['N']}")
    print(f"Numer of orders: {params['n']}")
    print(f"ALl pick nodes: {params['P']}")
    print(f"ALl delivery nodes: {params['D']}")
    print(f"Earlist time: {params['e_i']}")
    print(len(params['e_i']))
    print(f"Lastest time: {params['l_i']}")
    print(len(params['l_i']))
    print("========================")

def main(args):
    # 随机函数的参数设置
    stochasitc_setting = {
        'coordinates': np.random.uniform,
        'coordinates_params': {'low': 0, 'high': 5},
        'sample': np.random.randint,
        'sample_params': {'low': 0, 'high': 5},
        'demand': np.random.poisson,
        'demand_params': {'lam': 2},
    }

    # 从命令行中获取值
    n_r = args.n_r
    n_c = args.n_c
    time_range = args.time_range
    time_stepsize = args.time_stepsize
    time_delay = args.time_delay
    v = args.v
    B = args.B
    b = args.b
    Q = args.Q
    M = args.M
    # 包含所有参数的dict
    params = ParametersGenerator(n_r, n_c, stochasitc_setting, time_range, time_stepsize, time_delay, v, B, b, Q, M)
    # 打印参数
    print_params(params)
    # model
    model = create_model(params)

    # 模型求解
    # ...

if __name__ == "__main__":
    # 在命令行输入：python main.py -v 3 -B 25 -b 0.6 -Q 8 -M 1500 -n_r 3 -n_c 8 --time_range 180 --time_stepsize 15 --time_delay 90
    # 来修改默认参数
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-v", type=int, default=2, help="Number of vehicles")
    parser.add_argument("-B", type=float, default=20, help="Battery capacity")
    parser.add_argument("-b", type=float, default=0.5, help="Battery consumption rate")
    parser.add_argument("-Q", type=int, default=6, help="Maximum capacity")
    parser.add_argument("-M", type=int, default=1000, help="Big M")
    parser.add_argument("-n_r", type=int, default=1, help="Number of restaurants")
    parser.add_argument("-n_c", type=int, default=1, help="Number of customers")
    parser.add_argument("--time_range", type=int, default=10, help="Whole service time")
    parser.add_argument("--time_stepsize", type=int, default=10, help="Size of each time step")
    parser.add_argument("--time_delay", type=int, default=60, help="Latest delay time")
    args = parser.parse_args()

    main(args)
    print('end')



