import random
from instance import PDPTWInstance
from solution import PDPTWSolution
from solver import greedy_insertion_init
from operators import RemovalOperators, RepairOperators, SwapOperators
import numpy as np

def main():
    # 设置随机种子
    random.seed(42)

    # 参数设置
    n = 20  # pickup点的数量
    map_size = 3  # 地图大小
    speed = 4  # 车辆速度
    extra_time = 10  # delivery点时间窗口起始时间的额外时间
    num_vehicles = 5  # 车辆数量
    vehicle_capacity = 5  # 车辆容量
    battery_capacity = 240  # 电池容量
    battery_consume_rate = 1  # 电池消耗率

    # 生成PDPTW实例
    instance = PDPTWInstance(n, map_size, speed, extra_time, seed=1234)

    df = instance.to_dataframe()

    # Reference matrix for SISR removal
    df_orders = df.iloc[1:, :]
    stops = df_orders[['X', 'Y']].values
    e_times = df_orders[['ReadyTime']].values
    l_times = df_orders[['DueDate']].values

    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    d_matrix = []
    for i in range(n):
        d_list = []
        for j in range(n):
            d_1 = euclidean_distance(stops[i], stops[j])  # pickup-pickup
            t_1 = abs(e_times[i] - e_times[j]) / 60 * speed
            dt_1 = d_1 + t_1 * 0.1

            d_2 = euclidean_distance(stops[i], stops[j + n])  # pickup-dropoff
            t_2 = abs(e_times[i] - l_times[j + n]) / 60 * speed
            dt_2 = d_2 + t_2 * 0.1

            d_3 = euclidean_distance(stops[i + n], stops[j])  # dropoff-pickup
            t_3 = abs(e_times[j] - l_times[i + n]) / 60 * speed
            dt_3 = d_3 + t_3 * 0.1

            d_4 = euclidean_distance(stops[i + n], stops[j + n])  # dropoff-dropoff
            t_4 = abs(e_times[j + n] - l_times[i + n]) / 60 * speed
            dt_4 = d_4 + t_4 * 0.1

            d_min = min(dt_1, dt_2, dt_3, dt_4)[0]
            d_list.append(d_min)
        d_matrix.append(d_list)

    # 使用贪心插入法构建初始解
    initial_solution = greedy_insertion_init(instance, num_vehicles, vehicle_capacity, battery_capacity, battery_consume_rate)

    print("Initial Solution:")
    print(initial_solution.routes)
    initial_solution.plot_solution()

    # 创建移除算子
    removal_operators = RemovalOperators(initial_solution)

    # 移除数量和参数 p
    num_removals = 3
    p = 3

    # new_solution, removed_requests = removal_operators.shaw_removal(num_removals, p)
    new_solution, removed_requests = removal_operators.SISR_removal(200, 10, d_matrix)
    print("Removed requests:", removed_requests)
    print("New routes after removal:")
    print(new_solution.routes)
    print("Is new solution feasible?", new_solution.is_feasible())


if __name__ == "__main__":
    main()


