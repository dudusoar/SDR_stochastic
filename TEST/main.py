import random
from instance import PDPTWInstance
from solution import PDPTWSolution
from solver import greedy_insertion_init
from operators import RemovalOperators, RepairOperators, SwapOperators

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

    new_solution, removed_requests = removal_operators.shaw_removal(num_removals, p)
    print("Removed requests:", removed_requests)
    print("New routes after removal:")
    print(new_solution.routes)
    print("Is new solution feasible?", new_solution.is_feasible())


if __name__ == "__main__":
    main()


