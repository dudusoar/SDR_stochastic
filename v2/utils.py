import numpy as np
import matplotlib.pyplot as plt

def generate_time_intervals(time_range:int, time_step:int)-> tuple:
    '''
    产生时间区间
    '''
    intervals = tuple((i, min(i+time_step,time_range)) for i in range(0,time_range,time_step))
    return intervals

def split_pairs_demand(pairs, demand_table, time_interval):
    """
    根据给定的时间间隔处理需求数据，区分提货点和送货点。

    :param pairs: 所有可能的提货点和送货点配对列表。
    :param demand_table: Pandas DataFrame格式的需求表。
    :param time_interval: 要处理的时间间隔。
    :return: 包含所有节点需求的字典。
    """
    demand_dict = {}
    # 特殊点需求为0
    n = len(pairs)
    for i in range(2 * n + 3):  # 包括所有特殊点
        demand_dict[i] = 0

    if demand_table is not None and time_interval is not None:
        for pair in pairs:
            pickup, delivery = pair
            total_demand = demand_table.loc[pair, time_interval]
            # 提货点需求为正，送货点需求为负
            demand_dict[pickup] = total_demand
            demand_dict[delivery] = -total_demand

    return demand_dict

if __name__ == '__main__':
    pairs = [(1, 4), (2, 5), (3, 4), (3,6)]
    n = max(pairs, key=lambda x: max(x))[1] // 2
    print(n)



