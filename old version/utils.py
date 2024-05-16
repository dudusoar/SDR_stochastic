import numpy as np
import matplotlib.pyplot as plt

def generate_time_intervals(time_range:int, time_step:int)-> tuple:
    '''
    产生时间区间
    '''
    intervals = tuple((i, min(i+time_step,time_range)) for i in range(0,time_range,time_step))
    return intervals

def plot_nodes(pairs, coordinates, demand_table=None, time_interval=None):
    """
    绘制所有节点，并可选地显示每个节点的累计需求和所属时间间隔。

    :param coordinates: 包含所有节点坐标的字典。
    :param demand_table: 可选的，Pandas DataFrame格式的需求表。
    :param time_interval: 可选的，表示当前需求表对应的时间间隔。 e.g. 'o-10'
    """
    n = len(pairs)

    # 假设最后三个节点是特殊节点
    special_coords = {0: 'Depot', 2*n+1: 'Destination', 2*n+2: 'Charging Station'}
    fig, ax = plt.subplots()

    # 绘制提货点和送货点
    for node, (x, y) in coordinates.items():
        if node in special_coords:  # 特殊节点
            ax.scatter(x, y, s=100, label=special_coords[node], edgecolors='black', facecolors='none')
        else:
            color = 'blue' if node <= n else 'green'  # 提货点蓝色，送货点绿色
            ax.scatter(x, y, color=color)

            # 如果提供了demand_table，则显示累计需求
            if demand_table is not None and time_interval is not None:
                # 根据time_interval选取需求量
                total_demand = demand_table.loc[node, time_interval] if node in demand_table.index else 0
                total_demand = np.int32(total_demand)
                ax.text(x, y, f"[{node}, {total_demand}]", fontsize=9, ha='right')

    # 添加图例
    ax.legend()

    # 添加轴标签和标题
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    if time_interval is not None:
        ax.set_title(f'Demand for Time Interval: {time_interval}')
    else:
        ax.set_title('Node Locations')

    # 显示图形
    plt.show()


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



