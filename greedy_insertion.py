# greedy_insertion function:
# Sort orders by earliest service time.
# For each order, try to insert it into each vehicle's route to find the optimal insertion location.
# If a feasible insertion location cannot be found, try to insert the order by delaying the delivery time.
# If it still cannot be inserted, mark the order as unassigned.


# can_insert function:
# Check if it is possible to insert an order at a given location on a given vehicle.
# Check the time window constraints, power constraints and load constraints.


# can_insert_late function:
# Check if the order can be inserted by extending the delivery time.
# Check the time window constraints, power constraints and load constraints.


# insert function:
# Insert an order at a given location for a given vehicle.
# Update the vehicle's arrival time, battery capacity and load.
def greedy_insertion(params):
    solution = Solution(params)

    # 按最早服务时间对订单进行排序
    sorted_orders = sorted(range(1, params['n'] + 1), key=lambda x: params['e_i'][x])

    for order in sorted_orders:
        pickup, delivery = order, order + params['n']

        # 找到一辆可行的车辆
        assigned = False
        for k in range(solution.v):
            for i in range(len(solution.routes[k]) + 1):
                if can_insert(solution, k, i, pickup, delivery, params):
                    insert(solution, k, i, pickup, delivery, params)
                    assigned = True
                    break
            if assigned:
                break

        # 如果没有找到可行的车辆,尝试延后插入
        if not assigned:
            for k in range(solution.v):
                for i in range(len(solution.routes[k]) + 1):
                    if can_insert_late(solution, k, i, pickup, delivery, params):
                        insert(solution, k, i, pickup, delivery, params)
                        assigned = True
                        break
                if assigned:
                    break

        # 如果还是没有找到可行解,则将该订单标记为未分配
        if not assigned:
            print(f"Order {order} cannot be assigned to any vehicle.")

    return solution


def can_insert(solution, k, i, pickup, delivery, params):
    """检查在车辆k的路线的位置i处插入取货点pickup和交货点delivery是否可行"""
    route = solution.routes[k]

    # 检查取货点和交货点是否满足时间窗约束
    earliest_pickup_time = max(params['e_i'][pickup],
                               solution.arrival_times[k][i - 1] + params['time_matrix'][route[i - 1]][pickup])
    earliest_delivery_time = max(params['e_i'][delivery],
                                 earliest_pickup_time + params['time_matrix'][pickup][delivery])
    if earliest_pickup_time > params['l_i'][pickup] or earliest_delivery_time > params['l_i'][delivery]:
        return False

    # 检查在取货点和交货点之后的节点是否满足时间窗约束
    for j in range(i, len(route)):
        node = route[j]
        earliest_time = max(params['e_i'][node], earliest_delivery_time + params['time_matrix'][delivery][node])
        if earliest_time > params['l_i'][node]:
            return False
        earliest_delivery_time = earliest_time

    # 检查电量约束
    battery = solution.battery_levels[k][i - 1] - params['b'] * params['time_matrix'][route[i - 1]][pickup]
    if battery < params['b'] * params['time_matrix'][pickup][delivery]:
        return False
    battery -= params['b'] * params['time_matrix'][pickup][delivery]
    for j in range(i, len(route)):
        node = route[j]
        if battery < params['b'] * params['time_matrix'][delivery][node]:
            return False
        battery -= params['b'] * params['time_matrix'][delivery][node]

    # 检查载重约束
    capacity = solution.capacities[k][i - 1] + params['p_i'][pickup]
    if capacity > params['Q']:
        return False
    capacity += params['p_i'][delivery]
    for j in range(i, len(route)):
        node = route[j]
        capacity += params['p_i'][node]
        if capacity > params['Q']:
            return False

    return True


def can_insert_late(solution, k, i, pickup, delivery, params):
    """检查是否可以通过延后交货时间来插入订单"""
    route = solution.routes[k]

    # 检查取货点是否满足时间窗约束
    earliest_pickup_time = max(params['e_i'][pickup],
                               solution.arrival_times[k][i - 1] + params['time_matrix'][route[i - 1]][pickup])
    if earliest_pickup_time > params['l_i'][pickup]:
        return False

    # 检查在取货点之后的节点是否满足时间窗约束
    earliest_delivery_time = earliest_pickup_time + params['time_matrix'][pickup][delivery]
    for j in range(i, len(route)):
        node = route[j]
        earliest_time = max(params['e_i'][node], earliest_delivery_time + params['time_matrix'][delivery][node])
        if earliest_time > params['l_i'][node]:
            return False
        earliest_delivery_time = earliest_time

    # 检查电量和载重约束
    battery = solution.battery_levels[k][i - 1] - params['b'] * params['time_matrix'][route[i - 1]][pickup]
    capacity = solution.capacities[k][i - 1] + params['p_i'][pickup]
    if battery < params['b'] * params['time_matrix'][pickup][delivery] or capacity > params['Q']:
        return False
    battery -= params['b'] * params['time_matrix'][pickup][delivery]
    capacity += params['p_i'][delivery]
    for j in range(i, len(route)):
        node = route[j]
        if battery < params['b'] * params['time_matrix'][delivery][node] or capacity > params['Q']:
            return False
        battery -= params['b'] * params['time_matrix'][delivery][node]
        capacity += params['p_i'][node]

    return True


def insert(solution, k, i, pickup, delivery, params):
    """在车辆k的路线的位置i处插入取货点pickup和交货点delivery"""
    route = solution.routes[k]
    route.insert(i, pickup)
    route.insert(i + 1, delivery)

    # 更新到达时间
    solution.arrival_times[k][i] = max(params['e_i'][pickup],
                                       solution.arrival_times[k][i - 1] + params['time_matrix'][route[i - 1]][pickup])
    solution.arrival_times[k][i + 1] = max(params['e_i'][delivery],
                                           solution.arrival_times[k][i] + params['time_matrix'][pickup][delivery])
    for j in range(i + 2, len(route)):
        node = route[j]
        solution.arrival_times[k][j] = max(params['e_i'][node],
                                           solution.arrival_times[k][j - 1] + params['time_matrix'][route[j - 1]][node])

    # 更新电量
    solution.battery_levels[k][i] = solution.battery_levels[k][i - 1] - params['b'] * \
                                    params['time_matrix'][route[i - 1]][pickup]
    solution.battery_levels[k][i + 1] = solution.battery_levels[k][i] - params['b'] * params['time_matrix'][pickup][
        delivery]
    for j in range(i + 2, len(route)):
        node = route[j]
        solution.battery_levels[k][j] = solution.battery_levels[k][j - 1] - params['b'] * \
                                        params['time_matrix'][route[j - 1]][node]

    # 更新载重
    solution.capacities[k][i] = solution.capacities[k][i - 1] + params['p_i'][pickup]
    solution.capacities[k][i + 1] = solution.capacities[k][i] + params['p_i'][delivery]
    for j in range(i + 2, len(route)):
        node = route[j]
        solution.capacities[k][j] = solution.capacities[k][j - 1] + params['p_i'][node]