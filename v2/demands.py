import numpy as np
import pandas as pd

def generate_demand_table(pairs, time_intervals, sample_dist, sample_params, demand_dist, demand_params) -> pd.DataFrame:
    """
    Generate the demand table with additional columns for pickup and delivery points.
    :param pairs: list of (pickup, delivery) tuples.
    :param time_intervals: list of time interval tuples.
    :param sample_dist: function to sample number of demands.
    :param sample_params: parameters for the sampling distribution function.
    :param demand_dist: function to generate demand values.
    :param demand_params: parameters for the demand distribution function.
    :return: Pandas DataFrame representing the demand table with additional columns for pickup and delivery.
    """
    # Initialize demand dictionary
    demand_dict = {pair: [0] * len(time_intervals) for pair in pairs}

    # Traverse each time interval and generate demands
    for idx, _ in enumerate(time_intervals):
        num_samples = sample_dist(**sample_params)
        sampled_pairs = np.random.choice(len(pairs), size=num_samples, replace=True)

        # Generate order quantity for each sampled pair
        for sample in sampled_pairs:
            pair = pairs[sample]
            order_quantity = 0
            # Ensure order_quantity is not 0
            while order_quantity <= 0:
                order_quantity = demand_dist(**demand_params)
            demand_dict[pair][idx] += order_quantity

    # Convert the dictionary to Pandas DataFrame
    demand_table = pd.DataFrame(demand_dict, index=[f"{start}-{end}" for start, end in time_intervals])
    # Transpose DataFrame, so each row represents a pair and each column represents a time interval
    demand_table = demand_table.T

    # Add columns for pickup and delivery points
    demand_table['Pickup'] = [pair[0] for pair in pairs]
    demand_table['Delivery'] = [pair[1] for pair in pairs]

    # Rearrange columns to move Pickup and Delivery to the front
    cols = ['Pickup', 'Delivery'] + [col for col in demand_table.columns if col not in ['Pickup', 'Delivery']]
    demand_table = demand_table[cols]

    # Reset index to start from 1
    demand_table.reset_index(drop=True, inplace=True)
    demand_table.index += 1

    return demand_table

def generate_order_details(demand_table, locations):
    """
    Generate a detailed DataFrame for each order based on the demand table and location coordinates.

    :param demand_table: DataFrame with demand information per time interval.
    :param locations: Dictionary with coordinates for each restaurant and customer.
    :return: DataFrame with detailed order information including real indices.
    """
    label_list = []
    t_list = []
    x_list = []
    y_list = []
    real_index_list = []

    # Iterate over each time interval column in demand_table
    for j in range(2, demand_table.shape[1]):  # Assuming the first two columns are 'Pickup' and 'Delivery'
        time_start = int(demand_table.columns[j].split('-')[0])  # Extract start of the time interval
        # Iterate over each order pair
        for i in range(demand_table.shape[0]):
            orders_count = demand_table.iloc[i, j]
            if orders_count > 0:
                # Generate order details for each order
                for _ in range(orders_count):
                    # Restaurant information
                    pickup_label = demand_table.iloc[i, 0]
                    delivery_label = demand_table.iloc[i, 1]
                    label_list.append(i * 2 + 1)
                    x_list.append(locations[pickup_label][0])
                    y_list.append(locations[pickup_label][1])
                    t_list.append(time_start)
                    real_index_list.append(pickup_label)

                    # Customer information
                    label_list.append(i * 2 + 2)
                    x_list.append(locations[delivery_label][0])
                    y_list.append(locations[delivery_label][1])
                    t_list.append(time_start + 10)  # Assuming some fixed travel or processing time
                    real_index_list.append(delivery_label)

    df_point = pd.DataFrame({
        "label": label_list,
        "x": x_list,
        "y": y_list,
        "earliest": t_list,
        "real_index": real_index_list
    })

    return df_point


# Example usage
if __name__ == "__main__":

    # 需求表测试
    # pairs = [(1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6)]
    # time_intervals = ((0, 10), (10, 20), (20, 30))
    # demand_table = generate_demand_table(
    #     pairs, time_intervals,
    #     sample_dist=np.random.randint, sample_params={'low': 1, 'high': 10},
    #     demand_dist=np.random.poisson, demand_params={'lam': 5})
    # print('打印需求表')
    # print(demand_table)

    # order pair测试
    from real_map import RealMap
    real_map = RealMap(n_r=2, n_c=4)
    coordinates = real_map.generate_coordinates(np.random.uniform, {'low': 0, 'high': 10})
    pairs = real_map.generate_pairs()
    time_intervals = ((0, 10), (10, 20), (20, 30))
    demand_table = generate_demand_table(
        pairs, time_intervals,
        sample_dist=np.random.randint, sample_params={'low': 1, 'high': 10},
        demand_dist=np.random.poisson, demand_params={'lam': 5})
    order_pairs_table = generate_order_details(demand_table, coordinates)
    print(order_pairs_table)


    # from utils import split_pairs_demand
    # time_interval = '0-10'
    # demand_dict = split_pairs_demand(pairs, demand_table, time_interval)
    # print('打印需求字典')
    # print(demand_dict)



