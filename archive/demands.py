import numpy as np
import pandas as pd


class DemandGenerator:
    def __init__(self, time_range, time_step, restaurants, customers, random_params):
        """
        Initialize the DemandGenerator instance and generate the demand table.
        :param time_range: The total time range.
        :param time_step: The step size for each time interval.
        :param restaurants: List of restaurant indices.
        :param customers: List of customer indices.
        :param random_params: A dictionary containing the random functions and their parameters.
        """
        self.time_range = time_range
        self.time_step = time_step
        self.restaurants = restaurants
        self.customers = customers

        # Extract random functions and their parameters from the random_params dictionary
        self.sample_dist = random_params['sample_dist']['function']
        self.sample_params = random_params['sample_dist']['params']
        self.demand_dist = random_params['demand_dist']['function']
        self.demand_params = random_params['demand_dist']['params']

        self.time_intervals = self.generate_time_intervals()
        self.pairs = self.generate_pairs()
        self.demand_table = self.generate_demand_table()

    def generate_time_intervals(self) -> tuple:
        """
        Generate time intervals.
        :return: A tuple of time intervals.
        """
        intervals = tuple(
            (i, min(i + self.time_step, self.time_range)) for i in range(0, self.time_range, self.time_step))
        return intervals

    def generate_pairs(self):
        """
        Generate all restaurant-customer pairs.
        :return: A list of restaurant-customer pairs.
        """
        pairs = [(r, c) for r in self.restaurants for c in self.customers]
        return pairs

    def generate_demand_table(self) -> pd.DataFrame:
        """
        Generate the demand table with additional columns for pickup and delivery points.
        :return: A Pandas DataFrame representing the demand table.
        """
        # Initialize demand dictionary
        demand_dict = {pair: [0] * len(self.time_intervals) for pair in self.pairs}

        # Traverse each time interval and generate demands
        for idx, _ in enumerate(self.time_intervals):
            num_samples = self.sample_dist(**self.sample_params)
            sampled_pairs = np.random.choice(len(self.pairs), size=num_samples, replace=True)

            # Generate order quantity for each sampled pair
            for sample in sampled_pairs:
                pair = self.pairs[sample]
                order_quantity = 0
                # Ensure order_quantity is not 0
                while order_quantity <= 0:
                    order_quantity = self.demand_dist(**self.demand_params)
                demand_dict[pair][idx] += order_quantity

        # Convert the dictionary to Pandas DataFrame
        demand_table = pd.DataFrame(demand_dict, index=[f"{start}-{end}" for start, end in self.time_intervals])
        # Transpose DataFrame, so each row represents a pair and each column represents a time interval
        demand_table = demand_table.T

        # Add columns for pickup and delivery points
        demand_table['Pickup'] = [pair[0] for pair in self.pairs]
        demand_table['Delivery'] = [pair[1] for pair in self.pairs]

        # Rearrange columns to move Pickup and Delivery to the front
        cols = ['Pickup', 'Delivery'] + [col for col in demand_table.columns if col not in ['Pickup', 'Delivery']]
        demand_table = demand_table[cols]

        # Reset index to start from 1
        demand_table.reset_index(drop=True, inplace=True)
        demand_table.index += 1

        return demand_table


# 使用示例
if __name__ == "__main__":
    time_range = 30
    time_step = 10
    restaurants = [1, 2]
    customers = [3, 4, 5, 6]

    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 1, 'high': 3}
        },
        'demand_dist': {
            'function': np.random.poisson,
            'params': {'lam': 1}
        }
    }

    demands = DemandGenerator(time_range, time_step, restaurants, customers, random_params)
    print(demands.demand_table)
