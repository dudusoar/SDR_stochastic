import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable
import matplotlib.pyplot as plt
import seaborn as sns

class DemandGenerator:
    def __init__(self, time_range: int, time_step: int, restaurants: List[int], customers: List[int], random_params: Dict):
        """
        Initialize the DemandGenerator instance and generate the demand table.

        Args:
            time_range (int): The total time range.
            time_step (int): The step size for each time interval.
            restaurants (List[int]): List of restaurant indices.
            customers (List[int]): List of customer indices.
            random_params (Dict): A dictionary containing the random functions and their parameters.
        """
        self.time_range = time_range
        self.time_step = time_step
        self.restaurants = restaurants
        self.customers = customers

        self.sample_dist: Callable = random_params['sample_dist']['function']
        self.sample_params: Dict = random_params['sample_dist']['params']
        self.demand_dist: Callable = random_params['demand_dist']['function']
        self.demand_params: Dict = random_params['demand_dist']['params']

        self.time_intervals: List[Tuple[int, int]] = self._generate_time_intervals()
        self.pairs: List[Tuple[int, int]] = self._generate_pairs()
        self.demand_table: pd.DataFrame = self._generate_demand_table()

    def _generate_time_intervals(self) -> List[Tuple[int, int]]:
        """Generate time intervals."""
        return [(i, min(i + self.time_step, self.time_range)) for i in range(0, self.time_range, self.time_step)]

    def _generate_pairs(self) -> List[Tuple[int, int]]:
        """Generate all restaurant-customer pairs."""
        return [(r, c) for r in self.restaurants for c in self.customers]

    def _generate_demand_table(self) -> pd.DataFrame:
        """Generate the demand table with additional columns for pickup and delivery points."""
        demand_dict = {pair: np.zeros(len(self.time_intervals), dtype=int) for pair in self.pairs}

        for idx in range(len(self.time_intervals)):
            num_samples = self.sample_dist(**self.sample_params)
            sampled_pairs = np.random.choice(len(self.pairs), size=num_samples, replace=True)

            for sample in sampled_pairs:
                pair = self.pairs[sample]
                order_quantity = max(0, self.demand_dist(**self.demand_params))
                demand_dict[pair][idx] += order_quantity

        demand_table = pd.DataFrame(demand_dict, index=[f"{start}-{end}" for start, end in self.time_intervals]).T

        demand_table['Pickup'] = [pair[0] for pair in self.pairs]
        demand_table['Delivery'] = [pair[1] for pair in self.pairs]

        cols = ['Pickup', 'Delivery'] + [col for col in demand_table.columns if col not in ['Pickup', 'Delivery']]
        demand_table = demand_table[cols]

        return demand_table.reset_index(drop=True).rename_axis(None)

    def get_demand_table(self) -> pd.DataFrame:
        """Return the generated demand table."""
        return self.demand_table

    def plot_demand_heatmap(self):
        """Plot a heatmap of the demand table."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.demand_table.iloc[:, 2:], cmap="YlOrRd", annot=True, fmt="d")
        plt.title("Demand Heatmap")
        plt.xlabel("Time Intervals")
        plt.ylabel("Restaurant-Customer Pairs")
        plt.show()

if __name__ == "__main__":
    time_range = 30
    time_step = 10
    restaurants = [1, 2]
    customers = [3, 4, 5, 6]

    random_params = {
        'sample_dist': {
            'function': np.random.randint,
            'params': {'low': 7.5, 'high': 8}
        },
        'demand_dist': {
            'function': np.random.poisson,
            'params': {'lam': 10/3/8}
        }
    }

    demand_gen = DemandGenerator(time_range, time_step, restaurants, customers, random_params)
    print(demand_gen.get_demand_table())
    demand_gen.plot_demand_heatmap()

