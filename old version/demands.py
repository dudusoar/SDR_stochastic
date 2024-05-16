import numpy as np
import pandas as pd

def generate_demand_table(pairs,time_intervals,
              sample_dist,sample_params,
              demand_dist,demand_params) -> pd.DataFrame:
    """
    Generate the demand table.
    :return: Pandas DataFrame, represents the demand table.
    """
    # Initialization
    demand_dict = {pair: [0] * len(time_intervals) for pair in pairs}

    # Traverse each time interval
    for idx, _ in enumerate(time_intervals):
        num_samples = sample_dist(**sample_params)
        # Randomly draw samples from all pairs (allowing repeats)
        sampled_pairs = np.random.choice(len(pairs), size=num_samples, replace=True)

        # Generate order quantity for each sampled pair
        for sample in sampled_pairs:
            pair = pairs[sample]
            order_quantity = 0
            # make sure the order_quantity is not 0
            while order_quantity <= 0:
              order_quantity = demand_dist(**demand_params)
            demand_dict[pair][idx] += order_quantity

    # Convert the dictionary to Pandas DataFrame
    demand_table = pd.DataFrame(demand_dict, index=[f"{start}-{end}" for start, end in time_intervals])
    # Transpose DataFrame, so each row represents a pair and each column represents a time interval
    demand_table = demand_table.T

    return demand_table

# Example usage
if __name__ == "__main__":
  pairs = [(1, 4), (2, 5), (3, 6)]
  time_intervals = ((0, 10), (10, 20), (20, 30))

  # Demand table
  demand_table = generate_demand_table(
      pairs, time_intervals,
      sample_dist=np.random.randint, sample_params={'low': 1, 'high': 10},
      demand_dist=np.random.poisson, demand_params={'lam': 5})

  print(demand_table)

  from utils import split_pairs_demand
  time_interval = '0-10'
  demand_dict = split_pairs_demand(pairs, demand_table, time_interval)
  print(demand_dict)
