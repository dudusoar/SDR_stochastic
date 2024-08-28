import numpy as np
import pandas as pd

class OrdersGenerator:
    def __init__(self, demand_table, time_params, node_type_dict):
        """
        Initialize the OrderPairGenerator instance.
        :param demand_table: DataFrame with demand information per time interval.
        :param time_params: Dict of
        """
        self.demand_table = demand_table
        # total number of orders
        self.total_orders = int(demand_table.iloc[:, 2:].sum().sum())

        # time related
        self.time_window_length = time_params['time_window_length']
        self.service_time = time_params['service_time']
        self.delivery_time_buffer = time_params['delivery_time_buffer']

        # node type
        self.node_types_dict = node_type_dict

        # Generate index lists
        self.order_pairs = []
        self.pickup_dict = {}
        self.delivery_dict = {}
        # Generate order pairs and dictionaries
        self.generate_order_pairs()

    def generate_order_pairs(self):
        """
        Generate detailed order pairs and populate the pickup and delivery dictionaries.
        """
        count = 0
        for j in range(2, self.demand_table.shape[1]):  # Columns excluding 'Pickup' and 'Delivery'
            time_start = int(self.demand_table.columns[j].split('-')[0])  # Start time of the interval
            for i in range(self.demand_table.shape[0]):
                orders_count = self.demand_table.iloc[i, j]
                if orders_count > 0:
                    pickup_label = self.demand_table.iloc[i, 0]
                    delivery_label = self.demand_table.iloc[i, 1]

                    for _ in range(orders_count):
                        count += 1
                        self.order_pairs.append((pickup_label, delivery_label))
                        # Store pickup information
                        self.pickup_dict[count] = pickup_label
                        # Store delivery information
                        self.delivery_dict[count + self.total_orders] = delivery_label

    def generate_order_table(self):
        """
        Generate the order table with the specified columns.
        :return: Pandas DataFrame containing the order details.
        """
        order_table_data = []

        for order_index in range(1, self.total_orders + 1):
            # Pickup details
            pickup_real_index = self.pickup_dict[order_index]
            delivery_real_index = self.delivery_dict[order_index + self.total_orders]

            # Pickup entry
            order_table_data.append([
                order_index,  # ID
                self.node_types_dict[pickup_real_index],  # Type
                1,  # Demand for pickup
                float('inf'),  # DueDate for pickup is infinity
                self.service_time,  # Service Time
                order_index + self.total_orders,  # PartnerID
                pickup_real_index  # Real index
            ])

            # Delivery entry
            order_table_data.append([
                order_index + self.total_orders,  # ID
                self.node_types_dict[delivery_real_index],  # Type
                -1,  # Demand for delivery
                self.time_window_length + self.delivery_time_buffer,  # DueDate
                self.service_time,  # Service Time
                order_index,  # PartnerID
                delivery_real_index  # Real index
            ])

        # Convert the list to a DataFrame
        order_table = pd.DataFrame(order_table_data, columns=[
            'ID', 'Type', 'Demand', 'DueDate', 'ServiceTime', 'PartnerID', 'RealIndex'
        ])

        return order_table

# Example usage
if __name__ == "__main__":
    demand_table = pd.DataFrame({
        "Pickup": [1, 1, 2, 2],
        "Delivery": [3, 4, 5, 6],
        "0-10": [2, 1, 0, 3],
        "10-20": [1, 0, 2, 0],
        "20-30": [0, 2, 1, 1]
    })

    time_params = {
        'time_window_length': 30,
        'service_time': 5,
        'delivery_time_buffer': 10
    }

    node_type_dict = {
        1: 'restaurant',
        2: 'restaurant',
        3: 'customer',
        4: 'customer',
        5: 'customer',
        6: 'customer'
    }

    generator = OrdersGenerator(demand_table, time_params, node_type_dict)
    order_table = generator.generate_order_table()
    print(order_table)