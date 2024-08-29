import matplotlib.pyplot as plt
from collections import defaultdict

def plot_real_map(real_map, show_index=False):
    """
    Plot real map
    :param show_index: if True，show the indices of the nodes
    """
    # index and coordinates
    coordinates = real_map.coordinates
    restaurants = real_map.restaurants
    customers = real_map.customers
    restaurant_coords = [coordinates[i] for i in restaurants]
    customer_coords = [coordinates[i] for i in customers]

    # plot
    plt.figure(figsize=(8, 8))

    # plot depot
    plt.scatter(coordinates[0][0], coordinates[0][1], c='red', marker='s', s=100, label='Depot')
    # plot restaurants
    plt.scatter([p[0] for p in restaurant_coords], [p[1] for p in restaurant_coords], c='blue', marker='o',
                label='Restaurant')
    # plot customers
    plt.scatter([d[0] for d in customer_coords], [d[1] for d in customer_coords], c='green', marker='d',
                label='Customer')

    # add indices
    if show_index:
        for i in restaurants:
            plt.annotate(f'R{i}', (coordinates[i][0], coordinates[i][1]), textcoords='offset points',
                         xytext=(0, 5), ha='center')
        for i in customers:
            plt.annotate(f'C{i}', (coordinates[i][0], coordinates[i][1]), textcoords='offset points',
                         xytext=(0, 5), ha='center')

    # set the ranges for X axis and Y axis
    x_coords = [coord[0] for coord in coordinates.values()]
    y_coords = [coord[1] for coord in coordinates.values()]
    plt.xlim(min(x_coords) - 0.5, max(x_coords) + 0.5)
    plt.ylim(min(y_coords) - 0.5, max(y_coords) + 0.5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('RealMap Instance')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_instance(order_df):
    """
    Plot the PDPTW instance orders, highlighting overlapping locations with order IDs.
    :param order_df: DataFrame containing the order details.
    """
    plt.figure(figsize=(8, 8))

    # Create a dictionary to track all orders pointing to the same location
    location_orders = defaultdict(list)

    # Initialize a set to track labels already added to the legend
    labels_added = set()

    for _, row in order_df.iterrows():
        location = (row['X'], row['Y'])
        location_orders[location].append(row['ID'])

        # Determine the label only if it hasn't been added yet
        if row['Type'] == 'cp':
            label = 'Pickup' if 'Pickup' not in labels_added else ''
            plt.scatter(row['X'], row['Y'], c='blue', marker='o', s=100, label=label)
            labels_added.add('Pickup')
        elif row['Type'] == 'cd':
            label = 'Delivery' if 'Delivery' not in labels_added else ''
            plt.scatter(row['X'], row['Y'], c='green', marker='d', s=100, label=label)
            labels_added.add('Delivery')
        elif row['Type'] == 'depot':
            label = 'Depot' if 'Depot' not in labels_added else ''
            plt.scatter(row['X'], row['Y'], c='red', marker='s', s=100, label=label)
            labels_added.add('Depot')
        elif row['Type'] == 'charging':
            label = 'Charging Station' if 'Charging Station' not in labels_added else ''
            plt.scatter(row['X'], row['Y'], c='purple', marker='^', s=100, label=label)
            labels_added.add('Charging Station')

    # Annotate overlapping points
    for location, orders in location_orders.items():
        if len(orders) > 1:
            plt.text(location[0], location[1] + 0.02, f"[{', '.join(map(str, orders))}]", fontsize=8, ha='center')
        else:
            plt.text(location[0], location[1] + 0.02, f"{orders[0]}", fontsize=8, ha='center')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('PDPTW Orders Plot')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()