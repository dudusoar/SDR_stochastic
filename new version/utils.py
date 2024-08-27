import matplotlib.pyplot as plt

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
