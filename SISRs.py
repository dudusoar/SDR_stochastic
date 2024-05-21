
# SISRs Class: The initialization method accepts a distance matrix, demand array, vehicle capacity, and initial solution.
# calculate_cost Method: Calculates the total cost of a solution.
# ruin Method: Ruins the current solution by removing adjacent strings of customers to create slack.
# recreate Method: Reinserts the removed customers using a greedy insertion method with a blinking mechanism.
# calculate_route_demand Method: Calculates the total demand for a specific route.
# fleet_minimization Method: Minimizes the fleet size based on an absences-based acceptance criterion.
# optimize Method: Uses simulated annealing to optimize the SISRs algorithm.

import random
import numpy as np

class SISRs:
    def __init__(self, distance_matrix, demand, vehicle_capacity, initial_solution):
        self.distance_matrix = distance_matrix
        self.demand = demand
        self.vehicle_capacity = vehicle_capacity
        self.solution = initial_solution
        self.absence_counters = np.zeros(len(demand))
        
    def calculate_cost(self, solution):
        cost = 0
        for route in solution:
            if route:
                cost += self.distance_matrix[0][route[0]]
                for i in range(1, len(route)):
                    cost += self.distance_matrix[route[i - 1]][route[i]]
                cost += self.distance_matrix[route[-1]][0]
        return cost
    
    def ruin(self, solution):
        c_bar = 10
        Lmax = 10
        
        num_customers = sum(len(route) for route in solution)
        k_max = 4 * c_bar / (1 + Lmax) - 1
        ks = int(np.floor(random.uniform(1, k_max + 1)))
        
        customers_to_remove = []
        for _ in range(ks):
            route = random.choice(solution)
            if route:
                l_max = min(Lmax, len(route))
                l = int(np.floor(random.uniform(1, l_max + 1)))
                start = random.randint(0, len(route) - l)
                customers_to_remove.extend(route[start:start + l])
                del route[start:start + l]
        
        absent_customers = set(customers_to_remove)
        return solution, absent_customers

    def recreate(self, solution, absent_customers):
        beta = 0.01
        
        for customer in absent_customers:
            best_cost = float('inf')
            best_position = None
            best_route = None
            
            for route in solution:
                if self.calculate_route_demand(route) + self.demand[customer] <= self.vehicle_capacity:
                    for i in range(len(route) + 1):
                        if random.uniform(0, 1) < 1 - beta:
                            new_route = route[:i] + [customer] + route[i:]
                            cost = self.calculate_cost([new_route])
                            if cost < best_cost:
                                best_cost = cost
                                best_position = i
                                best_route = route
            
            if best_route is None:
                best_route = []
                solution.append(best_route)
                best_position = 0
            
            best_route.insert(best_position, customer)
        
        return solution
    
    def calculate_route_demand(self, route):
        return sum(self.demand[customer] for customer in route)
    
    def fleet_minimization(self, initial_solution):
        best_solution = initial_solution
        for customer in range(len(self.demand)):
            self.absence_counters[customer] = 0
        
        while True:
            current_solution, absent_customers = self.ruin(initial_solution)
            new_solution = self.recreate(current_solution, absent_customers)
            new_absent_customers = sum(self.absence_counters[customer] for customer in absent_customers)
            
            if len(absent_customers) < len(new_absent_customers):
                initial_solution = new_solution
                for customer in absent_customers:
                    self.absence_counters[customer] += 1
                
                if len(absent_customers) == 0:
                    best_solution = new_solution
                    route_to_remove = min(best_solution, key=lambda r: sum(self.absence_counters[c] for c in r))
                    best_solution.remove(route_to_remove)
            
            if self.calculate_cost(best_solution) < self.calculate_cost(initial_solution):
                initial_solution = best_solution
                break
        
        return best_solution
    
    def optimize(self, iterations=1000, initial_temperature=100, final_temperature=1):
        current_solution = self.solution
        best_solution = current_solution
        T = initial_temperature
        cooling_rate = (final_temperature / initial_temperature) ** (1 / iterations)
        
        for _ in range(iterations):
            new_solution, absent_customers = self.ruin(current_solution)
            new_solution = self.recreate(new_solution, absent_customers)
            
            delta_cost = self.calculate_cost(new_solution) - self.calculate_cost(current_solution)
            if delta_cost < 0 or random.uniform(0, 1) < np.exp(-delta_cost / T):
                current_solution = new_solution
            
            if self.calculate_cost(new_solution) < self.calculate_cost(best_solution):
                best_solution = new_solution
            
            T *= cooling_rate
        
        return best_solution

if __name__ == "__main__":
    # 示例使用
    distance_matrix = np.array([
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0]
    ])
    demand = [0, 1, 2, 1]
    vehicle_capacity = 4
    initial_solution = [[1, 2], [3]]

    sirs = SISRs(distance_matrix, demand, vehicle_capacity, initial_solution)
    best_solution = sirs.optimize()
    print("Best Solution:", best_solution)
