"""ALNS (Adaptive Large Neighborhood Search) solver for PDPTW problems.

This module implements the ALNS algorithm for solving PDPTW problems with
battery constraints, including charging station insertion.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
import time

# Local imports
from .operators import RemovalOperators, RepairOperators
from vrp_toolkit.problems.pdptw import PDPTWInstance, PDPTWSolution


@dataclass
class ALNSConfig:
    """Configuration parameters for ALNS algorithm."""
    
    # Operator parameters
    num_removal: int = 5
    p: float = 4.0  # Shaw removal parameter
    k: int = 3  # Regret insertion parameter
    L_max: int = 5  # SISR removal max string length
    avg_remove_order: float = 2.0  # SISR average remove order
    d_matrix: Optional[np.ndarray] = None  # Distance matrix for SISR
    
    # ALNS parameters
    max_no_improve: int = 100
    segment_length: int = 100
    num_segments: int = 10
    r: float = 0.1  # Weight update rate
    sigma: Tuple[float, float, float] = (33.0, 9.0, 13.0)  # Reward scores
    
    # Simulated annealing parameters
    start_temp: float = 10000.0
    cooling_rate: float = 0.99
    
    # Convergence criteria
    cost_ci_obj_diff_threshold: float = 0.1  # Convergence threshold
    cost_ci_window_size: int = 25  # Window size for best cost list
    
    # Operator indices (configurable)
    removal_indices: List[int] = None  # Default: [0, 2, 3] (Shaw, Worst, SISR)
    repair_indices: List[int] = None   # Default: [0, 1] (Greedy, Regret)
    
    # Charging station parameters
    charging_station_index: Optional[int] = None  # If None, assumes last index
    
    def __post_init__(self):
        """Set default values for lists."""
        if self.removal_indices is None:
            self.removal_indices = [0, 2, 3]  # Shaw, Worst, SISR
        if self.repair_indices is None:
            self.repair_indices = [0, 1]  # Greedy, Regret


def greedy_insertion_initial_solution(
    instance: PDPTWInstance,
    num_vehicles: int,
    vehicle_capacity: float,
    battery_capacity: float,
    battery_consume_rate: float,
    penalty_unvisit: float,
    penalty_delay: float
) -> PDPTWSolution:
    """Construct initial solution using greedy insertion.
    
    Args:
        instance: PDPTW problem instance
        num_vehicles: Number of available vehicles
        vehicle_capacity: Maximum capacity per vehicle
        battery_capacity: Maximum battery capacity
        battery_consume_rate: Battery consumption rate per distance unit
        penalty_unvisit: Penalty for unvisited nodes
        penalty_delay: Penalty for time window delays
        
    Returns:
        Initial feasible solution
    """
    routes = []
    pickup_nodes = list(range(1, instance.n + 1))
    pickup_nodes.sort(key=lambda x: instance.time_windows[x][0])  # Sort by pickup start time

    for vehicle_id in range(num_vehicles):
        route = [0, 0]

        while pickup_nodes:
            best_pickup_node = None
            best_insertion_index = None
            best_objective_value = float('inf')

            for pickup_node in pickup_nodes:
                delivery_node = pickup_node + instance.n
                for insertion_index in range(1, len(route)):
                    new_route = route[:insertion_index] + [pickup_node, delivery_node] + route[insertion_index:]
                    temp_solution = PDPTWSolution(
                        instance, vehicle_capacity, battery_capacity, battery_consume_rate,
                        [new_route], penalty_unvisit, penalty_delay
                    )

                    if temp_solution.is_feasible():
                        objective_value = temp_solution.objective_function()
                        if objective_value < best_objective_value:
                            best_pickup_node = pickup_node
                            best_insertion_index = insertion_index
                            best_objective_value = objective_value

            if best_pickup_node is not None:
                pickup_nodes.remove(best_pickup_node)
                route = route[:best_insertion_index] + [best_pickup_node, best_pickup_node + instance.n] + route[best_insertion_index:]
            else:
                break

        routes.append(route)

    solution = PDPTWSolution(
        instance, vehicle_capacity, battery_capacity, battery_consume_rate,
        routes, penalty_unvisit, penalty_delay
    )

    return solution


class ALNS:
    """Core ALNS algorithm implementation.
    
    This class implements the Adaptive Large Neighborhood Search algorithm
    for PDPTW problems with battery constraints and charging station insertion.
    """
    
    def __init__(
        self,
        initial_solution: PDPTWSolution,
        config: ALNSConfig,
        dist_matrix: np.ndarray,
        battery_capacity: float
    ):
        """Initialize ALNS solver.
        
        Args:
            initial_solution: Starting solution for ALNS
            config: Algorithm configuration parameters
            dist_matrix: Distance matrix for charging insertion calculations
            battery_capacity: Battery capacity for charging constraint
        """
        # Solution storage
        self.current_solution = deepcopy(initial_solution)
        self.best_solution = deepcopy(initial_solution)
        self.charging_solution = deepcopy(initial_solution)
        
        # Adjust charging solution battery capacity
        robot_speed = self.current_solution.instance.robot_speed
        self.charging_solution.battery_capacity = battery_capacity * 2 / robot_speed * 60
        
        self.best_charging_solution = None
        self.best_charging_route = []

        # Operator parameters from config
        self.num_removal = config.num_removal
        self.p = config.p
        self.k = config.k
        self.L_max = config.L_max
        self.avg_remove_order = config.avg_remove_order
        self.d_matrix = config.d_matrix
        
        self.dist_matrix = dist_matrix
        self.battery_capacity = battery_capacity

        # ALNS parameters from config
        self.max_no_improve = config.max_no_improve
        self.segment_length = config.segment_length
        self.num_segments = config.num_segments
        self.r = config.r
        self.sigma1, self.sigma2, self.sigma3 = config.sigma

        # Simulated annealing parameters
        self.start_temp = config.start_temp
        self.cooling_rate = config.cooling_rate
        self.current_temp = config.start_temp

        # Convergence criteria
        self.cost_ci_obj_diff_threshold = config.cost_ci_obj_diff_threshold
        self.cost_ci_window_size = config.cost_ci_window_size
        
        # Charging station index
        self.charging_station_index = config.charging_station_index
        if self.charging_station_index is None:
            self.charging_station_index = len(self.dist_matrix) - 1

        # ======== Initialization============
        # Operator indices
        self.removal_list = config.removal_indices
        self.repair_list = config.repair_indices

        # Weights initialization
        self.removal_weights = np.zeros((self.num_segments, len(self.removal_list)))
        self.repair_weights = np.zeros((self.num_segments, len(self.repair_list)))
        
        # Initial weights (equal distribution)
        self.removal_weights[0] = np.ones(len(self.removal_list)) / len(self.removal_list)
        self.repair_weights[0] = np.ones(len(self.repair_list)) / len(self.repair_list)

        # Scores
        self.removal_scores = np.zeros((self.num_segments, len(self.removal_list)))
        self.repair_scores = np.zeros((self.num_segments, len(self.repair_list)))

        # Theta: number of times each operator is used
        self.removal_theta = np.zeros((self.num_segments, len(self.removal_list)))
        self.repair_theta = np.zeros((self.num_segments, len(self.repair_list)))

    def select_operator(self, weights: np.ndarray) -> int:
        """Select operator using roulette wheel selection.
        
        Args:
            weights: Weight array for operators
            
        Returns:
            Index of selected operator
        """
        total_weight = np.sum(weights)
        probabilities = weights / total_weight
        cumulative_probabilities = np.cumsum(probabilities)
        random_number = random.random()
        for i, probability in enumerate(cumulative_probabilities):
            if random_number < probability:
                return i
        return len(weights) - 1  # select the last one

    def total_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route.
        
        Args:
            route: List of node indices
            
        Returns:
            Total distance of the route
        """
        dist = 0.0
        for i in range(len(route) - 1):
            dist += self.dist_matrix[route[i]][route[i + 1]]
        return dist

    def run(self) -> Tuple[PDPTWSolution, PDPTWSolution]:
        """Run the ALNS algorithm.
        
        Returns:
            Tuple of (best_solution, best_charging_solution)
        """
        num_no_improve = 0
        segment = 0
        r = self.r
        start_time = time.time()
        best_obj_diff = 100.0
        best_obj_list = []
        insert_index = self.charging_station_index

        cost_ci_best = float('inf')  # recording cost after charging insertion
        cost_ci_obj_diff = 100.0
        cost_ci_best_list = []

        while segment < self.num_segments and cost_ci_obj_diff > self.cost_ci_obj_diff_threshold:
            # (time and information)
            segment_start_time = time.time()
            print(f"Segment {segment + 1} / {self.num_segments}")

            # ================================== A new segment begins ==================================
            # Update the weights for the current segment
            if segment > 0:
                for i in range(len(self.removal_list)):
                    self.removal_weights[segment][i] = (
                        self.removal_weights[segment - 1][i] * (1 - r)
                        + r * self.removal_scores[segment - 1][i] / max(1, self.removal_theta[segment - 1][i])
                    )
                for i in range(len(self.repair_list)):
                    self.repair_weights[segment][i] = (
                        self.repair_weights[segment - 1][i] * (1 - r)
                        + r * self.repair_scores[segment - 1][i] / max(1, self.repair_theta[segment - 1][i])
                    )

            for iteration in range(self.segment_length):
                # ================================== select the operators ==================================
                ## removal 
                removal_operators = RemovalOperators(self.current_solution)
                removal_idx = self.select_operator(self.removal_weights[segment])

                if removal_idx == 0:
                    removed_solution = removal_operators.shaw_removal(self.num_removal, self.p)
                elif removal_idx == 1:
                    removed_solution = removal_operators.random_removal(self.num_removal)
                elif removal_idx == 2:
                    removed_solution = removal_operators.worst_removal(self.num_removal)
                elif removal_idx == 3:
                    removed_solution = removal_operators.SISR_removal(self.L_max, self.avg_remove_order, self.d_matrix)

                removed_solution.update_all()
                unvisited_pairs = removed_solution.unvisited_pairs

                ## repair
                repair_operators = RepairOperators(removed_solution)
                repair_idx = self.select_operator(self.repair_weights[segment])
                if repair_idx == 0:
                    repair_solution = repair_operators.greedy_insertion(unvisited_pairs)
                elif repair_idx == 1:
                    repair_solution = repair_operators.regret_insertion(unvisited_pairs, self.k)

                # ================================== update the count ==================================
                self.removal_theta[segment][removal_idx] += 1
                self.repair_theta[segment][repair_idx] += 1

                # ================================== update the scores ==================================
                repair_solution.update_all()
                new_objective = repair_solution.objective_function()

                current_objective = self.current_solution.objective_function()
                best_objective = self.best_solution.objective_function()

                if new_objective < best_objective:  # sigma1
                    self.best_solution = deepcopy(repair_solution)
                    self.current_solution = deepcopy(repair_solution)
                    num_no_improve = 0
                    self.removal_scores[segment][removal_idx] += self.sigma1
                    self.repair_scores[segment][repair_idx] += self.sigma1
                elif new_objective < current_objective:  # sigma2
                    self.current_solution = deepcopy(repair_solution)
                    num_no_improve = 0
                    self.removal_scores[segment][removal_idx] += self.sigma2
                    self.repair_scores[segment][repair_idx] += self.sigma2
                else:  # sigma3
                    acceptance_probability = np.exp(-(new_objective - current_objective) / self.current_temp)
                    if random.random() < acceptance_probability:
                        self.current_solution = deepcopy(repair_solution)
                        self.removal_scores[segment][removal_idx] += self.sigma3
                        self.repair_scores[segment][repair_idx] += self.sigma3
                    num_no_improve += 1
    
                #================================== Add charging insertion ==================================
                if segment > 0:
                    z = 0 # number of routes which does not need charge
                    routes_charge = deepcopy(repair_solution.routes)
                    
                    for route_id, route_1 in enumerate(routes_charge):
                        
                        route_best = []
                        
                        if self.total_distance(route_1) <= self.battery_capacity:
                            z += 1
                            continue
                
                        c_best = float('inf')
                        
                        for i in range(2, len(route_1) - 1):
                            route_copy = route_1[:i] + [insert_index] + route_1[i:]

                            if self.total_distance(route_copy) > 2 * self.battery_capacity:
                                continue
                            else:
                                self.charging_solution.routes[route_id] = route_copy
                                self.charging_solution.update_all()
                                # update the best objective function
                                cr_best = self.charging_solution.objective_function()
                                if cr_best < c_best:
                                    second_index = route_copy.index(insert_index)
                                    subroute_1 = route_copy[:second_index + 1]
                                    subroute_2 = route_copy[second_index:]
                                    dist_1 = self.total_distance(subroute_1)
                                    dist_2 = self.total_distance(subroute_2)
                                    if (dist_1 <= self.battery_capacity) & (dist_2 <= self.battery_capacity):
                                        c_best = cr_best
                                        route_best = deepcopy(route_copy)
                                        routes_charge[route_id] = route_best
                        if len(route_best) > 0:
                            z += 1

                    if z == len(routes_charge):
                        self.charging_solution.routes = routes_charge
                        self.charging_solution.update_all()
                        cost_ci = self.charging_solution.objective_function()
                        if cost_ci < cost_ci_best:
                            cost_ci_best = cost_ci
                            self.best_charging_solution = deepcopy(self.charging_solution)
                    cost_ci_best_list.append(cost_ci_best)

                if len(cost_ci_best_list) >= self.cost_ci_window_size:
                    cost_ci_best_list = cost_ci_best_list[-self.cost_ci_window_size:]
                    cost_ci_obj_diff = np.mean(cost_ci_best_list) - cost_ci_best
                else:
                    cost_ci_obj_diff = 100.0

            print(f"Best objective: {best_objective}, Best charging cost: {cost_ci_best}, Diff: {cost_ci_obj_diff}")

            # (time spent on this segment)
            segment_end_time = time.time()
            segment_duration = segment_end_time - segment_start_time
            print(f"Segment {segment + 1} completed in {segment_duration:.2f} seconds")

            # update the segment, temperature
            segment += 1
            self.current_temp *= self.cooling_rate

            # === End of the segment

        # (time spend on the whole process)
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"ALNS run completed in {total_duration:.2f} seconds")

        return self.best_solution, self.best_charging_solution

    # ============================================= plot ==========================================================
    def plot_scores(self):
        """Plot operator scores over segments."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        segments = range(self.removal_scores.shape[0])

        # removal scores
        for i in range(len(self.removal_list)):
            plt.plot(segments, self.removal_scores[:, i],
                     label=f'Shaw Removal' if i == 0 else (f'Random Removal' if i == 1 else 'Worst Removal'))

        # repair scores
        for i in range(len(self.repair_list)):
            plt.plot(segments, self.repair_scores[:, i], label=f'Greedy Insertion' if i == 0 else 'Regret Insertion')

        plt.xlabel('Segment')
        plt.ylabel('Scores')
        plt.title('Scores of Operators')
        plt.xticks(segments)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_theta(self):
        """Plot operator usage counts over segments."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        segments = range(self.removal_theta.shape[0])

        # removal theta
        for i in range(len(self.removal_list)):
            plt.plot(segments, self.removal_theta[:, i], label=f'Shaw Removal' if i == 0 else (
                f'Worst Removal' if i == 1 else f'SISR Removal' if i == 2 else 'SISR Removal'))

        # repair theta
        for i in range(len(self.repair_list)):
            plt.plot(segments, self.repair_theta[:, i], label=f'Greedy Insertion' if i == 0 else 'Regret Insertion')

        plt.xlabel('Segment')
        plt.ylabel('Theta (Usage Count)')
        plt.title('Usage Count of Operators')
        plt.xticks(segments)
        plt.legend()
        plt.grid(True)
        plt.show()