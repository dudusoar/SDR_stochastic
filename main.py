from solution_v2 import Solution
from solver import GreedyInsertion

# Initialize parameters
params = # ...

# Create the solver instance
solver = GreedyInsertion(params)

# Solve the problem
initial_routes = solver.solve()

# Update the solution and evaluate
solution = Solution(params)
solution.routes = initial_routes
solution.check_feasibility()
if solution.feasible:
    obj_value = solution.evaluate()
    print(f"Objective value: {obj_value}")
else:
    print("Solution is not feasible.")