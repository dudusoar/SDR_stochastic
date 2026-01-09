"""
VRP-Toolkit Playground - Interactive Learning Interface

Stage 1 MVP: Basic problem → solve → visualize workflow
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import vrp_toolkit
sys.path.insert(0, str(Path(__file__).parent.parent / "vrp-toolkit"))

try:
    from vrp_toolkit.problems.pdptw import PDPTWInstance
    from vrp_toolkit.algorithms.alns import ALNS, ALNSConfig, greedy_insertion_initial_solution
    from vrp_toolkit.algorithms.base import PDPTWProblemAdapter
    from vrp_toolkit.data.generators import OrderGenerator, DemandGenerator
    from vrp_toolkit.data.map import RealMap
    from vrp_toolkit.visualization.problem import PDPTWVisualizer
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("💡 Make sure vrp-toolkit is installed: `cd vrp-toolkit && uv pip install -e .`")
    st.stop()

# Page config
st.set_page_config(
    page_title="VRP-Toolkit Playground",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'instance' not in st.session_state:
    st.session_state.instance = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'cost_history' not in st.session_state:
    st.session_state.cost_history = None

# Title
st.title("🚀 VRP-Toolkit Playground")
st.markdown("**Interactive learning environment for Vehicle Routing Problems**")
st.markdown("---")

# Status indicator
col1, col2, col3 = st.columns(3)
with col1:
    instance_status = "✅" if st.session_state.instance is not None else "⏳"
    st.markdown(f"**Instance:** {instance_status}")
with col2:
    solution_status = "✅" if st.session_state.solution is not None else "⏳"
    st.markdown(f"**Solution:** {solution_status}")
with col3:
    if st.session_state.instance is not None:
        node_count = len(st.session_state.instance.indices)
        st.markdown(f"**Nodes:** {node_count}")
    else:
        st.markdown("**Nodes:** --")

st.markdown("---")

# Step 1: Problem Definition
st.header("📦 Step 1: Define Problem")

with st.expander("ℹ️ What is PDPTW?", expanded=False):
    st.markdown("""
    **Pickup-Delivery Problem with Time Windows (PDPTW)**

    - **Vehicles** pick up items from restaurants and deliver to customers
    - **Time windows** constrain when pickups/deliveries can occur
    - **Battery constraints** limit vehicle range (with charging stations)
    - **Objective** minimize total travel distance
    """)

col1, col2 = st.columns(2)

with col1:
    num_orders = st.slider(
        "Number of Orders",
        min_value=2,
        max_value=20,
        value=5,
        help="More orders = harder problem"
    )

with col2:
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        help="Same seed = same instance (for reproducibility)"
    )

if st.button("🎲 Generate Synthetic Instance", type="primary"):
    with st.spinner("Generating instance..."):
        try:
            # Set random seed for reproducibility
            np.random.seed(seed)

            # Generate synthetic map
            num_restaurants = max(2, num_orders // 3)
            real_map = RealMap(
                n_r=num_restaurants,
                n_c=num_orders,
                dist_function=np.random.uniform,
                dist_params={'low': 0, 'high': 100}
            )

            # Generate demands
            demand_gen = DemandGenerator(
                time_range=240,  # 4 hours in minutes
                time_step=30,     # 30-minute intervals
                restaurants=real_map.restaurants,
                customers=real_map.customers,
                random_params={
                    'sample_dist': {
                        'function': np.random.poisson,
                        'params': {'lam': 3}  # Average 3 orders per time interval
                    },
                    'demand_dist': {
                        'function': np.random.randint,
                        'params': {'low': 1, 'high': 5}  # 1-4 items per order
                    }
                }
            )
            demand_table = demand_gen.demand_table  # DemandGenerator generates on init

            # Generate orders
            order_gen = OrderGenerator(
                real_map=real_map,
                demand_table=demand_table,
                time_params={
                    'time_window_length': 30,
                    'service_time': 5,
                    'extra_time': 10,
                    'big_time': 1000
                },
                robot_speed=1.0
            )
            order_table = order_gen.order_table  # OrderGenerator generates on init

            # Create instance with all required parameters
            instance = PDPTWInstance(
                order_table=order_table,
                distance_matrix=real_map.distance_matrix,
                time_matrix=order_gen.time_matrix,
                robot_speed=1.0
            )
            st.session_state.instance = instance

            st.success(f"✅ Generated instance with {instance.n} orders and {len(instance.indices)} total nodes")

        except Exception as e:
            st.error(f"❌ Error generating instance: {e}")
            st.exception(e)

# Show instance summary if exists
if st.session_state.instance is not None:
    instance = st.session_state.instance

    st.subheader("📋 Instance Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Orders", instance.n)
    with col2:
        st.metric("Total Nodes", len(instance.indices))
    with col3:
        st.metric("Depot", instance.depot_index)
    with col4:
        has_charging = hasattr(instance, 'charging_index') and instance.charging_index is not None
        st.metric("Charging Station", "Yes" if has_charging else "No")

st.markdown("---")

# Step 2: Algorithm Configuration
st.header("⚙️ Step 2: Configure ALNS Algorithm")

with st.expander("ℹ️ What is ALNS?", expanded=False):
    st.markdown("""
    **Adaptive Large Neighborhood Search (ALNS)**

    An iterative improvement algorithm that:
    1. **Destroy** - Remove some requests from current solution
    2. **Repair** - Reinsert removed requests in better positions
    3. **Accept** - Accept improvement or explore worse solutions (simulated annealing)
    4. **Adapt** - Adjust operator weights based on performance
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Parameters")

    max_iterations = st.slider(
        "Max Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="More iterations = better solution (but slower)"
    )

    num_vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=10,
        value=3,
        help="Fleet size"
    )

    start_temp = st.number_input(
        "Start Temperature",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
        help="Higher = more exploration (accepts worse solutions initially)"
    )

with col2:
    st.subheader("Advanced Parameters")

    with st.expander("Show Advanced Options"):
        cooling_rate = st.slider(
            "Cooling Rate",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="How fast temperature decreases"
        )

        segment_length = st.number_input(
            "Segment Length",
            min_value=10,
            max_value=500,
            value=100,
            help="Iterations before adapting operator weights"
        )

        battery_capacity = st.number_input(
            "Battery Capacity",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=10.0,
            help="Vehicle battery capacity (recommended: 100.0 or higher for feasible solutions)"
        )

st.markdown("---")

# Step 3: Run Solver
st.header("🚀 Step 3: Run Solver")

if st.session_state.instance is None:
    st.warning("⚠️ Please generate an instance first (Step 1)")
else:
    if st.button("▶️ Run ALNS Solver", type="primary"):
        with st.spinner("🔄 Running ALNS algorithm..."):
            try:
                # Set seed for reproducibility
                np.random.seed(seed)

                # Create problem adapter
                problem = PDPTWProblemAdapter(st.session_state.instance)

                # Generate initial solution
                initial_solution = greedy_insertion_initial_solution(
                    problem=problem,
                    num_vehicles=num_vehicles,
                    vehicle_capacity=1000,
                    battery_capacity=battery_capacity,
                    battery_consume_rate=1.0,
                    penalty_unvisit=1000.0,
                    penalty_delay=100.0
                )

                if initial_solution is None:
                    st.error("❌ Failed to generate initial solution. Try increasing number of vehicles or battery capacity.")
                    st.stop()

                # Check if initial solution has empty routes (no nodes inserted)
                has_empty_routes = all(len(route) == 2 and route == [0, 0] for route in initial_solution.routes)
                if has_empty_routes:
                    st.error("""
                    ❌ Initial solution has empty routes (no nodes were inserted).

                    **Possible causes:**
                    1. **Battery capacity too low** - Increase battery capacity (try 200.0 or higher)
                    2. **Time windows too tight** - Generated instance may have infeasible time windows
                    3. **Vehicle capacity too low** - Increase vehicle capacity
                    4. **Distance between nodes too large** - Try generating a new instance with different seed

                    **Try:** Increase battery capacity to at least 200.0 and try again.
                    """)
                    st.stop()

                # Create ALNS config
                # Note: ALNSConfig uses num_segments * segment_length to calculate total iterations
                num_segments = max(1, max_iterations // segment_length)
                config = ALNSConfig(
                    num_segments=num_segments,
                    segment_length=segment_length,
                    start_temp=start_temp,
                    cooling_rate=cooling_rate,
                    seed=seed  # Use seed from UI for reproducibility
                )

                # Run ALNS
                alns = ALNS(
                    initial_solution=initial_solution._solution,
                    config=config,
                    dist_matrix=st.session_state.instance.distance_matrix,
                    battery_capacity=battery_capacity
                )

                # Run with progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Simple run (in real implementation, we'd add callbacks for live updates)
                alns.run()
                progress_bar.progress(1.0)

                # Store results
                st.session_state.solution = alns.best_solution
                st.session_state.cost_history = alns.cost_history

                st.success("✅ ALNS completed successfully!")

            except ValueError as e:
                st.error(f"❌ Invalid parameters: {e}")
                st.info("💡 Try adjusting num_vehicles or battery_capacity")
            except Exception as e:
                st.error(f"❌ Error during solving: {e}")
                st.exception(e)

st.markdown("---")

# Step 4: Results
if st.session_state.solution is not None:
    st.header("📊 Step 4: Results")

    solution = st.session_state.solution

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Cost",
            f"{solution.objective_value:.2f}",
            help="Total travel distance"
        )

    with col2:
        is_feasible = solution.is_feasible()
        st.metric(
            "Feasibility",
            "✅ Feasible" if is_feasible else "❌ Infeasible"
        )

        # Show warning if solution is infeasible
        if not is_feasible:
            st.warning("""
            Solution is infeasible (violates constraints).

            This could be because:
            - Battery capacity too low
            - Time window violations
            - Vehicle capacity exceeded
            - Pickup-delivery precedence violated

            Try increasing battery capacity or reducing problem size.
            """)

    with col3:
        st.metric(
            "Number of Routes",
            len(solution.routes)
        )

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["🗺️ Routes", "📈 Convergence", "📋 Details"])

    with tab1:
        st.subheader("Route Visualization")

        try:
            # Create route visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Use toolkit visualizer if available, otherwise simple plot
            try:
                visualizer = PDPTWVisualizer(st.session_state.instance)
                visualizer.visualize(solution, ax=ax)
            except:
                # Fallback: simple scatter plot
                coords = st.session_state.instance.order_table[['x', 'y']].values
                ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                ax.set_title(f"Solution (Cost: {solution.objective_value:.2f})")
                ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"❌ Error visualizing routes: {e}")

    with tab2:
        st.subheader("Convergence Plot")

        if st.session_state.cost_history is not None:
            cost_history = st.session_state.cost_history

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cost_history, linewidth=2, color='#1f77b4')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Total Cost')
            ax.set_title('ALNS Convergence')
            ax.grid(True, alpha=0.3)

            # Highlight best
            best_idx = cost_history.index(min(cost_history))
            best_cost = cost_history[best_idx]
            ax.plot(best_idx, best_cost, 'r*', markersize=15, label='Best Solution')
            ax.legend()

            st.pyplot(fig)
            plt.close()

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Cost", f"{cost_history[0]:.2f}")
            with col2:
                st.metric("Final Cost", f"{cost_history[-1]:.2f}")
            with col3:
                improvement = (cost_history[0] - cost_history[-1]) / cost_history[0] * 100
                st.metric("Improvement", f"{improvement:.1f}%")

    with tab3:
        st.subheader("Route Details")

        route_data = []
        for i, route in enumerate(solution.routes):
            route_data.append({
                'Route': i + 1,
                'Nodes': len(route),
                'Sequence': ' → '.join(map(str, route))
            })

        st.dataframe(route_data, use_container_width=True)

    # Save Experiment Button
    st.markdown("---")
    st.subheader("💾 Save Experiment")

    experiment_name = st.text_input(
        "Experiment Name (optional)",
        value=f"ALNS_{num_orders}_orders_{num_vehicles}_vehicles"
    )

    if st.button("💾 Save Experiment to runs/ directory"):
        try:
            import json
            import pandas as pd
            from datetime import datetime
            import os

            # Create timestamp for directory name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_dir = os.path.join("runs", timestamp)
            os.makedirs(experiment_dir, exist_ok=True)

            # Save configuration
            config = {
                "experiment_name": experiment_name,
                "timestamp": timestamp,
                "algorithm": "ALNS",
                "parameters": {
                    "num_orders": num_orders,
                    "num_vehicles": num_vehicles,
                    "seed": seed,
                    "max_iterations": max_iterations,
                    "start_temp": start_temp,
                    "cooling_rate": cooling_rate,
                    "segment_length": segment_length,
                    "battery_capacity": battery_capacity
                },
                "instance_info": {
                    "type": "synthetic_pdptw",
                    "n": st.session_state.instance.n if st.session_state.instance else "unknown",
                    "total_nodes": len(st.session_state.instance.indices) if st.session_state.instance else "unknown"
                }
            }

            config_path = os.path.join(experiment_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Save solution (simplified)
            solution_data = {
                "routes": solution.routes,
                "objective_value": float(solution.objective_value),
                "is_feasible": solution.is_feasible(),
                "num_routes": len(solution.routes)
            }

            solution_path = os.path.join(experiment_dir, "solution.json")
            with open(solution_path, 'w') as f:
                json.dump(solution_data, f, indent=2)

            # Save metrics
            metrics = {
                "total_cost": float(solution.objective_value),
                "feasibility": solution.is_feasible(),
                "route_count": len(solution.routes),
                "total_nodes": sum(len(route) for route in solution.routes),
                "cost_history": cost_history.tolist() if hasattr(cost_history, 'tolist') else cost_history if cost_history is not None else []
            }

            metrics_path = os.path.join(experiment_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            # Save instance info (simplified)
            if st.session_state.instance and hasattr(st.session_state.instance, 'order_table'):
                instance_info = {
                    "n": st.session_state.instance.n,
                    "depot_index": st.session_state.instance.depot_index,
                    "node_count": len(st.session_state.instance.indices)
                }
                instance_path = os.path.join(experiment_dir, "instance_info.json")
                with open(instance_path, 'w') as f:
                    json.dump(instance_info, f, indent=2)

            # Update EXPERIMENTS_LOG.md
            # Calculate improvement if cost_history exists
            improvement_text = ""
            if st.session_state.cost_history is not None and len(st.session_state.cost_history) > 1:
                initial_cost = st.session_state.cost_history[0]
                final_cost = st.session_state.cost_history[-1]
                if initial_cost > 0:
                    improvement_pct = (initial_cost - final_cost) / initial_cost * 100
                    improvement_text = f"- Improvement: {improvement_pct:.1f}%"

            log_entry = f"""
### {timestamp} - {experiment_name}

**Config:**
- Algorithm: ALNS
- Orders: {num_orders}
- Vehicles: {num_vehicles}
- Max Iterations: {max_iterations}
- Start Temperature: {start_temp}
- Seed: {seed}
- Battery Capacity: {battery_capacity}

**Results:**
- Best Cost: {solution.objective_value:.2f}
- Feasible: {'✅ Yes' if solution.is_feasible() else '❌ No'}
- Routes: {len(solution.routes)}
{improvement_text}

**Files:**
- {experiment_dir}/config.json
- {experiment_dir}/solution.json
- {experiment_dir}/metrics.json
- {experiment_dir}/instance_info.json

---

"""
            log_path = os.path.join("runs", "EXPERIMENTS_LOG.md")
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)

            st.success(f"✅ Experiment saved to {experiment_dir}")
            st.info(f"Log entry added to runs/EXPERIMENTS_LOG.md")

        except Exception as e:
            st.error(f"❌ Failed to save experiment: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 VRP-Toolkit Playground | Learn by Playing, Not by Reading Code</p>
    <p style='font-size: 0.8em;'>See <code>playground/VISION.md</code> for design philosophy</p>
</div>
""", unsafe_allow_html=True)
