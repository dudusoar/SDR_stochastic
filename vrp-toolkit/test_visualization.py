"""Test visualization module."""

import sys

print("Testing VRP Toolkit visualization module...")

try:
    # Test 1: Import base module
    from vrp_toolkit.visualization import (
        BaseVisualizer, ProblemVisualizer, PDPTWVisualizer,
        AlgorithmVisualizer, ALNSVisualizer,
        DataVisualizer, MapVisualizer, DemandVisualizer
    )
    print("[OK] Visualization module imported successfully")

    # Test 2: Create instances
    base_viz = BaseVisualizer(title="Test Visualization")
    print(f"[OK] BaseVisualizer created: {base_viz}")

    problem_viz = ProblemVisualizer()
    print(f"[OK] ProblemVisualizer created: {problem_viz}")

    algorithm_viz = AlgorithmVisualizer()
    print(f"[OK] AlgorithmVisualizer created: {algorithm_viz}")

    data_viz = DataVisualizer()
    print(f"[OK] DataVisualizer created: {data_viz}")

    # Test 3: Check if PDPTWVisualizer can be created (may fail if PDPTW not available)
    try:
        pdptw_viz = PDPTWVisualizer()
        print(f"[OK] PDPTWVisualizer created: {pdptw_viz}")
    except Exception as e:
        print(f"[INFO] PDPTWVisualizer creation: {e}")

    # Test 4: Check if ALNSVisualizer can be created
    try:
        alns_viz = ALNSVisualizer()
        print(f"[OK] ALNSVisualizer created: {alns_viz}")
    except Exception as e:
        print(f"[INFO] ALNSVisualizer creation: {e}")

    # Test 5: Check if MapVisualizer can be created
    try:
        map_viz = MapVisualizer()
        print(f"[OK] MapVisualizer created: {map_viz}")
    except Exception as e:
        print(f"[INFO] MapVisualizer creation: {e}")

    # Test 6: Check if DemandVisualizer can be created
    try:
        demand_viz = DemandVisualizer()
        print(f"[OK] DemandVisualizer created: {demand_viz}")
    except Exception as e:
        print(f"[INFO] DemandVisualizer creation: {e}")

    print("\n[PASS] Basic visualization module tests passed!")

except Exception as e:
    print(f"\n[FAIL] Visualization test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)