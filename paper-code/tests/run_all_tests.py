"""
Master test script to run all paper-code tests
Runs tests in sequence to verify the entire pipeline
"""
import sys
import os
import subprocess
import time

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = SCRIPT_DIR
PAPER_CODE_DIR = os.path.dirname(SCRIPT_DIR)

def run_test(test_file, test_name):
    """Run a single test script"""
    print("\n" + "=" * 80)
    print(f" {test_name}")
    print("=" * 80)
    print()

    start_time = time.time()
    test_path = os.path.join(TESTS_DIR, test_file)

    try:
        result = subprocess.run(
            [sys.executable, test_path],
            cwd=TESTS_DIR,
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n[PASS] {test_name} (took {elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[FAIL] {test_name} (took {elapsed:.1f}s)")
        return False


def main():
    """Run all tests in sequence"""
    print("\n" + "=" * 80)
    print(" " * 80)
    print(" PAPER-CODE FULL TEST SUITE ".center(80))
    print(" " * 80)
    print("=" * 80)

    tests = [
        ("test_01_data_layer.py", "TEST 1: Data Layer (RealMap + DemandGenerator)"),
        ("test_02_order_generation.py", "TEST 2: Order Generation"),
        ("test_03_instance_solution.py", "TEST 3: Instance & Solution"),
        ("test_04_solver.py", "TEST 4: Solver (Initial + ALNS)")
    ]

    results = []
    start_time = time.time()

    for test_file, test_name in tests:
        success = run_test(test_file, test_name)
        results.append((test_name, success))

    total_time = time.time() - start_time

    # Print summary
    print("\n\n" + "=" * 80)
    print(" " * 80)
    print(" TEST SUMMARY ".center(80))
    print(" " * 80)
    print("=" * 80)
    print()

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:8} - {test_name}")

    print()
    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 80)

    if passed == total:
        print("\n*** ALL TESTS PASSED! Paper-code is working correctly. ***")
        return True
    else:
        print(f"\n*** WARNING: {total - passed} test(s) failed. Please review the output above. ***")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
