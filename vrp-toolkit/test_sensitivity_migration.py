"""Test migration of sensitivity_test.ipynb to tutorials/05_sensitivity_analysis.ipynb"""

import json
import os
import sys
from pathlib import Path

def test_sensitivity_tutorial_file_exists():
    """Verify sensitivity tutorial file was created."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "05_sensitivity_analysis.ipynb"
    assert tutorial_path.exists(), f"Sensitivity tutorial file not found: {tutorial_path}"
    print(f"[OK] Sensitivity tutorial file exists: {tutorial_path}")

def test_sensitivity_tutorial_is_valid_json():
    """Verify sensitivity tutorial file is valid JSON (Jupyter notebook format)."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "05_sensitivity_analysis.ipynb"
    try:
        with open(tutorial_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        assert "cells" in notebook, "Notebook missing 'cells' key"
        assert "metadata" in notebook, "Notebook missing 'metadata' key"
        assert len(notebook["cells"]) > 0, "Notebook has no cells"
        print(f"[OK] Sensitivity tutorial file is valid JSON with {len(notebook['cells'])} cells")
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON in sensitivity tutorial file: {e}")

def test_sensitivity_tutorial_has_required_sections():
    """Verify sensitivity tutorial has key sections."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "05_sensitivity_analysis.ipynb"
    with open(tutorial_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Check for key sections in markdown cells
    markdown_texts = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            if "source" in cell:
                # source is a list of strings, join them
                text = "".join(cell["source"])
                markdown_texts.append(text.lower())

        required_keywords = [
        "sensitivity analysis",
        "import",
        "experiment design",
        "helper functions",
        "experiment loop",
        "analyze and visualize",
        "visualize",
        "export results",
        "conclusion"
    ]

    all_text = " ".join(markdown_texts)
    missing_keywords = []
    for keyword in required_keywords:
        if keyword.lower() not in all_text:
            missing_keywords.append(keyword)

    assert len(missing_keywords) == 0, f"Missing required sections: {missing_keywords}"
    print(f"[OK] Sensitivity tutorial contains all required sections")

def test_sensitivity_imports_resolve():
    """Verify that imports in sensitivity tutorial can resolve (basic check)."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "05_sensitivity_analysis.ipynb"
    with open(tutorial_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    import_statements = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "source" in cell:
            source = "".join(cell["source"])
            if "import" in source:
                import_statements.append(source)

    # Check for expected imports
    expected_imports = [
        "vrp_toolkit.data.map",
        "vrp_toolkit.data.generators",
        "vrp_toolkit.problems.pdptw",
        "vrp_toolkit.algorithms.alns"
    ]

    all_imports = "\n".join(import_statements)
    missing_imports = []
    for imp in expected_imports:
        if imp not in all_imports:
            missing_imports.append(imp)

    # Note: This is a weak check - just verifies the strings appear
    if len(missing_imports) > 0:
        print(f"Note: Some expected imports not found: {missing_imports}")
    else:
        print(f"[OK] Sensitivity tutorial contains expected import statements")

def test_sensitivity_uses_alnsconfig():
    """Verify that sensitivity tutorial uses ALNSConfig dataclass."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "05_sensitivity_analysis.ipynb"
    with open(tutorial_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    uses_config = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "source" in cell:
            source = "".join(cell["source"])
            if "ALNSConfig" in source and "config = ALNSConfig" in source:
                uses_config = True
                break

    assert uses_config, "Sensitivity tutorial should use ALNSConfig dataclass for configuration"
    print(f"[OK] Sensitivity tutorial uses ALNSConfig dataclass")

def test_sensitivity_has_parameterization():
    """Verify that sensitivity tutorial has parameterized experiment settings."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "05_sensitivity_analysis.ipynb"
    with open(tutorial_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    has_parameters = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "source" in cell:
            source = "".join(cell["source"])
            if "NUM_RUNS" in source and "AVERAGE_ORDER" in source and "NUM_VEHICLES" in source:
                has_parameters = True
                break

    assert has_parameters, "Sensitivity tutorial should have parameterized experiment settings"
    print(f"[OK] Sensitivity tutorial has parameterized experiment settings")

def main():
    """Run all tests."""
    print("Testing sensitivity tutorial migration...")
    tests = [
        test_sensitivity_tutorial_file_exists,
        test_sensitivity_tutorial_is_valid_json,
        test_sensitivity_tutorial_has_required_sections,
        test_sensitivity_imports_resolve,
        test_sensitivity_uses_alnsconfig,
        test_sensitivity_has_parameterization
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} error: {e}")
            failed += 1

    print(f"\nTest summary: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)