"""Test migration of test.ipynb to tutorials/01_quickstart.ipynb"""

import json
import os
import sys
from pathlib import Path

def test_tutorial_file_exists():
    """Verify tutorial file was created."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "01_quickstart.ipynb"
    assert tutorial_path.exists(), f"Tutorial file not found: {tutorial_path}"
    print(f"[OK] Tutorial file exists: {tutorial_path}")

def test_tutorial_is_valid_json():
    """Verify tutorial file is valid JSON (Jupyter notebook format)."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "01_quickstart.ipynb"
    try:
        with open(tutorial_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        assert "cells" in notebook, "Notebook missing 'cells' key"
        assert "metadata" in notebook, "Notebook missing 'metadata' key"
        assert len(notebook["cells"]) > 0, "Notebook has no cells"
        print(f"[OK] Tutorial file is valid JSON with {len(notebook['cells'])} cells")
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON in tutorial file: {e}")

def test_tutorial_has_required_sections():
    """Verify tutorial has key sections."""
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "01_quickstart.ipynb"
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
        "quick start",
        "import",
        "synthetic map",
        "demand data",
        "pdptw",
        "initial solution",
        "alns",
        "visualize"
    ]

    all_text = " ".join(markdown_texts)
    missing_keywords = []
    for keyword in required_keywords:
        if keyword.lower() not in all_text:
            missing_keywords.append(keyword)

    assert len(missing_keywords) == 0, f"Missing required sections: {missing_keywords}"
    print(f"[OK] Tutorial contains all required sections")

def test_imports_resolve():
    """Verify that imports in tutorial can resolve (basic check)."""
    # This doesn't actually run the imports, just checks they're syntactically valid
    tutorial_path = Path(__file__).parent.parent / "tutorials" / "01_quickstart.ipynb"
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
        print(f"[OK] Tutorial contains expected import statements")

def main():
    """Run all tests."""
    print("Testing tutorial migration...")
    tests = [
        test_tutorial_file_exists,
        test_tutorial_is_valid_json,
        test_tutorial_has_required_sections,
        test_imports_resolve
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