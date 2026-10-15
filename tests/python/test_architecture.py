"""Keep source dependencies aligned with the documented feature boundaries."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "cpp/cg"
INCLUDES = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
DEPENDENCIES = {
    "runtime": {"runtime"},
    "storage": {"runtime", "storage"},
    "matrix": {"runtime", "storage", "matrix"},
    "backend": {"runtime", "storage", "matrix", "backend"},
    "preconditioner": {
        "runtime",
        "storage",
        "matrix",
        "backend",
        "preconditioner",
    },
    "solver": {
        "runtime",
        "storage",
        "matrix",
        "backend",
        "preconditioner",
        "solver",
    },
}


def test_native_features_follow_dependency_direction():
    violations = []
    for path in CORE.rglob("*"):
        if path.suffix not in {".cpp", ".hpp", ".cu", ".cuh"}:
            continue
        parts = path.relative_to(CORE).parts
        if "pymod" in parts or parts[0] == "python" or len(parts) == 1:
            continue
        for include in INCLUDES.findall(path.read_text()):
            if include == "Python.h" or include.startswith("pybind11/"):
                violations.append((str(path.relative_to(ROOT)), include))
            elif include.startswith("cg/"):
                dependency = include.split("/")[1]
                if dependency not in DEPENDENCIES[parts[0]]:
                    violations.append((str(path.relative_to(ROOT)), include))
    assert not violations, violations


def test_product_python_package_does_not_depend_on_benchmark_tools():
    violations = []
    for path in (ROOT / "python/cgpy").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == "cgbench" or module.startswith("cgbench.")
                for module in modules
            ):
                violations.append((str(path.relative_to(ROOT)), node.lineno))
    assert not violations, violations
