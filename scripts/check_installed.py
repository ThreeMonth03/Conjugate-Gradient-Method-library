"""Verify an installed wheel without relying on the repository or provider prefix."""

import argparse
import importlib.util
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blas", choices=("OFF", "ON"), required=True)
    options = parser.parse_args()
    expected_blas = options.blas == "ON"
    # Load NumPy first to exercise coexistence with its independent BLAS.
    # Dynamic import keeps the intentional order independent of import sorting.
    np = importlib.import_module("numpy")
    import _cgcore
    import _cgpy
    import cgpy
    import utils

    assert utils is not None
    assert importlib.util.find_spec("autograd") is None
    assert Path(_cgcore.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert cgpy.blas_enabled() is expected_blas
    info = cgpy.build_info()["blas"]
    assert info["enabled"] is expected_blas
    assert _cgpy.Matrix.Naive_Matrix(2, 2).nrow == 2
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    expected = np.array([1 / 11, 7 / 11])
    modes = ["native", "blas"] if expected_blas else ["native"]
    for mode in modes:
        for algorithm in ("standard", "explicit"):
            result = cgpy.solve(matrix, rhs, dense_kernel=mode, algorithm=algorithm)
            assert result.converged
            np.testing.assert_allclose(result.x, expected, rtol=1e-8, atol=1e-12)
        plan = cgpy.prepare(matrix, dense_kernel=mode, preconditioner="jacobi")
        output = np.zeros(2)
        solved = plan.solve_into(rhs, output, workspace=plan.create_workspace())
        assert solved.converged
        np.testing.assert_allclose(output, expected)
    if expected_blas:
        assert info["integer_bits"] == 32
        assert info["thread_mode"] == "sequential"
        assert info["symbol_prefix"] == "cgpy_"
        assert "0.3.30" in info["configuration"]

        def solve_one(_):
            answer = cgpy.solve(matrix, rhs, dense_kernel="blas")
            assert answer.converged
            np.testing.assert_allclose(answer.x, expected)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(solve_one, range(32)))
        license_file = Path(cgpy.__file__).parent / "_licenses/OpenBLAS.txt"
        assert license_file.is_file()
    else:
        try:
            cgpy.prepare(matrix, dense_kernel="blas")
        except ValueError:
            pass
        else:
            raise AssertionError("disabled wheel silently accepted BLAS")
    csr = cgpy.CSRMatrix([0, 2, 4], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0], (2, 2))
    result = cgpy.solve(csr, rhs, preconditioner="jacobi", dense_kernel="native")
    assert result.converged
    np.testing.assert_allclose(result.x, expected)
    if sys.platform == "linux":
        for module in (_cgcore, _cgpy):
            dynamic = subprocess.run(
                ["readelf", "-d", module.__file__], capture_output=True, text=True, check=True
            ).stdout
            assert "openblas" not in dynamic.lower()
            assert "_deps" not in dynamic and "RUNPATH" not in dynamic and "RPATH" not in dynamic
            exports = subprocess.run(
                ["nm", "-D", "--defined-only", module.__file__],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            assert "cblas_" not in exports and "cgpy_" not in exports and "gotoblas" not in exports
    print(f"Installed {cgpy.__file__}; BLAS={expected_blas}; native/dense/CSR checks passed.")


if __name__ == "__main__":
    main()
