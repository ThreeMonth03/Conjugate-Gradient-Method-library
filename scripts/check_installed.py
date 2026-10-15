"""Verify an installed wheel without relying on the repository or provider prefix."""

import argparse
import concurrent.futures
import importlib.util
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blas", choices=("OFF", "ON"), required=True)
    parser.add_argument("--cuda", choices=("OFF", "ON"), default="OFF")
    options = parser.parse_args()
    expected_blas = options.blas == "ON"
    expected_cuda = options.cuda == "ON"
    # Load NumPy first to exercise coexistence with its independent BLAS.
    # Dynamic import keeps the intentional order independent of import sorting.
    np = importlib.import_module("numpy")
    import cgpy

    core = importlib.import_module("cgpy._core")

    assert importlib.util.find_spec("_cgpy") is None
    assert importlib.util.find_spec("utils") is None
    assert sys.version_info[:2] == (3, 12)
    assert (
        pathlib.Path(core.__file__)
        .resolve()
        .is_relative_to(pathlib.Path(sys.prefix).resolve())
    )
    assert cgpy.blas_enabled() is expected_blas
    info = cgpy.build_info()["blas"]
    assert info["enabled"] is expected_blas
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    expected = np.array([1 / 11, 7 / 11])
    modes = ["native", "blas"] if expected_blas else ["native"]
    for mode in modes:
        for algorithm in ("standard", "explicit"):
            result = cgpy.solve(
                matrix, rhs, dense_kernel=mode, algorithm=algorithm
            )
            assert result.converged
            np.testing.assert_allclose(
                result.x, expected, rtol=1e-8, atol=1e-12
            )
        plan = cgpy.prepare(matrix, dense_kernel=mode, preconditioner="jacobi")
        assert type(plan) is cgpy.PreparedSolver
        output = np.zeros(2)
        solved = plan.solve_into(
            rhs, output, workspace=plan.create_workspace()
        )
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(solve_one, range(32)))
        license_file = (
            pathlib.Path(cgpy.__file__).parent / "_licenses/OpenBLAS.txt"
        )
        assert license_file.is_file()
    else:
        try:
            cgpy.prepare(matrix, dense_kernel="blas")
        except ValueError:
            pass
        else:
            raise AssertionError("disabled wheel silently accepted BLAS")
    csr = cgpy.CsrMatrix([0, 2, 4], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0], (2, 2))
    result = cgpy.solve(
        csr, rhs, preconditioner="jacobi", dense_kernel="native"
    )
    assert result.converged
    np.testing.assert_allclose(result.x, expected)
    plan = cgpy.prepare(csr, preconditioner="jacobi")
    assert type(plan) is cgpy.PreparedSolver
    output = np.zeros(2)
    assert plan.solve_into(
        rhs, output, workspace=plan.create_workspace(), atol=0
    ).converged
    np.testing.assert_allclose(output, expected)
    assert not hasattr(cgpy, "CSRMatrix")
    assert not hasattr(cgpy, "PreparedCsrSolver")
    from cgpy import cuda

    assert cuda.enabled() is expected_cuda
    modules = [core]
    if expected_cuda:
        assert cuda.build_info()
        gpu_module = importlib.import_module("cgpy._cuda")
        assert (
            pathlib.Path(gpu_module.__file__)
            .resolve()
            .is_relative_to(pathlib.Path(sys.prefix).resolve())
        )
        modules.append(gpu_module)
        if cuda.is_available():
            gpu_matrix = cuda.DenseMatrix.from_host(matrix)
            gpu_rhs = cuda.Vector.from_host(rhs)
            for mode in ("native", "cublas"):
                for algorithm in ("standard", "explicit"):
                    for fused in (False, True):
                        answer = cuda.solve(
                            gpu_matrix,
                            gpu_rhs,
                            dense_kernel=mode,
                            algorithm=algorithm,
                            fused=fused,
                        )
                        assert answer.converged
                        np.testing.assert_allclose(
                            answer.x.to_host(), expected
                        )
                plan = cuda.prepare(
                    gpu_matrix, dense_kernel=mode, preconditioner="jacobi"
                )
                output = plan.create_vector()
                solved = plan.solve_into(
                    gpu_rhs, output, workspace=plan.create_workspace()
                )
                assert solved.converged
                np.testing.assert_allclose(output.to_host(), expected)
            gpu_csr = cuda.CsrMatrix.from_host(csr)
            answer = cuda.solve(gpu_csr, gpu_rhs, preconditioner="jacobi")
            assert answer.converged
            np.testing.assert_allclose(answer.x.to_host(), expected)
            print(
                "Installed CUDA wheel: native/cuBLAS/CSR/Jacobi device checks passed."
            )
        else:
            print(
                "Installed CUDA wheel imported; device checks skipped: no usable GPU."
            )
    else:
        assert importlib.util.find_spec("cgpy._cuda") is None
        assert not cuda.is_available()
        assert importlib.util.find_spec("nvidia") is None
    if sys.platform == "linux":
        for module in modules:
            dynamic = subprocess.run(
                ["readelf", "-d", module.__file__],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            assert "openblas" not in dynamic.lower()
            assert (
                "_deps" not in dynamic
                and "RUNPATH" not in dynamic
                and "RPATH" not in dynamic
            )
            assert "/cuda" not in dynamic and "/torch" not in dynamic
            if module.__name__ != "cgpy._cuda":
                assert (
                    "libcudart" not in dynamic and "libcublas" not in dynamic
                )
            else:
                assert (
                    "libcudart.so.12" in dynamic
                    and "libcublas.so.12" in dynamic
                )
            exports = subprocess.run(
                ["nm", "-D", "--defined-only", module.__file__],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            assert (
                "cblas_" not in exports
                and "cgpy_" not in exports
                and "gotoblas" not in exports
            )
    print(
        f"Installed {cgpy.__file__}; BLAS={expected_blas}; CUDA={expected_cuda}; native/dense/CSR checks passed."
    )


if __name__ == "__main__":
    main()
