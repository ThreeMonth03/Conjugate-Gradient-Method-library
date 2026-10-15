"""Verify explicit native/BLAS modes through the public Python boundary."""

import math
from concurrent.futures import ThreadPoolExecutor

import cgpy
import numpy as np
import pytest
from cgbench.backends import make_executor
from cgbench.cases import SUITES, make_case
from cgbench.collector import check_outcome


@pytest.fixture(params=["native", "blas"])
def dense_kernel(request):
    if request.param == "blas" and not cgpy.blas_enabled():
        pytest.skip("optional BLAS provider is not enabled")
    return request.param


@pytest.fixture
def dense_system():
    size = 33
    matrix = np.diag(np.geomspace(1.0, 50.0, size))
    matrix += np.diag(np.full(size - 1, -0.125), 1)
    matrix += np.diag(np.full(size - 1, -0.125), -1)
    truth = np.linspace(-1.0, 1.0, size)
    return matrix, matrix @ truth, truth


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "algorithm,preconditioner",
    [("standard", "none"), ("standard", "jacobi"), ("explicit", "none")],
)
def test_modes_preserve_inputs_and_verify_true_residual(
    dense_kernel, dense_system, fused, algorithm, preconditioner
):
    matrix, rhs, truth = dense_system
    initial = np.full(rhs.shape, 0.25)
    original = matrix.copy(), rhs.copy(), initial.copy()
    result = cgpy.solve(
        matrix,
        rhs,
        initial,
        dense_kernel=dense_kernel,
        fused=fused,
        algorithm=algorithm,
        preconditioner=preconditioner,
        rtol=1e-10,
        maxiter=300,
    )
    assert result.converged
    assert math.hypot(*(rhs - matrix @ result.x)) <= 1e-10 * math.hypot(*rhs)
    np.testing.assert_allclose(result.x, truth, rtol=1e-7, atol=1e-8)
    for actual, before in zip((matrix, rhs, initial), original, strict=True):
        np.testing.assert_array_equal(actual, before)


def test_prepared_modes_snapshot_and_support_separate_concurrent_workspaces(
    dense_kernel, dense_system
):
    matrix, rhs, truth = dense_system
    plan = cgpy.prepare(
        matrix, dense_kernel=dense_kernel, preconditioner="jacobi"
    )
    assert plan.dense_kernel == dense_kernel
    matrix.fill(np.nan)

    def solve(_):
        output = np.zeros_like(rhs)
        workspace = plan.create_workspace()
        info = plan.solve_into(rhs, output, workspace=workspace, rtol=1e-10)
        assert info.converged
        np.testing.assert_allclose(output, truth, rtol=1e-7, atol=1e-8)
        assert (
            plan.solve_into(
                rhs, output, workspace=workspace, rtol=1e-10
            ).iterations
            == 0
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(solve, range(4)))


def test_unavailable_blas_and_unsupported_mode_do_not_fall_back(dense_system):
    matrix, rhs, _ = dense_system
    output = np.zeros_like(rhs)
    workspace = cgpy.Workspace(rhs.size)
    for invalid in ("auto", "unknown"):
        with pytest.raises(ValueError, match="dense_kernel"):
            cgpy.prepare(matrix, dense_kernel=invalid)
        with pytest.raises(ValueError, match="dense_kernel"):
            cgpy.solve_into(
                matrix, rhs, output, workspace=workspace, dense_kernel=invalid
            )
    if not cgpy.blas_enabled():
        with pytest.raises(ValueError, match="BLAS"):
            cgpy.prepare(matrix, dense_kernel="blas")
        with pytest.raises(ValueError, match="BLAS"):
            cgpy.solve_into(
                matrix, rhs, output, workspace=workspace, dense_kernel="blas"
            )
    np.testing.assert_array_equal(output, 0.0)


def test_blas_rejects_multiple_threads_and_csr():
    if not cgpy.blas_enabled():
        pytest.skip("optional BLAS provider is not enabled")
    with pytest.raises(ValueError, match="1"):
        cgpy.prepare(np.eye(2), dense_kernel="blas", num_threads=2)
    matrix = cgpy.CSRMatrix([0, 1, 2], [0, 1], [1.0, 2.0], (2, 2))
    for prepare in (
        lambda: cgpy.prepare(matrix, dense_kernel="blas"),
        lambda: cgpy.solve(matrix, [1.0, 1.0], dense_kernel="blas"),
    ):
        with pytest.raises(ValueError, match="dense|CSR"):
            prepare()


def test_provider_metadata_describes_actual_build():
    metadata = cgpy.build_info()["blas"]
    assert metadata["enabled"] == cgpy.blas_enabled()
    if cgpy.blas_enabled():
        assert metadata["provider"] == "OpenBLAS"
        assert metadata["integer_bits"] == 32
        assert metadata["thread_mode"] == "sequential"
        assert metadata["symbol_prefix"] == "cgpy_"
        assert "OpenBLAS" in metadata["configuration"]
    else:
        assert metadata["provider"] is None
        assert metadata["configuration"] == "disabled"


@pytest.mark.parametrize("scope", ["api", "prepared"])
@pytest.mark.parametrize("candidate", ["native", "native-cg", "native-pcg"])
def test_native_profile_candidates_keep_native_kernel_despite_conflicting_options(
    scope, candidate
):
    matrix, rhs, initial, _ = make_case(SUITES["smoke"][0], 42)
    options = {
        "algorithm": "standard",
        "fused": True,
        "rtol": 1e-8,
        "atol": 1e-12,
        "maxiter": 2000,
        "residual_refresh": 50,
        "dense_kernel": "blas",
    }
    arrays = matrix, rhs, initial
    executor = make_executor(candidate, arrays, options, scope, 1)
    assert executor.benchmark_metadata()["dense_kernel"] == "native"
    assert check_outcome(executor(), arrays, options)["eligible"]


@pytest.mark.parametrize("scope", ["api", "prepared"])
def test_profile_blas_candidate_uses_public_api_and_shared_accuracy_gate(
    scope,
):
    matrix, rhs, initial, _ = make_case(SUITES["smoke"][0], 42)
    options = {
        "algorithm": "standard",
        "fused": True,
        "rtol": 1e-8,
        "atol": 1e-12,
        "maxiter": 2000,
        "residual_refresh": 50,
        "dense_kernel": "native",
    }
    arrays = matrix, rhs, initial
    if not cgpy.blas_enabled():
        with pytest.raises(ImportError, match="BLAS-enabled"):
            make_executor("native-blas", arrays, options, scope, 1)
        return
    executor = make_executor("native-blas", arrays, options, scope, 1)
    for _ in range(2):
        assert check_outcome(executor(), arrays, options)["eligible"]
    assert executor.benchmark_metadata()["dense_kernel"] == "blas"
