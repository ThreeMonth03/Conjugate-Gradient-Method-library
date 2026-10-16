"""CUDA Python orchestration contracts independent of a CUDA runtime."""

from types import SimpleNamespace

import pytest
from cgpy import cuda
from cgpy._runtime import cuda as cuda_runtime
from cgpy.solver.cuda import api


@pytest.fixture
def runtime(monkeypatch):
    created = []
    calls = []
    info = SimpleNamespace(converged=True)

    def workspace(size, device, stream, dense_kernel, execution_mode):
        value = SimpleNamespace(
            size=size,
            device=device,
            stream=stream,
            dense_kernel=dense_kernel,
            execution_mode=execution_mode,
        )
        created.append(value)
        return value

    def solve_into(
        A,
        b,
        x,
        *,
        workspace,
        rtol,
        atol,
        maxiter,
        algorithm,
        fused,
        residual_refresh,
        preconditioner,
    ):
        calls.append((A, b, x, workspace, preconditioner))
        return info

    extension = SimpleNamespace(Workspace=workspace, solve_into=solve_into)
    monkeypatch.setattr(cuda_runtime, "load", lambda: extension)
    return SimpleNamespace(
        extension=extension, created=created, calls=calls, info=info
    )


@pytest.mark.parametrize(
    "selector,value",
    [
        ("device", 2),
        ("stream", 17),
        ("dense_kernel", "cublas"),
        ("execution_mode", "device"),
    ],
)
def test_strict_api_rejects_duplicate_workspace_settings_before_runtime_load(
    monkeypatch, selector, value
):
    def forbidden():
        raise AssertionError("duplicate selector reached the CUDA runtime")

    monkeypatch.setattr(cuda_runtime, "load", forbidden)
    with pytest.raises(TypeError, match=selector):
        cuda.solve_into(
            object(),
            object(),
            object(),
            workspace=object(),
            **{selector: value},
        )


@pytest.mark.parametrize("kernel", ["native", "cublas"])
@pytest.mark.parametrize("mode", ["host", "device"])
def test_strict_api_uses_workspace_without_a_repeated_selector(
    runtime, kernel, mode
):
    workspace = cuda.Workspace(
        2, device=3, stream=17, dense_kernel=kernel, execution_mode=mode
    )
    matrix, rhs, output = object(), object(), object()
    assert (
        cuda.solve_into(
            matrix, rhs, output, workspace=workspace, preconditioner="jacobi"
        )
        is runtime.info
    )
    assert runtime.calls == [
        (matrix, rhs, output, workspace._native, "jacobi")
    ]
    assert (
        workspace.device,
        workspace.stream,
        workspace.dense_kernel,
        workspace.execution_mode,
    ) == (3, 17, kernel, mode)
    assert len(runtime.created) == 1


@pytest.mark.parametrize("mode", ["host", "device"])
def test_convenience_selects_backend_when_creating_its_workspace(
    monkeypatch, runtime, mode
):
    output = object()
    allocations = []

    def initial_output(b, x0, device, stream):
        allocations.append((b, x0, device, stream))
        return output, 2

    monkeypatch.setattr(api, "_initial_output", initial_output)
    matrix, rhs, initial = object(), object(), object()
    answer = cuda.solve(
        matrix,
        rhs,
        initial,
        device=3,
        stream=17,
        dense_kernel="cublas",
        execution_mode=mode,
    )
    assert answer.x is output and answer.info is runtime.info
    assert allocations == [(rhs, initial, 3, 17)]
    assert len(runtime.created) == 1
    workspace = runtime.created[0]
    assert workspace.dense_kernel == "cublas"
    assert workspace.execution_mode == mode
    assert runtime.calls == [(matrix, rhs, output, workspace, "none")]


def test_session_lock_covers_validation_output_allocation_and_execution(
    monkeypatch, runtime
):
    session = cuda.Session(2, dense_kernel="cublas", execution_mode="device")
    matrix, rhs = object(), object()
    allocations = []
    reentry_checks = []

    def reject_reentry(stage):
        with pytest.raises(ValueError, match="session is already in use"):
            session.solve(matrix, rhs)
        reentry_checks.append(stage)

    class ReentrantCount:
        def __index__(self):
            reject_reentry("validation")
            return 2

    def initial_output(b, x0, device, stream):
        reject_reentry("allocation")
        output = object()
        allocations.append(output)
        return output, 2

    native_solve = runtime.extension.solve_into

    def solve_into(*args, **kwargs):
        reject_reentry("execution")
        return native_solve(*args, **kwargs)

    monkeypatch.setattr(api, "_initial_output", initial_output)
    runtime.extension.solve_into = solve_into
    answer = session.solve(matrix, rhs, maxiter=ReentrantCount())
    assert answer.info is runtime.info and allocations == [answer.x]
    assert reentry_checks == ["validation", "allocation", "execution"]
    assert len(runtime.created) == 1


@pytest.mark.parametrize("stage", ["validation", "allocation", "execution"])
def test_session_recovers_after_failure_and_preserves_independent_outputs(
    monkeypatch, runtime, stage
):
    session = cuda.Session(2)
    matrix, rhs = object(), object()
    fail = True

    class Count:
        def __index__(self):
            if fail and stage == "validation":
                raise RuntimeError("validation sentinel")
            return 2

    def initial_output(b, x0, device, stream):
        if fail and stage == "allocation":
            raise RuntimeError("allocation sentinel")
        return object(), 2

    native_solve = runtime.extension.solve_into

    def solve_into(*args, **kwargs):
        if fail and stage == "execution":
            raise RuntimeError("execution sentinel")
        return native_solve(*args, **kwargs)

    monkeypatch.setattr(api, "_initial_output", initial_output)
    runtime.extension.solve_into = solve_into
    with pytest.raises(RuntimeError, match=f"{stage} sentinel"):
        session.solve(matrix, rhs, maxiter=Count())
    fail = False
    first = session.solve(matrix, rhs)
    second = session.solve(matrix, rhs)
    assert first.info is second.info is runtime.info
    assert first.x is not second.x
    assert len(runtime.created) == 1
