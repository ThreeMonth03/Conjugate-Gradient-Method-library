#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cg/cg.hpp>

#include <cg/python/array_conversion.hpp>
#include <cg/python/module.hpp>
#include <cg/solver/pymod/options.hpp>

#include <algorithm>
#include <atomic>
#include <memory>
#include <utility>

namespace py = pybind11;

namespace cg
{
namespace python
{
namespace
{
struct PythonWorkspace
{
    explicit PythonWorkspace(std::size_t size) : value(size) {}
    explicit PythonWorkspace(cg::Workspace workspace) : value(std::move(workspace)) {}
    cg::Workspace value;
    std::atomic<bool> active{false};
};

struct WorkspaceUseGuard
{
    explicit WorkspaceUseGuard(PythonWorkspace &workspace) : workspace(workspace)
    {
        if (workspace.active.exchange(true))
        {
            throw std::invalid_argument("Workspace is already in use by another solve");
        }
    }
    WorkspaceUseGuard(const WorkspaceUseGuard &) = delete;
    WorkspaceUseGuard &operator=(const WorkspaceUseGuard &) = delete;
    ~WorkspaceUseGuard() { workspace.active.store(false); }
    PythonWorkspace &workspace;
};

cg::DenseKernel parse_dense_kernel(const std::string &name)
{
    if (name == "native")
    {
        return cg::DenseKernel::native;
    }
    if (name == "blas")
    {
        return cg::DenseKernel::blas;
    }
    throw std::invalid_argument("dense_kernel must be 'native' or 'blas'");
}
const char *dense_kernel_name(cg::DenseKernel kernel)
{
    return kernel == cg::DenseKernel::blas ? "blas" : "native";
}
void validate_operator_kernel(cg::DenseMatrixView, cg::DenseKernel) {}
void validate_operator_kernel(const cg::CsrMatrix &, cg::DenseKernel kernel)
{
    if (kernel == cg::DenseKernel::blas)
    {
        throw std::invalid_argument("dense_kernel='blas' requires a dense matrix; CSR uses native kernels");
    }
}
template <class Operator>
cg::SolveInfo run(const Operator &A, const py::array &b, py::array &x, PythonWorkspace &workspace, int threads,
                  bool fused, const cg::CgOptions &settings, cg::PreconditionerKind preconditioner,
                  cg::DenseKernel kernel = cg::DenseKernel::native)
{
    validate_operator_kernel(A, kernel);
    validate_vector_array(b, false);
    validate_vector_array(x, true);
    const cg::ConstVectorView bv{static_cast<const double *>(b.data()), static_cast<std::size_t>(b.size())};
    const cg::VectorView xv{static_cast<double *>(x.mutable_data()), static_cast<std::size_t>(x.size())};
    const cg::CpuBackend backend(threads, fused, kernel);
    WorkspaceUseGuard use(workspace);
    py::gil_scoped_release release;
    return cg::solve_into(A, bv, xv, workspace.value, backend, settings, preconditioner);
}
cg::SolveInfo run_prepared(const cg::PreparedSolver &plan, const py::array &b, py::array x, PythonWorkspace &workspace,
                           double rtol, double atol, std::size_t maxiter, const std::string &algorithm, bool fused,
                           std::size_t residual_refresh)
{
    validate_vector_array(b, false);
    validate_vector_array(x, true);
    const cg::ConstVectorView bv{static_cast<const double *>(b.data()), static_cast<std::size_t>(b.size())};
    const cg::VectorView xv{static_cast<double *>(x.mutable_data()), static_cast<std::size_t>(x.size())};
    const auto options = parse_solver_options(rtol, atol, maxiter, algorithm, residual_refresh);
    WorkspaceUseGuard use(workspace);
    py::gil_scoped_release release;
    return plan.solve_into(bv, xv, workspace.value, options, fused);
}
cg::DenseMatrix snapshot_dense_matrix(const py::array &A)
{
    const auto view = as_dense_matrix_view(A);
    // The view contract guarantees contiguous storage; the core validates the
    // prepared operator's dimensions independently of the Python boundary.
    const auto count = static_cast<std::size_t>(A.size());
    std::vector<double> values(count);
    if (count != 0)
    {
        std::copy_n(view.data(), count, values.data());
    }
    return {view.rows(), view.cols(), std::move(values)};
}

auto bind_prepared(py::module_ &m)
{
    using Plan = cg::PreparedSolver;
    return py::class_<Plan>(m, "PreparedSolver", R"doc(
An owning matrix snapshot with fixed thread, preconditioner and kernel settings.

Notes
-----
Construct through ``cgpy.prepare`` to normalize a dense input or wrap a public
CsrMatrix. The prepared object owns its matrix and optional Jacobi inverse.
Each overlapping solve needs a separate workspace and output. All calls are
synchronous, and native computation releases the GIL.
)doc")
        .def(
            "create_workspace",
            [](const Plan &plan) { return std::make_unique<PythonWorkspace>(plan.create_workspace()); }, R"doc(
Allocate reusable solver vectors matching this system.

Returns
-------
Workspace
    New independent workspace for one active solve at a time.
)doc")
        .def_property_readonly("size", &Plan::size, "int: Matrix dimension and workspace size.")
        .def_property_readonly(
            "shape", [](const Plan &plan) { return py::make_tuple(plan.size(), plan.size()); },
            "tuple of int: Square operator dimensions.")
        .def_property_readonly("preconditioner_nbytes", &Plan::preconditioner_nbytes,
                               "int: Stored Jacobi inverse bytes; zero without Jacobi.")
        .def_property_readonly("num_threads", &Plan::num_threads, "int: CPU thread count fixed at preparation time.")
        .def_property_readonly(
            "dense_kernel", [](const Plan &plan) { return dense_kernel_name(plan.dense_kernel()); },
            "str: Explicit 'native' or 'blas' kernel selected at preparation.")
        .def_property_readonly("preconditioner_setup_ns", &Plan::preconditioner_setup_ns,
                               "int: Jacobi construction time in nanoseconds; zero without Jacobi.")
        .def_property_readonly(
            "preconditioner", [](const Plan &plan) { return cg::preconditioner_name(plan.preconditioner()); },
            "str: 'none' or 'jacobi', fixed at preparation time.")
        .def("solve_into", &run_prepared, py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(),
             py::arg("workspace"), py::arg("rtol") = 1e-8, py::arg("atol") = 0.0, py::arg("maxiter") = 1000,
             py::arg("algorithm") = "standard", py::arg("fused") = true, py::arg("residual_refresh") = 50,
             R"doc(
Solve in place using the prepared operator and caller-owned vectors.

Parameters
----------
b : numpy.ndarray
    Aligned C-contiguous float64 right-hand side of shape ``(size,)``.
x : numpy.ndarray
    Writable aligned C-contiguous float64 initial guess and output, disjoint
    from ``b``. Reset it before repeating a solve with the same initial guess.
workspace : Workspace
    Matching workspace, unused by overlapping calls.
rtol, atol : float, default=1e-8, 0.0
    Finite nonnegative tolerances. True-residual threshold is
    ``max(atol, rtol * ||b||2)``.
maxiter : int, default=1000
    Maximum updates; zero checks only the initial guess.
algorithm : {'standard', 'explicit'}, default='standard'
    Residual recurrence or explicit-residual formulation. Jacobi requires standard.
fused : bool, default=True
    Fuse supported CPU operations independently of the fixed dense kernel.
residual_refresh : int, default=50
    Standard-CG replacement/restart interval; zero disables periodic refresh.

Returns
-------
SolveInfo
    Convergence/failure status, independently checked residual and counts.

Raises
------
ValueError
    Invalid arrays, options, workspace use, aliasing or algorithm/preconditioner.
TypeError
    Negative, out-of-range or nonintegral native iteration counts.

Notes
-----
The matrix, thread count, dense kernel and preconditioner are fixed by
preparation. Borrowed arrays must remain alive and free from conflicting
mutation until the synchronous call returns. The GIL is released for the solve.
)doc");
}
} // namespace

void register_cpu_solver(py::module_ &m)
{
    py::class_<cg::SolveInfo>(m, "SolveInfo", R"doc(
Read-only numerical status and operation counts for a completed native call.

Attributes
----------
status : str
    ``converged``, ``max_iterations``, ``non_positive_curvature`` or
    ``non_finite``. Negative curvature has its own status; zero/subnormal
    required products are reported as outside the usable double range.
converged : bool
    Whether the independently evaluated true residual met the tolerance.
iterations : int
    Completed solution updates, excluding the initial residual check.
matvecs : int
    Matrix-vector products, including initial, refresh and final checks.
residual_norm : float
    Independently evaluated final Euclidean true-residual norm.
restarts : int
    Direction restarts after a true-residual replacement.

Notes
-----
Numerical breakdown and iteration limits return this record instead of raising
an exception. ``non_finite`` includes unusable inner-product underflow/overflow;
the reported true residual can still be finite. Invalid inputs raise exceptions.
)doc")
        .def_property_readonly(
            "status", [](const cg::SolveInfo &info) { return cg::status_name(info.status); },
            "str: Convergence, iteration-limit or numerical-failure status.")
        .def_property_readonly(
            "converged", [](const cg::SolveInfo &info) { return info.status == cg::Status::converged; },
            "bool: Whether the true-residual stopping rule was satisfied.")
        .def_readonly("iterations", &cg::SolveInfo::iterations, "int: Number of completed solution updates.")
        .def_readonly("matvecs", &cg::SolveInfo::matvecs,
                      "int: Matrix-vector products, including true-residual checks.")
        .def_readonly("residual_norm", &cg::SolveInfo::residual_norm,
                      "float: Independently evaluated final Euclidean residual norm.")
        .def_readonly("restarts", &cg::SolveInfo::restarts, "int: Direction restarts following residual replacement.");
    py::class_<PythonWorkspace>(m, "Workspace", R"doc(
Own reusable residual, direction, product, scratch and preconditioned vectors.

Parameters
----------
size : int
    Nonnegative system dimension. A solve requires a matching nonempty matrix.

Raises
------
TypeError
    Size is negative, out of native integer range or nonintegral.

Notes
-----
One workspace can serve only one active call. Overlapping use raises ValueError;
concurrent solves need independent workspaces and outputs. The solver reuses
these five vectors instead of allocating them during each prepared solve.

Examples
--------
>>> import cgpy
>>> workspace = cgpy.Workspace(2)
>>> workspace.size
2
)doc")
        .def(py::init<std::size_t>(), py::arg("size"))
        .def_property_readonly(
            "size", [](const PythonWorkspace &w) { return w.value.size(); },
            "int: Current vector length, required to match the matrix dimension.")
        .def_property_readonly(
            "capacity", [](const PythonWorkspace &w) { return w.value.capacity(); },
            "int: Allocated vector capacity, at least the current size.");
    bind_prepared(m).def(
        py::init(
            [](py::object A, int threads, const std::string &preconditioner, const std::string &kernel)
            {
                const auto selected = parse_dense_kernel(kernel);
                const auto kind = parse_preconditioner_kind(preconditioner);
                if (py::isinstance<cg::CsrMatrix>(A))
                {
                    return cg::PreparedSolver(A.cast<const cg::CsrMatrix &>(), threads, kind, selected);
                }
                if (!py::isinstance<py::array>(A))
                {
                    throw py::type_error("A must be a NumPy array or CsrMatrix");
                }
                return cg::PreparedSolver(snapshot_dense_matrix(A.cast<py::array>()), threads, kind, selected);
            }),
        py::arg("A"), py::arg("num_threads") = 1, py::arg("preconditioner") = "none",
        py::arg("dense_kernel") = "native");
    m.def(
        "solve_into",
        [](py::object A, const py::array &b, py::array x, PythonWorkspace &workspace, int threads, double rtol,
           double atol, std::size_t maxiter, const std::string &algorithm, bool fused, std::size_t residual_refresh,
           const std::string &preconditioner, const std::string &kernel)
        {
            const auto selected = parse_dense_kernel(kernel);
            const auto kind = parse_preconditioner_kind(preconditioner);
            const auto options = parse_solver_options(rtol, atol, maxiter, algorithm, residual_refresh);
            if (py::isinstance<cg::CsrMatrix>(A))
            {
                return run(A.cast<const cg::CsrMatrix &>(), b, x, workspace, threads, fused, options, kind, selected);
            }
            if (!py::isinstance<py::array>(A))
            {
                throw py::type_error("A must be a NumPy array or CsrMatrix");
            }
            return run(as_dense_matrix_view(A.cast<py::array>()), b, x, workspace, threads, fused, options, kind,
                       selected);
        },
        py::arg("A"), py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(), py::arg("workspace"),
        py::arg("num_threads") = 1, py::arg("rtol") = 1e-8, py::arg("atol") = 0.0, py::arg("maxiter") = 1000,
        py::arg("algorithm") = "standard", py::arg("fused") = true, py::arg("residual_refresh") = 50,
        py::arg("preconditioner") = "none", py::arg("dense_kernel") = "native",
        "Strict dense/CSR solve. Use cgpy.solve_into for the public array/option contract.");
}

} // namespace python
} // namespace cg
