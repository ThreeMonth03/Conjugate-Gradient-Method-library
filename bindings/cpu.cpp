#include <cg/csr.hpp>
#include <cg/preconditioner.hpp>
#include <cg/solver.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace
{
struct PythonWorkspace
{
    explicit PythonWorkspace(std::size_t n) : value(n) {}
    cg::Workspace value;
    std::atomic<bool> active{false};
};

struct WorkspaceUse
{
    explicit WorkspaceUse(PythonWorkspace &workspace) : workspace(workspace)
    {
        if (workspace.active.exchange(true))
        {
            throw std::invalid_argument("Workspace is already in use by another solve");
        }
    }
    ~WorkspaceUse() { workspace.active.store(false); }
    PythonWorkspace &workspace;
};

void vector_array(const py::array &value, bool writable)
{
    if (!value.dtype().is(py::dtype::of<double>()) || value.ndim() != 1 ||
        !(value.flags() & py::array::c_style))
    {
        throw std::invalid_argument("Expected a contiguous one-dimensional float64 array");
    }
    if (writable && !value.writeable())
    {
        throw std::invalid_argument("Output array must be writable");
    }
    if (value.size() != 0 && reinterpret_cast<std::uintptr_t>(value.data()) % alignof(double) != 0)
    {
        throw std::invalid_argument("Expected an aligned float64 array");
    }
}

cg::DenseMatrixView matrix_view(const py::array &value)
{
    if (!value.dtype().is(py::dtype::of<double>()) || value.ndim() != 2 ||
        !(value.flags() & py::array::c_style))
    {
        throw std::invalid_argument("Expected a C-contiguous two-dimensional float64 matrix");
    }
    return {static_cast<const double *>(value.data()), static_cast<std::size_t>(value.shape(0)),
            static_cast<std::size_t>(value.shape(1))};
}

cg::CgOptions options(double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                      std::size_t residual_refresh)
{
    cg::CgOptions result;
    result.rtol = rtol;
    result.atol = atol;
    result.maxiter = maxiter;
    result.residual_refresh = residual_refresh;
    if (algorithm == "standard")
    {
        result.algorithm = cg::Algorithm::recurrence;
    }
    else if (algorithm == "explicit")
    {
        result.algorithm = cg::Algorithm::explicit_residual;
    }
    else
    {
        throw std::invalid_argument("algorithm must be 'standard' or 'explicit'");
    }
    return result;
}

cg::DenseKernel dense_kernel(const std::string &name)
{
    if (name == "native")
        return cg::DenseKernel::native;
    if (name == "blas")
        return cg::DenseKernel::blas;
    throw std::invalid_argument("dense_kernel must be 'native' or 'blas'");
}

const char *kernel_name(cg::DenseKernel kernel)
{
    return kernel == cg::DenseKernel::blas ? "blas" : "native";
}

void validate_operator_kernel(cg::DenseMatrixView, cg::DenseKernel) {}

void validate_operator_kernel(const cg::CsrMatrix &, cg::DenseKernel kernel)
{
    if (kernel == cg::DenseKernel::blas)
        throw std::invalid_argument(
            "dense_kernel='blas' requires a dense matrix; CSR uses native kernels");
}

template <class Operator>
std::unique_ptr<cg::JacobiPreconditioner> prepare_preconditioner(const Operator &A,
                                                                 const std::string &name)
{
    if (name == "none")
        return {};
    if (name == "jacobi")
        return std::make_unique<cg::JacobiPreconditioner>(A);
    throw std::invalid_argument("preconditioner must be 'none' or 'jacobi'");
}

template <class Operator>
cg::SolveInfo run(const Operator &A, const py::array &b, py::array &x, PythonWorkspace &workspace,
                  int threads, bool fused, const cg::CgOptions &settings,
                  const cg::JacobiPreconditioner *M = nullptr,
                  cg::DenseKernel kernel = cg::DenseKernel::native)
{
    validate_operator_kernel(A, kernel);
    vector_array(b, false);
    vector_array(x, true);
    const cg::ConstVectorView bv{static_cast<const double *>(b.data()),
                                 static_cast<std::size_t>(b.size())};
    const cg::VectorView xv{static_cast<double *>(x.mutable_data()),
                            static_cast<std::size_t>(x.size())};
    const cg::CpuBackend backend(threads, fused, kernel);
    WorkspaceUse use(workspace);
    cg::SolveInfo result;
    {
        py::gil_scoped_release release;
        result = cg::solve_into(A, bv, xv, workspace.value, backend, settings, M);
    }
    return result;
}

cg::DenseMatrix snapshot(const py::array &A)
{
    const auto view = matrix_view(A);
    if (view.rows == 0 || view.rows != view.cols)
        throw std::invalid_argument("A must be a nonempty square matrix");
    return {view.rows, view.cols,
            std::vector<double>(view.data, view.data + view.rows * view.cols)};
}

cg::DenseMatrixView operator_view(const cg::DenseMatrix &A) { return A.view(); }
const cg::CsrMatrix &operator_view(const cg::CsrMatrix &A) { return A; }

template <class Matrix> class PreparedSolver
{
  public:
    PreparedSolver(Matrix matrix, int threads, const std::string &preconditioner,
                   const std::string &kernel)
        : matrix_(std::move(matrix)), threads_(threads), dense_kernel_(dense_kernel(kernel))
    {
        threads_ = cg::CpuBackend(threads, true, dense_kernel_).threads();
        validate_operator_kernel(operator_view(matrix_), dense_kernel_);
        if (matrix_.rows() == 0 || matrix_.rows() != matrix_.cols())
            throw std::invalid_argument("A must be a nonempty square matrix");
        const auto start = std::chrono::steady_clock::now();
        preconditioner_ = prepare_preconditioner(operator_view(matrix_), preconditioner);
        if (preconditioner_)
            setup_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    }
    std::unique_ptr<PythonWorkspace> create_workspace() const
    {
        return std::make_unique<PythonWorkspace>(matrix_.rows());
    }
    cg::SolveInfo solve_into(const py::array &b, py::array x, PythonWorkspace &workspace,
                             double rtol, double atol, std::size_t maxiter,
                             const std::string &algorithm, bool fused,
                             std::size_t residual_refresh) const
    {
        return run(operator_view(matrix_), b, x, workspace, threads_, fused,
                   options(rtol, atol, maxiter, algorithm, residual_refresh), preconditioner_.get(),
                   dense_kernel_);
    }
    std::size_t size() const { return matrix_.rows(); }
    int threads() const { return threads_; }
    const char *kernel() const { return kernel_name(dense_kernel_); }
    std::uint64_t setup_ns() const { return setup_ns_; }
    const char *preconditioner_name() const { return preconditioner_ ? "jacobi" : "none"; }

  private:
    Matrix matrix_;
    int threads_;
    cg::DenseKernel dense_kernel_;
    std::unique_ptr<cg::JacobiPreconditioner> preconditioner_;
    std::uint64_t setup_ns_ = 0;
};

template <class Matrix> auto bind_prepared(py::module_ &m, const char *name)
{
    using Plan = PreparedSolver<Matrix>;
    return py::class_<Plan>(m, name, R"doc(
An owning matrix snapshot with fixed thread, preconditioner and kernel settings.

Notes
-----
Construct through ``cgpy.prepare`` to normalize a dense input or wrap a public
CSRMatrix. The prepared object owns its matrix and optional Jacobi inverse.
Each overlapping solve needs a separate workspace and output. All calls are
synchronous, and native computation releases the GIL.
)doc")
        .def("create_workspace", &Plan::create_workspace, R"doc(
Allocate reusable solver vectors matching this system.

Returns
-------
Workspace
    New independent workspace for one active solve at a time.
)doc")
        .def_property_readonly("size", &Plan::size, "int: Matrix dimension and workspace size.")
        .def_property_readonly("num_threads", &Plan::threads,
                               "int: CPU thread count fixed at preparation time.")
        .def_property_readonly("dense_kernel", &Plan::kernel,
                               "str: Explicit 'native' or 'blas' kernel selected at preparation.")
        .def_property_readonly("preconditioner_setup_ns", &Plan::setup_ns,
                               "int: Jacobi construction time in nanoseconds; zero without Jacobi.")
        .def_property_readonly("preconditioner", &Plan::preconditioner_name,
                               "str: 'none' or 'jacobi', fixed at preparation time.")
        .def("solve_into", &Plan::solve_into, py::arg("b").noconvert(), py::arg("x").noconvert(),
             py::kw_only(), py::arg("workspace"), py::arg("rtol") = 1e-8, py::arg("atol") = 0.0,
             py::arg("maxiter") = 1000, py::arg("algorithm") = "standard", py::arg("fused") = true,
             py::arg("residual_refresh") = 50,
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
    Residual recurrence or legacy explicit formulation. Jacobi requires standard.
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

PYBIND11_MODULE(_cgcore, m)
{
    m.doc() = "Synchronous CPU CG kernels with caller-owned output and reusable workspace";
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
            "converged",
            [](const cg::SolveInfo &info) { return info.status == cg::Status::converged; },
            "bool: Whether the true-residual stopping rule was satisfied.")
        .def_readonly("iterations", &cg::SolveInfo::iterations,
                      "int: Number of completed solution updates.")
        .def_readonly("matvecs", &cg::SolveInfo::matvecs,
                      "int: Matrix-vector products, including true-residual checks.")
        .def_readonly("residual_norm", &cg::SolveInfo::residual_norm,
                      "float: Independently evaluated final Euclidean residual norm.")
        .def_readonly("restarts", &cg::SolveInfo::restarts,
                      "int: Direction restarts following residual replacement.");
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
    py::class_<cg::CsrMatrix>(m, "CSRMatrix",
                              "Owning canonical CSR storage; construct through cgpy.CSRMatrix.")
        .def(py::init(
                 [](std::vector<std::size_t> indptr, std::vector<std::size_t> indices,
                    std::vector<double> data, std::pair<std::size_t, std::size_t> shape)
                 {
                     return cg::CsrMatrix(shape.first, shape.second, std::move(indptr),
                                          std::move(indices), std::move(data));
                 }),
             py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("shape"))
        .def_property_readonly(
            "shape", [](const cg::CsrMatrix &A) { return py::make_tuple(A.rows(), A.cols()); },
            "tuple of int: Matrix dimensions (rows, columns).")
        .def_property_readonly("nnz", &cg::CsrMatrix::nnz,
                               "int: Number of stored entries, including stored zeros.");
    bind_prepared<cg::DenseMatrix>(m, "PreparedSolver")
        .def(py::init(
                 [](const py::array &A, int threads, const std::string &preconditioner,
                    const std::string &kernel)
                 {
                     return std::make_unique<PreparedSolver<cg::DenseMatrix>>(
                         snapshot(A), threads, preconditioner, kernel);
                 }),
             py::arg("A").noconvert(), py::arg("num_threads") = 1,
             py::arg("preconditioner") = "none", py::arg("dense_kernel") = "native");
    bind_prepared<cg::CsrMatrix>(m, "PreparedCsrSolver")
        .def(py::init<cg::CsrMatrix, int, const std::string &, const std::string &>(), py::arg("A"),
             py::arg("num_threads") = 1, py::arg("preconditioner") = "none",
             py::arg("dense_kernel") = "native");
    m.def(
        "solve_csr_into",
        [](const cg::CsrMatrix &A, const py::array &b, py::array x, PythonWorkspace &workspace,
           int threads, double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
           bool fused, std::size_t residual_refresh, const std::string &preconditioner,
           const std::string &kernel)
        {
            const auto selected = dense_kernel(kernel);
            validate_operator_kernel(A, selected);
            const auto M = prepare_preconditioner(A, preconditioner);
            return run(A, b, x, workspace, threads, fused,
                       options(rtol, atol, maxiter, algorithm, residual_refresh), M.get(),
                       selected);
        },
        py::arg("A"), py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(),
        py::arg("workspace"), py::arg("num_threads") = 1, py::arg("rtol") = 1e-8,
        py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
        py::arg("fused") = true, py::arg("residual_refresh") = 50,
        py::arg("preconditioner") = "none", py::arg("dense_kernel") = "native",
        "Low-level strict CSR solve. Use cgpy.solve_into for the public array/option contract.");
    m.def(
        "solve_into",
        [](const py::array &A, const py::array &b, py::array x, PythonWorkspace &workspace,
           int threads, double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
           bool fused, std::size_t residual_refresh, const std::string &preconditioner,
           const std::string &kernel)
        {
            const auto M = prepare_preconditioner(matrix_view(A), preconditioner);
            return run(matrix_view(A), b, x, workspace, threads, fused,
                       options(rtol, atol, maxiter, algorithm, residual_refresh), M.get(),
                       dense_kernel(kernel));
        },
        py::arg("A").noconvert(), py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(),
        py::arg("workspace"), py::arg("num_threads") = 1, py::arg("rtol") = 1e-8,
        py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
        py::arg("fused") = true, py::arg("residual_refresh") = 50,
        py::arg("preconditioner") = "none", py::arg("dense_kernel") = "native",
        "Low-level strict dense solve. Use cgpy.solve_into for the public array/option contract.");
    m.def("openmp_enabled", &cg::CpuBackend::openmp_enabled, R"doc(
Report OpenMP support in the extension imported by this process.

Returns
-------
bool
    Whether native kernel calls can request more than one CPU thread.
)doc");
    m.def("blas_enabled", &cg::CpuBackend::blas_enabled,
          R"doc(
Report optional sequential BLAS support in the imported extension.

Returns
-------
bool
    Whether ``dense_kernel='blas'`` is available for dense, one-thread calls.

Notes
-----
This reports the actual imported binary. It does not change kernel selection
or inspect another build directory. Native remains the default kernel.
)doc");
    m.def(
        "build_info",
        []
        {
            py::dict result;
            result["compiler"] = CG_COMPILER;
            result["build_type"] = CG_BUILD_TYPE;
            result["flags"] = CG_BUILD_FLAGS;
            result["openmp"] = cg::CpuBackend::openmp_enabled();
            py::dict blas;
            const bool enabled = cg::CpuBackend::blas_enabled();
            blas["enabled"] = enabled;
            blas["configuration"] = cg::CpuBackend::blas_info();
            if (enabled)
            {
                blas["provider"] = "OpenBLAS";
                blas["integer_bits"] = 32;
                blas["thread_mode"] = "sequential";
                blas["symbol_prefix"] = "cgpy_";
            }
            else
            {
                blas["provider"] = py::none();
                blas["integer_bits"] = py::none();
                blas["thread_mode"] = py::none();
                blas["symbol_prefix"] = py::none();
            }
            result["blas"] = blas;
            return result;
        },
        R"doc(
Describe the native extension imported by this Python process.

Returns
-------
dict
    Compiler, build type, flags and OpenMP support. The ``blas`` dictionary
    records ``enabled``, provider/configuration, integer width, thread mode
    and symbol prefix. Provider-specific fields are None without BLAS.

Notes
-----
BLAS-enabled builds use the private sequential LP64 OpenBLAS provider. Build
information belongs to this loaded binary, not another preset or installed wheel.
)doc");
}
