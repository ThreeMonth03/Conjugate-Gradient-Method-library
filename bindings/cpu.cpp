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
                  const cg::JacobiPreconditioner *M = nullptr)
{
    vector_array(b, false);
    vector_array(x, true);
    const cg::ConstVectorView bv{static_cast<const double *>(b.data()),
                                 static_cast<std::size_t>(b.size())};
    const cg::VectorView xv{static_cast<double *>(x.mutable_data()),
                            static_cast<std::size_t>(x.size())};
    const cg::CpuBackend backend(threads, fused);
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
    PreparedSolver(Matrix matrix, int threads, const std::string &preconditioner)
        : matrix_(std::move(matrix)), threads_(cg::CpuBackend(threads).threads())
    {
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
                   options(rtol, atol, maxiter, algorithm, residual_refresh),
                   preconditioner_.get());
    }
    std::size_t size() const { return matrix_.rows(); }
    int threads() const { return threads_; }
    std::uint64_t setup_ns() const { return setup_ns_; }
    const char *preconditioner_name() const { return preconditioner_ ? "jacobi" : "none"; }

  private:
    Matrix matrix_;
    int threads_;
    std::unique_ptr<cg::JacobiPreconditioner> preconditioner_;
    std::uint64_t setup_ns_ = 0;
};

template <class Matrix> auto bind_prepared(py::module_ &m, const char *name)
{
    using Plan = PreparedSolver<Matrix>;
    return py::class_<Plan>(m, name)
        .def("create_workspace", &Plan::create_workspace)
        .def_property_readonly("size", &Plan::size)
        .def_property_readonly("num_threads", &Plan::threads)
        .def_property_readonly("preconditioner_setup_ns", &Plan::setup_ns)
        .def_property_readonly("preconditioner", &Plan::preconditioner_name)
        .def("solve_into", &Plan::solve_into, py::arg("b").noconvert(), py::arg("x").noconvert(),
             py::kw_only(), py::arg("workspace"), py::arg("rtol") = 1e-8, py::arg("atol") = 0.0,
             py::arg("maxiter") = 1000, py::arg("algorithm") = "standard", py::arg("fused") = true,
             py::arg("residual_refresh") = 50,
             "Solve with aligned contiguous float64 arrays. Invalid arrays raise ValueError; "
             "negative or nonintegral iteration counts raise TypeError at the native boundary.");
}
} // namespace

PYBIND11_MODULE(_cgcore, m)
{
    m.doc() = "Synchronous CPU CG kernels with caller-owned output and reusable workspace";
    py::class_<cg::SolveInfo>(m, "SolveInfo")
        .def_property_readonly("status", [](const cg::SolveInfo &info)
                               { return cg::status_name(info.status); })
        .def_property_readonly("converged", [](const cg::SolveInfo &info)
                               { return info.status == cg::Status::converged; })
        .def_readonly("iterations", &cg::SolveInfo::iterations)
        .def_readonly("matvecs", &cg::SolveInfo::matvecs)
        .def_readonly("residual_norm", &cg::SolveInfo::residual_norm)
        .def_readonly("restarts", &cg::SolveInfo::restarts);
    py::class_<PythonWorkspace>(m, "Workspace")
        .def(py::init<std::size_t>(), py::arg("size"))
        .def_property_readonly("size", [](const PythonWorkspace &w) { return w.value.size(); })
        .def_property_readonly("capacity",
                               [](const PythonWorkspace &w) { return w.value.capacity(); });
    py::class_<cg::CsrMatrix>(m, "CSRMatrix")
        .def(py::init(
                 [](std::vector<std::size_t> indptr, std::vector<std::size_t> indices,
                    std::vector<double> data, std::pair<std::size_t, std::size_t> shape)
                 {
                     return cg::CsrMatrix(shape.first, shape.second, std::move(indptr),
                                          std::move(indices), std::move(data));
                 }),
             py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("shape"))
        .def_property_readonly("shape", [](const cg::CsrMatrix &A)
                               { return py::make_tuple(A.rows(), A.cols()); })
        .def_property_readonly("nnz", &cg::CsrMatrix::nnz);
    bind_prepared<cg::DenseMatrix>(m, "PreparedSolver")
        .def(py::init(
                 [](const py::array &A, int threads, const std::string &preconditioner)
                 {
                     return std::make_unique<PreparedSolver<cg::DenseMatrix>>(snapshot(A), threads,
                                                                              preconditioner);
                 }),
             py::arg("A").noconvert(), py::arg("num_threads") = 1,
             py::arg("preconditioner") = "none");
    bind_prepared<cg::CsrMatrix>(m, "PreparedCsrSolver")
        .def(py::init<cg::CsrMatrix, int, const std::string &>(), py::arg("A"),
             py::arg("num_threads") = 1, py::arg("preconditioner") = "none");
    m.def(
        "solve_csr_into",
        [](const cg::CsrMatrix &A, const py::array &b, py::array x, PythonWorkspace &workspace,
           int threads, double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
           bool fused, std::size_t residual_refresh, const std::string &preconditioner)
        {
            const auto M = prepare_preconditioner(A, preconditioner);
            return run(A, b, x, workspace, threads, fused,
                       options(rtol, atol, maxiter, algorithm, residual_refresh), M.get());
        },
        py::arg("A"), py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(),
        py::arg("workspace"), py::arg("num_threads") = 1, py::arg("rtol") = 1e-8,
        py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
        py::arg("fused") = true, py::arg("residual_refresh") = 50,
        py::arg("preconditioner") = "none");
    m.def(
        "solve_into",
        [](const py::array &A, const py::array &b, py::array x, PythonWorkspace &workspace,
           int threads, double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
           bool fused, std::size_t residual_refresh, const std::string &preconditioner)
        {
            const auto M = prepare_preconditioner(matrix_view(A), preconditioner);
            return run(matrix_view(A), b, x, workspace, threads, fused,
                       options(rtol, atol, maxiter, algorithm, residual_refresh), M.get());
        },
        py::arg("A").noconvert(), py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(),
        py::arg("workspace"), py::arg("num_threads") = 1, py::arg("rtol") = 1e-8,
        py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
        py::arg("fused") = true, py::arg("residual_refresh") = 50,
        py::arg("preconditioner") = "none");
    m.def("openmp_enabled", &cg::CpuBackend::openmp_enabled);
    m.def("build_info",
          []
          {
              py::dict result;
              result["compiler"] = CG_COMPILER;
              result["build_type"] = CG_BUILD_TYPE;
              result["flags"] = CG_BUILD_FLAGS;
              result["openmp"] = cg::CpuBackend::openmp_enabled();
              return result;
          });
}
