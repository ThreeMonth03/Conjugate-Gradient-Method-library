#include <cg/solver.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <atomic>
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

cg::SolveInfo run(cg::DenseMatrixView A, const py::array &b, py::array &x,
                  PythonWorkspace &workspace, int threads, bool fused,
                  const cg::CgOptions &settings)
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
        result = cg::solve_into(A, bv, xv, workspace.value, backend, settings);
    }
    return result;
}

class PreparedSolver
{
  public:
    PreparedSolver(const py::array &A, int threads)
        : matrix_(snapshot(A)), threads_(cg::CpuBackend(threads).threads())
    {
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
        return run(matrix_.view(), b, x, workspace, threads_, fused,
                   options(rtol, atol, maxiter, algorithm, residual_refresh));
    }

    std::size_t size() const { return matrix_.rows(); }
    int threads() const { return threads_; }

  private:
    static cg::DenseMatrix snapshot(const py::array &A)
    {
        const auto view = matrix_view(A);
        if (view.rows == 0 || view.rows != view.cols)
        {
            throw std::invalid_argument("A must be a nonempty square matrix");
        }
        return {view.rows, view.cols,
                std::vector<double>(view.data, view.data + view.rows * view.cols)};
    }
    cg::DenseMatrix matrix_;
    int threads_;
};
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
    py::class_<PreparedSolver>(m, "PreparedSolver")
        .def(py::init<const py::array &, int>(), py::arg("A").noconvert(),
             py::arg("num_threads") = 1)
        .def("create_workspace", &PreparedSolver::create_workspace)
        .def_property_readonly("size", &PreparedSolver::size)
        .def_property_readonly("num_threads", &PreparedSolver::threads)
        .def("solve_into", &PreparedSolver::solve_into, py::arg("b").noconvert(),
             py::arg("x").noconvert(), py::kw_only(), py::arg("workspace"), py::arg("rtol") = 1e-8,
             py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
             py::arg("fused") = true, py::arg("residual_refresh") = 50,
             "Solve with aligned contiguous float64 arrays. Invalid arrays raise ValueError; "
             "negative or nonintegral iteration counts raise TypeError at the native boundary.");
    m.def(
        "solve_into",
        [](const py::array &A, const py::array &b, py::array x, PythonWorkspace &workspace,
           int threads, double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
           bool fused, std::size_t residual_refresh)
        {
            return run(matrix_view(A), b, x, workspace, threads, fused,
                       options(rtol, atol, maxiter, algorithm, residual_refresh));
        },
        py::arg("A").noconvert(), py::arg("b").noconvert(), py::arg("x").noconvert(), py::kw_only(),
        py::arg("workspace"), py::arg("num_threads") = 1, py::arg("rtol") = 1e-8,
        py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
        py::arg("fused") = true, py::arg("residual_refresh") = 50);
    m.def("openmp_enabled", &cg::CpuBackend::openmp_enabled);
}
