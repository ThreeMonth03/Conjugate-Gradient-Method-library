#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>
#include <cg/python/module.hpp>
#include <cg/cuda.hpp>
#include <cg/solver/cuda/prepared_solver.hpp>
#include <cg/solver/pymod/options.hpp>

#include <memory>

namespace cg
{
namespace python
{
namespace
{

gpu::DenseKernel parse_dense_kernel(const std::string &name)
{
    if (name == "native")
    {
        return gpu::DenseKernel::native;
    }
    if (name == "cublas")
    {
        return gpu::DenseKernel::cublas;
    }
    throw std::invalid_argument("dense_kernel must be 'native' or 'cublas'");
}
const char *dense_kernel_name(gpu::DenseKernel kernel)
{
    return kernel == gpu::DenseKernel::native ? "native" : "cublas";
}

gpu::ExecutionMode parse_execution_mode(const std::string &name)
{
    if (name == "host")
    {
        return gpu::ExecutionMode::host;
    }
    if (name == "device")
    {
        return gpu::ExecutionMode::device;
    }
    throw std::invalid_argument("execution_mode must be 'host' or 'device'");
}
const char *execution_mode_name(gpu::ExecutionMode mode)
{
    return mode == gpu::ExecutionMode::host ? "host" : "device";
}

std::unique_ptr<gpu::Workspace> create_workspace(std::size_t size, int device, std::uintptr_t stream,
                                                 const std::string &kernel_name, const std::string &mode_name)
{
    const auto kernel = parse_dense_kernel(kernel_name);
    const auto mode = parse_execution_mode(mode_name);
    py::gil_scoped_release release;
    const gpu::Context context(device, stream);
    return std::make_unique<gpu::Workspace>(size, context, kernel, mode);
}

std::unique_ptr<gpu::PreparedSolver> prepare_device(py::object matrix, int device, std::uintptr_t stream,
                                                    const std::string &kernel_name,
                                                    const std::string &preconditioner_name)
{
    const auto kernel = parse_dense_kernel(kernel_name);
    const auto preconditioner = parse_preconditioner_kind(preconditioner_name);
    const gpu::Context context(device, stream);
    if (py::isinstance<gpu::CsrMatrix>(matrix))
    {
        const auto &source = matrix.cast<const gpu::CsrMatrix &>();
        py::gil_scoped_release release;
        return std::make_unique<gpu::PreparedSolver>(gpu::CsrMatrix(source.view(), context), context, preconditioner,
                                                     kernel);
    }
    if (py::isinstance<gpu::DenseMatrix>(matrix))
    {
        const auto &source = matrix.cast<const gpu::DenseMatrix &>();
        py::gil_scoped_release release;
        return std::make_unique<gpu::PreparedSolver>(gpu::DenseMatrix(source.view(), context), context, preconditioner,
                                                     kernel);
    }
    const auto source = device_array(matrix, 2, context, false);
    py::gil_scoped_release release;
    return std::make_unique<gpu::PreparedSolver>(gpu::DenseMatrix(as_dense_matrix_view(source), context), context,
                                                 preconditioner, kernel);
}

std::unique_ptr<gpu::PreparedSolver> prepare_host(py::object matrix, int device, std::uintptr_t stream,
                                                  const std::string &kernel_name,
                                                  const std::string &preconditioner_name)
{
    const auto kernel = parse_dense_kernel(kernel_name);
    const auto preconditioner = parse_preconditioner_kind(preconditioner_name);
    const gpu::Context context(device, stream);
    if (py::isinstance<cg::CsrMatrix>(matrix))
    {
        const auto &source = matrix.cast<const cg::CsrMatrix &>();
        py::gil_scoped_release release;
        return std::make_unique<gpu::PreparedSolver>(gpu::CsrMatrix(source, context), context, preconditioner, kernel);
    }
    const auto source = matrix.cast<py::array>();
    const auto view = as_dense_matrix_view(source);
    py::gil_scoped_release release;
    return std::make_unique<gpu::PreparedSolver>(gpu::DenseMatrix(view, context), context, preconditioner, kernel);
}

cg::SolveInfo solve_prepared(const gpu::PreparedSolver &solver, py::object b, py::object x, gpu::Workspace &workspace,
                             double rtol, double atol, const py::object &maxiter, const std::string &algorithm,
                             bool fused, const py::object &refresh)
{
    const auto settings = parse_solver_options(rtol, atol, maxiter, algorithm, refresh);
    const auto bv = device_array(b, 1, workspace.context(), false);
    const auto xv = device_array(x, 1, workspace.context(), true);
    py::gil_scoped_release release;
    return solver.solve_into(as_const_vector_view(bv), as_mutable_vector_view(xv), workspace, settings, fused);
}

cg::SolveInfo solve(py::object matrix, py::object b, py::object x, gpu::Workspace &workspace, double rtol, double atol,
                    const py::object &maxiter, const std::string &algorithm, bool fused, const py::object &refresh,
                    const std::string &preconditioner_name)
{
    const auto preconditioner = parse_preconditioner_kind(preconditioner_name);
    const auto settings = parse_solver_options(rtol, atol, maxiter, algorithm, refresh);
    const auto bv = device_array(b, 1, workspace.context(), false);
    const auto xv = device_array(x, 1, workspace.context(), true);
    if (py::isinstance<gpu::CsrMatrix>(matrix))
    {
        const auto &A = matrix.cast<const gpu::CsrMatrix &>();
        py::gil_scoped_release release;
        return gpu::solve_into(A, as_const_vector_view(bv), as_mutable_vector_view(xv), workspace, settings, fused,
                               preconditioner);
    }
    gpu::DenseMatrixView A;
    DeviceArray owner;
    if (py::isinstance<gpu::DenseMatrix>(matrix))
    {
        A = matrix.cast<const gpu::DenseMatrix &>().view();
    }
    else
    {
        owner = device_array(matrix, 2, workspace.context(), false);
        A = as_dense_matrix_view(owner);
    }
    py::gil_scoped_release release;
    return gpu::solve_into(A, as_const_vector_view(bv), as_mutable_vector_view(xv), workspace, settings, fused,
                           preconditioner);
}

} // namespace

void register_cuda_solver(py::module_ &m)
{
    py::class_<gpu::Workspace>(m, "Workspace")
        .def(py::init(&create_workspace), py::arg("size"), py::arg("device") = 0, py::arg("stream") = 0,
             py::arg("dense_kernel") = "native", py::arg("execution_mode") = "host")
        .def_property_readonly("size", &gpu::Workspace::size)
        .def_property_readonly("device", &gpu::Workspace::device)
        .def_property_readonly("stream", &gpu::Workspace::stream)
        .def_property_readonly("dense_kernel",
                               [](const gpu::Workspace &value) { return dense_kernel_name(value.dense_kernel()); })
        .def_property_readonly("execution_mode",
                               [](const gpu::Workspace &value) { return execution_mode_name(value.execution_mode()); });
    py::class_<gpu::PreparedSolver>(m, "PreparedSolver")
        .def(py::init(&prepare_device), py::arg("matrix"), py::arg("device") = 0, py::arg("stream") = 0,
             py::arg("dense_kernel") = "native", py::arg("preconditioner") = "none")
        .def_static("from_host", &prepare_host, py::arg("matrix"), py::arg("device") = 0, py::arg("stream") = 0,
                    py::arg("dense_kernel") = "native", py::arg("preconditioner") = "none")
        .def_property_readonly("size", &gpu::PreparedSolver::size)
        .def_property_readonly("device", &gpu::PreparedSolver::device)
        .def_property_readonly("dense_kernel",
                               [](const gpu::PreparedSolver &value) { return dense_kernel_name(value.dense_kernel()); })
        .def_property_readonly("preconditioner", [](const gpu::PreparedSolver &value)
                               { return cg::preconditioner_name(value.preconditioner()); })
        .def_property_readonly("preconditioner_setup_ns", &gpu::PreparedSolver::preconditioner_setup_ns)
        .def_property_readonly("preconditioner_nbytes", &gpu::PreparedSolver::preconditioner_nbytes)
        .def("solve_into", &solve_prepared, py::arg("b"), py::arg("x"), py::kw_only(), py::arg("workspace"),
             py::arg("rtol") = 1e-8, py::arg("atol") = 0.0, py::arg("maxiter") = 1000,
             py::arg("algorithm") = "standard", py::arg("fused") = true, py::arg("residual_refresh") = 50);
    m.def("solve_into", &solve, py::arg("A"), py::arg("b"), py::arg("x"), py::kw_only(), py::arg("workspace"),
          py::arg("rtol") = 1e-8, py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
          py::arg("fused") = true, py::arg("residual_refresh") = 50, py::arg("preconditioner") = "none");
}

} // namespace python
} // namespace cg
