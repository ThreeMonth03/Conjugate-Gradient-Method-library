#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>
#include <cg/python/module.hpp>
#include <cg/cuda.hpp>

#include <memory>

#include <chrono>

namespace cg
{
namespace python
{

namespace
{
gpu::DenseKernel kernel(const std::string &name)
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

struct CudaWorkspace
{
    CudaWorkspace(std::size_t size, int device, std::uintptr_t stream) : context(device, stream), value(size, context)
    {
    }
    gpu::Context context;
    gpu::Workspace value;
};

template <class Matrix>
std::unique_ptr<gpu::JacobiPreconditioner> preconditioner(const Matrix &A, const gpu::Context &context,
                                                          const std::string &name)
{
    if (name == "none")
    {
        return {};
    }
    if (name == "jacobi")
    {
        return std::make_unique<gpu::JacobiPreconditioner>(A, context);
    }
    throw std::invalid_argument("preconditioner must be 'none' or 'jacobi'");
}

class PreparedCudaSolver
{
    public:
    PreparedCudaSolver(py::object matrix, int device, std::uintptr_t stream, const std::string &kernel_name,
                       const std::string &preconditioner_name, bool from_host = false)
        : m_device(device), m_kernel(kernel(kernel_name)), m_preconditioner_name(preconditioner_name)
    {
        const gpu::Context context(device, stream);
        if (from_host && py::isinstance<cg::CsrMatrix>(matrix))
        {
            if (m_kernel != gpu::DenseKernel::native)
            {
                throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
            }
            m_csr = std::make_unique<gpu::CsrMatrix>(matrix.cast<const cg::CsrMatrix &>(), context);
        }
        else if (from_host)
        {
            m_dense = std::make_unique<gpu::DenseMatrix>(host_matrix(matrix.cast<py::array>()), context);
        }
        else if (py::isinstance<gpu::CsrMatrix>(matrix))
        {
            if (m_kernel != gpu::DenseKernel::native)
            {
                throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
            }
            m_csr = std::make_unique<gpu::CsrMatrix>(matrix.cast<const gpu::CsrMatrix &>().view(), context);
        }
        else if (py::isinstance<gpu::DenseMatrix>(matrix))
        {
            m_dense = std::make_unique<gpu::DenseMatrix>(matrix.cast<const gpu::DenseMatrix &>().view(), context);
        }
        else
        {
            const auto input = device_array(matrix, 2, context, false);
            m_dense = std::make_unique<gpu::DenseMatrix>(matrix_view(input), context);
        }
        if (size() == 0 || size() != (m_dense ? m_dense->cols() : m_csr->cols()))
        {
            throw std::invalid_argument("CG requires a nonempty square matrix");
        }
        const auto start = std::chrono::steady_clock::now();
        m_preconditioner = m_dense ? preconditioner(m_dense->view(), context, preconditioner_name)
                                   : preconditioner(*m_csr, context, preconditioner_name);
        if (m_preconditioner)
        {
            m_setup_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }
    }
    std::size_t size() const { return m_dense ? m_dense->rows() : m_csr->rows(); }
    int device() const { return m_device; }
    std::string kernel_name() const { return m_kernel == gpu::DenseKernel::native ? "native" : "cublas"; }
    const std::string &preconditioner_name() const { return m_preconditioner_name; }
    std::uint64_t setup_ns() const { return m_setup_ns; }
    std::size_t preconditioner_bytes() const { return m_preconditioner ? size() * sizeof(double) : 0; }
    cg::SolveInfo solve(py::object b, py::object x, CudaWorkspace &workspace, double rtol, double atol,
                        std::size_t maxiter, const std::string &algorithm, bool fused, std::size_t refresh) const
    {
        if (workspace.context.device() != m_device)
        {
            throw std::invalid_argument("workspace device does not match prepared solver");
        }
        const auto bv = device_array(b, 1, workspace.context, false);
        const auto xv = device_array(x, 1, workspace.context, true);
        const auto settings = options(rtol, atol, maxiter, algorithm, refresh);
        py::gil_scoped_release release;
        return m_dense ? gpu::solve_into(m_dense->view(), vector_view(bv), mutable_view(xv), workspace.value,
                                         workspace.context, settings, fused, m_kernel, m_preconditioner.get())
                       : gpu::solve_into(*m_csr, vector_view(bv), mutable_view(xv), workspace.value, workspace.context,
                                         settings, fused, m_kernel, m_preconditioner.get());
    }

    private:
    int m_device;
    gpu::DenseKernel m_kernel;
    std::string m_preconditioner_name;
    std::unique_ptr<gpu::DenseMatrix> m_dense;
    std::unique_ptr<gpu::CsrMatrix> m_csr;
    std::unique_ptr<gpu::JacobiPreconditioner> m_preconditioner;
    std::uint64_t m_setup_ns = 0;
};

cg::SolveInfo solve(py::object matrix, py::object b, py::object x, CudaWorkspace &workspace, double rtol, double atol,
                    std::size_t maxiter, const std::string &algorithm, bool fused, std::size_t refresh,
                    const std::string &preconditioner_name, const std::string &kernel_name)
{
    const auto selected = kernel(kernel_name);
    const auto settings = options(rtol, atol, maxiter, algorithm, refresh);
    const auto bv = device_array(b, 1, workspace.context, false);
    const auto xv = device_array(x, 1, workspace.context, true);
    if (py::isinstance<gpu::CsrMatrix>(matrix))
    {
        if (selected != gpu::DenseKernel::native)
        {
            throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
        }
        const auto &A = matrix.cast<const gpu::CsrMatrix &>();
        py::gil_scoped_release release;
        const auto M = preconditioner(A, workspace.context, preconditioner_name);
        return gpu::solve_into(A, vector_view(bv), mutable_view(xv), workspace.value, workspace.context, settings,
                               fused, selected, M.get());
    }
    gpu::DenseMatrixView A;
    DeviceArray owner;
    if (py::isinstance<gpu::DenseMatrix>(matrix))
    {
        A = matrix.cast<const gpu::DenseMatrix &>().view();
    }
    else
    {
        owner = device_array(matrix, 2, workspace.context, false);
        A = matrix_view(owner);
    }
    py::gil_scoped_release release;
    const auto M = preconditioner(A, workspace.context, preconditioner_name);
    return gpu::solve_into(A, vector_view(bv), mutable_view(xv), workspace.value, workspace.context, settings, fused,
                           selected, M.get());
}
} // namespace

void register_cuda_solver(py::module_ &m)
{
    py::class_<CudaWorkspace>(m, "Workspace")
        .def(py::init<std::size_t, int, std::uintptr_t>(), py::arg("size"), py::arg("device") = 0,
             py::arg("stream") = 0)
        .def_property_readonly("size", [](const CudaWorkspace &value) { return value.value.size(); })
        .def_property_readonly("device", [](const CudaWorkspace &value) { return value.context.device(); })
        .def_property_readonly("stream", [](const CudaWorkspace &value) { return value.context.stream(); });
    py::class_<PreparedCudaSolver>(m, "PreparedSolver")
        .def(py::init<py::object, int, std::uintptr_t, const std::string &, const std::string &>(), py::arg("matrix"),
             py::arg("device") = 0, py::arg("stream") = 0, py::arg("dense_kernel") = "native",
             py::arg("preconditioner") = "none")
        .def_static(
            "from_host",
            [](py::object matrix, int device, std::uintptr_t stream, const std::string &kernel_name,
               const std::string &preconditioner_name)
            {
                return std::make_unique<PreparedCudaSolver>(matrix, device, stream, kernel_name, preconditioner_name,
                                                            true);
            },
            py::arg("matrix"), py::arg("device") = 0, py::arg("stream") = 0, py::arg("dense_kernel") = "native",
            py::arg("preconditioner") = "none")
        .def_property_readonly("size", &PreparedCudaSolver::size)
        .def_property_readonly("device", &PreparedCudaSolver::device)
        .def_property_readonly("dense_kernel", &PreparedCudaSolver::kernel_name)
        .def_property_readonly("preconditioner", &PreparedCudaSolver::preconditioner_name)
        .def_property_readonly("preconditioner_setup_ns", &PreparedCudaSolver::setup_ns)
        .def_property_readonly("preconditioner_nbytes", &PreparedCudaSolver::preconditioner_bytes)
        .def("solve_into", &PreparedCudaSolver::solve, py::arg("b"), py::arg("x"), py::kw_only(), py::arg("workspace"),
             py::arg("rtol") = 1e-8, py::arg("atol") = 0.0, py::arg("maxiter") = 1000,
             py::arg("algorithm") = "standard", py::arg("fused") = true, py::arg("residual_refresh") = 50);
    m.def("solve_into", &solve, py::arg("A"), py::arg("b"), py::arg("x"), py::kw_only(), py::arg("workspace"),
          py::arg("rtol") = 1e-8, py::arg("atol") = 0.0, py::arg("maxiter") = 1000, py::arg("algorithm") = "standard",
          py::arg("fused") = true, py::arg("residual_refresh") = 50, py::arg("preconditioner") = "none",
          py::arg("dense_kernel") = "native");
}

} // namespace python
} // namespace cg
