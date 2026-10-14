#include <cg/cuda.hpp>
#include <cuda_runtime_api.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace gpu = cg::cuda;
namespace
{
void check(cudaError_t error)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error));
}

class DeviceScope
{
  public:
    explicit DeviceScope(int device)
    {
        check(cudaGetDevice(&previous_));
        check(cudaSetDevice(device));
    }
    ~DeviceScope() { cudaSetDevice(previous_); }

  private:
    int previous_ = 0;
};

struct Array
{
    py::object owner;
    void *data;
    std::vector<std::size_t> shape;
    std::size_t itemsize;
    int device;
    std::string typestr;
};

Array device_array(py::object object, int ndim, const gpu::Context &context, bool writable,
                   bool integer = false)
{
    if (!py::hasattr(object, "__cuda_array_interface__"))
        throw std::invalid_argument("expected a device array with __cuda_array_interface__");
    const auto interface = object.attr("__cuda_array_interface__").cast<py::dict>();
    const int version = interface["version"].cast<int>();
    if (version != 2 && version != 3)
        throw std::invalid_argument("CUDA Array Interface version must be 2 or 3");
    if (interface.contains("mask") && !interface["mask"].is_none())
        throw std::invalid_argument("masked CUDA arrays are unsupported");
    const auto shape = interface["shape"].cast<std::vector<std::size_t>>();
    if (shape.size() != static_cast<std::size_t>(ndim))
        throw std::invalid_argument("CUDA array dimension mismatch");
    const auto typestr = interface["typestr"].cast<std::string>();
    const bool f64 = typestr == "<f8" || typestr == "=f8";
    const bool i32 = typestr == "<i4" || typestr == "=i4";
    const bool i64 = typestr == "<i8" || typestr == "=i8";
    if (integer ? !(i32 || i64) : !f64)
        throw std::invalid_argument(integer ? "CUDA CSR indices require signed int32 or int64"
                                            : "CUDA arrays require native float64");
    const std::size_t itemsize = i32 ? 4 : 8;
    if (interface.contains("strides") && !interface["strides"].is_none())
    {
        const auto strides = interface["strides"].cast<std::vector<py::ssize_t>>();
        if (strides.size() != shape.size())
            throw std::invalid_argument("CUDA strides must match dimensions");
        std::size_t expected = itemsize;
        for (std::size_t i = shape.size(); i-- != 0;)
        {
            if (shape[i] > 1 &&
                (strides[i] < 0 || static_cast<std::size_t>(strides[i]) != expected))
                throw std::invalid_argument("CUDA arrays must be C-contiguous");
            if (shape[i] != 0 && expected > SIZE_MAX / shape[i])
                throw std::invalid_argument("CUDA shape byte size overflows size_t");
            expected *= shape[i];
        }
    }
    const auto data = interface["data"].cast<py::tuple>();
    if (data.size() != 2 || (writable && data[1].cast<bool>()))
        throw std::invalid_argument("CUDA output must be writable");
    const auto address = data[0].cast<std::uintptr_t>();
    std::size_t elements = 1;
    for (const auto dimension : shape)
    {
        if (dimension != 0 && elements > SIZE_MAX / dimension)
            throw std::invalid_argument("CUDA shape size overflows size_t");
        elements *= dimension;
    }
    if (elements > SIZE_MAX / itemsize || address % itemsize != 0)
        throw std::invalid_argument("CUDA array size or alignment is invalid");
    DeviceScope selected(context.device());
    if (elements != 0)
    {
        if (address == 0)
            throw std::invalid_argument("nonempty CUDA arrays require device storage");
        cudaPointerAttributes attributes{};
        const auto error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void *>(address));
        if (error != cudaSuccess)
        {
            cudaGetLastError();
            throw std::invalid_argument("CUDA array pointer is not valid device storage");
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != context.device())
            throw std::invalid_argument("CUDA array pointer and requested device do not match");
    }
    if (version == 3 && interface.contains("stream") && !interface["stream"].is_none())
    {
        const auto producer = interface["stream"].cast<std::uintptr_t>();
        if (producer == 0)
            throw std::invalid_argument("CUDA Array Interface producer stream cannot be zero");
        const auto consumer = context.stream() == 0 ? std::uintptr_t{1} : context.stream();
        if (producer != consumer)
        {
            cudaEvent_t ready = nullptr;
            check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            try
            {
                check(cudaEventRecord(ready, reinterpret_cast<cudaStream_t>(producer)));
                check(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(context.stream()), ready));
            }
            catch (...)
            {
                cudaEventDestroy(ready);
                throw;
            }
            check(cudaEventDestroy(ready));
        }
    }
    return {std::move(object),
            reinterpret_cast<void *>(address),
            shape,
            itemsize,
            context.device(),
            typestr};
}

gpu::ConstVectorView vector_view(const Array &array)
{
    return {static_cast<const double *>(array.data), array.shape[0], array.device};
}
gpu::VectorView mutable_view(const Array &array)
{
    return {static_cast<double *>(array.data), array.shape[0], array.device};
}
gpu::DenseMatrixView matrix_view(const Array &array)
{
    return {static_cast<const double *>(array.data), array.shape[0], array.shape[1], array.shape[1],
            array.device};
}
void host_array(const py::array &array, int ndim)
{
    if (!array.dtype().is(py::dtype::of<double>()) || array.ndim() != ndim ||
        !(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
        throw std::invalid_argument("host input must be aligned C-contiguous float64");
}
cg::DenseMatrixView host_matrix(const py::array &array)
{
    host_array(array, 2);
    return {static_cast<const double *>(array.data()), static_cast<std::size_t>(array.shape(0)),
            static_cast<std::size_t>(array.shape(1))};
}
cg::CgOptions options(double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                      std::size_t refresh)
{
    cg::CgOptions value;
    value.rtol = rtol;
    value.atol = atol;
    value.maxiter = maxiter;
    value.residual_refresh = refresh;
    if (algorithm == "standard")
        value.algorithm = cg::Algorithm::recurrence;
    else if (algorithm == "explicit")
        value.algorithm = cg::Algorithm::explicit_residual;
    else
        throw std::invalid_argument("algorithm must be 'standard' or 'explicit'");
    return value;
}
gpu::DenseKernel kernel(const std::string &name)
{
    if (name == "native")
        return gpu::DenseKernel::native;
    if (name == "cublas")
        return gpu::DenseKernel::cublas;
    throw std::invalid_argument("dense_kernel must be 'native' or 'cublas'");
}

struct Workspace
{
    Workspace(std::size_t size, int device, std::uintptr_t stream)
        : context(device, stream), value(size, context)
    {
    }
    gpu::Context context;
    gpu::Workspace value;
};

template <class Matrix>
std::unique_ptr<gpu::JacobiPreconditioner>
preconditioner(const Matrix &A, const gpu::Context &context, const std::string &name)
{
    if (name == "none")
        return {};
    if (name == "jacobi")
        return std::make_unique<gpu::JacobiPreconditioner>(A, context);
    throw std::invalid_argument("preconditioner must be 'none' or 'jacobi'");
}

class Plan
{
  public:
    Plan(py::object matrix, int device, std::uintptr_t stream, const std::string &kernel_name,
         const std::string &preconditioner_name, bool from_host = false)
        : device_(device), kernel_(kernel(kernel_name)), preconditioner_(preconditioner_name)
    {
        const gpu::Context context(device, stream);
        if (from_host && py::isinstance<cg::CsrMatrix>(matrix))
        {
            if (kernel_ != gpu::DenseKernel::native)
                throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
            csr_ = std::make_unique<gpu::CsrMatrix>(matrix.cast<const cg::CsrMatrix &>(), context);
        }
        else if (from_host)
            dense_ =
                std::make_unique<gpu::DenseMatrix>(host_matrix(matrix.cast<py::array>()), context);
        else if (py::isinstance<gpu::CsrMatrix>(matrix))
        {
            if (kernel_ != gpu::DenseKernel::native)
                throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
            csr_ = std::make_unique<gpu::CsrMatrix>(matrix.cast<const gpu::CsrMatrix &>().view(),
                                                    context);
        }
        else if (py::isinstance<gpu::DenseMatrix>(matrix))
            dense_ = std::make_unique<gpu::DenseMatrix>(
                matrix.cast<const gpu::DenseMatrix &>().view(), context);
        else
        {
            const auto input = device_array(matrix, 2, context, false);
            dense_ = std::make_unique<gpu::DenseMatrix>(matrix_view(input), context);
        }
        if (size() == 0 || size() != (dense_ ? dense_->cols() : csr_->cols()))
            throw std::invalid_argument("CG requires a nonempty square matrix");
        const auto start = std::chrono::steady_clock::now();
        M_ = dense_ ? preconditioner(dense_->view(), context, preconditioner_name)
                    : preconditioner(*csr_, context, preconditioner_name);
        if (M_)
            setup_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    }
    std::size_t size() const { return dense_ ? dense_->rows() : csr_->rows(); }
    int device() const { return device_; }
    std::string kernel_name() const
    {
        return kernel_ == gpu::DenseKernel::native ? "native" : "cublas";
    }
    const std::string &preconditioner_name() const { return preconditioner_; }
    std::uint64_t setup_ns() const { return setup_ns_; }
    std::size_t preconditioner_bytes() const { return M_ ? size() * sizeof(double) : 0; }
    cg::SolveInfo solve(py::object b, py::object x, Workspace &workspace, double rtol, double atol,
                        std::size_t maxiter, const std::string &algorithm, bool fused,
                        std::size_t refresh) const
    {
        if (workspace.context.device() != device_)
            throw std::invalid_argument("workspace device does not match prepared solver");
        const auto bv = device_array(b, 1, workspace.context, false);
        const auto xv = device_array(x, 1, workspace.context, true);
        const auto settings = options(rtol, atol, maxiter, algorithm, refresh);
        py::gil_scoped_release release;
        return dense_ ? gpu::solve_into(dense_->view(), vector_view(bv), mutable_view(xv),
                                        workspace.value, workspace.context, settings, fused,
                                        kernel_, M_.get())
                      : gpu::solve_into(*csr_, vector_view(bv), mutable_view(xv), workspace.value,
                                        workspace.context, settings, fused, kernel_, M_.get());
    }

  private:
    int device_;
    gpu::DenseKernel kernel_;
    std::string preconditioner_;
    std::unique_ptr<gpu::DenseMatrix> dense_;
    std::unique_ptr<gpu::CsrMatrix> csr_;
    std::unique_ptr<gpu::JacobiPreconditioner> M_;
    std::uint64_t setup_ns_ = 0;
};

cg::SolveInfo solve(py::object matrix, py::object b, py::object x, Workspace &workspace,
                    double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                    bool fused, std::size_t refresh, const std::string &preconditioner_name,
                    const std::string &kernel_name)
{
    const auto selected = kernel(kernel_name);
    const auto settings = options(rtol, atol, maxiter, algorithm, refresh);
    const auto bv = device_array(b, 1, workspace.context, false);
    const auto xv = device_array(x, 1, workspace.context, true);
    if (py::isinstance<gpu::CsrMatrix>(matrix))
    {
        if (selected != gpu::DenseKernel::native)
            throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
        const auto &A = matrix.cast<const gpu::CsrMatrix &>();
        py::gil_scoped_release release;
        const auto M = preconditioner(A, workspace.context, preconditioner_name);
        return gpu::solve_into(A, vector_view(bv), mutable_view(xv), workspace.value,
                               workspace.context, settings, fused, selected, M.get());
    }
    gpu::DenseMatrixView A;
    Array owner;
    if (py::isinstance<gpu::DenseMatrix>(matrix))
        A = matrix.cast<const gpu::DenseMatrix &>().view();
    else
    {
        owner = device_array(matrix, 2, workspace.context, false);
        A = matrix_view(owner);
    }
    py::gil_scoped_release release;
    const auto M = preconditioner(A, workspace.context, preconditioner_name);
    return gpu::solve_into(A, vector_view(bv), mutable_view(xv), workspace.value, workspace.context,
                           settings, fused, selected, M.get());
}
} // namespace

PYBIND11_MODULE(_cgcuda, m)
{
    py::module_::import("_cgcore"); // SolveInfo is registered once by the CPU module.
    m.doc() = "Optional synchronous CUDA CG with explicit device storage and streams";
    m.def(
        "is_available", [](int device) { return device >= 0 && device < gpu::device_count(); },
        py::arg("device") = 0);
    m.def(
        "synchronize",
        [](int device, std::uintptr_t stream) { gpu::Context(device, stream).synchronize(); },
        py::arg("device") = 0, py::arg("stream") = 0);
    m.def(
        "runtime_info",
        [](int device)
        {
            DeviceScope selected(device);
            cudaDeviceProp property{};
            check(cudaGetDeviceProperties(&property, device));
            std::ostringstream uuid;
            uuid << "GPU-" << std::hex << std::setfill('0');
            for (std::size_t i = 0; i != 16; ++i)
            {
                if (i == 4 || i == 6 || i == 8 || i == 10)
                    uuid << '-';
                uuid << std::setw(2)
                     << static_cast<unsigned>(static_cast<unsigned char>(property.uuid.bytes[i]));
            }
            py::dict result;
            result["device"] = device;
            result["name"] = property.name;
            result["uuid"] = uuid.str();
            result["compute_capability"] = py::make_tuple(property.major, property.minor);
            result["total_memory"] = property.totalGlobalMem;
            result["runtime_version"] = gpu::runtime_version();
            result["driver_api_version"] = gpu::driver_version();
            return result;
        },
        py::arg("device") = 0);
    m.def("build_info",
          []
          {
              py::dict result;
              result["enabled"] = true;
              result["toolkit_version"] = CG_CUDA_TOOLKIT_VERSION;
              result["architectures"] = CG_CUDA_ARCHITECTURES;
              result["compiler"] = CG_CUDA_COMPILER;
              result["host_compiler"] = CG_CUDA_HOST_COMPILER;
              result["build_type"] = CG_CUDA_BUILD_TYPE;
              result["flags"] = CG_CUDA_BUILD_FLAGS;
              result["scalar_dtype"] = "float64";
              return result;
          });
    py::class_<gpu::Vector>(m, "Vector")
        .def(
            py::init([](std::size_t size, int device, std::uintptr_t stream)
                     { return std::make_unique<gpu::Vector>(size, gpu::Context(device, stream)); }),
            py::arg("size"), py::arg("device") = 0, py::arg("stream") = 0)
        .def_property_readonly("size", &gpu::Vector::size)
        .def_property_readonly("device", &gpu::Vector::device)
        .def_property_readonly("__cuda_array_interface__",
                               [](const gpu::Vector &value)
                               {
                                   py::dict result;
                                   result["shape"] = py::make_tuple(value.size());
                                   result["typestr"] = "<f8";
                                   result["data"] = py::make_tuple(
                                       reinterpret_cast<std::uintptr_t>(value.data()), false);
                                   result["strides"] = py::none();
                                   result["stream"] = py::none();
                                   result["version"] = 3;
                                   return result;
                               })
        .def(
            "fill",
            [](gpu::Vector &value, double scalar, std::uintptr_t stream)
            {
                const gpu::Context context(value.device(), stream);
                py::gil_scoped_release release;
                value.fill(scalar, context);
            },
            py::arg("value"), py::arg("stream") = 0)
        .def(
            "copy_from_host",
            [](gpu::Vector &value, const py::array &host, std::uintptr_t stream)
            {
                host_array(host, 1);
                const gpu::Context context(value.device(), stream);
                const cg::ConstVectorView source{static_cast<const double *>(host.data()),
                                                 static_cast<std::size_t>(host.size())};
                py::gil_scoped_release release;
                value.upload(source, context);
            },
            py::arg("source").noconvert(), py::arg("stream") = 0)
        .def(
            "copy_from_device",
            [](gpu::Vector &value, py::object source, std::uintptr_t stream)
            {
                const gpu::Context context(value.device(), stream);
                const auto array = device_array(source, 1, context, false);
                py::gil_scoped_release release;
                value.copy_from_device(vector_view(array), context);
            },
            py::arg("source"), py::arg("stream") = 0)
        .def(
            "to_host",
            [](const gpu::Vector &value, std::uintptr_t stream)
            {
                py::array_t<double> output(value.size());
                const cg::VectorView destination{output.mutable_data(), value.size()};
                const gpu::Context context(value.device(), stream);
                {
                    py::gil_scoped_release release;
                    value.download(destination, context);
                }
                return output;
            },
            py::arg("stream") = 0);
    py::class_<gpu::DenseMatrix>(m, "DenseMatrix")
        .def_static(
            "from_host",
            [](const py::array &host, int device, std::uintptr_t stream)
            {
                const auto source = host_matrix(host);
                const gpu::Context context(device, stream);
                py::gil_scoped_release release;
                auto result = std::make_unique<gpu::DenseMatrix>(source, context);
                return result;
            },
            py::arg("source").noconvert(), py::arg("device") = 0, py::arg("stream") = 0)
        .def_static(
            "from_device",
            [](py::object source, int device, std::uintptr_t stream)
            {
                const gpu::Context context(device, stream);
                const auto array = device_array(source, 2, context, false);
                py::gil_scoped_release release;
                auto result = std::make_unique<gpu::DenseMatrix>(matrix_view(array), context);
                return result;
            },
            py::arg("source"), py::arg("device") = 0, py::arg("stream") = 0)
        .def_property_readonly("shape", [](const gpu::DenseMatrix &value)
                               { return py::make_tuple(value.rows(), value.cols()); })
        .def_property_readonly("device", &gpu::DenseMatrix::device);
    py::class_<gpu::CsrMatrix>(m, "CSRMatrix")
        .def_static(
            "from_host",
            [](const cg::CsrMatrix &host, int device, std::uintptr_t stream)
            {
                const gpu::Context context(device, stream);
                py::gil_scoped_release release;
                auto result = std::make_unique<gpu::CsrMatrix>(host, context);
                return result;
            },
            py::arg("source"), py::arg("device") = 0, py::arg("stream") = 0)
        .def_static(
            "from_device",
            [](py::object indptr, py::object indices, py::object data,
               std::pair<std::size_t, std::size_t> shape, int device, std::uintptr_t stream)
            {
                const gpu::Context context(device, stream);
                const auto pointers = device_array(indptr, 1, context, false, true);
                const auto columns = device_array(indices, 1, context, false, true);
                const auto values = device_array(data, 1, context, false);
                if (pointers.itemsize != columns.itemsize || pointers.shape[0] != shape.first + 1 ||
                    columns.shape[0] != values.shape[0])
                    throw std::invalid_argument(
                        "CUDA CSR index types or array shapes do not match");
                const gpu::CsrMatrixView source{pointers.data,
                                                columns.data,
                                                static_cast<const double *>(values.data),
                                                shape.first,
                                                shape.second,
                                                values.shape[0],
                                                pointers.itemsize == 4 ? gpu::IndexType::int32
                                                                       : gpu::IndexType::int64,
                                                device};
                py::gil_scoped_release release;
                auto result = std::make_unique<gpu::CsrMatrix>(source, context);
                return result;
            },
            py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("shape"),
            py::arg("device") = 0, py::arg("stream") = 0)
        .def_property_readonly("shape", [](const gpu::CsrMatrix &value)
                               { return py::make_tuple(value.rows(), value.cols()); })
        .def_property_readonly("nnz", &gpu::CsrMatrix::nnz)
        .def_property_readonly("device", &gpu::CsrMatrix::device);
    py::class_<Workspace>(m, "Workspace")
        .def(py::init<std::size_t, int, std::uintptr_t>(), py::arg("size"), py::arg("device") = 0,
             py::arg("stream") = 0)
        .def_property_readonly("size", [](const Workspace &value) { return value.value.size(); })
        .def_property_readonly("device",
                               [](const Workspace &value) { return value.context.device(); })
        .def_property_readonly("stream",
                               [](const Workspace &value) { return value.context.stream(); });
    py::class_<Plan>(m, "PreparedSolver")
        .def(py::init<py::object, int, std::uintptr_t, const std::string &, const std::string &>(),
             py::arg("matrix"), py::arg("device") = 0, py::arg("stream") = 0,
             py::arg("dense_kernel") = "native", py::arg("preconditioner") = "none")
        .def_static(
            "from_host",
            [](py::object matrix, int device, std::uintptr_t stream, const std::string &kernel_name,
               const std::string &preconditioner_name)
            {
                return std::make_unique<Plan>(matrix, device, stream, kernel_name,
                                              preconditioner_name, true);
            },
            py::arg("matrix"), py::arg("device") = 0, py::arg("stream") = 0,
            py::arg("dense_kernel") = "native", py::arg("preconditioner") = "none")
        .def_property_readonly("size", &Plan::size)
        .def_property_readonly("device", &Plan::device)
        .def_property_readonly("dense_kernel", &Plan::kernel_name)
        .def_property_readonly("preconditioner", &Plan::preconditioner_name)
        .def_property_readonly("preconditioner_setup_ns", &Plan::setup_ns)
        .def_property_readonly("preconditioner_nbytes", &Plan::preconditioner_bytes)
        .def("solve_into", &Plan::solve, py::arg("b"), py::arg("x"), py::kw_only(),
             py::arg("workspace"), py::arg("rtol") = 1e-8, py::arg("atol") = 0.0,
             py::arg("maxiter") = 1000, py::arg("algorithm") = "standard", py::arg("fused") = true,
             py::arg("residual_refresh") = 50);
    m.def("solve_into", &solve, py::arg("A"), py::arg("b"), py::arg("x"), py::kw_only(),
          py::arg("workspace"), py::arg("rtol") = 1e-8, py::arg("atol") = 0.0,
          py::arg("maxiter") = 1000, py::arg("algorithm") = "standard", py::arg("fused") = true,
          py::arg("residual_refresh") = 50, py::arg("preconditioner") = "none",
          py::arg("dense_kernel") = "native");
}
