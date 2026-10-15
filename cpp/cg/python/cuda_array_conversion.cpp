#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>

namespace cg
{
namespace python
{

void check(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error));
    }
}

DeviceScope::DeviceScope(int device)
{
    check(cudaGetDevice(&m_previous));
    check(cudaSetDevice(device));
}
DeviceScope::~DeviceScope()
{
    (void)cudaSetDevice(m_previous);
}

DeviceArray device_array(py::object object, int ndim, const gpu::Context &context, bool writable, bool integer)
{
    if (!py::hasattr(object, "__cuda_array_interface__"))
    {
        throw std::invalid_argument("expected a device array with __cuda_array_interface__");
    }
    const auto interface = object.attr("__cuda_array_interface__").cast<py::dict>();
    const int version = interface["version"].cast<int>();
    if (version != 2 && version != 3)
    {
        throw std::invalid_argument("CUDA Array Interface version must be 2 or 3");
    }
    if (interface.contains("mask") && !interface["mask"].is_none())
    {
        throw std::invalid_argument("masked CUDA arrays are unsupported");
    }
    const auto shape = interface["shape"].cast<std::vector<std::size_t>>();
    if (shape.size() != static_cast<std::size_t>(ndim))
    {
        throw std::invalid_argument("CUDA array dimension mismatch");
    }
    const auto typestr = interface["typestr"].cast<std::string>();
    const bool f64 = typestr == "<f8" || typestr == "=f8";
    const bool i32 = typestr == "<i4" || typestr == "=i4";
    const bool i64 = typestr == "<i8" || typestr == "=i8";
    if (integer ? !(i32 || i64) : !f64)
    {
        throw std::invalid_argument(integer ? "CUDA CSR indices require signed int32 or int64"
                                            : "CUDA arrays require native float64");
    }
    const std::size_t itemsize = i32 ? 4 : 8;
    if (interface.contains("strides") && !interface["strides"].is_none())
    {
        const auto strides = interface["strides"].cast<std::vector<py::ssize_t>>();
        if (strides.size() != shape.size())
        {
            throw std::invalid_argument("CUDA strides must match dimensions");
        }
        std::size_t expected = itemsize;
        for (std::size_t i = shape.size(); i-- != 0;)
        {
            if (shape[i] > 1 && (strides[i] < 0 || static_cast<std::size_t>(strides[i]) != expected))
            {
                throw std::invalid_argument("CUDA arrays must be C-contiguous");
            }
            if (shape[i] != 0 && expected > SIZE_MAX / shape[i])
            {
                throw std::invalid_argument("CUDA shape byte size overflows size_t");
            }
            expected *= shape[i];
        }
    }
    const auto data = interface["data"].cast<py::tuple>();
    if (data.size() != 2 || (writable && data[1].cast<bool>()))
    {
        throw std::invalid_argument("CUDA output must be writable");
    }
    const auto address = data[0].cast<std::uintptr_t>();
    std::size_t elements = 1;
    for (const auto dimension : shape)
    {
        if (dimension != 0 && elements > SIZE_MAX / dimension)
        {
            throw std::invalid_argument("CUDA shape size overflows size_t");
        }
        elements *= dimension;
    }
    if (elements > SIZE_MAX / itemsize || address % itemsize != 0)
    {
        throw std::invalid_argument("CUDA array size or alignment is invalid");
    }
    DeviceScope selected(context.device());
    if (elements != 0)
    {
        if (address == 0)
        {
            throw std::invalid_argument("nonempty CUDA arrays require device storage");
        }
        cudaPointerAttributes attributes{};
        const auto error = cudaPointerGetAttributes(&attributes, reinterpret_cast<void *>(address));
        if (error != cudaSuccess)
        {
            cudaGetLastError();
            throw std::invalid_argument("CUDA array pointer is not valid device storage");
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != context.device())
        {
            throw std::invalid_argument("CUDA array pointer and requested device do not match");
        }
    }
    if (version == 3 && interface.contains("stream") && !interface["stream"].is_none())
    {
        const auto producer = interface["stream"].cast<std::uintptr_t>();
        if (producer == 0)
        {
            throw std::invalid_argument("CUDA Array Interface producer stream cannot be zero");
        }
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
    return {std::move(object), reinterpret_cast<void *>(address), shape, itemsize, context.device(), typestr};
}

gpu::ConstVectorView vector_view(const DeviceArray &array)
{
    return {static_cast<const double *>(array.data), array.shape[0], array.device};
}
gpu::VectorView mutable_view(const DeviceArray &array)
{
    return {static_cast<double *>(array.data), array.shape[0], array.device};
}
gpu::DenseMatrixView matrix_view(const DeviceArray &array)
{
    return {static_cast<const double *>(array.data), array.shape[0], array.shape[1], array.shape[1], array.device};
}
void host_array(const py::array &array, int ndim)
{
    if (!array.dtype().is(py::dtype::of<double>()) || array.ndim() != ndim || !(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
    {
        throw std::invalid_argument("host input must be aligned C-contiguous float64");
    }
}
cg::DenseMatrixView host_matrix(const py::array &array)
{
    host_array(array, 2);
    return {static_cast<const double *>(array.data()), static_cast<std::size_t>(array.shape(0)),
            static_cast<std::size_t>(array.shape(1))};
}

} // namespace python
} // namespace cg
