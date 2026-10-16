#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>
#include <cg/runtime/cuda/detail/event.hpp>

namespace cg
{
namespace python
{
namespace
{

py::object required_field(const py::dict &interface, const char *name)
{
    if (!interface.contains(name))
    {
        throw std::invalid_argument(std::string("CUDA Array Interface requires '") + name + "'");
    }
    return interface[name];
}

template <typename T> T metadata_integer(py::handle value, const char *name)
{
    if (!PyIndex_Check(value.ptr()))
    {
        throw std::invalid_argument(std::string("CUDA ") + name + " must contain integers");
    }
    // Preserve exceptions raised by user-defined __index__ implementations.
    const auto index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!index)
    {
        throw py::error_already_set();
    }
    try
    {
        return index.cast<T>();
    }
    catch (const py::cast_error &)
    {
        throw std::invalid_argument(std::string("CUDA ") + name + " integer is out of range");
    }
}

py::tuple metadata_tuple(py::handle value, const char *name)
{
    if (!py::isinstance<py::tuple>(value))
    {
        throw std::invalid_argument(std::string("CUDA ") + name + " must be a tuple");
    }
    return py::reinterpret_borrow<py::tuple>(value);
}

} // namespace

DeviceArray device_array(py::object object, int ndim, const gpu::Context &context, bool writable, bool integer)
{
    py::object descriptor;
    try
    {
        descriptor = object.attr("__cuda_array_interface__");
    }
    catch (const py::error_already_set &error)
    {
        if (!error.matches(PyExc_AttributeError))
        {
            throw;
        }
        throw std::invalid_argument("expected a device array with __cuda_array_interface__");
    }
    if (!py::isinstance<py::dict>(descriptor))
    {
        throw std::invalid_argument("CUDA Array Interface must be a dictionary");
    }
    const auto interface = py::reinterpret_borrow<py::dict>(descriptor);
    const int version = metadata_integer<int>(required_field(interface, "version"), "version");
    if (version != 2 && version != 3)
    {
        throw std::invalid_argument("CUDA Array Interface version must be 2 or 3");
    }
    if (interface.contains("mask") && !interface["mask"].is_none())
    {
        throw std::invalid_argument("masked CUDA arrays are unsupported");
    }
    const auto dimensions = metadata_tuple(required_field(interface, "shape"), "shape");
    if (dimensions.size() != static_cast<std::size_t>(ndim))
    {
        throw std::invalid_argument("CUDA array dimension mismatch");
    }
    std::vector<std::size_t> shape;
    shape.reserve(dimensions.size());
    for (const auto dimension : dimensions)
    {
        shape.push_back(metadata_integer<std::size_t>(dimension, "shape"));
    }
    const auto dtype = required_field(interface, "typestr");
    if (!py::isinstance<py::str>(dtype))
    {
        throw std::invalid_argument("CUDA typestr must be a string");
    }
    const auto typestr = dtype.cast<std::string>();
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
        const auto strides = metadata_tuple(interface["strides"], "strides");
        if (strides.size() != shape.size())
        {
            throw std::invalid_argument("CUDA strides must match dimensions");
        }
        std::size_t expected = itemsize;
        for (std::size_t i = shape.size(); i-- != 0;)
        {
            const auto stride = metadata_integer<py::ssize_t>(strides[i], "strides");
            if (shape[i] > 1 && (stride < 0 || static_cast<std::size_t>(stride) != expected))
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
    const auto data = metadata_tuple(required_field(interface, "data"), "data");
    if (data.size() != 2 || !py::isinstance<py::bool_>(data[1]))
    {
        throw std::invalid_argument("CUDA data must contain an address and readonly flag");
    }
    if (writable && data[1].cast<bool>())
    {
        throw std::invalid_argument("CUDA output must be writable");
    }
    const auto address = metadata_integer<std::uintptr_t>(data[0], "data address");
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
    gpu::detail::DeviceGuard selected(context.device());
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
        const auto producer = metadata_integer<std::uintptr_t>(interface["stream"], "stream");
        if (producer == 0)
        {
            throw std::invalid_argument("CUDA Array Interface producer stream cannot be zero");
        }
        const auto consumer = context.stream() == 0 ? std::uintptr_t{1} : context.stream();
        if (producer != consumer)
        {
            const gpu::detail::Event ready(context.device());
            ready.record(reinterpret_cast<cudaStream_t>(producer));
            ready.wait(gpu::detail::stream(context));
        }
    }
    return {std::move(object), reinterpret_cast<void *>(address), shape, itemsize, context.device()};
}

gpu::ConstVectorView as_const_vector_view(const DeviceArray &array)
{
    return {static_cast<const double *>(array.data), array.shape[0], array.device};
}
gpu::VectorView as_mutable_vector_view(const DeviceArray &array)
{
    return {static_cast<double *>(array.data), array.shape[0], array.device};
}
gpu::DenseMatrixView as_dense_matrix_view(const DeviceArray &array)
{
    return {static_cast<const double *>(array.data), array.shape[0], array.shape[1], array.shape[1], array.device};
}
} // namespace python
} // namespace cg
