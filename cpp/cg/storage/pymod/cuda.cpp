#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>
#include <cg/python/module.hpp>
#include <cg/storage/cuda/vector.hpp>
#include <cg/storage/detail/validation.hpp>

#include <memory>

namespace cg
{
namespace python
{

void register_cuda_storage(py::module_ &m)
{
    py::class_<gpu::Vector>(m, "Vector")
        .def(py::init([](std::size_t size, int device, std::uintptr_t stream)
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
                                   result["data"] =
                                       py::make_tuple(reinterpret_cast<std::uintptr_t>(value.data()), false);
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
                validate_host_array(host, 1);
                const gpu::Context context(value.device(), stream);
                cg::detail::validate_raw_vector(static_cast<const double *>(host.data()),
                                                static_cast<std::size_t>(host.size()));
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
                value.copy_from_device(as_vector_view(array), context);
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
}

} // namespace python
} // namespace cg
