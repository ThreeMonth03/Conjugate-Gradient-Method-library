#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>
#include <cg/python/module.hpp>
#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix.hpp>

#include <memory>

namespace cg
{
namespace python
{

void register_cuda_matrix(py::module_ &m)
{
    py::class_<gpu::DenseMatrix>(m, "DenseMatrix")
        .def_static(
            "from_host",
            [](const py::array &host, int device, std::uintptr_t stream)
            {
                const auto source = as_host_matrix_view(host);
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
                auto result = std::make_unique<gpu::DenseMatrix>(as_dense_matrix_view(array), context);
                return result;
            },
            py::arg("source"), py::arg("device") = 0, py::arg("stream") = 0)
        .def_property_readonly("shape",
                               [](const gpu::DenseMatrix &value) { return py::make_tuple(value.rows(), value.cols()); })
        .def_property_readonly("device", &gpu::DenseMatrix::device);
    py::class_<gpu::CsrMatrix>(m, "CsrMatrix")
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
            [](py::object indptr, py::object indices, py::object data, std::pair<std::size_t, std::size_t> shape,
               int device, std::uintptr_t stream)
            {
                const gpu::Context context(device, stream);
                const auto pointers = device_array(indptr, 1, context, false, true);
                const auto columns = device_array(indices, 1, context, false, true);
                const auto values = device_array(data, 1, context, false);
                if (pointers.itemsize != columns.itemsize || pointers.shape[0] != shape.first + 1 ||
                    columns.shape[0] != values.shape[0])
                {
                    throw std::invalid_argument("CUDA CSR index types or array shapes do not match");
                }
                const gpu::CsrMatrixView source{pointers.data,
                                                columns.data,
                                                static_cast<const double *>(values.data),
                                                shape.first,
                                                shape.second,
                                                values.shape[0],
                                                pointers.itemsize == 4 ? gpu::IndexType::int32 : gpu::IndexType::int64,
                                                device};
                py::gil_scoped_release release;
                auto result = std::make_unique<gpu::CsrMatrix>(source, context);
                return result;
            },
            py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("shape"), py::arg("device") = 0,
            py::arg("stream") = 0)
        .def_property_readonly("shape",
                               [](const gpu::CsrMatrix &value) { return py::make_tuple(value.rows(), value.cols()); })
        .def_property_readonly("nnz", &gpu::CsrMatrix::nnz)
        .def_property_readonly("device", &gpu::CsrMatrix::device);
}

} // namespace python
} // namespace cg
