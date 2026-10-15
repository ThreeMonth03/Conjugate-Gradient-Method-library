#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <cg/matrix/csr_matrix.hpp>

#include <cg/python/module.hpp>

namespace py = pybind11;

namespace cg
{
namespace python
{

void register_cpu_matrix(py::module_ &m)
{
    py::class_<cg::CsrMatrix>(m, "CSRMatrix", "Owning canonical CSR storage; construct through cgpy.CSRMatrix.")
        .def(py::init(
                 [](std::vector<std::size_t> indptr, std::vector<std::size_t> indices, std::vector<double> data,
                    std::pair<std::size_t, std::size_t> shape)
                 {
                     return cg::CsrMatrix(shape.first, shape.second, std::move(indptr), std::move(indices),
                                          std::move(data));
                 }),
             py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("shape"))
        .def_property_readonly(
            "shape", [](const cg::CsrMatrix &A) { return py::make_tuple(A.rows(), A.cols()); },
            "tuple of int: Matrix dimensions (rows, columns).")
        .def_property_readonly("nnz", &cg::CsrMatrix::nnz, "int: Number of stored entries, including stored zeros.");
}

} // namespace python
} // namespace cg
