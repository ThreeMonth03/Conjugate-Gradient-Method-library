#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cg/matrix/csr_matrix.hpp>

#include <cg/python/array_conversion.hpp>
#include <cg/python/module.hpp>
#include <cg/runtime/detail/checked_size.hpp>

#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace cg
{
namespace python
{

namespace
{
template <class Index> std::vector<std::size_t> copy_indices(const py::array &array)
{
    const auto count = static_cast<std::size_t>(array.size());
    (void)cg::detail::checked_product(count, sizeof(Index));
    std::vector<std::size_t> result(count);
    const auto *source = static_cast<const char *>(array.data());
    for (std::size_t i = 0; i < count; ++i)
    {
        // Foreign buffers do not establish a C++ scalar object's type. A byte
        // copy also handles equivalent NumPy integer aliases without aliasing
        // assumptions; the compiler can reduce this fixed-size copy to a load.
        Index value;
        std::memcpy(&value, source + i * sizeof(Index), sizeof(Index));
        if constexpr (std::is_signed_v<Index>)
        {
            if (value < 0)
            {
                throw std::invalid_argument("CSR indices must be nonnegative");
            }
        }
        if (!std::in_range<std::size_t>(value))
        {
            throw py::type_error("CSR indices must fit size_t");
        }
        result[i] = static_cast<std::size_t>(value);
    }
    return result;
}

std::vector<std::size_t> snapshot_indices(const py::array &array)
{
    if (array.ndim() != 1 || !(array.flags() & py::array::c_style))
    {
        throw std::invalid_argument("CSR index buffers must be contiguous and one-dimensional");
    }
    const auto dtype = array.dtype();
    const auto byteorder = dtype.byteorder();
    if ((byteorder == '>' && std::endian::native == std::endian::little) ||
        (byteorder == '<' && std::endian::native == std::endian::big))
    {
        throw std::invalid_argument("CSR index buffers must use native byte order");
    }
    const bool signed_index = dtype.kind() == 'i';
    if (!signed_index && dtype.kind() != 'u')
    {
        throw std::invalid_argument("CSR index arrays must contain integers");
    }
    switch (dtype.itemsize())
    {
    case 1:
        return signed_index ? copy_indices<std::int8_t>(array) : copy_indices<std::uint8_t>(array);
    case 2:
        return signed_index ? copy_indices<std::int16_t>(array) : copy_indices<std::uint16_t>(array);
    case 4:
        return signed_index ? copy_indices<std::int32_t>(array) : copy_indices<std::uint32_t>(array);
    case 8:
        return signed_index ? copy_indices<std::int64_t>(array) : copy_indices<std::uint64_t>(array);
    default:
        throw std::invalid_argument("unsupported CSR integer width");
    }
}

std::vector<double> snapshot_values(const py::array &array)
{
    const auto view = as_const_vector_view(array);
    if (view.empty())
    {
        return {};
    }
    return {view.begin(), view.end()};
}
} // namespace

void register_cpu_matrix(py::module_ &m)
{
    py::class_<cg::CsrMatrix>(m, "CsrMatrix", "Owning canonical CSR storage; construct through cgpy.CsrMatrix.")
        .def(py::init(
                 [](const py::array &indptr, const py::array &indices, const py::array &data,
                    std::pair<std::size_t, std::size_t> shape)
                 {
                     return cg::CsrMatrix(shape.first, shape.second, snapshot_indices(indptr),
                                          snapshot_indices(indices), snapshot_values(data));
                 }),
             py::arg("indptr").noconvert(), py::arg("indices").noconvert(), py::arg("data").noconvert(),
             py::arg("shape"))
        .def_property_readonly(
            "shape", [](const cg::CsrMatrix &A) { return py::make_tuple(A.rows(), A.cols()); },
            "tuple of int: Matrix dimensions (rows, columns).")
        .def_property_readonly("nnz", &cg::CsrMatrix::nnz, "int: Number of stored entries, including stored zeros.");
}

} // namespace python
} // namespace cg
