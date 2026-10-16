#pragma once

#include <pybind11/pybind11.h>

#include <cg/python/array_conversion.hpp>
#include <pybind11/stl.h>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/runtime/cuda/context.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/storage/cuda/vector_view.hpp>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu = cg::cuda;

namespace cg
{
namespace python
{

inline void check(cudaError_t error)
{
    gpu::detail::check(error, "CUDA");
}

struct DeviceArray
{
    py::object owner;
    void *data;
    std::vector<std::size_t> shape;
    std::size_t itemsize;
    int device;
};

DeviceArray device_array(py::object object, int ndim, const gpu::Context &context, bool writable, bool integer = false);
gpu::ConstVectorView as_const_vector_view(const DeviceArray &array);
gpu::VectorView as_mutable_vector_view(const DeviceArray &array);
gpu::DenseMatrixView as_dense_matrix_view(const DeviceArray &array);

} // namespace python
} // namespace cg
