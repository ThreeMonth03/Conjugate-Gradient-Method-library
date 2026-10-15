#pragma once

#include <pybind11/pybind11.h>

#include <cg/python/array_conversion.hpp>
#include <pybind11/stl.h>
#include <cg/backend/cuda/context.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
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

void check(cudaError_t error);
class DeviceScope
{
    public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope &) = delete;
    DeviceScope &operator=(const DeviceScope &) = delete;

    private:
    int m_previous = 0;
};

struct DeviceArray
{
    py::object owner;
    void *data;
    std::vector<std::size_t> shape;
    std::size_t itemsize;
    int device;
    std::string typestr;
};

DeviceArray device_array(py::object object, int ndim, const gpu::Context &context, bool writable, bool integer = false);
gpu::ConstVectorView vector_view(const DeviceArray &array);
gpu::VectorView mutable_view(const DeviceArray &array);
gpu::DenseMatrixView matrix_view(const DeviceArray &array);
void host_array(const py::array &array, int ndim);
cg::DenseMatrixView host_matrix(const py::array &array);

} // namespace python
} // namespace cg
