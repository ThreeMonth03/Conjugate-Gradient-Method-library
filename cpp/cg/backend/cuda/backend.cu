#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>

#include <stdexcept>

namespace cg
{
namespace cuda
{
namespace detail
{
Backend::Backend(ExecutionResources &resources, bool fused, DenseKernel kernel)
    : m_resources(resources), m_fused(fused), m_kernel(kernel)
{
    if (kernel != DenseKernel::native && kernel != DenseKernel::cublas)
    {
        throw std::invalid_argument("unsupported CUDA dense kernel");
    }
    if (resources.kernel != kernel)
    {
        throw std::invalid_argument("CUDA workspace dense kernel does not match selected kernel");
    }
}

} // namespace detail
} // namespace cuda
} // namespace cg
