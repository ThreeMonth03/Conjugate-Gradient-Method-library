#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{
Backend::Backend(ExecutionResources &resources, bool fused) : m_resources(resources), m_fused(fused) {}

} // namespace detail
} // namespace cuda
} // namespace cg
