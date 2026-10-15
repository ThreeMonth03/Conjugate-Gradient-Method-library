#pragma once

#include <cg/backend/cuda/backend.hpp>
#include <cg/solver/options.hpp>

namespace cg
{
namespace cuda
{

class JacobiPreconditioner;

// Both algorithms and their numerical/refresh policy are shared with the CPU
// solver. Return is synchronous on context.stream(), including scalar checks.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options = {}, bool fused = true, DenseKernel kernel = DenseKernel::native,
                     const JacobiPreconditioner *M = nullptr);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options = {}, bool fused = true, DenseKernel kernel = DenseKernel::native,
                     const JacobiPreconditioner *M = nullptr);

} // namespace cuda
} // namespace cg
