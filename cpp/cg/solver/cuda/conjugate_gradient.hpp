#pragma once

#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/preconditioner/kind.hpp>
#include <cg/solver/cuda/workspace.hpp>
#include <cg/solver/options.hpp>
#include <cg/solver/solve_info.hpp>

namespace cg
{
namespace cuda
{

class JacobiPreconditioner;

// Algorithms retain the CPU numerical/refresh policy. The explicit workspace
// execution mode selects the host-controlled or device-scalar engine. Return
// is synchronous on context.stream(), including scalar/status checks.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options = {}, bool fused = true, DenseKernel kernel = DenseKernel::native,
                     const JacobiPreconditioner *M = nullptr);

// One-shot policy: optional Jacobi setup belongs to this call. PreparedSolver
// caches that setup when repeated solves are required.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options, bool fused, DenseKernel kernel, PreconditionerKind preconditioner);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options, bool fused, DenseKernel kernel, PreconditionerKind preconditioner);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options = {}, bool fused = true, DenseKernel kernel = DenseKernel::native,
                     const JacobiPreconditioner *M = nullptr);

} // namespace cuda
} // namespace cg
