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

// Algorithms retain the CPU numerical/refresh policy. The workspace selects
// the device, stream, dense kernel, and host-controlled or device-scalar engine.
// Return is synchronous on workspace.stream(), including scalar/status checks.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CgOptions &options = {}, bool fused = true, const JacobiPreconditioner *M = nullptr);

// One-shot policy: optional Jacobi setup belongs to this call. PreparedSolver
// caches that setup when repeated solves are required.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                     bool fused, PreconditionerKind preconditioner);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CgOptions &options, bool fused, PreconditionerKind preconditioner);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CgOptions &options = {}, bool fused = true, const JacobiPreconditioner *M = nullptr);

} // namespace cuda
} // namespace cg
