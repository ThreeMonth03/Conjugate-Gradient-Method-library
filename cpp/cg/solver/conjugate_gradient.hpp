#pragma once

#include <cg/backend/cpu/backend.hpp>
#include <cg/preconditioner/kind.hpp>
#include <cg/solver/options.hpp>
#include <cg/solver/solve_info.hpp>
#include <cg/solver/workspace.hpp>

namespace cg
{

class JacobiPreconditioner;

// x is an initial guess and output. The output and workspace must be disjoint
// from A and b. The workspace must match A.rows(); prepare it outside timing.
// Inputs are validated once, and no vector storage is allocated by the solve.
// M is an optional prepared Jacobi preconditioner; it requires recurrence CG.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options = {}, const JacobiPreconditioner *M = nullptr);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options = {}, const JacobiPreconditioner *M = nullptr);

// One-shot setup keeps the matrix borrowed and creates only the selected
// preconditioner. PreparedSolver owns reusable setup for repeated solves.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options, PreconditionerKind preconditioner);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options, PreconditionerKind preconditioner);

} // namespace cg
