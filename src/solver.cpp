#include "cg/solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cg
{
namespace
{

void validate_solve(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                    const CgOptions &options)
{
    detail::validate(A);
    detail::validate(b);
    detail::validate(x);
    if (A.rows == 0 || A.rows != A.cols || b.size != A.rows || x.size != A.rows)
    {
        throw std::invalid_argument("CG requires a nonempty square matrix and matching vectors");
    }
    if (workspace.size() != A.rows)
    {
        throw std::invalid_argument(
            "prepare the workspace for the matrix dimension before solving");
    }
    if (!std::isfinite(options.rtol) || !std::isfinite(options.atol) || options.rtol < 0.0 ||
        options.atol < 0.0)
    {
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    }
    if (options.algorithm != Algorithm::explicit_residual)
    {
        throw std::invalid_argument("unsupported CG algorithm");
    }
    if (detail::overlaps(b, x) || detail::overlaps(A, x))
    {
        throw std::invalid_argument("CG output must be disjoint from inputs");
    }
    const ConstVectorView vectors[] = {workspace.residual(), workspace.direction(),
                                       workspace.product(), workspace.scratch()};
    for (const auto vector : vectors)
    {
        if (detail::overlaps(vector, x) || detail::overlaps(vector, b) ||
            detail::overlaps(A, vector))
        {
            throw std::invalid_argument("workspace must be disjoint from CG inputs and output");
        }
    }
}

SolveInfo solve_explicit(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                         const CpuBackend &backend, const CgOptions &options)
{
    SolveInfo info;
    auto r = workspace.residual();
    auto p = workspace.direction();
    auto ap = workspace.product();
    const double b_norm = std::sqrt(backend.norm2(b));
    double rho = backend.residual(A, b, x, r, false);
    ++info.matvecs;
    info.residual_norm = std::sqrt(rho);
    if (!std::isfinite(b_norm) || !std::isfinite(info.residual_norm))
    {
        info.status = Status::non_finite;
        return info;
    }
    const double tolerance = std::max(options.atol, options.rtol * b_norm);
    if (info.residual_norm <= tolerance)
    {
        info.status = Status::converged;
        return info;
    }
    backend.axpby(-1.0, r, 0.0, r, p);
    for (std::size_t iteration = 0; iteration < options.maxiter; ++iteration)
    {
        const double denominator = backend.apply_dot(A, p, ap);
        ++info.matvecs;
        const double numerator = backend.dot(r, p);
        if (!std::isfinite(denominator) || !std::isfinite(numerator))
        {
            info.status = Status::non_finite;
            return info;
        }
        if (denominator <= 0.0)
        {
            info.status = Status::non_positive_curvature;
            return info;
        }
        const double alpha = -numerator / denominator;
        if (!std::isfinite(alpha))
        {
            info.status = Status::non_finite;
            return info;
        }
        backend.axpy(alpha, p, x);
        ++info.iterations;
        rho = backend.residual(A, b, x, r, false);
        ++info.matvecs;
        info.residual_norm = std::sqrt(rho);
        if (!std::isfinite(info.residual_norm))
        {
            info.status = Status::non_finite;
            return info;
        }
        if (info.residual_norm <= tolerance)
        {
            info.status = Status::converged;
            return info;
        }
        if (info.iterations == options.maxiter)
        {
            break;
        }
        const double beta = backend.dot(r, ap) / denominator;
        if (!std::isfinite(beta))
        {
            info.status = Status::non_finite;
            return info;
        }
        backend.update_direction(r, p, beta, -1.0);
    }
    return info;
}

} // namespace

const char *status_name(Status status) noexcept
{
    switch (status)
    {
    case Status::converged:
        return "converged";
    case Status::max_iterations:
        return "max_iterations";
    case Status::non_positive_curvature:
        return "non_positive_curvature";
    case Status::non_finite:
        return "non_finite";
    }
    return "unknown";
}

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options)
{
    validate_solve(A, b, x, workspace, options);
    return solve_explicit(A, b, x, workspace, backend, options);
}

} // namespace cg
