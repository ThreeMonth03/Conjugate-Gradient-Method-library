#include "cg/solver.hpp"

#include "cg/csr.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cg
{
namespace
{

std::size_t rows(DenseMatrixView A) { return A.rows; }
std::size_t rows(const CsrMatrix &A) { return A.rows(); }
std::size_t cols(DenseMatrixView A) { return A.cols; }
std::size_t cols(const CsrMatrix &A) { return A.cols(); }
void validate_matrix(DenseMatrixView A) { detail::validate(A); }
void validate_matrix(const CsrMatrix &) {} // Validated, immutable owning storage.

template <class Matrix>
void validate_solve(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                    const CgOptions &options)
{
    validate_matrix(A);
    detail::validate(b);
    detail::validate(x);
    if (rows(A) == 0 || rows(A) != cols(A) || b.size != rows(A) || x.size != rows(A))
    {
        throw std::invalid_argument("CG requires a nonempty square matrix and matching vectors");
    }
    if (workspace.size() != rows(A))
    {
        throw std::invalid_argument(
            "prepare the workspace for the matrix dimension before solving");
    }
    if (!std::isfinite(options.rtol) || !std::isfinite(options.atol) || options.rtol < 0.0 ||
        options.atol < 0.0)
    {
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    }
    if (options.algorithm != Algorithm::explicit_residual &&
        options.algorithm != Algorithm::recurrence)
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

template <class Matrix>
SolveInfo solve_explicit(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
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

template <class Matrix>
SolveInfo solve_recurrence(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                           const CpuBackend &backend, const CgOptions &options)
{
    SolveInfo info;
    auto r = workspace.residual();
    auto p = workspace.direction();
    auto ap = workspace.product();
    auto true_r = workspace.scratch();
    const double b_norm = std::sqrt(backend.norm2(b));
    double rho = backend.residual(A, b, x, r);
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
    backend.copy(r, p);
    // Recurrence residuals can drift. Every exit after an iteration reports a
    // freshly computed norm, including breakdown and iteration-limit exits.
    const auto finish = [&](Status status)
    {
        const double true_rho = backend.residual(A, b, x, true_r);
        ++info.matvecs;
        info.residual_norm = std::sqrt(true_rho);
        info.status = std::isfinite(info.residual_norm) ? status : Status::non_finite;
        if (status != Status::non_finite && info.residual_norm <= tolerance)
        {
            info.status = Status::converged;
        }
        return info;
    };
    for (std::size_t iteration = 0; iteration < options.maxiter; ++iteration)
    {
        const double denominator = backend.apply_dot(A, p, ap);
        ++info.matvecs;
        if (!std::isfinite(denominator))
        {
            return finish(Status::non_finite);
        }
        if (denominator <= 0.0)
        {
            return finish(Status::non_positive_curvature);
        }
        const double alpha = rho / denominator;
        if (!std::isfinite(alpha))
        {
            return finish(Status::non_finite);
        }
        const double next_rho = backend.update_x_r_norm2(x, r, p, ap, alpha);
        ++info.iterations;
        if (!std::isfinite(next_rho))
        {
            return finish(Status::non_finite);
        }
        const bool candidate = std::sqrt(next_rho) <= tolerance;
        const bool refresh =
            options.residual_refresh != 0 && info.iterations % options.residual_refresh == 0;
        const bool last_iteration = info.iterations == options.maxiter;
        if (candidate || refresh || last_iteration)
        {
            const double true_rho = backend.residual(A, b, x, true_r);
            ++info.matvecs;
            info.residual_norm = std::sqrt(true_rho);
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
            if (last_iteration)
            {
                return info;
            }
            // Replacing the residual also restarts the direction. This avoids
            // carrying conjugacy assumptions across an explicit replacement.
            backend.copy(true_r, r);
            backend.copy(r, p);
            rho = true_rho;
            ++info.restarts;
            continue;
        }
        const double beta = next_rho / rho;
        if (!std::isfinite(beta))
        {
            return finish(Status::non_finite);
        }
        backend.update_direction(r, p, beta);
        rho = next_rho;
    }
    // maxiter=0: the initial residual is already the true residual.
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
    return options.algorithm == Algorithm::explicit_residual
               ? solve_explicit(A, b, x, workspace, backend, options)
               : solve_recurrence(A, b, x, workspace, backend, options);
}

SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options)
{
    validate_solve(A, b, x, workspace, options);
    return options.algorithm == Algorithm::explicit_residual
               ? solve_explicit(A, b, x, workspace, backend, options)
               : solve_recurrence(A, b, x, workspace, backend, options);
}

} // namespace cg
