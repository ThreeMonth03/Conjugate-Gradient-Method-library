#pragma once

#include <cg/solver/options.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cg
{
namespace detail
{

inline bool unusable_squared_scalar(double value)
{
    return !std::isfinite(value) || value < std::numeric_limits<double>::min();
}

template <class Matrix, class Input, class Output, class Work, class Backend, class Norm>
SolveInfo solve_explicit(const Matrix &A, Input b, Output x, Work &workspace, const Backend &backend,
                         const CgOptions &options, Norm norm_from_squared)
{
    SolveInfo info;
    auto r = workspace.residual();
    auto p = workspace.direction();
    auto ap = workspace.product();
    const double b_norm = backend.norm(b);
    double rho = backend.residual(A, b, x, r, false);
    ++info.matvecs;
    info.residual_norm = norm_from_squared(r, rho);
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
    if (options.maxiter == 0)
    {
        return info;
    }
    if (unusable_squared_scalar(rho))
    {
        info.status = Status::non_finite;
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
        if (denominator < 0.0)
        {
            info.status = Status::non_positive_curvature;
            return info;
        }
        if (unusable_squared_scalar(denominator))
        {
            info.status = Status::non_finite;
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
        info.residual_norm = norm_from_squared(r, rho);
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
        if (unusable_squared_scalar(rho))
        {
            info.status = Status::non_finite;
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

template <class Matrix, class Input, class Output, class Work, class Backend, class Preconditioner, class Norm>
SolveInfo solve_recurrence(const Matrix &A, Input b, Output x, Work &workspace, const Backend &backend,
                           const CgOptions &options, const Preconditioner *M, Norm norm_from_squared)
{
    SolveInfo info;
    auto r = workspace.residual();
    auto p = workspace.direction();
    auto ap = workspace.product();
    auto true_r = workspace.scratch();
    auto z = workspace.preconditioned();
    const double b_norm = backend.norm(b);
    double rho = backend.residual(A, b, x, r);
    ++info.matvecs;
    info.residual_norm = norm_from_squared(r, rho);
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
    if (options.maxiter == 0)
    {
        return info;
    }
    if (unusable_squared_scalar(rho))
    {
        info.status = Status::non_finite;
        return info;
    }
    // Recurrence residuals can drift. Every exit after an iteration reports a
    // freshly computed norm, including breakdown and iteration-limit exits.
    const auto finish = [&](Status status)
    {
        const double true_rho = backend.residual(A, b, x, true_r);
        ++info.matvecs;
        info.residual_norm = norm_from_squared(true_r, true_rho);
        info.status = std::isfinite(info.residual_norm) ? status : Status::non_finite;
        if (status != Status::non_finite && info.residual_norm <= tolerance)
        {
            info.status = Status::converged;
        }
        return info;
    };
    if (M != nullptr)
    {
        rho = M->apply_dot(r, z, backend);
        if (unusable_squared_scalar(rho))
        {
            return finish(Status::non_finite);
        }
        backend.copy(z, p);
    }
    else
    {
        backend.copy(r, p);
    }
    for (std::size_t iteration = 0; iteration < options.maxiter; ++iteration)
    {
        const double denominator = backend.apply_dot(A, p, ap);
        ++info.matvecs;
        if (!std::isfinite(denominator))
        {
            return finish(Status::non_finite);
        }
        if (denominator < 0.0)
        {
            return finish(Status::non_positive_curvature);
        }
        if (unusable_squared_scalar(denominator))
        {
            return finish(Status::non_finite);
        }
        const double alpha = rho / denominator;
        if (!std::isfinite(alpha))
        {
            return finish(Status::non_finite);
        }
        const double residual_squared = backend.update_x_r_norm2(x, r, p, ap, alpha);
        ++info.iterations;
        const double recurrence_norm = norm_from_squared(r, residual_squared);
        if (!std::isfinite(recurrence_norm))
        {
            return finish(Status::non_finite);
        }
        const bool candidate = recurrence_norm <= tolerance;
        if (!candidate && unusable_squared_scalar(residual_squared))
        {
            return finish(Status::non_finite);
        }
        const bool refresh = options.residual_refresh != 0 && info.iterations % options.residual_refresh == 0;
        const bool last_iteration = info.iterations == options.maxiter;
        if (candidate || refresh || last_iteration)
        {
            const double true_rho = backend.residual(A, b, x, true_r);
            ++info.matvecs;
            info.residual_norm = norm_from_squared(true_r, true_rho);
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
            if (unusable_squared_scalar(true_rho))
            {
                info.status = Status::non_finite;
                return info;
            }
            if (last_iteration)
            {
                return info;
            }
            // Replacing the residual also restarts the direction. This avoids
            // carrying conjugacy assumptions across an explicit replacement.
            backend.copy(true_r, r);
            if (M != nullptr)
            {
                rho = M->apply_dot(r, z, backend);
                if (unusable_squared_scalar(rho))
                {
                    info.status = Status::non_finite;
                    return info;
                }
                backend.copy(z, p);
            }
            else
            {
                backend.copy(r, p);
                rho = true_rho;
            }
            ++info.restarts;
            continue;
        }
        const double next_rho = M == nullptr ? residual_squared : M->apply_dot(r, z, backend);
        if (unusable_squared_scalar(next_rho))
        {
            return finish(Status::non_finite);
        }
        const double beta = next_rho / rho;
        if (!std::isfinite(beta))
        {
            return finish(Status::non_finite);
        }
        backend.update_direction(M == nullptr ? r : z, p, beta);
        rho = next_rho;
    }
    // maxiter=0: the initial residual is already the true residual.
    return info;
}

} // namespace detail
} // namespace cg
