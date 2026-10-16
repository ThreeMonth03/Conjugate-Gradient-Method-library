#pragma once

#include <cg/solver/solve_info.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace example
{

inline constexpr std::array<double, 4> matrix_values = {4.0, 1.0, 1.0, 3.0};
inline constexpr std::array<std::array<double, 2>, 2> right_hand_sides = {{{6.0, 7.0}, {-3.0, 2.0}}};

inline std::array<double, 2> exact_solution(std::span<const double> b)
{
    return {(3.0 * b[0] - b[1]) / 11.0, (4.0 * b[1] - b[0]) / 11.0};
}

inline void check_solution(std::span<const double> b, std::span<const double> x, const cg::SolveInfo &info)
{
    const auto expected = exact_solution(b);
    const double residual = std::hypot(b[0] - (4.0 * x[0] + x[1]), b[1] - (x[0] + 3.0 * x[1]));
    const double tolerance = 1e-11 * std::max(1.0, std::hypot(b[0], b[1]));
    if (info.status != cg::Status::converged || !std::isfinite(residual) || residual > tolerance ||
        !std::isfinite(info.residual_norm) || std::abs(info.residual_norm - residual) > tolerance)
    {
        throw std::runtime_error("solve did not converge to the independently checked true residual");
    }
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        if (!std::isfinite(x[i]) || std::abs(x[i] - expected[i]) > tolerance)
        {
            throw std::runtime_error("solve disagrees with the analytical solution");
        }
    }
}

inline void check_initial_counts(const cg::SolveInfo &info, cg::Status status)
{
    if (info.status != status || info.iterations != 0 || info.matvecs != 1 || info.restarts != 0)
    {
        throw std::runtime_error("initial residual check violated the solve counting contract");
    }
}

inline void check_zero_output(std::span<const double> x)
{
    if (std::any_of(x.begin(), x.end(), [](double value) { return value != 0.0; }))
    {
        throw std::runtime_error("zero-iteration solve changed its output");
    }
}

} // namespace example
