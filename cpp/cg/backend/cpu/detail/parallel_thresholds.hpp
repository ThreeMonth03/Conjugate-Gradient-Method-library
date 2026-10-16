#pragma once

#include <cstddef>

namespace cg
{
namespace detail
{

// Existing OpenMP activation heuristics; tuning requires workload measurements.
inline constexpr std::size_t openmp_min_vector_elements = 16'384;
inline constexpr std::size_t openmp_min_dense_rows = 64;
inline constexpr std::size_t openmp_min_dense_columns = 128;
inline constexpr std::size_t openmp_min_csr_rows = 4'096;

} // namespace detail
} // namespace cg
