#include "cg/backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace cg {
namespace {

void same_size(ConstVectorView x, ConstVectorView y) {
    detail::validate(x);
    detail::validate(y);
    if (x.size != y.size) {
        throw std::invalid_argument("vector dimensions do not match");
    }
}

void no_overlap(ConstVectorView x, ConstVectorView y) {
    if (detail::overlaps(x, y)) {
        throw std::invalid_argument("operation requires disjoint storage");
    }
}

void exact_or_disjoint(ConstVectorView input, ConstVectorView output) {
    if (input.data != output.data) {
        no_overlap(input, output);
    }
}

void validate_apply(DenseMatrixView A, ConstVectorView x, VectorView output) {
    detail::validate(A);
    detail::validate(x);
    detail::validate(output);
    if (A.cols != x.size || A.rows != output.size) {
        throw std::invalid_argument("matrix-vector dimensions do not match");
    }
    no_overlap(x, output);
    if (detail::overlaps(A, output)) {
        throw std::invalid_argument("matrix and output storage overlap");
    }
}

} // namespace

CpuBackend::CpuBackend(int threads, bool fused) : threads_(threads), fused_(fused) {
    if (threads < 1) {
        throw std::invalid_argument("threads must be a positive integer");
    }
    if (threads > 1 && !openmp_enabled()) {
        throw std::invalid_argument("multiple threads require an OpenMP build");
    }
}

bool CpuBackend::openmp_enabled() noexcept {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

void CpuBackend::copy(ConstVectorView source, VectorView destination) const {
    same_size(source, destination);
    exact_or_disjoint(source, destination);
    if (source.data != destination.data && source.size != 0) {
        std::copy_n(source.data, source.size, destination.data);
    }
}

double CpuBackend::dot(ConstVectorView x, ConstVectorView y) const {
    same_size(x, y);
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : sum) schedule(static) \
    num_threads(threads_) if (threads_ > 1 && x.size >= 16384)
#endif
    for (std::size_t i = 0; i < x.size; ++i) {
        sum += x.data[i] * y.data[i];
    }
    return sum;
}

double CpuBackend::norm2(ConstVectorView x) const { return dot(x, x); }

void CpuBackend::axpy(double alpha, ConstVectorView x, VectorView y) const {
    same_size(x, y);
    exact_or_disjoint(x, y);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(threads_) \
    if (threads_ > 1 && x.size >= 16384)
#endif
    for (std::size_t i = 0; i < x.size; ++i) {
        y.data[i] += alpha * x.data[i];
    }
}

void CpuBackend::axpby(double alpha, ConstVectorView x, double beta,
                       ConstVectorView y, VectorView output) const {
    same_size(x, y);
    same_size(x, output);
    exact_or_disjoint(x, output);
    exact_or_disjoint(y, output);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(threads_) \
    if (threads_ > 1 && x.size >= 16384)
#endif
    for (std::size_t i = 0; i < x.size; ++i) {
        output.data[i] = alpha * x.data[i] + beta * y.data[i];
    }
}

void CpuBackend::apply(DenseMatrixView A, ConstVectorView x, VectorView output) const {
    validate_apply(A, x, output);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) \
    if (threads_ > 1 && A.rows >= 64 && A.cols >= 128)
#endif
    for (std::size_t i = 0; i < A.rows; ++i) {
        const auto* row = A.data + i * A.row_stride;
        double sum = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : sum)
#endif
        for (std::size_t j = 0; j < A.cols; ++j) {
            sum += row[j] * x.data[j];
        }
        output.data[i] = sum;
    }
}

double CpuBackend::apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const {
    apply(A, p, ap);
    return dot(p, ap);
}

double CpuBackend::residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x,
                            VectorView r, bool b_minus_ax) const {
    same_size(b, r);
    no_overlap(b, r);
    apply(A, x, r);
    axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
    return norm2(r);
}

double CpuBackend::update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p,
                                   ConstVectorView ap, double alpha) const {
    same_size(x, r);
    same_size(x, p);
    same_size(x, ap);
    no_overlap(x, r);
    no_overlap(x, p);
    no_overlap(x, ap);
    no_overlap(r, p);
    no_overlap(r, ap);
    axpy(alpha, p, x);
    axpy(-alpha, ap, r);
    return norm2(r);
}

void CpuBackend::update_direction(ConstVectorView r, VectorView p, double beta,
                                  double residual_scale) const {
    same_size(r, p);
    no_overlap(r, p);
    axpby(residual_scale, r, beta, p, p);
}

} // namespace cg
