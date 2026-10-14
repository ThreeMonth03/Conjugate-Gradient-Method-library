#ifndef CG_BACKEND_HPP
#define CG_BACKEND_HPP

#include "cg/storage.hpp"

namespace cg
{

class CsrMatrix;

// All methods are synchronous. Thread settings are local to this context and
// never change OpenMP's process-wide defaults. Inputs must remain alive and
// unmodified by other threads until the operation returns.
class CpuBackend
{
  public:
    explicit CpuBackend(int threads = 1, bool fused = true);
    int threads() const noexcept { return threads_; }
    bool fused() const noexcept { return fused_; }
    static bool openmp_enabled() noexcept;

    void copy(ConstVectorView source, VectorView destination) const;
    double dot(ConstVectorView x, ConstVectorView y) const;
    double norm2(ConstVectorView x) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y,
               VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
    // The residual sign is explicit to preserve the legacy algorithm exactly.
    double residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    void apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const;
    double apply_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const;
    double residual(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    double update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                            double alpha) const;
    void update_direction(ConstVectorView r, VectorView p, double beta,
                          double residual_scale = 1.0) const;

  private:
    int threads_;
    bool fused_;
};

} // namespace cg

#endif
