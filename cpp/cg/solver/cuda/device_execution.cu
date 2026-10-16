#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/backend/cuda/detail/reduction.cuh>
#include <cg/runtime/cuda/detail/launch.hpp>
#include <cg/solver/cuda/detail/device_execution.hpp>
#include <cg/solver/detail/numerical_policy.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cg
{
namespace cuda
{
namespace detail
{

struct DeviceState
{
    IterationPacket packet;
    double rho = 0.0;
    double denominator = 0.0;
    double numerator = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double residual_squared = 0.0;
    double next_rho = 0.0;
    double tolerance = 0.0;
    bool last = false;
    bool refresh = false;
};

namespace
{

constexpr double minimum_squared_scalar = std::numeric_limits<double>::min();

__device__ bool unusable(double value)
{
    return !isfinite(value) || value < minimum_squared_scalar;
}
__device__ void stop(DeviceState *state, Status status)
{
    state->packet.status = status;
    state->packet.decision = IterationDecision::stopped;
}

__global__ void initialize_state(DeviceState *state, double rho, double tolerance, double norm)
{
    *state = DeviceState{};
    state->rho = rho;
    state->tolerance = tolerance;
    state->packet.residual_norm = norm;
}

__global__ void prepare_iteration_alpha(DeviceState *state, bool explicit_residual, bool last, bool refresh)
{
    // Metric kernels only write denominator/numerator, so reset the packet
    // here after they finish on the workspace stream. Keep the prior residual
    // norm for an alpha failure, which advances neither x nor the iteration.
    state->packet.status = Status::max_iterations;
    state->packet.decision = IterationDecision::proceed;
    state->packet.advanced = false;
    state->last = last;
    state->refresh = refresh;
    const double denominator = state->denominator;
    if (!isfinite(denominator) || (explicit_residual && !isfinite(state->numerator)))
    {
        stop(state, Status::non_finite);
        return;
    }
    if (denominator < 0.0)
    {
        stop(state, Status::non_positive_curvature);
        return;
    }
    if (unusable(denominator))
    {
        stop(state, Status::non_finite);
        return;
    }
    state->alpha = (explicit_residual ? -state->numerator : state->rho) / denominator;
    if (!isfinite(state->alpha))
    {
        stop(state, Status::non_finite);
        return;
    }
    // Updates are ordered after this check on the workspace stream. A CUDA
    // launch/execution error throws instead of returning this numerical packet.
    state->packet.advanced = true;
}

__device__ void decide_residual_exit(DeviceState *state, bool explicit_residual)
{
    if (state->packet.decision != IterationDecision::proceed)
    {
        return;
    }
    if (!isfinite(state->packet.residual_norm))
    {
        stop(state, Status::non_finite);
        return;
    }
    const bool candidate = state->packet.residual_norm <= state->tolerance;
    if (explicit_residual && candidate)
    {
        stop(state, Status::converged);
        return;
    }
    if ((!candidate || explicit_residual) && unusable(state->residual_squared))
    {
        stop(state, Status::non_finite);
        return;
    }
    if (explicit_residual)
    {
        if (state->last)
        {
            stop(state, Status::max_iterations);
        }
    }
    else if (candidate || state->refresh || state->last)
    {
        state->packet.decision = IterationDecision::verify_true;
    }
}

__global__ void evaluate_residual_and_compute_beta(DeviceState *state, bool explicit_residual, bool preconditioned)
{
    // Residual exits take priority over an unusable next-rho or beta. The
    // next-rho kernels do not consume this decision and were always submitted
    // before the packet read, including candidate/refresh/last/failure paths.
    decide_residual_exit(state, explicit_residual);
    if (state->packet.decision != IterationDecision::proceed)
    {
        return;
    }
    if (explicit_residual)
    {
        state->beta = state->next_rho / state->denominator;
    }
    else
    {
        const double next = preconditioned ? state->next_rho : state->residual_squared;
        if (unusable(next))
        {
            stop(state, Status::non_finite);
            return;
        }
        state->beta = next / state->rho;
        state->rho = next;
    }
    if (!isfinite(state->beta))
    {
        stop(state, Status::non_finite);
    }
}

__global__ void guarded_axpy(ConstVectorView input, VectorView output, const DeviceState *state, bool subtract)
{
    if (state->packet.decision != IterationDecision::proceed)
    {
        return;
    }
    const double alpha = subtract ? -state->alpha : state->alpha;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < output.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        output.data()[i] = alpha * input.data()[i] + output.data()[i];
    }
}

__global__ void guarded_update(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                               const DeviceState *state, double *partials)
{
    // This condition is uniform for the grid and precedes every barrier/store.
    // Substituting alpha=0 would still corrupt x for nonfinite p (0 * Inf).
    if (state->packet.decision != IterationDecision::proceed)
    {
        if (threadIdx.x == 0)
        {
            partials[blockIdx.x] = 0.0;
        }
        return;
    }
    const double alpha = state->alpha;
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < x.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        x.data()[i] += alpha * p.data()[i];
        const double value = r.data()[i] - alpha * ap.data()[i];
        r.data()[i] = value;
        sum += value * value;
    }
    const double total = block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}

__global__ void guarded_direction(ConstVectorView residual, VectorView p, const DeviceState *state,
                                  double residual_scale)
{
    if (state->packet.decision != IterationDecision::proceed)
    {
        return;
    }
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < p.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        p.data()[i] = residual_scale * residual.data()[i] + state->beta * p.data()[i];
    }
}

IterationPacket read_packet(DeviceExecutionResources &control, const Context &context)
{
    check(cudaMemcpyAsync(&control.host, &control.state.get()->packet, sizeof(IterationPacket), cudaMemcpyDeviceToHost,
                          stream(context)),
          "cudaMemcpyAsync iteration status");
    context.synchronize();
    return control.host;
}

void reset_state(DeviceExecutionResources &control, const Context &context, double rho, double tolerance, double norm)
{
    initialize_state<<<1, 1, 0, stream(context)>>>(control.state.get(), rho, tolerance, norm);
    launch_check();
}

template <class Matrix>
SolveInfo solve_device_impl(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                            const Backend &backend, ExecutionResources &execution, DeviceExecutionResources &control,
                            const CgOptions &options, bool fused, const JacobiPreconditioner *M)
{
    SolveInfo info;
    const Context &context = execution.context;
    const bool explicit_residual = options.algorithm == Algorithm::explicit_residual;
    auto r = workspace.residual();
    auto p = workspace.direction();
    auto ap = workspace.product();
    auto true_r = workspace.scratch();
    auto z = workspace.preconditioned();
    const double b_norm = backend.norm(b);
    double rho = backend.residual_squared_norm(
        A, b, x, r, explicit_residual ? ResidualSign::product_minus_rhs : ResidualSign::rhs_minus_product);
    ++info.matvecs;
    info.residual_norm = backend.norm_from_squared(r, rho);
    const auto initial = cg::detail::initial_residual_decision(b_norm, info.residual_norm, rho, options);
    if (initial.status)
    {
        info.status = *initial.status;
        return info;
    }
    const double tolerance = initial.tolerance;
    const auto finish = [&](Status status)
    {
        const double true_rho = backend.residual_squared_norm(A, b, x, true_r);
        ++info.matvecs;
        info.residual_norm = backend.norm_from_squared(true_r, true_rho);
        info.status = cg::detail::true_residual_status(status, info.residual_norm, tolerance);
        return info;
    };
    if (explicit_residual)
    {
        backend.axpby(-1.0, r, 0.0, r, p);
    }
    else if (M != nullptr)
    {
        rho = backend.multiply_and_dot(M->inverse_diagonal(), r, z);
        if (cg::detail::unusable_squared_scalar(rho))
        {
            return finish(Status::non_finite);
        }
        backend.copy(z, p);
    }
    else
    {
        backend.copy(r, p);
    }
    reset_state(control, context, rho, tolerance, info.residual_norm);
    auto *state = control.state.get();
    for (std::size_t iteration = 0; iteration < options.maxiter; ++iteration)
    {
        const bool last = iteration + 1 == options.maxiter;
        const bool refresh = options.residual_refresh != 0 && (iteration + 1) % options.residual_refresh == 0;
        backend.apply_and_dot_into(A, p, ap, &state->denominator);
        ++info.matvecs;
        if (explicit_residual)
        {
            backend.dot_into(r, p, &state->numerator);
        }
        prepare_iteration_alpha<<<1, 1, 0, stream(context)>>>(state, explicit_residual, last, refresh);
        launch_check();
        if (explicit_residual)
        {
            // cuBLAS GEMV cannot be conditionally skipped by a device flag.
            // Read alpha validity before submitting the true-residual matvec,
            // preserving physical work and counts even on curvature/failure.
            const auto alpha_packet = read_packet(control, context);
            if (alpha_packet.decision == IterationDecision::stopped)
            {
                info.status = alpha_packet.status;
                return info;
            }
            guarded_axpy<<<blocks(x.size()), block_size, 0, stream(context)>>>(p, x, state, false);
            launch_check();
            backend.residual_squared_norm_into(A, b, x, r, &state->residual_squared, ResidualSign::product_minus_rhs);
            ++info.matvecs;
        }
        else if (fused)
        {
            const auto count = blocks(x.size());
            guarded_update<<<count, block_size, 0, stream(context)>>>(x, r, p, ap, state, execution.partials);
            launch_check();
            backend.finish_reduction_into(count, &state->residual_squared);
        }
        else
        {
            guarded_axpy<<<blocks(x.size()), block_size, 0, stream(context)>>>(p, x, state, false);
            launch_check();
            guarded_axpy<<<blocks(r.size()), block_size, 0, stream(context)>>>(ap, r, state, true);
            launch_check();
            backend.dot_into(r, r, &state->residual_squared);
        }
        backend.norm_from_squared_into(r, &state->residual_squared, &state->packet.residual_norm);
        if (explicit_residual && !last)
        {
            backend.dot_into(r, ap, &state->next_rho);
        }
        else if (!explicit_residual && M != nullptr)
        {
            backend.multiply_and_dot_into(M->inverse_diagonal(), r, z, &state->next_rho);
        }
        evaluate_residual_and_compute_beta<<<1, 1, 0, stream(context)>>>(state, explicit_residual, M != nullptr);
        launch_check();
        guarded_direction<<<blocks(p.size()), block_size, 0, stream(context)>>>(M == nullptr ? r : z, p, state,
                                                                                explicit_residual ? -1.0 : 1.0);
        launch_check();
        const auto packet = read_packet(control, context);
        if (packet.advanced)
        {
            ++info.iterations;
        }
        if (explicit_residual)
        {
            info.residual_norm = packet.residual_norm;
            if (packet.decision == IterationDecision::stopped)
            {
                info.status = packet.status;
                return info;
            }
            continue;
        }
        if (packet.decision == IterationDecision::stopped)
        {
            return finish(packet.status);
        }
        if (packet.decision == IterationDecision::verify_true)
        {
            const double true_rho = backend.residual_squared_norm(A, b, x, true_r);
            ++info.matvecs;
            info.residual_norm = backend.norm_from_squared(true_r, true_rho);
            if (const auto status =
                    cg::detail::residual_checkpoint_status(info.residual_norm, true_rho, tolerance, last))
            {
                info.status = *status;
                return info;
            }
            backend.copy(true_r, r);
            if (M != nullptr)
            {
                rho = backend.multiply_and_dot(M->inverse_diagonal(), r, z);
                if (cg::detail::unusable_squared_scalar(rho))
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
            reset_state(control, context, rho, tolerance, info.residual_norm);
        }
    }
    return info;
}

} // namespace

DeviceExecutionResources::DeviceExecutionResources(const Context &context)
    : state(device_allocation<DeviceState>(1, context))
{
}
DeviceExecutionResources::~DeviceExecutionResources() = default;

SolveInfo solve_device(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Backend &backend,
                       ExecutionResources &execution, DeviceExecutionResources &control, const CgOptions &options,
                       bool fused, const JacobiPreconditioner *M)
{
    return solve_device_impl(A, b, x, workspace, backend, execution, control, options, fused, M);
}
SolveInfo solve_device(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                       const Backend &backend, ExecutionResources &execution, DeviceExecutionResources &control,
                       const CgOptions &options, bool fused, const JacobiPreconditioner *M)
{
    return solve_device_impl(A, b, x, workspace, backend, execution, control, options, fused, M);
}

} // namespace detail
} // namespace cuda
} // namespace cg
