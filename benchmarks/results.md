# CPU benchmark results — 2026-10-15

These measurements support a specific CPU advantage for CSR CG and for Jacobi PCG on diagonally scaled Poisson systems. They retain every measured case, including dense losses and faster SuperLU reuse. They compare CPU eager PyTorch, NumPy/SciPy operations, official SciPy CG/PCG, and SuperLU general sparse LU; they do not evaluate `torch.compile`, sparse Cholesky, or every direct solver.

All eight artifacts use source `ccc8ec7a76a569e4e2505251af3cb7d2469eb594`. All 162 candidate/case results passed independent stable true-residual checks before and after timing. The [manifest](results/2026-10-15/manifest.json) records original SHA-256 checksums, binary identities, sizes, scope, threads and affinity. Raw timing files are copied unchanged.

## Method and environment

Hardware: Intel Core i7-13700K, 24 logical CPUs, WSL2/Linux 6.6.87.2. The model comes from a separate [lscpu snapshot](results/2026-10-15/hardware.json) taken on the same host after measurement; the original JSON CPU label remains generic `x86_64`. One-thread runs use affinity `[0]`; four-thread runs use `[0,2,4,6]`. These are WSL2 logical CPU IDs; physical core or P-core isolation is not established by the artifacts. Python 3.12.3; NumPy 2.2.6 with OpenBLAS 0.3.29; SciPy 1.15.3 with its OpenBLAS 0.3.28; Torch 2.8.0+cpu with MKL 2024.2 and AVX2. Native builds use GNU 16.0.1, Release `-O3 -DNDEBUG`, without fast-math; four-thread runs enable OpenMP. Runtime pools match the requested 1/4 threads and Torch interop uses one thread. Each raw artifact records the actual loaded native binary SHA-256 and build metadata.

All times below are **median milliseconds per solve**, obtained from each round's elapsed nanoseconds divided by repetitions. Main runs use 3 warmups, 30 Williams rounds and 5 repetitions: five complete six-row cycles. The large suite uses 1 warmup, 6 rounds and 1 repetition: one complete cycle. Positions and directed within-round predecessor pairs are balanced. Every raw round is retained; `cgbench compare` computes p5/p95 of round averages, which are not individual-call tail latency guarantees or independent-run confidence intervals.

Shared inputs are deterministic float64, seed `20261015`, with a known seeded solution and `x0=0`. Requested tolerance is `max(1e-12, 1e-8 * ||b||₂)`, `maxiter=2000`. Native and implementation references use standard CG, true-residual refresh/restart every 50 iterations, and final verification; native fused kernels are enabled. Eligibility uses an independent scaled norm (`math.hypot`) and permits only a 64-epsilon summation margin. Official SciPy CG/PCG uses its public library recurrence without configured periodic replacement, so strategy comparisons can have different iteration counts. Its counts are instrumented only on the first untimed correctness call; formal timing uses CSR directly without a callback.

`prepared` caches native operator/workspace/Jacobi setup; `api` includes output/workspace allocation and per-call Jacobi construction. Official SciPy allocates its own iterative workspace in both scopes. All calls charge resetting the output from `x0`, including direct candidates, whose mathematics ignores the initial guess. Inputs, canonical CSR/native-wrapper construction, and initial sparse format conversion are prepared outside timing, including API scope. SuperLU `total` factors and solves every timed call; `reuse` factors once outside timing. Both include their final true-residual evaluation; CSR→CSC conversion is excluded and recorded separately. Reuse remains reuse in `api` scope.

## Dense implementation comparison

Ratios are baseline median / native median; values above one favor native. All three candidates share the CG recurrence and residual policy. Every dense/CSR implementation case records identical iterations and matvec counts across candidates. NumPy and Torch use preallocated vector operations; Torch runs CPU eager operations under inference mode.

| threads | n | Native ms | NumPy-out ms | Torch-eager-out ms | NumPy/native | Torch/native | iter/matvec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 0.0401 | 0.2617 | 0.4400 | 6.523× | 10.968× | 49/51 |
| 1 | 256 | 0.6550 | 0.7652 | 1.0610 | 1.168× | 1.620× | 67/70 |
| 1 | 1024 | 11.2226 | 8.8882 | 9.1303 | 0.792× | 0.814× | 70/73 |
| 1 | 2048 | 126.8924 | 80.9526 | 89.9656 | 0.638× | 0.709× | 70/73 |
| 4 | 64 | 0.0617 | 0.2125 | 0.3367 | 3.444× | 5.457× | 49/51 |
| 4 | 256 | 0.3056 | 0.6051 | 0.6555 | 1.980× | 2.145× | 67/70 |
| 4 | 1024 | 4.1187 | 5.3735 | 3.3042 | 1.305× | 0.802× | 70/73 |
| 4 | 2048 | 49.1773 | 37.7599 | 45.1775 | 0.768× | 0.919× | 70/73 |

Native wins at n=64/256 in both thread settings, but loses to both libraries at n=2048. At n=1024 it loses to both at one thread, and beats NumPy while losing to Torch at four threads. Four threads also make the native n=64 case slower than one thread.

## CSR implementation comparison

These cases use the shared two-dimensional Dirichlet Poisson fixture. `scipy-out` means this project's same-policy Python CG loop with public SciPy CSR matvec and NumPy vector updates; it is separate from official `scipy.sparse.linalg.cg` below. Its CSR matvec product allocation/copy is included in timing.

| threads | grid / n | Native ms | SciPy-out ms | Torch-eager-out ms | SciPy/native | Torch/native | iter/matvec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 / 64 | 0.0097 | 0.1543 | 0.2015 | 15.838× | 20.689× | 26/28 |
| 1 | 32 / 1024 | 0.3626 | 1.3117 | 1.3464 | 3.618× | 3.713× | 129/133 |
| 1 | 64 / 4096 | 2.8523 | 6.0579 | 4.9494 | 2.124× | 1.735× | 290/297 |
| 1 | 128 / 16384 | 16.2138 | 31.1988 | 19.5595 | 1.924× | 1.206× | 395/404 |
| 4 | 8 / 64 | 0.0209 | 0.1410 | 0.2229 | 6.741× | 10.661× | 26/28 |
| 4 | 32 / 1024 | 0.4481 | 1.1998 | 1.2720 | 2.678× | 2.839× | 129/133 |
| 4 | 64 / 4096 | 3.8211 | 6.1291 | 35.9296 | 1.604× | 9.403× | 290/297 |
| 4 | 128 / 16384 | 5.9082 | 37.3314 | 17.2837 | 6.319× | 2.925× | 395/404 |

Native beats both references in all eight CSR rows. Its four-thread run is slower than one thread for grids 8/32/64 and faster for grid 128. The table reports the observed whole-solve behavior, including dispatch and thread overhead.

## Sparse solver choices

Scaled cases use `A = D * Poisson * D` with positive diagonal `D` spanning 1..20. This scaling parameter is not an exact condition number. The neutral case uses the constant diagonal Poisson operator. PCG uses a Jacobi inverse diagonal. These tables compare solver choices with different recurrence/setup policies; they are not same-algorithm implementation speedup claims. SuperLU achieves tighter actual residuals than the requested iterative tolerance.

### Default suite, prepared scope (ms)

| case / n | Native CG | Native PCG | SciPy CG | SciPy PCG | LU total | LU reuse |
| --- | --- | --- | --- | --- | --- | --- |
| scaled-poisson-2d-16 / 256 | 0.3855 | 0.0489 | 2.4718 | 0.4837 | 0.3593 | 0.0264 |
| scaled-poisson-2d-32 / 1024 | 3.9723 | 0.3667 | 7.1950 | 1.2077 | 1.5682 | 0.0791 |
| scaled-poisson-3d-6 / 216 | 0.1567 | 0.0257 | 1.4619 | 0.2378 | 0.3693 | 0.0221 |
| scaled-poisson-3d-10 / 1000 | 1.3622 | 0.1411 | 3.3779 | 0.4779 | 3.2956 | 0.1145 |
| poisson-2d-16-neutral / 256 | 0.0369 | 0.0439 | 0.4214 | 0.4309 | 0.2523 | 0.0183 |

### Default suite, API scope (ms)

| case / n | Native CG | Native PCG | SciPy CG | SciPy PCG | LU total | LU reuse |
| --- | --- | --- | --- | --- | --- | --- |
| scaled-poisson-2d-16 / 256 | 0.3671 | 0.0537 | 2.2860 | 0.4497 | 0.2951 | 0.0224 |
| scaled-poisson-2d-32 / 1024 | 3.7211 | 0.3553 | 6.7482 | 1.1249 | 1.4371 | 0.0727 |
| scaled-poisson-3d-6 / 216 | 0.1738 | 0.0363 | 1.5581 | 0.2825 | 0.4047 | 0.0228 |
| scaled-poisson-3d-10 / 1000 | 1.4899 | 0.1651 | 3.6362 | 0.5318 | 3.6259 | 0.1206 |
| poisson-2d-16-neutral / 256 | 0.0464 | 0.0563 | 0.4501 | 0.4929 | 0.2704 | 0.0211 |

For the four scaled cases, native Jacobi PCG reduces iterations 555→48, 1498→115, 252→25 and 444→37, giving 6.10–10.83× lower prepared median time than native CG. The neutral case stays at 50 iterations and PCG adds overhead. SuperLU reuse is faster than native PCG in all five default cases in both scopes; its prior factorization is excluded from those solve times. Prepared and API artifacts are separate process runs, so small reversed timings do not isolate allocation cost.

### Large unscaled suite, prepared scope (ms)

These constant-diagonal cases give Jacobi no iteration benefit. Six single-call rounds provide a bounded larger-scale comparison, with less sampling than the default suite.

| case / n | Native CG | Native PCG | SciPy CG | SciPy PCG | LU total | LU reuse |
| --- | --- | --- | --- | --- | --- | --- |
| poisson-2d-64-large / 4096 | 3.0995 | 3.5854 | 4.0941 | 4.6194 | 6.6440 | 0.5216 |
| poisson-2d-128-large / 16384 | 16.6445 | 19.5010 | 24.4526 | 26.1848 | 34.0955 | 2.7351 |
| poisson-3d-20-large / 8000 | 2.1545 | 2.4136 | 2.6627 | 3.0328 | 163.1439 | 2.7112 |
| poisson-3d-30-large / 27000 | 10.2716 | 11.8192 | 12.2027 | 14.3893 | 2394.1336 | 17.3421 |

SuperLU reuse remains faster on the two-dimensional cases. Native CG is faster than both SuperLU candidates on the two three-dimensional cases in this run. At 3D grid 30, native CG is 10.27 ms, SuperLU total 2394.13 ms, and reuse 17.34 ms; the LU solution has a much tighter residual, so these are not equal-error timings.

### Accuracy and iteration counts

Entries below use the original prepared runs. Default API and independent replay record the same counts. Each iterative column is iterations/matvecs; the residual range covers all four iterative candidates. Direct candidates report zero iterative updates and one final residual matvec. Requested thresholds and actual residuals are shown rather than treating all accepted answers as equally accurate.

| case | Native CG | Native PCG | SciPy CG | SciPy PCG | iterative true residual | LU true residual | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scaled-poisson-2d-16 | 555/568 | 48/50 | 282/283 | 48/49 | 5.819e-05–6.260e-05 | 1.274e-12 | 6.386e-05 |
| scaled-poisson-2d-32 | 1498/1529 | 115/119 | 609/610 | 92/93 | 1.116e-04–1.456e-04 | 3.442e-12 | 1.473e-04 |
| scaled-poisson-3d-6 | 252/259 | 25/27 | 185/186 | 25/26 | 3.946e-05–8.780e-05 | 1.280e-12 | 8.968e-05 |
| scaled-poisson-3d-10 | 444/454 | 37/39 | 304/305 | 37/38 | 2.031e-04–2.094e-04 | 7.183e-12 | 2.149e-04 |
| poisson-2d-16-neutral | 50/52 | 50/52 | 50/51 | 50/51 | 4.829e-07–4.829e-07 | 1.247e-14 | 6.319e-07 |
| poisson-2d-64-large | 290/297 | 290/297 | 172/173 | 172/173 | 2.773e-06–2.775e-06 | 6.679e-14 | 2.810e-06 |
| poisson-2d-128-large | 395/404 | 395/404 | 320/321 | 320/321 | 5.661e-06–5.704e-06 | 1.506e-13 | 5.749e-06 |
| poisson-3d-20-large | 85/88 | 85/88 | 70/71 | 70/71 | 4.714e-06–5.157e-06 | 3.662e-13 | 5.817e-06 |
| poisson-3d-30-large | 116/120 | 116/120 | 100/101 | 100/101 | 1.012e-05–1.038e-05 | 9.516e-13 | 1.063e-05 |

### Setup and stored arrays

These are **single untimed setup/correctness probes**, not distributions of setup latency. The Jacobi phase measures inverse-diagonal construction; operator snapshot/setup is recorded separately in raw JSON. The table uses the reuse candidate's factorization probe and factor arrays. LU-total's factorization remains inside each solve measurement above. CSR→CSC conversion time and CSC arrays are also recorded in each raw artifact.

Shared input bytes count the original NumPy/SciPy A arrays (with their original index dtype), b and x0. They exclude native owning CSR copies, solver workspace and the known solution. Native owning CSR can use wider indices. Factor bytes count exported L/U CSC arrays plus permutation arrays. They exclude internal SuperLU/allocator storage and **are not peak RSS**. These columns describe stored arrays, not complete implementation memory usage.

| case | Jacobi setup ms | inverse bytes | reuse factor setup ms | L/U/permutation bytes | CSC bytes | shared input bytes |
| --- | --- | --- | --- | --- | --- | --- |
| scaled-poisson-2d-16 | 0.001807 | 2048 | 0.291847 | 69504 | 15620 | 19716 |
| scaled-poisson-2d-32 | 0.005689 | 8192 | 1.440815 | 464400 | 64004 | 80388 |
| scaled-poisson-3d-6 | 0.001299 | 1728 | 0.387847 | 125624 | 16420 | 19876 |
| scaled-poisson-3d-10 | 0.004246 | 8000 | 2.850964 | 1509648 | 80804 | 96804 |
| poisson-2d-16-neutral | 0.001173 | 2048 | 0.242061 | 69504 | 15620 | 19716 |
| poisson-2d-64-large | 0.031520 | 32768 | 5.500635 | 2713032 | 259076 | 324612 |
| poisson-2d-128-large | 0.096357 | 131072 | 31.420603 | 14917032 | 1042436 | 1304580 |
| poisson-3d-20-large | 0.034159 | 64000 | 157.403911 | 44726488 | 675204 | 803204 |
| poisson-3d-30-large | 0.098942 | 216000 | 2488.392033 | 310561976 | 2311204 | 2743204 |

## Independent process replay

The saved default prepared spec was replayed in a fresh worker process. Input hashes, candidate set, resolved sampling, actual Williams orders, iterations and matvec counts match the original. The replay's medians below are kept separate from the original samples; process-to-process variation remains visible.

| case / n | Native CG | Native PCG | SciPy CG | SciPy PCG | LU total | LU reuse |
| --- | --- | --- | --- | --- | --- | --- |
| scaled-poisson-2d-16 / 256 | 0.3556 | 0.0431 | 2.2518 | 0.4368 | 0.3223 | 0.0201 |
| scaled-poisson-2d-32 / 1024 | 3.8835 | 0.3557 | 7.0855 | 1.1723 | 1.4719 | 0.0670 |
| scaled-poisson-3d-6 / 216 | 0.1646 | 0.0253 | 1.5536 | 0.2544 | 0.3908 | 0.0204 |
| scaled-poisson-3d-10 / 1000 | 1.4454 | 0.1474 | 3.5785 | 0.5202 | 3.3463 | 0.1086 |
| poisson-2d-16-neutral / 256 | 0.0388 | 0.0450 | 0.4360 | 0.4581 | 0.2662 | 0.0171 |

## Reproduce

Run from the repository root after creating the local `.venv` as documented in [README](../README.md#local-development), then install the benchmark lock below. It includes Torch, which the test lock omits; without it the implementation comparison would have fewer candidates. See [reproducible comparisons](../README.md#reproducible-comparisons). The commands make new artifacts under `build/benchmarks`; compilation/setup is outside the solve timers. They spell out the recorded solver and sampling settings. On another host, choose available affinity IDs and record the changed hardware/environment. These measurements belong to the source revision above; replay checks inputs/scheduling and records the current revision, binary and environment, which should also be inspected.

```bash
# Benchmark dependencies stay inside the project virtual environment/cache.
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock

# Three original one-thread prepared comparisons.
for suite in cpu-dense-f64-v1 cpu-csr-f64-v1 sparse-solvers; do
    taskset -c 0 make pyprof PRESET=cpu-release \
        PYPROF_SUITE="$suite" PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
        PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
        PYPROF_OUTPUT="$PWD/build/benchmarks/reproduced-${suite}-t1-prepared.json" \
        PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'
done

# Default strategy comparison with API allocations/setup charged.
taskset -c 0 make pyprof PRESET=cpu-release \
    PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=api \
    PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
    PYPROF_OUTPUT="$PWD/build/benchmarks/reproduced-sparse-t1-api.json" \
    PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'

# Two original four-thread implementation comparisons.
for suite in cpu-dense-f64-v1 cpu-csr-f64-v1; do
    taskset -c 0,2,4,6 make pyprof PRESET=cpu-openmp \
        PYPROF_SUITE="$suite" PYPROF_THREADS=4 PYPROF_SCOPE=prepared \
        PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
        PYPROF_OUTPUT="$PWD/build/benchmarks/reproduced-${suite}-t4-prepared.json" \
        PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'
done

# Bounded larger sparse comparison.
taskset -c 0 make pyprof PRESET=cpu-release \
    PYPROF_SUITE=sparse-solvers-large PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
    PYPROF_ROUNDS=6 PYPROF_REPETITIONS=1 \
    PYPROF_OUTPUT="$PWD/build/benchmarks/reproduced-sparse-large-t1-prepared.json" \
    PYPROF_ARGS='--warmups 1 --seed 20261015 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'

# Independent replay; preserves the saved candidate set and actual schedule.
taskset -c 0 env PYTHONPATH="$PWD/build/cpu-release/python:$PWD/python" \
    .venv/bin/python -m cgbench replay \
    benchmarks/results/2026-10-15/final-sparse-t1-prepared.json \
    --output build/benchmarks/replayed-sparse-t1.json

# Read-only summaries include p5/p95 from stored round-average samples.
PYTHONPATH="$PWD/python" .venv/bin/python -m cgbench compare \
    benchmarks/results/2026-10-15/final-*.json
```

In the strategy suite, candidate suffixes `-cg` and `-pcg` explicitly select none/Jacobi, overriding the general `--preconditioner` default. See [benchmark contracts](README.md) for adapter scope, fixture generation, replay and correctness details.

## Original raw evidence

| artifact | suite | scope | threads | rounds × repetitions |
| --- | --- | --- | --- | --- |
| [final-csr-t1-prepared.json](results/2026-10-15/final-csr-t1-prepared.json) | cpu-csr-f64-v1 | prepared | 1 | 30 × 5 |
| [final-csr-t4-prepared.json](results/2026-10-15/final-csr-t4-prepared.json) | cpu-csr-f64-v1 | prepared | 4 | 30 × 5 |
| [final-dense-t1-prepared.json](results/2026-10-15/final-dense-t1-prepared.json) | cpu-dense-f64-v1 | prepared | 1 | 30 × 5 |
| [final-dense-t4-prepared.json](results/2026-10-15/final-dense-t4-prepared.json) | cpu-dense-f64-v1 | prepared | 4 | 30 × 5 |
| [final-sparse-large-t1-prepared.json](results/2026-10-15/final-sparse-large-t1-prepared.json) | sparse-solvers-large | prepared | 1 | 6 × 1 |
| [final-sparse-t1-api.json](results/2026-10-15/final-sparse-t1-api.json) | sparse-solvers | api | 1 | 30 × 5 |
| [final-sparse-t1-prepared.json](results/2026-10-15/final-sparse-t1-prepared.json) | sparse-solvers | prepared | 1 | 30 × 5 |
| [final-sparse-t1-replay.json](results/2026-10-15/final-sparse-t1-replay.json) | sparse-solvers | prepared | 1 | 30 × 5 |

The eight original JSON files total 467,088 bytes. Checksums, native binary hashes and hardware provenance are in the [manifest](results/2026-10-15/manifest.json). Raw JSON retains every input hash, nanosecond block, timing order, eligibility result, true residual, setup probe, package version, thread setting and compiler identity.
