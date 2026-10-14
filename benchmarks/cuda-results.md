# CUDA benchmark results — 2026-10-15

Prepared device-resident native CUDA had a lower measured median than PyTorch CUDA eager on every fixture here. Dense Jacobi in API scope had a higher median than PyTorch at all four sizes. CSR Jacobi preserves the CG iteration counts and adds work on this constant-diagonal fixture. All eight configurations and their losses are retained below; these results do not establish universal GPU, cuBLAS or framework superiority.

Measured source: **`6d3ebb16a4a0f47732326045c1a0a55782f91cb7`**. This revision precedes the report/artifact commit. All eight formal runs completed before any tracked source modification; only the two existing untracked research files were present. The [manifest](results/2026-10-15-cuda/manifest.json) contains original checksums, complete actual commands, source and binary identities. Ten raw files—eight formal comparisons and two tiny replay validations—were copied unchanged after measurement.

## Hardware and build

GPU: NVIDIA GeForce RTX 2060, compute capability 7.5, device ordinal 0, approximately 6 GiB (6,442,123,264 reported bytes). UUID `GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302`; NVIDIA display-driver release 591.86. The supported driver CUDA API version is separately recorded as 13010; the runtime API version is 12080. Native CUDA compiler/toolkit is 12.8.93, architecture `75`, host GNU 13.3.0, Release with recorded flags `-O3 -DNDEBUG`. The optional CPU BLAS and OpenMP builds are disabled in these artifacts.

Host: Intel Core i7-13700K, WSL2/Linux 6.6.87.2, 24 logical CPUs, Python 3.12.3. Every process is pinned with `taskset -c 0`; library/host thread limits and Torch intraop/interop are one. This affinity names a WSL logical CPU and does not demonstrate physical-core isolation. NumPy 2.2.6/SciPy 1.15.3 construct and independently verify host fixtures. GPU reference: PyTorch 2.8.0+cu128, CUDA 12.8, float64 with TF32 disabled and no mixed precision. Torch deterministic-algorithm enforcement is false; reproducible inputs and observed residuals are recorded, without a bitwise-reproducibility claim.

Loaded CUDA extension SHA-256: `f4e77c0314be523ae6780a1597ff0de6bc6a0f330b98c657b09c577d0d6bb22f`. Loaded CPU extension SHA-256: `67531bd75aca2de079a67e240d51b18caca500cd05f40f301acbc482a2fe7bf6`. Both identities are constant across the formal and replay runs. Each raw artifact also retains actual module paths, runtime/build dictionaries, GPU UUID and physical `nvidia-smi` inventory.

GPU clocks, thermal state and Windows/display activity were not controlled. Some p5–p95 intervals overlap, including dense-2048 prepared Jacobi. Median rankings describe these samples; they do not establish statistical significance or a consistent advantage on repeated independent runs.

## Timing and fairness contract

All values are **median milliseconds per solve**, followed by **[p5–p95]** of per-round average times. Dense uses 30 rounds × 3 solves with 3 warmups per candidate; CSR uses 30 rounds × 1 solve with 3 warmups. Dense's three-candidate Williams design has six rows/five complete cycles; CSR's two-candidate design has two rows/fifteen complete cycles. Positions and directed within-round predecessor pairs are balanced. These are repeated blocks inside one process per configuration, not independent process replications or individual-call tail guarantees.

Timing uses completed wall time: a fence before starting each candidate block, then another fence after its last solve and before stopping the clock. It includes every device x0 reset, native host/control/scalar synchronization, and Python eager control/scalar synchronization. Native calls are synchronous on stream 0; Torch uses its default stream with a device fence. `CUDA_LAUNCH_BLOCKING` is unset, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Compute Sanitizer execution is separate from these measurements.

Both scopes keep input matrices, b and x0 on the GPU. `prepared` caches matrix snapshot, output, workspace and Jacobi. `api` includes fresh native output/workspace resources and per-call Jacobi; Torch API allocates tensors through its caching allocator and retains its framework runtime resources. Matrix upload/snapshot, canonical CSR conversion and eligibility downloads are outside both timing scopes. Single untimed completed setup/upload/Jacobi probes remain in the raw data; they include first-use effects and are not steady-state setup distributions. The API/prepared batches run in separate processes, so subtracting their medians does not isolate a particular allocation phase. No host-transfer or end-to-end latency comparison is claimed.

All candidates receive the same exact float64 inputs: seed `20261015`, x0=0, standard CG, `rtol=1e-8`, `atol=1e-12`, `maxiter=2000`, residual refresh/restart every 50 updates, native fusion enabled. Candidate names explicitly choose native CUDA or dense cuBLAS; no threshold selects the backend. The `torch-cuda-eager` candidate is this project's CG loop built from public PyTorch CUDA operations, rather than an official PyTorch CG solver. It uses the same recurrence and true-residual policy, with eager preallocated prepared operations. cuBLAS is dense only. This compares public solve behavior with eager Torch, rather than `torch.compile`, CUDA Graphs, a GPU direct solver, or kernel-only event timing.

All **80 formal candidate/case results** passed independent stable CPU true-residual checks before and after timing. Every candidate records the same iteration/matvec counts within its case. Exact input hashes are identical across scopes/preconditioners for each fixture. Different floating-point reductions can produce slightly different accepted residuals; equal error or bitwise answers are not assumed.

## Dense results

Dense fixtures are symmetric strictly diagonally dominant matrices with diagonal scale parameter 100; this parameter is not their exact condition number. Ratios are Torch median / named native-or-cuBLAS median: **above one favors the named CUDA candidate**. Every row's iteration/matvec count is shared by all three candidates.

| M | scope | n | Native ms [p5–p95] | cuBLAS ms [p5–p95] | Torch ms [p5–p95] | Torch/native | Torch/cuBLAS | iter/matvec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | prepared | 64 | 6.1315 [5.5288–6.8216] | 6.2148 [5.6948–6.4958] | 11.9994 [11.3500–13.5792] | 1.957× | 1.931× | 49/51 |
| none | prepared | 256 | 8.6112 [8.1405–9.4791] | 8.3784 [7.8764–9.2537] | 16.3782 [14.8247–18.2635] | 1.902× | 1.955× | 67/70 |
| none | prepared | 1024 | 11.3483 [10.7215–12.6136] | 9.9039 [9.4584–10.7193] | 16.4007 [15.5007–17.6991] | 1.445× | 1.656× | 70/73 |
| none | prepared | 2048 | 18.3767 [18.0155–19.3238] | 16.8248 [16.1379–17.7859] | 22.9550 [22.1366–24.2642] | 1.249× | 1.364× | 70/73 |
| none | api | 64 | 8.8116 [8.1640–9.8874] | 8.8940 [8.1536–9.7393] | 15.4370 [14.1193–17.5446] | 1.752× | 1.736× | 49/51 |
| none | api | 256 | 11.6796 [10.8191–14.1263] | 11.4278 [10.4522–13.8319] | 20.7257 [19.8008–22.6405] | 1.775× | 1.814× | 67/70 |
| none | api | 1024 | 14.2387 [13.6514–16.1960] | 12.7405 [12.1063–13.8802] | 20.6343 [19.1393–25.5663] | 1.449× | 1.620× | 70/73 |
| none | api | 2048 | 21.7394 [21.0434–22.5869] | 19.9793 [19.4223–21.8189] | 26.3218 [25.3167–28.4057] | 1.211× | 1.317× | 70/73 |
| jacobi | prepared | 64 | 1.0777 [0.9780–1.5740] | 1.1068 [0.9990–1.5797] | 2.2646 [1.8725–3.0045] | 2.101× | 2.046× | 5/7 |
| jacobi | prepared | 256 | 0.9000 [0.8601–1.3171] | 0.8742 [0.8322–1.4108] | 1.6115 [1.4289–2.2755] | 1.791× | 1.843× | 4/6 |
| jacobi | prepared | 1024 | 1.2437 [1.1691–1.7080] | 1.1016 [1.0324–1.6057] | 1.8063 [1.5024–2.5632] | 1.452× | 1.640× | 4/6 |
| jacobi | prepared | 2048 | 1.5431 [1.3292–1.9549] | 1.3059 [1.2128–2.0588] | 1.7247 [1.5899–2.0442] | 1.118× | 1.321× | 3/5 |
| jacobi | api | 64 | 3.0451 [2.9058–3.6960] | 3.0968 [2.9008–3.8508] | 2.2362 [2.0617–2.9233] | 0.734× | 0.722× | 5/7 |
| jacobi | api | 256 | 2.8852 [2.7037–3.3109] | 2.9572 [2.7253–3.5014] | 1.9897 [1.7795–2.8037] | 0.690× | 0.673× | 4/6 |
| jacobi | api | 1024 | 2.9641 [2.6735–3.4943] | 2.8785 [2.6282–3.4618] | 1.9499 [1.7441–2.6093] | 0.658× | 0.677× | 4/6 |
| jacobi | api | 2048 | 3.1001 [2.9367–3.5919] | 2.9321 [2.8191–3.3170] | 2.1226 [1.8960–2.7302] | 0.685× | 0.724× | 3/5 |

With no preconditioner, native and cuBLAS have lower medians than eager Torch in both scopes at all four sizes. Jacobi reduces updates from 49/67/70/70 to 5/4/4/3. Its prepared median is lower than Torch at every size, but its API median is higher at every size: native 2.89–3.10 ms versus Torch 1.95–2.24 ms. Explicit cuBLAS also has higher medians in those four API Jacobi rows. cuBLAS has slightly higher medians than the custom kernel for several small cases; no automatic crossover rule is inferred.

## CSR results

The shared two-dimensional Dirichlet Poisson grids have diagonal 4. Jacobi is therefore a scalar inverse and does not improve conditioning. It leaves update counts at 26/129/290/395 and adds preconditioner/reduction work. These cases establish behavior for a neutral Jacobi fixture, not its potential benefit on variably scaled sparse systems.

| M | scope | grid / n | Native ms [p5–p95] | Torch ms [p5–p95] | Torch/native | iter/matvec |
| --- | --- | --- | --- | --- | --- | --- |
| none | prepared | 8 / 64 | 3.0453 [2.9106–4.2924] | 5.6538 [5.2622–7.2216] | 1.857× | 26/28 |
| none | prepared | 32 / 1024 | 15.4805 [14.2131–16.5986] | 28.2484 [26.4809–32.3113] | 1.825× | 129/133 |
| none | prepared | 64 / 4096 | 33.4554 [32.7183–37.1609] | 63.7745 [61.4152–68.2877] | 1.906× | 290/297 |
| none | prepared | 128 / 16384 | 51.1318 [46.5387–57.8908] | 92.1287 [86.8644–102.5793] | 1.802× | 395/404 |
| none | api | 8 / 64 | 4.9860 [4.7505–6.8917] | 5.8776 [5.3284–8.8609] | 1.179× | 26/28 |
| none | api | 32 / 1024 | 18.2143 [16.2771–21.1439] | 29.6476 [27.1746–34.0975] | 1.628× | 129/133 |
| none | api | 64 / 4096 | 38.5098 [35.1875–41.5713] | 69.9089 [63.2895–78.7443] | 1.815× | 290/297 |
| none | api | 128 / 16384 | 53.2059 [50.3060–63.8343] | 92.1661 [85.7574–105.3291] | 1.732× | 395/404 |
| jacobi | prepared | 8 / 64 | 4.5916 [4.3385–6.3191] | 8.5638 [7.8588–10.7557] | 1.865× | 26/28 |
| jacobi | prepared | 32 / 1024 | 23.8808 [21.5462–27.8935] | 42.0058 [39.7512–47.6718] | 1.759× | 129/133 |
| jacobi | prepared | 64 / 4096 | 52.7241 [48.8249–66.0076] | 98.3060 [92.4234–110.8748] | 1.865× | 290/297 |
| jacobi | prepared | 128 / 16384 | 79.0034 [73.6615–98.2985] | 145.9772 [130.0647–170.5285] | 1.848× | 395/404 |
| jacobi | api | 8 / 64 | 6.4958 [6.1321–8.3881] | 9.2400 [8.0655–12.8448] | 1.422× | 26/28 |
| jacobi | api | 32 / 1024 | 24.9432 [23.9180–29.3167] | 40.5634 [39.4436–43.5988] | 1.626× | 129/133 |
| jacobi | api | 64 / 4096 | 56.4907 [50.2472–75.9376] | 99.2850 [91.1849–123.5182] | 1.758× | 290/297 |
| jacobi | api | 128 / 16384 | 81.8643 [75.3686–104.2133] | 139.4909 [128.7492–178.4611] | 1.704× | 395/404 |

Native has a lower median than eager Torch in every measured CSR row. Jacobi has higher medians than no preconditioner in both implementations/scopes on these inputs; for grid 128 prepared native time increases from 51.13 to 79.00 ms. Reusable GPU storage helps avoid repeated resource construction, but this data does not show that a GPU is faster than a CPU solve or that Jacobi is universally beneficial.

## Accuracy

The table shows the minimum/maximum independently checked residual among candidates from the prepared run; API has the same counts and corresponding residuals. The requested threshold is `max(atol, rtol*||b||₂)`. CSR's none/Jacobi probes share counts and residuals. Full precision for all scopes remains in raw JSON.

| fixture | M | iter/matvec | true residual range | threshold |
| --- | --- | --- | --- | --- |
| dense-64 | none | 49/51 | 2.044745e-06–2.206060e-06 | 2.737557e-06 |
| dense-256 | none | 67/70 | 5.650814e-06–5.992621e-06 | 6.342306e-06 |
| dense-1024 | none | 70/73 | 9.505000e-06–9.505000e-06 | 1.079068e-05 |
| dense-2048 | none | 70/73 | 1.322930e-05–1.322930e-05 | 1.474103e-05 |
| dense-64 | jacobi | 5/7 | 1.141019e-07–1.141019e-07 | 2.737557e-06 |
| dense-256 | jacobi | 4/6 | 1.098968e-06–1.098968e-06 | 6.342306e-06 |
| dense-1024 | jacobi | 4/6 | 1.722285e-07–1.722285e-07 | 1.079068e-05 |
| dense-2048 | jacobi | 3/5 | 1.245009e-05–1.245009e-05 | 1.474103e-05 |
| poisson-8 | none/jacobi | 26/28 | 2.012273e-07–2.012273e-07 | 3.054924e-07 |
| poisson-32 | none/jacobi | 129/133 | 1.327602e-06–1.327602e-06 | 1.367232e-06 |
| poisson-64 | none/jacobi | 290/297 | 2.772948e-06–2.772948e-06 | 2.810075e-06 |
| poisson-128 | none/jacobi | 395/404 | 5.661370e-06–5.661370e-06 | 5.749348e-06 |

Raw `input_stored_bytes` describe shared host NumPy/SciPy arrays with their original index dtype, b and x0; they exclude owning GPU snapshots/workspace. Preconditioner bytes describe inverse arrays. These are not total GPU allocations, runtime/handle storage or peak process/GPU memory.

## Reproduce the measured revision

Use a fresh checkout, existing compatible NVIDIA driver, Python 3.12 and `/usr/bin/g++-13`. The pinned bootstrap/dependencies stay inside the checkout; no global toolkit or shell changes are required. Another host should choose its own compatible architecture and affinity, preserving the changed metadata. The recorded architecture and commands below target this RTX 2060.

```bash
git clone https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library.git cgpy-cuda-reproduce
cd cgpy-cuda-reproduce
git checkout --detach 6d3ebb16a4a0f47732326045c1a0a55782f91cb7
python3.12 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --cache-dir .cache/pip \
    -r requirements/cuda-benchmark.lock
.venv-cuda/bin/python scripts/setup_cuda.py --prefix .deps/cuda-12.8
make build PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" \
    CUDA_ROOT="$PWD/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    CUDA_ARCHITECTURES=75 JOBS=2

# Eight fixed comparisons; each implementation is selected explicitly.
for storage in dense csr; do
    if [ "$storage" = dense ]; then
        candidates=cuda-native,cuda-cublas,torch-cuda-eager
        repetitions=3
    else
        candidates=cuda-native,torch-cuda-eager
        repetitions=1
    fi
    for preconditioner in none jacobi; do
        for scope in prepared api; do
            taskset -c 0 make pyprof PRESET=cuda-release \
                PYTHON="$PWD/.venv-cuda/bin/python" JOBS=2 \
                PYPROF_SUITE="cuda-${storage}-f64-v1" PYPROF_THREADS=1 \
                PYPROF_SCOPE="$scope" PYPROF_ROUNDS=30 \
                PYPROF_REPETITIONS="$repetitions" \
                PYPROF_OUTPUT="$PWD/build/benchmarks/reproduced-cuda-${storage}-${preconditioner}-${scope}-t1.json" \
                PYPROF_ARGS="--device cuda --cuda-device 0 --implementations $candidates --seed 20261015 --warmups 3 --algorithm standard --fused --preconditioner $preconditioner --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50"
        done
    done
done

# Replay one newly generated artifact with its saved candidates/sampling/order.
taskset -c 0 env PYTHONPATH="$PWD/build/cuda-release/python:$PWD/python" \
    .venv-cuda/bin/python -m cgbench replay \
    build/benchmarks/reproduced-cuda-dense-jacobi-prepared-t1.json \
    --output build/benchmarks/replayed-cuda-dense-jacobi-prepared-t1.json
```

The measured checkout predates these archived files; preserve an archived JSON separately if replaying it after switching revisions. Replay verifies input hashes, candidates, resolved sampling and actual timing orders, and records the current environment/binary identity. Inspect those identities rather than expecting identical elapsed time or binary bytes across different hosts/build paths. From the report checkout, existing measurements can be summarized without GPU execution:

```bash
PYTHONPATH="$PWD/python" .venv-cuda/bin/python -m cgbench compare \
    benchmarks/results/2026-10-15-cuda/cuda-dense-*-t1.json \
    benchmarks/results/2026-10-15-cuda/cuda-csr-*-t1.json
```

## Raw evidence and replay validation

All ten raw files total 428,900 bytes. The [manifest](results/2026-10-15-cuda/manifest.json) records SHA-256/bytes and the exact actual argv for every formal run, including the original build output locations. The tiny smoke replay uses three candidates, 6 rounds × 1 call, zero warmups and the same seed/solver policy. Its two fresh worker processes match source/binary, input hashes, candidates, orders, sampling and iteration/matvec counts. It validates replay plumbing and is not pooled into formal performance statistics.

| artifact | scope | M | rounds × repetitions |
| --- | --- | --- | --- |
| [cuda-csr-jacobi-api-t1.json](results/2026-10-15-cuda/cuda-csr-jacobi-api-t1.json) | api | jacobi | 30 × 1 |
| [cuda-csr-jacobi-prepared-t1.json](results/2026-10-15-cuda/cuda-csr-jacobi-prepared-t1.json) | prepared | jacobi | 30 × 1 |
| [cuda-csr-none-api-t1.json](results/2026-10-15-cuda/cuda-csr-none-api-t1.json) | api | none | 30 × 1 |
| [cuda-csr-none-prepared-t1.json](results/2026-10-15-cuda/cuda-csr-none-prepared-t1.json) | prepared | none | 30 × 1 |
| [cuda-dense-jacobi-api-t1.json](results/2026-10-15-cuda/cuda-dense-jacobi-api-t1.json) | api | jacobi | 30 × 3 |
| [cuda-dense-jacobi-prepared-t1.json](results/2026-10-15-cuda/cuda-dense-jacobi-prepared-t1.json) | prepared | jacobi | 30 × 3 |
| [cuda-dense-none-api-t1.json](results/2026-10-15-cuda/cuda-dense-none-api-t1.json) | api | none | 30 × 3 |
| [cuda-dense-none-prepared-t1.json](results/2026-10-15-cuda/cuda-dense-none-prepared-t1.json) | prepared | none | 30 × 3 |
| [cuda-smoke-replay-input.json](results/2026-10-15-cuda/cuda-smoke-replay-input.json) | tiny replay | none | 6 × 1 |
| [cuda-smoke-replay-output.json](results/2026-10-15-cuda/cuda-smoke-replay-output.json) | tiny replay | none | 6 × 1 |

See [benchmark contracts](README.md) for the shared fixtures/scheduling and [CUDA API documentation](../docs/cuda.rst) for ownership, borrowed streams and device-resident setup boundaries.
