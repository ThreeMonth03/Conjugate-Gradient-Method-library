# Explicit dense BLAS results

Measured on 2026-10-15 (Asia/Taipei) at source revision
`abdc0424d10d1a9a569481439c824a80bdd207e2`. Every candidate passed the
independent true-residual gate before and after timing: 32 results across four
dense sizes, four implementations and two scopes. Both kernel modes were
selected explicitly; there is no automatic selection or size threshold.

The BLAS candidate had a lower median than NumPy and eager CPU PyTorch in each
measured case. In prepared scope the ratios were 1.01–8.89 for NumPy and
1.06–13.51 for PyTorch; in API scope they were 1.06–7.72 and 1.10–11.24.
Ratios are `reference median / BLAS median`. These are measurements on this
machine, not universal speed
guarantees. The prepared 2048 comparison with NumPy differs by only about 1%;
the timing distributions overlap, so that case does not establish a consistent
performance advantage. The native mode remains slower than the BLAS-backed
references at larger dense sizes.

## Environment and sampling

- Intel Core i7-13700K, WSL2 Linux 6.6.87.2, Python 3.12.3.
- One thread, process affinity restricted to logical CPU 0. This does not
  guarantee a dedicated physical core or isolation from the Windows host.
- Project core: GNU C++ 16.0.1, Release `-O3 -DNDEBUG`, OpenMP disabled.
- Private provider: OpenBLAS 0.3.30, sequential, LP64, locking enabled,
  private symbols; CORE2/HASWELL kernels with the SSE3/SSSE3 baseline.
- NumPy 2.2.6 with OpenBLAS 0.3.29; PyTorch 2.8.0+cpu built with MKL 2024.2.
  Artifacts record actual thread settings and the loaded binary hash.
- Seed 20261015; standard CG, no preconditioner, fused native vector operations,
  `rtol=1e-8`, `atol=1e-12`, `maxiter=2000`, residual refresh every 50 updates.
- Three warmups and five calls per timed round. Thirty requested rounds become
  32 to complete the four-candidate Williams cycle. Results below are median
  milliseconds per call; the raw JSON retains every round and execution order.

All four implementations used the same input bytes and iteration/matvec counts:
49/51 at size 64, 67/70 at size 256, and 70/73 at sizes 1024 and 2048.
Input preparation stays outside timing. Prepared scope reuses operator and
workspace; API scope includes public one-shot validation and output/workspace
allocation. Both scopes reset the same initial guess on every call. See the
[benchmark contract](README.md) for detailed reference and ownership costs.

The recorded Git status contains only the two unrelated, untracked research
files `compare_direct.py` and `cg_vs_direct_results.json`; tracked source was
clean. Those files were preserved and were not used by this benchmark.

## Complete measurements

Prepared scope ([raw artifact](results/2026-10-15-blas/dense-t1-prepared.json)):

| Size | Native ms | BLAS ms | NumPy ms | PyTorch ms |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.034298 | 0.023092 | 0.205357 | 0.312009 |
| 256 | 0.577629 | 0.395360 | 0.545323 | 0.697875 |
| 1024 | 10.526326 | 7.459030 | 8.597074 | 8.692266 |
| 2048 | 87.065250 | 55.457808 | 56.078584 | 58.964700 |

API scope ([raw artifact](results/2026-10-15-blas/dense-t1-api.json)):

| Size | Native ms | BLAS ms | NumPy ms | PyTorch ms |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.043802 | 0.029642 | 0.228767 | 0.333154 |
| 256 | 0.607748 | 0.400338 | 0.660380 | 0.830903 |
| 1024 | 10.056445 | 6.894183 | 7.582507 | 8.053867 |
| 2048 | 86.794532 | 54.096938 | 57.598952 | 59.408369 |

## Reproduce and replay

Install the project-local test and benchmark locks, then run the timing
commands at the recorded revision. Repeat on the same hardware and affinity
for a timing comparison; different machines can produce different rankings.

The report and raw files were committed after the measured source revision.
The replay example uses a checkout containing those artifacts. To replay at
the older revision, copy the raw JSON outside the checkout before switching
revisions and pass that retained absolute path instead.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip \
  -r requirements/test.lock -r requirements/benchmark.lock
taskset -c 0 make pyprof PRESET=cpu-blas \
  PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
  PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
  PYPROF_OUTPUT=build/blas-dense-prepared.json \
  PYPROF_ARGS='--implementations native,native-blas,numpy-out,torch-eager-out'
taskset -c 0 make pyprof PRESET=cpu-blas \
  PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
  PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
  PYPROF_OUTPUT=build/blas-dense-api.json \
  PYPROF_ARGS='--implementations native,native-blas,numpy-out,torch-eager-out'
PYTHONPATH="$PWD/build/cpu-blas/python:$PWD/python" taskset -c 0 \
  .venv/bin/python -m cgbench replay \
  benchmarks/results/2026-10-15-blas/dense-t1-prepared.json \
  --output build/blas-dense-replay.json
```

Replay verifies inputs, eligible candidates, sampling and execution order;
it does not promise identical wall time. Original
[native CPU and sparse CG/PCG/direct measurements](results.md) remain available
with their own source revision and environment.
