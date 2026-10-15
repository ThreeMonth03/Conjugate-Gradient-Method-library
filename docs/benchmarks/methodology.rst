Methodology and reproduction
============================

Run from the repository root after following :doc:`../build`. CPU comparison
dependencies use ``requirements/benchmark.lock``; GPU comparisons use
``requirements/cuda-benchmark.lock`` in the separate CUDA environment.

Suites and candidates
---------------------

.. csv-table::
   :header: "Suite", "Inputs", "Default comparison"
   :widths: 24, 39, 37

   "cpu-dense-f64-v1", "Float64 SPD dense: n=64, 256, 1024, 2048", "Native / NumPy-out / eager Torch"
   "cpu-csr-f64-v1", "2D Poisson CSR: grid=8, 32, 64, 128", "Native / SciPy-out / eager Torch"
   "sparse-solvers", "Scaled 2D/3D Poisson and one neutral case", "Native and official SciPy CG/PCG / SuperLU"
   "sparse-solvers-large", "Unscaled 2D grids 64/128 and 3D grids 20/30", "Same solver choices; opt-in"
   "cuda-dense-f64-v1 / cuda-csr-f64-v1", "The same dense / CSR input fixtures", "Explicit native CUDA / cuBLAS / eager Torch CUDA"

Dense inputs are symmetric and strictly diagonally dominant, with diagonal
range parameter 100. CSR uses the Dirichlet Poisson operator. Scaled sparse
cases use ``A = D P D``, with positive diagonal D spanning 1..20. These
parameters describe generation, not exact condition numbers. Fixtures
have a seeded known solution and zero initial guess; input hashes include the
exact matrix/vector bytes, since a seed alone does not fix every library's
random-number behavior.

Implementation candidates share the selected CG recurrence, residual policy
and preconditioner. ``scipy-out`` is this project's CG loop using public SciPy
CSR matvec and NumPy vector operations; its product allocation/copy is timed.
Torch uses eager operations with preallocated prepared workspace.
The explicit ``native-blas`` candidate uses the private dense BLAS kernel,
with no automatic backend crossover.

The strategy suite instead calls official ``scipy.sparse.linalg.cg`` and
SuperLU ``splu``. Official SciPy has its own recurrence and stopping policy;
its iterations can differ from native CG. Candidate suffixes ``-cg`` / ``-pcg``
select no preconditioner / Jacobi. Direct and iterative answers can have
different accepted errors.

Timing boundaries
------------------

* ``prepared`` reuses the matrix, output, workspace and Jacobi setup.
  Each timed solve restores the initial output.
* ``api`` includes output/workspace allocation, native validation and per-call
  Jacobi setup. Input generation, canonical CSR conversion and library-ready
  matrix construction stay outside both scopes.
* Official SciPy CG allocates iterative workspace in both scopes.
  SuperLU ``total`` factors and solves on every call; ``reuse`` factors before
  timing in either scope. CSR-to-CSC conversion is outside both.
* CUDA starts with device-resident inputs. Completed wall time includes host
  dispatch, scalar synchronization, output reset and a completion fence.
  Uploads and correctness downloads stay outside solve timing. This is
  neither kernel-only timing nor end-to-end host-to-host latency.

Setup probes and stored-array counts are supplementary measurements.
Stored bytes exclude owning snapshots, workspace, allocator and internal
solver storage; they are not peak process or GPU memory.

Accuracy and sampling
---------------------

Default options are standard CG, fused native vector operations, seed
20261015, rtol=1e-8, atol=1e-12, maxiter=2000, and true-residual refresh/restart
every 50 updates. The ``explicit`` algorithm is a reference formulation with
two matvecs per update; it cannot be combined with Jacobi PCG.

Before and after timing, every candidate must converge and independently
satisfy ``||b - A x||2 <= max(atol, rtol * ||b||2)``,
with only a 64-machine-epsilon relative summation margin. A scaled
``math.hypot`` norm guards against overflow/underflow in this check.
Invalid answers are excluded and fail the run; missing optional candidates
remain visible as ineligible. Matvec counts include initial and final
residual evaluations.

Each run starts a fresh process with native thread limits set before numeric
imports. Warmups are untimed. A Williams schedule balances candidate
positions and predecessor pairs; requested rounds are rounded up to a
complete cycle. Each sample times a fixed number of repeated solves.
Median and p5–p95 describe round averages, not independent-run confidence
intervals or individual-call tail latency.

Use the same hardware, affinity and compiler for comparisons, and inspect
actual thread-pool settings. Shared CI checks correctness and collects
artifacts; speed thresholds require a stable measurement host.

Run, summarize and replay
--------------------------

.. code-block:: bash

   make pyprof PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_THREADS=1 \
       PYPROF_SCOPE=prepared PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
       PYPROF_ARGS='--warmups 3'
   PYTHONPATH=build/cpu-release/python .venv/bin/python -m cgbench compare \
       build/cpu-release/benchmarks/cpu-csr-f64-v1-t1-prepared.json
   PYTHONPATH=build/cpu-release/python .venv/bin/python -m cgbench replay \
       build/cpu-release/benchmarks/cpu-csr-f64-v1-t1-prepared.json \
       --output build/benchmarks/replayed-csr.json

JSON records raw nanoseconds, actual schedules, options, input hashes,
residuals/counts, revision, hardware, package versions, threads and the loaded
native binary identity. Keep separate process runs separate. Replay verifies
regenerated inputs and the saved schedule while recording the current
environment; it does not promise identical elapsed times.

Compare two builds
------------------

Build the baseline revision and current checkout separately with matching
compiler, optimization and backend settings. Generate fresh artifacts for
each build with the commands above, alternating their launch order across
several process runs. Compare the new files using the staged package:

.. code-block:: bash

   PYTHONPATH=build/cpu-release/python .venv/bin/python scripts/compare_performance.py \
       --baseline build/benchmarks/baseline-*.json \
       --current build/benchmarks/current-*.json \
       --output build/benchmarks/comparison.json

Its bootstrap resamples artifact blocks and rounds within each block; few
process runs provide limited evidence about process variance.

The paired runner generates fresh fixtures directly from a suite. Each case
starts two fresh interpreters and keeps them for that case's adjacent solves,
so previous cases cannot change its allocator history. It needs neither
previous JSON nor recorded iteration counts.

When comparing native cores across a benchmark-harness change, stage the
same Python harness and wrapper for both extensions. This isolates the core;
it does not measure changes to the Python wrapper. The following example
preserves each extension's real path for build metadata:

.. code-block:: bash

   .venv/bin/python - /absolute/baseline/build/cpu-release/python \
       "$PWD/build/cpu-release/python" <<'PY'
   import pathlib
   import shutil
   import sys

   current = pathlib.Path(sys.argv[2])
   for label, native in zip(("baseline", "current"), sys.argv[1:], strict=True):
       stage = pathlib.Path("build/comparison") / label
       for package in ("cgbench", "cgpy"):
           shutil.copytree(current / package, stage / package, dirs_exist_ok=True,
                           ignore=shutil.ignore_patterns("*.so", "*.pyd", "__pycache__"))
       binary, = pathlib.Path(native).glob("cgpy/_core*.so")
       link = stage / "cgpy" / binary.name
       link.unlink(missing_ok=True)
       link.symlink_to(binary.resolve())
   PY

.. code-block:: bash

   PYTHONPATH=build/cpu-release/python .venv/bin/python scripts/run_paired_performance.py \
       --baseline-pythonpath "$PWD/build/comparison/baseline" \
       --current-pythonpath "$PWD/build/comparison/current" \
       --suite cpu-csr-f64-v1 --scope prepared --implementations native \
       --output build/benchmarks/paired-csr.json

Both builds must match input hashes, solver options and numerical work, and
pass the independent residual gate. The default protocol uses five warmup
pairs and 50 alternating pairs, batching toward 10 ms per sample with a cap
of 1000 solves. It checks the last solve's status and counts in each timed
batch, with residual and input checks before and after sampling.
The 95% interval resamples ten contiguous blocks of five pairs. These
intervals are conditional on the two worker processes and do not establish
variation across process launches or hosts. Per-case coverage is not adjusted
for simultaneous comparisons.

Both comparison scripts use a 5% material-regression budget: an interval
entirely at or below 1.05 supports no material regression, one entirely above
1.05 indicates regression, and an interval crossing 1.05 is inconclusive.
Preserve both fresh-process and paired results when their conclusions differ.
