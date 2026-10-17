Methodology and reproduction
============================

Run from the repository root after following :doc:`../build`. CPU comparison
dependencies use ``requirements/benchmark.lock``; GPU comparisons use
``requirements/cuda-benchmark.lock`` in the separate CUDA environment.

Run the comparison suite after installing its dependencies:

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
   make pyprof PYPROF_SUITE=sparse-solvers

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
  matrix construction stay outside solve timing.
* CUDA-only ``session`` reuses iteration vectors and execution resources,
  allocating a fresh output and optional Jacobi inverse on every call for native
  and eager Torch. It does not snapshot/cache the matrix or preconditioner.
  Output reset and per-call setup are timed; session construction/destruction
  stay outside sampling. Torch retains its documented caching allocator.
  Existing ``api``/``prepared`` specifications keep their meanings.
* Official SciPy CG allocates iterative workspace in both scopes.
  SuperLU ``total`` factors and solves on every call; ``reuse`` factors before
  timing in either scope. CSR-to-CSC conversion is outside both.
* CUDA starts with device-resident inputs. Completed wall time includes host
  dispatch, scalar synchronization, output reset and a completion fence.
  Uploads and correctness downloads stay outside solve timing. This is
  neither kernel-only timing nor end-to-end host-to-host latency.

CUDA defaults to native ``execution_mode="host"``. Select
``--cuda-execution-mode device`` explicitly to benchmark native device scalar
control; this adds ``solver.execution_mode="device"`` to the artifact. Default
host specs retain their existing schema and replay behavior. The flag selects
only native CUDA execution: the eager Torch reference keeps its own scalar
control and records that the native mode does not apply to it. Both modes use
the same input fixtures, numerical options, reset and completion boundaries.
Device status packets, residual checks and final completion remain timed.
Keep distinct mode artifacts and inspect their operation counts before
interpreting a latency difference.

Scopes describe different allocation/setup contracts. Keep their artifacts
separate: the strict comparator rejects mismatched scopes, even if operation
counts match. Session reuse is not an improvement to the fresh API contract.
CPU candidates reject session scope. No hidden pool or automatic selection
is introduced.

For a session comparison on resident inputs:

.. code-block:: bash

   make pyprof PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" \
       PYPROF_SUITE=smoke-cuda-dense PYPROF_SCOPE=session \
       PYPROF_THREADS=1 PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_ARGS='--device cuda --cuda-execution-mode device --implementations cuda-native,cuda-cublas,torch-cuda-eager --preconditioner jacobi --warmups 3'

Use ``PYPROF_SCOPE=api`` or ``prepared`` in separate artifacts to compare the
other contracts. For CSR, select ``cuda-native,torch-cuda-eager``.

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

New native and same-policy reference results also record ``restarts``: the
number of successful search-direction replacements after a true-residual
checkpoint. Initial direction setup is not a restart. Convergence, iteration
limits or numerical failure before direction replacement do not increment
the count; the explicit algorithm reports zero. Counts reset on every solve,
including prepared and Session reuse. Before/after gates require matching
iterations, matvecs and restart counts, and profiler JSON records all three.

Official SciPy Krylov solvers and direct LU report ``restarts=null`` because
this native restart definition is unavailable; summaries show ``n/a``.
Schema-1 artifacts without the optional field remain readable and are never
rewritten or assigned a fabricated zero. Strict block and paired comparisons
reject different known counts or known/unknown mixtures. When both sides are
unknown, a legacy comparison remains available with
``restart_parity_verified=false`` and an explicit limitation: matching
iterations/matvecs alone does not establish full numerical-work parity.

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

For a fixed eight-process-block comparison with complementary ABBA/BAAB
orders, immutable plans and boundary environment records, see
:doc:`campaigns`. It launches the same public benchmark commands after builds
finish and preserves the original 5% gate and all framework controls.

.. toctree::
   :hidden:

   campaigns

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

.. _benchmark-diagnostic-profiling:

Diagnostic profiling
--------------------

``make pyprof`` remains the comparison entry point: it launches the shared
``cgbench`` harness with independently verified solutions, multiple candidates
and recorded sampling schedules. Use ``scripts/profile_solve.py`` to inspect
one implementation and one exact suite case in a fresh process. It reuses the
shared CLI, fixtures, executors and true-residual gates; its samples are
diagnostics, not framework comparison scores.

Build the selected preset before profiling, then keep its package path and
environment fixed. For a native CPU solve:

.. code-block:: bash

   taskset -c 0 env PYTHONPATH="$PWD/build/cpu-release/python" \
       .venv/bin/python scripts/profile_solve.py \
       --suite cpu-dense-f64-v1 --case dense-1024 --implementation native \
       --scope prepared --probe solve --threads 1 --warmups 3 --repetitions 10 \
       --output build/profiles/cpu-native-prepared-dense1024.json

``--probe solve`` times the existing adapter's completed call, including its
initial-guess reset. API scope includes per-call result/workspace/preconditioner
setup and lifecycle; prepared scope reuses the resources defined by the
adapter. Native ``--probe workspace`` and ``--probe prepare`` instead time
public resource construction **and destruction**, including completion fences.
The prepare probe snapshots the operator and sets up the selected
preconditioner. They are separate observations: do not subtract independently
sampled medians to attribute a solve's internal costs.

For CUDA, use the project-local CUDA environment and an existing Nsight Systems
installation. ``--cuda-profiler`` brackets only the repeated diagnostic calls;
the script loads the public NVIDIA runtime from that virtual environment and
adds an NVTX range per call. For example:

.. code-block:: bash

   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
       --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
       --output build/profiles/cublas-jacobi-api-dense1024 \
       taskset -c 0 env PYTHONPATH="$PWD/build/cuda-release/python" \
       .venv-cuda/bin/python scripts/profile_solve.py \
       --suite cuda-dense-f64-v1 --case dense-1024 \
       --implementation cuda-cublas --device cuda --cuda-device 0 \
       --scope api --probe solve --preconditioner jacobi --threads 1 \
       --warmups 3 --repetitions 10 --cuda-profiler \
       --output build/profiles/cublas-jacobi-api-dense1024.json

Change the probe to ``workspace`` or ``prepare`` for a separate resource
capture. Fixture creation, input uploads, independent correctness downloads,
warmups, metadata collection, NVTX push/pop and profiler start/stop are outside
the recorded wall interval. Profiling adds overhead, and CUDA API durations
include host waiting; API and kernel totals are not additive components of
unprofiled solve latency.

JSON records raw completed-wall samples, input hashes, solver options,
before/after correctness gates, adapter setup and environment metadata. Its
``metric`` is ``diagnostic_phase_completed_wall`` and ``framework_score`` is
false. For a frozen staged package, optionally pass its ``identity.json`` using
``--stage-identity``; loaded module hashes are checked before and after capture.
Keep raw JSON and profiler output under ``build/`` and retain unsuccessful
captures when reporting evidence.

When investigating host readback, inspect allocation, transfer and completion
calls alongside kernels. ``cudaMemcpyAsync`` can block for pageable host
memory; its name alone does not establish overlap. Preserve stream completion
and storage lifetime when comparing readback strategies, as described in the
`NVIDIA CUDA 12.8.1 synchronization documentation
<https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/api-sync-behavior.html>`_.
