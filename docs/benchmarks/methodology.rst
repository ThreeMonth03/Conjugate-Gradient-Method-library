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
