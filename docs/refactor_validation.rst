Refactor validation
===================

Recorded on 2026-10-15 on macOS ARM64, using CPython 3.12.9, AppleClang 21
and NumPy 2.2.6. Native CPU comparisons use Release ``-O3 -DNDEBUG`` builds,
one thread and the same compiler, interpreter, inputs and solver options.

The baseline is an untouched archive of revision
``fa99ef888606d5f34e2a38660b525f3883b4be4b``. The refactored core/source fingerprint is
``fdfef79c3b040e1c1c1eadde2b7125a43e4a22689ee2626a0a79c1eec81ba368``. Native binary hashes and full environment metadata
are recorded in the measurement artifacts.

Correctness and build checks
----------------------------

* ``make test``: 157 Python tests and 51 native tests passed. Another 78
  Python cases and 13 native cases were skipped for optional capabilities.
* ``make gtest PRESET=cpu-asan``: all 51 available native cases passed with
  AddressSanitizer and UndefinedBehaviorSanitizer; 13 optional cases skipped.
* ``make lint``: Ruff lint/format and C++/CUDA clang-format checks passed.
  The native-only clang-tidy build also passed.
* ``make docs doctest demo``: documentation builds with warnings as errors;
  54 doctests and the dense, CSR and prepared-solve examples passed.
* ``make test-dist``: an sdist was relocated, built in PEP 517 isolation,
  installed outside the repository and exercised using only declared runtime
  dependencies. Retired modules were absent. Native dense/CSR CG/PCG and
  prepared solves passed the installed-wheel checks.
* The built wheel also passed installed dense/CSR/prepared checks with
  NumPy 1.26.0, the declared runtime floor for Python 3.12. The isolated
  distribution check used NumPy 2.2.6.
* A native-only build with Python package discovery disabled was installed
  into a separate prefix. An independent C++ consumer compiled against the
  installed public headers/archive and passed a true-residual check.
* All 23 public CPU/CUDA headers compiled independently without Python or
  CUDA SDK includes. CUDA-specific binding/support units and non-kernel
  sources passed host syntax checks against official CUDA 12.8 headers.
  The CUDA kernel backend passed Clang 22 host/device frontend checks for
  ``sm_75``. This does not verify CUDA linking or device execution.

Numerical tests retain independent residual and known-solution checks,
convergence/failure statuses, aliasing, snapshots, workspace reuse and
floating-point range behavior. Explicit-residual tests select that reference
algorithm explicitly; C++ and Python now default to standard recurrence CG.
A new ownership test checks independent DenseMatrix copies, moved-from empty
views and self-move safety.

Performance protocol and evidence
---------------------------------

The unchanged material-regression budget is 5%. Timing data are accepted only
when input hashes, convergence, iteration/matvec counts and independent true
residual checks agree.

First, fresh workers ran four baseline and four current blocks in the order
``A B B A B A A B``. Every block used 5 warmups, 20 rounds and 3 repetitions.
All 32 raw artifacts are retained. The initial comparisons are also retained.
Final pooled median changes ranged from -3.23% to +2.66%, but seven per-case
hierarchical bootstrap intervals remained inconclusive at the 5% budget.
These measurements do not establish a 5% bound across arbitrary processes.

A follow-up uses two persistent workers with separate package paths and
alternating adjacent ``A B`` / ``B A`` rounds. Each case has 5 warmup pairs and
50 measured pairs, with a shared repetition count targeting at least 10 ms
per sample (capped at 1000). Timing happens inside the worker and excludes IPC;
prepared output resets and API allocation/setup remain included. Correctness
is checked before and after measurement, and counts are checked each round.

A deterministic bootstrap resamples ten contiguous blocks of five paired
ratios. All 22 cases have a 95% upper bound below 1.05. The largest paired upper
bound is 1.0244: this supports no material regression for the measured native
CPU workloads in these two workers.

.. list-table:: Paired current/baseline timing ratios, expressed as changes
   :header-rows: 1
   :widths: 28 12 35 25

   * - Workload and scope
     - Cases
     - Median paired changes
     - Largest 95% upper bound
   * - Dense prepared
     - 4
     - -1.19% to +0.18%
     - +0.64%
   * - Dense API
     - 4
     - -0.49% to +0.89%
     - +1.15%
   * - CSR prepared
     - 4
     - -1.31% to +1.85%
     - +2.10%
   * - Sparse CG/PCG API
     - 10
     - -5.54% to +2.01%
     - +2.44%

The paired intervals are conditional on the two workers. Short-block
resampling cannot capture every scheduler, thermal, cache or long-term
process effect, and per-case intervals are not simultaneous coverage.
The earlier fresh-worker uncertainty remains part of the evidence. These
measurements are not a general speedup claim.

Artifacts and reproduction
--------------------------

All data live in ``benchmarks/results/2026-10-15-refactor/``:

* ``manifest.json`` records source provenance, run order and commands.
* ``*-baseline-*.json`` and ``*-current-*.json`` contain all fresh-worker
  timings, inputs, solver policies and environment metadata.
* ``*-initial-comparison.json`` and ``*-comparison.json`` preserve the
  fresh-worker uncertainty results.
* ``paired-comparison.json`` contains all paired raw intervals, exact round
  orders, numerical checks, worker metadata and per-case decisions.

Build the archived baseline with the same compiler/interpreter and Release
flags as the current source. The manifest retains the original commands.
Fresh-worker artifacts can be compared with
``scripts/compare_performance.py``. To reproduce the paired follow-up:

.. code-block:: bash

   results=benchmarks/results/2026-10-15-refactor
   .venv/bin/python scripts/run_paired_performance.py \
       --comparisons "$results"/{csr-prepared,dense-api,dense-prepared,sparse-api}-comparison.json \
       --baseline-pythonpath "$PWD/build/baseline/cpu-release/python:$PWD/build/baseline/source/python" \
       --current-pythonpath "$PWD/build/cpu-release/python" \
       --output build/reproduced-paired.json

Unverified execution scopes
---------------------------

The host cannot run the project's Linux-only private OpenBLAS or CUDA builds,
and OpenMP is disabled in these native CPU measurements. Actual BLAS,
OpenMP, GPU runtime correctness and GPU performance need their Linux/real-GPU
jobs. The existing CI configurations were updated to Python 3.12 and the new
paths; no remote CI run is claimed by this local report.
