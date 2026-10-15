Refactor validation
===================

This historical check was recorded on 2026-10-15 for the feature-layout
refactor, using CPython 3.12.9, AppleClang 21 and NumPy 2.2.6 on macOS ARM64.
It compares native one-thread Release builds (-O3 -DNDEBUG) with the same
interpreter, compiler, inputs and solver options.

Baseline revision: ``fa99ef888606d5f34e2a38660b525f3883b4be4b``.
Refactored revision: ``ea8cbb37a316db1e7fd999369e2b99f10c4fb7b0``.
The measured core/source SHA-256 is
``fdfef79c3b040e1c1c1eadde2b7125a43e4a22689ee2626a0a79c1eec81ba368``.

Correctness and builds
----------------------

The original local checks passed:

* 157 Python tests and 51 native tests; optional capabilities skipped
  78 Python and 13 native cases.
* All 51 available native tests under AddressSanitizer/UndefinedBehaviorSanitizer.
* Ruff, clang-format and the native clang-tidy build.
* Warning-free Sphinx, 54 doctests and three examples.
* A relocated sdist-to-wheel build and installed dense/CSR/CG/PCG/prepared
  checks with NumPy 2.2.6 and separately with the runtime floor NumPy 1.26.0.
* A native-only installation and independent C++ consumer; all 23 public
  CPU/CUDA headers compile without Python or CUDA SDK includes.
* CUDA binding/support and non-kernel host syntax against CUDA 12.8 headers,
  plus Clang 22 CUDA host/device frontend checks for sm_75.

Numerical tests check independent residuals, known solutions, convergence
and failure status, aliasing, ownership, workspace reuse and floating-point
range behavior. CUDA syntax checks alone do not establish linking or device
execution. This local comparison did not measure Linux BLAS, OpenMP or GPU
runtime behavior.

Performance summary
--------------------

The material-regression budget is 5%. Inputs, convergence, iteration/matvec
counts and independent true-residual checks must agree before comparing time.

Fresh processes ran four baseline and four current blocks in order
A B B A B A A B. Each block used five warmups, 20 rounds and three solves per
round. Pooled median changes ranged from -3.23% to +2.66%, but seven per-case
hierarchical bootstrap intervals remained inconclusive at the 5% budget.
This does not establish a 5% bound across arbitrary processes.

A follow-up used two persistent workers with separate package paths and
alternating adjacent AB/BA rounds. Each case used five warmup pairs and
50 measured pairs, with a shared repetition count targeting 10 ms per sample
and capped at 1000. Worker-local timing excludes IPC; output resets and API
allocation/setup remain included. Correctness is checked before and after,
with counts checked each round.

.. csv-table:: Paired current/baseline changes
   :class: benchmark-results
   :header: "Workload / scope", "Cases", "Median changes", "Largest 95% upper bound"

   "Dense prepared", "4", "-1.19% to +0.18%", "+0.64%"
   "Dense API", "4", "-0.49% to +0.89%", "+1.15%"
   "CSR prepared", "4", "-1.31% to +1.85%", "+2.10%"
   "Sparse CG/PCG API", "10", "-5.54% to +2.01%", "+2.44%"

A deterministic bootstrap resamples ten contiguous blocks of five paired
ratios, using 10,000 resamples. All 22 per-case 95% upper bounds are below
1.05; the largest is 1.0244. This supports no material regression for these
native CPU workloads in those two workers.

The paired intervals are conditional on the workers and do not capture
every scheduler, thermal, cache or long-term process effect.
Intervals are per case, without simultaneous coverage.
The fresh-worker uncertainty remains part of the result; this is not a
general speedup claim.

Repeat the comparison
----------------------

Build both revisions with the same interpreter/compiler and Release flags.
The following example uses a fresh baseline directory and the current
checkout for the second worker. Results describe the versions actually
loaded; rerunning today does not recreate the original timing samples.

.. code-block:: bash

   mkdir -p build/baseline/source
   git archive fa99ef888606d5f34e2a38660b525f3883b4be4b \
       | tar -x -C build/baseline/source
   .venv/bin/cmake -S build/baseline/source -B build/baseline/cpu-release -G Ninja \
       -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
       -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
       -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
       -DCG_ENABLE_OPENMP=OFF -DCG_ENABLE_BLAS=OFF -DCG_ENABLE_CUDA=OFF
   .venv/bin/cmake --build build/baseline/cpu-release --target _cgcore
   make build PRESET=cpu-release

   baseline_pythonpath="$PWD/build/baseline/cpu-release/python:$PWD/build/baseline/source/python"
   current_pythonpath="$PWD/build/cpu-release/python"
   paired() {
       PYTHONPATH="$current_pythonpath" .venv/bin/python scripts/run_paired_performance.py \
           --baseline-pythonpath "$baseline_pythonpath" \
           --current-pythonpath "$current_pythonpath" \
           --suite "$1" --scope "$2" --implementations "$3" \
           --output "build/benchmarks/$4-paired.json"
   }
   paired cpu-dense-f64-v1 prepared native dense-prepared
   paired cpu-dense-f64-v1 api native dense-api
   paired cpu-csr-f64-v1 prepared native csr-prepared
   paired sparse-solvers api native-cg,native-pcg sparse-api

The paired runner constructs jobs from suite/options and checks both workers'
input hashes and numerical work. It needs no historical result file.
For independent fresh-process blocks, use the sampling and order above
with ``cgbench run``, then pass the newly generated baseline/current JSON to
``scripts/compare_performance.py --baseline ... --current ... --output ...``.
Raw outputs stay under ``build/``; only this summary is tracked.
