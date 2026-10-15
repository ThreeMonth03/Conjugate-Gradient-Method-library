Reproducible benchmarks
=======================

Install the separate CPU comparison dependencies locally, then run a shared
CLI/pytest fixture suite:

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
   make pyprof
   make pyprof PYPROF_SUITE=cpu-dense-f64-v1
   make pyprof PYPROF_SUITE=cpu-csr-f64-v1
   make pyprof PRESET=cpu-openmp PYPROF_THREADS=4
   make pyprof PYPROF_SCOPE=api
   make pyprof PYPROF_ARGS='--no-fused'
   make pyprof PRESET=cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 \
       PYPROF_ARGS='--implementations native,native-blas,numpy-out,torch-eager-out'

The default ``sparse-solvers`` suite compares native CG/Jacobi PCG, official
SciPy CG/PCG, and SuperLU factor-plus-solve versus factor reuse. The dense and
CSR implementation suites compare a shared CG policy with NumPy/SciPy and
eager CPU PyTorch vector operations. They do not evaluate every library mode
or every direct solver. Native and official SciPy strategy policies can have
different iteration counts.

The explicit ``native-blas`` candidate selects the optional BLAS dense kernel;
``native`` retains the native kernel in the same one-thread comparison. Both
use the common solver policy and remaining vector-fusion setting.

Candidates must satisfy an independently checked stable true residual before
and after measurement. Timed rounds use complete Williams cycles. Each round
times several repeated solves; reported percentiles summarize round averages,
not independent-call tail latencies. ``PYPROF_ROUNDS`` and
``PYPROF_REPETITIONS`` control sampling; ``PYPROF_ARGS`` forwards CLI flags.

``prepared`` reuses native matrix/workspace/Jacobi setup. ``api`` includes
output/workspace allocation and per-call Jacobi setup; canonical sparse
conversion remains outside both scopes. Official SciPy allocates iterative
workspace in both. SuperLU reuse factors outside timing in either scope;
the total candidate factors every timed call. CSC conversion is separate.
Stored-array counts exclude native owning copies, workspace and internal peak
allocator use, so they are not complete memory-use measurements.

JSON preserves raw nanoseconds, sampling, schedules, input hashes, counts,
actual residuals, solver options and the loaded native binary/build identity.
Replay regenerates and checks the saved inputs and actual schedule while
recording the current environment:

.. code-block:: bash

   PYTHONPATH="$PWD/build/cpu-release/python" .venv/bin/python -m cgbench replay \
       build/cpu-release/benchmarks/sparse-solvers-t1-prepared.json \
       --output build/benchmarks/replay.json
   PYTHONPATH="$PWD/build/cpu-release/python" .venv/bin/python -m cgbench compare \
       build/benchmarks/replay.json

Performance depends on matrix structure, conditioning, hardware, setup reuse,
threads and native/BLAS choice. Record affinity and compiler/library metadata;
shared CI enforces correctness instead of speed thresholds. The repository's
`benchmark contracts <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/blob/main/benchmarks/README.md>`_
and `recorded CPU results <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/blob/main/benchmarks/results.md>`_
retain the complete scope notes, reproducible commands and original artifacts.
Those recorded runs predate the optional BLAS kernel and remain unchanged.
The separate `explicit dense BLAS results <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/blob/main/benchmarks/blas-results.md>`_
cover both kernel modes and both scopes, with raw artifacts and timing limits.

Explicit CUDA comparisons
-------------------------

Build as described in :doc:`cuda`, then install GPU comparison dependencies
into the separate environment. Select the device and all candidates:

.. code-block:: bash

   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip -r requirements/cuda-benchmark.lock
   make pyprof PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" \
       PYPROF_SUITE=smoke-cuda-dense \
       PYPROF_ARGS='--device cuda --cuda-device 0 --implementations cuda-native,cuda-cublas,torch-cuda-eager'
   make pyprof PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" \
       PYPROF_SUITE=smoke-cuda-csr PYPROF_SCOPE=api \
       PYPROF_ARGS='--device cuda --cuda-device 0 --implementations cuda-native,torch-cuda-eager'

Use ``cuda-dense-f64-v1`` or ``cuda-csr-f64-v1`` for the corresponding full
fixture suites. CPU remains the CLI default; ``--device cuda`` requires an
explicit implementation list. cuBLAS accepts only dense operators. Algorithm,
fusion, refresh and preconditioner flags retain their explicit meanings.

Both GPU scopes start with resident device inputs. ``prepared`` caches matrix,
workspace and Jacobi setup; ``api`` allocates output/workspace and constructs
Jacobi per call. Each timed call restores the initial device vector. The timer
fences before the repeated-call block and after its last call, so completed
wall time includes host dispatch, iteration scalar synchronization and final
completion. This is not a GPU-event-only kernel measurement.

Host-to-device setup and correctness-gate downloads remain outside solve
timing. The artifact records upload/setup separately; neither scope is an
end-to-end host-to-host comparison. Retain GPU/driver/runtime/compiler metadata,
the actual loaded CUDA binary hash and device mapping with results. Logical
CUDA device ordinals need not match physical ``nvidia-smi`` indices. Shared CI
compiles CUDA without a GPU; execution proof requires an available device.

The `recorded CUDA results <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/blob/main/benchmarks/cuda-results.md>`_
include all dense/CSR CG and Jacobi PCG measurements in both scopes, original
samples, device/build identity and replay commands. They retain cases where
native CUDA has a higher median than eager PyTorch CUDA.
