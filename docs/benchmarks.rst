Benchmarks
==========

Run the comparison suite with:

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
   make pyprof PYPROF_SUITE=sparse-solvers

Benchmarks measure time to an independently verified solution. Choose
``prepared`` to reuse setup or ``api`` to include per-call allocation.
:doc:`benchmarks/methodology` defines their exact boundaries, fixtures,
correctness checks and sampling.

.. toctree::
   :maxdepth: 1

   benchmarks/current
   benchmarks/optimization-20261016
   benchmarks/main-20261016
   benchmarks/cub
   benchmarks/methodology
   benchmarks/cpu
   benchmarks/sparse
   benchmarks/cuda

Each report names its measurement dates and source revisions. The tables are
historical results, not fresh measurements of the current checkout.

Version control keeps the scripts and summary tables. New raw samples,
input hashes and environment metadata go under ``build/``; CI publishes its
smoke-run JSON as workflow artifacts. Documentation builds never require
benchmark execution.

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
