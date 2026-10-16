Benchmarks
==========

Benchmarks measure completed time to an independently verified solution.
The latest :doc:`2026-10-17 evaluation <benchmarks/restart-optimization-20261017>`
retains adopted changes, rejected experiments, uncertainty and every framework
control. Measurements describe their pinned source and harness, rather than
an unmeasured later ``main``.

Results and remaining limitations
---------------------------------

.. csv-table:: Restart-aware evaluation
   :class: benchmark-results
   :header: "Comparison", "Observation", "Limit"
   :widths: 30, 35, 35

   "Full CPU/CUDA baseline", "108 processes; 1,336 outcomes; 78 of 512 matching native/reference pairs have higher native medians", "Descriptive baseline; official SciPy/direct strategies have unknown restart counts"
   "Fresh cuBLAS resource ownership", "One fewer allocation/free pair per API call after consolidating owned storage and borrowing scratch", "Follow-up wall-time batch is inconclusive; selected after an initial rejected variant"
   "CPU four-row GEMV", "Rejected; production keeps the existing two-row kernel and retains stronger correctness tests", "Four native regressions; framework and unchanged BLAS controls also drifted"
   "CUDA stable norm/control", "One fewer kernel launch per recurrence reaching control; memcpy/status-sync counts unchanged", "Prepared batch has one Torch control regression; separate three-scope batch is inconclusive"
   "cuBLAS Jacobi fresh API, host/device", "All eight size/mode medians remain higher than matching Torch, ratio 1.11–1.46", "Session/prepared medians are lower; reuse results cannot replace the fresh contract"

The optimization studies retain all 176 fresh-process artifacts, 2,144 outcomes
and 536 native/control strict rows. They use a fixed 5% regression budget and
95% hierarchical intervals, without removing control drift or selecting a
passing subset. Within-budget does not prove a speedup. Sequential follow-ups
have unadjusted exploratory intervals. CPU component diagnostics and the 57
CUDA captures are separate evidence, not additive wall-time attribution.

Matching native and same-policy references verify known iterations, matvecs
and restarts. Official SciPy CG/PCG and direct methods retain unknown restart
counts. Framework comparisons use the same accepted tolerance; Torch executes
this project's eager recurrence, rather than an official Torch CG solver.
The :doc:`full baseline <benchmarks/restart-baseline-20261017>` retains all
remaining CPU, sparse-strategy and CUDA losses.

Measurement context
-------------------

Hardware: Intel Core i7-13700K under WSL2/Linux 6.6.87.2; NVIDIA GeForce
RTX 2060 (sm75) for CUDA. OpenMP uses two threads with logical affinity [0,2];
other runs use one thread with affinity [0]. Affinity does not isolate the
Windows host. GPU clocks, display activity and thermals were not controlled.

CPU builds use GNU 16.0.1; CUDA builds use GNU 13.3.0 and CUDA 12.8.93.
Release flags are portable ``-O3 -DNDEBUG`` without architecture-specific
optimization or fast math. NumPy is 2.2.6; separate environments use Torch
2.8.0+cpu and 2.8.0+cu128. The recorded CUDA driver is 610.88, driver API
13030 and runtime 12080. Earlier reports retain their own environments.

The public tags ``benchmarks/restart-baseline-20261017``,
``benchmarks/cuda-borrowed-scratch-20261017``,
``benchmarks/cpu-four-row-20261017`` (rejected) and
``benchmarks/cuda-norm-control-20261017`` identify actual measured revisions.
The complete report also preserves the rejected initial resource variant.
A later reference numerical-range precedence fix was validated after timing;
these framework times do not measure that new Python source.

Scope and reproduction
----------------------

FP64 inputs use rtol=1e-8, atol=1e-12, seed 20261017 and residual refresh every
50 updates. Fresh ``api`` charges per-call output, workspace and requested
Jacobi setup. CUDA ``session`` reuses execution resources with fresh output
and per-call Jacobi. ``prepared`` reuses output, workspace and Jacobi.
These scopes are compared separately. Backend, fusion and CUDA host/device
execution are explicit choices; no automatic selection is used.

CUDA inputs are resident. Completed wall timing includes dispatch, output
reset, scalar/status synchronization and completion fences; uploads and
independent correctness downloads are excluded. This is not host-to-host
latency or pure-kernel time. Resource probes and profiler captures have their
own documented boundaries.

Use the full report's ``make pyprof`` commands and :doc:`benchmarks/methodology`
for fresh reproduction, including sparse CG/PCG versus direct methods.
Dependencies, environments and build products stay project-local. Scripts
and summary tables are versioned; raw samples, binary/input hashes and
profiler captures remain under ``build/``. Reproduction generates new evidence
without previous JSON, and documentation builds do not run benchmarks.

Read further
------------

.. toctree::
   :maxdepth: 1

   benchmarks/methodology
   benchmarks/reports

Diagnostic profiling
--------------------

See :ref:`benchmark-diagnostic-profiling` for resource and Nsight Systems
capture commands.
