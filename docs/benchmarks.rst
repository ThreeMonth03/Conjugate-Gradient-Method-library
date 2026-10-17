Benchmarks
==========

Benchmarks measure completed time to an independently verified solution.
The latest :doc:`complete mode baseline <benchmarks/main-complete-20261017>`
measures compiled revision ``445275a`` across 134 configurations, 268 fresh
processes and 3,096 eligible candidate/case outcomes. Its 1,548 table rows
retain both forward/reverse process observations over 13 input fixtures.

Results and remaining limitations
---------------------------------

.. csv-table:: Complete mode baseline, 2026-10-17
   :class: benchmark-results
   :header: "Comparison", "Higher native medians / matched pairs", "Scope and limit"
   :widths: 30, 25, 45

   "CPU dense: serial, OpenMP and BLAS", "93 / 384", "Native and native-BLAS versus NumPy/Torch; all algorithm, fusion and scope rows retained"
   "CPU CSR: serial and OpenMP", "1 / 192", "The higher point is OpenMP explicit/unfused prepared Poisson-128 versus Torch; its ratio is 1.014"
   "CUDA native dense and CSR", "0 / 288", "All measured native medians are lower than matched Torch; some explicit dense pairs perform different work"
   "CUDA cuBLAS dense", "16 / 144", "All higher points are standard Jacobi fresh API: four sizes, both fusion flags and both execution modes; ratios 1.05–1.42"
   "Sparse CG/PCG/direct strategies", "25 / 80", "Different solver strategies and unknown external restart counts; descriptive time-to-solution comparisons"

These totals contain 1,088 matched native/reference pairs: 953 lower native
medians and 135 higher. A pair identifies a fixture and configuration, not a
unique matrix. The estimator is the median of two process medians; their
range is descriptive, without confidence intervals, statistical-significance
claims or an A/B adoption verdict. Every higher point remains in the full
report. CUDA Session/prepared results cannot replace the fresh API contract.

Recorded iterations, matvecs, restarts and their knownness repeat across both
processes for each candidate/configuration. No cross-scope/mode/preset work
changes were observed within the same candidate, algorithm and
preconditioner group. Across
frameworks, 980 matched pairs have equal known I/M/R, 28 explicit dense pairs
have different known work, and 80 strategy pairs have unknown restart parity.
Official SciPy CG/PCG and SuperLU retain unknown restart counts rather than
fabricated zeros. Work differences are not normalized into pure-kernel
ratios. Torch executes this project's eager recurrence, not an official Torch
CG solver; SuperLU is a general sparse direct solver.

The :doc:`earlier optimization evaluation <benchmarks/restart-optimization-20261017>`
retains the ownership variants, rejected four-row CPU experiment, launch
studies, control regressions and exploratory intervals at their measured
sources. Those historical results are separate from this complete baseline;
:doc:`benchmarks/reports` preserves every earlier report URL.

The subsequent :doc:`fixed A/B evaluation <benchmarks/main-optimization-20261017>`
records the CUDA device-restoration and CPU dense scheduling candidates against
the same compiled baseline, retaining every native and framework control and
each adoption decision. Read its source pins before applying a candidate result
to production. The :doc:`CUDA phase diagnostics <benchmarks/main-cuda-phases-20261017>`
keep fresh API, Session, prepared and workspace-only probes distinct;
:doc:`CPU GEMV diagnostics <benchmarks/main-cpu-components-20261017>` retain
independent numerical checks and compiler evidence. Diagnostic durations are
separate from the completed-solve A/B gate.

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

The public tag ``benchmarks/main-complete-20261017`` identifies actual compiled
source ``445275a3994e342b06de4093889f3f814c8706c6``. The full report preserves
compiled tree, staged binaries, Python harness, compiler controls and original
raw/import/log identities. The reference numerical-range fix is included in
this measured source. A later checkout or optimization is not covered merely
because it shares the same package version.

Scope and reproduction
----------------------

The matrix contains 90 standard configurations and 44 explicit-none
configurations. Both algorithms cover fused/separate execution, supported
CPU native/BLAS and CUDA native/cuBLAS backends, and their declared scopes.
Standard CG includes none/Jacobi; explicit recomputes the residual every
iteration and supports no preconditioner. CPU covers API/prepared, CUDA covers
API/Session/prepared with explicit host/device execution. CSR rejects BLAS
and cuBLAS. The five sparse-strategy fixtures retain all six CG/PCG/LU aliases
in API/prepared scopes with the standard algorithm.

FP64 inputs use rtol=1e-8, atol=1e-12, maxiter=2000 and seed 20261017.
Standard residual refresh is every 50 updates. Each artifact contains three
warmups and 36 complete balanced Williams rounds, one solve per round, with
all declared candidates. Configurations run forward and then in exact reverse.

Fresh ``api`` charges per-call output, workspace and requested Jacobi setup.
CUDA ``session`` reuses execution resources with fresh output and requested
per-call Jacobi. ``prepared`` reuses output, workspace and setup. These scopes
remain distinct; native host/device selection does not change Torch's control
engine. Backend, algorithm, fusion and execution mode are explicit choices.

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
