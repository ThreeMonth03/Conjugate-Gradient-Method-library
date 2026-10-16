Benchmarks
==========

Benchmarks measure time to an independently verified solution. The selected
results below are historical measurements from **2026-10-16**, using frozen
compiled revisions. They are not new measurements of the current ``main``.

Representative results
----------------------

.. csv-table:: Results and remaining limitations
   :header: "Comparison and scope", "Observation", "Limit"
   :widths: 32, 34, 34

   "CPU dense, n=2048, unfused; none/Jacobi, prepared/API", "32–42% lower median completed-wall time versus baseline", "Across n=64–2048, all 16 unfused rows meet the 5% budget; the original fused none/API batch retains 3 regressions and its pooled confirmation is inconclusive"
   "CUDA launch A/B, prepared/device, fused=True; dense/CSR, none/Jacobi", "21 of 24 native/cuBLAS rows meet the 5% regression budget; 3 remain inconclusive", "The complete batch is inconclusive; meeting the budget does not prove a speedup"
   "CUDA cuBLAS dense Jacobi, fresh API; host/device", "cuBLAS solve median times are higher than eager Torch in all 8 size/mode rows", "Per-call output, workspace and Jacobi setup are charged; prepared/Session results cannot replace this contract"

Every selected answer passes an independent true-residual check. Strict A/B
rows also preserve iteration/matvec counts. Framework median comparisons are
descriptive; Torch runs this project's eager CG/PCG recurrence, not an official
Torch CG solver. The :doc:`complete report <benchmarks/execution-reuse-20261016>`
retains every result, uncertainty interval and framework control.

Measurement context
-------------------

Hardware: Intel Core i7-13700K under WSL2/Linux 6.6.87.2; NVIDIA GeForce
RTX 2060 (sm75) for CUDA. Runs use one thread and logical CPU 0 affinity.
CPU builds use GNU 16.0.1; CUDA builds use GNU 13.3.0 and CUDA 12.8.93.
Both use Release ``-O3 -DNDEBUG``. NumPy is 2.2.6 and Torch is 2.8.0
(separate CPU/CUDA environments). GPU clocks, display activity and Windows
background load were not controlled.

.. csv-table:: Frozen compiled versions used by the summary
   :header: "Cohort", "Baseline", "Measured version"
   :widths: 34, 33, 33

   "CPU GEMV A/B", "`6bfb7b4905e7 <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/tree/6bfb7b4905e79381a3d09772016aab7a860a0dad>`_", "`7a7e6ea6e998 <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/tree/7a7e6ea6e998a193514a1bb060d3cc67c0aaac45>`_"
   "CUDA launch A/B", "`03379431a8cc <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/tree/03379431a8cc4426447f1d5dfd011d908d9976bb>`_", "`ca0d6afebeda <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/tree/ca0d6afebedaef1a1c4c7a5a2d2b4ffded6e1e2d>`_"

The CUDA fresh-API characterization also uses ``ca0d6afebeda``. Source, stage
and loaded-binary hashes are retained in the complete report; later rebases
do not replace those compiled identities.

FP64 inputs use rtol=1e-8, atol=1e-12 and residual refresh every 50 updates.
``prepared`` reuses output/workspace/Jacobi; ``api`` charges per-call setup.
CUDA ``session`` reuses execution resources with fresh output and per-call
Jacobi. CUDA timings use resident inputs and include host dispatch, scalar
synchronization, output reset and completion fences; uploads and correctness
downloads are excluded. These are completed solves, not kernel-only or
host-to-host latency. The methodology defines fixtures, sampling and scopes.

Read further
------------

.. toctree::
   :maxdepth: 1

   benchmarks/methodology
   benchmarks/reports

Version control keeps scripts and summary tables. Raw samples, input hashes
and environment metadata belong under ``build/``; CI publishes smoke-run JSON
as workflow artifacts. Reproduction generates fresh measurements without
requiring previous JSON, and documentation builds never run benchmarks.

Diagnostic profiling
--------------------

See :ref:`benchmark-diagnostic-profiling` for single-case resource and
Nsight Systems captures.
