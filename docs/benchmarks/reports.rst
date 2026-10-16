Detailed benchmark reports
==========================

These reports retain the full tables, failed comparisons, uncertainty and
reproduction details behind the :doc:`summary </benchmarks>`. Each describes
its measured source and environment; filenames such as ``current`` and
``main`` identify historical baselines, not the current checkout.
Existing report URLs remain unchanged.

2026-10-17
----------

.. toctree::
   :maxdepth: 1

   Restart-aware optimization decisions — rejected variants, allocation ownership and launch evaluation <restart-optimization-20261017>
   Restart-aware baseline — all CPU/CUDA scopes, remaining losses and seventeen diagnostics <restart-baseline-20261017>

2026-10-16
----------

.. toctree::
   :maxdepth: 1

   Execution reuse and launches — GEMV changes, allocation scopes and remaining losses <execution-reuse-20261016>
   cuBLAS API lifecycle — which construction and synchronization calls cost time? <api-lifecycle-20261016>
   CPU/CUDA optimization evaluation — which candidates meet the comparison gate? <optimization-20261016>
   Main baseline — performance before the optimization campaign <main-20261016>

2026-10-15
----------

.. toctree::
   :maxdepth: 1

   Cleanup baseline — dense, CSR and sparse strategies before later changes <current>
   CPU implementation results — C++20 comparison and native versus BLAS-backed references <cpu>
   Sparse solver results — CG/PCG versus official SciPy and SuperLU <sparse>
   CUDA results — native/cuBLAS versus eager Torch on resident inputs <cuda>

CUB adoption decision
---------------------

The CUB report was published on 2026-10-15; it does not record a measurement
date. Its source revisions, hardware and sampling protocol identify the
comparison. Correctness passed, but the complete adoption gate was not met.

.. toctree::
   :maxdepth: 1

   CUB block reduction — should it replace the existing implementation? <cub>

For fresh measurements, use :doc:`methodology`. Raw JSON, frozen binaries and
profiler traces referenced by historical reports are local evidence under
``build/``; a clean clone reruns the public commands.
