CPU implementation results
==========================

These historical measurements were made on 2026-10-15. The native dense
kernel loses to BLAS-backed references on several larger cases; CSR has lower
medians in the measured cases. Rankings apply to these inputs and environments.

Native dense and CSR
--------------------

Measured revision: ``ccc8ec7a76a569e4e2505251af3cb7d2469eb594``.

* Intel Core i7-13700K, WSL2/Linux 6.6.87.2, 24 logical CPUs.
* CPython 3.12.3; NumPy 2.2.6 / OpenBLAS 0.3.29; SciPy 1.15.3 /
  OpenBLAS 0.3.28; PyTorch 2.8.0+cpu / MKL 2024.2, AVX2.
* GNU C++ 16.0.1, Release -O3 -DNDEBUG, without fast-math.
  Four-thread native builds enable OpenMP.
* Affinity is logical CPU 0 at one thread, or 0,2,4,6 at four threads;
  this does not establish physical-core isolation from the Windows host.
  Runtime pools use the requested thread count and Torch interop uses one.

Times are median milliseconds per solve in prepared scope, with three
warmups, 30 Williams rounds and five solves per round. The common
:doc:`methodology` defines float64 inputs, solver options and verification.
All implementation candidates have identical iteration/matvec counts per
case. SciPy-out is the matching-policy loop, separate from the official
SciPy solver in :doc:`sparse`.

.. csv-table:: Dense prepared solves (ms)
   :class: benchmark-results
   :header: "Threads","n","Native","NumPy-out","Torch eager","Iter/matvec"

   "1","64","0.0401","0.2617","0.4400","49/51"
   "1","256","0.6550","0.7652","1.0610","67/70"
   "1","1024","11.2226","8.8882","9.1303","70/73"
   "1","2048","126.8924","80.9526","89.9656","70/73"
   "4","64","0.0617","0.2125","0.3367","49/51"
   "4","256","0.3056","0.6051","0.6555","67/70"
   "4","1024","4.1187","5.3735","3.3042","70/73"
   "4","2048","49.1773","37.7599","45.1775","70/73"

Native is faster at n=64/256, but slower than both references at n=2048.
At n=1024 it loses to both at one thread and to Torch at four threads.
Four threads increase the native n=64 median.

.. csv-table:: CSR prepared solves (ms)
   :class: benchmark-results
   :header: "Threads","Grid / n","Native","SciPy-out","Torch eager","Iter/matvec"

   "1","8 / 64","0.0097","0.1543","0.2015","26/28"
   "1","32 / 1024","0.3626","1.3117","1.3464","129/133"
   "1","64 / 4096","2.8523","6.0579","4.9494","290/297"
   "1","128 / 16384","16.2138","31.1988","19.5595","395/404"
   "4","8 / 64","0.0209","0.1410","0.2229","26/28"
   "4","32 / 1024","0.4481","1.1998","1.2720","129/133"
   "4","64 / 4096","3.8211","6.1291","35.9296","290/297"
   "4","128 / 16384","5.9082","37.3314","17.2837","395/404"

Native has a lower median than both references in all eight CSR rows.
Four native threads are slower than one on grids 8/32/64 and faster on
grid 128. These are whole-solve timings including dispatch/thread overhead.

Explicit dense BLAS
--------------------

Measured revision: ``abdc0424d10d1a9a569481439c824a80bdd207e2``,
also on 2026-10-15. The host and reference versions are as above. The native
build has OpenMP disabled and uses one thread on logical CPU 0.
Its private provider is sequential OpenBLAS 0.3.30, LP64, with locking and
CORE2/HASWELL kernels under an SSE3/SSSE3 baseline.

Each candidate uses standard CG without a preconditioner and the common
options. Three warmups and five solves per round are used; 30 requested
rounds become 32 to complete the four-candidate Williams cycle.
All 32 candidate/case results pass the independent residual gate.
Iteration/matvec counts are 49/51, 67/70, 70/73 and 70/73 by increasing size.

.. csv-table:: Dense native and explicit BLAS kernels (ms)
   :class: benchmark-results
   :header: "Scope","n","Native","BLAS","NumPy-out","Torch eager"

   "prepared","64","0.034298","0.023092","0.205357","0.312009"
   "prepared","256","0.577629","0.395360","0.545323","0.697875"
   "prepared","1024","10.526326","7.459030","8.597074","8.692266"
   "prepared","2048","87.065250","55.457808","56.078584","58.964700"
   "api","64","0.043802","0.029642","0.228767","0.333154"
   "api","256","0.607748","0.400338","0.660380","0.830903"
   "api","1024","10.056445","6.894183","7.582507","8.053867"
   "api","2048","86.794532","54.096938","57.598952","59.408369"

BLAS medians are lower than NumPy and Torch in these samples.
The prepared n=2048 NumPy comparison differs by about 1% and distributions
overlap, so it does not establish a consistent advantage.
The native kernel remains slower than BLAS-backed references at large sizes.
Prepared/API are separate process runs; subtracting their medians does not
isolate allocation cost.

Repeat the protocols
---------------------

The following commands run the same fixtures and sampling on the current
checkout, recording its identity. To measure the original implementation,
build its named revision in a separate checkout.

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
   for suite in cpu-dense-f64-v1 cpu-csr-f64-v1; do
       taskset -c 0 make pyprof PRESET=cpu-release \
           PYPROF_SUITE="$suite" PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
           PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 PYPROF_ARGS='--warmups 3'
       taskset -c 0,2,4,6 make pyprof PRESET=cpu-openmp \
           PYPROF_SUITE="$suite" PYPROF_THREADS=4 PYPROF_SCOPE=prepared \
           PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 PYPROF_ARGS='--warmups 3'
   done
   for scope in prepared api; do
       taskset -c 0 make pyprof PRESET=cpu-blas \
           PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE="$scope" \
           PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 \
           PYPROF_ARGS='--warmups 3 --implementations native,native-blas,numpy-out,torch-eager-out'
   done

Choose available affinity IDs on your host. New raw output stays under
``build/<preset>/benchmarks/``; :doc:`methodology` shows summary and replay
commands. These single-process timing percentiles do not establish
independent-run confidence or a universal speedup.
