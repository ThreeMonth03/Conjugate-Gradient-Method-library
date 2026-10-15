CPU implementation results
==========================

Measurements below cover a C++20 baseline comparison and historical
implementation comparisons. The native dense kernel loses to BLAS-backed
references on several larger cases. Rankings apply to the named inputs and
environments.

C++20 CPU comparison
--------------------

Measured on 2026-10-15 against baseline
``abdba2421a06b6fcdef5cb4439c6dbc9c4eeb394`` and measured implementation
``05cdc861a2cc2689fc98e15b252158924b0156c8``. This historical comparison uses an
Apple M1 on
macOS 26.6.2, CPython 3.12.9, NumPy 2.2.6 and SciPy 1.15.3.
Both builds use AppleClang 21.0.0.21000334, Release ``-O3 -DNDEBUG``,
one native thread, and no OpenMP, BLAS or CUDA. macOS affinity is not pinned.
Source and loaded-binary hashes were checked before and after measurement.

All 22 cases agree on input hashes, solver options and iteration/matvec
counts, and pass the independent true-residual gate. The common
:doc:`methodology` defines inputs, tolerances and timing boundaries.
Fresh-process sampling uses four blocks per build in A-B-B-A-B-A-A-B order,
with five warmups, 20 rounds and three solves per sample. The paired protocol
uses identical Python harness/wrapper files for both native cores, a new
worker pair per case, five warmup pairs and 50 alternating pairs, batching
toward 10 ms. The two measured revisions use the same CPU wrapper.
Both intervals use 10,000 bootstrap resamples and a 5% regression budget.

Fresh-process intervals support no material regression in 17 cases;
5 remain inconclusive.
All 22 paired intervals support no material regression; the largest upper
bound is 1.0312. Paired intervals are conditional on two persistent workers
per case. Neither protocol establishes performance on another host, with
other backend settings or on a GPU. Ratios are measured implementation/baseline; the
milliseconds below are fresh-process medians, and brackets are per-case
95% intervals without simultaneous coverage adjustment.

.. csv-table:: Dense prepared
   :class: benchmark-results
   :header: "Case / candidate","Iter/matvec","Baseline ms","C++20 ms","Fresh ratio (95%)","Paired ratio (95%)"

   "dense-64","49/51","0.0810","0.0810","1.0001 [0.9760, 1.0107]","0.9962 [0.9931, 0.9974]"
   "dense-256","67/70","1.5141","1.5131","0.9993 [0.9661, 1.0091]","1.0009 [0.9997, 1.0025]"
   "dense-1024","70/73","25.3936","25.3741","0.9992 [0.9485, 1.0057]","1.0131 [1.0071, 1.0312]"
   "dense-2048","70/73","100.0581","100.2743","1.0022 [0.9959, 1.0061]","1.0039 [1.0032, 1.0070]"

.. csv-table:: Dense API
   :class: benchmark-results
   :header: "Case / candidate","Iter/matvec","Baseline ms","C++20 ms","Fresh ratio (95%)","Paired ratio (95%)"

   "dense-64","49/51","0.0869","0.0878","1.0104 [0.9586, 1.1171]","0.9979 [0.9967, 0.9988]"
   "dense-256","67/70","1.5194","1.5411","1.0143 [1.0014, 1.0666]","1.0017 [0.9993, 1.0071]"
   "dense-1024","70/73","25.2638","25.4025","1.0055 [0.9937, 1.0744]","0.9896 [0.9877, 0.9929]"
   "dense-2048","70/73","100.3082","100.5569","1.0025 [0.9982, 1.0066]","1.0074 [1.0045, 1.0132]"

.. csv-table:: CSR prepared
   :class: benchmark-results
   :header: "Case / candidate","Iter/matvec","Baseline ms","C++20 ms","Fresh ratio (95%)","Paired ratio (95%)"

   "poisson-8","26/28","0.0113","0.0115","1.0135 [0.9854, 1.0895]","1.0046 [1.0036, 1.0060]"
   "poisson-32","129/133","0.5852","0.5780","0.9877 [0.9525, 1.0156]","0.9611 [0.9610, 0.9614]"
   "poisson-64","290/297","4.9544","4.9965","1.0085 [1.0007, 1.0158]","1.0143 [1.0110, 1.0177]"
   "poisson-128","395/404","26.0298","26.2076","1.0068 [1.0015, 1.0128]","0.9998 [0.9987, 1.0019]"

.. csv-table:: Sparse CG/PCG API
   :class: benchmark-results
   :header: "Case / candidate","Iter/matvec","Baseline ms","C++20 ms","Fresh ratio (95%)","Paired ratio (95%)"

   "scaled-poisson-2d-16 / CG","568/581","0.6132","0.6123","0.9984 [0.9896, 1.0041]","0.9985 [0.9975, 1.0026]"
   "scaled-poisson-2d-16 / PCG","48/50","0.0729","0.0725","0.9946 [0.9773, 1.0027]","0.9920 [0.9858, 0.9954]"
   "scaled-poisson-2d-32 / CG","1498/1529","6.7173","6.8584","1.0210 [0.9335, 1.0737]","1.0197 [1.0182, 1.0220]"
   "scaled-poisson-2d-32 / PCG","115/119","0.6638","0.6514","0.9813 [0.9568, 1.0219]","0.9944 [0.9919, 0.9998]"
   "scaled-poisson-3d-6 / CG","252/259","0.2764","0.2717","0.9829 [0.9621, 0.9969]","0.9954 [0.9930, 0.9994]"
   "scaled-poisson-3d-6 / PCG","25/27","0.0421","0.0413","0.9814 [0.9608, 1.0007]","0.9861 [0.9827, 0.9873]"
   "scaled-poisson-3d-10 / CG","444/454","2.1775","2.1611","0.9925 [0.9647, 1.0089]","0.9948 [0.9923, 0.9970]"
   "scaled-poisson-3d-10 / PCG","37/39","0.2385","0.2385","1.0000 [0.9766, 1.0129]","0.9973 [0.9947, 0.9990]"
   "poisson-2d-16-neutral / CG","50/52","0.0618","0.0611","0.9878 [0.9760, 0.9986]","1.0009 [0.9956, 1.0084]"
   "poisson-2d-16-neutral / PCG","50/52","0.0758","0.0751","0.9906 [0.9759, 0.9976]","0.9853 [0.9834, 0.9871]"

An earlier protocol kept the same workers across the entire CSR suite.
Nine worker-pair runs gave grid 32/64 intervals of [1.0050, 1.0574] and
[1.0053, 1.0560], both inconclusive. Between-case metadata and retained
executors created different allocation histories. The revised protocol
isolates cases and uses a common Python harness; those earlier samples
remain separate and are not pooled with the table above. Neither protocol
proves allocation or scheduling effects have been eliminated.

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
