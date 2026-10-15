CUDA results
============

Historical measurements made on 2026-10-15 at revision
``6d3ebb16a4a0f47732326045c1a0a55782f91cb7``.
Native CUDA has lower prepared medians than eager Torch on these fixtures,
but dense Jacobi API solves are slower. This does not establish that GPU
solves beat CPU solves or that one framework is universally faster.

Hardware and protocol
----------------------

* NVIDIA GeForce RTX 2060, compute capability 7.5, approximately 6 GiB,
  logical device 0; NVIDIA driver 591.86.
* CUDA toolkit/compiler 12.8.93, runtime API 12080, driver API 13010,
  architecture 75, GNU C++ 13.3.0, Release -O3 -DNDEBUG.
  CPU BLAS and OpenMP are disabled.
* Intel Core i7-13700K, WSL2/Linux 6.6.87.2, CPython 3.12.3;
  affinity is logical CPU 0, without physical-core isolation.
* NumPy 2.2.6, SciPy 1.15.3 and PyTorch 2.8.0+cu128.
  Host/native library and Torch intraop/interop limits are one.
  Torch uses float64, with TF32 and mixed precision disabled.

GPU clocks, thermal state and display activity were not controlled.
Torch deterministic-algorithm enforcement is off. Some p5–p95 ranges
overlap, including dense-2048 prepared Jacobi; medians do not establish
statistical significance across independent runs.

Both scopes use resident device inputs. Timed wall time includes host
dispatch, device output reset, scalar synchronization and completion fences.
Uploads and correctness downloads are excluded. Prepared reuses resources;
API includes output/workspace and Jacobi construction. Torch allocations
use its caching allocator. This is not a host-to-host or kernel-only metric.
See :doc:`methodology` for fixture and common solver settings.

Dense runs use three warmups, 30 Williams rounds and three solves per round;
CSR uses three warmups, 30 rounds and one solve per round.
All 80 formal candidate/case results pass independent CPU true-residual
checks before and after timing, with matching iteration/matvec counts
within each case. The Torch candidate is this project's matching-policy
CG loop using eager Torch operations, not an official Torch CG solver.

Dense
------

.. csv-table:: Dense median ms [p5–p95]
   :class: benchmark-results
   :header: "Preconditioner","Scope","n","Native","cuBLAS","Torch eager","Iter/matvec"

   "none","prepared","64","6.1315 [5.5288–6.8216]","6.2148 [5.6948–6.4958]","11.9994 [11.3500–13.5792]","49/51"
   "none","prepared","256","8.6112 [8.1405–9.4791]","8.3784 [7.8764–9.2537]","16.3782 [14.8247–18.2635]","67/70"
   "none","prepared","1024","11.3483 [10.7215–12.6136]","9.9039 [9.4584–10.7193]","16.4007 [15.5007–17.6991]","70/73"
   "none","prepared","2048","18.3767 [18.0155–19.3238]","16.8248 [16.1379–17.7859]","22.9550 [22.1366–24.2642]","70/73"
   "none","api","64","8.8116 [8.1640–9.8874]","8.8940 [8.1536–9.7393]","15.4370 [14.1193–17.5446]","49/51"
   "none","api","256","11.6796 [10.8191–14.1263]","11.4278 [10.4522–13.8319]","20.7257 [19.8008–22.6405]","67/70"
   "none","api","1024","14.2387 [13.6514–16.1960]","12.7405 [12.1063–13.8802]","20.6343 [19.1393–25.5663]","70/73"
   "none","api","2048","21.7394 [21.0434–22.5869]","19.9793 [19.4223–21.8189]","26.3218 [25.3167–28.4057]","70/73"
   "jacobi","prepared","64","1.0777 [0.9780–1.5740]","1.1068 [0.9990–1.5797]","2.2646 [1.8725–3.0045]","5/7"
   "jacobi","prepared","256","0.9000 [0.8601–1.3171]","0.8742 [0.8322–1.4108]","1.6115 [1.4289–2.2755]","4/6"
   "jacobi","prepared","1024","1.2437 [1.1691–1.7080]","1.1016 [1.0324–1.6057]","1.8063 [1.5024–2.5632]","4/6"
   "jacobi","prepared","2048","1.5431 [1.3292–1.9549]","1.3059 [1.2128–2.0588]","1.7247 [1.5899–2.0442]","3/5"
   "jacobi","api","64","3.0451 [2.9058–3.6960]","3.0968 [2.9008–3.8508]","2.2362 [2.0617–2.9233]","5/7"
   "jacobi","api","256","2.8852 [2.7037–3.3109]","2.9572 [2.7253–3.5014]","1.9897 [1.7795–2.8037]","4/6"
   "jacobi","api","1024","2.9641 [2.6735–3.4943]","2.8785 [2.6282–3.4618]","1.9499 [1.7441–2.6093]","4/6"
   "jacobi","api","2048","3.1001 [2.9367–3.5919]","2.9321 [2.8191–3.3170]","2.1226 [1.8960–2.7302]","3/5"

Without preconditioning, both CUDA implementations have lower medians
than Torch at all sizes in both scopes. Jacobi reduces iterations to
5/4/4/3 from 49/67/70/70, but native Jacobi API medians are 2.89–3.10 ms
versus Torch's 1.95–2.24 ms. cuBLAS also loses in those four API rows.
Small cases sometimes favor the custom kernel over cuBLAS; no crossover
rule follows from these samples.

CSR
----

The Poisson diagonal is constant 4, so Jacobi is scalar scaling without
a conditioning benefit. It leaves iterations unchanged and adds work.

.. csv-table:: CSR median ms [p5–p95]
   :class: benchmark-results
   :header: "Preconditioner","Scope","Grid / n","Native","Torch eager","Iter/matvec"

   "none","prepared","8 / 64","3.0453 [2.9106–4.2924]","5.6538 [5.2622–7.2216]","26/28"
   "none","prepared","32 / 1024","15.4805 [14.2131–16.5986]","28.2484 [26.4809–32.3113]","129/133"
   "none","prepared","64 / 4096","33.4554 [32.7183–37.1609]","63.7745 [61.4152–68.2877]","290/297"
   "none","prepared","128 / 16384","51.1318 [46.5387–57.8908]","92.1287 [86.8644–102.5793]","395/404"
   "none","api","8 / 64","4.9860 [4.7505–6.8917]","5.8776 [5.3284–8.8609]","26/28"
   "none","api","32 / 1024","18.2143 [16.2771–21.1439]","29.6476 [27.1746–34.0975]","129/133"
   "none","api","64 / 4096","38.5098 [35.1875–41.5713]","69.9089 [63.2895–78.7443]","290/297"
   "none","api","128 / 16384","53.2059 [50.3060–63.8343]","92.1661 [85.7574–105.3291]","395/404"
   "jacobi","prepared","8 / 64","4.5916 [4.3385–6.3191]","8.5638 [7.8588–10.7557]","26/28"
   "jacobi","prepared","32 / 1024","23.8808 [21.5462–27.8935]","42.0058 [39.7512–47.6718]","129/133"
   "jacobi","prepared","64 / 4096","52.7241 [48.8249–66.0076]","98.3060 [92.4234–110.8748]","290/297"
   "jacobi","prepared","128 / 16384","79.0034 [73.6615–98.2985]","145.9772 [130.0647–170.5285]","395/404"
   "jacobi","api","8 / 64","6.4958 [6.1321–8.3881]","9.2400 [8.0655–12.8448]","26/28"
   "jacobi","api","32 / 1024","24.9432 [23.9180–29.3167]","40.5634 [39.4436–43.5988]","129/133"
   "jacobi","api","64 / 4096","56.4907 [50.2472–75.9376]","99.2850 [91.1849–123.5182]","290/297"
   "jacobi","api","128 / 16384","81.8643 [75.3686–104.2133]","139.4909 [128.7492–178.4611]","395/404"

Native has a lower median than Torch in every measured CSR row.
Jacobi increases both implementations' medians; for grid 128,
prepared native time rises from 51.13 to 79.00 ms. This neutral fixture
does not describe Jacobi's benefit on variably scaled systems.

Accuracy
---------

API probes have the same counts and corresponding residuals.
Candidates can have different accepted errors due to floating-point
reductions; bitwise equivalence is not assumed.

.. csv-table:: Prepared accuracy across candidates
   :class: benchmark-results
   :header: "Fixture","Preconditioner","Iter/matvec","True residual range","Threshold"

   "dense-64","none","49/51","2.044745e-06–2.206060e-06","2.737557e-06"
   "dense-256","none","67/70","5.650814e-06–5.992621e-06","6.342306e-06"
   "dense-1024","none","70/73","9.505000e-06–9.505000e-06","1.079068e-05"
   "dense-2048","none","70/73","1.322930e-05–1.322930e-05","1.474103e-05"
   "dense-64","jacobi","5/7","1.141019e-07–1.141019e-07","2.737557e-06"
   "dense-256","jacobi","4/6","1.098968e-06–1.098968e-06","6.342306e-06"
   "dense-1024","jacobi","4/6","1.722285e-07–1.722285e-07","1.079068e-05"
   "dense-2048","jacobi","3/5","1.245009e-05–1.245009e-05","1.474103e-05"
   "poisson-8","none/jacobi","26/28","2.012273e-07–2.012273e-07","3.054924e-07"
   "poisson-32","none/jacobi","129/133","1.327602e-06–1.327602e-06","1.367232e-06"
   "poisson-64","none/jacobi","290/297","2.772948e-06–2.772948e-06","2.810075e-06"
   "poisson-128","none/jacobi","395/404","5.661370e-06–5.661370e-06","5.749348e-06"

Repeat the protocol
--------------------

Follow :doc:`../cuda` to build on a compatible GPU host.
These commands use the same fixtures/sampling on the current checkout.
To measure the original source, build the named revision separately.
Choose your own compatible architecture and affinity IDs.

.. code-block:: bash

   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip -r requirements/cuda-benchmark.lock
   for storage in dense csr; do
       if [ "$storage" = dense ]; then
           candidates=cuda-native,cuda-cublas,torch-cuda-eager
           repetitions=3
       else
           candidates=cuda-native,torch-cuda-eager
           repetitions=1
       fi
       for preconditioner in none jacobi; do
           for scope in prepared api; do
               taskset -c 0 make pyprof PRESET=cuda-release \
                   PYTHON="$PWD/.venv-cuda/bin/python" \
                   PYPROF_SUITE="cuda-$storage-f64-v1" PYPROF_THREADS=1 \
                   PYPROF_SCOPE="$scope" PYPROF_ROUNDS=30 \
                   PYPROF_REPETITIONS="$repetitions" \
                   PYPROF_OUTPUT="build/benchmarks/cuda-$storage-$preconditioner-$scope.json" \
                   PYPROF_ARGS="--device cuda --cuda-device 0 --implementations $candidates --preconditioner $preconditioner --warmups 3"
           done
       done
   done

New JSON contains input hashes, device/build identity, actual scheduling
and raw samples under ``build/``. :doc:`methodology` shows local summary and
replay commands. A smoke replay checks this plumbing; it is kept separate
from formal performance statistics.


Explicit execution-mode measurements
------------------------------------

The historical tables above use host scalar control. To collect new resident
samples for both native execution modes, use a separate output for each mode:

.. code-block:: bash

   for mode in host device; do
       make pyprof PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" \
           PYPROF_SUITE=smoke-cuda-dense PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
           PYPROF_OUTPUT="build/benchmarks/cuda-$mode-smoke.json" \
           PYPROF_ARGS="--device cuda --cuda-execution-mode $mode --implementations cuda-native,cuda-cublas,torch-cuda-eager"
   done

This smoke command verifies the selected mode and comparison plumbing; it
does not establish a performance ranking. Select ``cuda-dense-f64-v1`` or
``cuda-csr-f64-v1`` for the full fixtures. CSR candidates are
``cuda-native,torch-cuda-eager``. Torch keeps its eager reference control in
both runs; only the native candidate changes execution mode. See
:doc:`methodology` for timing, accuracy and independent-run requirements.
