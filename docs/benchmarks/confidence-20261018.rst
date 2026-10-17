Same-code process calibration, 2026-10-18
=========================================

The fixed same-code calibration completed all seven configurations and retained
112 fresh-worker artifacts, 1,408 eligible candidate/case outcomes and 88 primary
comparison rows. Of those rows, 33 supported no material regression within the
original 5% budget, 55 were inconclusive and none had a lower confidence bound
above 1.05. All four campaign-level verdicts remained ``inconclusive``.

This measures the procedure's ability to distinguish two aliases of the same
build on this host. It does not measure an optimization, establish a speedup,
or establish that the host will remain stable. In particular, zero strict
regressions does not turn the 55 inconclusive rows into passes. The A/B labels
below are baseline/current aliases of identical payloads; a NumPy row compares
NumPy with itself, and a PyTorch row compares PyTorch with itself.

The earlier :doc:`main-complete-20261017` framework baseline and
:doc:`main-optimization-20261017` rejected trials retain their original sources,
protocols, outcomes and URLs. These observations are a separate calibration;
no earlier trial was replaced, combined with this one or corrected using a
framework control.

Protocol and sources
--------------------

The solver packages were compiled from commit
``84c88c98c895103395e1b6d6e98f7618848a14d3``, tree
``57a755d537e516b91c70c1b555f2803753389862``. Within each preset,
``baseline`` and ``current`` refer to the exact same frozen package root and
file manifest, including the native extension. Presets have separate builds;
they are not aliases of each other.

The measured orchestration source is commit
``a4655b94c0893e6b5f68a102ea4730292baab074``, tree
``e8f521211627ea1b2798ceef50cc48ff25b3bac1``. Its frozen runner SHA256 is
``ca9844ab88a5fe99c8a05cc1403786e2492ee746bbf43aa78d0c47831ae7c654``.
A later failure-cleanup repair has separate source and validation; it does not
change the successful launch commands used here. These measurements must not
be relabeled as measurements of that later runner revision.

Each of two time windows runs ABBA followed by BAAB for every configuration.
Window 2 reverses configuration order. Every slot launches the public
``python -m cgbench run`` interface with a fresh timing worker. This gives eight
artifact blocks per side, each with three warmups, 36 complete Williams rounds
and one invocation per round: 288 normalized timing rounds per side and row.
The windows have a declared 60-second pause; that pause does not establish
thermal equilibrium or independence of the physical host.

Fixture seed is 20261017. Every configuration uses float64, ``rtol=1e-8``,
``atol=1e-12``, ``maxiter=2000`` and ``residual_refresh=50``. Ordered cases are
``dense-64``, ``dense-256``, ``dense-1024`` and ``dense-2048`` from the shared
``cpu-dense-f64-v1`` / ``cuda-dense-f64-v1`` fixtures. Input hashes, actual
iterations, matvecs and known restarts match across all same-candidate A/B
blocks. Every retained outcome passed the independent true-residual gate.
No work normalization, row exclusions, selective retries or control-drift
subtraction/division was applied.

The original, unchanged comparator has SHA256
``b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063``.
It uses 10,000 artifact-then-round hierarchical bootstrap samples, seed
20261018, inclusive percentile 95% intervals and the fixed 5% budget. An upper
bound at most 1.05 supports no material regression; a lower bound above 1.05
indicates regression; overlap is inconclusive. Every native and unchanged
framework control participates in the same primary decision. A supported row
is not evidence of a statistically established speedup.

.. list-table::
   :header-rows: 1
   :widths: 5 15 10 12 10 8 12 12

   * - ID
     - Preset
     - Scope
     - Algorithm
     - M
     - Fusion
     - Native mode
     - Threads / logical CPUs
   * - C1
     - cpu-release
     - prepared
     - standard
     - none
     - on
     - host
     - 1 / 0
   * - C2
     - cpu-blas
     - prepared
     - explicit
     - none
     - off
     - host
     - 1 / 0
   * - C3
     - cpu-openmp
     - prepared
     - standard
     - jacobi
     - on
     - host
     - 2 / 0,2
   * - C4
     - cuda-release
     - api
     - standard
     - jacobi
     - on
     - host
     - 1 / 0
   * - C5
     - cuda-release
     - api
     - standard
     - jacobi
     - on
     - device
     - 1 / 0
   * - C6
     - cuda-release
     - session
     - standard
     - jacobi
     - on
     - host
     - 1 / 0
   * - C7
     - cuda-release
     - prepared
     - standard
     - jacobi
     - on
     - device
     - 1 / 0

C1 and C3 retain ``native,numpy-out,torch-eager-out``. C2 retains
``native,native-blas,numpy-out,torch-eager-out``. C4--C7 retain
``cuda-native,cuda-cublas,torch-cuda-eager``. The native CUDA mode selection
is recorded as inapplicable to the eager PyTorch reference; host/device do
not select a different PyTorch engine. There is no CPU Session measurement.

Fresh CUDA API includes output, workspace and per-call Jacobi setup; resident
input uploads are outside the timer. Session reuses the iterative workspace
while charging fresh output and per-call Jacobi. Prepared reuses output,
workspace and preconditioner; each measured invocation still resets x0.
Completed-wall fences remain in the declared timing protocol, while correctness
downloads remain outside the measured intervals. Framework allocator/handle
caches keep their normal behavior; fresh API does not mean a cold framework
runtime on every solve. No hidden native reuse or AUTO selection is introduced.

Observed environment and limitations
------------------------------------

The host reports an Intel Core i7-13700K, 24 logical CPUs and
Linux 6.6.87.2 under WSL2. Python is 3.12.3, NumPy 2.2.6, SciPy 1.15.3 and
threadpoolctl 3.6.0. CPU campaigns use PyTorch 2.8.0+cpu; the CUDA campaign
uses PyTorch 2.8.0+cu128. CPU native builds report GNU 16.0.1; the CUDA host
build reports GNU 13.3.0 and NVIDIA CUDA compiler 12.8.93, architecture 75.
Native Release flags are ``-O3 -DNDEBUG`` without fast math or host ISA tuning.
The explicit BLAS build uses the private, sequential LP64 OpenBLAS 0.3.30
provider with ``cgpy_`` symbols; NumPy reports its separate OpenBLAS 0.3.29
provider. Provider/build inventories remain in the raw artifacts.

The GPU is an NVIDIA GeForce RTX 2060, compute capability 7.5, UUID
``GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302``. The driver reports 610.88;
CUDA runtime is 12080 and driver API version 13030. CUDA controls are explicitly
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and no ``CUDA_LAUNCH_BLOCKING``. All six
thread-pool environment variables, observed pool counts, Torch intraop/interop
counts and logical CPU affinity were checked. ``OMP_DYNAMIC`` and
``MKL_DYNAMIC`` are false. Logical CPU affinity does not identify isolated
physical cores or exclude Windows activity.

Boundary telemetry records UTC and monotonic timestamps, Linux load/stat/CPU
pressure, available sensors and read-only ``nvidia-smi`` output before and
after each child, outside solve timers. CPU sysfs frequency and temperature
sensors are unavailable in these captures. ``/proc/cpuinfo`` MHz strings do
not establish the physical CPU clock during a solve. GPU query values are
boundary observations, not a continuous account of clocks, utilization,
thermal state or another application's activity. Telemetry is diagnostic;
its own driver queries can perturb the system.

One additional manual read-only GPU utilization/temperature query occurred
during the CUDA calibration. It reported ``23, 50``. The last completed slot
observed before that query was 43; the latest slot recorded at disclosure was
60. Exact query start/end timestamps were not instrumented. The event
was recorded at ``2026-10-17T16:48:14.175741+00:00`` in
``additional-observer-query.json``; that later recording time is not its
execution timestamp. Every row and block remains present, without exclusions,
correction, rerun or pooling. This extra query is not evidence of an isolated
or continuously monitored host.

Eight fresh artifact blocks per side provide more process observations than
the earlier two-block trials, but only two declared time windows were observed.
Fresh processes do not prove independent environmental states. The unchanged
primary comparator resamples rounds independently within each artifact;
serial dependence can make intervals too narrow. Intervals are per row,
without simultaneous-coverage adjustment. WSL does not isolate Windows host
load, display activity, power management or thermal behavior. These limits
apply even where the 5% gate is supported. This calibration does not quantify
an improvement in false-regression rate against earlier trials from another
time window, and cannot attribute a timing shift to a specific disturbance.

Before calibration, an initial CPU setup test failed because the inherited
``CG_REQUIRE_CUDA=1`` requirement was applied to a CPU-only build. Its original
log and summary remain unchanged. The corrected setup cleared that requirement
for CPU presets and enabled it only for CUDA. All four corrected build/test
commands succeeded before timing began. This was a setup correction, not a
retrial or exclusion of a started measurement campaign.

Complete primary outcomes
-------------------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Configs
     - Artifacts
     - Eligible outcomes
     - Rows
     - Within budget
     - Inconclusive
     - Regression
   * - cpu-release
     - 1
     - 16
     - 192
     - 12
     - 0
     - 12
     - 0
   * - cpu-blas
     - 1
     - 16
     - 256
     - 16
     - 5
     - 11
     - 0
   * - cpu-openmp
     - 1
     - 16
     - 192
     - 12
     - 3
     - 9
     - 0
   * - cuda-release
     - 4
     - 64
     - 768
     - 48
     - 25
     - 23
     - 0
   * - Total
     - 7
     - 112
     - 1408
     - 88
     - 33
     - 55
     - 0

All four campaign manifests have ``status=complete`` and
``verdict=inconclusive``. Their command exit status is 1 because inconclusive
is not a supported pass; it is not a worker crash or numerical failure.
The observations were collected between 2026-10-18 00:28:35 and 00:48:50 in
Asia/Taipei (2026-10-17 16:28:35--16:48:50 UTC).

In the following tables, A/B milliseconds are pooled medians over all 288
normalized rounds on that side, not medians of the eight process medians.
``W`` means ``supported_no_material_regression`` and ``I`` means
``inconclusive``. ``I/M/R`` gives the verified iteration/matvec/restart counts
for that same candidate on both aliases. B/A is the comparator's original
ratio of pooled medians. Displayed decimal values are rounded; the reports
retain full JSON precision.

C1 -- cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - native
     - 0.036478500
     - 0.036458500
     - 0.999451732
     - [0.857970870, 1.176264066]
     - I
     - 49/51/0
   * - dense-64
     - numpy-out
     - 0.247816500
     - 0.253927000
     - 1.024657357
     - [0.953013705, 1.110543751]
     - I
     - 49/51/0
   * - dense-64
     - torch-eager-out
     - 0.413731500
     - 0.440459500
     - 1.064602284
     - [0.934780557, 1.200728111]
     - I
     - 49/51/0
   * - dense-256
     - native
     - 0.441359000
     - 0.461859500
     - 1.046448583
     - [0.957039092, 1.147097782]
     - I
     - 67/70/1
   * - dense-256
     - numpy-out
     - 0.754586500
     - 0.799749000
     - 1.059850660
     - [0.935422520, 1.320781649]
     - I
     - 67/70/1
   * - dense-256
     - torch-eager-out
     - 0.902173000
     - 1.030391000
     - 1.142121300
     - [0.976227476, 1.364493844]
     - I
     - 67/70/1
   * - dense-1024
     - native
     - 10.877847500
     - 10.986021000
     - 1.009944385
     - [0.954194925, 1.062288370]
     - I
     - 70/73/1
   * - dense-1024
     - numpy-out
     - 11.584009500
     - 11.688656000
     - 1.009033703
     - [0.968076061, 1.059169991]
     - I
     - 70/73/1
   * - dense-1024
     - torch-eager-out
     - 12.745391000
     - 12.844314000
     - 1.007761472
     - [0.962674901, 1.062781969]
     - I
     - 70/73/1
   * - dense-2048
     - native
     - 74.685592000
     - 76.539489500
     - 1.024822693
     - [0.971689800, 1.093725354]
     - I
     - 70/73/1
   * - dense-2048
     - numpy-out
     - 61.558876500
     - 62.171188000
     - 1.009946762
     - [0.971609721, 1.100218654]
     - I
     - 70/73/1
   * - dense-2048
     - torch-eager-out
     - 65.439674000
     - 66.826310000
     - 1.021189531
     - [0.985161210, 1.091859318]
     - I
     - 70/73/1

C2 -- cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - native
     - 0.060915500
     - 0.060741500
     - 0.997143584
     - [0.833278434, 1.035042225]
     - W
     - 49/99/0
   * - dense-64
     - native-blas
     - 0.046028000
     - 0.044047000
     - 0.956960980
     - [0.784658332, 1.070160933]
     - I
     - 50/101/0
   * - dense-64
     - numpy-out
     - 0.365628500
     - 0.342559500
     - 0.936905903
     - [0.795059364, 1.067032071]
     - I
     - 50/101/0
   * - dense-64
     - torch-eager-out
     - 0.612165500
     - 0.587366500
     - 0.959489713
     - [0.765387188, 1.213168078]
     - I
     - 49/99/0
   * - dense-256
     - native
     - 0.832015000
     - 0.830772000
     - 0.998506037
     - [0.932858785, 1.051655260]
     - I
     - 65/131/0
   * - dense-256
     - native-blas
     - 0.591559000
     - 0.587602000
     - 0.993310895
     - [0.808875717, 1.095287667]
     - I
     - 65/131/0
   * - dense-256
     - numpy-out
     - 1.180804000
     - 1.194225000
     - 1.011365985
     - [0.914962405, 1.149700282]
     - I
     - 65/131/0
   * - dense-256
     - torch-eager-out
     - 1.512979000
     - 1.514156500
     - 1.000778266
     - [0.839536306, 1.124478410]
     - I
     - 65/131/0
   * - dense-1024
     - native
     - 20.453768000
     - 20.201086000
     - 0.987646188
     - [0.950111738, 1.017758486]
     - W
     - 69/139/0
   * - dense-1024
     - native-blas
     - 19.224594500
     - 19.137412500
     - 0.995465080
     - [0.943557892, 1.039939604]
     - W
     - 69/139/0
   * - dense-1024
     - numpy-out
     - 20.989331500
     - 20.834847500
     - 0.992639880
     - [0.943212008, 1.032535597]
     - W
     - 69/139/0
   * - dense-1024
     - torch-eager-out
     - 23.203545500
     - 23.041773000
     - 0.993028113
     - [0.943142955, 1.034957393]
     - W
     - 69/139/0
   * - dense-2048
     - native
     - 148.032970500
     - 151.788187000
     - 1.025367433
     - [0.925235561, 1.144349838]
     - I
     - 68/137/0
   * - dense-2048
     - native-blas
     - 114.550510500
     - 118.164948500
     - 1.031553225
     - [0.914674573, 1.165967152]
     - I
     - 68/137/0
   * - dense-2048
     - numpy-out
     - 117.635083000
     - 121.202122500
     - 1.030322922
     - [0.891473878, 1.181774149]
     - I
     - 68/137/0
   * - dense-2048
     - torch-eager-out
     - 124.830293000
     - 130.597615500
     - 1.046201305
     - [0.919222680, 1.203120045]
     - I
     - 68/137/0

C3 -- cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - native
     - 0.010440500
     - 0.010346500
     - 0.990996600
     - [0.961697969, 1.020191083]
     - W
     - 5/7/0
   * - dense-64
     - numpy-out
     - 0.029012000
     - 0.028550000
     - 0.984075555
     - [0.960963892, 1.017699023]
     - W
     - 5/7/0
   * - dense-64
     - torch-eager-out
     - 0.044832500
     - 0.045174000
     - 1.007617242
     - [0.969232652, 1.049710303]
     - W
     - 5/7/0
   * - dense-256
     - native
     - 0.038686500
     - 0.033943500
     - 0.877399093
     - [0.706015027, 1.174343343]
     - I
     - 4/6/0
   * - dense-256
     - numpy-out
     - 0.063162500
     - 0.058882000
     - 0.932230358
     - [0.654672503, 1.327264641]
     - I
     - 4/6/0
   * - dense-256
     - torch-eager-out
     - 0.090306500
     - 0.082682000
     - 0.915570861
     - [0.590233346, 1.364876785]
     - I
     - 4/6/0
   * - dense-1024
     - native
     - 1.088244000
     - 0.971866000
     - 0.893058910
     - [0.656242819, 1.295332512]
     - I
     - 4/6/0
   * - dense-1024
     - numpy-out
     - 1.397346500
     - 1.452268500
     - 1.039304496
     - [0.740325484, 1.545161471]
     - I
     - 4/6/0
   * - dense-1024
     - torch-eager-out
     - 1.327797500
     - 1.153067500
     - 0.868406139
     - [0.681905432, 1.154333336]
     - I
     - 4/6/0
   * - dense-2048
     - native
     - 7.245700500
     - 7.402156500
     - 1.021592943
     - [0.951930773, 1.109022282]
     - I
     - 3/5/0
   * - dense-2048
     - numpy-out
     - 6.326082500
     - 6.198597000
     - 0.979847639
     - [0.779014883, 1.067923799]
     - I
     - 3/5/0
   * - dense-2048
     - torch-eager-out
     - 7.256162000
     - 7.371341000
     - 1.015873267
     - [0.952578214, 1.094900050]
     - I
     - 3/5/0

C4 -- cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - cuda-native
     - 0.973382500
     - 0.970757000
     - 0.997302705
     - [0.959199197, 1.029853624]
     - W
     - 5/7/0
   * - dense-64
     - cuda-cublas
     - 2.599223000
     - 2.552911000
     - 0.982182368
     - [0.924884842, 1.013225501]
     - W
     - 5/7/0
   * - dense-64
     - torch-cuda-eager
     - 2.075354000
     - 2.041694500
     - 0.983781321
     - [0.942057862, 1.038661441]
     - W
     - 5/7/0
   * - dense-256
     - cuda-native
     - 0.871065500
     - 0.876715500
     - 1.006486309
     - [0.974348428, 1.031659770]
     - W
     - 4/6/0
   * - dense-256
     - cuda-cublas
     - 2.494266500
     - 2.505634000
     - 1.004557452
     - [0.980173710, 1.028987772]
     - W
     - 4/6/0
   * - dense-256
     - torch-cuda-eager
     - 1.787447000
     - 1.818148000
     - 1.017175894
     - [0.973617948, 1.061824775]
     - I
     - 4/6/0
   * - dense-1024
     - cuda-native
     - 1.156404000
     - 1.167750000
     - 1.009811450
     - [0.991715451, 1.028847427]
     - W
     - 4/6/0
   * - dense-1024
     - cuda-cublas
     - 2.681247500
     - 2.722697000
     - 1.015459035
     - [0.988296020, 1.036793228]
     - W
     - 4/6/0
   * - dense-1024
     - torch-cuda-eager
     - 1.825714500
     - 1.837918000
     - 1.006684232
     - [0.968517482, 1.054593297]
     - I
     - 4/6/0
   * - dense-2048
     - cuda-native
     - 1.480894500
     - 1.492668000
     - 1.007950262
     - [0.986748532, 1.027180139]
     - W
     - 3/5/0
   * - dense-2048
     - cuda-cublas
     - 2.932352500
     - 2.931416500
     - 0.999680802
     - [0.980069469, 1.016471723]
     - W
     - 3/5/0
   * - dense-2048
     - torch-cuda-eager
     - 2.017374000
     - 2.009443500
     - 0.996068899
     - [0.951776726, 1.023582800]
     - W
     - 3/5/0

C5 -- cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - cuda-native
     - 0.700862000
     - 0.688858500
     - 0.982873233
     - [0.937871677, 1.026712615]
     - W
     - 5/7/0
   * - dense-64
     - cuda-cublas
     - 2.334348500
     - 2.336503500
     - 1.000923170
     - [0.956361980, 1.061953464]
     - I
     - 5/7/0
   * - dense-64
     - torch-cuda-eager
     - 2.055879500
     - 2.068202500
     - 1.005994028
     - [0.948864842, 1.059520854]
     - I
     - 5/7/0
   * - dense-256
     - cuda-native
     - 0.672045000
     - 0.676133000
     - 1.006082926
     - [0.961887822, 1.050206816]
     - I
     - 4/6/0
   * - dense-256
     - cuda-cublas
     - 2.272285500
     - 2.299828000
     - 1.012121056
     - [0.974395476, 1.057859328]
     - I
     - 4/6/0
   * - dense-256
     - torch-cuda-eager
     - 1.778866000
     - 1.774317500
     - 0.997443034
     - [0.956262134, 1.044812401]
     - W
     - 4/6/0
   * - dense-1024
     - cuda-native
     - 0.958423000
     - 0.964118000
     - 1.005942053
     - [0.964407802, 1.054036065]
     - I
     - 4/6/0
   * - dense-1024
     - cuda-cublas
     - 2.459574500
     - 2.466921500
     - 1.002987102
     - [0.967266373, 1.049748515]
     - W
     - 4/6/0
   * - dense-1024
     - torch-cuda-eager
     - 1.832807000
     - 1.823488000
     - 0.994915449
     - [0.953456637, 1.037041800]
     - W
     - 4/6/0
   * - dense-2048
     - cuda-native
     - 1.310687000
     - 1.299524500
     - 0.991483474
     - [0.932765678, 1.051592577]
     - I
     - 3/5/0
   * - dense-2048
     - cuda-cublas
     - 2.693607000
     - 2.693731000
     - 1.000046035
     - [0.940558067, 1.044620375]
     - W
     - 3/5/0
   * - dense-2048
     - torch-cuda-eager
     - 1.992150500
     - 1.972030500
     - 0.989900361
     - [0.950965101, 1.028961607]
     - W
     - 3/5/0

C6 -- cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - cuda-native
     - 1.055452000
     - 1.051959000
     - 0.996690517
     - [0.937398389, 1.054995893]
     - I
     - 5/7/0
   * - dense-64
     - cuda-cublas
     - 1.090300500
     - 1.067735500
     - 0.979303871
     - [0.915756349, 1.048680615]
     - W
     - 5/7/0
   * - dense-64
     - torch-cuda-eager
     - 2.301979000
     - 2.324570500
     - 1.009813947
     - [0.944174286, 1.092038733]
     - I
     - 5/7/0
   * - dense-256
     - cuda-native
     - 0.962850000
     - 0.947361500
     - 0.983913901
     - [0.944722994, 1.021418662]
     - W
     - 4/6/0
   * - dense-256
     - cuda-cublas
     - 0.952832500
     - 0.957803000
     - 1.005216552
     - [0.947626180, 1.035667109]
     - W
     - 4/6/0
   * - dense-256
     - torch-cuda-eager
     - 2.071974500
     - 2.080366000
     - 1.004050002
     - [0.943678302, 1.062849749]
     - I
     - 4/6/0
   * - dense-1024
     - cuda-native
     - 1.231823500
     - 1.232015500
     - 1.000155866
     - [0.937151470, 1.059762990]
     - I
     - 4/6/0
   * - dense-1024
     - cuda-cublas
     - 1.129131500
     - 1.138191000
     - 1.008023423
     - [0.952470843, 1.061290913]
     - I
     - 4/6/0
   * - dense-1024
     - torch-cuda-eager
     - 2.041687000
     - 2.042009000
     - 1.000157713
     - [0.918963897, 1.109550688]
     - I
     - 4/6/0
   * - dense-2048
     - cuda-native
     - 1.590005500
     - 1.544001000
     - 0.971066452
     - [0.934676822, 1.017661536]
     - W
     - 3/5/0
   * - dense-2048
     - cuda-cublas
     - 1.420603500
     - 1.382065500
     - 0.972872093
     - [0.949243406, 1.017365711]
     - W
     - 3/5/0
   * - dense-2048
     - torch-cuda-eager
     - 2.188302500
     - 2.124503000
     - 0.970845210
     - [0.924256285, 1.041721268]
     - W
     - 3/5/0

C7 -- cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 9 16 12 12 10 20 7 8

   * - Case
     - Candidate
     - A ms
     - B ms
     - B/A
     - 95% interval
     - Verdict
     - I/M/R
   * - dense-64
     - cuda-native
     - 0.686690500
     - 0.684183500
     - 0.996349156
     - [0.900202610, 1.110989711]
     - I
     - 5/7/0
   * - dense-64
     - cuda-cublas
     - 0.752553000
     - 0.735382000
     - 0.977183002
     - [0.892936367, 1.073842090]
     - I
     - 5/7/0
   * - dense-64
     - torch-cuda-eager
     - 2.075066000
     - 2.012004000
     - 0.969609641
     - [0.862456567, 1.084288885]
     - I
     - 5/7/0
   * - dense-256
     - cuda-native
     - 0.635981000
     - 0.644305000
     - 1.013088441
     - [0.961525500, 1.074139747]
     - I
     - 4/6/0
   * - dense-256
     - cuda-cublas
     - 0.653569500
     - 0.652732500
     - 0.998719340
     - [0.934631061, 1.068020189]
     - I
     - 4/6/0
   * - dense-256
     - torch-cuda-eager
     - 1.712591500
     - 1.712064000
     - 0.999691987
     - [0.942103080, 1.081600347]
     - I
     - 4/6/0
   * - dense-1024
     - cuda-native
     - 0.929282000
     - 0.920169000
     - 0.990193504
     - [0.878957070, 1.034421628]
     - W
     - 4/6/0
   * - dense-1024
     - cuda-cublas
     - 0.809870000
     - 0.795822000
     - 0.982654006
     - [0.858664074, 1.037646781]
     - W
     - 4/6/0
   * - dense-1024
     - torch-cuda-eager
     - 1.799256500
     - 1.710087500
     - 0.950441196
     - [0.785608551, 1.075179618]
     - I
     - 4/6/0
   * - dense-2048
     - cuda-native
     - 1.320069500
     - 1.328411500
     - 1.006319364
     - [0.982908623, 1.042312209]
     - W
     - 3/5/0
   * - dense-2048
     - cuda-cublas
     - 1.151646000
     - 1.149937500
     - 0.998516471
     - [0.971460097, 1.056480319]
     - I
     - 3/5/0
   * - dense-2048
     - torch-cuda-eager
     - 1.843685500
     - 1.878265500
     - 1.018755910
     - [0.950309287, 1.108193191]
     - I
     - 3/5/0

Eight process medians per side
------------------------------

The eight values in each cell are process-block medians in milliseconds,
ordered exactly as the comparator's ``baseline_files`` or ``current_files``
array. For either side the first four belong to window 1 and the last four
to window 2; within a window they retain actual ABBA then BAAB slot order.
These are descriptive process observations, not independent whole-host
replications or a replacement estimator for the primary decision.

C1 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - native
     - 0.034863000, 0.035210500, 0.039969500, 0.033544500, 0.036896000, 0.028922500, 0.040526500, 0.044745500
     - 0.039333000, 0.032091500, 0.035785000, 0.037543500, 0.035688000, 0.030919500, 0.037484500, 0.036767000
   * - dense-64
     - numpy-out
     - 0.246572500, 0.243467500, 0.268640500, 0.238535500, 0.257019000, 0.200451500, 0.272417000, 0.297465500
     - 0.321841500, 0.244299000, 0.241008000, 0.257795000, 0.269883000, 0.246159000, 0.258627000, 0.256960000
   * - dense-64
     - torch-eager-out
     - 0.404574000, 0.416888000, 0.454255000, 0.360999500, 0.432832000, 0.305040000, 0.535784500, 0.578903000
     - 0.556850500, 0.416606500, 0.410318000, 0.436422000, 0.446673000, 0.425699000, 0.450299500, 0.435097500
   * - dense-256
     - native
     - 0.450488000, 0.437183500, 0.463028000, 0.418411000, 0.458039000, 0.379728000, 0.560662500, 0.531689000
     - 0.556308500, 0.445460500, 0.478523500, 0.463751500, 0.475161000, 0.434665000, 0.453568000, 0.445390500
   * - dense-256
     - numpy-out
     - 0.756385000, 0.766068000, 0.840736500, 0.627218000, 0.876137000, 0.519323500, 0.880085500, 0.910001000
     - 0.796256500, 0.786690500, 0.721878500, 0.865352500, 0.717317000, 0.787590000, 0.850203000, 0.825656500
   * - dense-256
     - torch-eager-out
     - 0.903430000, 0.847167000, 1.076970500, 0.808805000, 0.994489000, 0.644816000, 1.168479500, 1.274132500
     - 1.253483500, 0.959984500, 1.005450000, 1.010615500, 1.024958500, 1.044639000, 0.996766500, 1.039443000
   * - dense-1024
     - native
     - 10.611854500, 11.900283500, 10.535237000, 10.662038000, 10.778511500, 10.864079500, 11.459580000, 10.452886000
     - 11.983497000, 10.901240500, 11.343744500, 11.577150500, 11.296724500, 10.245993500, 10.308624500, 10.315050500
   * - dense-1024
     - numpy-out
     - 11.738073000, 12.361164500, 11.576706000, 11.208671500, 11.396704000, 11.080290000, 11.707361000, 11.391166000
     - 12.590694500, 11.440718000, 11.789707000, 12.494155000, 12.461959500, 11.290409500, 11.105924000, 11.168976500
   * - dense-1024
     - torch-eager-out
     - 13.094442000, 13.721380500, 12.449621500, 12.031225500, 12.802075000, 12.399951000, 13.401078500, 12.439695000
     - 14.039811500, 13.647669500, 12.876563000, 13.522929000, 13.116406500, 12.241545000, 12.194187500, 12.220482500
   * - dense-2048
     - native
     - 78.086862000, 72.193961500, 77.527322000, 75.435834000, 74.000481000, 74.495811500, 74.943425500, 70.459720500
     - 96.208170000, 77.903885500, 81.018674000, 73.159933500, 72.095060000, 76.715016500, 71.795599000, 80.053109500
   * - dense-2048
     - numpy-out
     - 66.081534000, 60.454254000, 64.414973000, 59.438171000, 62.246008500, 59.620978000, 62.627675000, 61.454071000
     - 78.940696500, 63.606289500, 60.289337500, 61.941228000, 60.292884500, 61.888964500, 60.662190500, 62.981402500
   * - dense-2048
     - torch-eager-out
     - 68.309388500, 69.527296500, 71.377085000, 64.139753000, 66.375117500, 65.565928500, 63.925974000, 64.815065000
     - 91.092626500, 68.107557500, 67.788995500, 65.213681000, 64.954951000, 67.455641500, 65.763718500, 69.265656500

C2 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - native
     - 0.052457500, 0.060795500, 0.062797000, 0.064647000, 0.059932000, 0.059742000, 0.062656500, 0.062066000
     - 0.062592000, 0.048838000, 0.062016500, 0.061497000, 0.062031000, 0.061875000, 0.049893000, 0.063677000
   * - dense-64
     - native-blas
     - 0.040221500, 0.045990500, 0.041195500, 0.049693000, 0.045913000, 0.045085500, 0.048376500, 0.046600500
     - 0.047200000, 0.035205000, 0.046908000, 0.048127000, 0.054950000, 0.047343500, 0.034457500, 0.039453000
   * - dense-64
     - numpy-out
     - 0.303790000, 0.347884500, 0.341876500, 0.389832500, 0.373756000, 0.337599500, 0.387809500, 0.378109500
     - 0.365808500, 0.270206500, 0.388960000, 0.378348000, 0.380363000, 0.376568500, 0.297512500, 0.350238000
   * - dense-64
     - torch-eager-out
     - 0.523617000, 0.577233000, 0.675729500, 0.652441000, 0.641911500, 0.513616000, 0.697876000, 0.632148500
     - 0.665561000, 0.406075000, 0.689107000, 0.671817500, 0.693466500, 0.714361000, 0.501410000, 0.559348500
   * - dense-256
     - native
     - 0.805921000, 0.835670000, 0.830871000, 0.895662000, 0.812581500, 0.816715000, 0.850265500, 0.851611000
     - 0.844728000, 0.730272000, 0.830772000, 0.850372500, 0.878237000, 0.832477500, 0.774864000, 0.927911500
   * - dense-256
     - native-blas
     - 0.535125000, 0.748705000, 0.547132500, 0.591166500, 0.583421500, 0.549045500, 0.575121500, 0.806010000
     - 0.586864000, 0.478805500, 0.525086500, 0.755438000, 0.601404500, 0.590517500, 0.525746000, 0.630983000
   * - dense-256
     - numpy-out
     - 1.143487500, 1.105344500, 1.192008500, 1.377552000, 1.220759500, 1.181670500, 1.229261500, 1.087061000
     - 1.292224500, 1.009186500, 1.201823500, 1.107744500, 1.287730000, 1.305933000, 1.089487500, 1.447479500
   * - dense-256
     - torch-eager-out
     - 1.347873500, 1.648238500, 1.739750500, 1.542157000, 1.533026000, 1.420811000, 1.613581000, 1.579182000
     - 1.622044500, 1.041704000, 1.565463000, 1.592122500, 1.696052500, 1.405526000, 1.273283000, 1.775769500
   * - dense-1024
     - native
     - 20.010429500, 20.277092000, 21.126848500, 23.511977500, 20.714939000, 20.429846000, 20.323844000, 19.476441000
     - 20.486658500, 20.727956500, 20.144060500, 21.555562000, 20.475919000, 18.777469500, 19.599590000, 19.716078500
   * - dense-1024
     - native-blas
     - 19.046026000, 18.702500000, 20.454149500, 21.555899000, 19.457936000, 19.531430000, 18.864539000, 17.614002000
     - 19.597870500, 19.683844500, 19.335895000, 19.941817500, 19.542984000, 17.910844500, 18.502737500, 19.077455000
   * - dense-1024
     - numpy-out
     - 20.226633500, 20.532795500, 21.753460500, 23.860933000, 21.688469500, 20.777657000, 20.185352000, 20.503456500
     - 21.005725500, 21.223596500, 20.448095500, 21.894136000, 22.023093000, 19.681422500, 19.982001000, 20.632702500
   * - dense-1024
     - torch-eager-out
     - 21.795185000, 23.166457500, 23.982652500, 26.824400000, 24.039930500, 23.268672500, 22.568099500, 21.965041000
     - 23.274487000, 23.199040000, 23.030598000, 24.684153500, 23.427295000, 21.375570000, 22.319379000, 23.197778000
   * - dense-2048
     - native
     - 148.001566000, 147.432839500, 139.506271500, 142.186951000, 185.588234000, 176.226529500, 143.792315000, 137.086221500
     - 143.142068000, 146.659050500, 143.030876000, 152.773396500, 161.842878500, 212.869596500, 147.181861000, 140.529140500
   * - dense-2048
     - native-blas
     - 111.019615500, 113.300648000, 113.790834500, 111.775844000, 143.583941000, 137.015266000, 117.297930000, 108.982142000
     - 112.228563500, 116.887367500, 112.403745500, 122.703934500, 134.110912500, 174.987923000, 108.170463000, 106.487269500
   * - dense-2048
     - numpy-out
     - 114.614108000, 116.103846500, 115.265909500, 115.941028000, 146.726132500, 145.748993000, 119.685960500, 114.987416500
     - 111.229478000, 122.023513500, 115.026734500, 121.334125500, 134.574519500, 183.147272000, 116.313342000, 114.874081000
   * - dense-2048
     - torch-eager-out
     - 120.556644500, 123.634555500, 122.694030000, 121.781264500, 168.438647500, 150.415628500, 119.204039000, 121.784406500
     - 124.029810500, 131.368845000, 124.004652000, 127.956085000, 149.462738000, 186.075534500, 126.793812500, 122.240494000

C3 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - native
     - 0.011725000, 0.010162000, 0.010372000, 0.010030500, 0.010595000, 0.010435000, 0.010422500, 0.010442000
     - 0.010549500, 0.010417000, 0.010401500, 0.010244000, 0.010181500, 0.010080500, 0.010293000, 0.011429500
   * - dense-64
     - numpy-out
     - 0.032217500, 0.028953500, 0.028563000, 0.027921500, 0.028783500, 0.028607500, 0.029028500, 0.029040000
     - 0.028976000, 0.029176000, 0.028516500, 0.028280000, 0.028037500, 0.028477500, 0.028174000, 0.030966500
   * - dense-64
     - torch-eager-out
     - 0.049517500, 0.044644000, 0.044625000, 0.043959000, 0.044992500, 0.043547000, 0.043965000, 0.045521000
     - 0.043990500, 0.046891500, 0.044528000, 0.044772000, 0.043387000, 0.044567000, 0.044626000, 0.049738000
   * - dense-256
     - native
     - 0.032212500, 0.030298500, 0.639454500, 0.045252500, 0.093240500, 0.040354000, 0.032674500, 0.037433000
     - 0.045586500, 0.031061500, 0.034607500, 0.039941000, 0.030278000, 0.043906000, 0.028943500, 0.030499500
   * - dense-256
     - numpy-out
     - 0.051136500, 0.046824000, 0.143120000, 0.080212500, 0.163523500, 0.068906500, 0.062035000, 0.056071500
     - 0.091220500, 0.052049000, 0.071950500, 0.044645500, 0.054875500, 0.089058500, 0.051149500, 0.058215500
   * - dense-256
     - torch-eager-out
     - 0.066494500, 0.061189500, 0.704291000, 0.112444000, 0.212067500, 0.110353500, 0.072216000, 0.071343000
     - 0.116433500, 0.078399000, 0.089643500, 0.077447000, 0.060044500, 0.117491500, 0.078203000, 0.064983000
   * - dense-1024
     - native
     - 1.341747500, 1.256959000, 1.194963000, 0.710565500, 1.210974500, 0.727304000, 0.778420500, 1.165052500
     - 0.761817000, 0.738948000, 0.825430500, 0.997547500, 1.249928000, 0.749993500, 0.778445000, 1.266317500
   * - dense-1024
     - numpy-out
     - 1.276987500, 1.273802500, 1.187841000, 2.191399000, 2.082795500, 2.036614500, 2.067059500, 1.229934000
     - 2.016046500, 1.889496500, 2.090487500, 1.250801500, 1.200829500, 2.255819500, 2.051360500, 1.267919000
   * - dense-1024
     - torch-eager-out
     - 1.457255500, 1.376946500, 1.483178000, 0.893649500, 1.530526500, 0.927867000, 0.966329500, 1.385263500
     - 0.892529000, 0.872634500, 1.162492500, 1.124062500, 1.342858500, 0.859049000, 0.870773500, 1.518936500
   * - dense-2048
     - native
     - 7.422519500, 8.150682500, 6.335358500, 7.829901000, 6.925826500, 7.003889500, 7.375794000, 7.676340000
     - 7.311218000, 7.079666000, 7.936069000, 7.026778500, 6.829259000, 7.673926000, 7.204788500, 8.351912000
   * - dense-2048
     - numpy-out
     - 6.641674500, 6.374474000, 6.076810000, 5.328452000, 6.085180500, 6.331504500, 6.132924500, 7.062382500
     - 4.927955500, 4.568638500, 6.580330000, 4.885934000, 6.198597000, 6.471636000, 6.134253000, 7.321415500
   * - dense-2048
     - torch-eager-out
     - 7.085855000, 7.086835000, 6.318175500, 7.585919500, 7.461632000, 7.354397500, 7.555504500, 8.034053500
     - 7.181113500, 7.278538500, 7.866048000, 7.476059500, 6.954293500, 7.358854000, 7.291377000, 8.553031500

C4 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - cuda-native
     - 0.976143000, 0.949691500, 0.952035000, 0.991161000, 0.998563000, 0.982062500, 0.975260000, 0.985992000
     - 0.876854500, 0.974678500, 0.905259000, 0.946858500, 1.007061000, 1.008706500, 1.028120500, 0.970945500
   * - dense-64
     - cuda-cublas
     - 2.607696000, 2.542307000, 2.557175500, 2.578921000, 2.664830000, 2.744041000, 2.552429000, 2.642349500
     - 2.344620000, 2.499267000, 2.328487000, 2.403569000, 2.613824500, 2.647318000, 2.622622500, 2.623077500
   * - dense-64
     - torch-cuda-eager
     - 1.980618500, 1.969853500, 1.979269500, 1.959567000, 2.113384000, 2.142670000, 2.081057000, 2.145244000
     - 1.932675500, 2.060033500, 1.926367000, 1.998845500, 2.079747000, 2.132832500, 2.027840500, 2.125937500
   * - dense-256
     - cuda-native
     - 0.851905500, 0.853903500, 0.862256000, 0.866259500, 0.876144000, 0.889931500, 0.897293500, 0.911229500
     - 0.855503000, 0.880720500, 0.860502500, 0.872413000, 0.889819000, 0.874563000, 0.915609500, 0.872218000
   * - dense-256
     - cuda-cublas
     - 2.456307500, 2.448579000, 2.480162500, 2.478660500, 2.467317000, 2.559714500, 2.558647000, 2.580137500
     - 2.415924500, 2.523289000, 2.457862000, 2.461346000, 2.535822500, 2.549416500, 2.566963000, 2.562143500
   * - dense-256
     - torch-cuda-eager
     - 1.676317000, 1.726075500, 1.734398500, 1.742137500, 1.793414500, 1.891293500, 1.786850000, 1.900978500
     - 1.724811000, 1.864139500, 1.739151500, 1.767076500, 1.953556000, 1.812269000, 1.879501000, 1.824911000
   * - dense-1024
     - cuda-native
     - 1.132669000, 1.132016500, 1.125673500, 1.158574500, 1.191535500, 1.191804500, 1.159253000, 1.167353000
     - 1.142580500, 1.166621500, 1.143958000, 1.169321000, 1.156803500, 1.199830500, 1.191224500, 1.189778000
   * - dense-1024
     - cuda-cublas
     - 2.641191000, 2.601588000, 2.649325500, 2.708300000, 2.696074000, 2.777890500, 2.675384000, 2.722685500
     - 2.660499000, 2.755576500, 2.602457000, 2.669799000, 2.780062000, 2.799525500, 2.739350000, 2.765297000
   * - dense-1024
     - torch-cuda-eager
     - 1.731847000, 1.743886000, 1.758826500, 1.891018000, 1.846597500, 1.889727000, 1.808225000, 1.941500500
     - 1.819836000, 1.883837000, 1.763989500, 1.772376500, 1.786962000, 1.950385500, 1.826557000, 1.903889000
   * - dense-2048
     - cuda-native
     - 1.470833000, 1.469142500, 1.502085000, 1.479514500, 1.474236500, 1.507788000, 1.482153000, 1.489214000
     - 1.472334000, 1.492035500, 1.462012500, 1.461981500, 1.488686500, 1.506625000, 1.546110000, 1.523580500
   * - dense-2048
     - cuda-cublas
     - 2.922558000, 2.900387500, 2.908624000, 2.908229500, 2.942451000, 2.974046000, 2.896662000, 3.036700000
     - 2.854246000, 2.920671000, 2.869858000, 2.881927500, 2.913216000, 3.017363000, 2.987138500, 2.978406500
   * - dense-2048
     - torch-cuda-eager
     - 1.983146500, 1.962876500, 1.948925500, 1.932407000, 2.002816000, 2.099613000, 2.104754000, 2.161927000
     - 1.991019500, 2.045032000, 2.007544000, 1.988578000, 1.979927000, 2.005490000, 2.011573000, 2.064135500

C5 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - cuda-native
     - 0.756791000, 0.691030000, 0.657041500, 0.655092000, 0.724902500, 0.658757000, 0.706506000, 0.706460000
     - 0.710017000, 0.701116000, 0.675827500, 0.636075000, 0.698282500, 0.665916000, 0.668224000, 0.720749000
   * - dense-64
     - cuda-cublas
     - 2.511867000, 2.318306500, 2.160590000, 2.078786500, 2.366746000, 2.040850500, 2.391917000, 2.379148500
     - 2.409987500, 2.379011500, 2.317249000, 2.055190000, 2.345526500, 2.172257500, 2.328450500, 2.376427500
   * - dense-64
     - torch-cuda-eager
     - 2.174865000, 1.959278000, 1.975350000, 1.987072000, 2.067515000, 1.993320000, 2.144297000, 2.066993500
     - 2.229091000, 2.018002000, 1.977033500, 1.860167500, 2.068802000, 2.060211500, 2.006243000, 2.187362500
   * - dense-256
     - cuda-native
     - 0.718240500, 0.650980000, 0.654710500, 0.650864500, 0.693693500, 0.675327500, 0.667311000, 0.669096500
     - 0.728553500, 0.663016500, 0.652378500, 0.664527500, 0.663932000, 0.682754500, 0.679008000, 0.679376500
   * - dense-256
     - cuda-cublas
     - 2.470525500, 2.242598500, 2.220877000, 2.206293000, 2.285800000, 2.272715000, 2.285420500, 2.301141500
     - 2.481290000, 2.286401500, 2.245318500, 2.204976000, 2.281471000, 2.357268500, 2.277582500, 2.382239000
   * - dense-256
     - torch-cuda-eager
     - 1.941165500, 1.742722500, 1.730577500, 1.701637500, 1.835770500, 1.776708500, 1.761172000, 1.768981000
     - 1.928669500, 1.716596000, 1.658214000, 1.714363500, 1.777292000, 1.839180000, 1.773860000, 1.782791000
   * - dense-1024
     - cuda-native
     - 1.020732500, 0.923289500, 0.933372000, 0.952477000, 0.958310500, 0.987663500, 0.962260500, 0.954437500
     - 1.031538000, 0.942935000, 0.926795500, 0.931413000, 0.967317500, 0.956157000, 0.957643500, 0.997703500
   * - dense-1024
     - cuda-cublas
     - 2.623421000, 2.406624500, 2.355571000, 2.428770000, 2.420221500, 2.518552500, 2.460275500, 2.475390500
     - 2.665447000, 2.403424000, 2.395508500, 2.385966500, 2.464929500, 2.502450000, 2.469516000, 2.550288000
   * - dense-1024
     - torch-cuda-eager
     - 2.025963000, 1.753304000, 1.768614500, 1.768458500, 1.825223500, 1.892802000, 1.850138000, 1.836816000
     - 1.945506500, 1.765888000, 1.785111000, 1.745575500, 1.844140000, 1.845400500, 1.796078500, 1.862946000
   * - dense-2048
     - cuda-native
     - 1.459068000, 1.239852000, 1.251723000, 1.249078000, 1.352463000, 1.342723000, 1.350989000, 1.252674500
     - 1.259080500, 1.244780500, 1.231785000, 1.313897500, 1.343711500, 1.371732000, 1.293001500, 1.304332500
   * - dense-2048
     - cuda-cublas
     - 2.978474000, 2.554766500, 2.628190500, 2.578413000, 2.773440500, 2.776611000, 2.809223500, 2.651693000
     - 2.630277000, 2.606518000, 2.534772500, 2.694192000, 2.779888500, 2.853963000, 2.716538500, 2.687824000
   * - dense-2048
     - torch-cuda-eager
     - 2.117021500, 1.863862500, 1.935893000, 1.955716500, 1.985149500, 2.000111500, 2.045383500, 1.932164000
     - 1.939356500, 1.901047000, 1.888920500, 1.950274000, 1.998780500, 2.080354500, 2.001809000, 1.999463500

C6 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - cuda-native
     - 1.058852000, 0.960998000, 1.192425500, 1.021661000, 1.089756000, 1.058205000, 1.005072000, 1.075331000
     - 1.017685000, 1.015317000, 1.097058000, 1.048227500, 1.018771000, 1.160319000, 1.038598500, 1.116000500
   * - dense-64
     - cuda-cublas
     - 1.112173000, 1.015508000, 1.248267500, 1.048002500, 1.099931000, 1.120944500, 1.035932000, 1.125719500
     - 1.033318500, 1.001883500, 1.081862500, 1.069224000, 1.037975000, 1.195224000, 1.082638500, 1.151985000
   * - dense-64
     - torch-cuda-eager
     - 2.396633000, 2.240985500, 2.767619500, 2.189033000, 2.370450000, 2.291004500, 2.188399500, 2.362297000
     - 2.252341500, 2.228690500, 2.538618000, 2.263202500, 2.232969000, 2.596406000, 2.304693500, 2.629142000
   * - dense-256
     - cuda-native
     - 0.960969000, 0.947113500, 1.056390000, 1.024687000, 0.946758500, 0.954209000, 0.925633500, 0.940284000
     - 0.912178000, 0.920938000, 0.950869500, 0.925538000, 0.940972500, 0.987976500, 0.946692000, 0.999968500
   * - dense-256
     - cuda-cublas
     - 0.948960000, 0.926711500, 1.059967500, 0.993359500, 0.942748500, 0.956602500, 0.924325000, 0.931454500
     - 0.911110000, 0.913101500, 0.949858500, 0.950523000, 0.956357500, 1.000899000, 0.966539500, 0.987141000
   * - dense-256
     - torch-cuda-eager
     - 2.101947500, 2.020676000, 2.348385000, 2.124956000, 2.056069500, 1.973004000, 2.030604500, 2.032169500
     - 1.961052000, 1.949164500, 2.149007000, 1.996247000, 2.018325500, 2.269686500, 2.091043500, 2.287014000
   * - dense-1024
     - cuda-native
     - 1.233264500, 1.171732000, 1.387385500, 1.307418000, 1.143720500, 1.256560000, 1.214936000, 1.189288000
     - 1.146746000, 1.195410000, 1.353063000, 1.208310500, 1.247749000, 1.312411500, 1.233504500, 1.296120500
   * - dense-1024
     - cuda-cublas
     - 1.129131500, 1.073292000, 1.223961000, 1.189760000, 1.050132500, 1.152245500, 1.126715500, 1.101087500
     - 1.060200000, 1.094270000, 1.220965500, 1.095484000, 1.127544500, 1.187061000, 1.167152000, 1.200936000
   * - dense-1024
     - torch-cuda-eager
     - 2.154927500, 1.938299000, 2.389294500, 2.188384500, 1.748322500, 2.068449500, 2.003670500, 1.957254500
     - 1.776475000, 1.950867500, 2.277662000, 1.968101500, 1.985837000, 2.370008000, 2.013941000, 2.212728500
   * - dense-2048
     - cuda-native
     - 1.566349000, 1.545473500, 1.647134000, 1.674490500, 1.566443500, 1.558577500, 1.570602000, 1.603178000
     - 1.506743500, 1.478755000, 1.718992000, 1.516905500, 1.567551500, 1.613867000, 1.514526000, 1.547271000
   * - dense-2048
     - cuda-cublas
     - 1.406331500, 1.357600000, 1.463036500, 1.460465000, 1.391031000, 1.410578000, 1.406675500, 1.397940500
     - 1.368574000, 1.320223000, 1.513920500, 1.352095000, 1.416017500, 1.439498500, 1.380636500, 1.364791000
   * - dense-2048
     - torch-cuda-eager
     - 2.253799000, 2.152138000, 2.214669000, 2.285615500, 2.138135000, 2.150512000, 2.135652000, 2.187812000
     - 2.094800500, 2.017490500, 2.435348000, 2.077767000, 2.175602000, 2.307237000, 2.067454500, 2.037637500

C7 process observations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 36 36

   * - Case
     - Candidate
     - A1--A8 ms
     - B1--B8 ms
   * - dense-64
     - cuda-native
     - 0.602499000, 0.676204500, 0.641782500, 0.840580500, 0.728921500, 0.754733000, 0.702643000, 0.796951000
     - 0.678258000, 0.648314500, 0.703109000, 0.894646000, 0.638390000, 0.673202500, 0.860620500, 0.658208000
   * - dense-64
     - cuda-cublas
     - 0.655174500, 0.736147500, 0.728911500, 1.026963500, 0.755851500, 0.756554500, 0.769478000, 0.973403000
     - 0.714897500, 0.716759500, 0.759471500, 0.965431500, 0.687136500, 0.719517000, 0.932687500, 0.718472000
   * - dense-64
     - torch-cuda-eager
     - 1.809247500, 2.190159000, 1.956555000, 2.783302500, 2.089637000, 2.196478500, 1.949247500, 2.543216000
     - 1.943441000, 1.917333500, 1.966230000, 2.761847500, 1.999103500, 2.027765500, 2.721109000, 2.005371000
   * - dense-256
     - cuda-native
     - 0.596219500, 0.627107000, 0.637750500, 0.783451500, 0.643968000, 0.616033000, 0.762353000, 0.640074500
     - 0.664694000, 0.636190500, 0.630275000, 0.855263000, 0.629816000, 0.621216500, 0.679869500, 0.635617000
   * - dense-256
     - cuda-cublas
     - 0.611068500, 0.686863000, 0.648642500, 0.715774000, 0.635426000, 0.637167500, 0.857536500, 0.682934000
     - 0.667691000, 0.645010000, 0.671346000, 0.910342500, 0.621762500, 0.628735500, 0.696579000, 0.631433500
   * - dense-256
     - torch-cuda-eager
     - 1.570924500, 1.749644500, 1.721078500, 1.992389000, 1.665163500, 1.698908500, 1.900321500, 1.808255500
     - 1.740680500, 1.626315500, 1.706004000, 2.377185000, 1.692031000, 1.671273500, 1.923590000, 1.651795500
   * - dense-1024
     - cuda-native
     - 0.882587500, 0.912479000, 0.893514500, 1.120369000, 0.948060000, 1.051670500, 0.915294000, 1.036278500
     - 0.948602500, 0.899119000, 0.896024000, 0.943596000, 0.991591000, 0.911206000, 0.961334500, 0.908214500
   * - dense-1024
     - cuda-cublas
     - 0.757975000, 0.795062500, 0.769881500, 1.017732500, 0.829684000, 0.930255500, 0.796767000, 0.934736500
     - 0.817977500, 0.773250000, 0.789941500, 0.793373000, 0.947414000, 0.779345500, 0.831898000, 0.786994500
   * - dense-1024
     - torch-cuda-eager
     - 1.548684500, 1.790191000, 1.614466500, 2.452524000, 1.706811500, 2.190169500, 1.696837000, 2.057936000
     - 1.756196500, 1.636901000, 1.684351000, 1.672165000, 2.179268500, 1.710087500, 1.895507500, 1.631982500
   * - dense-2048
     - cuda-native
     - 1.266488000, 1.334597500, 1.298695000, 1.425639500, 1.332806000, 1.322630500, 1.302552000, 1.315012000
     - 1.343006500, 1.265816500, 1.301113500, 1.460985000, 1.382675000, 1.367156000, 1.318720000, 1.313486500
   * - dense-2048
     - cuda-cublas
     - 1.098609000, 1.177786500, 1.140402500, 1.235242500, 1.172779500, 1.153630500, 1.133032500, 1.162116000
     - 1.161543000, 1.091221500, 1.124997000, 1.243638000, 1.265490500, 1.212367000, 1.139346500, 1.131442500
   * - dense-2048
     - torch-cuda-eager
     - 1.745136000, 1.959729500, 1.794697000, 2.179952000, 1.804593000, 1.964306000, 1.790214000, 1.834134500
     - 1.875910500, 1.622160000, 1.804604500, 2.242824500, 2.171582000, 2.153555500, 1.900842500, 1.800214000

Reproduce the declared work
---------------------------

The public ``make pyprof`` commands below reproduce the seven declared
input/mode/scope configurations through the same fixtures, adapters,
true-residual checks and Williams scheduler exercised by pytest. Each command
is one artifact; it does not by itself reproduce the balanced confidence
calibration. Use separate build directories and retain each output under a
fresh name. Build before the quiet timing campaign; ``make pyprof`` has a
build prerequisite and must not target an immutable stage as ``BUILD_DIR``.
Exact historical reproduction also requires the solver/tool revisions,
compiler/runtime/provider controls and frozen identities above.

.. code-block:: bash

   common='--seed 20261017 --warmups 3 --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50'
   cpu_refs='--implementations native,numpy-out,torch-eager-out'
   blas_refs='--implementations native,native-blas,numpy-out,torch-eager-out'
   taskset -c 0 make pyprof PRESET=cpu-release PYTHON=.venv/bin/python \
       BUILD_DIR=build/reproduce-confidence/cpu-release \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 \
       PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 \
       PYPROF_ARGS="$common $cpu_refs --algorithm standard --preconditioner none --fused"
   taskset -c 0 make pyprof PRESET=cpu-blas PYTHON=.venv/bin/python \
       BUILD_DIR=build/reproduce-confidence/cpu-blas \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 \
       PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 \
       PYPROF_ARGS="$common $blas_refs --algorithm explicit --preconditioner none --no-fused"
   taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON=.venv/bin/python \
       BUILD_DIR=build/reproduce-confidence/cpu-openmp \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 \
       PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 \
       PYPROF_ARGS="$common $cpu_refs --algorithm standard --preconditioner jacobi --fused"
   for selection in api-host api-device session-host prepared-device; do
       scope=${selection%-*}
       mode=${selection#*-}
       env -u CUDA_LAUNCH_BLOCKING CUBLAS_WORKSPACE_CONFIG=:4096:8 \
           taskset -c 0 make pyprof PRESET=cuda-release PYTHON=.venv-cuda/bin/python \
           BUILD_DIR=build/reproduce-confidence/cuda-release \
           CUDA_ROOT="$PWD/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 \
           CUDA_ARCHITECTURES=75 CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
           PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE="$scope" PYPROF_THREADS=1 \
           PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 \
           PYPROF_OUTPUT="build/reproduce-confidence/cuda-$selection.json" \
           PYPROF_ARGS="$common --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode $mode --algorithm standard --preconditioner jacobi --fused"
   done

For the full balanced procedure, follow :doc:`campaigns`: create one input
plan per preset, declaring both aliases of the same frozen stage and the
configurations above. Freeze the plan with ``--plan-only``, then execute it
once in a new directory. The original four input plans and expanded schedules
remain in the evidence directory below. Do not execute an already started
output directory or replace it after seeing its outcome. A repeat is a new,
separately declared campaign whose outcome must remain alongside this one.

Evidence identities
-------------------

All paths in this section are relative to the retained local evidence root
``build/reliable-performance-20261018/``. Manifests bind every raw artifact and
log digest, telemetry snapshot, launch command and comparison report; plans
bind complete stage payloads, compiled-source declarations, interpreter hashes
and tool hashes. They preserve the original successful execution's provenance.
This report does not rerun timing or recompute confidence intervals.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - SHA256
   * - calibration-analysis.json
     - ``4974aaeb7bd314a2c39ec27b9bce81d940b0b246c9540442981a09a5cd0f9ef7``
   * - source-baseline.json
     - ``09403d4df478aec01c5aa72fe4893b5ee43191da353eb80c1619aab1f034d218``
   * - tool-source.json
     - ``63f9e0a3d658bf83a16887881737dfaeecdec0864fe4c9a1a029b42acc43ec0d``
   * - calibration-execution-summary.json
     - ``ed7def359c6a05e000e8c3cf92ba2b7030f74fdf32f7c263ba376760da8013f4``
   * - additional-observer-query.json
     - ``c1bd5c48383c9a9598e2b75b294f3d17c6063d9064eb3b43186affa262669152``
   * - setup-summary.json
     - ``9ee0ed4a37181d3af5e247fdb6700068b68d7bbd2069c2ff91d5231a04be214f``
   * - setup-cpu-release.log
     - ``2667d6d5bd91e4a832752b04cac8075daef692785c06b2b3495484ab3bf287f9``
   * - setup-fixed-summary.json
     - ``d03db540537c9b7953d1bacef0df7cc241a82c7e5dd5f89d3a76d34239e7ab28``
   * - setup-fixed-cpu-release.log
     - ``d88b60fb9d192470087cf14b036ef4772f9a0b2a08e7a4811c4c1ea0cd598183``
   * - setup-fixed-cpu-blas.log
     - ``a4645f8bf0a22e66bbb861ffd4b3c7523ed97fc0a6c2fef57a2e1689e9feb321``
   * - setup-fixed-cpu-openmp.log
     - ``fe13e0b32153e312652d0d790be06ffa5ea7c7c95801319d374c41e0b0187325``
   * - setup-fixed-cuda-release.log
     - ``2683cb572a8ab3056023721c519ef7bc5ad99675b1da6c4fabdc10913a539ac4``
   * - setup_baseline.py
     - ``4cf3f8e44889ec6e6a2d3050b58917b274b2289c43ecb0c9c0be2b911c63185c``
   * - setup_baseline_fixed.py
     - ``9f5b2a40201eea95c92e2397ed5d59f75918c32c669cfc5d5b730d9fb7e634ff``
   * - run_calibrations.py
     - ``11c01c4ddb5b20084cf619f502ccdc066bd013e633e1a846c254843cb7632170``
   * - freeze_stage.py
     - ``6e4e06991f095c31d073df67ac2418d2eee52acc4d1226b4e47f108c1d20a2b4``
   * - protocol.py
     - ``e9d8ee8e95b06c5a41416f807615b5f23020775702f4cb838dce7dd687ebcb0a``

cpu-release campaign and stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - SHA256
   * - inputs/calibration-cpu-release.json
     - ``f150533756be7710643961ff54919454358bbc4557f4927f9f5870188e08a209``
   * - calibration-cpu-release/plan.json
     - ``9f78ef6014d36a7d797c3f425ce50230824ecdc7f7ce02611daaddb28643da18``
   * - calibration-cpu-release/manifest.json
     - ``9635ade929760cd3787901edfd097782f5fbc92bc3e9beef490f87ea7be76ab9``
   * - calibration-cpu-release/reports/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard.json
     - ``8cc4f174702bed6d2b3e5896c8aa7e5fd211456f44cb8af64e0abbf76d134d36``
   * - baseline-stages/cpu-release/source-files.json
     - ``de44d54de1c7133d20fbfb3099a8e412d489a4f5e9f59fb1e53e1d8fee2d2cf8``
   * - baseline-stages/cpu-release/libcg_core.a
     - ``e47471ca51084098a11eee8f0b45bb1b06b8a78b27f0b868408d8e709de6f3da``
   * - baseline-stages/cpu-release/compile_commands.json
     - ``eb4fd1947170009fe5cc972f3f792e928755bab5bb342b0594a32ebc3eeee632``
   * - baseline-stages/cpu-release/identity.json
     - ``65d54c622017689d2eabdd543d5cbdc15e70b2030088512972d9e82248883f38``
   * - baseline-stages/cpu-release/CMakeCache.txt
     - ``3c3a873efe97716f8670ca14a727c4dec59b030cfd8e2d2cbc3d12b9f298fcf7``
   * - baseline-stages/cpu-release/python/cgpy/_core.cpython-312-x86_64-linux-gnu.so
     - ``76168203038c854eec161fb28eac73b832ec0d0341d76e0266dfdc3514318611``

cpu-blas campaign and stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - SHA256
   * - inputs/calibration-cpu-blas.json
     - ``aac26ae4643fd417a1674e16302a1497c52097fe75ebb69efa770535382754ae``
   * - calibration-cpu-blas/plan.json
     - ``9cc77c414382eb4b27bb983ff3535ffea46c73081a7fad1ab7a3dd1d23f7ce27``
   * - calibration-cpu-blas/manifest.json
     - ``5c8f114588219f50c619bb6619c12be28257e4909c0b775818cbf1f1eff849bf``
   * - calibration-cpu-blas/reports/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit.json
     - ``ba7f317df061a084b408e9aa99479d25118f636879051ade6fb1058bbe0da2e5``
   * - baseline-stages/cpu-blas/source-files.json
     - ``de44d54de1c7133d20fbfb3099a8e412d489a4f5e9f59fb1e53e1d8fee2d2cf8``
   * - baseline-stages/cpu-blas/libcg_core.a
     - ``fe9159eb82deff9837978f28fbd51b57ad0aeb097bf68fecb3995e60d4ce5352``
   * - baseline-stages/cpu-blas/compile_commands.json
     - ``7cd5720a683e3378b3048dce648138a042012ca42eb21d59b4e7ab68abffff8c``
   * - baseline-stages/cpu-blas/identity.json
     - ``76c69c450baed44d866848639ed8f8f9da3dc4f72f0764d62a9d255a9538bbda``
   * - baseline-stages/cpu-blas/CMakeCache.txt
     - ``9a1dc6e556b6dad28dc1d49491297e5c6a00015b44df41c435644c61c734e6ae``
   * - baseline-stages/cpu-blas/python/cgpy/_core.cpython-312-x86_64-linux-gnu.so
     - ``81f5e99ede9ebf0d62e2ffcef1b4529183ce335a9d2e713d6a2d5373c7f63702``

cpu-openmp campaign and stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - SHA256
   * - inputs/calibration-cpu-openmp.json
     - ``91e6dcdcc17ce724bd8289952607eab9f56fc50d727782b7e92bc81dc71bb00c``
   * - calibration-cpu-openmp/plan.json
     - ``d82172afabf4cdd391dc11789e75eccac01af4bcd99862e2f6bd14019683e797``
   * - calibration-cpu-openmp/manifest.json
     - ``0e226a0c3d9924e46545fcdc4349069b2e284663c10f1b21492e7238e141cf95``
   * - calibration-cpu-openmp/reports/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard.json
     - ``00890d2d41340b7f88db91c3eb7587c372236186d284b26b980cc1bd5949d42c``
   * - baseline-stages/cpu-openmp/source-files.json
     - ``de44d54de1c7133d20fbfb3099a8e412d489a4f5e9f59fb1e53e1d8fee2d2cf8``
   * - baseline-stages/cpu-openmp/libcg_core.a
     - ``166e96b104572f1c717186c838d08e208d3391055914d8cd24334535368ceb65``
   * - baseline-stages/cpu-openmp/compile_commands.json
     - ``2a95ea3794ac02fb5fb173180aaeeb90b9b3141ae14a89ca76a95119550743ad``
   * - baseline-stages/cpu-openmp/identity.json
     - ``6d611ecf0a45a4e5f989e57c5b34528487d6bad5ec055ae60892cb9a0f57b2bb``
   * - baseline-stages/cpu-openmp/CMakeCache.txt
     - ``24c3bf3d3d0061c9a2c5a86e5271a9dc236e524bc24010ca11046498ec139250``
   * - baseline-stages/cpu-openmp/python/cgpy/_core.cpython-312-x86_64-linux-gnu.so
     - ``17d44c0b297d33607aef6c51084304e1955b3e5961fec29409e4a13a57cb1ea1``

cuda-release campaign and stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - SHA256
   * - inputs/calibration-cuda-release.json
     - ``afbe3b042fdb4dfde3f433f651f970f543dd8cad75a6cf5a915d6f955eb95b7e``
   * - calibration-cuda-release/plan.json
     - ``befb348aa4161d88a0a2f5101dd350d75f6daecf5b79b659ac751256377d920a``
   * - calibration-cuda-release/manifest.json
     - ``f22ee7194333605172a49abe73d50bfea245d38ba3909a099c370b662ad9516f``
   * - calibration-cuda-release/reports/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard.json
     - ``81b5248bcf731d5e80ba27ed634e63a34a0b5f7ccc1f657eda64b15b1b4236d8``
   * - calibration-cuda-release/reports/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard.json
     - ``03296e4b86f2c31a9c0f24bf6aad8feaa986570c378add8ec43e7b5a0e1ec283``
   * - calibration-cuda-release/reports/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard.json
     - ``18604528de49057c8457f946ca8de6fc0110d5e1a5732d4cd6280ffc02e20427``
   * - calibration-cuda-release/reports/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard.json
     - ``49da031110a71447bc7403cb73485e53d627bef392e88873a0db5b7136ab2e7a``
   * - baseline-stages/cuda-release/source-files.json
     - ``de44d54de1c7133d20fbfb3099a8e412d489a4f5e9f59fb1e53e1d8fee2d2cf8``
   * - baseline-stages/cuda-release/libcg_core.a
     - ``b6776cf9cd999ea45206ca9b5552a747575162b87f403d5297f4e5b04a35a0a9``
   * - baseline-stages/cuda-release/compile_commands.json
     - ``416932af06bed2811f8b00140891a95f8cb170ff53428d580cd1e89be47399a6``
   * - baseline-stages/cuda-release/identity.json
     - ``3a4bdbc2df5b44e8cbdc94f7b2d4de8558d9514659a0e70dd872109e3d3fd551``
   * - baseline-stages/cuda-release/CMakeCache.txt
     - ``0a4eccd32d33c0b3f5188e7434b78bdde5d532401c33cbd64c4f6dee526cb848``
   * - baseline-stages/cuda-release/python/cgpy/_cuda.cpython-312-x86_64-linux-gnu.so
     - ``2b95332f5e7cfa622e5de9eeb109ed8d6b2798f7e9eb9f8ee7d686a9c1f71fbe``
   * - baseline-stages/cuda-release/python/cgpy/_core.cpython-312-x86_64-linux-gnu.so
     - ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``

Fixture identities
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Case
     - Input SHA256
   * - dense-64
     - ``b2e274e240704161b83bff5de50cf7b8ba9e3848b098f027c1ba8b5fccf82a29``
   * - dense-256
     - ``6a6622788ec7f29156523e562995e09db8ec35032185f8c76ae6fccf2165a4a0``
   * - dense-1024
     - ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``
   * - dense-2048
     - ``8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c``

