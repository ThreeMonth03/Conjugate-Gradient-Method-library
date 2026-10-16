Restart-aware CPU and CUDA baseline, 2026-10-17
===============================================

This baseline retains all 54 configurations, 108 fresh-process artifacts and
1,336 eligible candidate/case results. The complete tables contain 668 rows,
each with both fixed process observations. Ratios describe this host and run
order; they are point estimates without confidence or regression claims.

Across both fresh processes, every recorded iteration/matvec/restart signature
repeats; external restart observations remain explicitly unknown. No known
work changes across scopes, native execution modes or presets were observed
for the same candidate and fixture. CUDA retains
eight higher medians against eager Torch: fresh cuBLAS Jacobi API on all four
dense sizes in both native execution modes. Its Session/prepared comparisons,
and all cuda-native dense/CSR comparisons, have lower measured medians.
CPU implementation comparisons retain 45 higher medians out of 288 matched
native/reference pairs; all 64 CPU CSR pairs have lower medians. The separate
solver-strategy inventory retains 25 higher medians out of 80 pairs, with
unknown external restart counts and different solver policies. These are
scoped descriptive observations, not confidence or universal speed claims.

The public measured-source tag is
`benchmarks/restart-baseline-20261017 <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/tree/benchmarks/restart-baseline-20261017>`__.
The NVIDIA driver recorded in this campaign is **610.88**, with driver API
**13030**; historical reports used a different driver and sampling protocol.
Their timings are not silently pooled with this baseline.

The next candidates target three specific costs. Private CUDA execution
storage can combine its partial reductions, scalar and optional full 4 MiB
cuBLAS scratch into one RAII allocation, while retaining fresh handle ownership
and the existing synchronization contract. CPU GEMV can process four rows
together and share loaded x values, preserving each row's four-lane reduction
and the serial outer accumulation order; register pressure may offset that
reuse. OpenMP partitions on odd shapes may change and require explicit
validation and work-count disclosure. A CUDA device candidate can run the
existing stable-norm collective inside the residual/beta check and remove one separate post-update norm
launch. All threads must complete the collective before thread zero evaluates
the existing residual-exit and beta logic; guarded work, packet copies,
synchronization and exit priorities stay intact. An explicit invalid-alpha
exit saves no launch. These are hypotheses for later fixed A/B evaluation,
not measured improvements or permission to relax iteration/matvec/restart
parity. All framework controls and inconclusive or regression results must
remain visible.

Actual compiled commit: ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``; compiled tree: ``06f324529806d3edf005e4ecbb6a3c27f048f946``.

Method and scope
----------------

The fixed order is all 54 configurations forward, then exactly in reverse.
Every artifact uses seed 20261017, three warmups, 36 complete balanced
Williams rounds and one solve per round. All declared candidates run in every
artifact. The estimator is the median of the two process medians, and the
range brackets those two medians. Reverse/forward compares those process
medians directly; it does not establish drift, causation or significance.

Each table records iterations/matvecs/restarts as I/M/R. A direction restart
is counted only when a new CG/PCG direction is actually installed after
residual recomputation; the initial direction and unsuccessful attempts are
excluded. Official SciPy CG/PCG and SuperLU restart counts are explicitly unknown.
Unknown restart counts do not establish numerical-work parity. Known work
differences remain visible and are not normalized into a pure kernel ratio.

CUDA inputs are GPU-resident. Upload and independent correctness downloads
are outside solve timing; x0 reset, dispatch, required scalar synchronization
and completed final work are included. API allocates output and workspace on
every call, plus Jacobi data when requested. Session reuses iterative workspace
but allocates fresh output and per-call Jacobi data when requested. Prepared
reuses setup, output and workspace.
Native host/device execution is explicit; the Torch project reference keeps
its own implementation and discloses that native execution mode does not apply.
See :doc:`methodology` for the full allocation and timing contracts, including
framework caching allocators and the distinction between tensor creation and
traced CUDA allocation calls.

CPU OpenMP uses two threads and logical affinity [0,2]; all other presets use
one thread and affinity [0]. WSL affinity does not isolate the physical host.
Host load, Windows display use, GPU clocks and thermal state are uncontrolled.
Different frameworks, scopes, modes and solver strategies may perform different
accepted-tolerance work. No cross-scope strict comparison or universal speed
claim is made. SciPy official solvers have a different refresh policy; SuperLU
is a general sparse direct solver. LU-total includes factorization and solve,
whereas LU-reuse retains its factorization. Neither is a pure CG kernel.

Repair provenance
-----------------

The original campaign stopped after numerical worker 29 completed successfully:
the parent audit incorrectly indexed null Torch metadata for the six-strategy
cohort, which intentionally imports no Torch. Its failed manifest, claim, plan,
all 29 raw/import/log sets and original frozen tools remain unchanged. A separate
one-time audit repair revalidated those 29 without running them again, then ran
only original positions 30–108, in the original order. It changes only applicable
Torch metadata auditing. All sampling, numerical policies and other environment
and complete threadpool gates remain exact. The continuation introduces a wall
time gap, so the campaign must not be described as uninterrupted or retry-free
without this qualification. No numerical worker was repeated or discarded.

Repeated same-configuration I/M/R gate: **passed**.

Hardware and source identity
----------------------------

.. code-block:: json

   {
     "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
     "machine": "x86_64",
     "cpu": "13th Gen Intel(R) Core(TM) i7-13700K",
     "logical_cpus": 24
   }

* cpu-release ``_core``: ``76168203038c854eec161fb28eac73b832ec0d0341d76e0266dfdc3514318611``.
* cpu-blas ``_core``: ``81f5e99ede9ebf0d62e2ffcef1b4529183ce335a9d2e713d6a2d5373c7f63702``.
* cpu-openmp ``_core``: ``17d44c0b297d33607aef6c51084304e1955b3e5961fec29409e4a13a57cb1ea1``.
* cuda-release ``_core``: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
* cuda-release ``_cuda``: ``760af83cc19b2e0c6b06f9373166d7bd7022d4a2a779677f5a3d141dec774240``.

.. csv-table:: Frozen stage and host compiler
   :class: benchmark-results
   :header-rows: 1

   Preset,Identity SHA256,Compiler,Flags
   cpu-release,82f66c8a54d2c5a029ab6b62941d37650c1d2f07441ad6dd5e852b052e567b38,GNU 16.0.1, -O3 -DNDEBUG
   cpu-blas,c2ee4d3b54cb2f013799122cfb6bb254f102582931ebe16f341bbb3698ace8a9,GNU 16.0.1, -O3 -DNDEBUG
   cpu-openmp,425d959e2b85fc58d2bdfcd3bc1f41f95845478f6989adb1ff723d237920375d,GNU 16.0.1, -O3 -DNDEBUG
   cuda-release,7a956531880eaee793f22d75c4cc748f53c20ee8d5569fc0f2fe4cf67411d416,GNU 13.3.0, -O3 -DNDEBUG

CUDA runtime, device mapping, compiler and package controls:

.. code-block:: json

   {
     "cuda": {
       "device": 0,
       "environment": {
         "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
         "CUDA_DEVICE_ORDER": null,
         "CUDA_LAUNCH_BLOCKING": null,
         "CUDA_VISIBLE_DEVICES": null,
         "PYTORCH_CUDA_ALLOC_CONF": null
       },
       "nvidia_smi": [
         {
           "driver_version": "610.88",
           "name": "NVIDIA GeForce RTX 2060",
           "pci_bus_id": "00000000:01:00.0",
           "physical_index": "0",
           "uuid": "GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302"
         }
       ],
       "torch": {
         "compute_capability": [
           7,
           5
         ],
         "cudnn_allow_tf32": false,
         "deterministic_algorithms": false,
         "device": 0,
         "matmul_allow_tf32": false,
         "multiprocessor_count": 30,
         "name": "NVIDIA GeForce RTX 2060",
         "runtime_version": "12.8",
         "total_memory_bytes": 6442123264,
         "uuid": "725be96d-2b67-c8ec-fa10-5d9b23a2c302"
       }
     },
     "native": {
       "runtime_info": {
         "compute_capability": [
           7,
           5
         ],
         "device": 0,
         "driver_api_version": 13030,
         "name": "NVIDIA GeForce RTX 2060",
         "runtime_version": 12080,
         "total_memory": 6442123264,
         "uuid": "GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302"
       },
       "build_info": {
         "architectures": "75",
         "build_type": "Release",
         "compiler": "NVIDIA 12.8.93",
         "enabled": true,
         "flags": " -O3 -DNDEBUG",
         "host_compiler": "GNU 13.3.0",
         "scalar_dtype": "float64",
         "toolkit_version": "12.8.93"
       }
     },
     "packages": {
       "cgpy": "0.4.0",
       "numpy": "2.2.6",
       "scipy": "1.15.3",
       "threadpoolctl": "3.6.0",
       "torch": "2.8.0+cu128"
     }
   }

Matched native/reference summary
--------------------------------

Lower/higher counts compare the median of process medians for every matched
fixture. Native/reference ratios below one indicate a lower measured native
median. Strategy comparisons remain separately labelled; unknown/different
work is descriptive. Every higher median appears in the loss inventory.

.. csv-table:: All matched scopes and controls
   :class: benchmark-results
   :header-rows: 1

   Configuration,Native,Reference,Lower/higher/equal,Ratio range,I/M/R parity
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,native,numpy-out,3/1/0,0.1800–1.1590,same
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,native,torch-eager-out,3/1/0,0.1117–1.1357,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,native,numpy-out,3/1/0,0.1909–1.2051,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,native,torch-eager-out,3/1/0,0.1190–1.1727,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,native,numpy-out,3/1/0,0.1486–1.2058,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,native,torch-eager-out,3/1/0,0.0872–1.0603,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,native,numpy-out,3/1/0,0.1512–1.2784,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,native,torch-eager-out,3/1/0,0.0917–1.1265,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,native,numpy-out,3/1/0,0.3982–1.1186,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,3/1/0,0.2037–1.1051,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,native,numpy-out,3/1/0,0.3064–1.0529,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,native,torch-eager-out,4/0/0,0.1525–0.9917,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,numpy-out,2/2/0,0.1904–1.4177,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,3/1/0,0.1038–1.3392,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native,numpy-out,2/2/0,0.2195–1.1483,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native,torch-eager-out,4/0/0,0.1219–0.9640,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native,numpy-out,3/1/0,0.1641–1.1081,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native,torch-eager-out,3/1/0,0.1010–1.0545,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native-blas,numpy-out,4/0/0,0.1564–0.9611,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native-blas,torch-eager-out,4/0/0,0.0962–0.9146,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host,native,numpy-out,3/1/0,0.1711–1.1090,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host,native,torch-eager-out,3/1/0,0.1054–1.0509,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host,native-blas,numpy-out,4/0/0,0.1625–0.9339,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host,native-blas,torch-eager-out,4/0/0,0.1001–0.8849,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native,numpy-out,3/1/0,0.1248–1.1434,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native,torch-eager-out,3/1/0,0.0775–1.0342,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native-blas,numpy-out,4/0/0,0.1155–0.9886,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native-blas,torch-eager-out,4/0/0,0.0718–0.8942,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host,native,numpy-out,3/1/0,0.1529–1.2543,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host,native,torch-eager-out,3/1/0,0.0897–1.2421,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host,native-blas,numpy-out,4/0/0,0.1163–0.9685,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host,native-blas,torch-eager-out,4/0/0,0.0682–0.9591,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native,numpy-out,3/1/0,0.2361–1.1012,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,4/0/0,0.1294–0.9291,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native-blas,numpy-out,4/0/0,0.2766–0.9585,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native-blas,torch-eager-out,4/0/0,0.1516–0.8087,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host,native,numpy-out,3/1/0,0.2747–1.0174,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host,native,torch-eager-out,4/0/0,0.1398–0.9063,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host,native-blas,numpy-out,4/0/0,0.3100–0.9326,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host,native-blas,torch-eager-out,4/0/0,0.1578–0.8308,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,numpy-out,2/2/0,0.1728–1.2329,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,3/1/0,0.0977–1.2466,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native-blas,numpy-out,3/1/0,0.1713–1.1815,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native-blas,torch-eager-out,3/1/0,0.0968–1.1946,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native,numpy-out,2/2/0,0.2349–1.0818,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native,torch-eager-out,3/1/0,0.1464–1.0944,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native-blas,numpy-out,3/1/0,0.2307–1.0135,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native-blas,torch-eager-out,3/1/0,0.1438–1.0252,same
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host,native,scipy-out,4/0/0,0.1131–0.4946,same
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host,native,torch-eager-out,4/0/0,0.0847–0.6763,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host,native,scipy-out,4/0/0,0.0742–0.5254,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host,native,torch-eager-out,4/0/0,0.0568–0.7004,same
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host,native,scipy-out,4/0/0,0.1365–0.5128,same
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,4/0/0,0.0906–0.6425,same
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host,native,scipy-out,4/0/0,0.0706–0.5079,same
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,4/0/0,0.0499–0.6409,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,native,numpy-out,2/2/0,0.3029–1.2600,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,native,torch-eager-out,4/0/0,0.2004–0.9944,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,native,numpy-out,2/2/0,0.2723–1.3425,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,native,torch-eager-out,3/1/0,0.1772–1.0750,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,native,numpy-out,2/2/0,0.4660–1.3375,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,3/1/0,0.2759–1.1799,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,numpy-out,3/1/0,0.3731–1.1487,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,4/0/0,0.2352–0.8438,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host,native,scipy-out,4/0/0,0.2050–0.4902,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host,native,torch-eager-out,4/0/0,0.1366–0.6862,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host,native,scipy-out,4/0/0,0.1399–0.5269,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host,native,torch-eager-out,4/0/0,0.0674–0.6576,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host,native,scipy-out,4/0/0,0.2193–0.6221,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,4/0/0,0.0915–0.7053,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host,native,scipy-out,4/0/0,0.1779–0.5435,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,4/0/0,0.0925–0.7407,same
   cpu-release-sparse-solvers-none-api-fused1-host,native-cg,scipy-cg,5/0/0,0.0976–0.5162,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-cg,scipy-pcg,3/2/0,0.0932–2.8684,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-cg,scipy-lu-total,3/2/0,0.1931–3.1175,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-cg,scipy-lu-reuse,0/5/0,1.6631–35.6254,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-pcg,scipy-cg,5/0/0,0.0262–0.1260,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-pcg,scipy-pcg,5/0/0,0.1171–0.2726,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-pcg,scipy-lu-total,5/0/0,0.0574–0.2963,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,native-pcg,scipy-lu-reuse,1/4/0,0.8859–3.3854,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-cg,scipy-cg,5/0/0,0.0957–0.5397,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-cg,scipy-pcg,3/2/0,0.0857–3.0481,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-cg,scipy-lu-total,3/2/0,0.1647–2.8751,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-cg,scipy-lu-reuse,0/5/0,1.7888–30.9123,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-pcg,scipy-cg,5/0/0,0.0179–0.1113,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-pcg,scipy-pcg,5/0/0,0.0997–0.2679,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-pcg,scipy-lu-total,5/0/0,0.0485–0.2527,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,native-pcg,scipy-lu-reuse,2/3/0,0.7578–2.7172,unknown
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4010–0.7485,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.5415–0.7529,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2798–0.6071,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.4083–0.6587,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4029–0.7419,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4197–0.6876,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2848–0.6295,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3232–0.6028,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4043–0.7348,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4183–0.6933,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2822–0.6283,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3012–0.5991,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4286–0.6687,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-cublas,torch-cuda-eager,0/4/0,1.2156–1.3759,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3332–0.5736,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-cublas,torch-cuda-eager,0/4/0,1.1223–1.3302,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.3967–0.6316,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4263–0.5874,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3135–0.6068,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3310–0.5396,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4298–0.7426,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4573–0.6640,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3291–0.7134,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3739–0.6594,same
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4746–0.5103,same
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3443–0.3592,same
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4985–0.5062,same
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3384–0.3485,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4627–0.5085,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3298–0.3589,same
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4509–0.5301,same
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2559–0.3162,same
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4644–0.5242,same
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2705–0.2963,same
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4967–0.5092,same
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2741–0.2968,same

Complete remaining higher-median inventory
------------------------------------------

.. csv-table:: Every native/reference point-estimate loss
   :class: benchmark-results
   :header-rows: 1

   Configuration,Case,Native/reference,Native ms,Reference ms,Ratio,Native I/M/R,Reference I/M/R,Parity
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/numpy-out,133.250233,114.968729,1.15901,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/torch-eager-out,133.250233,117.332482,1.13566,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,dense-2048,native/numpy-out,119.545844,99.196311,1.20514,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,dense-2048,native/torch-eager-out,119.545844,101.940282,1.17270,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/numpy-out,132.480457,109.864913,1.20585,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/torch-eager-out,132.480457,124.946043,1.06030,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,dense-2048,native/numpy-out,134.345307,105.088718,1.27840,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,dense-2048,native/torch-eager-out,134.345307,119.261869,1.12647,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/numpy-out,8.917253,7.971514,1.11864,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/torch-eager-out,8.917253,8.069224,1.10509,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,dense-2048,native/numpy-out,8.576768,8.145480,1.05295,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-1024,native/numpy-out,1.459701,1.403025,1.04040,4/6/0,4/6/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/numpy-out,10.457162,7.376163,1.41770,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/torch-eager-out,10.457162,7.808286,1.33924,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-1024,native/numpy-out,1.489243,1.418102,1.05017,4/6/0,4/6/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native/numpy-out,9.262359,8.066068,1.14831,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/numpy-out,132.933591,119.968866,1.10807,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/torch-eager-out,132.933591,126.057971,1.05454,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host,dense-2048,native/numpy-out,117.896980,106.307222,1.10902,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host,dense-2048,native/torch-eager-out,117.896980,112.187574,1.05089,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/numpy-out,132.347169,115.752917,1.14336,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/torch-eager-out,132.347169,127.967336,1.03423,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host,dense-2048,native/numpy-out,141.413416,112.746579,1.25426,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host,dense-2048,native/torch-eager-out,141.413416,113.849544,1.24211,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/numpy-out,7.491425,6.802998,1.10119,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host,dense-2048,native/numpy-out,7.807740,7.674391,1.01738,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-1024,native/numpy-out,1.619652,1.597488,1.01387,4/6/0,4/6/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/numpy-out,10.426763,8.456992,1.23292,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/torch-eager-out,10.426763,8.364180,1.24660,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native-blas/numpy-out,9.992125,8.456992,1.18152,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native-blas/torch-eager-out,9.992125,8.364180,1.19463,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-1024,native/numpy-out,1.864183,1.726260,1.07990,4/6/0,4/6/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native/numpy-out,11.152470,10.308947,1.08182,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native/torch-eager-out,11.152470,10.190689,1.09438,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native-blas/numpy-out,10.447604,10.308947,1.01345,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native-blas/torch-eager-out,10.447604,10.190689,1.02521,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,dense-1024,native/numpy-out,12.283158,11.516861,1.06654,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/numpy-out,90.669160,71.958069,1.26003,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,dense-1024,native/numpy-out,14.538949,13.107376,1.10922,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/numpy-out,101.712580,75.760981,1.34255,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/torch-eager-out,101.712580,94.619034,1.07497,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-256,native/numpy-out,0.079250,0.076524,1.03562,4/6/0,4/6/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/numpy-out,10.305668,7.704990,1.33753,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/torch-eager-out,10.305668,8.734395,1.17989,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/numpy-out,12.494393,10.876795,1.14872,3/5/0,3/5/0,same
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-16,native-cg/scipy-lu-total,0.566271,0.456687,1.23995,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-16,native-cg/scipy-lu-reuse,0.566271,0.051512,10.99293,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-16,native-pcg/scipy-lu-reuse,0.092243,0.051512,1.79070,48/50/0,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-32,native-cg/scipy-pcg,5.768379,2.010995,2.86842,1799/1836/35,90/91/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-32,native-cg/scipy-lu-total,5.768379,1.850295,3.11755,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-32,native-cg/scipy-lu-reuse,5.768379,0.161918,35.62537,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-2d-32,native-pcg/scipy-lu-reuse,0.548156,0.161918,3.38540,118/122/2,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-3d-6,native-cg/scipy-lu-reuse,0.247445,0.051081,4.84414,253/260/5,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-3d-6,native-pcg/scipy-lu-reuse,0.062159,0.051081,1.21686,24/26/0,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-3d-10,native-cg/scipy-pcg,1.600837,0.922748,1.73486,387/396/7,37/38/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,scaled-poisson-3d-10,native-cg/scipy-lu-reuse,1.600837,0.279344,5.73070,387/396/7,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,poisson-2d-16-neutral,native-cg/scipy-lu-reuse,0.072834,0.043793,1.66315,51/54/1,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host,poisson-2d-16-neutral,native-pcg/scipy-lu-reuse,0.094036,0.043793,2.14730,51/54/1,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-16,native-cg/scipy-lu-total,0.562015,0.448662,1.25265,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-16,native-cg/scipy-lu-reuse,0.562015,0.053874,10.43193,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-16,native-pcg/scipy-lu-reuse,0.075033,0.053874,1.39274,48/50/0,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-32,native-cg/scipy-pcg,5.714016,1.874628,3.04808,1799/1836/35,90/91/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-32,native-cg/scipy-lu-total,5.714016,1.987406,2.87511,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-32,native-cg/scipy-lu-reuse,5.714016,0.184846,30.91231,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-2d-32,native-pcg/scipy-lu-reuse,0.502265,0.184846,2.71721,118/122/2,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-3d-6,native-cg/scipy-lu-reuse,0.227615,0.041420,5.49526,253/260/5,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-3d-10,native-cg/scipy-pcg,1.554626,0.875958,1.77477,387/396/7,37/38/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,scaled-poisson-3d-10,native-cg/scipy-lu-reuse,1.554626,0.279542,5.56133,387/396/7,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,poisson-2d-16-neutral,native-cg/scipy-lu-reuse,0.051486,0.028782,1.78883,51/54/1,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host,poisson-2d-16-neutral,native-pcg/scipy-lu-reuse,0.059888,0.028782,2.08072,51/54/1,0/1/unknown,unknown
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-64,cuda-cublas/torch-cuda-eager,3.287730,2.704691,1.21557,5/7/0,5/7/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-256,cuda-cublas/torch-cuda-eager,3.130028,2.506757,1.24864,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-1024,cuda-cublas/torch-cuda-eager,3.317847,2.594690,1.27871,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-2048,cuda-cublas/torch-cuda-eager,3.858382,2.804317,1.37587,3/5/0,3/5/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-64,cuda-cublas/torch-cuda-eager,2.771716,2.469720,1.12228,5/7/0,5/7/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-256,cuda-cublas/torch-cuda-eager,2.710148,2.141640,1.26545,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-1024,cuda-cublas/torch-cuda-eager,2.879996,2.165131,1.33017,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-2048,cuda-cublas/torch-cuda-eager,3.000265,2.317655,1.29453,3/5/0,3/5/0,same

Complete per-configuration measurements
---------------------------------------

All 668 candidate rows follow. Forward/reverse ms are single-process medians;
the reported median and range summarize those two medians. TrueR/tol is the
largest independently measured residual-to-threshold ratio across both process outcomes.

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.050589,0.049608,0.050098,0.049608–0.050589,0.98061,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.274377,0.282243,0.278309,0.274377–0.282243,1.02867,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.425265,0.471396,0.448331,0.425265–0.471396,1.10848,49/51/0,49/51/0,0.635707
   dense-256,native,0.583891,0.533357,0.558624,0.533357–0.583891,0.91345,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.816049,0.767957,0.792003,0.767957–0.816049,0.94107,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.217837,1.131007,1.174422,1.131007–1.217837,0.92870,67/70/1,67/70/1,0.972357
   dense-1024,native,10.263201,10.331992,10.297596,10.263201–10.331992,1.00670,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,10.556204,11.000840,10.778522,10.556204–11.000840,1.04212,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,11.625245,12.452298,12.038771,11.625245–12.452298,1.07114,70/73/1,70/73/1,0.928014
   dense-2048,native,129.604324,136.896141,133.250233,129.604324–136.896141,1.05626,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,108.713886,121.223572,114.968729,108.713886–121.223572,1.11507,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,106.472242,128.192722,117.332482,106.472242–128.192722,1.20400,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.050617,0.065831,0.058224,0.050617–0.065831,1.30056,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.266272,0.343811,0.305042,0.266272–0.343811,1.29120,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.421787,0.557095,0.489441,0.421787–0.557095,1.32080,49/51/0,49/51/0,0.635707
   dense-256,native,0.569477,0.614353,0.591916,0.569477–0.614353,1.07880,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.921752,0.979056,0.950404,0.921752–0.979056,1.06217,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.211672,1.420653,1.316162,1.211672–1.420653,1.17247,67/70/1,67/70/1,0.972357
   dense-1024,native,10.062846,10.870784,10.466815,10.062846–10.870784,1.08029,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,10.761135,11.356390,11.058762,10.761135–11.356390,1.05532,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,12.078561,12.822276,12.450418,12.078561–12.822276,1.06157,70/73/1,70/73/1,0.928014
   dense-2048,native,117.446623,121.645064,119.545844,117.446623–121.645064,1.03575,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,98.183524,100.209098,99.196311,98.183524–100.209098,1.02063,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,101.036497,102.844067,101.940282,101.036497–102.844067,1.01789,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.040063,0.041937,0.041000,0.040063–0.041937,1.04676,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.269653,0.281983,0.275818,0.269653–0.281983,1.04573,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.447479,0.492793,0.470136,0.447479–0.492793,1.10127,49/51/0,49/51/0,0.635707
   dense-256,native,0.500435,0.491074,0.495754,0.491074–0.500435,0.98129,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.921965,0.849639,0.885802,0.849639–0.921965,0.92155,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.159045,1.041246,1.100145,1.041246–1.159045,0.89837,67/70/1,67/70/1,0.972357
   dense-1024,native,12.051404,12.425268,12.238336,12.051404–12.425268,1.03102,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,12.574818,12.499436,12.537128,12.499436–12.574818,0.99401,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,13.970036,14.309067,14.139551,13.970036–14.309067,1.02427,70/73/1,70/73/1,0.928014
   dense-2048,native,138.492134,126.468780,132.480457,126.468780–138.492134,0.91318,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,114.493167,105.236658,109.864913,105.236658–114.493167,0.91915,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,130.219097,119.672988,124.946043,119.672988–130.219097,0.91901,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.042717,0.046120,0.044418,0.042717–0.046120,1.07965,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.278622,0.308964,0.293793,0.278622–0.308964,1.10890,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.439085,0.529401,0.484243,0.439085–0.529401,1.20569,49/51/0,49/51/0,0.635707
   dense-256,native,0.470108,0.535496,0.502802,0.470108–0.535496,1.13909,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.800376,0.904173,0.852274,0.800376–0.904173,1.12969,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.972432,1.174504,1.073468,0.972432–1.174504,1.20780,67/70/1,67/70/1,0.972357
   dense-1024,native,12.530190,13.425578,12.977884,12.530190–13.425578,1.07146,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,12.885563,13.654557,13.270060,12.885563–13.654557,1.05968,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,14.251731,15.625897,14.938814,14.251731–15.625897,1.09642,70/73/1,70/73/1,0.928014
   dense-2048,native,116.015233,152.675381,134.345307,116.015233–152.675381,1.31599,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,97.316035,112.861401,105.088718,97.316035–112.861401,1.15974,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,106.375054,132.148684,119.261869,106.375054–132.148684,1.24229,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.016102,0.027731,0.021917,0.016102–0.027731,1.72219,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.044175,0.065918,0.055046,0.044175–0.065918,1.49220,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.076211,0.138986,0.107598,0.076211–0.138986,1.82369,5/7/0,5/7/0,0.0796191
   dense-256,native,0.076865,0.048633,0.062749,0.048633–0.076865,0.63270,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.111382,0.064640,0.088011,0.064640–0.111382,0.58035,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.201485,0.100276,0.150880,0.100276–0.201485,0.49768,4/6/0,4/6/0,0.209854
   dense-1024,native,1.172178,0.978461,1.075319,0.978461–1.172178,0.83474,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.356137,1.089579,1.222858,1.089579–1.356137,0.80344,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.634070,1.323235,1.478652,1.323235–1.634070,0.80978,4/6/0,4/6/0,0.013684
   dense-2048,native,9.863568,7.970938,8.917253,7.970938–9.863568,0.80812,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,7.460780,8.482249,7.971514,7.460780–8.482249,1.13691,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,8.377263,7.761186,8.069224,7.761186–8.377263,0.92646,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.020803,0.020701,0.020752,0.020701–0.020803,0.99507,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.071941,0.063525,0.067733,0.063525–0.071941,0.88302,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.156307,0.115942,0.136125,0.115942–0.156307,0.74176,5/7/0,5/7/0,0.0796191
   dense-256,native,0.068527,0.050184,0.059355,0.050184–0.068527,0.73234,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.098479,0.064352,0.081415,0.064352–0.098479,0.65345,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.175479,0.114980,0.145229,0.114980–0.175479,0.65524,4/6/0,4/6/0,0.209854
   dense-1024,native,1.322199,0.997249,1.159724,0.997249–1.322199,0.75424,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.379174,1.098027,1.238601,1.098027–1.379174,0.79615,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.644464,1.368481,1.506473,1.368481–1.644464,0.83217,4/6/0,4/6/0,0.013684
   dense-2048,native,8.907108,8.246427,8.576768,8.246427–8.907108,0.92583,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,7.505701,8.785259,8.145480,7.505701–8.785259,1.17048,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,10.167063,7.130846,8.648955,7.130846–10.167063,0.70137,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.009305,0.009268,0.009287,0.009268–0.009305,0.99597,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.050819,0.046746,0.048782,0.046746–0.050819,0.91985,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.090208,0.088741,0.089475,0.088741–0.090208,0.98373,5/7/0,5/7/0,0.0796191
   dense-256,native,0.058610,0.058899,0.058755,0.058610–0.058899,1.00493,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.077350,0.101597,0.089473,0.077350–0.101597,1.31347,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.110781,0.145580,0.128180,0.110781–0.145580,1.31413,4/6/0,4/6/0,0.209854
   dense-1024,native,1.495202,1.424200,1.459701,1.424200–1.495202,0.95251,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.454393,1.351657,1.403025,1.351657–1.454393,0.92936,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.683917,1.522064,1.602990,1.522064–1.683917,0.90388,4/6/0,4/6/0,0.013684
   dense-2048,native,10.129389,10.784935,10.457162,10.129389–10.784935,1.06472,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,6.969513,7.782814,7.376163,6.969513–7.782814,1.11669,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,8.576457,7.040115,7.808286,7.040115–8.576457,0.82087,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.008143,0.011373,0.009758,0.008143–0.011373,1.39664,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.030625,0.058293,0.044459,0.030625–0.058293,1.90346,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.047711,0.112388,0.080049,0.047711–0.112388,2.35556,5/7/0,5/7/0,0.0796191
   dense-256,native,0.052291,0.045480,0.048886,0.045480–0.052291,0.86974,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.085148,0.066389,0.075768,0.066389–0.085148,0.77969,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.117332,0.107577,0.112454,0.107577–0.117332,0.91686,4/6/0,4/6/0,0.209854
   dense-1024,native,1.442635,1.535851,1.489243,1.442635–1.535851,1.06461,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.500218,1.335986,1.418102,1.335986–1.500218,0.89053,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.602432,1.791591,1.697011,1.602432–1.791591,1.11804,4/6/0,4/6/0,0.013684
   dense-2048,native,10.840718,7.684000,9.262359,7.684000–10.840718,0.70881,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.890513,7.241622,8.066068,7.241622–8.890513,0.81453,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,12.059352,7.158016,9.608684,7.158016–12.059352,0.59357,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.053525,0.039368,0.046447,0.039368–0.053525,0.73550,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.047697,0.040835,0.044265,0.040835–0.047697,0.85613,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.305694,0.260435,0.283065,0.260435–0.305694,0.85195,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.513709,0.406396,0.460052,0.406396–0.513709,0.79110,49/51/0,49/51/0,0.635707
   dense-256,native,0.468253,0.489287,0.478770,0.468253–0.489287,1.04492,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.483517,0.346407,0.414963,0.346407–0.483517,0.71643,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.882363,0.729287,0.805825,0.729287–0.882363,0.82652,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.168486,0.945461,1.056973,0.945461–1.168486,0.80913,67/70/1,67/70/1,0.972357
   dense-1024,native,9.383907,10.308972,9.846440,9.383907–10.308972,1.09858,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,9.079334,9.296114,9.187724,9.079334–9.296114,1.02388,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,10.282751,11.168822,10.725787,10.282751–11.168822,1.08617,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,11.824490,12.324386,12.074438,11.824490–12.324386,1.04228,70/73/1,70/73/1,0.928014
   dense-2048,native,139.178176,126.689007,132.933591,126.689007–139.178176,0.91026,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,119.352720,111.242902,115.297811,111.242902–119.352720,0.93205,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,121.711591,118.226140,119.968866,118.226140–121.711591,0.97136,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,129.235583,122.880358,126.057971,122.880358–129.235583,0.95082,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.043864,0.055230,0.049547,0.043864–0.055230,1.25911,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.045482,0.048628,0.047055,0.045482–0.048628,1.06919,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.268346,0.310850,0.289598,0.268346–0.310850,1.15839,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.424707,0.515177,0.469942,0.424707–0.515177,1.21302,49/51/0,49/51/0,0.635707
   dense-256,native,0.492723,0.552494,0.522608,0.492723–0.552494,1.12131,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.359643,0.381453,0.370548,0.359643–0.381453,1.06064,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.743630,0.817676,0.780653,0.743630–0.817676,1.09957,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.033114,1.215331,1.124223,1.033114–1.215331,1.17638,67/70/1,67/70/1,0.972357
   dense-1024,native,10.395900,10.586603,10.491252,10.395900–10.586603,1.01834,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,8.943068,9.835743,9.389406,8.943068–9.835743,1.09982,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,10.903348,10.711827,10.807588,10.711827–10.903348,0.98243,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,12.564355,12.827360,12.695858,12.564355–12.827360,1.02093,70/73/1,70/73/1,0.928014
   dense-2048,native,121.507041,114.286919,117.896980,114.286919–121.507041,0.94058,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,100.240492,98.319106,99.279799,98.319106–100.240492,0.98083,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,108.567942,104.046502,106.307222,104.046502–108.567942,0.95835,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,113.232318,111.142831,112.187574,111.142831–113.232318,0.98155,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.043214,0.039928,0.041571,0.039928–0.043214,0.92398,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.040933,0.036040,0.038486,0.036040–0.040933,0.88045,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.331375,0.334884,0.333130,0.331375–0.334884,1.01059,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.555198,0.517583,0.536390,0.517583–0.555198,0.93225,49/51/0,49/51/0,0.635707
   dense-256,native,0.581584,0.575875,0.578730,0.575875–0.581584,0.99018,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.552561,0.374275,0.463418,0.374275–0.552561,0.67735,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.972070,1.011027,0.991549,0.972070–1.011027,1.04008,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.301757,1.295851,1.298804,1.295851–1.301757,0.99546,67/70/1,67/70/1,0.972357
   dense-1024,native,14.434668,14.578450,14.506559,14.434668–14.578450,1.00996,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,13.303029,13.041578,13.172303,13.041578–13.303029,0.98035,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,14.826606,14.761462,14.794034,14.761462–14.826606,0.99561,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,16.878217,16.316642,16.597430,16.316642–16.878217,0.96673,70/73/1,70/73/1,0.928014
   dense-2048,native,130.347935,134.346404,132.347169,130.347935–134.346404,1.03068,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,113.307170,115.556966,114.432068,113.307170–115.556966,1.01986,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,118.402690,113.103144,115.752917,113.103144–118.402690,0.95524,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,124.742797,131.191875,127.967336,124.742797–131.191875,1.05170,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.046344,0.045989,0.046166,0.045989–0.046344,0.99232,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.037699,0.032557,0.035128,0.032557–0.037699,0.86360,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.305907,0.298156,0.302031,0.298156–0.305907,0.97466,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.509519,0.520110,0.514814,0.509519–0.520110,1.02079,49/51/0,49/51/0,0.635707
   dense-256,native,0.546888,0.498870,0.522880,0.498870–0.546888,0.91220,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.377406,0.335495,0.356451,0.335495–0.377406,0.88895,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.946373,0.838908,0.892640,0.838908–0.946373,0.88645,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.182116,1.009961,1.096039,1.009961–1.182116,0.85437,67/70/1,67/70/1,0.972357
   dense-1024,native,12.981480,13.294125,13.137803,12.981480–13.294125,1.02408,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,12.155437,12.528750,12.342094,12.155437–12.528750,1.03071,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,13.520736,13.351536,13.436136,13.351536–13.520736,0.98749,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,15.235684,14.947266,15.091475,14.947266–15.235684,0.98107,70/73/1,70/73/1,0.928014
   dense-2048,native,129.217569,153.609262,141.413416,129.217569–153.609262,1.18876,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,106.649608,111.735711,109.192660,106.649608–111.735711,1.04769,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,111.037550,114.455608,112.746579,111.037550–114.455608,1.03078,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,106.761567,120.937522,113.849544,106.761567–120.937522,1.13278,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.012939,0.014642,0.013791,0.012939–0.014642,1.13161,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.014404,0.017907,0.016156,0.014404–0.017907,1.24323,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.044361,0.072469,0.058415,0.044361–0.072469,1.63362,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.077462,0.135730,0.106595,0.077462–0.135730,1.75222,5/7/0,5/7/0,0.0796191
   dense-256,native,0.060484,0.065600,0.063042,0.060484–0.065600,1.08457,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.042458,0.066055,0.054256,0.042458–0.066055,1.55579,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.090227,0.109854,0.100041,0.090227–0.109854,1.21753,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.154482,0.205096,0.179789,0.154482–0.205096,1.32764,4/6/0,4/6/0,0.209854
   dense-1024,native,0.968295,1.181615,1.074955,0.968295–1.181615,1.22030,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,0.955680,1.133167,1.044423,0.955680–1.133167,1.18572,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.089816,1.184227,1.137021,1.089816–1.184227,1.08663,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.312776,1.593068,1.452922,1.312776–1.593068,1.21351,4/6/0,4/6/0,0.013684
   dense-2048,native,7.716738,7.266113,7.491425,7.266113–7.716738,0.94160,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,6.607337,6.433462,6.520399,6.433462–6.607337,0.97368,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,6.792844,6.813153,6.802998,6.792844–6.813153,1.00299,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,7.705364,8.421064,8.063214,7.705364–8.421064,1.09288,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.018052,0.021626,0.019839,0.018052–0.021626,1.19802,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.019854,0.024927,0.022390,0.019854–0.024927,1.25549,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.057590,0.086868,0.072229,0.057590–0.086868,1.50840,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.121066,0.162775,0.141921,0.121066–0.162775,1.34452,5/7/0,5/7/0,0.0796191
   dense-256,native,0.051174,0.075386,0.063280,0.051174–0.075386,1.47312,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.041072,0.063065,0.052069,0.041072–0.063065,1.53549,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.077878,0.112239,0.095058,0.077878–0.112239,1.44122,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.144300,0.206072,0.175186,0.144300–0.206072,1.42808,4/6/0,4/6/0,0.209854
   dense-1024,native,0.952948,1.114768,1.033858,0.952948–1.114768,1.16981,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,0.928964,1.065311,0.997138,0.928964–1.065311,1.14677,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.070072,1.296002,1.183037,1.070072–1.296002,1.21114,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.338801,1.500650,1.419726,1.338801–1.500650,1.12089,4/6/0,4/6/0,0.013684
   dense-2048,native,7.480561,8.134920,7.807740,7.480561–8.134920,1.08747,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,5.971742,8.342902,7.157322,5.971742–8.342902,1.39706,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,6.138254,9.210529,7.674391,6.138254–9.210529,1.50051,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,7.366924,9.862422,8.614673,7.366924–9.862422,1.33874,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.009880,0.008651,0.009266,0.008651–0.009880,0.87566,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.009879,0.008495,0.009187,0.008495–0.009879,0.85990,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.057813,0.049426,0.053620,0.049426–0.057813,0.85492,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.109567,0.080148,0.094858,0.080148–0.109567,0.73150,5/7/0,5/7/0,0.0796191
   dense-256,native,0.041719,0.057737,0.049728,0.041719–0.057737,1.38393,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.033060,0.050499,0.041779,0.033060–0.050499,1.52746,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.065851,0.099066,0.082458,0.065851–0.099066,1.50440,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.084765,0.150554,0.117659,0.084765–0.150554,1.77613,4/6/0,4/6/0,0.209854
   dense-1024,native,1.575236,1.664069,1.619652,1.575236–1.664069,1.05639,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,1.256692,1.657712,1.457202,1.256692–1.657712,1.31911,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.520336,1.674641,1.597488,1.520336–1.674641,1.10149,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.740174,2.007630,1.873902,1.740174–2.007630,1.15370,4/6/0,4/6/0,0.013684
   dense-2048,native,10.000932,10.852593,10.426763,10.000932–10.852593,1.08516,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,9.810232,10.174018,9.992125,9.810232–10.174018,1.03708,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.603800,8.310184,8.456992,8.310184–8.603800,0.96587,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,8.327561,8.400799,8.364180,8.327561–8.400799,1.00879,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.007410,0.007789,0.007599,0.007410–0.007789,1.05122,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.007430,0.007502,0.007466,0.007430–0.007502,1.00956,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.031887,0.032828,0.032358,0.031887–0.032828,1.02949,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.050237,0.053567,0.051902,0.050237–0.053567,1.06628,5/7/0,5/7/0,0.0796191
   dense-256,native,0.043449,0.058312,0.050880,0.043449–0.058312,1.34207,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.041952,0.058560,0.050256,0.041952–0.058560,1.39588,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.060811,0.087112,0.073962,0.060811–0.087112,1.43251,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.082849,0.148594,0.115722,0.082849–0.148594,1.79357,4/6/0,4/6/0,0.209854
   dense-1024,native,1.731756,1.996609,1.864183,1.731756–1.996609,1.15294,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,1.521902,1.585338,1.553620,1.521902–1.585338,1.04168,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.790915,1.661606,1.726260,1.661606–1.790915,0.92780,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,2.032251,2.223143,2.127697,2.032251–2.223143,1.09393,4/6/0,4/6/0,0.013684
   dense-2048,native,10.569294,11.735644,11.152470,10.569294–11.735644,1.11035,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,10.259472,10.635735,10.447604,10.259472–10.635735,1.03667,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,10.515744,10.102151,10.308947,10.102151–10.515744,0.96067,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,9.604243,10.777135,10.190689,9.604243–10.777135,1.12212,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.025326,0.026416,0.025871,0.025326–0.026416,1.04300,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.259256,0.198120,0.228688,0.198120–0.259256,0.76419,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.362777,0.247992,0.305385,0.247992–0.362777,0.68359,26/28/0,26/28/0,0.579621
   poisson-32,native,0.437107,0.426339,0.431723,0.426339–0.437107,0.97536,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.766803,1.587707,1.677255,1.587707–1.766803,0.89863,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.635541,1.565416,1.600478,1.565416–1.635541,0.95712,118/122/2,118/122/2,0.840189
   poisson-64,native,2.527417,2.383729,2.455573,2.383729–2.527417,0.94315,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,6.358015,5.602874,5.980444,5.602874–6.358015,0.88123,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.467104,4.652737,5.059921,4.652737–5.467104,0.85104,201/207/4,201/207/4,0.907427
   poisson-128,native,28.823542,29.507218,29.165380,28.823542–29.507218,1.02372,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,59.529992,58.415465,58.972729,58.415465–59.529992,0.98128,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,42.324429,43.920038,43.122234,42.324429–43.920038,1.03770,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.012954,0.014412,0.013683,0.012954–0.014412,1.11251,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.183952,0.185022,0.184487,0.183952–0.185022,1.00581,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.232786,0.248907,0.240846,0.232786–0.248907,1.06925,26/28/0,26/28/0,0.579621
   poisson-32,native,0.341132,0.396581,0.368856,0.341132–0.396581,1.16254,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.311948,1.481064,1.396506,1.311948–1.481064,1.12890,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.388529,1.552283,1.470406,1.388529–1.552283,1.11793,118/122/2,118/122/2,0.840189
   poisson-64,native,2.200787,2.319198,2.259993,2.200787–2.319198,1.05380,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.378894,5.205381,5.292137,5.205381–5.378894,0.96774,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.476667,4.651119,4.563893,4.476667–4.651119,1.03897,201/207/4,201/207/4,0.907427
   poisson-128,native,29.496267,31.969688,30.732978,29.496267–31.969688,1.08386,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,58.979563,58.020257,58.499910,58.020257–58.979563,0.98373,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,43.320298,44.443843,43.882070,43.320298–44.443843,1.02594,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.032274,0.035319,0.033797,0.032274–0.035319,1.09436,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.238952,0.256290,0.247621,0.238952–0.256290,1.07256,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.354777,0.391651,0.373214,0.354777–0.391651,1.10394,26/28/0,26/28/0,0.579621
   poisson-32,native,0.470132,0.503678,0.486905,0.470132–0.503678,1.07135,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.658735,1.697019,1.677877,1.658735–1.697019,1.02308,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.870100,1.973956,1.922028,1.870100–1.973956,1.05553,118/122/2,118/122/2,0.840189
   poisson-64,native,2.968628,2.808124,2.888376,2.808124–2.968628,0.94593,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,6.954732,6.537453,6.746092,6.537453–6.954732,0.94000,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.779999,6.147450,5.963724,5.779999–6.147450,1.06357,201/207/4,201/207/4,0.907427
   poisson-128,native,34.972147,37.282167,36.127157,34.972147–37.282167,1.06605,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,67.721823,73.170770,70.446296,67.721823–73.170770,1.08046,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,53.229109,59.231209,56.230159,53.229109–59.231209,1.11276,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.018916,0.016609,0.017763,0.016609–0.018916,0.87802,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.286291,0.216630,0.251460,0.216630–0.286291,0.75668,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.396840,0.314812,0.355826,0.314812–0.396840,0.79330,26/28/0,26/28/0,0.579621
   poisson-32,native,0.465917,0.457056,0.461487,0.457056–0.465917,0.98098,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.885886,1.672838,1.779362,1.672838–1.885886,0.88703,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,2.111523,1.853050,1.982286,1.853050–2.111523,0.87759,118/122/2,118/122/2,0.840189
   poisson-64,native,2.880112,2.933520,2.906816,2.880112–2.933520,1.01854,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,7.125383,6.400830,6.763107,6.400830–7.125383,0.89831,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,6.464707,6.453167,6.458937,6.453167–6.464707,0.99821,201/207/4,201/207/4,0.907427
   poisson-128,native,33.510543,33.933828,33.722186,33.510543–33.933828,1.01263,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,67.269954,65.516504,66.393230,65.516504–67.269954,0.97393,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,53.587795,51.643910,52.615853,51.643910–53.587795,0.96373,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.063271,0.061144,0.062208,0.061144–0.063271,0.96639,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.205037,0.205723,0.205379,0.205037–0.205723,1.00335,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.316026,0.304739,0.310383,0.304739–0.316026,0.96428,49/51/0,49/51/0,0.635707
   dense-256,native,0.364721,0.411164,0.387943,0.364721–0.411164,1.12734,67/70/1,67/70/1,0.993806
   dense-256,numpy-out,0.662627,1.082239,0.872433,0.662627–1.082239,1.63326,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.818121,1.148995,0.983558,0.818121–1.148995,1.40443,67/70/1,67/70/1,0.972357
   dense-1024,native,10.965793,13.600523,12.283158,10.965793–13.600523,1.24027,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,10.896076,12.137645,11.516861,10.896076–12.137645,1.11395,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,12.248268,17.115843,14.682056,12.248268–17.115843,1.39741,70/73/1,70/73/1,0.928014
   dense-2048,native,93.139776,88.198544,90.669160,88.198544–93.139776,0.94695,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,71.250376,72.665762,71.958069,71.250376–72.665762,1.01986,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,99.255246,83.098213,91.176729,83.098213–99.255246,0.83722,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.055848,0.054640,0.055244,0.054640–0.055848,0.97838,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.203923,0.201829,0.202876,0.201829–0.203923,0.98973,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.314262,0.309256,0.311759,0.309256–0.314262,0.98407,49/51/0,49/51/0,0.635707
   dense-256,native,0.348855,0.369202,0.359028,0.348855–0.369202,1.05832,67/70/1,67/70/1,0.993806
   dense-256,numpy-out,0.826943,0.872141,0.849542,0.826943–0.872141,1.05466,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.887800,1.057684,0.972742,0.887800–1.057684,1.19135,67/70/1,67/70/1,0.972357
   dense-1024,native,13.883033,15.194865,14.538949,13.883033–15.194865,1.09449,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,13.078618,13.136135,13.107376,13.078618–13.136135,1.00440,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,16.583150,17.675429,17.129290,16.583150–17.675429,1.06587,70/73/1,70/73/1,0.928014
   dense-2048,native,108.289352,95.135807,101.712580,95.135807–108.289352,0.87853,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,81.933498,69.588465,75.760981,69.588465–81.933498,0.84933,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,94.660541,94.577527,94.619034,94.577527–94.660541,0.99912,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.021537,0.019845,0.020691,0.019845–0.021537,0.92144,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.044631,0.044168,0.044400,0.044168–0.044631,0.98964,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.078053,0.071923,0.074988,0.071923–0.078053,0.92146,5/7/0,5/7/0,0.0796191
   dense-256,native,0.115130,0.043369,0.079250,0.043369–0.115130,0.37670,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.084996,0.068052,0.076524,0.068052–0.084996,0.80064,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.261472,0.104804,0.183138,0.104804–0.261472,0.40083,4/6/0,4/6/0,0.209854
   dense-1024,native,1.065217,1.009434,1.037325,1.009434–1.065217,0.94763,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.167746,1.105117,1.136432,1.105117–1.167746,0.94637,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.312561,1.205148,1.258855,1.205148–1.312561,0.91817,4/6/0,4/6/0,0.013684
   dense-2048,native,10.471840,10.139496,10.305668,10.139496–10.471840,0.96826,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,7.222506,8.187473,7.704990,7.222506–8.187473,1.13361,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,9.166802,8.301989,8.734395,8.301989–9.166802,0.90566,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.011655,0.011732,0.011693,0.011655–0.011732,1.00656,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.032245,0.030433,0.031338,0.030433–0.032245,0.94380,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.050005,0.049415,0.049710,0.049415–0.050005,0.98821,5/7/0,5/7/0,0.0796191
   dense-256,native,0.070796,0.036514,0.053656,0.036514–0.070796,0.51577,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.063011,0.060200,0.061605,0.060200–0.063011,0.95538,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.174957,0.091398,0.133177,0.091398–0.174957,0.52240,4/6/0,4/6/0,0.209854
   dense-1024,native,1.982139,0.975012,1.478576,0.975012–1.982139,0.49190,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.698503,1.468522,1.583512,1.468522–1.698503,0.86460,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,2.374221,1.360953,1.867587,1.360953–2.374221,0.57322,4/6/0,4/6/0,0.013684
   dense-2048,native,11.332840,13.655945,12.494393,11.332840–13.655945,1.20499,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,10.422716,11.330873,10.876795,10.422716–11.330873,1.08713,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,13.696452,15.917632,14.807042,13.696452–15.917632,1.16217,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.029795,0.059320,0.044557,0.029795–0.059320,1.99094,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.147498,0.287226,0.217362,0.147498–0.287226,1.94732,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.224794,0.427529,0.326161,0.224794–0.427529,1.90187,26/28/0,26/28/0,0.579621
   poisson-32,native,0.429674,0.605009,0.517342,0.429674–0.605009,1.40807,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.119220,1.848999,1.484109,1.119220–1.848999,1.65204,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.274064,1.916092,1.595078,1.274064–1.916092,1.50392,118/122/2,118/122/2,0.840189
   poisson-64,native,2.427446,2.653018,2.540232,2.427446–2.653018,1.09293,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.150322,6.213375,5.181849,4.150322–6.213375,1.49708,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,3.999192,5.606717,4.802954,3.999192–5.606717,1.40196,201/207/4,201/207/4,0.907427
   poisson-128,native,32.011201,29.172898,30.592049,29.172898–32.011201,0.91133,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,75.547713,76.871238,76.209475,75.547713–76.871238,1.01752,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,42.427236,46.730128,44.578682,42.427236–46.730128,1.10142,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.023658,0.027541,0.025599,0.023658–0.027541,1.16416,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.147366,0.218642,0.183004,0.147366–0.218642,1.48367,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.446858,0.313082,0.379970,0.313082–0.446858,0.70063,26/28/0,26/28/0,0.579621
   poisson-32,native,0.427194,0.507199,0.467197,0.427194–0.507199,1.18728,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.170940,1.594799,1.382869,1.170940–1.594799,1.36198,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.265589,1.607089,1.436339,1.265589–1.607089,1.26984,118/122/2,118/122/2,0.840189
   poisson-64,native,2.578861,2.497129,2.537995,2.497129–2.578861,0.96831,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.151237,5.483015,4.817126,4.151237–5.483015,1.32081,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,3.907198,4.560803,4.234000,3.907198–4.560803,1.16728,201/207/4,201/207/4,0.907427
   poisson-128,native,29.158123,31.872338,30.515231,29.158123–31.872338,1.09309,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,78.988898,75.770064,77.379481,75.770064–78.988898,0.95925,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,44.568305,48.238836,46.403570,44.568305–48.238836,1.08236,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.039215,0.042877,0.041046,0.039215–0.042877,1.09338,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.185137,0.189256,0.187196,0.185137–0.189256,1.02225,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.294318,0.602969,0.448643,0.294318–0.602969,2.04870,26/28/0,26/28/0,0.579621
   poisson-32,native,0.537406,0.517035,0.527220,0.517035–0.537406,0.96209,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.344966,1.328162,1.336564,1.328162–1.344966,0.98751,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.748533,1.599521,1.674027,1.599521–1.748533,0.91478,118/122/2,118/122/2,0.840189
   poisson-64,native,3.015597,2.872832,2.944214,2.872832–3.015597,0.95266,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.808878,4.657072,4.732975,4.657072–4.808878,0.96843,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.272380,5.197838,5.235109,5.197838–5.272380,0.98586,201/207/4,201/207/4,0.907427
   poisson-128,native,34.588732,38.069504,36.329118,34.588732–38.069504,1.10063,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,82.637017,84.796574,83.716795,82.637017–84.796574,1.02613,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,50.418396,52.606243,51.512319,50.418396–52.606243,1.04339,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.028361,0.047122,0.037741,0.028361–0.047122,1.66151,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.168113,0.256084,0.212098,0.168113–0.256084,1.52328,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.385250,0.430345,0.407797,0.385250–0.430345,1.11705,26/28/0,26/28/0,0.579621
   poisson-32,native,0.488675,0.602248,0.545462,0.488675–0.602248,1.23241,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.302156,1.758353,1.530254,1.302156–1.758353,1.35034,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.526337,1.907891,1.717114,1.526337–1.907891,1.24998,118/122/2,118/122/2,0.840189
   poisson-64,native,2.834501,3.070550,2.952526,2.834501–3.070550,1.08328,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.680625,6.183395,5.432010,4.680625–6.183395,1.32106,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.093900,5.818389,5.456144,5.093900–5.818389,1.14223,201/207/4,201/207/4,0.907427
   poisson-128,native,32.379157,35.703153,34.041155,32.379157–35.703153,1.10266,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,62.960403,81.457278,72.208840,62.960403–81.457278,1.29379,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,40.160783,51.759282,45.960032,40.160783–51.759282,1.28880,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-sparse-solvers-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   scaled-poisson-2d-16,native-cg,0.534712,0.597830,0.566271,0.534712–0.597830,1.11804,597/610/11,597/610/11,0.970433
   scaled-poisson-2d-16,native-pcg,0.094597,0.089889,0.092243,0.089889–0.094597,0.95023,48/50/0,48/50/0,0.91593
   scaled-poisson-2d-16,scipy-cg,3.460262,3.544083,3.502172,3.460262–3.544083,1.02422,276/277/unknown,276/277/unknown,0.846418
   scaled-poisson-2d-16,scipy-pcg,0.765618,0.809412,0.787515,0.765618–0.809412,1.05720,48/49/unknown,48/49/unknown,0.91593
   scaled-poisson-2d-16,scipy-lu-total,0.449873,0.463500,0.456687,0.449873–0.463500,1.03029,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-16,scipy-lu-reuse,0.051803,0.051222,0.051512,0.051222–0.051803,0.98877,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-32,native-cg,5.548388,5.988370,5.768379,5.548388–5.988370,1.07930,1799/1836/35,1799/1836/35,0.976976
   scaled-poisson-2d-32,native-pcg,0.506568,0.589744,0.548156,0.506568–0.589744,1.16419,118/122/2,118/122/2,0.909328
   scaled-poisson-2d-32,scipy-cg,10.719684,11.629991,11.174838,10.719684–11.629991,1.08492,633/634/unknown,633/634/unknown,0.899425
   scaled-poisson-2d-32,scipy-pcg,1.937689,2.084300,2.010995,1.937689–2.084300,1.07566,90/91/unknown,90/91/unknown,0.843882
   scaled-poisson-2d-32,scipy-lu-total,1.737091,1.963499,1.850295,1.737091–1.963499,1.13034,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-2d-32,scipy-lu-reuse,0.154951,0.168884,0.161918,0.154951–0.168884,1.08992,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-3d-6,native-cg,0.238012,0.256877,0.247445,0.238012–0.256877,1.07926,253/260/5,253/260/5,0.876984
   scaled-poisson-3d-6,native-pcg,0.058056,0.066261,0.062159,0.058056–0.066261,1.14132,24/26/0,24/26/0,0.71185
   scaled-poisson-3d-6,scipy-cg,2.214169,2.532484,2.373327,2.214169–2.532484,1.14376,180/181/unknown,180/181/unknown,0.837854
   scaled-poisson-3d-6,scipy-pcg,0.464274,0.488905,0.476589,0.464274–0.488905,1.05305,24/25/unknown,24/25/unknown,0.71185
   scaled-poisson-3d-6,scipy-lu-total,0.492134,0.525659,0.508897,0.492134–0.525659,1.06812,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-6,scipy-lu-reuse,0.046690,0.055473,0.051081,0.046690–0.055473,1.18813,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-10,native-cg,1.537725,1.663949,1.600837,1.537725–1.663949,1.08208,387/396/7,387/396/7,0.977579
   scaled-poisson-3d-10,native-pcg,0.238577,0.256388,0.247483,0.238577–0.256388,1.07465,37/39/0,37/39/0,0.956242
   scaled-poisson-3d-10,scipy-cg,5.119989,5.974416,5.547203,5.119989–5.974416,1.16688,296/297/unknown,296/297/unknown,0.995554
   scaled-poisson-3d-10,scipy-pcg,0.859248,0.986248,0.922748,0.859248–0.986248,1.14780,37/38/unknown,37/38/unknown,0.956242
   scaled-poisson-3d-10,scipy-lu-total,4.305481,4.315884,4.310683,4.305481–4.315884,1.00242,0/1/unknown,0/1/unknown,3.15518e-08
   scaled-poisson-3d-10,scipy-lu-reuse,0.264462,0.294227,0.279344,0.264462–0.294227,1.11255,0/1/unknown,0/1/unknown,3.15518e-08
   poisson-2d-16-neutral,native-cg,0.068999,0.076669,0.072834,0.068999–0.076669,1.11115,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,native-pcg,0.083746,0.104326,0.094036,0.083746–0.104326,1.24574,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,scipy-cg,0.577333,0.915670,0.746502,0.577333–0.915670,1.58603,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-pcg,0.646484,0.915754,0.781119,0.646484–0.915754,1.41651,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-lu-total,0.334745,0.419540,0.377142,0.334745–0.419540,1.25331,0/1/unknown,0/1/unknown,1.84387e-08
   poisson-2d-16-neutral,scipy-lu-reuse,0.041099,0.046487,0.043793,0.041099–0.046487,1.13111,0/1/unknown,0/1/unknown,1.84387e-08

.. csv-table:: cpu-release-sparse-solvers-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   scaled-poisson-2d-16,native-cg,0.517064,0.606966,0.562015,0.517064–0.606966,1.17387,597/610/11,597/610/11,0.970433
   scaled-poisson-2d-16,native-pcg,0.070279,0.079787,0.075033,0.070279–0.079787,1.13529,48/50/0,48/50/0,0.91593
   scaled-poisson-2d-16,scipy-cg,3.072085,3.771473,3.421779,3.072085–3.771473,1.22766,276/277/unknown,276/277/unknown,0.846418
   scaled-poisson-2d-16,scipy-pcg,0.637318,0.773864,0.705591,0.637318–0.773864,1.21425,48/49/unknown,48/49/unknown,0.91593
   scaled-poisson-2d-16,scipy-lu-total,0.439088,0.458236,0.448662,0.439088–0.458236,1.04361,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-16,scipy-lu-reuse,0.053377,0.054372,0.053874,0.053377–0.054372,1.01864,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-32,native-cg,5.389068,6.038965,5.714016,5.389068–6.038965,1.12060,1799/1836/35,1799/1836/35,0.976976
   scaled-poisson-2d-32,native-pcg,0.484953,0.519577,0.502265,0.484953–0.519577,1.07140,118/122/2,118/122/2,0.909328
   scaled-poisson-2d-32,scipy-cg,9.558303,11.614915,10.586609,9.558303–11.614915,1.21516,633/634/unknown,633/634/unknown,0.899425
   scaled-poisson-2d-32,scipy-pcg,1.660964,2.088292,1.874628,1.660964–2.088292,1.25728,90/91/unknown,90/91/unknown,0.843882
   scaled-poisson-2d-32,scipy-lu-total,1.852135,2.122677,1.987406,1.852135–2.122677,1.14607,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-2d-32,scipy-lu-reuse,0.169090,0.200602,0.184846,0.169090–0.200602,1.18637,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-3d-6,native-cg,0.217884,0.237346,0.227615,0.217884–0.237346,1.08933,253/260/5,253/260/5,0.876984
   scaled-poisson-3d-6,native-pcg,0.040835,0.037076,0.038955,0.037076–0.040835,0.90796,24/26/0,24/26/0,0.71185
   scaled-poisson-3d-6,scipy-cg,2.061424,2.293613,2.177518,2.061424–2.293613,1.11264,180/181/unknown,180/181/unknown,0.837854
   scaled-poisson-3d-6,scipy-pcg,0.353809,0.411241,0.382525,0.353809–0.411241,1.16232,24/25/unknown,24/25/unknown,0.71185
   scaled-poisson-3d-6,scipy-lu-total,0.463294,0.521690,0.492492,0.463294–0.521690,1.12605,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-6,scipy-lu-reuse,0.038492,0.044348,0.041420,0.038492–0.044348,1.15212,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-10,native-cg,1.371450,1.737801,1.554626,1.371450–1.737801,1.26713,387/396/7,387/396/7,0.977579
   scaled-poisson-3d-10,native-pcg,0.188773,0.234893,0.211833,0.188773–0.234893,1.24432,37/39/0,37/39/0,0.956242
   scaled-poisson-3d-10,scipy-cg,4.411201,5.671074,5.041138,4.411201–5.671074,1.28561,296/297/unknown,296/297/unknown,0.995554
   scaled-poisson-3d-10,scipy-pcg,0.715481,1.036435,0.875958,0.715481–1.036435,1.44858,37/38/unknown,37/38/unknown,0.956242
   scaled-poisson-3d-10,scipy-lu-total,3.820412,4.914178,4.367295,3.820412–4.914178,1.28630,0/1/unknown,0/1/unknown,3.15518e-08
   scaled-poisson-3d-10,scipy-lu-reuse,0.250433,0.308652,0.279542,0.250433–0.308652,1.23248,0/1/unknown,0/1/unknown,3.15518e-08
   poisson-2d-16-neutral,native-cg,0.055699,0.047274,0.051486,0.047274–0.055699,0.84874,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,native-pcg,0.063313,0.056462,0.059888,0.056462–0.063313,0.89180,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,scipy-cg,0.590382,0.485745,0.538064,0.485745–0.590382,0.82276,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-pcg,0.635419,0.565505,0.600462,0.565505–0.635419,0.88997,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-lu-total,0.317781,0.307580,0.312680,0.307580–0.317781,0.96790,0/1/unknown,0/1/unknown,1.84387e-08
   poisson-2d-16-neutral,scipy-lu-reuse,0.029965,0.027599,0.028782,0.027599–0.029965,0.92106,0/1/unknown,0/1/unknown,1.84387e-08

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,6.269868,6.455184,6.362526,6.269868–6.455184,1.02956,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,8.358752,9.695800,9.027276,8.358752–9.695800,1.15996,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,15.577025,16.159938,15.868482,15.577025–16.159938,1.03742,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,9.144025,9.188360,9.166193,9.144025–9.188360,1.00485,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,11.574677,12.131001,11.852839,11.574677–12.131001,1.04806,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,21.514676,22.260022,21.887349,21.514676–22.260022,1.03464,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,14.692116,12.655358,13.673737,12.655358–14.692116,0.86137,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,14.241922,13.583410,13.912666,13.583410–14.241922,0.95376,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.149862,22.703437,21.926649,21.149862–22.703437,1.07346,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,20.681981,21.198042,20.940011,20.681981–21.198042,1.02495,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,21.002477,21.119265,21.060871,21.002477–21.119265,1.00556,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.129382,27.820101,27.974742,27.820101–28.129382,0.98901,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.574724,4.467877,4.521300,4.467877–4.574724,0.97664,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,6.300969,6.894224,6.597596,6.300969–6.894224,1.09415,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.270546,16.049218,16.159882,16.049218–16.270546,0.98640,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,6.466888,7.245254,6.856071,6.466888–7.245254,1.12036,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,9.271701,9.300287,9.285994,9.271701–9.300287,1.00308,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.872080,22.278460,21.575270,20.872080–22.278460,1.06738,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,8.890242,9.414016,9.152129,8.890242–9.414016,1.05892,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,10.138445,10.608897,10.373671,10.138445–10.608897,1.04640,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.482633,21.033517,21.258075,21.033517–21.482633,0.97909,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,18.176967,17.638852,17.907910,17.638852–18.176967,0.97040,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,19.418498,19.443857,19.431177,19.418498–19.443857,1.00131,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,30.213943,28.785822,29.499882,28.785822–30.213943,0.95273,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,6.390088,6.678574,6.534331,6.390088–6.678574,1.04515,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,7.050249,7.140918,7.095584,7.050249–7.140918,1.01286,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.175825,16.260447,16.218136,16.175825–16.260447,1.00523,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,11.215874,9.572181,10.394028,9.572181–11.215874,0.85345,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,9.448244,9.791449,9.619847,9.448244–9.791449,1.03632,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,23.218738,22.618169,22.918453,22.618169–23.218738,0.97413,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,13.059845,14.301699,13.680772,13.059845–14.301699,1.09509,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,13.133909,12.029058,12.581483,12.029058–13.133909,0.91588,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,22.941660,22.123548,22.532604,22.123548–22.941660,0.96434,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,21.619573,21.911151,21.765362,21.619573–21.911151,1.01349,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,19.898805,20.447979,20.173392,19.898805–20.447979,1.02760,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.666466,29.008131,29.337298,29.008131–29.666466,0.97781,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.632907,4.391675,4.512291,4.391675–4.632907,0.94793,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,5.534070,4.708995,5.121532,4.708995–5.534070,0.85091,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.487116,15.204584,15.845850,15.204584–16.487116,0.92221,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,6.558118,6.589540,6.573829,6.558118–6.589540,1.00479,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,7.504784,6.673646,7.089214,6.673646–7.504784,0.88925,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.278962,19.969123,20.124042,19.969123–20.278962,0.98472,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,9.545454,8.515592,9.030523,8.515592–9.545454,0.89211,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,10.033996,8.542699,9.288348,8.542699–10.033996,0.85138,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.216919,19.953305,20.585112,19.953305–21.216919,0.94044,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,18.392514,17.585358,17.988936,17.585358–18.392514,0.95611,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,17.605696,16.847078,17.226387,16.847078–17.605696,0.95691,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.526996,27.630161,28.578578,27.630161–29.526996,0.93576,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,7.068707,6.610287,6.839497,6.610287–7.068707,0.93515,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,7.620498,6.534897,7.077697,6.534897–7.620498,0.85754,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,18.092991,15.743479,16.918235,15.743479–18.092991,0.87014,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,10.120936,9.058238,9.589587,9.058238–10.120936,0.89500,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,9.971676,8.975055,9.473365,8.975055–9.971676,0.90005,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,22.431897,21.210103,21.820999,21.210103–22.431897,0.94553,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,13.465141,13.510642,13.487892,13.465141–13.510642,1.00338,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,12.509892,11.347420,11.928656,11.347420–12.509892,0.90708,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,22.408939,22.341279,22.375109,22.341279–22.408939,0.99698,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,21.540948,21.946701,21.743824,21.540948–21.946701,1.01884,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,20.024139,21.005498,20.514819,20.024139–21.005498,1.04901,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.149499,30.029636,29.589568,29.149499–30.029636,1.03019,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,5.014330,4.470805,4.742568,4.470805–5.014330,0.89161,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,5.308808,4.814369,5.061589,4.814369–5.308808,0.90686,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.866513,16.743166,16.804839,16.743166–16.866513,0.99269,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,7.790847,7.551984,7.671416,7.551984–7.790847,0.96934,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,8.893674,7.753810,8.323742,7.753810–8.893674,0.87183,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,25.838038,23.760219,24.799129,23.760219–25.838038,0.91958,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,9.068542,9.750016,9.409279,9.068542–9.750016,1.07515,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,8.616156,8.980066,8.798111,8.616156–8.980066,1.04224,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.176460,23.614414,22.395437,21.176460–23.614414,1.11513,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,18.083897,18.408777,18.246337,18.083897–18.408777,1.01797,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,17.617959,17.176741,17.397350,17.176741–17.617959,0.97496,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.017050,29.065641,29.041345,29.017050–29.065641,1.00167,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.211434,1.106880,1.159158,1.106880–1.211434,0.91369,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,3.524750,3.050709,3.287730,3.050709–3.524750,0.86551,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.841259,2.568123,2.704691,2.568123–2.841259,0.90387,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,1.053285,1.111425,1.082355,1.053285–1.111425,1.05520,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,3.147820,3.112236,3.130028,3.112236–3.147820,0.98870,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.508514,2.504999,2.506757,2.504999–2.508514,0.99860,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.305438,1.538012,1.421725,1.305438–1.538012,1.17816,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,3.067125,3.568569,3.317847,3.067125–3.568569,1.16349,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.288226,2.901154,2.594690,2.288226–2.901154,1.26786,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.671071,2.079653,1.875362,1.671071–2.079653,1.24450,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,3.681907,4.034857,3.858382,3.681907–4.034857,1.09586,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.766433,2.842201,2.804317,2.766433–2.842201,1.02739,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.844137,0.801747,0.822942,0.801747–0.844137,0.94978,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,2.823151,2.720280,2.771716,2.720280–2.823151,0.96356,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.543452,2.395989,2.469720,2.395989–2.543452,0.94202,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.737121,0.742673,0.739897,0.737121–0.742673,1.00753,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,2.700133,2.720162,2.710148,2.700133–2.720162,1.00742,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.093466,2.189814,2.141640,2.093466–2.189814,1.04602,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.081598,1.063954,1.072776,1.063954–1.081598,0.98369,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,2.910116,2.849876,2.879996,2.849876–2.910116,0.97930,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.160598,2.169665,2.165131,2.160598–2.169665,1.00420,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.309529,1.349296,1.329413,1.309529–1.349296,1.03037,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,2.911624,3.088907,3.000265,2.911624–3.088907,1.06089,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.304098,2.331212,2.317655,2.304098–2.331212,1.01177,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.166698,1.236079,1.201388,1.166698–1.236079,1.05947,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.368300,1.214211,1.291255,1.214211–1.368300,0.88739,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,3.145095,2.912489,3.028792,2.912489–3.145095,0.92604,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,1.053506,1.072344,1.062925,1.053506–1.072344,1.01788,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,1.037602,1.057431,1.047516,1.037602–1.057431,1.01911,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.463527,2.405117,2.434322,2.405117–2.463527,0.97629,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.458046,1.355160,1.406602,1.355160–1.458046,0.92944,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.376270,1.241769,1.309020,1.241769–1.376270,0.90227,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.867322,2.355589,2.611455,2.355589–2.867322,0.82153,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.802081,1.738736,1.770409,1.738736–1.802081,0.96485,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.732698,1.560234,1.646466,1.560234–1.732698,0.90046,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.786026,2.820343,2.803184,2.786026–2.820343,1.01232,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.871212,0.887842,0.879527,0.871212–0.887842,1.01909,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,0.988029,0.921473,0.954751,0.921473–0.988029,0.93264,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.855711,2.756023,2.805868,2.756023–2.855711,0.96509,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.866282,0.800258,0.833270,0.800258–0.866282,0.92378,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.852172,0.840460,0.846316,0.840460–0.852172,0.98626,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.753281,2.360670,2.556976,2.360670–2.753281,0.85740,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.312025,1.168481,1.240253,1.168481–1.312025,0.89059,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.067850,0.971641,1.019745,0.971641–1.067850,0.90990,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.689312,2.542680,2.615996,2.542680–2.689312,0.94548,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.379028,1.506625,1.442827,1.379028–1.506625,1.09253,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.240682,1.325757,1.283219,1.240682–1.325757,1.06857,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.347926,2.407871,2.377898,2.347926–2.407871,1.02553,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.044819,1.061816,1.053317,1.044819–1.061816,1.01627,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.157545,1.094470,1.126008,1.094470–1.157545,0.94551,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.607155,2.293962,2.450558,2.293962–2.607155,0.87987,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,1.069243,1.297870,1.183556,1.069243–1.297870,1.21382,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.953793,1.032093,0.992942,0.953793–1.032093,1.08209,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.085648,2.256552,2.171100,2.085648–2.256552,1.08194,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.273699,1.353054,1.313377,1.273699–1.353054,1.06230,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.210381,1.136579,1.173480,1.136579–1.210381,0.93903,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.109475,2.010085,2.059780,2.010085–2.109475,0.95288,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.616982,1.583636,1.600310,1.583636–1.616982,0.97938,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.438358,1.423598,1.430978,1.423598–1.438358,0.98974,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.213152,2.096893,2.155022,2.096893–2.213152,0.94747,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.771944,0.730599,0.751271,0.730599–0.771944,0.94644,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,0.877695,0.829494,0.853595,0.829494–0.877695,0.94508,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.338792,2.227499,2.283146,2.227499–2.338792,0.95241,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.706674,0.726784,0.716729,0.706674–0.726784,1.02846,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.747998,0.758755,0.753377,0.747998–0.758755,1.01438,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.916270,1.884136,1.900203,1.884136–1.916270,0.98323,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.077309,1.027431,1.052370,1.027431–1.077309,0.95370,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,0.936435,0.874920,0.905678,0.874920–0.936435,0.93431,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.197793,1.954895,2.076344,1.954895–2.197793,0.88948,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.428058,1.448763,1.438411,1.428058–1.448763,1.01450,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.404498,1.254845,1.329671,1.254845–1.404498,0.89345,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.079083,1.953738,2.016411,1.953738–2.079083,0.93971,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.885093,3.470811,3.677952,3.470811–3.885093,0.89337,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.332026,7.167731,7.749878,7.167731–8.332026,0.86026,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,18.332634,17.693562,18.013098,17.693562–18.332634,0.96514,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,37.590233,33.008491,35.299362,33.008491–37.590233,0.87811,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,30.250225,27.911440,29.080832,27.911440–30.250225,0.92269,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,59.845757,56.827901,58.336829,56.827901–59.845757,0.94957,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,92.288784,92.596307,92.442546,92.288784–92.596307,1.00333,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,191.132182,190.753439,190.942811,190.753439–191.132182,0.99802,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.559531,2.471026,2.515278,2.471026–2.559531,0.96542,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.494133,6.983801,7.238967,6.983801–7.494133,0.93190,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,12.514065,10.576862,11.545464,10.576862–12.514065,0.84520,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,33.674654,33.387313,33.530983,33.387313–33.674654,0.99147,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,21.703441,20.173485,20.938463,20.173485–21.703441,0.92951,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,61.456304,55.143028,58.299666,55.143028–61.456304,0.89727,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,61.918585,59.780676,60.849631,59.780676–61.918585,0.96547,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,178.432959,170.958261,174.695610,170.958261–178.432959,0.95811,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.767923,3.533099,3.650511,3.533099–3.767923,0.93768,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.448449,7.125253,7.286851,7.125253–7.448449,0.95661,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,16.587828,16.482698,16.535263,16.482698–16.587828,0.99366,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,33.575213,32.759799,33.167506,32.759799–33.575213,0.97571,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,29.062711,27.724816,28.393764,27.724816–29.062711,0.95397,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,58.534835,54.955627,56.745231,54.955627–58.534835,0.93885,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,97.000312,94.312532,95.656422,94.312532–97.000312,0.97229,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,191.921350,186.011503,188.966427,186.011503–191.921350,0.96921,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.650469,2.572040,2.611255,2.572040–2.650469,0.97041,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.067220,7.341147,7.704184,7.341147–8.067220,0.91000,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,13.120899,11.646092,12.383496,11.646092–13.120899,0.88760,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,37.923114,35.256024,36.589569,35.256024–37.923114,0.92967,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,20.516031,19.966020,20.241025,19.966020–20.516031,0.97319,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,57.597618,58.968199,58.282909,57.597618–58.968199,1.02380,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,59.537474,59.073042,59.305258,59.073042–59.537474,0.99220,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,171.740273,168.622966,170.181620,168.622966–171.740273,0.98185,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.529113,3.973931,3.751523,3.529113–3.973931,1.12604,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.588621,8.627436,8.108028,7.588621–8.627436,1.13689,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,16.878124,16.663894,16.771009,16.663894–16.878124,0.98731,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,32.972352,32.989258,32.980805,32.972352–32.989258,1.00051,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,27.467398,27.835566,27.651481,27.467398–27.835566,1.01340,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,57.833641,56.870973,57.352307,56.870973–57.833641,0.98335,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,91.217444,91.376964,91.297204,91.217444–91.376964,1.00175,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,181.941115,181.987231,181.964173,181.941115–181.987231,1.00025,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.588757,2.531217,2.559987,2.531217–2.588757,0.97777,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.924133,7.271503,7.597818,7.271503–7.924133,0.91764,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,11.774907,10.625133,11.200020,10.625133–11.774907,0.90235,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,35.064780,32.848205,33.956492,32.848205–35.064780,0.93679,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,20.476464,19.910350,20.193407,19.910350–20.476464,0.97235,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,57.374994,55.168747,56.271871,55.168747–57.374994,0.96155,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,60.620983,59.283214,59.952099,59.283214–60.620983,0.97793,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,179.094012,173.430265,176.262139,173.430265–179.094012,0.96838,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,4.917222,4.944645,4.930934,4.917222–4.944645,1.00558,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,11.312478,10.559404,10.935941,10.559404–11.312478,0.93343,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,23.855392,23.817508,23.836450,23.817508–23.855392,0.99841,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,47.499001,46.666033,47.082517,46.666033–47.499001,0.98246,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,44.505910,43.449912,43.977911,43.449912–44.505910,0.97627,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,90.658630,87.005657,88.832144,87.005657–90.658630,0.95971,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,144.228448,135.515278,139.871863,135.515278–144.228448,0.93959,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,272.165276,255.508382,263.836829,255.508382–272.165276,0.93880,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.013419,3.276654,3.145036,3.013419–3.276654,1.08735,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,12.057346,12.522046,12.289696,12.057346–12.522046,1.03854,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,14.902662,13.686991,14.294826,13.686991–14.902662,0.91843,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,47.935492,42.489023,45.212257,42.489023–47.935492,0.88638,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,23.798264,23.298237,23.548251,23.298237–23.798264,0.97899,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,83.995767,79.299497,81.647632,79.299497–83.995767,0.94409,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,70.821213,66.641038,68.731125,66.641038–70.821213,0.94098,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,262.833375,238.693155,250.763265,238.693155–262.833375,0.90815,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,4.727530,5.082607,4.905068,4.727530–5.082607,1.07511,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.034593,11.091574,10.563083,10.034593–11.091574,1.10533,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,23.029080,25.046150,24.037616,23.029080–25.046150,1.08759,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,45.495182,48.937574,47.216378,45.495182–48.937574,1.07566,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,42.377652,46.052616,44.215134,42.377652–46.052616,1.08672,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,83.706879,89.147672,86.427275,83.706879–89.147672,1.06500,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,138.503123,131.591778,135.047450,131.591778–138.503123,0.95010,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,262.359687,252.849628,257.604657,252.849628–262.359687,0.96375,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.240227,3.095162,3.167695,3.095162–3.240227,0.95523,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.837675,11.948662,11.393169,10.837675–11.948662,1.10251,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,14.637109,13.225475,13.931292,13.225475–14.637109,0.90356,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,45.745833,48.401271,47.073552,45.745833–48.401271,1.05805,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,23.800019,25.067473,24.433746,23.800019–25.067473,1.05325,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,80.961163,83.952953,82.457058,80.961163–83.952953,1.03695,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,62.192920,64.549769,63.371344,62.192920–64.549769,1.03790,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,233.326570,235.204184,234.265377,233.326570–235.204184,1.00805,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.429498,4.937259,5.183378,4.937259–5.429498,0.90934,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.741996,10.127461,10.434729,10.127461–10.741996,0.94279,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,24.197599,23.290596,23.744098,23.290596–24.197599,0.96252,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,47.050039,46.773108,46.911573,46.773108–47.050039,0.99411,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,43.531404,39.209492,41.370448,39.209492–43.531404,0.90072,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,84.338177,78.165466,81.251822,78.165466–84.338177,0.92681,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,129.998343,127.608635,128.803489,127.608635–129.998343,0.98162,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,256.056913,251.328148,253.692530,251.328148–256.056913,0.98153,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.905692,2.744146,2.824919,2.744146–2.905692,0.94440,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.725719,9.886356,10.306037,9.886356–10.725719,0.92174,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,13.556830,12.554493,13.055661,12.554493–13.556830,0.92606,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,49.112409,45.048780,47.080595,45.048780–49.112409,0.91726,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,23.852474,23.135055,23.493765,23.135055–23.852474,0.96992,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,76.604730,81.722741,79.163735,76.604730–81.722741,1.06681,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,66.953125,66.986375,66.969750,66.953125–66.986375,1.00050,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,237.335892,238.938524,238.137208,237.335892–238.938524,1.00675,627/641/12,627/641/12,0.989568

Numerical-work observations
---------------------------

Same-configuration forward/reverse changes: 0.

Cross-scope/mode/preset groups with different observed work: 0.
These groups are descriptive and do not fail the within-configuration gate.
Their complete process observations remain in the analysis JSON.

Reproduction and retained evidence
----------------------------------

Use the project-local installation steps in :doc:`../install` and :doc:`../cuda`.
Build before measuring; never use a frozen stage as BUILD_DIR. CPU and CUDA
use their respective local environments and lock files. Check out the actual
compiled source rather than a later merge head. New runs reproduce fixtures,
scope and schedule contracts, not historical timings. The complete ignored
driver, plans, raw files, logs and import sidecars remain local evidence and
are not downloadable from this documentation. Public Make remains sufficient
to rerun every configuration without the ignored audit tool.

.. code-block:: sh

   git fetch origin tag benchmarks/restart-baseline-20261017
   git switch --detach 1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7
   PROJECT="$PWD"
   export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   unset CUDA_LAUNCH_BLOCKING

Run these 54 commands in order, then in exact reverse order with distinct
output names (``-r02``). Set all six thread variables to two only for the
cpu-openmp rows, and one for every other row. CPU affinity is encoded in each
command. The Make target may configure/build before its worker; for a quiet
measurement, finish all builds first and use the equivalent public cgbench
CLI from the plan, with PYTHONPATH pointing to that completed build.
Compiler executable paths must resolve to the recorded versions; different
toolchains create a new environment rather than reproducing this one.

.. code-block:: sh

   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=sparse-solvers PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-sparse-solvers-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=sparse-solvers PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cpu-release-sparse-solvers-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/restart-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''

Manifest SHA256: ``ff9c2a520b7426d9ea4335a47ae04b6bfa1b38acbb0c2e1cfb03d6944ad5ecef``.
Plan SHA256: ``74916360892a8e8798b9d9de20b0eac61faaef17613985b0c2a7b13fd357aa1d``.
Characterization SHA256: ``9d99ef7a90189780100216e9dda0e36f9030f2c3a72b0f4ff8fdf21524cc3bc7``.
Renderer SHA256: ``af180cd5ecb8e916450024b0d5fabd0f6b7088f9f41e0e720b95ea5552154ab0``.

.. csv-table:: Original stopped-campaign evidence
   :class: benchmark-results
   :header-rows: 1

   Path,SHA256
   build/restart-optimization-20261017/baseline/plan.json,ed77eac6d083b80fa6e18aa652e854c960b1193e1c6771ceb20afc485359661c
   build/restart-optimization-20261017/baseline/manifest.json,6f6c83cc68817bf5b37d96367a97dc86623b3d275bf797494fda94d526a9940d
   build/restart-optimization-20261017/baseline/execution-claim.json,07bc77bb8d558ba04e68dc45d15334d4d0826822d0802cc4a57f06f635b9af1f

.. csv-table:: Frozen audit and worker source provenance
   :class: benchmark-results
   :header-rows: 1

   Path,SHA256
   build/restart-optimization-20261017/run_baseline.py,b368f781a3aa6516e366770b2d1f8c57df04da0f21f1c0274331c913ecc1ba51
   build/restart-optimization-20261017/protocol.py,8814adcd9e90fa800e3654c248b1554fbc19e74a4a773dfcbe61e60187afdc0c
   build/restart-optimization-20261017/worker.py,038fa8530e6ed876f52dcea09497e804840f74ebabb008d4adaa99db24f9456c
   build/restart-optimization-20261017/freeze_stage.py,ea90c787125e9bb2f7977f40a25e5d4fe4e94afbe2a237b2dc3893ca678b68ab
   build/restart-optimization-20261017/README.md,3f308246cf5540f79b24b43bd492cd57b35b6fe7e1f8c1e7a2977f6165cbea14
   build/performance-optimization/ab_runner.py,54668d5459fec4041f125d21d38a6fc73074c5b7dedc539a765a8bde254c7ace
   build/performance-optimization/run_baseline.py,213afaa21053d4f65dea506158dc65d60cc4587033bc388d0efccb362e76233f
   build/restart-optimization-20261017/baseline-audit-repair/run_baseline.py,7173f06e038d6de74596ec06c506136705d3be7a8a346f6d3cb9d235c77d7cff
   build/restart-optimization-20261017/baseline-audit-repair/protocol.py,5276f6ddface2b5011df30c4feea845e4ccd91e90b00d9d8c0bf092315f9dced
   build/restart-optimization-20261017/baseline-audit-repair/worker.py,038fa8530e6ed876f52dcea09497e804840f74ebabb008d4adaa99db24f9456c
   build/restart-optimization-20261017/baseline-audit-repair/freeze_stage.py,ea90c787125e9bb2f7977f40a25e5d4fe4e94afbe2a237b2dc3893ca678b68ab
   build/restart-optimization-20261017/baseline-audit-repair/README.md,7ba0ff2cc21ce90529945aa3c84b13b9a40a4a8d90ab00ad5de9a18e73e2cc55

.. csv-table:: All retained raw/import sidecars
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Imports SHA256,Log SHA256,Reused
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-r01.json,e09c132dddf75e5246c905803e557c914a91ff1c7fa4b2fa0c2ac1f6ceda1d35,db17fc4354eedf73a413ffd0c11fdd422ae5b54c804309e67dc5d5f81cec8a63,7def3c9a304665391c3a3426de90893720e68bbd3e2544c754fa0d3948a4dc81,True
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-r01.json,c7665f7ecc2b9fa954ace21ee9e816eea80be75567472f8b6c8b4c72b1b27dae,63d89cd4ca9bf8d2281945361c0e215b29fa5366bfc11b89d76226799dad3a04,e20e719f06f135b2ab1d1c6faeab7a104554a79418e148423d563a8c9bf9386c,True
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-r01.json,f65fc27cb8456826cbdbb0cd5f784e59981a0540f88a1b6b0481fbc0e892c7c5,f2ccb39ba2fb0b200dc3be6241fb2fe539a7012d6581eab74f5e91ef063dae9d,5c4935516f9824799163e168e7b905ed6c5324d2310d03a09db22e20a871848d,True
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-r01.json,999a407c7a2cfa9224bd1789a0e58e7a6dc2211d2be239147abeb23c1789eb75,adc18e2a6c03565c793133bbbf9ca0835c4fb87370c49b77ce5d2d84dc9110cd,df9d2f657f81de88d6007e346ef92c6422b14c8521f18c31e4f9d67f84b21483,True
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-r01.json,035ce8718526786e034cf17f6ea70eb7366dda48bc6b83a78d9a2bdd1b817607,7a68607f0ae261e6a00ed9307da66757d26398e3206838ea71564850c58995e6,b51580a1c174434b139a0166dca4d1be2b47328346b371b9c01fa2d079a93d87,True
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-r01.json,7f7286bbb648a2c4878138114e5e3ec12701b9fd9d6d5ccd60a16019a93802f7,59609fc6168a39991cde7551c0338ff255a2c9b157a42b08f4cf327a3350b24b,211c65b74396a795acc574b96d333fbf2acfba99dc7acf2cf22aed5f998ed1b7,True
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01.json,4f48bf747d0088647744ed596016618dfd9fc9cb764417eb0d48d82b6565d502,4720cc583f7a62d8d3e0823751de68b4dbeb0e3f9c7b2fc4f50fe17407fbf269,419fa4e4ac2daf282551997b783d652fdfdfbc77c2a6024541823ca55ebf871b,True
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r01.json,baeb04247ff8f4711a89ef747170b5dbac0c44b2dfde2b59404091c7c0c829bf,fa67c329f1e22d28f51bf7fac60c470818653dd875ad7023ece324506dff34cb,c6da7bdaf61275c48ac45d05061582fd69a501b3206a746ccc30e0fb0cf5ca08,True
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-r01.json,ea40bd4d1c5768378e9a158d4c4a9564d91bebd9c713deca69e5bccd2181a9bc,eb9c26b471d7fa55988b6b59bc33d2be41b28bcb2852b202a3970d9f33d8ad0b,bac36a9fc93bb967db9956271441bb831fcca93ddec929ed8418b005d98e2707,True
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-r01.json,bfb4873e655e93735dcda5963d606838137de8c8c8bf3d353acfc24462a03403,5ccce3ed7df2594626f8bf5fba339d6257477d64cbf8882c358d5c79f450f32f,5973bfdf671dba046697d7b115f6c4be37161725fbc604a9ee845c7015b348a8,True
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-r01.json,add507a53bcdcae9b0f50f444ee71175767de011396749a43f0dedc72d4cbedd,8d39d9fbbffe99712e987c3893a246626a935dbfb5794196cd5f154128ae892e,69995985fad349559fde96f7b85196e893a7b7f151cea040fc5526a3bc59472a,True
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-r01.json,b1787f3e8fd8189e60185955c68ad3f88b204d7e602f0a9b158219920afca154,46f186ebe12a991bdf872493834febee1f75177580cf8c24c955ae5101370d1e,c4f090bc5f751dd6296d33028554cb9b09dd2c3051d161903ce4fe8bfa711dec,True
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-r01.json,15615c64d6b03a25c8da9ed7179fd05dfa5de1948e8787c4aa82b2ddd01961ec,fc4b0180c008aaedbfd2d8b7f0b68bb86d0092334d4b9d1e9c802858743730d7,03b6e30a3a3d89694871ff67f75cdf23cb5d950484553ec2cf67b72b17b57452,True
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-r01.json,1faf0822866b14cc8c72e97b72dde086e8a458e1fb7bcdd51db6a63ff11bd7c7,788a266fdb3fd7e632bbf33cbde9ae8f36733c01eeb920555f56329c390a3369,dfce013ba2ec72b132d2279ee6a5a84d26ce9a32f3a7c4a63d3fd5c7df7afc6e,True
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01.json,66c066097b168ab88da1207cb6b172d8f40981ad6aa2e205ada31d867051ddf9,0479137e95864b92dc2e788fecdc41facb725ba0c38f4c03eed068699426b935,edeec24585034731e139331661f0c8b11d8dc0ddadd7a931ffd360dd7b5e9b56,True
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r01.json,c94359a4457d537a11f1cf276a172396780dc5a00a8fdbacfa2ce8a91a172764,5b1e7788c910225cf45f4ec6144182ccfba3d3744d2c45250d75a2aa7f5957b5,337dff78b0044a10a40001483448311e577351b4419cabe8a9663ebd8dfd8598,True
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-r01.json,436e947af5eb9c35dbb0fa250916264fd2479d1cfcda0fb9d4d4342588dcf4fe,9ceb797c714639d04ccd638c8c4647ef89f19dec681a6e104c34bf15e4b450ee,210897dd850023ebf467be4a3025b7fdecc9799bd3d43d7f19e8d76f5200e012,True
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-r01.json,0f2d4648a5186fe6f941ba619b70d99aad950145766cdcce8d9d08ae0cd1df2c,6364df8840f5cc9351d8da5688b595335bd45422b45c9e59e3cf4a6bc5f59197,0036b75495320a305aa068a4c16f7009beac5dfae0dfdd060c2f169dbf07384f,True
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-r01.json,1ebb51975b1c90ecb509b5f650de6e7586058ef271ac7fef60ea89658fb3ba5e,be5d40a7fed3185362e184ac6b86fe2a208eb9b45622b9db227feb7d839cf841,f09358a59421087db41f3d34970e2bb47041750f5159455322872ad42c5295d0,True
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r01.json,5fcbbff7ec0feeb9af54f33282ba15d6cbab82bfde5f263a77cc1787580fd5ec,cd8ecae368d55990ace5da7911bbaf2c3f3dcf3c63d337244172f668ef43c0dd,61eefdd84b18305bc203a633f2349a6b170b86e0754bd6dd17767e8ec922df38,True
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-r01.json,42a2b5c09f84570d63bdfb0b3c6337d5d945ea662ed95293e087322a600d589c,b07753214b8f8bac87f357f6c28ec06c0c43b01a113ae104d3d3c900b70b88cd,88fc59566c3d7146c55975a546fcb6f88205a43e6a02ba0172c85593cb95f222,True
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-r01.json,81ba4817c676af2c7559ee00be6d3d189a750a3b5bcc4c227a5c4cfbc4869923,8ca32e93414a256888b3a3b928185d4c05a1ad5eb3d048b2c856a04cb8016aad,1cd0fbbdb78b23fb1d9208d28828ceaa52a89e2e235d1d35fc4a4e7a5b45f1e4,True
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-r01.json,bfa984c88fc21a1955acace9b91ad59832abe282d43f877b95a3e64c06cb8464,b9399f6ba6807584eeb7754f848e853732bba75f8cbc3247bc1a7e2b80fb613c,2c3de2186249a395981fa4703c498c5c58da5b1997b97cc3624b7f6c9a8c8930,True
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r01.json,02af6b789d5e581054075fc7b7ec1ea9f54901bbd857136e2bd6ff710fa06d8f,ef1270bfe9b6ac2309861396e2ecd8dccf7fe3d94ca3ec536ff47ed307148203,61e65819a49375bf5dd71cae2d869cbd1e9986b5f505fb2d597487b262af270e,True
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-r01.json,3494dc2e4384e77721a218ea2ae6f800410735e6e574ac61c62c320e73448231,19c0c989834d7545add018cd28ac25b515ee24b7e818695df71e62c854f6bc65,03516dcf2165bcec19465839684f4d88e410bc94ac728a391c934f34ec160305,True
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-r01.json,5bdcbd7f7ca89733ee583bc2963ce0dbcdf6656f904a40ea28dc09d6a8983f3c,40f697dfea144001422f3ddd86927beb6e0d0d747b1aaf1d19d2d501d3b28838,a9580abb085a1aaba50110927902faa0e3a891da13d59e1044469b557ceb5ed7,True
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-r01.json,717440675adfbf65df3d6fdcb29e29b16d97101fa7093e64e223e63dfa3c9895,16caa9e458f6c935e02a935cdaf6a1e5518a3e594562ceb889f3817d40513e3d,a1e37de3a813e85a2479904201dfc26764638839f2bd51c177782dd0d22e8832,True
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r01.json,a4c3145fc401832c520f0984f914dff1eb29d4da63fee71bb08b64476aebefbf,da0937058b8f3f2060dea3987f7383a325d6c05cca75d1992f238459541fd1cd,edbfd64eb4b8ed63d9b767cfff64e40604805a0f02fa894c0715d89bdcd1b498,True
   cpu-release-sparse-solvers-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline/raw/cpu-release-sparse-solvers-none-api-fused1-host-r01.json,79833fd5d0bb1794c1a9d147ed110649d93e4e0e58c315052736ea662b3fd4f8,80a48eb822bd1eb8694c7ffee97f007e553f6199ce989395b2973d4153bc4c97,bf4be36880f59f8bb974952343f92dd2b056a3dcf239f04f4e46658aa5925e1a,True
   cpu-release-sparse-solvers-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-sparse-solvers-none-prepared-fused1-host-r01.json,f0ff4b122baa1bcba283c421c29e9ddf71bf535f290519b9f122c399487d71a0,14ab38728041ef68c3c16de44cc2490b6d2de196577f0cd5a528a35c01531cbc,150f98c59b4e154c93b817c6f72c4abb93a6580fea0e11e3a3d7d6a4caef2f56,False
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-r01.json,dd6b2a70b34029edafdbf15c37bbed048467b3d1aae4f34a731470c4bf28f457,d151f0c6590353350c87f081325cb87164d2d84fc85a8aaf4b0d1bc51e5f48a7,8a162dd8e55aca92f4647525a4542c314c33cad8f4e40356355cbd0ba3c65b06,False
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-r01.json,16f61b4c89d655ab428646c95d3ff80fea2e581bafd30213bd6c2d9f77faac3e,13d2b4a783ba6d11a4bf3eaeb9e0022c5ecd0ef6310e3807f4d716468a3d412a,eb35fa03c14e45d05b05dd75c20171724db87a5e6f380a4f50b63405d654c8b1,False
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-r01.json,f3beb50809d18647c1bdb340e157c8177d4bb04547bbc4578dae103b12f23ec4,73128130500f3a74afd5b4b8035daee38670a5d1727753d2b2c3e87b4246d038,065bfc51c10c438dbfd43131a86cba7e98dd85593f7ae141ad1fbdd5744aba11,False
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-r01.json,5c32c5c2210a91c3de17a412ed5cd7ffd330602c3f0930ee15fb8d3fca675d31,ba4bbfba2aba832d92eb9f47d6fb8cf55a9935e86c495a56d340dbb851f17c7a,c4f83dff1dda21cac45d02504946a13578262438d1ec0e7537a192fb88ac7559,False
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-r01.json,e9f8ae90122124bfc1bfe55e961bf09ae27ba2c5beac34aeee48164ea56c2402,27d159b0e04b9c5d4637e91189fa85e18362b33c01f7c2e7110933b159b13efe,29bb74b6fe22a84be4269a28b1fce0500c82a3ce0a3f7cb78ae116aa0a0435c3,False
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-r01.json,e7338bd714c7fd9b86acaec73f3b42a61089037198e357ccbda94a510bc32630,1ec0252f96a5ad9db4eb8459409f342d3cf8d3a77e8202169aa290a83df1d9b3,1641e54dae34ee8599e91b8d0417ce301d50daa2a1777674c7373e68bbf24805,False
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-r01.json,e219c676e385e6c462dfb6243cb1903691a9be3573b3f3139155e93a0a1b6a2e,ed61e244821ca4999bfc68d510b3f22fb93846fa0f5b737e79201d356e66f640,7d542cf1f69e6c9eb6519caadf66637ff3e5cf49c16180a7eb1fbe56f0481051,False
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-r01.json,be1c89c9a633604e0bcd236da64ecc959d027ea48522d4b209b622ba8d7a5875,64bf0f80f3e0c9c63f880dbc91803b4e1291dbe6177f7800b34a0d77c0db3627,791c5609f0af59dd4f53c66986f3984acf41772c30cf48b4cbaaf6d6ddf85561,False
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-r01.json,a3e41fae775f0a0db6460e676a4a768200042d35e280c04e1e860aab90e701dc,6737136c26540f20957470189e8a0d2957235957a60d10d13c7891299475ae05,6c35284644f6259a2e89da815996ce9d026b4ab13bcbcbbae8f0f984449a2a61,False
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-r01.json,897db817abd3a9c3114dcb2cea595ccd1d39002527f163ef37aaa52835f5505a,978bab38409c3c1a9067d5b9eae5a04ccc52210391d7a50d893ba34c87f07bc5,490ab445e06ff35a0630e2f12889229d64195d3dc77bb11e5c9461e3da7b8e98,False
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-r01.json,2f8e9514568c4e382794ea7e2136b4a786df926eb95d9027a3759f189827f23e,6650877489ca3a69f769903e1a86b6eebffe6b5b2308a103e7d50b765c4ec410,0357ccf28e96706af364e0e714f27d075e47e841ffed648115e63b04421978fc,False
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-r01.json,3077d97489d1c6d6a5845ade5c13e808f2b0485bcf7ea8143c34143405682128,a746a0c8d3c3026241adf341ca94efc42fa25963c6c10cba3e9a4c0226ec2955,512a83a21d752655e21dcb783f06b95a41b2385e5cefa26da08e9c4957820300,False
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-r01.json,0a01ddbacdd380ce8b1ac61ce478f70ade5f2cfb70597800a1229fdffd42113f,a877366760fa0c2c05bd6a326bc6ce3ff083028e696d70d2fb04c1262f8f4814,2c347f452fd5491fbe008841f1f9169059f421314293175db463159833767b86,False
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-r01.json,a350e85e4a10e7c9c81cf7e78fe48c4eb479304acccb96b84578baa9c711d9a1,684ab5db397e652ed2039d5af2679f069a83cb3a0f98074c93bde4ef77c9b06f,cb023fb6a3e6fda8794571d7673b7324042bc639a21f8de3d20e3feaf695b7f9,False
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-r01.json,207d0fe95256ded52f3345640f47675c68a53d1c9ca321cd6cbdbb7894cb2e27,4ef36afe33f223cb89abc8fa27cf437b4ee65bcbe143eaabd91595a8dcc47743,1ea0390fec8079c0e95aa3a33eb7dd4af7183f5406163e64cca81123918f8f13,False
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-r01.json,46783582d8d853f5a50fb193c624dc4146d833692567dfc10b68f8929dac5467,848c1d50a526674c9cbf296ddc81f4a16b7abb7041270ac916d31cf0df4b1060,80834e33808c5c7a87d3b2585e9ae4cfa7fb3dffc813554b24778d72a8112017,False
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-r01.json,621676fd81f53868ad7e3357bd52a958c7c2b483c2ff7035d46e4c0b2e782b7b,0617d30d1c83d85c80d777c15fafab2e44e32c0bcfd5315ae9317484ca122260,a29bf26a77c273105e92647b451fb87e0858867212c4de21bc9cfec113fb24ef,False
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-r01.json,4bc3df462863e9c3bab84007a54a2eb7c4ba7b05a2f20dd376deeafa20f9adcd,404822d336d7b03db96a7ddb1eab3b1c8fe2f2aca7ce27f6e1167b1848a981a6,629a9172cd5e5288871f4dafd0a8848baa6f4678840b708a48c8a0f6a0ff6030,False
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-r01.json,33214b53082e621ab71f3175196d70bb31a8db39add542ac64b56651bfcfee7e,e2c36894a1938da4a168dcb3df83890a680da1afd99b2f4ceff76765bf9098ee,9175a514870176a0e42b36ac5a3c76f8f465ace8cd5722cae9b13654f1e64f8a,False
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-r01.json,2425e44bae25705b6bd9755cc55f70b143f7f1ca3f9f542947a1a03380830922,d923ac29d9e15dad0d75e28752a9080e27c694b4700c577d3d98bb228e05f1cb,7d6bcddc304e1e9abdb314798ca5ac08f584abc61aa245ad9ed152a883b35614,False
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-r01.json,765d1eb449724a9344726ce55fcc28b61879accefee3cda1a2086308162c0083,65687f9afe8a7b8749c1b16978db5456f449e37e54b59cca1f800438476ade0c,035a4a4a2db696c1b3b82851447e23ed0d8eee626a034aa955d32b7a7da4a320,False
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-r01.json,1260ea424238f914011e0037bcbcdbe84a210d740004feaf966b475564fb7a05,e6d10b5ef1b956e2e19873e6aa57897145b5e913d8d813329251c58b92b731b4,4eb6033090c427df2b20f9aae941f51021883d5e98bff6e4ba6a15f3c373d5a0,False
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-r01.json,a7e1085990f5de52a9704b48a44263f510c53f523937dcffc5358b67bce28ca5,33f896438beb5364c0d06be2769ca70ef34cdc743793416df27b53fc625aeb7f,5cffaabb812a089c51ca0517ee79601a0d3a2d9e776a600e83c40a9b2186e30e,False
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-r01,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-r01.json,71980ef375976ac92a0ac69af185dc056083eb5423f1fcb956b52fb6c27d826a,11e6981a86a53ce225d6241c8133005a0594e7c0bd1bb65ca71e1d84f325f934,69e1d071eae6cb799cc2543effc4c953fb3317efcc2ff71d2cac679840a4d478,False
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-r02.json,44305fb264cf25e691e02ea257336dc68474c6c2e50a429c7c05a0ce7d67a8b6,fdc0ab7e6df21e11d1ff4a2fafe4739e917cfb57acff4c75367633deb794ce56,0d0770da08a4784403e608dccf0e5c3be68cccf2426d56e1c764681d3254e324,False
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-r02.json,d441b74f1aa7c5118369ddd4564ca16da21112ce0fcc9216012fab599a6f523b,980380a8ac7fa1bac21e7ddd1d9071f5b283dda999cde40f6aa1496852dee9da,f9d951ccaff4524416a5aac8020decc1ead41c4dea34465023db8963de4bd031,False
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-r02.json,1179b9da0396184a4dc14df6137a432b198815e82230fdcd60a4eb9fe4b8e0f8,83187d950d929267fb72241b5562e58477bc625a550702cd448d27c00f4eb88a,9bba6bb5c4497862299dc710ce0e5e2cf3415dcfe18465fb3747eb392003a00b,False
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-r02.json,4151cf1fe8b96de10c998ce5a67ee1d80b45e5d74245ef8e2f439a227f7fccf3,bc693b9c9d79f782762a3cded2724c69d1f850062436a2ec8bbbeba544e875f9,d492411bf1caa279aacd5aee5ff64e094b58a548367f189703a246c68b1d516e,False
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-r02.json,74c8971d68c169299254e25e4d6ca4d728d15c864abfabacd04728bf3006687f,e9bdc2102f1713fe15cbe04e825aed0aec5386a8225d81ebb53d5531434d79f3,a53c43dc56d11c20001ad9d39dfbc0b93dabfe6977345e0b3c900faeba36ad5d,False
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-r02.json,8a326c9e3e6c8239bc030d37541d816488f383a28bbb0b9fcb13b03cb69ee3e5,1f7e780daf59586fbe90ff9c2d3f2b0d0b27967da2a618014905934f62b4d57c,570a1071d746db1035432a69e227ca70dab8f9fea02e0ea5b8f71497e50872f2,False
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-r02.json,d9834deb0f34408f89be53368fa566e29ef92bf49dc03913aa32a890fc77ab6f,445d601c043f7b7a79f1f4f54e0db243e9f1332807bde53a2110e1850f4371cf,cce19ed2e6fbe56a301e27283fd9f4e93fd0ec87fb6128c6ae63e9d5969e1404,False
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-r02.json,511cf883d2b162dd8aeefaac5cfd35cedf0d1671cc88e3c22f65776c89b1a241,902a5278dde5c66752c8ae96444cacf9b11dc669f9ba8d8ab0c81208e0e5c70b,a99114041d082782fb2407acb862781e07422b0a5ea757583ecb19ab859db364,False
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-r02.json,6383cd8c90096b23f3d8fa2a1f5b0fbceab81abb6fa7ad2bb5a9e34cc0494f16,26065f3a38435dfb65f022af32aceebccb991bf80bd6bdd8f56a509b4adfa8b6,ddad6d5d3ef904412e1e886a1d60560f04105b151865aec41470991aaea39a80,False
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-r02.json,886a5c776529d16c7628ed80451e6950d220cbaccf3b5d7528ebe15aaf9fba6a,1da99c2d3696775fa1db9d40d1983e7c74c958be4bc689789a9a62f2cbf9053a,a92db9898c7495398227215e2d89a97a3528b93008e00684a1cd5ee23884ace1,False
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-r02.json,b531397adcf78faa8beda0d0298ccec6cd99ea58d07097772b1e12b4348145a1,d523486a5a7dfa0a84b35af195741f04b0b267b496d40a7f063f7c11d96478b2,7c50a215221c99b19d92b3a20686d8f67cb4a1e93c548b1df2a386e06eb34d89,False
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-r02.json,057b5ba0be1ef6e92721cc3fd554c6736b769b4b13b99ebb896ccc097e8a143c,16009c9ec2a0e2d3c3055cb84abc92338ded46b2d173542821040899f4df461e,c6ec6c6dc8727d57bda036faf4a3856b3788c0a8a550c5c334c740a2d18e5cbd,False
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-r02.json,1f5b37a40686cc15dbc5a6ec58b44683af5292571d6340092072f2b6e56169de,dbd656eaf657c70bbd5f5e1644fb8495879b4060deee6269971bbbfc2028e3b4,7f309070ebcc3d2b6c852a4a8a61b889a0b252c753ddb8754a9eb37b824b8710,False
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-r02.json,5a49c8d5fb344ee87d91486c16d31c554beeb17dc3f91b9344ef80f38f477656,b822833ab1e05ba890aad308e789e7036f710ad9dfd9a367582e47bb57299559,cc207cc7d2e9ac824666126d09cbdc236ab8df4319f965edf6023963ac18d606,False
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-r02.json,c4e5f84985146898f54c3498cb31754f223bac2c484ccaece425204ef8bd26f3,906fbc79be0dcaccf1057e2aee793384df5ac669e4342489807b2e22da308145,10dd39892b93be37b4bbf7c63606637db3fa9f3123a23b0dee33b6bd13224b56,False
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-r02.json,b277d76e7f3a24da6042ef181a0d68ec1331c3e7d3678699e44acc40f0378ba9,6a7642a9a48ddc5d7442e2a617c42f113ed73b13f022bb8325b7650dd34fd039,c959a55ef057f904cd3ed04d883f93d167db91030431e2caccb31dbb1fb7f52b,False
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-r02.json,f229c4792ace4bd42595cdb45ca33d55333b8c7210169e6b3045106822e28133,212c0d2a412f3e27654f516a20c261f7c5095223099775b8fa181e7e0e869785,c84e0a7d28291643963df601b309aa8a157011dd06768029b4005db65f0177f1,False
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-r02.json,2e0e456bc2de9c108e1369cea2eee885bc72ec75168371cddeeade5518dfd494,4b48ac314f4ddd8d38edc780e01437c45aa9ee3fc802b2a9ae0ca3f66fba622a,4c3dd13b3c53bdf2374e14cc25c36359d7910339591230177a19df9c571a19af,False
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-r02.json,a5b4695e4c2a158e56878323edddda9bc4415b648c5081107af956ff131dc433,81bfb7be7cfbf9c9b2d003074589ff7e7e885b544a12a2e5e1f2153a28ea6612,709d7f8041f33f012b36ac5c264641f77a8f3fbafe7ee407c4e93d7284ea1b3f,False
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-r02.json,e85ae0bfafcc2236b3a34b4791e0b4fa6f9a6e4fa639332faac1d545fbcb7b58,b3844fe25f6e842d0fdfb33b36d44b2688afff05f06a5b632673e5778bb6aa56,b23e9e83507aa601fdd56ceb0e31646ab9276764280cba05facfa250acf43e21,False
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-r02.json,bcd2b556e89809025bb1bf780c13c878696555ba5d2a8ed49f4ac8bddad83a45,303f8d56fe2ed2a9a4864e5c89e561490d7ffb18caad9052fdd529053e9a6c2d,f85c2aae5f87dc307d0e1178512fbbc21a4a0914c3e94bfb1e17950f501cf274,False
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-r02.json,8eb81ee197de38e42118ddcf9fee7402fc7d7e4c68df46e8fe1e7eadf397e705,e291d18e1f5953e0e61ca8d3e8ac533bb70d6132b9e288345a9b84ede9623ae6,2524940ae7b6b5a22a46fc3bbaa1682ab0571233f740ba02c65a7a335b848fa5,False
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-r02.json,76f0af277353987439cb0e0c3d2caa0537f2d33b2e3962da7c862c5960d311f7,97ece1be3e458dee3b51884e89f4c3c84c1810b6ceb9081b5a34b18121d0ae86,bea11bf4b4e974b4bf88f15e6f4be519a7592b89c3713b06619fb3ac6e29f4ba,False
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-r02.json,eaab342fa0681bd75ac065cd855141780ef48f4ba5ec783a8c1f295c6104c4f1,49d33b82025ffe85dbfa106c1f99f095295fff43d24e3e4e7c9482ba0ea9fdbc,2da84d6a6006bbae776cf861cbbdb4ab9054a3cda834b7ec070af6fde50fc7c8,False
   cpu-release-sparse-solvers-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-sparse-solvers-none-prepared-fused1-host-r02.json,ec8a33d5e361900e55ea13468b7b010b73b0522c01c7ef1216462c8bcef4eea6,9780ca9813b3fda1e8dfa5cf52ddd0133d0304733aa5b7975bb310d56a78e490,20dbbe33cb796d0c4a527740fdf940ff7760c97275f438bf2738745b90e6a510,False
   cpu-release-sparse-solvers-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-sparse-solvers-none-api-fused1-host-r02.json,46ed639a82e1de8cd19179625f8ecbaf13605b8370ad4ccb9ca13e85587903ab,6b7fb121649aa2a67728eeeab5fe970d26b20e10948f204644049772076b4418,125ff8a9b7de564ce62390885e123205d86a7e61faa075473ba69a76488c6f91,False
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r02.json,aa9df6bee56519fd8efc63222500e3658d797d37d836040444315d52b4e6c02b,46265e9b8e276470c93b7985351088b7d4be46824d2dc39f629938c65849b22d,dcbb8c5e17f8c6c804ef3a6962ac29de445c3dc9316d8c7a6a0d7e8f4963a09d,False
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-r02.json,be10b42b8efb8ff11436f0abfccac9623f21ca46dfbb5ae5d3140f6c4facd18f,d089ff104b31840e8e1efa05cffe0c469af485cfbb19126398f2ed965516396c,ef27e52d06c899c858136dcd713ff78bb9a685f6f20ad3264e2f0991b915a734,False
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-r02.json,86adf8ca262f4aa16628c4ee0638d6c4ac901f062f36352dd96738e7ca56bbf5,dca3778363c70f40141276ccc7f08c216c40eedbf25c0b28d0fd35c36fa0fcef,5ad54394efebf566f8405a6928cf1c50bcc54b3e7f466f068140dbf775f719cd,False
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-r02.json,0a9c4f7810f4ef45f6375209a16991c8552b75887f8a2e1f839623ae35de4d89,e15d8526ebf595c2bf931f8be500311624561f9a0fc6ba4b344ad6b71ed8adb9,a81781be7cc7eeb6750f1ff7f071ba31100f261f0ad7dc10f0a49e92800a2145,False
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r02.json,f3fe0d8aa3c15fb1b3c101f318fa18402ef93d3159bdc1d919a83a42287a2454,fee638a63bbf58c0203fbb1387d8118bb7e1cd3ee151ad9f1d258f28e3fc0270,33dc0cd4a21f90837fdff0a698f098ca557fbe9add8f624f3bbda9f24351dde4,False
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-r02.json,bb8f80adddbbeaf42f256de726740ad756d49fa84b410622f8f63f0b25eabaf0,d33cc272c5d3b4fb4f769ea227e9c08d59fdbad327a071826f6d54b38854d913,460b8b9802f600cf5c818fbad8f0c14aefa93907a6c644bf9d34620d9171bbd5,False
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-r02.json,be7ef8b4b4de019964af46f797bc7cbc25e4979bf7d5372724ba147374c5b9e4,824416231aa6319215641348e3af04f2982171f11b8bb17f408954806b785460,6ddda05cb688e28e0187e925529776c96e8df4ac4fb4300804ceecb20bb26d8c,False
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-r02.json,c34054c49189a6a5a222e308bf16b1c798814909baea825a23764daface81f06,6d70c0967db670e0252a5d12b5383be6662f7629e10051d2824283713009bcae,a8ef1e409cf589f2d85b4dd10779e5919489a313a3bd59056805dcfa119e63b6,False
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-r02.json,da62af8ab96fab7c3f98938ac77c484aa840f3c0a50aee113aa7acd81b66dc68,19291f2ef4d928f1fc50b8fe50f7cc9a5fe4b40193725978edd2ab17e2b29f90,b06bd1564be0e033b04d4e95c7e0353480889a1a78bd19cc7386275f05c2b57b,False
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-r02.json,3fe2a933f3641c4a1b0b33c630ce97137a7b84fefdc655d5ae03db246f4591d8,64a3f48868426a831ddaeaa2b06b786cb6d9b62af4604f4386885c10eabf31c4,28cae8055425f7b5f98eab3c7d9d02d2068a0734faba9af766ecca2bfc820f17,False
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-r02.json,7761a51198b99401a464fc183fbebfe67c8472bccd0ec47646965cc5fdaf4d73,51ca183aac46d6604469b128d560a3d3d2696498db2b8d04d1d81e8dfc291f0b,832dd1a935f5845c0424d0d654bd7d4c69090536b718378e8e76247cc0f1c733,False
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-r02.json,69e8c821c191bc5dd9d233d6104a50205c50b2f725b375c6b1bbc9925ea5eb12,5e827fe947e9c79cfdf0a233b5deae3004565d650f9a951ade8de85e10011018,e8c367e0f6088d8625d115cf5f0cefe71a9947ccd04697851ed23177bcc9796b,False
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r02.json,865d8fd8dae9e266e5796f523d4e602129124e9572582652006503365515981c,cdcdc7256aacd25df927dc25bd28ba7d0c957d4db824f50ad3cd478ec8f9fc51,ff45770e339acce06ba9ba8161135b14293b1332120aa7d8ba0e142445e68874,False
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r02.json,4427ff291ae37f26bdfa9b618ce7e97c8200ac1aeb6067ee1a464eb408b48cef,756f48557e5e79a2591bcf1cbc0860e79cd34513599743a2b754b98b721d4506,f97735e1805fc708089d3125a3144b3aa935012c2b1be1160364b8c6fca5f440,False
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-r02.json,6f8074f83f0f1fde5c56c77068c3c54ac202f76105c3a1c8ae3f006f41e6034d,74e28ccfe0af1b8addef16ac2e41fb77ff891b303978bf5298b71727d3bec475,de0dc194f05b5523fb423e3a311553590c5a1e2d4c5f993fa2fe86eb7eda0e85,False
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-r02.json,0c034f8f45a4b67f404bc5f532ca12ddb7375266f6090c0841027b8dc5cc8a07,7ec5a8c6d5c3b45028a9441117b0d68fe6c95fc4979d7852c35231bae39e5313,184bb579a6bf0cd95d24b257553108f837a7be810e84f2445c7eb554fb67d480,False
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-r02.json,646ad076a88cc5c4d56d9d6bf5f9ed13fedd6fd484b3d8985dbdd58b3dcd3d42,e4d6eae0249d2906cf452c1ef97283828dd12f7c23623b341b8809e36d736f09,6f3157d2c53c320009b5ae798556e4896e0bd1e42958f29b388e2d0a236a048f,False
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-r02.json,eaf19efcc868e25729b9516b93fe696d0977cf7557741bb5e8ae9c0fe4c97f1d,199db581603f1d0be10c404b39fde1d60c05ccf14501c9462cbb056eb69d354e,45d4e062efe89bf9f6df584432d390e6c6825edd2b6386d3b588cc5ab3ad8306,False
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-r02.json,cbce501c5189ccd8f47cf9f2c21b559802084631f89613650bbf384bebe16b26,1dd0f699cc02eb4ea29f9514b7442898644a1a57803093b2d046a1d34a627a00,93e22c9be3904cbcf56e49c69dba2c03ebbfef87d71a5d69fe1205579969a354,False
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-r02.json,a36c88131fd85331317fb6b0278148c9950a56b58fa8223cc0f9156dbb01feed,1bc548c6f3f3a420438899fcc3e65c8fad64ced51672edc14b18075198cc276d,28ff34103b5770a678f47156893d6084847207c20f3682678dae2a7f9398c76f,False
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-r02.json,8b8710cfc9ab220ace33badac4a770b16074a10c07503ad19833380f2a20ec2a,b5d8966169c469a119f557dabcc6a137f63ca132787ee1e540650bd4011ceb93,cf77df9d4996f73b41d45e42ae90e8b2f347a9030461dea8f38433857ceba076,False
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-r02.json,cb503efee9de61a66c8cd06b906f241c1154eeb2d579d237a966b2c5566d97ab,22470be58450f97095ca9d4c7cebecb1e094577f4f9b8dbdd5d885aaa6b0c121,c9e203f7f5b95d6b8ecd6f703cde1072cce806d7ff4035ab3d3179f4b7974a24,False
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-r02.json,b72e07b5d0cb004d7f73c120ea6d369658f8e06ed77c036f7cf803a2e880a583,e92275031181fabde2e13c44552294011810802b711d1ac1afbb244163e66cf6,5d46adde8446c87d396a692b6878f971778df4b73fb1510ffa3d05a29c5b5999,False
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-r02.json,30032236203c5218a5115f486ec37da40d620cfe2137dd14c9fce961bc88fb92,4d703d6de7f87a0bd722e1ac3b36987366f202fe251b4cf248fcfc39258b865e,3320e9728430292dfc0b7b547df57b414744bd9b4764a58be1f6d992845252b0,False
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-r02.json,f6a86557671374204e87d04fce968bb4d555ae102182d28e8ea04bcf00e3732d,e962e20dac9dcc11708da712b4257ffdb59791c88bd6063145a6f6c7b4532c74,df5e71aeb4adb32878f418da3fb102dfaf5c24db1fc21176b24f70e09b03be4c,False
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-r02.json,fc5b9dfa363c72690583d0079b8894ebff06540b279f31efc57a103443ad0201,ce90814504b6023368dbfe1321c9bb2b04b6ccd6521eeb89e1e519aad57ecf8d,6c13d1ba26945bb3d848f6ee5634ee0288d697ae23fc2a12d37e0c2ff93e0019,False
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-r02.json,bb76025f6f8736b95cac6aa02f077df84bf78d38e07db634ac668935b77a3d64,1acd0153e9484052252ca26e866a370ea580525947a90dffc493ff3dee1ff741,9de98305d6a9bae87c7aa883d738a794787adc326824d1d9fd33c85480ae243d,False
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-r02,build/restart-optimization-20261017/baseline-repaired/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-r02.json,6409a69fe931f8051390528b265d0554632b7cdb02995c563e1a7df996797717,44214c57d42c5e36b7de41550331fbd5e34afe776f974e813db783fa82b02b21,c6ec68e806417acd1fd016dab67af8b4b91436dc010356adc63725f735d9fc95,False


Profiling appendix: seventeen CUDA diagnostics
----------------------------------------------

For dense-1024 Jacobi, cuBLAS fresh API records 80/90 cudaMalloc/free calls
across the ten-call device capture; Session records 20/20 and prepared 0/0.
The device workspace-only probe records 60/70 and no kernel events. All their
gated solves perform 4 updates, 6 matvecs and 0 direction restarts. These
observations support inspecting resource construction and destruction, but
do not assign a fraction of fresh API latency to that phase. Zero traced
CUDA allocation calls also do not establish that a framework creates no
fresh output or Jacobi tensors.

These instrumented captures are separate from the uninstrumented baseline
and are not framework scores. All ten-repetition captures retain passing
before/after independent true-residual gates with identical known iterations,
matvecs and direction restarts. Resource probes gate a solve outside the
resource-only probe; their gated work does not describe operations performed
inside Workspace/prepare construction and destruction.

All CUDA API counts below are totals over each complete ten-call cudaProfiler
capture. The driver synchronizes before every NVTX range as well as within
the measured phase. Full-capture stream-sync counts therefore include those
inter-call fences and must not be interpreted as solver-only counts. Launch
API counts are distinct from the recorded GPU kernel instances.

NVTX total sums only the ten exact outer iteration ranges, which do not
overlap each other. NVTX phase time, Python completed-wall medians, CUDA API
durations and GPU kernel durations have different boundaries and can overlap.
They must not be added into latency fractions or inferred by subtracting
different scope medians. API calls can wait for prior GPU work. Captures have
three warmups, seed 20261017, one host thread, affinity [0] and ten repetitions.
Clocks, thermal state, Windows display activity and physical host isolation
are uncontrolled. Single-process instrumentation does not establish a speedup.

Workspace probes include construction and destruction. Prepare probes include
the operator snapshot, optional Jacobi setup and destruction. Session workspace
construction is outside its solve capture; output and optional Jacobi remain per-call.
No historical pinned-host-scalar allocation is claimed as a current cost.
Empty workspace kernel exports record zero instances only when the unchanged
Nsight stats log explicitly confirms no CUDA kernel data. An absent export
remains unreported; its absence is not converted into a fabricated zero.

.. csv-table:: All seventeen full-capture diagnostics
   :class: benchmark-results
   :header-rows: 1

   Capture,Instrumented wall median ms,10 outer NVTX total ms,Malloc/free,Launch APIs,GPU kernels,Kernel CSV,Memcpy/async,Stream/device sync,Gated I/M/R,Before TrueR,Before threshold
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,3.710663,41.866971,80/90,670,670,recorded,0/120,140/30,5/7/0,2.43445e-07,3.05762e-06
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,2.916256,34.481943,80/90,560,560,recorded,0/110,130/30,4/6/0,1.05477e-06,5.02623e-06
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,3.711502,40.730392,80/90,560,560,recorded,0/110,130/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,3.371965,35.248615,80/90,450,450,recorded,0/100,120/30,3/5/0,1.23878e-05,1.48107e-05
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,3.626654,41.482650,70/80,400,400,recorded,0/180,200/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,1.636027,17.867452,50/60,0,0,empty; no kernel events,0/0,20/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,1.868264,19.568045,60/70,0,0,empty; no kernel events,0/0,20/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,0.979402,12.676295,20/20,10,10,recorded,0/10,40/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-api-device-solve-jacobi-standard,1.261705,16.898935,50/50,500,500,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,4.074745,48.270787,0/0,720,720,recorded,0/210,190/20,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-session-device-solve-jacobi-standard,1.208967,15.409862,20/20,500,500,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,1.124743,14.017973,20/20,560,560,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,3.201754,36.395886,0/0,720,720,recorded,0/210,190/20,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,1.432609,17.084414,0/0,490,490,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,1.390553,18.336522,0/0,550,550,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,17.861432,175.607085,0/0,11990,11990,recorded,0/1320,1300/0,118/122/2,1.21108e-06,1.44144e-06
   cuda-native-dense-1024-prepared-device-solve-none-explicit,21.695802,215.801420,0/0,9030,9030,recorded,0/1410,1440/0,69/139/0,8.92006e-06,1.04058e-05

All CUDA API calls remain in the table below, including profiler and completion
fences outside the NVTX ranges. Raw original CSV rows and API durations remain
in the analysis JSON; durations are not combined into additive attribution.

.. csv-table:: Complete observed API count inventory
   :class: benchmark-results
   :header-rows: 1

   Capture,API,Calls per ten-call capture
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaFree,90
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaLaunchKernel,670
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMalloc,80
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMemcpyAsync,120
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaStreamSynchronize,140
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaFree,90
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMalloc,80
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaFree,90
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMalloc,80
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaFree,90
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaLaunchKernel,450
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMalloc,80
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaLaunchKernel,400
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMemcpyAsync,180
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaStreamSynchronize,200
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaFree,60
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaMalloc,50
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaStreamSynchronize,20
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaFree,70
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaMalloc,60
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaStreamSynchronize,20
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaFree,20
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaLaunchKernel,10
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaMalloc,20
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaMemcpy2DAsync,10
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaMemcpyAsync,10
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cudaStreamSynchronize,40
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaFree,50
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaLaunchKernel,500
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaMalloc,50
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,cuProfilerStart,1
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,cudaDeviceSynchronize,20
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,cudaLaunchKernel,720
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,cudaMemcpyAsync,210
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,cudaStreamSynchronize,190
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaFree,20
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaLaunchKernel,500
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaMalloc,20
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaFree,20
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaMalloc,20
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaStreamSynchronize,130
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,cuProfilerStart,1
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,cudaDeviceSynchronize,20
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,cudaLaunchKernel,720
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,cudaMemcpyAsync,210
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,cudaStreamSynchronize,190
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaLaunchKernel,490
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaLaunchKernel,550
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaLaunchKernel,11990
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaMemcpyAsync,1320
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaStreamSynchronize,1300
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuProfilerStart,1
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaLaunchKernel,9030
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaMemcpyAsync,1410
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaStreamGetDevice_v12080,30
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaStreamSynchronize,1440

Compiled source and retained evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actual compiled source: ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``; tree: ``06f324529806d3edf005e4ecbb6a3c27f048f946``.

Profile driver and campaign runner are different files and are independently
hashed below. Raw JSON, CSVs, logs, Nsight reports and frozen source/binaries
remain local ignored evidence; these paths are not public artifact downloads.

.. csv-table:: Source identities
   :class: benchmark-results
   :header-rows: 1

   Role,Path,SHA256
   profile_driver,build/restart-optimization-20261017/baseline-stages/cuda-release/tools/profile_solve.py,20b805f3bd95a9f0c1a5d23105c17888e4bd285b754575c1540661df87b7e0c4
   runner,build/restart-optimization-20261017/profile_current.py,eda0e6d1779698c6cc85d419c269cfb58150ef2c4cd8aad317ce066ea7f11d5e
   stage_identity,build/restart-optimization-20261017/baseline-stages/cuda-release/identity.json,7a956531880eaee793f22d75c4cc748f53c20ee8d5569fc0f2fe4cf67411d416
   manifest,build/restart-optimization-20261017/profiles-baseline/manifest.json,26d67c9d638a096bec24f03eab4274052706d7ca98aedc3cbd4c69a37bf334ac
   plan,build/restart-optimization-20261017/profiles-baseline/plan.json,1dcd1c68e3f57f69031eec7893568f859476a5c4c06409a83d3cf66136133e8e
   renderer,build/restart-optimization-20261017/render_profiles.py,bdfa1742428454b7cf75869710f528a0f34235038a18936ca4ce7d6bb138182c
   render_helpers,build/restart-optimization-20261017/render_baseline.py,af180cd5ecb8e916450024b0d5fabd0f6b7088f9f41e0e720b95ea5552154ab0

.. csv-table:: Every retained diagnostic artifact
   :class: benchmark-results
   :header-rows: 1

   Capture,Filename,SHA256
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard-stats.log,cd2ffcd93c9b812be9577b99e697850ef22c07ebb3995596d80e0decffb277a5
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.json,cb3aec02fc1bd7070c389306d6876f52dfb6fe89fbfcde0b2fdd4f368af62aac
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.log,273ecf6e8afca954c543c38fd151c6bb2264184ea1274ce110a55b7f26e4b985
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.nsys-rep,e9aadf3b59157de2a25ec8087268bd08a2c54c1eefd95c3037798dcec742d63f
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.sqlite,b1f77618e02ff6df8351f1472c51e7ef7f7a081de59afe9708c203d711228c3a
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_cuda_api_sum.csv,faf38535d5b3d09cb839adafb7e59af9785adc8c4e3ee4dbf83686283142c281
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,d8251fdabd8a62610a43c30aeb3ff01b3cc3400f655a166df69f60c64226b585
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,07ef20b449472563abc5208fd507c61d738c7e0838ddc0ab22ee80869e432e4e
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard-stats.log,65c51fbb846d17b8b033414d28fef0a0720800d35ac0ca8195131fe197b65a26
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.json,9dd566c3614d8be80975b5226b460ca12bc22ba973bcbc8cb324d238a85638c6
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.log,af2a6a8bb73335f93aac4036cb89be40bb1b6073fd9a32c50f19813eb4b00133
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.nsys-rep,d00cac17707cc4a6d4ddc13c1be171d9afb2b139d5e9059a4d4e20cde5497184
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.sqlite,1460b49c5cab621ccc869e8ec1edcdfc660ab1f1c03e6974818ce9e5045d0989
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_cuda_api_sum.csv,1e9c02e55bbc3705be87f01c23a6f4e6fce878846a48c36f6cf38b13f5326652
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,b820fc1b8b18f24b4614548d2cbf3de17a70bacccbeef6d721a246571860cad2
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,74f5169f8791be483e652deedc010317bdd8bd607f4b483c9b006df0da5f8de2
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard-stats.log,f5ebc1bed6146eebeef0a91743aaa5c3202d61c868dee8fb2e5c8ea81d263608
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.json,669799b8da6919871ac61b47cb262236d7d1669b9653406abb746f408c603ad9
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.log,5e6ce3814438141bd5bc4c7e51e9f57d6d2445d5731999e6b2d9d919d22cff74
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.nsys-rep,29d1194e1e574bf6aced89d59dce37d84356596a13ff15902d62a6f8a123cbf6
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.sqlite,b6c41e20b2b363eed3759a775b3edcd58e32c2829ba94fb28e2df0d6a9dc55cd
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,ef815a39d41c49cd682b8b57be3d3f759b80f2b19df6e450f54cdfb033832978
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,1eee9ec0f1e3ef479a5e7ce13f1acb6771839c40ba4caee2edaa91c2802ed6d4
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,10f3e35d0c9abba8adb8b1d6f767463e5c6ed8b36985f3892bf072a7dc71ebef
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard-stats.log,b3b6313d0c9f0108aa13536572c008bd89eb69316ad3d0b80abd00391aacf632
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.json,0c6430702709503821afaabf2e9334c769db9d9102efc0d166fdd3a7a80961c7
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.log,3d5defdee1246c4cf030783c3c8c78f63026035ec0f265c08421ffcd55e4e195
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.nsys-rep,545775261b6e68a8211bca55d408e7fb61fe19f97d132fb461a0f22e070f63ee
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.sqlite,faa90eb5254a84661a0e4c2eabb460eaf9bde0ca2d68f301dfb6e1da86405c62
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_cuda_api_sum.csv,015be3268a338f7f813e5820a06092bfe1f8cff9faa3310fe3aa5bc514fb91f1
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,913f41d65b37c77a71d51a4e34b967d535b0d84ac7c16dc83517ad667b6aafe2
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,6940bf061170bb0ec6b7434b4d7ee28de37980202c476f323a380386620302e8
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard-stats.log,9fb42ed49468fdd3d3b7523bd2a65fe3843bd2753bde6fe57c51b03ffba25621
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.json,1dabd8d66bdff72fb60bbc35a60953032052a6364f2faa6836f9625e5a8bcf7a
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.log,61658e570ba006da258fa060f90fbce378fd06494ce96006327954814ddd16f1
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.nsys-rep,d71a78c8dbcc8eefacfa14c7784db68b872c7bdc248c83d28572ba24997ac0f1
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.sqlite,96da2f7b7f2bacbb480e0363344e560e701cec0524de16f437521ce8a7a4fdd3
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_cuda_api_sum.csv,263a7fbb3206aa97acda067174d4909a1792ea1195561b333f5aa327dd461d66
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_cuda_gpu_kern_sum.csv,81d89882b728e7e2ac4352b689b7586af68c300682bd0e2908b49a125008d4d9
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_nvtx_pushpop_sum.csv,6181ba8c9f683252f6450af4a13bd0032f1acb6bd0b48dc71b3436165385ae2a
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard-stats.log,cdea0d87341ea5a12352c0d3918ad39b4e54bcaa0baba3748af46af312295642
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.json,3dc9ecc50283b010b4e357d57bbd414530d0320c7b5d037f47e7509b2a6db75d
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.log,265a1b64cef2b24237000a0290cce41084e1d29a8dda32cdaf4433d85c334d45
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.nsys-rep,f529e1995464bb02163a533708ea582ae9cd74f342c4e96fa1a4c031a1908300
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.sqlite,e86e545f91b8d6f982335823e250ea78a36eb88174f6ce91d8ee263ad98e361e
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_cuda_api_sum.csv,40a0f93f1449752dc45dc55f6b91c1a41eb3558c5213df67b80033cd071b4a11
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_cuda_gpu_kern_sum.csv,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_nvtx_pushpop_sum.csv,c777143aa2d07da6db36cec8a8ca734fe60eb8082efc7808ea52607b876c7cc9
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard-stats.log,bdda8bacc62f587f32ff2f557083dcd7451db60903d8fde94108edf156c7683a
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.json,14937b25b31d5d43da6b90fe15ef2d3201011382ed79dd3e7fd48a99ee1e3fc6
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.log,92cbc6981ba420efd07800cb33a334706ba4c485bc5ba76cc3997e1d9f64d147
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.nsys-rep,9a25f71cdecb3a1435777ad3ec53d5ea4340b65c70a447e4d67a7389d676671c
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.sqlite,8026e5b36f103d98959bbfb6780841ed5ef22906bcaf4d91c5dd5736611f6d70
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_cuda_api_sum.csv,967ab2a426752b4204f6a757429dc60938353a6dd4533a379538ffe9c61359d5
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_cuda_gpu_kern_sum.csv,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_nvtx_pushpop_sum.csv,2c5da3dd703b13db6fab229f034b4be00cbfe12b5e49835ea5c0fa15fc32b09d
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard-stats.log,336232dbca9cc3090e608912d36c40b03763850661dbfb9447a4fedbee117a25
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.json,a05c063370de2a98dd3b8fc25a6bb85b8d40b728efe6ce4fddebee0f92b1d932
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.log,00a6f78128745b75b2a84079e79a6142b659aae2e629cd384425a29cda1fb53b
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.nsys-rep,c27b23e9dea50c5075d5fcb6dcbf91084acff15031aac99b7b7dbdb086485590
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.sqlite,1659ed25f9d39f0dc2b02657a1d765bbe6c1343bc6f0c0397508e14fa5cd41c4
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_cuda_api_sum.csv,be116d7afa73183246c2229a12abde526ee2674ddb91d5f0c22c4685a655f914
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_cuda_gpu_kern_sum.csv,cb84ff5e05ff400d35a206bec109f127d6507695b86347c8b776bb5ab7aeda81
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_nvtx_pushpop_sum.csv,a8ed9d1ac4dace0d945c311f9c10a579769b3ef70ee51e7d531445b2eda14cca
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard-stats.log,3ead9c9f01a28f4d689ad9306a7b41cc775b174cc5205be1cd4940b4d843721f
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.json,f3ab382a71ee6a35c74e2743764cd36ce1d67cc79539d27011f62df391ca10db
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.log,d3e1bb62ba9f9235c51de0f1ab4dedfca6ce48a007ffa14a18097f9b11ede9a8
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.nsys-rep,df76f135a9952f97b232c82a5adc4bc96c3952061fef2de76e7a9b0e7fbb2d55
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.sqlite,8c5eca2aedf61a913dba393ef6141b4beb33a44f8a0dad080553d57a9e50cc1c
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,b139b7806cb872a3d145f11ce166dcef36703b9156e383434fdfa23c9a7efc36
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,89610801b887b5e7b652d3a142ffa726855856f4fa29f9a766ee1e8a54d08efa
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,77aa045aaa636f56bdcfe1ae2c463bbcd56ee7d23ba9ad2ad9813e732b2f33f8
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard-stats.log,d281a865123bd4964a4176381e01a42c659cc00d934ddd4cc16570372a1b2a81
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.json,a100d24fc754740269dc05a27785da24978f7f056c0b0e5ee952f6d7ec816e38
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.log,10c3bfa736f2550938bb7c60667da6b00c9b8e1e4653a511366f0816be56d66f
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.nsys-rep,c15e9047ef7b1a506e016cb9989d695d548f66ac304167641a01af4c6b10fdd1
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.sqlite,be538ba5b5d948a7c18998565b055db46a2813bfd753fb2fb06f3fb12470de43
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,7d7a392569501d744c8ca3c69c6d870d7f1c25e3ae0c32c294dbb5baedf4e337
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,3a8562ec354be02e04ed8ab012e11ebe5359cb3c2bc43909ff6083c2e39fbf62
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,e2b493af6cb7e8f26af09a66f61245b5ff056ce47f9b3f1c2da021d9d92e6c14
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard-stats.log,de1a3676de048878873cf4b1ec63c0c8c289208ba0d28e2af4f88c82fdab947e
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.json,8e348a57d416065c79e93f6cfee34236965181904d53579643b42511cd3e1e81
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.log,55af0e36584b8cda4a41ce8d6287c91d1b0fa137e8c91703ba94fc85019db6d5
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.nsys-rep,a94b0cd599f65fe44285e283ca6a278812f6d30611fae4cb15e9b16ac8f9c891
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.sqlite,3bf0ea2d690d28630d79a993c46e876e9e3734a6a6f0d06236222aa82f6e3985
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,e13210778ed77adbe20eccf04583043e8003c58cf3c8d40a2e93293c0026028a
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,f89dbf52f7ced12a0994b8251d13afb40202684721a0a66b2516aea17c09efe0
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,3271df9f9188cbbb97d68482d183f6847467a445740e6cb8b660e4339a481526
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard-stats.log,dbd418d496db5bf865c2adfc1070da9ff7716adce894814133757aafb66c5574
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.json,bb47d9ceafc1ecb2d287e3b8a7974a83f1187ebb92092bf0b3d474aa3834c86d
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.log,8898011a23583c50507e4daa3c9885f31d254ddde66f79f5a2f2e98089f0aeb3
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.nsys-rep,e58fffc4f7320ebc110b4051704487301ec852bdce35acd73003d8a407275696
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.sqlite,ee7ffd229ca3590befd160c679f06a1b7c5e2333b7ecb3f22433fb21c51c2a7d
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,dc05a20661aa0f5fa457dd3d0a4b76e53daa58e950d02cb760a78ddcae769d79
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,a254f5b8bc74e35a4195a1e5a54a1faf1bb47b0d393b3c31c32cda88d9680e8a
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,745059b005383dba9018309c5029477a3e064a636ef78521887fb24433d5772d
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard-stats.log,6ac47568d5ebd5eddf6211a2202c4e2cebc55fcad1d18d42c506e2cc2667b6a7
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.json,7f5dcf1e11942f98e73caa1295137683eeaac3271f04e162e1ed31cf28cf4785
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.log,f65cccdeff677d70f8c90ef89e2118c0d401f13670016cbf5db8a0bb441b6e85
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.nsys-rep,f0ce2636b259876695bb0c8a9a914c78ad0407f620b016fb609aeb3c359fc2e4
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.sqlite,c79154e58a3fa6344671d2702076b7a09bc81574f8a2a8fec57b07ea25bff09a
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,4dd964abc449f88d369a45302add66ac9eeb5bb6627c8282fe58a1e9ecb7ad03
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,e6d0de931ecfb9e37d71d40d60181f96a326dfb8e1d60b0050c51433656d4dda
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,34d533f5b03a5777acde853ba0022041d98415f909e0a7b9c74fc7ad9f5b7aa0
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard-stats.log,fd6c122ea7f8957105401e982ba78136b1fa8738307612b54ee2099b9c0f6ea6
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.json,50ae41c8f9d1172ce788097bd2bb1877e814113c43a0556366009996835df728
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.log,a6db9df1a45519f37af18dc8f6a2be80f44dbd53034e562e38d2908a47f87520
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,25ef0e234cd202d99d7a15e19f867a982319534382fb439b4839898effe6c52e
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.sqlite,47a2d72a1bf34910dfe1ab4c11fa2a6a7e0cd214dee7bef7e298fdae1704ff1c
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,c0fdc0947fc8569c8285f0b944a3abda680cbf66fc0b4826c966e7eae2e08d4f
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,47ed248c5100d4352aa7ac8e1b3dfb1744527dc58d0fe0e77a52512b52ec8562
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,727aada084f0264b1f79960a1e468fc12417e3f45a44de6e3165891e68b49b3c
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard-stats.log,14ef39d87f6a58b34c319037fd3708a4cb5cce8f8f4e5b6193310565964f5322
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.json,d8e3ababf04ce4ba64c3abe1436e1eebfe311c5449e96c94dd762feadf246772
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.log,5beca91753f1377e259a6f7bba5cf43dc9896f8f8ed37abe8ef4a46e38ed05a4
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,c7b1f37b97e4309f3c70d386fa46269353f233ba0650296787ea4d3416db15b1
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.sqlite,586fe85730a38e4c854ff47ed363fd9d23f22e0c9ee71d8db5235f47e641d722
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,eeffd849db180912ac9519acda4ed0ed0932256b42f8d2e4e839873f2ccce87c
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,be9eabea3ecf8dce2434baea7f5cdf1ca58edad7fd559932292f78cafb9aa0dd
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,021a745ac1a5f2c59f27f037b304e2646ad977f6c1345f62d3529eb5c3507b5f
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard-stats.log,eeb6179539155e9f674e11a8abda8108721c6bee5c06198681175729f411a7e2
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.json,35fac0d4171450af287e112583bb595c9202df492a9cf82c53f21d2041fffbbf
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.log,007e516e3e755ece67916f7ba2a6559ec497e6e764245dfdf1d658225120b373
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.nsys-rep,adaf460e8ed825865be906b5e67463949218a2fecd9f111a0c9ebfd1c4012a93
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.sqlite,42b1fd1fc75333528fc14653bafe5cc358d3a67c77187e86d680dad6714d4277
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,0e22d69d4ecc8a6a64d0121936cc7e96b8ee48b4467831022ded4d0be2a52d18
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,cfa3cddab963ec934ce920858f9c12468384c06e7ee15d02cb9bbcf89d1b8719
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,3a52c7708e05274042ded24efa979e349aa007a15c2947d88616c8c0520710d5
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit-stats.log,3233b12e6e371a7430c4ae920ec8c797c7d96527c83c849befed034f73ddbecd
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.json,fc6bce4e5978e197e6af854ea8c8eefd934c5ead0c22e04eed90242771118816
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.log,17cbbd401377faaa784c33ec83936846d524fd36a5105f006421670f56acb1de
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.nsys-rep,4536319c464e0a268b187c9486169c04abf61aa4feafac438a2f537aeaaf787d
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.sqlite,cc996bdec8132b6309ae1c8670c3ee303181c97753d8c050bbb392ca1f0a44aa
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_api_sum.csv,d37e93f5666c535bd012388736cb99c7f93fea040ceb6f1a025df9b1aa894247
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_gpu_kern_sum.csv,9f71501e94181cb8684cb7df93d7936a04f780b43694f1307be6f6ff859ea01d
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_nvtx_pushpop_sum.csv,31b1d57111e1e8655bf5480baf08357746a2c52355745766c5a1b582d63a4016

To reproduce a capture, use the public source-install and project-local CUDA
instructions in :doc:`../install` and :doc:`../cuda`, build the recorded source,
then run the matching public ``scripts/profile_solve.py`` inside Nsight Systems
with capture-range=cudaProfilerApi and cuda,cublas,nvtx tracing. The ignored
orchestrator is not needed. Nsight installation, WSL permissions and GPU
availability are external prerequisites. Exact frozen command argv and all
original summary CSV rows are retained in the analysis JSON.

For example, after building the recorded source with the CUDA command above:

.. code-block:: sh

   PROJECT="$PWD"
   mkdir -p build/restart-profile-reproduction
   unset CUDA_LAUNCH_BLOCKING
   CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=build/reproduce-cuda-release/python \
   taskset -c 0 nsys profile --capture-range=cudaProfilerApi \
     --capture-range-end=stop --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
     --output=build/restart-profile-reproduction/cublas-1024-api \
     "$PROJECT/.venv-cuda/bin/python" scripts/profile_solve.py \
     --suite cuda-dense-f64-v1 --implementation cuda-cublas --case dense-1024 \
     --device cuda --cuda-device 0 --scope api --probe solve \
     --preconditioner jacobi --algorithm standard --fused \
     --cuda-execution-mode device --seed 20261017 --threads 1 \
     --warmups 3 --repetitions 10 --rtol 1e-8 --atol 1e-12 \
     --maxiter 2000 --residual-refresh 50 --cuda-profiler \
     --output build/restart-profile-reproduction/cublas-1024-api.json
   nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_pushpop_sum \
     --format csv --output build/restart-profile-reproduction/cublas-1024-api \
     build/restart-profile-reproduction/cublas-1024-api.nsys-rep

This example uses a completed public build and creates new capture files. It
omits the original local frozen-stage identity, which is not shipped in Git.
Change the case, scope, implementation, probe and algorithm flags to match each
listed capture; keep its numerical and sampling settings fixed.
