Complete current-main CPU and CUDA baseline, 2026-10-17
=======================================================

This baseline retains all 134 configurations, 268 fresh-process artifacts and
3,096 eligible candidate/case results. The complete tables contain 1,548 rows,
each with both fixed process observations. Ratios describe this host and run
order; they are point estimates without confidence or regression claims.

Actual compiled commit: ``445275a3994e342b06de4093889f3f814c8706c6``; compiled tree: ``5d7ba07832dc7576810ddd5d07d66e4b87ef8109``.

Method and scope
----------------

The fixed order is all 134 configurations forward, then exactly in reverse.
The 90 standard configurations precede 44 explicit-none configurations.
Both algorithms retain all supported fusion, scope and execution combinations;
Jacobi with explicit is unsupported. The five-fixture strategy suite is standard
only and retains all six fixed aliases. Explicit restarts are known zero.
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
and completed final work are included. API allocates output, workspace and
optional per-call Jacobi data. Session reuses iterative workspace but allocates fresh
output and optional per-call Jacobi data. Prepared reuses setup, output and workspace.
Native host/device execution is explicit; the Torch project reference keeps
its own implementation and discloses that native execution mode does not apply.

CPU OpenMP uses two threads and logical affinity [0,2]; all other presets use
one thread and affinity [0]. WSL affinity does not isolate the physical host.
Host load, Windows display use, GPU clocks and thermal state are uncontrolled.
Different frameworks, scopes, modes and solver strategies may perform different
accepted-tolerance work. No cross-scope strict comparison or universal speed
claim is made. SciPy official solvers have a different refresh policy; SuperLU
is a general sparse direct solver. LU-total includes factorization and solve,
whereas LU-reuse retains its factorization. Neither is a pure CG kernel.

This is a new fixed campaign: each planned fresh process is executed once,
with an exclusive execution claim. Earlier rejected/adopted trials are historical
and are not pooled into these estimates. Every raw artifact and control is kept.

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
* cuda-release ``_cuda``: ``2b95332f5e7cfa622e5de9eeb109ed8d6b2798f7e9eb9f8ee7d686a9c1f71fbe``.

.. csv-table:: Frozen stage and host compiler
   :class: benchmark-results
   :header-rows: 1

   Preset,Identity SHA256,Compiler,Flags
   cpu-release,be6e40ee8f066fbefba26eca707c5301430f10d4f46b5561274557f8ba6bf6a5,GNU 16.0.1, -O3 -DNDEBUG
   cpu-blas,33001f2b16b7ede19651304355913c1431aafc661d22354ef311b55496343929,GNU 16.0.1, -O3 -DNDEBUG
   cpu-openmp,3344786f97ab3b7e265a92546580064cef6b1b7f5a95e645bfb59507dd9e1d0b,GNU 16.0.1, -O3 -DNDEBUG
   cuda-release,f4a79e809aa636e857d50b5dce39f9eb11f94910901ea18213cd395a5c066ae2,GNU 13.3.0, -O3 -DNDEBUG

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
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard,native,numpy-out,3/1/0,0.1955–1.2340,same
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard,native,torch-eager-out,3/1/0,0.1148–1.1838,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard,native,numpy-out,3/1/0,0.2008–1.2683,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard,native,torch-eager-out,3/1/0,0.1271–1.2110,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native,numpy-out,3/1/0,0.1408–1.5273,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native,torch-eager-out,3/1/0,0.0816–1.2209,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native,numpy-out,3/1/0,0.1610–1.3502,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native,torch-eager-out,3/1/0,0.0938–1.1405,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native,numpy-out,3/1/0,0.3528–1.2015,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native,torch-eager-out,3/1/0,0.1876–1.2069,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native,numpy-out,3/1/0,0.4140–1.3167,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native,torch-eager-out,3/1/0,0.2096–1.1283,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native,numpy-out,2/2/0,0.2458–1.4281,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native,torch-eager-out,3/1/0,0.1385–1.1534,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native,numpy-out,2/2/0,0.2169–1.7098,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native,torch-eager-out,3/1/0,0.1242–1.3402,same
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard,native,scipy-out,4/0/0,0.1265–0.5162,same
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard,native,torch-eager-out,4/0/0,0.0920–0.6936,same
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard,native,scipy-out,4/0/0,0.1404–0.6666,same
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard,native,torch-eager-out,4/0/0,0.1045–0.9016,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard,native,scipy-out,4/0/0,0.0710–0.4952,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard,native,torch-eager-out,4/0/0,0.0532–0.6961,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard,native,scipy-out,4/0/0,0.0704–0.6356,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard,native,torch-eager-out,4/0/0,0.0459–0.8673,same
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard,native,scipy-out,4/0/0,0.1253–0.5164,same
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard,native,torch-eager-out,4/0/0,0.0864–0.6680,same
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard,native,scipy-out,4/0/0,0.1324–0.6974,same
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard,native,torch-eager-out,4/0/0,0.0856–0.9045,same
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard,native,scipy-out,4/0/0,0.0811–0.5572,same
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard,native,torch-eager-out,4/0/0,0.0588–0.7127,same
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard,native,scipy-out,4/0/0,0.0794–0.6818,same
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard,native,torch-eager-out,4/0/0,0.0553–0.8899,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard,native,numpy-out,3/1/0,0.3074–1.1392,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard,native,torch-eager-out,3/1/0,0.2023–1.1148,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard,native,numpy-out,2/2/0,0.4281–1.2231,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard,native,torch-eager-out,3/1/0,0.2815–1.0632,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native,numpy-out,3/1/0,0.2796–1.3675,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native,torch-eager-out,3/1/0,0.1829–1.0592,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native,numpy-out,2/2/0,0.3854–1.3184,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native,torch-eager-out,3/1/0,0.2517–1.0450,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native,numpy-out,2/2/0,0.4685–2.7708,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native,torch-eager-out,4/0/0,0.2741–0.9136,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native,numpy-out,3/1/0,0.3101–1.1157,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native,torch-eager-out,4/0/0,0.3273–0.9195,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native,numpy-out,3/1/0,0.4026–1.2480,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native,torch-eager-out,3/1/0,0.2385–1.0037,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native,numpy-out,3/1/0,0.5024–1.4679,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native,torch-eager-out,3/1/0,0.3170–1.2685,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard,native,scipy-out,4/0/0,0.1945–0.5715,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard,native,torch-eager-out,4/0/0,0.1348–0.9035,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard,native,scipy-out,4/0/0,0.2851–0.5469,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard,native,torch-eager-out,4/0/0,0.2007–0.8480,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard,native,scipy-out,4/0/0,0.1525–0.5032,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard,native,torch-eager-out,4/0/0,0.1068–0.6895,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard,native,scipy-out,4/0/0,0.2405–0.5813,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard,native,torch-eager-out,4/0/0,0.1779–0.9110,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard,native,scipy-out,4/0/0,0.2011–0.5352,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard,native,torch-eager-out,4/0/0,0.1341–0.6877,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard,native,scipy-out,4/0/0,0.2875–0.6542,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard,native,torch-eager-out,4/0/0,0.1760–0.9072,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard,native,scipy-out,4/0/0,0.1615–0.4943,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard,native,torch-eager-out,4/0/0,0.1016–0.7704,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard,native,scipy-out,4/0/0,0.2513–0.6031,same
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard,native,torch-eager-out,4/0/0,0.1656–0.8266,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard,native,numpy-out,3/1/0,0.1684–1.1544,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard,native,torch-eager-out,3/1/0,0.1066–1.1077,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard,native-blas,numpy-out,4/0/0,0.1625–0.9415,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard,native-blas,torch-eager-out,4/0/0,0.1028–0.9034,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard,native,numpy-out,3/1/0,0.1960–1.1707,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard,native,torch-eager-out,3/1/0,0.1202–1.1179,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard,native-blas,numpy-out,4/0/0,0.1548–0.9489,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard,native-blas,torch-eager-out,4/0/0,0.0949–0.9061,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native,numpy-out,3/1/0,0.1359–1.4479,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native,torch-eager-out,3/1/0,0.0836–1.1798,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native-blas,numpy-out,3/1/0,0.1245–1.1142,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,native-blas,torch-eager-out,4/0/0,0.0767–0.9079,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native,numpy-out,2/2/0,0.1524–1.4339,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native,torch-eager-out,3/1/0,0.0890–1.1943,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native-blas,numpy-out,3/1/0,0.1336–1.1423,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,native-blas,torch-eager-out,4/0/0,0.0780–0.9514,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native,numpy-out,3/1/0,0.2459–1.1045,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native,torch-eager-out,3/1/0,0.1349–1.0284,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native-blas,numpy-out,4/0/0,0.3253–0.9718,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,native-blas,torch-eager-out,4/0/0,0.1784–0.9049,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native,numpy-out,3/1/0,0.2574–1.1631,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native,torch-eager-out,3/1/0,0.1379–1.0896,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native-blas,numpy-out,4/0/0,0.3020–0.9277,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,native-blas,torch-eager-out,4/0/0,0.1617–0.8691,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native,numpy-out,2/2/0,0.2004–1.2910,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native,torch-eager-out,3/1/0,0.1150–1.0124,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native-blas,numpy-out,2/2/0,0.2050–1.1930,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,native-blas,torch-eager-out,4/0/0,0.1177–0.9356,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native,numpy-out,2/2/0,0.1922–1.4963,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native,torch-eager-out,3/1/0,0.1045–1.1736,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native-blas,numpy-out,3/1/0,0.1992–1.1298,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,native-blas,torch-eager-out,4/0/0,0.1083–0.8862,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4683–0.7783,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.5518–0.7914,same
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4863–0.7619,same
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.5769–0.7866,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3113–0.6165,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4009–0.6653,same
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3478–0.6449,same
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4451–0.6672,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4519–0.7377,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4317–0.6989,same
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5138–0.7749,same
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.5081–0.7125,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3094–0.6238,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3266–0.6108,same
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3502–0.6217,same
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3499–0.5957,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4411–0.7495,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4494–0.6971,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4986–0.7622,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4966–0.7166,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3053–0.6004,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3247–0.5601,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3215–0.6604,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3469–0.6092,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.3816–0.6195,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard,cuda-cublas,torch-cuda-eager,0/4/0,1.2060–1.3778,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4409–0.6055,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard,cuda-cublas,torch-cuda-eager,0/4/0,1.1811–1.4166,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2858–0.5430,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard,cuda-cublas,torch-cuda-eager,0/4/0,1.0512–1.2164,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3697–0.7309,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard,cuda-cublas,torch-cuda-eager,0/4/0,1.1235–1.3367,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4143–0.6293,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4281–0.5786,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.3822–0.4976,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3675–0.6281,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2326–0.4389,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.2458–0.3992,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2947–0.5642,same
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3030–0.5094,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4097–0.6122,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.3787–0.5502,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.3971–0.8613,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.4022–0.6087,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2580–0.5918,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.2703–0.5387,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2879–0.5766,same
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard,cuda-cublas,torch-cuda-eager,4/0/0,0.2936–0.5716,same
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4842–0.5017,same
   cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5351–0.5685,same
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3300–0.3510,same
   cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3877–0.4039,same
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4716–0.5092,same
   cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5503–0.5824,same
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3032–0.3602,same
   cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3891–0.4047,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4161–0.5126,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5124–0.5704,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3152–0.3435,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3737–0.3893,same
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.4960–0.5275,same
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5261–0.5781,same
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2614–0.3153,same
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3708–0.4311,same
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5023–0.5277,same
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5172–0.5733,same
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2622–0.2751,same
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3661–0.4325,same
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5182–0.5252,same
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard,cuda-native,torch-cuda-eager,4/0/0,0.5374–0.5617,same
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.2619–0.2718,same
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard,cuda-native,torch-cuda-eager,4/0/0,0.3504–0.4066,same
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-cg,scipy-cg,5/0/0,0.1085–0.5533,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-cg,scipy-pcg,3/2/0,0.1061–3.1182,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-cg,scipy-lu-total,3/2/0,0.2041–3.0577,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-cg,scipy-lu-reuse,0/5/0,2.2690–34.5955,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-pcg,scipy-cg,5/0/0,0.0265–0.1351,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-pcg,scipy-pcg,5/0/0,0.1229–0.2961,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-pcg,scipy-lu-total,5/0/0,0.0600–0.2792,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,native-pcg,scipy-lu-reuse,1/4/0,0.9556–3.1584,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-cg,scipy-cg,5/0/0,0.0869–0.5454,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-cg,scipy-pcg,3/2/0,0.0800–3.1452,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-cg,scipy-lu-total,3/2/0,0.1666–3.0513,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-cg,scipy-lu-reuse,0/5/0,1.5459–36.6741,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-pcg,scipy-cg,5/0/0,0.0176–0.1015,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-pcg,scipy-pcg,5/0/0,0.0934–0.2584,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-pcg,scipy-lu-total,5/0/0,0.0490–0.2507,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,native-pcg,scipy-lu-reuse,2/3/0,0.7967–3.0130,unknown
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit,native,numpy-out,3/1/0,0.2029–1.2416,"different,same"
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit,native,torch-eager-out,3/1/0,0.1193–1.1949,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit,native,numpy-out,3/1/0,0.2126–1.1839,"different,same"
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit,native,torch-eager-out,3/1/0,0.1228–1.1587,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native,numpy-out,2/2/0,0.1685–1.5151,"different,same"
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native,torch-eager-out,3/1/0,0.0925–1.1508,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native,numpy-out,2/2/0,0.1695–1.6517,"different,same"
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native,torch-eager-out,3/1/0,0.0997–1.2080,same
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit,native,scipy-out,4/0/0,0.1047–0.6065,same
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit,native,torch-eager-out,4/0/0,0.0765–0.7552,same
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit,native,scipy-out,4/0/0,0.1129–0.6868,same
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit,native,torch-eager-out,4/0/0,0.0805–0.8930,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit,native,scipy-out,4/0/0,0.0682–0.6027,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit,native,torch-eager-out,4/0/0,0.0493–0.7586,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit,native,scipy-out,4/0/0,0.0806–0.6970,same
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit,native,torch-eager-out,4/0/0,0.0556–0.8869,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit,native,numpy-out,2/2/0,0.3874–1.1723,"different,same"
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit,native,torch-eager-out,3/1/0,0.2413–1.0454,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit,native,numpy-out,2/2/0,0.4876–1.1148,"different,same"
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit,native,torch-eager-out,3/1/0,0.3011–1.0902,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native,numpy-out,2/2/0,0.3580–1.4428,"different,same"
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native,torch-eager-out,3/1/0,0.2246–1.0809,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native,numpy-out,2/2/0,0.4421–1.4743,"different,same"
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native,torch-eager-out,3/1/0,0.2897–1.0889,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit,native,scipy-out,4/0/0,0.2178–0.5528,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit,native,torch-eager-out,4/0/0,0.1498–0.8011,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit,native,scipy-out,4/0/0,0.2649–0.6191,same
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit,native,torch-eager-out,4/0/0,0.1842–0.9619,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit,native,scipy-out,4/0/0,0.1796–0.5374,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit,native,torch-eager-out,4/0/0,0.1245–0.7909,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit,native,scipy-out,4/0/0,0.2257–0.6834,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit,native,torch-eager-out,3/1/0,0.1509–1.0140,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit,native,numpy-out,3/1/0,0.1920–1.1958,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit,native,torch-eager-out,3/1/0,0.1097–1.1243,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit,native-blas,numpy-out,4/0/0,0.1638–0.9366,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit,native-blas,torch-eager-out,4/0/0,0.0936–0.8805,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit,native,numpy-out,3/1/0,0.2096–1.2247,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit,native,torch-eager-out,3/1/0,0.1228–1.1608,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit,native-blas,numpy-out,4/0/0,0.1591–0.9569,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit,native-blas,torch-eager-out,4/0/0,0.0932–0.9070,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native,numpy-out,2/2/0,0.1654–1.5012,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native,torch-eager-out,3/1/0,0.1050–1.1016,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native-blas,numpy-out,3/1/0,0.1088–1.3064,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,native-blas,torch-eager-out,4/0/0,0.0691–0.9587,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native,numpy-out,2/2/0,0.1741–1.6872,"different,same"
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native,torch-eager-out,3/1/0,0.0982–1.2340,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native-blas,numpy-out,3/1/0,0.1257–1.2519,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,native-blas,torch-eager-out,4/0/0,0.0709–0.9156,"different,same"
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5025–0.7933,"different,same"
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.5755–0.7591,same
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5200–0.8066,"different,same"
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.5642–0.7895,same
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3230–0.6608,"different,same"
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.3869–0.6501,same
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3542–0.6605,"different,same"
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.4098–0.6530,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5253–0.7958,"different,same"
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.5216–0.7378,same
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5439–0.7988,"different,same"
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.5294–0.7440,same
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3140–0.6498,"different,same"
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.3333–0.5968,same
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3476–0.6592,"different,same"
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.3540–0.6136,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5073–0.7772,"different,same"
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.5134–0.7274,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5269–0.7888,"different,same"
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.5127–0.7356,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3186–0.6592,"different,same"
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.3486–0.6099,same
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3581–0.6571,"different,same"
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit,cuda-cublas,torch-cuda-eager,4/0/0,0.3601–0.6118,same
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5163–0.5321,same
   cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5561–0.5804,same
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3344–0.3574,same
   cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3696–0.3772,same
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5417–0.5611,same
   cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5279–0.5754,same
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3367–0.3614,same
   cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3459–0.3786,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5275–0.5559,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit,cuda-native,torch-cuda-eager,4/0/0,0.5458–0.5584,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3378–0.3537,same
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit,cuda-native,torch-cuda-eager,4/0/0,0.3528–0.3760,same

Complete remaining higher-median inventory
------------------------------------------

.. csv-table:: Every native/reference point-estimate loss
   :class: benchmark-results
   :header-rows: 1

   Configuration,Case,Native/reference,Native ms,Reference ms,Ratio,Native I/M/R,Reference I/M/R,Parity
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard,dense-2048,native/numpy-out,118.588357,96.098132,1.23403,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard,dense-2048,native/torch-eager-out,118.588357,100.178468,1.18377,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-2048,native/numpy-out,142.107887,112.044927,1.26831,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-2048,native/torch-eager-out,142.107887,117.351714,1.21096,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native/numpy-out,138.475132,90.664556,1.52733,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native/torch-eager-out,138.475132,113.421785,1.22089,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native/numpy-out,134.941597,99.939536,1.35023,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native/torch-eager-out,134.941597,118.314620,1.14053,70/73/1,70/73/1,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,dense-2048,native/numpy-out,8.334790,6.936876,1.20152,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,dense-2048,native/torch-eager-out,8.334790,6.905791,1.20693,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,dense-2048,native/numpy-out,8.253083,6.267792,1.31674,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,dense-2048,native/torch-eager-out,8.253083,7.314555,1.12831,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-1024,native/numpy-out,1.515092,1.462283,1.03611,4/6/0,4/6/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native/numpy-out,10.311896,7.220870,1.42807,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native/torch-eager-out,10.311896,8.940090,1.15344,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-1024,native/numpy-out,1.694382,1.533767,1.10472,4/6/0,4/6/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native/numpy-out,12.063710,7.055812,1.70976,3/5/0,3/5/0,same
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native/torch-eager-out,12.063710,9.001329,1.34021,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard,dense-2048,native/numpy-out,75.148380,65.967623,1.13917,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard,dense-2048,native/torch-eager-out,75.148380,67.409035,1.11481,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-1024,native/numpy-out,14.286623,11.897912,1.20077,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-2048,native/numpy-out,89.872103,73.479853,1.22308,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-2048,native/torch-eager-out,89.872103,84.530394,1.06319,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native/numpy-out,98.465767,72.001631,1.36755,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native/torch-eager-out,98.465767,92.964135,1.05918,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-1024,native/numpy-out,14.196387,13.431586,1.05694,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native/numpy-out,85.095562,64.542916,1.31843,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native/torch-eager-out,85.095562,81.430632,1.04501,70/73/1,70/73/1,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,dense-256,native/numpy-out,0.356534,0.128675,2.77082,4/6/0,4/6/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,dense-2048,native/numpy-out,9.376753,6.680343,1.40363,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,dense-2048,native/numpy-out,8.627184,7.732798,1.11566,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native/numpy-out,11.981528,9.600205,1.24805,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native/torch-eager-out,11.981528,11.937428,1.00369,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native/numpy-out,11.526751,7.852403,1.46793,3/5/0,3/5/0,same
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native/torch-eager-out,11.526751,9.087063,1.26848,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard,dense-2048,native/numpy-out,136.315649,118.078639,1.15445,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard,dense-2048,native/torch-eager-out,136.315649,123.063056,1.10769,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-2048,native/numpy-out,144.486615,123.416500,1.17072,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard,dense-2048,native/torch-eager-out,144.486615,129.246507,1.11792,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native/numpy-out,154.476798,106.688623,1.44792,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native/torch-eager-out,154.476798,130.934282,1.17980,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard,dense-2048,native-blas/numpy-out,118.874061,106.688623,1.11421,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-1024,native/numpy-out,12.425722,12.317437,1.00879,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native/numpy-out,129.354202,90.209478,1.43393,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native/torch-eager-out,129.354202,108.312423,1.19427,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard,dense-2048,native-blas/numpy-out,103.043682,90.209478,1.14227,70/73/1,70/73/1,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,dense-2048,native/numpy-out,8.591651,7.778998,1.10447,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard,dense-2048,native/torch-eager-out,8.591651,8.354039,1.02844,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,dense-2048,native/numpy-out,7.860153,6.757744,1.16313,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard,dense-2048,native/torch-eager-out,7.860153,7.213606,1.08963,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-1024,native/numpy-out,1.661739,1.495724,1.11099,4/6/0,4/6/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-1024,native-blas/numpy-out,1.495793,1.495724,1.00005,4/6/0,4/6/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native/numpy-out,8.483690,6.571461,1.29099,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native/torch-eager-out,8.483690,8.379441,1.01244,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard,dense-2048,native-blas/numpy-out,7.839590,6.571461,1.19298,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-1024,native/numpy-out,1.734412,1.650489,1.05085,4/6/0,4/6/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native/numpy-out,10.727597,7.169557,1.49627,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native/torch-eager-out,10.727597,9.140786,1.17360,3/5/0,3/5/0,same
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard,dense-2048,native-blas/numpy-out,8.100409,7.169557,1.12983,3/5/0,3/5/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard,dense-64,cuda-cublas/torch-cuda-eager,3.574000,2.733711,1.30738,5/7/0,5/7/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard,dense-256,cuda-cublas/torch-cuda-eager,3.050396,2.529308,1.20602,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard,dense-1024,cuda-cublas/torch-cuda-eager,3.198078,2.398124,1.33358,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard,dense-2048,cuda-cublas/torch-cuda-eager,3.275883,2.377629,1.37779,3/5/0,3/5/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard,dense-64,cuda-cublas/torch-cuda-eager,3.300000,2.793938,1.18113,5/7/0,5/7/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard,dense-256,cuda-cublas/torch-cuda-eager,2.918551,2.060270,1.41659,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard,dense-1024,cuda-cublas/torch-cuda-eager,2.975176,2.457740,1.21053,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard,dense-2048,cuda-cublas/torch-cuda-eager,3.219470,2.494042,1.29086,3/5/0,3/5/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard,dense-64,cuda-cublas/torch-cuda-eager,2.942527,2.799279,1.05117,5/7/0,5/7/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard,dense-256,cuda-cublas/torch-cuda-eager,2.659251,2.319322,1.14656,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard,dense-1024,cuda-cublas/torch-cuda-eager,2.754136,2.368290,1.16292,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard,dense-2048,cuda-cublas/torch-cuda-eager,2.956296,2.430269,1.21645,3/5/0,3/5/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard,dense-64,cuda-cublas/torch-cuda-eager,3.061647,2.668084,1.14751,5/7/0,5/7/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard,dense-256,cuda-cublas/torch-cuda-eager,2.646861,2.355929,1.12349,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard,dense-1024,cuda-cublas/torch-cuda-eager,2.887714,2.463480,1.17221,4/6/0,4/6/0,same
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard,dense-2048,cuda-cublas/torch-cuda-eager,3.012754,2.253914,1.33668,3/5/0,3/5/0,same
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-16,native-cg/scipy-lu-total,0.524980,0.420542,1.24834,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-16,native-cg/scipy-lu-reuse,0.524980,0.044539,11.78698,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-16,native-pcg/scipy-lu-reuse,0.084638,0.044539,1.90031,48/50/0,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-32,native-cg/scipy-pcg,5.234211,1.678617,3.11817,1799/1836/35,90/91/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-32,native-cg/scipy-lu-total,5.234211,1.711803,3.05772,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-32,native-cg/scipy-lu-reuse,5.234211,0.151298,34.59549,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-2d-32,native-pcg/scipy-lu-reuse,0.477865,0.151298,3.15845,118/122/2,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-3d-6,native-cg/scipy-lu-reuse,0.229177,0.043192,5.30596,253/260/5,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-3d-6,native-pcg/scipy-lu-reuse,0.056012,0.043192,1.29682,24/26/0,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-3d-10,native-cg/scipy-pcg,1.425526,0.770721,1.84960,387/396/7,37/38/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,scaled-poisson-3d-10,native-cg/scipy-lu-reuse,1.425526,0.238824,5.96895,387/396/7,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,poisson-2d-16-neutral,native-cg/scipy-lu-reuse,0.066163,0.029160,2.26901,51/54/1,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-api-fused1-host-standard,poisson-2d-16-neutral,native-pcg/scipy-lu-reuse,0.076712,0.029160,2.63075,51/54/1,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-16,native-cg/scipy-lu-total,0.534291,0.410766,1.30072,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-16,native-cg/scipy-lu-reuse,0.534291,0.041362,12.91736,597/610/11,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-16,native-pcg/scipy-lu-reuse,0.064396,0.041362,1.55688,48/50/0,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-32,native-cg/scipy-pcg,5.366177,1.706146,3.14520,1799/1836/35,90/91/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-32,native-cg/scipy-lu-total,5.366177,1.758676,3.05126,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-32,native-cg/scipy-lu-reuse,5.366177,0.146320,36.67413,1799/1836/35,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-2d-32,native-pcg/scipy-lu-reuse,0.440864,0.146320,3.01300,118/122/2,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-3d-6,native-cg/scipy-lu-reuse,0.218823,0.040093,5.45784,253/260/5,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-3d-10,native-cg/scipy-pcg,1.417507,0.764225,1.85483,387/396/7,37/38/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,scaled-poisson-3d-10,native-cg/scipy-lu-reuse,1.417507,0.245362,5.77720,387/396/7,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,poisson-2d-16-neutral,native-cg/scipy-lu-reuse,0.057049,0.036902,1.54594,51/54/1,0/1/unknown,unknown
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard,poisson-2d-16-neutral,native-pcg/scipy-lu-reuse,0.066589,0.036902,1.80446,51/54/1,0/1/unknown,unknown
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-2048,native/numpy-out,220.748359,177.791309,1.24162,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-2048,native/torch-eager-out,220.748359,184.737752,1.19493,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-2048,native/numpy-out,212.830680,179.777652,1.18386,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-2048,native/torch-eager-out,212.830680,183.683694,1.15868,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-1024,native/numpy-out,22.757705,22.545109,1.00943,69/139/0,69/139/0,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native/numpy-out,281.827204,186.009797,1.51512,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native/torch-eager-out,281.827204,244.897260,1.15080,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-1024,native/numpy-out,24.210890,23.464051,1.03183,69/139/0,69/139/0,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native/numpy-out,291.849515,176.697884,1.65169,68/137/0,68/137/0,same
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native/torch-eager-out,291.849515,241.590420,1.20803,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-1024,native/numpy-out,20.311607,20.091234,1.01097,69/139/0,69/139/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-2048,native/numpy-out,138.859858,118.452883,1.17228,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-2048,native/torch-eager-out,138.859858,132.826935,1.04542,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-1024,native/numpy-out,20.843526,20.175528,1.03311,69/139/0,69/139/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-2048,native/numpy-out,141.523616,126.947609,1.11482,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-2048,native/torch-eager-out,141.523616,129.820262,1.09015,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-1024,native/numpy-out,22.975393,22.550152,1.01886,69/139/0,69/139/0,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native/numpy-out,179.602010,124.485870,1.44275,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native/torch-eager-out,179.602010,166.166413,1.08086,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-1024,native/numpy-out,23.716005,21.913575,1.08225,69/139/0,69/139/0,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native/numpy-out,186.699366,126.632969,1.47433,68/137/0,68/137/0,same
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native/torch-eager-out,186.699366,171.460670,1.08888,68/137/0,68/137/0,same
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit,poisson-128,native/torch-eager-out,37.508587,36.991538,1.01398,325/651/0,325/651/0,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-2048,native/numpy-out,205.294025,171.672435,1.19585,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit,dense-2048,native/torch-eager-out,205.294025,182.593021,1.12433,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-2048,native/numpy-out,205.291733,167.623239,1.22472,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit,dense-2048,native/torch-eager-out,205.291733,176.855918,1.16079,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-1024,native/numpy-out,23.980320,23.084946,1.03879,69/139/0,69/139/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native/numpy-out,251.192581,167.331394,1.50117,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native/torch-eager-out,251.192581,228.033074,1.10156,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit,dense-2048,native-blas/numpy-out,218.608018,167.331394,1.30644,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-1024,native/numpy-out,24.508141,23.734554,1.03259,69/139/0,69/139/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native/numpy-out,273.468731,162.080812,1.68724,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native/torch-eager-out,273.468731,221.617462,1.23397,68/137/0,68/137/0,same
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit,dense-2048,native-blas/numpy-out,202.901941,162.080812,1.25186,68/137/0,68/137/0,same

Complete per-configuration measurements
---------------------------------------

All 1,548 candidate rows follow. Forward/reverse ms are single-process medians;
the reported median and range summarize those two medians. TrueR/tol is the
largest independently measured residual-to-threshold ratio across both process outcomes.

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.042042,0.051391,0.046717,0.042042–0.051391,1.22240,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.240072,0.237900,0.238986,0.237900–0.240072,0.99095,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.414176,0.399929,0.407052,0.399929–0.414176,0.96560,49/51/0,49/51/0,0.635707
   dense-256,native,0.480451,0.451024,0.465738,0.451024–0.480451,0.93875,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.675650,0.704204,0.689927,0.675650–0.704204,1.04226,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.883778,0.917929,0.900854,0.883778–0.917929,1.03864,67/70/1,67/70/1,0.972357
   dense-1024,native,10.164416,9.101714,9.633065,9.101714–10.164416,0.89545,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,10.199874,9.658535,9.929205,9.658535–10.199874,0.94693,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,11.915261,10.702595,11.308927,10.702595–11.915261,0.89823,70/73/1,70/73/1,0.928014
   dense-2048,native,129.304374,107.872339,118.588357,107.872339–129.304374,0.83425,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,98.857035,93.339228,96.098132,93.339228–98.857035,0.94418,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,103.470585,96.886352,100.178468,96.886352–103.470585,0.93637,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.051488,0.053455,0.052471,0.051488–0.053455,1.03818,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.256653,0.265984,0.261318,0.256653–0.265984,1.03636,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.399806,0.425864,0.412835,0.399806–0.425864,1.06518,49/51/0,49/51/0,0.635707
   dense-256,native,0.499359,0.529795,0.514577,0.499359–0.529795,1.06095,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.646805,0.884293,0.765549,0.646805–0.884293,1.36717,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.899253,1.226476,1.062864,0.899253–1.226476,1.36388,67/70/1,67/70/1,0.972357
   dense-1024,native,10.297732,9.449076,9.873404,9.449076–10.297732,0.91759,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,11.239047,10.133664,10.686355,10.133664–11.239047,0.90165,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,12.638330,11.497200,12.067765,11.497200–12.638330,0.90971,70/73/1,70/73/1,0.928014
   dense-2048,native,160.947046,123.268728,142.107887,123.268728–160.947046,0.76590,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,118.130549,105.959305,112.044927,105.959305–118.130549,0.89697,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,125.354652,109.348776,117.351714,109.348776–125.354652,0.87232,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.039837,0.044827,0.042332,0.039837–0.044827,1.12527,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.290721,0.310719,0.300720,0.290721–0.310719,1.06879,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.489409,0.548724,0.519067,0.489409–0.548724,1.12120,49/51/0,49/51/0,0.635707
   dense-256,native,0.472610,0.502826,0.487718,0.472610–0.502826,1.06393,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.768911,0.753502,0.761207,0.753502–0.768911,0.97996,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.954792,1.056223,1.005507,0.954792–1.056223,1.10623,67/70/1,67/70/1,0.972357
   dense-1024,native,11.622953,13.117042,12.369997,11.622953–13.117042,1.12855,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,12.858482,12.668935,12.763708,12.668935–12.858482,0.98526,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,13.582341,14.714301,14.148321,13.582341–14.714301,1.08334,70/73/1,70/73/1,0.928014
   dense-2048,native,125.944733,151.005531,138.475132,125.944733–151.005531,1.19898,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,93.356235,87.972876,90.664556,87.972876–93.356235,0.94234,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,104.198953,122.644617,113.421785,104.198953–122.644617,1.17702,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.046035,0.043688,0.044862,0.043688–0.046035,0.94903,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.290687,0.266733,0.278710,0.266733–0.290687,0.91759,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.498188,0.458132,0.478160,0.458132–0.498188,0.91960,49/51/0,49/51/0,0.635707
   dense-256,native,0.525966,0.477165,0.501565,0.477165–0.525966,0.90722,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.914776,0.686519,0.800648,0.686519–0.914776,0.75048,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.142359,1.014901,1.078630,1.014901–1.142359,0.88843,67/70/1,67/70/1,0.972357
   dense-1024,native,11.956514,12.590690,12.273602,11.956514–12.590690,1.05304,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,13.205251,12.163403,12.684327,12.163403–13.205251,0.92110,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,14.001599,14.265955,14.133777,14.001599–14.265955,1.01888,70/73/1,70/73/1,0.928014
   dense-2048,native,110.933374,158.949820,134.941597,110.933374–158.949820,1.43284,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,104.025889,95.853183,99.939536,95.853183–104.025889,0.92144,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,101.765540,134.863699,118.314620,101.765540–134.863699,1.32524,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.014788,0.018214,0.016501,0.014788–0.018214,1.23164,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.041266,0.052284,0.046775,0.041266–0.052284,1.26701,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.070036,0.105834,0.087935,0.070036–0.105834,1.51116,5/7/0,5/7/0,0.0796191
   dense-256,native,0.064266,0.061747,0.063007,0.061747–0.064266,0.96081,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.106849,0.082109,0.094479,0.082109–0.106849,0.76846,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.177039,0.163768,0.170404,0.163768–0.177039,0.92503,4/6/0,4/6/0,0.209854
   dense-1024,native,0.930840,0.943727,0.937284,0.930840–0.943727,1.01384,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.034040,0.995771,1.014906,0.995771–1.034040,0.96299,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.206333,1.231712,1.219022,1.206333–1.231712,1.02104,4/6/0,4/6/0,0.013684
   dense-2048,native,9.000502,7.669079,8.334790,7.669079–9.000502,0.85207,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,7.527196,6.346555,6.936876,6.346555–7.527196,0.84315,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,7.082562,6.729018,6.905791,6.729018–7.082562,0.95008,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.025145,0.027469,0.026307,0.025145–0.027469,1.09245,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.067018,0.060069,0.063544,0.060069–0.067018,0.89631,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.129183,0.121844,0.125514,0.121844–0.129183,0.94319,5/7/0,5/7/0,0.0796191
   dense-256,native,0.055496,0.061199,0.058348,0.055496–0.061199,1.10276,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.079061,0.095608,0.087334,0.079061–0.095608,1.20931,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.146375,0.160874,0.153624,0.146375–0.160874,1.09905,4/6/0,4/6/0,0.209854
   dense-1024,native,0.964394,0.944508,0.954451,0.944508–0.964394,0.97938,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.101705,1.014166,1.057936,1.014166–1.101705,0.92054,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.362163,1.218393,1.290278,1.218393–1.362163,0.89445,4/6/0,4/6/0,0.013684
   dense-2048,native,9.127887,7.378279,8.253083,7.378279–9.127887,0.80832,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,6.078946,6.456638,6.267792,6.078946–6.456638,1.06213,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,7.508487,7.120622,7.314555,7.120622–7.508487,0.94834,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.007623,0.008434,0.008029,0.007623–0.008434,1.10645,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.031535,0.033780,0.032657,0.031535–0.033780,1.07122,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.049126,0.066850,0.057988,0.049126–0.066850,1.36078,5/7/0,5/7/0,0.0796191
   dense-256,native,0.062302,0.039245,0.050773,0.039245–0.062302,0.62990,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.113279,0.052251,0.082765,0.052251–0.113279,0.46126,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.154772,0.071721,0.113246,0.071721–0.154772,0.46339,4/6/0,4/6/0,0.209854
   dense-1024,native,1.356553,1.673633,1.515092,1.356553–1.673633,1.23374,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.395158,1.529409,1.462283,1.395158–1.529409,1.09623,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.552245,1.796318,1.674281,1.552245–1.796318,1.15724,4/6/0,4/6/0,0.013684
   dense-2048,native,9.869562,10.754231,10.311896,9.869562–10.754231,1.08964,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.073026,6.368714,7.220870,6.368714–8.073026,0.78889,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,9.024957,8.855222,8.940090,8.855222–9.024957,0.98119,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.008440,0.009772,0.009106,0.008440–0.009772,1.15788,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.032049,0.051934,0.041991,0.032049–0.051934,1.62044,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.051867,0.094785,0.073326,0.051867–0.094785,1.82749,5/7/0,5/7/0,0.0796191
   dense-256,native,0.052969,0.051977,0.052473,0.051977–0.052969,0.98126,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.084215,0.064677,0.074446,0.064677–0.084215,0.76800,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.139233,0.094829,0.117031,0.094829–0.139233,0.68109,4/6/0,4/6/0,0.209854
   dense-1024,native,1.481245,1.907518,1.694382,1.481245–1.907518,1.28778,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.519797,1.547738,1.533767,1.519797–1.547738,1.01838,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.838465,2.091711,1.965088,1.838465–2.091711,1.13775,4/6/0,4/6/0,0.013684
   dense-2048,native,12.971573,11.155848,12.063710,11.155848–12.971573,0.86002,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,7.261483,6.850140,7.055812,6.850140–7.261483,0.94335,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,8.826097,9.176562,9.001329,8.826097–9.176562,1.03971,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.031871,0.021333,0.026602,0.021333–0.031871,0.66935,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.229409,0.191270,0.210340,0.191270–0.229409,0.83375,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.317082,0.260898,0.288990,0.260898–0.317082,0.82281,26/28/0,26/28/0,0.579621
   poisson-32,native,0.440489,0.446593,0.443541,0.440489–0.446593,1.01386,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.616805,1.747361,1.682083,1.616805–1.747361,1.08075,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.713000,1.787205,1.750103,1.713000–1.787205,1.04332,118/122/2,118/122/2,0.840189
   poisson-64,native,2.299392,2.560213,2.429802,2.299392–2.560213,1.11343,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.633023,6.247995,5.940509,5.633023–6.247995,1.10917,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.843758,5.481753,5.162756,4.843758–5.481753,1.13171,201/207/4,201/207/4,0.907427
   poisson-128,native,26.601875,31.203138,28.902506,26.601875–31.203138,1.17297,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,51.332512,60.643647,55.988079,51.332512–60.643647,1.18139,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,39.256869,44.079685,41.668277,39.256869–44.079685,1.12285,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.025451,0.029089,0.027270,0.025451–0.029089,1.14296,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.176896,0.211620,0.194258,0.176896–0.211620,1.19630,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.228000,0.293880,0.260940,0.228000–0.293880,1.28894,26/28/0,26/28/0,0.579621
   poisson-32,native,0.449923,0.510583,0.480253,0.449923–0.510583,1.13482,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.372335,1.534296,1.453315,1.372335–1.534296,1.11802,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.357682,1.718571,1.538127,1.357682–1.718571,1.26581,118/122/2,118/122/2,0.840189
   poisson-64,native,2.712429,2.965480,2.838954,2.712429–2.965480,1.09329,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.948282,5.812907,5.380594,4.948282–5.812907,1.17473,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.244886,5.122977,4.683931,4.244886–5.122977,1.20686,201/207/4,201/207/4,0.907427
   poisson-128,native,36.630992,37.167636,36.899313,36.630992–37.167636,1.01465,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,54.520058,56.186492,55.353275,54.520058–56.186492,1.03057,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,40.153928,41.697735,40.925832,40.153928–41.697735,1.03845,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.013849,0.013518,0.013683,0.013518–0.013849,0.97606,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.194431,0.191082,0.192756,0.191082–0.194431,0.98278,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.244871,0.269095,0.256983,0.244871–0.269095,1.09893,26/28/0,26/28/0,0.579621
   poisson-32,native,0.366952,0.385020,0.375986,0.366952–0.385020,1.04924,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.618348,1.614594,1.616471,1.614594–1.618348,0.99768,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.630098,1.616294,1.623196,1.616294–1.630098,0.99153,118/122/2,118/122/2,0.840189
   poisson-64,native,2.332558,2.288721,2.310639,2.288721–2.332558,0.98121,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.186020,5.612672,5.399346,5.186020–5.612672,1.08227,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.541574,5.185937,4.863755,4.541574–5.185937,1.14188,201/207/4,201/207/4,0.907427
   poisson-128,native,28.316420,27.603905,27.960163,27.603905–28.316420,0.97484,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,57.283204,55.636119,56.459662,55.636119–57.283204,0.97125,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,39.501411,40.837140,40.169276,39.501411–40.837140,1.03381,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.014404,0.013977,0.014190,0.013977–0.014404,0.97036,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.219375,0.183497,0.201435,0.183497–0.219375,0.83645,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.285718,0.332267,0.308992,0.285718–0.332267,1.16292,26/28/0,26/28/0,0.579621
   poisson-32,native,0.482189,0.496997,0.489593,0.482189–0.496997,1.03071,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.698614,1.607366,1.652990,1.607366–1.698614,0.94628,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.754672,1.664168,1.709420,1.664168–1.754672,0.94842,118/122/2,118/122/2,0.840189
   poisson-64,native,3.232619,3.325260,3.278939,3.232619–3.325260,1.02866,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,6.067937,5.852534,5.960236,5.852534–6.067937,0.96450,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.442797,5.325077,5.383937,5.325077–5.442797,0.97837,201/207/4,201/207/4,0.907427
   poisson-128,native,38.668094,34.542443,36.605269,34.542443–38.668094,0.89331,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,61.890016,53.289668,57.589842,53.289668–61.890016,0.86104,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,43.463581,40.949259,42.206421,40.949259–43.463581,0.94215,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.031088,0.035380,0.033234,0.031088–0.035380,1.13806,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.243182,0.287440,0.265311,0.243182–0.287440,1.18199,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.356943,0.412011,0.384477,0.356943–0.412011,1.15428,26/28/0,26/28/0,0.579621
   poisson-32,native,0.461438,0.531527,0.496482,0.461438–0.531527,1.15189,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.695724,2.010363,1.853044,1.695724–2.010363,1.18555,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.864926,2.164917,2.014921,1.864926–2.164917,1.16086,118/122/2,118/122/2,0.840189
   poisson-64,native,3.029975,2.789119,2.909547,2.789119–3.029975,0.92051,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,7.400319,6.233682,6.817000,6.233682–7.400319,0.84235,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,6.478155,5.983615,6.230886,5.983615–6.478155,0.92366,201/207/4,201/207/4,0.907427
   poisson-128,native,31.827206,32.274634,32.050920,31.827206–32.274634,1.01406,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,61.124602,63.018026,62.071314,61.124602–63.018026,1.03098,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,47.410352,48.552697,47.981524,47.410352–48.552697,1.02409,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.027803,0.033178,0.030491,0.027803–0.033178,1.19330,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.221638,0.239010,0.230324,0.221638–0.239010,1.07838,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.345060,0.367364,0.356212,0.345060–0.367364,1.06464,26/28/0,26/28/0,0.579621
   poisson-32,native,0.518365,0.601826,0.560095,0.518365–0.601826,1.16101,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.576717,1.787078,1.681897,1.576717–1.787078,1.13342,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.631835,2.054653,1.843244,1.631835–2.054653,1.25911,118/122/2,118/122/2,0.840189
   poisson-64,native,3.422883,3.590146,3.506514,3.422883–3.590146,1.04887,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.602214,6.450267,6.026241,5.602214–6.450267,1.15138,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.276373,5.785624,5.530998,5.276373–5.785624,1.09652,201/207/4,201/207/4,0.907427
   poisson-128,native,42.787598,43.882900,43.335249,42.787598–43.882900,1.02560,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,61.642609,62.636544,62.139576,61.642609–62.636544,1.01612,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,46.456491,49.362879,47.909686,46.456491–49.362879,1.06256,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.012495,0.023141,0.017819,0.012495–0.023141,1.85199,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.193285,0.246199,0.219742,0.193285–0.246199,1.27376,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.268411,0.338045,0.303228,0.268411–0.338045,1.25943,26/28/0,26/28/0,0.579621
   poisson-32,native,0.424665,0.452130,0.438397,0.424665–0.452130,1.06467,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.576444,1.795003,1.685723,1.576444–1.795003,1.13864,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.707086,2.006509,1.856798,1.707086–2.006509,1.17540,118/122/2,118/122/2,0.840189
   poisson-64,native,2.696801,2.864968,2.780884,2.696801–2.864968,1.06236,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.769901,6.482422,6.126162,5.769901–6.482422,1.12349,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.473414,6.438897,5.956155,5.473414–6.438897,1.17640,201/207/4,201/207/4,0.907427
   poisson-128,native,35.322666,34.903525,35.113096,34.903525–35.322666,0.98813,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,63.728974,62.301761,63.015368,62.301761–63.728974,0.97760,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,49.091096,49.440229,49.265663,49.091096–49.440229,1.00711,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.021358,0.018082,0.019720,0.018082–0.021358,0.84663,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.266699,0.230255,0.248477,0.230255–0.266699,0.86335,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.396689,0.316368,0.356528,0.316368–0.396689,0.79752,26/28/0,26/28/0,0.579621
   poisson-32,native,0.576849,0.511960,0.544404,0.511960–0.576849,0.88751,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.903189,1.629072,1.766131,1.629072–1.903189,0.85597,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.961772,1.773869,1.867821,1.773869–1.961772,0.90422,118/122/2,118/122/2,0.840189
   poisson-64,native,3.646832,3.266466,3.456649,3.266466–3.646832,0.89570,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,7.312289,5.771456,6.541872,5.771456–7.312289,0.78928,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,6.596646,5.750839,6.173743,5.750839–6.596646,0.87178,201/207/4,201/207/4,0.907427
   poisson-128,native,43.479674,44.708432,44.094053,43.479674–44.708432,1.02826,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,64.797979,64.545672,64.671826,64.545672–64.797979,0.99611,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,48.550772,50.543152,49.546962,48.550772–50.543152,1.04104,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.065118,0.061072,0.063095,0.061072–0.065118,0.93785,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.212755,0.197722,0.205239,0.197722–0.212755,0.92934,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.322556,0.301318,0.311937,0.301318–0.322556,0.93416,49/51/0,49/51/0,0.635707
   dense-256,native,0.348325,0.330579,0.339452,0.330579–0.348325,0.94905,67/70/1,67/70/1,0.993806
   dense-256,numpy-out,0.628718,0.607151,0.617935,0.607151–0.628718,0.96570,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.785703,0.748074,0.766889,0.748074–0.785703,0.95211,67/70/1,67/70/1,0.972357
   dense-1024,native,10.653280,9.878929,10.266105,9.878929–10.653280,0.92731,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,12.091913,10.287214,11.189564,10.287214–12.091913,0.85075,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,15.964161,11.481675,13.722918,11.481675–15.964161,0.71922,70/73/1,70/73/1,0.928014
   dense-2048,native,77.355496,72.941265,75.148380,72.941265–77.355496,0.94294,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,62.671601,69.263644,65.967623,62.671601–69.263644,1.10518,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,66.797571,68.020499,67.409035,66.797571–68.020499,1.01831,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.083082,0.149986,0.116534,0.083082–0.149986,1.80528,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.190599,0.353869,0.272234,0.190599–0.353869,1.85662,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.284835,0.542995,0.413915,0.284835–0.542995,1.90635,49/51/0,49/51/0,0.635707
   dense-256,native,0.387273,0.438877,0.413075,0.387273–0.438877,1.13325,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.632659,0.507797,0.570228,0.507797–0.632659,0.80264,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.844494,0.782814,0.813654,0.782814–0.844494,0.92696,67/70/1,67/70/1,0.972357
   dense-1024,native,14.093803,14.479444,14.286623,14.093803–14.479444,1.02736,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,11.297049,12.498774,11.897912,11.297049–12.498774,1.10638,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,16.849991,17.164399,17.007195,16.849991–17.164399,1.01866,70/73/1,70/73/1,0.928014
   dense-2048,native,98.013868,81.730339,89.872103,81.730339–98.013868,0.83387,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,76.399649,70.560057,73.479853,70.560057–76.399649,0.92357,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,91.552282,77.508507,84.530394,77.508507–91.552282,0.84660,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.063662,0.053000,0.058331,0.053000–0.063662,0.83252,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.216528,0.200747,0.208638,0.200747–0.216528,0.92712,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.352818,0.284973,0.318896,0.284973–0.352818,0.80771,49/51/0,49/51/0,0.635707
   dense-256,native,0.341975,0.355901,0.348938,0.341975–0.355901,1.04072,67/70/1,67/70/1,0.993806
   dense-256,numpy-out,0.545014,0.561560,0.553287,0.545014–0.561560,1.03036,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.715966,0.723168,0.719567,0.715966–0.723168,1.01006,67/70/1,67/70/1,0.972357
   dense-1024,native,13.386845,12.512398,12.949621,12.512398–13.386845,0.93468,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,14.221844,12.897692,13.559768,12.897692–14.221844,0.90689,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,15.802729,14.285600,15.044164,14.285600–15.802729,0.90400,70/73/1,70/73/1,0.928014
   dense-2048,native,89.425431,107.506102,98.465767,89.425431–107.506102,1.20219,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,72.284600,71.718662,72.001631,71.718662–72.284600,0.99217,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,89.125669,96.802603,92.964135,89.125669–96.802603,1.08614,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.092463,0.076744,0.084603,0.076744–0.092463,0.83000,49/51/0,49/51/0,0.607749
   dense-64,numpy-out,0.247906,0.191110,0.219508,0.191110–0.247906,0.77090,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.373443,0.298687,0.336065,0.298687–0.373443,0.79982,49/51/0,49/51/0,0.635707
   dense-256,native,0.425837,0.416687,0.421262,0.416687–0.425837,0.97851,67/70/1,67/70/1,0.971216
   dense-256,numpy-out,0.726081,0.858204,0.792142,0.726081–0.858204,1.18197,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.892516,1.002409,0.947462,0.892516–1.002409,1.12313,67/70/1,67/70/1,0.972357
   dense-1024,native,15.091962,13.300812,14.196387,13.300812–15.091962,0.88132,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,13.421449,13.441722,13.431586,13.421449–13.441722,1.00151,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,19.357654,14.348959,16.853306,14.348959–19.357654,0.74126,70/73/1,70/73/1,0.928014
   dense-2048,native,84.067048,86.124076,85.095562,84.067048–86.124076,1.02447,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,71.891871,57.193961,64.542916,57.193961–71.891871,0.79556,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,83.798396,79.062867,81.430632,79.062867–83.798396,0.94349,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.019507,0.018941,0.019224,0.018941–0.019507,0.97098,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.041964,0.040109,0.041036,0.040109–0.041964,0.95582,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.071847,0.068420,0.070134,0.068420–0.071847,0.95231,5/7/0,5/7/0,0.0796191
   dense-256,native,0.058183,0.654885,0.356534,0.058183–0.654885,11.25551,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.094093,0.163256,0.128675,0.094093–0.163256,1.73504,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.143640,0.752301,0.447971,0.143640–0.752301,5.23741,4/6/0,4/6/0,0.209854
   dense-1024,native,0.600295,0.923859,0.762077,0.600295–0.923859,1.53901,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,2.151398,0.973448,1.562423,0.973448–2.151398,0.45247,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,0.887486,1.091255,0.989371,0.887486–1.091255,1.22960,4/6/0,4/6/0,0.013684
   dense-2048,native,11.041360,7.712146,9.376753,7.712146–11.041360,0.69848,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.124178,5.236508,6.680343,5.236508–8.124178,0.64456,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,12.902162,7.625003,10.263582,7.625003–12.902162,0.59099,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.024474,0.024078,0.024276,0.024078–0.024474,0.98382,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.044460,0.041927,0.043194,0.041927–0.044460,0.94303,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.074584,0.073756,0.074170,0.073756–0.074584,0.98889,5/7/0,5/7/0,0.0796191
   dense-256,native,0.056453,0.045241,0.050847,0.045241–0.056453,0.80139,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.088272,0.084177,0.086224,0.084177–0.088272,0.95360,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.136444,0.130961,0.133702,0.130961–0.136444,0.95981,4/6/0,4/6/0,0.209854
   dense-1024,native,0.638065,0.565876,0.601970,0.565876–0.638065,0.88686,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.809114,2.073077,1.941095,1.809114–2.073077,1.14591,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,0.950419,0.887305,0.918862,0.887305–0.950419,0.93359,4/6/0,4/6/0,0.013684
   dense-2048,native,9.675095,7.579272,8.627184,7.579272–9.675095,0.78338,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.824330,6.641267,7.732798,6.641267–8.824330,0.75261,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,11.084858,7.680822,9.382841,7.680822–11.084858,0.69291,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.013201,0.012908,0.013054,0.012908–0.013201,0.97777,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.032443,0.032413,0.032428,0.032413–0.032443,0.99909,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.055272,0.054213,0.054742,0.054213–0.055272,0.98086,5/7/0,5/7/0,0.0796191
   dense-256,native,0.044707,0.036771,0.040739,0.036771–0.044707,0.82248,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.075810,0.054335,0.065073,0.054335–0.075810,0.71673,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.107923,0.071289,0.089606,0.071289–0.107923,0.66055,4/6/0,4/6/0,0.209854
   dense-1024,native,1.391698,1.629409,1.510553,1.391698–1.629409,1.17081,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,2.153242,1.291100,1.722171,1.291100–2.153242,0.59961,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,2.730889,1.600640,2.165765,1.600640–2.730889,0.58612,4/6/0,4/6/0,0.013684
   dense-2048,native,13.534679,10.428379,11.981528,10.428379–13.534679,0.77049,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,12.174665,7.025744,9.600205,7.025744–12.174665,0.57708,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,15.051042,8.823814,11.937428,8.823814–15.051042,0.58626,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.014805,0.015130,0.014967,0.014805–0.015130,1.02192,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.029285,0.030294,0.029790,0.029285–0.030294,1.03444,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.046651,0.047774,0.047212,0.046651–0.047774,1.02405,5/7/0,5/7/0,0.0796191
   dense-256,native,0.035788,0.035820,0.035804,0.035788–0.035820,1.00089,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.065765,0.058417,0.062091,0.058417–0.065765,0.88828,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.086708,0.071508,0.079108,0.071508–0.086708,0.82469,4/6/0,4/6/0,0.209854
   dense-1024,native,1.158195,1.726882,1.442538,1.158195–1.726882,1.49101,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.637968,1.407477,1.522723,1.407477–1.637968,0.85928,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.398050,1.776625,1.587338,1.398050–1.776625,1.27079,4/6/0,4/6/0,0.013684
   dense-2048,native,10.734421,12.319081,11.526751,10.734421–12.319081,1.14762,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.081654,7.623151,7.852403,7.623151–8.081654,0.94327,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,8.334122,9.840003,9.087063,8.334122–9.840003,1.18069,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.041316,0.029912,0.035614,0.029912–0.041316,0.72397,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.218425,0.147822,0.183124,0.147822–0.218425,0.67676,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.310314,0.217961,0.264137,0.217961–0.310314,0.70239,26/28/0,26/28/0,0.579621
   poisson-32,native,0.497779,0.592323,0.545051,0.497779–0.592323,1.18993,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.494228,1.160658,1.327443,1.160658–1.494228,0.77676,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.423922,1.267337,1.345630,1.267337–1.423922,0.89003,118/122/2,118/122/2,0.840189
   poisson-64,native,2.361214,2.377178,2.369196,2.361214–2.377178,1.00676,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.201628,4.097287,4.649457,4.097287–5.201628,0.78769,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.378582,3.684352,4.031467,3.684352–4.378582,0.84145,201/207/4,201/207/4,0.907427
   poisson-128,native,35.136682,35.390229,35.263456,35.136682–35.390229,1.00722,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,60.020209,63.385089,61.702649,60.020209–63.385089,1.05606,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,37.574818,40.484437,39.029627,37.574818–40.484437,1.07744,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.068317,0.062086,0.065201,0.062086–0.068317,0.90879,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.235587,0.221821,0.228704,0.221821–0.235587,0.94157,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.337468,0.312248,0.324858,0.312248–0.337468,0.92527,26/28/0,26/28/0,0.579621
   poisson-32,native,0.745983,0.806215,0.776099,0.745983–0.806215,1.08074,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.746311,1.715153,1.730732,1.715153–1.746311,0.98216,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.755145,1.742765,1.748954,1.742765–1.755145,0.99295,118/122/2,118/122/2,0.840189
   poisson-64,native,3.381817,3.290828,3.336323,3.290828–3.381817,0.97309,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,6.191736,6.009927,6.100831,6.009927–6.191736,0.97064,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.435448,5.062534,5.248991,5.062534–5.435448,0.93139,201/207/4,201/207/4,0.907427
   poisson-128,native,44.689079,35.475184,40.082132,35.475184–44.689079,0.79382,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,83.924302,72.853578,78.388940,72.853578–83.924302,0.86809,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,52.670189,41.859197,47.264693,41.859197–52.670189,0.79474,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.023560,0.031370,0.027465,0.023560–0.031370,1.33147,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.148369,0.211906,0.180138,0.148369–0.211906,1.42824,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.217687,0.296834,0.257261,0.217687–0.296834,1.36358,26/28/0,26/28/0,0.579621
   poisson-32,native,0.435385,0.463230,0.449308,0.435385–0.463230,1.06395,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.173388,1.385987,1.279688,1.173388–1.385987,1.18118,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.227880,1.397877,1.312879,1.227880–1.397877,1.13845,118/122/2,118/122/2,0.840189
   poisson-64,native,2.779656,2.448133,2.613894,2.448133–2.779656,0.88073,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.712214,5.677868,5.195041,4.712214–5.677868,1.20493,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.192302,4.940224,4.566263,4.192302–4.940224,1.17840,201/207/4,201/207/4,0.907427
   poisson-128,native,35.512594,30.658029,33.085311,30.658029–35.512594,0.86330,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,80.389177,71.046376,75.717776,71.046376–80.389177,0.88378,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,49.794169,46.170793,47.982481,46.170793–49.794169,0.92723,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.044290,0.049455,0.046872,0.044290–0.049455,1.11662,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.191856,0.197958,0.194907,0.191856–0.197958,1.03180,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.267942,0.258943,0.263443,0.258943–0.267942,0.96641,26/28/0,26/28/0,0.579621
   poisson-32,native,0.672226,0.700132,0.686179,0.672226–0.700132,1.04151,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.469928,1.576434,1.523181,1.469928–1.576434,1.07246,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.468215,1.685547,1.576881,1.468215–1.685547,1.14802,118/122/2,118/122/2,0.840189
   poisson-64,native,3.157139,3.010328,3.083734,3.010328–3.157139,0.95350,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.289020,5.320510,5.304765,5.289020–5.320510,1.00595,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,4.556838,4.522960,4.539899,4.522960–4.556838,0.99257,201/207/4,201/207/4,0.907427
   poisson-128,native,38.371007,35.992477,37.181742,35.992477–38.371007,0.93801,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,71.531558,66.790138,69.160848,66.790138–71.531558,0.93372,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,40.270799,41.354943,40.812871,40.270799–41.354943,1.02692,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.040844,0.040235,0.040539,0.040235–0.040844,0.98509,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.220495,0.182654,0.201574,0.182654–0.220495,0.82839,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.335720,0.268749,0.302234,0.268749–0.335720,0.80051,26/28/0,26/28/0,0.579621
   poisson-32,native,0.571348,0.516239,0.543793,0.516239–0.571348,0.90355,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.686878,1.278064,1.482471,1.278064–1.686878,0.75765,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.839819,1.441993,1.640905,1.441993–1.839819,0.78377,118/122/2,118/122/2,0.840189
   poisson-64,native,3.044243,2.737413,2.890828,2.737413–3.044243,0.89921,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,6.059470,4.743365,5.401417,4.743365–6.059470,0.78280,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,6.178381,4.970459,5.574420,4.970459–6.178381,0.80449,201/207/4,201/207/4,0.907427
   poisson-128,native,36.647467,33.688196,35.167831,33.688196–36.647467,0.91925,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,85.071947,81.775348,83.423648,81.775348–85.071947,0.96125,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,54.988901,47.285028,51.136964,47.285028–54.988901,0.85990,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.079436,0.072831,0.076133,0.072831–0.079436,0.91685,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.262977,0.266624,0.264800,0.262977–0.266624,1.01387,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.457193,0.408010,0.432601,0.408010–0.457193,0.89242,26/28/0,26/28/0,0.579621
   poisson-32,native,0.823956,0.879211,0.851583,0.823956–0.879211,1.06706,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.715057,1.836024,1.775541,1.715057–1.836024,1.07053,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.865071,2.030029,1.947550,1.865071–2.030029,1.08845,118/122/2,118/122/2,0.840189
   poisson-64,native,3.962962,3.734192,3.848577,3.734192–3.962962,0.94227,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,5.990952,5.774729,5.882840,5.774729–5.990952,0.96391,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.831005,5.542179,5.686592,5.542179–5.831005,0.95047,201/207/4,201/207/4,0.907427
   poisson-128,native,47.644652,39.432088,43.538370,39.432088–47.644652,0.82763,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,82.517524,79.376525,80.947025,79.376525–82.517524,0.96194,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,49.194164,46.785250,47.989707,46.785250–49.194164,0.95103,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.032664,0.030293,0.031478,0.030293–0.032664,0.92741,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.178596,0.211229,0.194913,0.178596–0.211229,1.18272,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.291312,0.328406,0.309858,0.291312–0.328406,1.12733,26/28/0,26/28/0,0.579621
   poisson-32,native,0.489810,0.572708,0.531259,0.489810–0.572708,1.16924,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.296391,1.726076,1.511233,1.296391–1.726076,1.33145,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,1.470618,1.853272,1.661945,1.470618–1.853272,1.26020,118/122/2,118/122/2,0.840189
   poisson-64,native,2.839299,2.849884,2.844592,2.839299–2.849884,1.00373,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,4.964758,6.545261,5.755009,4.964758–6.545261,1.31834,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,5.063765,5.961767,5.512766,5.063765–5.961767,1.17734,201/207/4,201/207/4,0.907427
   poisson-128,native,34.138624,31.990070,33.064347,31.990070–34.138624,0.93706,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,71.434898,74.703031,73.068965,71.434898–74.703031,1.04575,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,41.378858,44.456725,42.917792,41.378858–44.456725,1.07438,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.059388,0.073602,0.066494,0.059388–0.073602,1.23934,26/28/0,26/28/0,0.579621
   poisson-8,scipy-out,0.223985,0.305271,0.264628,0.223985–0.305271,1.36291,26/28/0,26/28/0,0.579621
   poisson-8,torch-eager-out,0.331560,0.471430,0.401495,0.331560–0.471430,1.42185,26/28/0,26/28/0,0.579621
   poisson-32,native,0.927856,0.897625,0.912740,0.897625–0.927856,0.96742,118/122/2,118/122/2,0.840189
   poisson-32,scipy-out,1.978459,1.988294,1.983377,1.978459–1.988294,1.00497,118/122/2,118/122/2,0.840189
   poisson-32,torch-eager-out,2.359587,2.112452,2.236020,2.112452–2.359587,0.89526,118/122/2,118/122/2,0.840189
   poisson-64,native,4.320277,4.098594,4.209435,4.098594–4.320277,0.94869,201/207/4,201/207/4,0.907427
   poisson-64,scipy-out,7.229661,6.728774,6.979217,6.728774–7.229661,0.93072,201/207/4,201/207/4,0.907427
   poisson-64,torch-eager-out,6.944312,6.653569,6.798940,6.653569–6.944312,0.95813,201/207/4,201/207/4,0.907427
   poisson-128,native,45.513523,41.114100,43.313811,41.114100–45.513523,0.90334,627/641/12,627/641/12,0.989568
   poisson-128,scipy-out,78.160770,79.228008,78.694389,78.160770–79.228008,1.01365,627/641/12,627/641/12,0.989568
   poisson-128,torch-eager-out,56.558819,48.236694,52.397756,48.236694–56.558819,0.85286,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.047845,0.035664,0.041754,0.035664–0.047845,0.74539,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.049765,0.030783,0.040273,0.030783–0.049765,0.61856,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.289327,0.206425,0.247876,0.206425–0.289327,0.71346,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.468644,0.314779,0.391711,0.314779–0.468644,0.67168,49/51/0,49/51/0,0.635707
   dense-256,native,0.528804,0.477581,0.503193,0.477581–0.528804,0.90313,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.389044,0.342861,0.365953,0.342861–0.389044,0.88129,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.781128,0.687342,0.734235,0.687342–0.781128,0.87993,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.157508,1.006430,1.081969,1.006430–1.157508,0.86948,67/70/1,67/70/1,0.972357
   dense-1024,native,11.417768,9.387497,10.402632,9.387497–11.417768,0.82218,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,9.867431,8.487809,9.177620,8.487809–9.867431,0.86018,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,11.757089,10.167837,10.962463,10.167837–11.757089,0.86483,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,13.435463,11.549997,12.492730,11.549997–13.435463,0.85966,70/73/1,70/73/1,0.928014
   dense-2048,native,164.968861,107.662436,136.315649,107.662436–164.968861,0.65262,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,138.186431,84.154586,111.170509,84.154586–138.186431,0.60899,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,146.311026,89.846252,118.078639,89.846252–146.311026,0.61408,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,152.898857,93.227255,123.063056,93.227255–152.898857,0.60973,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.052136,0.054553,0.053344,0.052136–0.054553,1.04637,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.037596,0.046649,0.042123,0.037596–0.046649,1.24077,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.255474,0.288855,0.272164,0.255474–0.288855,1.13067,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.414225,0.473720,0.443973,0.414225–0.473720,1.14363,49/51/0,49/51/0,0.635707
   dense-256,native,0.526754,0.464294,0.495524,0.464294–0.526754,0.88143,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.380007,0.461967,0.420987,0.380007–0.461967,1.21568,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.736834,0.824732,0.780783,0.736834–0.824732,1.11929,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,1.047675,1.110575,1.079125,1.047675–1.110575,1.06004,67/70/1,67/70/1,0.972357
   dense-1024,native,11.374828,9.803777,10.589303,9.803777–11.374828,0.86188,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,10.340363,8.742738,9.541551,8.742738–10.340363,0.84550,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,12.231023,10.387963,11.309493,10.387963–12.231023,0.84931,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,13.769540,11.677738,12.723639,11.677738–13.769540,0.84808,70/73/1,70/73/1,0.928014
   dense-2048,native,172.656047,116.317183,144.486615,116.317183–172.656047,0.67369,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,141.448431,92.781802,117.115116,92.781802–141.448431,0.65594,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,149.178409,97.654590,123.416500,97.654590–149.178409,0.65462,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,155.537540,102.955473,129.246507,102.955473–155.537540,0.66193,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.038165,0.038717,0.038441,0.038165–0.038717,1.01444,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.034311,0.036173,0.035242,0.034311–0.036173,1.05428,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.268693,0.297223,0.282959,0.268693–0.297223,1.10618,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.432528,0.486619,0.459574,0.432528–0.486619,1.12506,49/51/0,49/51/0,0.635707
   dense-256,native,0.478583,0.480183,0.479383,0.478583–0.480183,1.00334,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.348215,0.454825,0.401520,0.348215–0.454825,1.30616,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.820505,0.687315,0.753911,0.687315–0.820505,0.83767,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.995193,0.911200,0.953197,0.911200–0.995193,0.91560,67/70/1,67/70/1,0.972357
   dense-1024,native,14.950404,13.834340,14.392373,13.834340–14.950404,0.92535,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,12.875426,12.475259,12.675343,12.475259–12.875426,0.96892,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,15.093210,13.886100,14.489655,13.886100–15.093210,0.92002,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,15.899546,15.638564,15.769055,15.638564–15.899546,0.98359,70/73/1,70/73/1,0.928014
   dense-2048,native,161.598892,147.354703,154.476798,147.354703–161.598892,0.91185,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,128.217840,109.530282,118.874061,109.530282–128.217840,0.85425,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,125.106196,88.271051,106.688623,88.271051–125.106196,0.70557,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,142.755481,119.113084,130.934282,119.113084–142.755481,0.83439,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.033356,0.052558,0.042957,0.033356–0.052558,1.57569,49/51/0,49/51/0,0.607749
   dense-64,native-blas,0.028772,0.046547,0.037660,0.028772–0.046547,1.61776,49/51/0,49/51/0,0.612054
   dense-64,numpy-out,0.226856,0.336828,0.281842,0.226856–0.336828,1.48476,49/51/0,49/51/0,0.61272
   dense-64,torch-eager-out,0.370632,0.594391,0.482511,0.370632–0.594391,1.60373,49/51/0,49/51/0,0.635707
   dense-256,native,0.456504,0.556252,0.506378,0.456504–0.556252,1.21850,67/70/1,67/70/1,0.971216
   dense-256,native-blas,0.296743,0.363063,0.329903,0.296743–0.363063,1.22350,67/70/1,67/70/1,0.979189
   dense-256,numpy-out,0.759151,0.957086,0.858119,0.759151–0.957086,1.26073,67/70/1,67/70/1,0.969964
   dense-256,torch-eager-out,0.837634,1.244927,1.041280,0.837634–1.244927,1.48624,67/70/1,67/70/1,0.972357
   dense-1024,native,11.899314,12.952129,12.425722,11.899314–12.952129,1.08848,70/73/1,70/73/1,0.928014
   dense-1024,native-blas,10.912426,11.975291,11.443858,10.912426–11.975291,1.09740,70/73/1,70/73/1,0.928014
   dense-1024,numpy-out,12.423796,12.211078,12.317437,12.211078–12.423796,0.98288,70/73/1,70/73/1,0.928014
   dense-1024,torch-eager-out,13.957047,13.866752,13.911900,13.866752–13.957047,0.99353,70/73/1,70/73/1,0.928014
   dense-2048,native,114.899253,143.809152,129.354202,114.899253–143.809152,1.25161,70/73/1,70/73/1,0.871762
   dense-2048,native-blas,91.075437,115.011927,103.043682,91.075437–115.011927,1.26282,70/73/1,70/73/1,0.871762
   dense-2048,numpy-out,93.703913,86.715043,90.209478,86.715043–93.703913,0.92542,70/73/1,70/73/1,0.871762
   dense-2048,torch-eager-out,99.183748,117.441098,108.312423,99.183748–117.441098,1.18408,70/73/1,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.017646,0.013355,0.015501,0.013355–0.017646,0.75686,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.025877,0.015130,0.020504,0.015130–0.025877,0.58472,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.080548,0.045523,0.063035,0.045523–0.080548,0.56517,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.155122,0.074734,0.114928,0.074734–0.155122,0.48178,5/7/0,5/7/0,0.0796191
   dense-256,native,0.071516,0.073370,0.072443,0.071516–0.073370,1.02591,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.068325,0.056414,0.062370,0.056414–0.068325,0.82568,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.124405,0.111528,0.117967,0.111528–0.124405,0.89649,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.185927,0.196160,0.191044,0.185927–0.196160,1.05504,4/6/0,4/6/0,0.209854
   dense-1024,native,0.994363,1.063040,1.028701,0.994363–1.063040,1.06907,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,0.987328,0.988691,0.988009,0.987328–0.988691,1.00138,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.151982,1.198027,1.175004,1.151982–1.198027,1.03997,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.446214,1.526860,1.486538,1.446214–1.526860,1.05576,4/6/0,4/6/0,0.013684
   dense-2048,native,9.161504,8.021797,8.591651,8.021797–9.161504,0.87560,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,8.346533,6.771986,7.559259,6.771986–8.346533,0.81135,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,8.638784,6.919213,7.778998,6.919213–8.638784,0.80095,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,9.115535,7.592543,8.354039,7.592543–9.115535,0.83292,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.016923,0.016057,0.016490,0.016057–0.016923,0.94880,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.020561,0.018127,0.019344,0.018127–0.020561,0.88160,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.067140,0.060976,0.064058,0.060976–0.067140,0.90818,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.121056,0.118166,0.119611,0.118166–0.121056,0.97613,5/7/0,5/7/0,0.0796191
   dense-256,native,0.055871,0.057530,0.056701,0.055871–0.057530,1.02968,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.054253,0.061865,0.058059,0.054253–0.061865,1.14033,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.096279,0.101916,0.099098,0.096279–0.101916,1.05855,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.158311,0.169340,0.163826,0.158311–0.169340,1.06966,4/6/0,4/6/0,0.209854
   dense-1024,native,0.958356,0.956309,0.957333,0.956309–0.958356,0.99786,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,0.934849,0.896965,0.915907,0.896965–0.934849,0.95948,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.043742,1.063552,1.053647,1.043742–1.063552,1.01898,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.353072,1.391025,1.372049,1.353072–1.391025,1.02805,4/6/0,4/6/0,0.013684
   dense-2048,native,7.614789,8.105517,7.860153,7.614789–8.105517,1.06444,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,5.673463,6.864556,6.269010,5.673463–6.864556,1.20994,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,6.290071,7.225416,6.757744,6.290071–7.225416,1.14870,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,6.543715,7.883497,7.213606,6.543715–7.883497,1.20474,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.007736,0.007105,0.007421,0.007105–0.007736,0.91856,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.008479,0.006707,0.007593,0.006707–0.008479,0.79101,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.042314,0.031752,0.037033,0.031752–0.042314,0.75038,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.078522,0.050543,0.064533,0.050543–0.078522,0.64369,5/7/0,5/7/0,0.0796191
   dense-256,native,0.040045,0.057048,0.048546,0.040045–0.057048,1.42458,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.031409,0.045639,0.038524,0.031409–0.045639,1.45305,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.063765,0.095916,0.079840,0.063765–0.095916,1.50421,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.087336,0.135789,0.111562,0.087336–0.135789,1.55478,4/6/0,4/6/0,0.209854
   dense-1024,native,1.456662,1.866816,1.661739,1.456662–1.866816,1.28157,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,1.350534,1.641051,1.495793,1.350534–1.641051,1.21511,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.404139,1.587309,1.495724,1.404139–1.587309,1.13045,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.807975,2.117674,1.962824,1.807975–2.117674,1.17130,4/6/0,4/6/0,0.013684
   dense-2048,native,7.395188,9.572193,8.483690,7.395188–9.572193,1.29438,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,6.501153,9.178027,7.839590,6.501153–9.178027,1.41175,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,6.499801,6.643121,6.571461,6.499801–6.643121,1.02205,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,6.924828,9.834055,8.379441,6.924828–9.834055,1.42012,3/5/0,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.007869,0.008282,0.008075,0.007869–0.008282,1.05248,5/7/0,5/7/0,0.0796191
   dense-64,native-blas,0.008012,0.008733,0.008372,0.008012–0.008733,1.08999,5/7/0,5/7/0,0.0796191
   dense-64,numpy-out,0.034245,0.049802,0.042024,0.034245–0.049802,1.45430,5/7/0,5/7/0,0.0796191
   dense-64,torch-eager-out,0.071297,0.083304,0.077300,0.071297–0.083304,1.16841,5/7/0,5/7/0,0.0796191
   dense-256,native,0.054619,0.042800,0.048710,0.042800–0.054619,0.78361,4/6/0,4/6/0,0.209854
   dense-256,native-blas,0.051535,0.043333,0.047434,0.043333–0.051535,0.84086,4/6/0,4/6/0,0.209854
   dense-256,numpy-out,0.077167,0.062476,0.069822,0.062476–0.077167,0.80963,4/6/0,4/6/0,0.209854
   dense-256,torch-eager-out,0.112276,0.092532,0.102404,0.092532–0.112276,0.82414,4/6/0,4/6/0,0.209854
   dense-1024,native,1.566780,1.902044,1.734412,1.566780–1.902044,1.21398,4/6/0,4/6/0,0.013684
   dense-1024,native-blas,1.353754,1.644030,1.498892,1.353754–1.644030,1.21442,4/6/0,4/6/0,0.013684
   dense-1024,numpy-out,1.684702,1.616276,1.650489,1.616276–1.684702,0.95938,4/6/0,4/6/0,0.013684
   dense-1024,torch-eager-out,1.857460,2.150910,2.004185,1.857460–2.150910,1.15798,4/6/0,4/6/0,0.013684
   dense-2048,native,10.879486,10.575709,10.727597,10.575709–10.879486,0.97208,3/5/0,3/5/0,0.836412
   dense-2048,native-blas,7.047077,9.153742,8.100409,7.047077–9.153742,1.29894,3/5/0,3/5/0,0.836412
   dense-2048,numpy-out,7.507474,6.831641,7.169557,6.831641–7.507474,0.90998,3/5/0,3/5/0,0.836412
   dense-2048,torch-eager-out,8.960299,9.321273,9.140786,8.960299–9.321273,1.04029,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,7.388080,7.539841,7.463960,7.388080–7.539841,1.02054,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,9.550935,9.815884,9.683410,9.550935–9.815884,1.02774,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,15.141148,16.736825,15.938987,15.141148–16.736825,1.10539,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,10.148627,10.543266,10.345947,10.148627–10.543266,1.03889,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,11.960757,12.175389,12.068073,11.960757–12.175389,1.01794,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,21.899011,21.839071,21.869041,21.839071–21.899011,0.99726,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,13.504428,12.831041,13.167735,12.831041–13.504428,0.95014,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,15.250528,13.639035,14.444781,13.639035–15.250528,0.89433,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,22.959388,21.031149,21.995269,21.031149–22.959388,0.91602,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,21.053797,21.923963,21.488880,21.053797–21.923963,1.04133,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,21.819496,21.879210,21.849353,21.819496–21.879210,1.00274,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,26.817182,28.401815,27.609498,26.817182–28.401815,1.05909,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,6.948031,8.946385,7.947209,6.948031–8.946385,1.28761,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,9.451536,11.149377,10.300457,9.451536–11.149377,1.17964,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,14.678539,17.521217,16.099878,14.678539–17.521217,1.19366,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,11.039473,11.923925,11.481699,11.039473–11.923925,1.08012,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,13.655520,13.585192,13.620356,13.585192–13.655520,0.99485,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,23.382777,23.840080,23.611428,23.382777–23.840080,1.01956,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,14.945037,14.368939,14.656988,14.368939–14.945037,0.96145,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,15.077550,15.120084,15.098817,15.077550–15.120084,1.00282,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,23.067629,22.110422,22.589025,22.110422–23.067629,0.95850,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,21.766157,21.890187,21.828172,21.766157–21.890187,1.00570,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,22.777109,22.297413,22.537261,22.297413–22.777109,0.97894,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.556719,27.746274,28.651497,27.746274–29.556719,0.93875,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,3.987453,5.112420,4.549937,3.987453–5.112420,1.28213,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,5.845959,7.240461,6.543210,5.845959–7.240461,1.23854,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,13.421046,15.807605,14.614325,13.421046–15.807605,1.17782,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,5.780264,6.797558,6.288911,5.780264–6.797558,1.17599,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,7.775747,8.367221,8.071484,7.775747–8.367221,1.07607,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,19.108040,21.163646,20.135843,19.108040–21.163646,1.10758,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,8.821112,9.919167,9.370140,8.821112–9.919167,1.12448,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,11.247437,11.411869,11.329653,11.247437–11.411869,1.01462,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,20.184331,21.682976,20.933653,20.184331–21.682976,1.07425,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,17.029689,19.768100,18.398894,17.029689–19.768100,1.16080,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,18.853548,20.857255,19.855401,18.853548–20.857255,1.10628,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.578105,31.111611,29.844859,28.578105–31.111611,1.08865,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.910508,5.628151,5.269330,4.910508–5.628151,1.14614,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,6.936994,8.123001,7.529998,6.936994–8.123001,1.17097,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,14.081965,16.221016,15.151490,14.081965–16.221016,1.15190,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,7.488791,7.543656,7.516223,7.488791–7.543656,1.00733,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,8.864387,9.447582,9.155985,8.864387–9.447582,1.06579,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,19.670103,21.467331,20.568717,19.670103–21.467331,1.09137,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,10.778197,10.389893,10.584045,10.389893–10.778197,0.96397,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,10.656004,11.210520,10.933262,10.656004–11.210520,1.05204,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.120302,21.104975,21.112638,21.104975–21.120302,0.99927,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,17.915996,18.991060,18.453528,17.915996–18.991060,1.06001,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,18.666763,19.516418,19.091590,18.666763–19.516418,1.04552,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.805473,28.427747,28.616610,28.427747–28.805473,0.98689,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,7.241627,8.203196,7.722411,7.241627–8.203196,1.13278,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,6.140199,8.614323,7.377261,6.140199–8.614323,1.40294,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,15.584111,18.591185,17.087648,15.584111–18.591185,1.19296,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,8.719755,11.634732,10.177243,8.719755–11.634732,1.33430,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,9.978030,10.910288,10.444159,9.978030–10.910288,1.09343,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.730290,23.343537,22.036914,20.730290–23.343537,1.12606,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,14.030023,13.306062,13.668042,13.306062–14.030023,0.94840,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,12.392817,12.596073,12.494445,12.392817–12.596073,1.01640,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,22.732283,22.232554,22.482418,22.232554–22.732283,0.97802,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,22.051188,21.462346,21.756767,21.462346–22.051188,0.97330,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,20.932686,20.290070,20.611378,20.290070–20.932686,0.96930,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,30.822573,28.160982,29.491778,28.160982–30.822573,0.91365,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,6.840152,8.749142,7.794647,6.840152–8.749142,1.27909,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,7.936485,8.496958,8.216722,7.936485–8.496958,1.07062,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,14.309202,16.033819,15.171511,14.309202–16.033819,1.12053,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,10.631960,12.229848,11.430904,10.631960–12.229848,1.15029,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,9.490118,11.658375,10.574246,9.490118–11.658375,1.22848,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,19.288610,22.331757,20.810184,19.288610–22.331757,1.15777,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,13.406925,14.555700,13.981312,13.406925–14.555700,1.08569,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,12.751259,13.106923,12.929091,12.751259–13.106923,1.02789,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.321700,22.248164,21.784932,21.321700–22.248164,1.04345,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,21.823242,23.583783,22.703513,21.823242–23.583783,1.08067,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,20.151145,21.601082,20.876113,20.151145–21.601082,1.07195,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.071414,30.525186,29.298300,28.071414–30.525186,1.08741,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.265998,5.048127,4.657063,4.265998–5.048127,1.18334,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,4.717384,5.114717,4.916051,4.717384–5.114717,1.08423,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,13.729828,16.371302,15.050565,13.729828–16.371302,1.19239,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,6.285208,7.384840,6.835024,6.285208–7.384840,1.17496,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,6.418367,6.802909,6.610638,6.418367–6.802909,1.05991,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,19.494487,20.847707,20.171097,19.494487–20.847707,1.06942,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,9.263727,9.812905,9.538316,9.263727–9.812905,1.05928,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,8.854515,9.233595,9.044055,8.854515–9.233595,1.04281,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,20.352436,21.306163,20.829300,20.352436–21.306163,1.04686,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,17.513602,18.270432,17.892017,17.513602–18.270432,1.04321,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,17.429637,17.611213,17.520425,17.429637–17.611213,1.01042,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.591409,28.775167,28.683288,28.591409–28.775167,1.00643,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,5.069512,5.998852,5.534182,5.069512–5.998852,1.18332,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,5.204862,6.134974,5.669918,5.204862–6.134974,1.17870,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,14.169868,17.434676,15.802272,14.169868–17.434676,1.23040,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,7.280553,8.011999,7.646276,7.280553–8.011999,1.10047,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,7.322037,7.773915,7.547976,7.322037–7.773915,1.06171,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.616244,22.521531,21.568887,20.616244–22.521531,1.09242,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,9.102507,10.990816,10.046662,9.102507–10.990816,1.20745,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,9.163472,10.041317,9.602395,9.163472–10.041317,1.09580,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,20.043273,22.794805,21.419039,20.043273–22.794805,1.13728,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,17.816853,18.333089,18.074971,17.816853–18.333089,1.02897,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,17.482836,17.153826,17.318331,17.153826–17.482836,0.98118,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.738614,28.405568,29.072091,28.405568–29.738614,0.95517,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,5.798606,7.958814,6.878710,5.798606–7.958814,1.37254,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,5.864696,8.151545,7.008121,5.864696–8.151545,1.38993,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,14.448935,16.742086,15.595510,14.448935–16.742086,1.15871,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,8.406523,11.492717,9.949620,8.406523–11.492717,1.36712,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,9.671088,11.068275,10.369682,9.671088–11.068275,1.14447,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,19.502623,23.715160,21.608892,19.502623–23.715160,1.21600,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,12.136881,14.353456,13.245168,12.136881–14.353456,1.18263,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,11.023504,13.384318,12.203911,11.023504–13.384318,1.21416,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,20.214645,23.343167,21.778907,20.214645–23.343167,1.15477,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,21.142829,22.386198,21.764514,21.142829–22.386198,1.05881,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,19.316028,21.168906,20.242466,19.316028–21.168906,1.09592,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.226523,29.852251,29.039387,28.226523–29.852251,1.05760,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,6.249249,8.344840,7.297044,6.249249–8.344840,1.33533,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,6.341066,8.340668,7.340867,6.341066–8.340668,1.31534,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,12.471711,16.797275,14.634493,12.471711–16.797275,1.34683,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,9.117527,11.654893,10.386210,9.117527–11.654893,1.27830,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,8.972304,11.117446,10.044875,8.972304–11.117446,1.23908,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,18.932217,21.523754,20.227985,18.932217–21.523754,1.13689,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,11.698420,14.552058,13.125239,11.698420–14.552058,1.24393,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,11.317683,13.455192,12.386437,11.317683–13.455192,1.18886,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,18.873849,22.483016,20.678432,18.873849–22.483016,1.19123,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,19.716413,22.464477,21.090445,19.716413–22.464477,1.13938,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,18.411963,21.244036,19.828000,18.411963–21.244036,1.15382,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,26.111364,29.229381,27.670373,26.111364–29.229381,1.11941,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.150162,5.249453,4.699807,4.150162–5.249453,1.26488,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,4.448444,5.430002,4.939223,4.448444–5.430002,1.22065,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,13.471479,16.668860,15.070170,13.471479–16.668860,1.23734,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,6.044571,7.107285,6.575928,6.044571–7.107285,1.17581,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,6.285266,7.703939,6.994602,6.285266–7.703939,1.22571,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.549149,22.528439,21.538794,20.549149–22.528439,1.09632,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,8.552240,10.031086,9.291663,8.552240–10.031086,1.17292,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,8.348430,9.552209,8.950319,8.348430–9.552209,1.14419,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,19.712516,22.374549,21.043533,19.712516–22.374549,1.13504,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,17.304814,18.545853,17.925334,17.304814–18.545853,1.07172,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,16.755182,16.688799,16.721991,16.688799–16.755182,0.99604,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.665460,30.046233,29.855847,29.665460–30.046233,1.01284,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.869879,5.682523,5.276200,4.869879–5.682523,1.16687,49/51/0,49/51/0,0.648887
   dense-64,cuda-cublas,5.293213,6.093776,5.693494,5.293213–6.093776,1.15124,49/51/0,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.574541,16.245954,16.410247,16.245954–16.574541,0.98018,49/51/0,49/51/0,0.611009
   dense-256,cuda-native,7.318039,8.337971,7.828005,7.318039–8.337971,1.13937,67/70/1,67/70/1,0.979365
   dense-256,cuda-cublas,7.480119,7.728248,7.604183,7.480119–7.728248,1.03317,67/70/1,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.358300,21.887217,21.122758,20.358300–21.887217,1.07510,67/70/1,67/70/1,0.977359
   dense-1024,cuda-native,9.409731,10.439578,9.924655,9.409731–10.439578,1.10944,70/73/1,70/73/1,0.928014
   dense-1024,cuda-cublas,9.092745,9.718826,9.405785,9.092745–9.718826,1.06886,70/73/1,70/73/1,0.928014
   dense-1024,torch-cuda-eager,19.935258,21.296287,20.615773,19.935258–21.296287,1.06827,70/73/1,70/73/1,0.928014
   dense-2048,cuda-native,18.019219,18.529881,18.274550,18.019219–18.529881,1.02834,70/73/1,70/73/1,0.871762
   dense-2048,cuda-cublas,16.423796,17.290662,16.857229,16.423796–17.290662,1.05278,70/73/1,70/73/1,0.871762
   dense-2048,torch-cuda-eager,26.937922,28.401789,27.669856,26.937922–28.401789,1.05434,70/73/1,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.157770,1.206902,1.182336,1.157770–1.206902,1.04244,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,3.620135,3.527865,3.574000,3.527865–3.620135,0.97451,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.606762,2.860660,2.733711,2.606762–2.860660,1.09740,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.927330,1.002968,0.965148,0.927330–1.002968,1.08157,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,2.924417,3.176375,3.050396,2.924417–3.176375,1.08616,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.307920,2.750697,2.529308,2.307920–2.750697,1.19185,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.239559,1.298792,1.269176,1.239559–1.298792,1.04779,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,2.881076,3.515081,3.198078,2.881076–3.515081,1.22006,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.381866,2.414382,2.398124,2.381866–2.414382,1.01365,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.440473,1.505507,1.472990,1.440473–1.505507,1.04515,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,3.265550,3.286215,3.275883,3.265550–3.286215,1.00633,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.165297,2.589960,2.377629,2.165297–2.589960,1.19612,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.146179,1.317341,1.231760,1.146179–1.317341,1.14933,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,2.885632,3.714368,3.300000,2.885632–3.714368,1.28719,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.508309,3.079566,2.793938,2.508309–3.079566,1.22775,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,1.016492,1.039505,1.027998,1.016492–1.039505,1.02264,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,2.744616,3.092487,2.918551,2.744616–3.092487,1.12675,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.943255,2.177286,2.060270,1.943255–2.177286,1.12043,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.336918,1.385344,1.361131,1.336918–1.385344,1.03622,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,3.051910,2.898442,2.975176,2.898442–3.051910,0.94971,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.106473,2.809006,2.457740,2.106473–2.809006,1.33351,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.498062,1.522241,1.510152,1.498062–1.522241,1.01614,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,3.086335,3.352606,3.219470,3.086335–3.352606,1.08627,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.090647,2.897438,2.494042,2.090647–2.897438,1.38591,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.759649,0.840352,0.800001,0.759649–0.840352,1.10624,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,2.699896,3.185157,2.942527,2.699896–3.185157,1.17973,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.538836,3.059722,2.799279,2.538836–3.059722,1.20517,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.745158,0.794312,0.769735,0.745158–0.794312,1.06596,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,2.673888,2.644614,2.659251,2.644614–2.673888,0.98905,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.071687,2.566958,2.319322,2.071687–2.566958,1.23907,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,0.991722,1.001945,0.996834,0.991722–1.001945,1.01031,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,2.692417,2.815854,2.754136,2.692417–2.815854,1.04585,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.198312,2.538268,2.368290,2.198312–2.538268,1.15464,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.295918,1.343188,1.319553,1.295918–1.343188,1.03648,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,2.961459,2.951133,2.956296,2.951133–2.961459,0.99651,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.406160,2.454378,2.430269,2.406160–2.454378,1.02004,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.984746,1.034511,1.009629,0.984746–1.034511,1.05054,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,3.008896,3.114397,3.061647,3.008896–3.114397,1.03506,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.331839,3.004330,2.668084,2.331839–3.004330,1.28840,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.874575,0.867614,0.871094,0.867614–0.874575,0.99204,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,2.648267,2.645456,2.646861,2.645456–2.648267,0.99894,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.094349,2.617510,2.355929,2.094349–2.617510,1.24980,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.143236,1.186316,1.164776,1.143236–1.186316,1.03768,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,2.921432,2.853996,2.887714,2.853996–2.921432,0.97692,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.317750,2.609209,2.463480,2.317750–2.609209,1.12575,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.429204,1.865522,1.647363,1.429204–1.865522,1.30529,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,2.958159,3.067349,3.012754,2.958159–3.067349,1.03691,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.192686,2.315142,2.253914,2.192686–2.315142,1.05585,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.124483,1.263607,1.194044,1.124483–1.263607,1.12372,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.172732,1.387525,1.280129,1.172732–1.387525,1.18316,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.585946,3.178784,2.882365,2.585946–3.178784,1.22925,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.977592,1.067706,1.022649,0.977592–1.067706,1.09218,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,1.000795,1.112863,1.056828,1.000795–1.112863,1.11198,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.093679,2.843080,2.468380,2.093679–2.843080,1.35793,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.326867,1.495416,1.411141,1.326867–1.495416,1.12703,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.223802,1.373082,1.298442,1.223802–1.373082,1.12198,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.220517,3.415272,2.817895,2.220517–3.415272,1.53805,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.649840,1.726847,1.688343,1.649840–1.726847,1.04668,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.460481,1.643984,1.552232,1.460481–1.643984,1.12564,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.351570,3.014181,2.682875,2.351570–3.014181,1.28177,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.351151,1.303483,1.327317,1.303483–1.351151,0.96472,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.378046,1.306140,1.342094,1.306140–1.378046,0.94782,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,3.315140,3.543055,3.429097,3.315140–3.543055,1.06875,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,1.214876,1.132788,1.173832,1.132788–1.214876,0.93243,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,1.128058,1.129540,1.128799,1.128058–1.129540,1.00131,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.559074,3.584125,3.071599,2.559074–3.584125,1.40056,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.561886,1.597176,1.579531,1.561886–1.597176,1.02259,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.409845,1.256375,1.333110,1.256375–1.409845,0.89114,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.655956,3.989781,3.322868,2.655956–3.989781,1.50220,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.851516,1.734572,1.793044,1.734572–1.851516,0.93684,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.721685,2.804892,2.263289,1.721685–2.804892,1.62915,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,3.173449,4.033178,3.603314,3.173449–4.033178,1.27091,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.845485,0.855754,0.850620,0.845485–0.855754,1.01215,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,0.914904,0.882532,0.898718,0.882532–0.914904,0.96462,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,3.070371,4.242339,3.656355,3.070371–4.242339,1.38170,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.849979,0.749111,0.799545,0.749111–0.849979,0.88133,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.824145,0.735233,0.779689,0.735233–0.824145,0.89212,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.505869,3.196061,2.850965,2.505869–3.196061,1.27543,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.089965,0.908963,0.999464,0.908963–1.089965,0.83394,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.020124,0.846159,0.933142,0.846159–1.020124,0.82947,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.857379,3.405731,3.131555,2.857379–3.405731,1.19191,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.436353,1.336902,1.386627,1.336902–1.436353,0.93076,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.272604,1.249688,1.261146,1.249688–1.272604,0.98199,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.475239,3.843485,3.159362,2.475239–3.843485,1.55277,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.042558,0.990086,1.016322,0.990086–1.042558,0.94967,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.035708,1.054074,1.044891,1.035708–1.054074,1.01773,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.603616,4.293239,3.448428,2.603616–4.293239,1.64895,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.945291,0.881071,0.913181,0.881071–0.945291,0.93206,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.966364,0.900671,0.933518,0.900671–0.966364,0.93202,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.459323,3.544773,3.002048,2.459323–3.544773,1.44136,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.246369,1.130578,1.188473,1.130578–1.246369,0.90710,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.176092,1.099299,1.137695,1.099299–1.176092,0.93470,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.319268,2.801404,2.560335,2.319268–2.801404,1.20788,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.660989,1.458502,1.559746,1.458502–1.660989,0.87809,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.488743,1.327604,1.408174,1.327604–1.488743,0.89176,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.516062,3.012647,2.764354,2.516062–3.012647,1.19737,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.100601,1.229872,1.165237,1.100601–1.229872,1.11746,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.158262,1.135809,1.147036,1.135809–1.158262,0.98062,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.556264,3.132474,2.844369,2.556264–3.132474,1.22541,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.951426,1.293198,1.122313,0.951426–1.293198,1.35922,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.988239,0.934577,0.961408,0.934577–0.988239,0.94570,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.928022,3.148795,2.538409,1.928022–3.148795,1.63317,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.345327,1.180954,1.263140,1.180954–1.345327,0.87782,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.189812,1.080414,1.135113,1.080414–1.189812,0.90805,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.093955,2.032537,2.063246,2.032537–2.093955,0.97067,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.622421,1.640928,1.631674,1.622421–1.640928,1.01141,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.499698,1.416413,1.458055,1.416413–1.499698,0.94447,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.190146,3.455147,2.822646,2.190146–3.455147,1.57759,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.208364,1.286409,1.247386,1.208364–1.286409,1.06459,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,1.246668,1.280185,1.263427,1.246668–1.280185,1.02689,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.575188,3.707629,3.141408,2.575188–3.707629,1.43975,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,1.073832,1.015043,1.044437,1.015043–1.073832,0.94525,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,1.020030,1.026629,1.023330,1.020030–1.026629,1.00647,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.987767,2.348249,2.168008,1.987767–2.348249,1.18135,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.335227,1.648956,1.492092,1.335227–1.648956,1.23496,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,1.212174,1.395776,1.303975,1.212174–1.395776,1.15146,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,1.888497,3.366410,2.627454,1.888497–3.366410,1.78259,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.637208,2.934849,2.286028,1.637208–2.934849,1.79259,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.473545,1.757233,1.615389,1.473545–1.757233,1.19252,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,1.968789,3.339254,2.654022,1.968789–3.339254,1.69609,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.700192,0.711565,0.705878,0.700192–0.711565,1.01624,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,0.739820,0.785729,0.762775,0.739820–0.785729,1.06205,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.053047,3.369981,2.711514,2.053047–3.369981,1.64145,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.629936,0.616184,0.623060,0.616184–0.629936,0.97817,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.679684,0.625491,0.652587,0.625491–0.679684,0.92027,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.709426,3.119823,2.414625,1.709426–3.119823,1.82507,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,0.916015,0.846924,0.881469,0.846924–0.916015,0.92457,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,0.802001,0.751275,0.776638,0.751275–0.802001,0.93675,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,1.739621,3.394673,2.567147,1.739621–3.394673,1.95139,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.315711,1.226647,1.271179,1.226647–1.315711,0.93231,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.157827,1.156524,1.157176,1.156524–1.157827,0.99888,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,1.775726,2.520533,2.148130,1.775726–2.520533,1.41944,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.963758,0.832795,0.898277,0.832795–0.963758,0.86411,5/7/0,5/7/0,0.0796191
   dense-64,cuda-cublas,0.954660,0.877113,0.915887,0.877113–0.954660,0.91877,5/7/0,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.306189,3.932992,3.119591,2.306189–3.932992,1.70541,5/7/0,5/7/0,0.0796191
   dense-256,cuda-native,0.849123,0.815497,0.832310,0.815497–0.849123,0.96040,4/6/0,4/6/0,0.209854
   dense-256,cuda-cublas,0.846487,0.777549,0.812018,0.777549–0.846487,0.91856,4/6/0,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.059194,2.435077,2.247136,2.059194–2.435077,1.18254,4/6/0,4/6/0,0.209854
   dense-1024,cuda-native,1.112873,1.025444,1.069158,1.025444–1.112873,0.92144,4/6/0,4/6/0,0.013684
   dense-1024,cuda-cublas,0.975430,0.933422,0.954426,0.933422–0.975430,0.95693,4/6/0,4/6/0,0.013684
   dense-1024,torch-cuda-eager,1.706629,2.732900,2.219765,1.706629–2.732900,1.60134,4/6/0,4/6/0,0.013684
   dense-2048,cuda-native,1.357654,1.346030,1.351842,1.346030–1.357654,0.99144,3/5/0,3/5/0,0.836412
   dense-2048,cuda-cublas,1.212064,1.468479,1.340271,1.212064–1.468479,1.21155,3/5/0,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.180013,2.509307,2.344660,2.180013–2.509307,1.15105,3/5/0,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.953743,4.717627,4.335685,3.953743–4.717627,1.19321,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.355117,10.348081,8.851599,7.355117–10.348081,1.40692,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,16.913905,20.837297,18.875601,16.913905–20.837297,1.23196,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,31.899965,43.347961,37.623963,31.899965–43.347961,1.35887,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,25.688656,34.220827,29.954742,25.688656–34.220827,1.33214,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,50.394022,73.328787,61.861404,50.394022–73.328787,1.45511,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,94.898618,110.348978,102.623798,94.898618–110.348978,1.16281,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,177.399956,232.904719,205.152337,177.399956–232.904719,1.31288,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,4.244235,4.621861,4.433048,4.244235–4.621861,1.08897,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.872612,8.022695,7.947654,7.872612–8.022695,1.01906,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,18.152743,19.140128,18.646436,18.152743–19.140128,1.05439,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,33.736421,35.961598,34.849009,33.736421–35.961598,1.06596,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,29.228119,33.258806,31.243462,29.228119–33.258806,1.13790,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,50.970216,60.898240,55.934228,50.970216–60.898240,1.19478,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,97.880626,101.488349,99.684488,97.880626–101.488349,1.03686,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,168.169475,182.547550,175.358512,168.169475–182.547550,1.08550,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.746244,3.081659,2.913952,2.746244–3.081659,1.12214,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.551004,8.053496,8.302250,8.053496–8.551004,0.94182,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,11.406969,11.678496,11.542732,11.406969–11.678496,1.02380,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,35.533563,34.430961,34.982262,34.430961–35.533563,0.96897,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,19.594797,20.419591,20.007194,19.594797–20.419591,1.04209,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,55.734679,59.334093,57.534386,55.734679–59.334093,1.06458,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,59.233131,58.357758,58.795445,58.357758–59.233131,0.98522,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,170.522150,178.278238,174.400194,170.522150–178.278238,1.04548,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.781807,3.807233,3.294520,2.781807–3.807233,1.36862,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.991185,9.002421,8.496803,7.991185–9.002421,1.12654,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,13.185258,13.932885,13.559072,13.185258–13.932885,1.05670,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,32.510807,34.634734,33.572770,32.510807–34.634734,1.06533,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,22.069748,21.387733,21.728741,21.387733–22.069748,0.96910,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,53.386456,56.340207,54.863332,53.386456–56.340207,1.05533,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,64.888992,71.115094,68.002043,64.888992–71.115094,1.09595,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,165.964238,177.738149,171.851193,165.964238–177.738149,1.07094,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.294605,4.273883,3.784244,3.294605–4.273883,1.29724,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,7.365749,8.682055,8.023902,7.365749–8.682055,1.17871,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,15.529962,18.812184,17.171073,15.529962–18.812184,1.21135,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,30.731232,36.712390,33.721811,30.731232–36.712390,1.19463,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,29.737540,32.205709,30.971624,29.737540–32.205709,1.08300,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,60.325650,63.188332,61.756991,60.325650–63.188332,1.04745,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,90.492998,95.864689,93.178843,90.492998–95.864689,1.05936,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,184.162988,187.776252,185.969620,184.162988–187.776252,1.01962,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.216565,4.646031,3.931298,3.216565–4.646031,1.44441,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,6.016868,8.051934,7.034401,6.016868–8.051934,1.33823,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,16.567733,20.368270,18.468001,16.567733–20.368270,1.22939,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,28.413035,35.002530,31.707782,28.413035–35.002530,1.23192,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,30.563132,33.844918,32.204025,30.563132–33.844918,1.10738,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,58.021315,59.019449,58.520381,58.021315–59.019449,1.01720,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,98.006164,98.418248,98.212206,98.006164–98.418248,1.00420,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,176.159762,177.530202,176.844982,176.159762–177.530202,1.00778,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.229884,3.161645,2.695764,2.229884–3.161645,1.41785,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,6.274695,8.693394,7.484045,6.274695–8.693394,1.38547,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,9.609951,12.174289,10.892120,9.609951–12.174289,1.26684,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,30.830132,41.018459,35.924295,30.830132–41.018459,1.33047,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,20.370172,22.673516,21.521844,20.370172–22.673516,1.11307,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,57.848404,69.496796,63.672600,57.848404–69.496796,1.20136,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,56.628636,63.274212,59.951424,56.628636–63.274212,1.11735,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,163.070205,195.047746,179.058976,163.070205–195.047746,1.19610,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.887554,3.858990,3.373273,2.887554–3.858990,1.33642,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,6.995171,9.676348,8.335759,6.995171–9.676348,1.38329,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,12.904003,15.914087,14.409045,12.904003–15.914087,1.23327,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,32.707009,41.356045,37.031527,32.707009–41.356045,1.26444,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,22.608746,24.336211,23.472479,22.608746–24.336211,1.07641,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,54.821064,63.090511,58.955787,54.821064–63.090511,1.15084,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,62.779173,85.011376,73.895275,62.779173–85.011376,1.35413,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,160.153200,216.942203,188.547701,160.153200–216.942203,1.35459,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.584310,3.568328,3.576319,3.568328–3.584310,0.99554,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.749085,8.439835,8.594460,8.439835–8.749085,0.96465,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,17.305524,17.319091,17.312307,17.305524–17.319091,1.00078,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,34.791559,32.760638,33.776099,32.760638–34.791559,0.94163,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,26.222099,29.567733,27.894916,26.222099–29.567733,1.12759,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,52.335146,56.737377,54.536262,52.335146–56.737377,1.08412,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,91.015559,104.978035,97.996797,91.015559–104.978035,1.15341,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,180.596931,213.472688,197.034809,180.596931–213.472688,1.18204,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.764217,4.204262,3.984239,3.764217–4.204262,1.11690,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.015558,7.537138,7.776348,7.537138–8.015558,0.94031,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,19.228953,18.897620,19.063286,18.897620–19.228953,0.98277,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,33.599140,33.244187,33.421664,33.244187–33.599140,0.98944,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,29.169411,31.173750,30.171580,29.169411–31.173750,1.06871,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,50.858056,55.277383,53.067719,50.858056–55.277383,1.08690,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,95.524874,99.062365,97.293620,95.524874–99.062365,1.03703,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,169.280975,176.818739,173.049857,169.280975–176.818739,1.04453,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.554952,3.491200,3.023076,2.554952–3.491200,1.36644,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,6.634130,10.967757,8.800943,6.634130–10.967757,1.65323,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,12.536847,12.700103,12.618475,12.536847–12.700103,1.01302,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,34.718755,43.348422,39.033588,34.718755–43.348422,1.24856,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,20.915987,21.325647,21.120817,20.915987–21.325647,1.01959,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,66.227778,67.798339,67.013059,66.227778–67.798339,1.02371,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,58.131022,66.314677,62.222849,58.131022–66.314677,1.14078,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,179.162864,195.428486,187.295675,179.162864–195.428486,1.09079,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.770839,4.090912,3.430875,2.770839–4.090912,1.47642,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.105996,9.519314,8.812655,8.105996–9.519314,1.17435,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,14.651304,13.426947,14.039126,13.426947–14.651304,0.91643,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,36.823327,36.787711,36.805519,36.787711–36.823327,0.99903,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,20.508901,21.126666,20.817783,20.508901–21.126666,1.03012,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,55.073205,56.346044,55.709625,55.073205–56.346044,1.02311,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,66.140225,76.657533,71.398878,66.140225–76.657533,1.15902,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,179.366487,202.587875,190.977181,179.366487–202.587875,1.12946,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,6.035025,6.562690,6.298857,6.035025–6.562690,1.08743,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,12.011012,13.388789,12.699901,12.011012–13.388789,1.11471,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,27.466591,28.715248,28.090919,27.466591–28.715248,1.04546,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,55.954591,54.059474,55.007032,54.059474–55.954591,0.96613,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,42.811315,47.084229,44.947772,42.811315–47.084229,1.09981,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,82.667100,88.067149,85.367125,82.667100–88.067149,1.06532,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,141.578713,143.568547,142.573630,141.578713–143.568547,1.01405,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,272.012573,268.529765,270.271169,268.529765–272.012573,0.98720,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.787070,7.655898,6.721484,5.787070–7.655898,1.32293,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,11.705145,13.846998,12.776071,11.705145–13.846998,1.18298,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,29.412704,31.246613,30.329659,29.412704–31.246613,1.06235,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,53.340626,54.200255,53.770440,53.340626–54.200255,1.01612,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,46.100887,50.652257,48.376572,46.100887–50.652257,1.09873,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,80.278444,87.073440,83.675942,80.278444–87.073440,1.08464,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,139.592426,143.426254,141.509340,139.592426–143.426254,1.02746,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,254.771085,253.280522,254.025804,253.280522–254.771085,0.99415,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,4.947540,3.532037,4.239789,3.532037–4.947540,0.71390,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,14.779913,12.112031,13.445972,12.112031–14.779913,0.81949,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,14.124313,13.699250,13.911781,13.699250–14.124313,0.96991,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,56.257556,49.576687,52.917122,49.576687–56.257556,0.88124,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,22.060816,22.379646,22.220231,22.060816–22.379646,1.01445,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,84.186624,84.622657,84.404640,84.186624–84.622657,1.00518,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,66.606074,64.798519,65.702296,64.798519–66.606074,0.97286,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,251.230685,251.391302,251.310994,251.230685–251.391302,1.00064,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.346497,4.681710,5.014104,4.681710–5.346497,0.87566,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,11.738956,11.522951,11.630954,11.522951–11.738956,0.98160,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,18.721540,19.300241,19.010891,18.721540–19.300241,1.03091,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,47.962210,49.322097,48.642154,47.962210–49.322097,1.02835,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,32.477410,31.169103,31.823257,31.169103–32.477410,0.95972,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,85.151085,82.534400,83.842742,82.534400–85.151085,0.96927,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,89.690601,96.727630,93.209115,89.690601–96.727630,1.07846,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,247.188594,255.565050,251.376822,247.188594–255.565050,1.03389,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.280696,6.282028,5.781362,5.280696–6.282028,1.18962,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.457569,12.563257,11.510413,10.457569–12.563257,1.20136,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,24.510977,28.044778,26.277877,24.510977–28.044778,1.14417,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,46.531264,53.072088,49.801676,46.531264–53.072088,1.14057,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,44.206047,46.665703,45.435875,44.206047–46.665703,1.05564,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,88.448246,88.831050,88.639648,88.448246–88.831050,1.00433,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,139.878559,142.400483,141.139521,139.878559–142.400483,1.01803,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,268.676754,267.662949,268.169851,267.662949–268.676754,0.99623,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.472203,5.820461,5.646332,5.472203–5.820461,1.06364,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,11.245523,10.586750,10.916137,10.586750–11.245523,0.94142,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,26.350228,27.126385,26.738306,26.350228–27.126385,1.02946,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,45.614617,47.658152,46.636384,45.614617–47.658152,1.04480,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,49.163685,47.164921,48.164303,47.164921–49.163685,0.95934,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,88.033161,86.147953,87.090557,86.147953–88.033161,0.97859,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,146.091064,135.413295,140.752179,135.413295–146.091064,0.92691,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,261.315197,250.598789,255.956993,250.598789–261.315197,0.95899,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.828651,3.405320,3.116985,2.828651–3.405320,1.20387,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.240820,12.419360,11.330090,10.240820–12.419360,1.21273,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,12.967400,13.771199,13.369300,12.967400–13.771199,1.06199,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,46.813300,51.142533,48.977916,46.813300–51.142533,1.09248,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,23.628617,23.486203,23.557410,23.486203–23.628617,0.99397,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,83.214690,91.172123,87.193406,83.214690–91.172123,1.09563,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,65.979955,62.335443,64.157699,62.335443–65.979955,0.94476,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,255.476547,233.824466,244.650506,233.824466–255.476547,0.91525,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.233687,5.051183,5.142435,5.051183–5.233687,0.96513,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.596666,13.184437,11.890552,10.596666–13.184437,1.24421,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,18.854922,20.088579,19.471750,18.854922–20.088579,1.06543,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,50.810088,55.564001,53.187044,50.810088–55.564001,1.09356,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,29.843491,32.086285,30.964888,29.843491–32.086285,1.07515,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,83.056647,84.061596,83.559122,83.056647–84.061596,1.01210,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,85.757809,96.430026,91.093918,85.757809–96.430026,1.12445,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,241.649061,249.169315,245.409188,241.649061–249.169315,1.03112,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,6.082927,6.335427,6.209177,6.082927–6.335427,1.04151,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,10.996603,12.648387,11.822495,10.996603–12.648387,1.15021,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,23.852525,28.832306,26.342416,23.852525–28.832306,1.20877,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,45.244993,55.609572,50.427282,45.244993–55.609572,1.22908,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,42.337661,47.919774,45.128718,42.337661–47.919774,1.13185,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,84.630370,89.535370,87.082870,84.630370–89.535370,1.05796,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,131.171696,143.354480,137.263089,131.171696–143.354480,1.09288,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,258.471665,268.862844,263.667255,258.471665–268.862844,1.04020,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,7.040244,6.163361,6.601803,6.163361–7.040244,0.87545,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,12.089534,12.479989,12.284762,12.089534–12.479989,1.03230,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,28.359338,30.373946,29.366642,28.359338–30.373946,1.07104,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,49.788010,57.235532,53.511771,49.788010–57.235532,1.14958,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,46.451634,47.616325,47.033980,46.451634–47.616325,1.02507,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,81.125938,86.340947,83.733442,81.125938–86.340947,1.06428,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,136.017871,140.279597,138.148734,136.017871–140.279597,1.03133,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,250.928659,252.543195,251.735927,250.928659–252.543195,1.00643,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.819551,2.698812,2.759181,2.698812–2.819551,0.95718,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,11.021698,10.051138,10.536418,10.051138–11.021698,0.91194,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,13.824635,13.143394,13.484014,13.143394–13.824635,0.95072,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,53.407184,46.030048,49.718616,46.030048–53.407184,0.86187,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,23.870569,23.209375,23.539972,23.209375–23.870569,0.97230,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,86.487411,86.747124,86.617267,86.487411–86.747124,1.00300,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,62.574328,77.347886,69.961106,62.574328–77.347886,1.23610,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,229.047147,292.253636,260.650392,229.047147–292.253636,1.27595,627/641/12,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.755644,4.208307,3.981976,3.755644–4.208307,1.12053,26/28/0,26/28/0,0.579621
   poisson-8,torch-cuda-eager,12.615665,10.113402,11.364534,10.113402–12.615665,0.80165,26/28/0,26/28/0,0.579621
   poisson-32,cuda-native,17.715136,18.926738,18.320937,17.715136–18.926738,1.06839,118/122/2,118/122/2,0.840189
   poisson-32,torch-cuda-eager,46.858942,45.112741,45.985841,45.112741–46.858942,0.96273,118/122/2,118/122/2,0.840189
   poisson-64,cuda-native,31.788029,32.037858,31.912944,31.788029–32.037858,1.00786,201/207/4,201/207/4,0.907427
   poisson-64,torch-cuda-eager,79.105656,77.874949,78.490303,77.874949–79.105656,0.98444,201/207/4,201/207/4,0.907427
   poisson-128,cuda-native,88.603889,87.894880,88.249385,87.894880–88.603889,0.99200,627/641/12,627/641/12,0.989568
   poisson-128,torch-cuda-eager,246.151021,229.086099,237.618560,229.086099–246.151021,0.93067,627/641/12,627/641/12,0.989568

.. csv-table:: cpu-release-sparse-solvers-none-api-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   scaled-poisson-2d-16,native-cg,0.540315,0.509646,0.524980,0.509646–0.540315,0.94324,597/610/11,597/610/11,0.970433
   scaled-poisson-2d-16,native-pcg,0.090324,0.078952,0.084638,0.078952–0.090324,0.87410,48/50/0,48/50/0,0.91593
   scaled-poisson-2d-16,scipy-cg,3.452895,2.817445,3.135169,2.817445–3.452895,0.81597,276/277/unknown,276/277/unknown,0.846418
   scaled-poisson-2d-16,scipy-pcg,0.779270,0.597607,0.688439,0.597607–0.779270,0.76688,48/49/unknown,48/49/unknown,0.91593
   scaled-poisson-2d-16,scipy-lu-total,0.464093,0.376990,0.420542,0.376990–0.464093,0.81232,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-16,scipy-lu-reuse,0.048341,0.040738,0.044539,0.040738–0.048341,0.84272,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-32,native-cg,5.357972,5.110451,5.234211,5.110451–5.357972,0.95380,1799/1836/35,1799/1836/35,0.976976
   scaled-poisson-2d-32,native-pcg,0.490270,0.465460,0.477865,0.465460–0.490270,0.94940,118/122/2,118/122/2,0.909328
   scaled-poisson-2d-32,scipy-cg,9.757430,9.161723,9.459577,9.161723–9.757430,0.93895,633/634/unknown,633/634/unknown,0.899425
   scaled-poisson-2d-32,scipy-pcg,1.792880,1.564355,1.678617,1.564355–1.792880,0.87254,90/91/unknown,90/91/unknown,0.843882
   scaled-poisson-2d-32,scipy-lu-total,1.798096,1.625509,1.711803,1.625509–1.798096,0.90402,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-2d-32,scipy-lu-reuse,0.159632,0.142963,0.151298,0.142963–0.159632,0.89558,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-3d-6,native-cg,0.230403,0.227951,0.229177,0.227951–0.230403,0.98936,253/260/5,253/260/5,0.876984
   scaled-poisson-3d-6,native-pcg,0.058702,0.053323,0.056012,0.053323–0.058702,0.90835,24/26/0,24/26/0,0.71185
   scaled-poisson-3d-6,scipy-cg,2.287034,1.939069,2.113052,1.939069–2.287034,0.84785,180/181/unknown,180/181/unknown,0.837854
   scaled-poisson-3d-6,scipy-pcg,0.433889,0.355990,0.394940,0.355990–0.433889,0.82046,24/25/unknown,24/25/unknown,0.71185
   scaled-poisson-3d-6,scipy-lu-total,0.467514,0.446949,0.457232,0.446949–0.467514,0.95601,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-6,scipy-lu-reuse,0.042679,0.043706,0.043192,0.042679–0.043706,1.02408,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-10,native-cg,1.465734,1.385318,1.425526,1.385318–1.465734,0.94514,387/396/7,387/396/7,0.977579
   scaled-poisson-3d-10,native-pcg,0.237123,0.219336,0.228230,0.219336–0.237123,0.92499,37/39/0,37/39/0,0.956242
   scaled-poisson-3d-10,scipy-cg,4.832266,4.346442,4.589354,4.346442–4.832266,0.89946,296/297/unknown,296/297/unknown,0.995554
   scaled-poisson-3d-10,scipy-pcg,0.825701,0.715741,0.770721,0.715741–0.825701,0.86683,37/38/unknown,37/38/unknown,0.956242
   scaled-poisson-3d-10,scipy-lu-total,3.852787,3.749750,3.801268,3.749750–3.852787,0.97326,0/1/unknown,0/1/unknown,3.15518e-08
   scaled-poisson-3d-10,scipy-lu-reuse,0.254467,0.223181,0.238824,0.223181–0.254467,0.87705,0/1/unknown,0/1/unknown,3.15518e-08
   poisson-2d-16-neutral,native-cg,0.068027,0.064299,0.066163,0.064299–0.068027,0.94521,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,native-pcg,0.079220,0.074203,0.076712,0.074203–0.079220,0.93667,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,scipy-cg,0.587584,0.548385,0.567985,0.548385–0.587584,0.93329,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-pcg,0.638450,0.608937,0.623694,0.608937–0.638450,0.95377,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-lu-total,0.328688,0.319582,0.324135,0.319582–0.328688,0.97229,0/1/unknown,0/1/unknown,1.84387e-08
   poisson-2d-16-neutral,scipy-lu-reuse,0.030441,0.027878,0.029160,0.027878–0.030441,0.91580,0/1/unknown,0/1/unknown,1.84387e-08

.. csv-table:: cpu-release-sparse-solvers-none-prepared-fused1-host-standard
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   scaled-poisson-2d-16,native-cg,0.527486,0.541096,0.534291,0.527486–0.541096,1.02580,597/610/11,597/610/11,0.970433
   scaled-poisson-2d-16,native-pcg,0.063289,0.065503,0.064396,0.063289–0.065503,1.03497,48/50/0,48/50/0,0.91593
   scaled-poisson-2d-16,scipy-cg,3.037864,3.567491,3.302678,3.037864–3.567491,1.17434,276/277/unknown,276/277/unknown,0.846418
   scaled-poisson-2d-16,scipy-pcg,0.620471,0.711698,0.666085,0.620471–0.711698,1.14703,48/49/unknown,48/49/unknown,0.91593
   scaled-poisson-2d-16,scipy-lu-total,0.404210,0.417321,0.410766,0.404210–0.417321,1.03243,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-16,scipy-lu-reuse,0.044775,0.037950,0.041362,0.037950–0.044775,0.84758,0/1/unknown,0/1/unknown,1.56573e-08
   scaled-poisson-2d-32,native-cg,5.252187,5.480167,5.366177,5.252187–5.480167,1.04341,1799/1836/35,1799/1836/35,0.976976
   scaled-poisson-2d-32,native-pcg,0.430709,0.451020,0.440864,0.430709–0.451020,1.04716,118/122/2,118/122/2,0.909328
   scaled-poisson-2d-32,scipy-cg,9.515899,10.163721,9.839810,9.515899–10.163721,1.06808,633/634/unknown,633/634/unknown,0.899425
   scaled-poisson-2d-32,scipy-pcg,1.548851,1.863441,1.706146,1.548851–1.863441,1.20311,90/91/unknown,90/91/unknown,0.843882
   scaled-poisson-2d-32,scipy-lu-total,1.722085,1.795265,1.758676,1.722085–1.795265,1.04249,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-2d-32,scipy-lu-reuse,0.140542,0.152098,0.146320,0.140542–0.152098,1.08222,0/1/unknown,0/1/unknown,2.13506e-08
   scaled-poisson-3d-6,native-cg,0.228518,0.209127,0.218823,0.209127–0.228518,0.91515,253/260/5,253/260/5,0.876984
   scaled-poisson-3d-6,native-pcg,0.041787,0.033378,0.037583,0.033378–0.041787,0.79877,24/26/0,24/26/0,0.71185
   scaled-poisson-3d-6,scipy-cg,2.205907,2.062259,2.134083,2.062259–2.205907,0.93488,180/181/unknown,180/181/unknown,0.837854
   scaled-poisson-3d-6,scipy-pcg,0.396861,0.337989,0.367425,0.337989–0.396861,0.85166,24/25/unknown,24/25/unknown,0.71185
   scaled-poisson-3d-6,scipy-lu-total,0.480862,0.447151,0.464007,0.447151–0.480862,0.92990,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-6,scipy-lu-reuse,0.042703,0.037483,0.040093,0.037483–0.042703,0.87775,0/1/unknown,0/1/unknown,1.87121e-08
   scaled-poisson-3d-10,native-cg,1.390527,1.444487,1.417507,1.390527–1.444487,1.03881,387/396/7,387/396/7,0.977579
   scaled-poisson-3d-10,native-pcg,0.198955,0.192011,0.195483,0.192011–0.198955,0.96510,37/39/0,37/39/0,0.956242
   scaled-poisson-3d-10,scipy-cg,4.399309,5.144191,4.771750,4.399309–5.144191,1.16932,296/297/unknown,296/297/unknown,0.995554
   scaled-poisson-3d-10,scipy-pcg,0.707151,0.821299,0.764225,0.707151–0.821299,1.16142,37/38/unknown,37/38/unknown,0.956242
   scaled-poisson-3d-10,scipy-lu-total,4.068103,3.912151,3.990127,3.912151–4.068103,0.96166,0/1/unknown,0/1/unknown,3.15518e-08
   scaled-poisson-3d-10,scipy-lu-reuse,0.268369,0.222356,0.245362,0.222356–0.268369,0.82855,0/1/unknown,0/1/unknown,3.15518e-08
   poisson-2d-16-neutral,native-cg,0.058151,0.055947,0.057049,0.055947–0.058151,0.96210,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,native-pcg,0.068784,0.064393,0.066589,0.064393–0.068784,0.93616,51/54/1,51/54/1,0.602546
   poisson-2d-16-neutral,scipy-cg,0.650528,0.661938,0.656233,0.650528–0.661938,1.01754,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-pcg,0.684202,0.741460,0.712831,0.684202–0.741460,1.08369,51/52/unknown,51/52/unknown,0.568679
   poisson-2d-16-neutral,scipy-lu-total,0.350553,0.334319,0.342436,0.334319–0.350553,0.95369,0/1/unknown,0/1/unknown,1.84387e-08
   poisson-2d-16-neutral,scipy-lu-reuse,0.041790,0.032015,0.036902,0.032015–0.041790,0.76610,0/1/unknown,0/1/unknown,1.84387e-08

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.075526,0.076386,0.075956,0.075526–0.076386,1.01139,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.355329,0.393456,0.374393,0.355329–0.393456,1.10730,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.572473,0.700849,0.636661,0.572473–0.700849,1.22425,49/99/0,49/99/0,0.794401
   dense-256,native,0.854576,0.933584,0.894080,0.854576–0.933584,1.09245,65/131/0,65/131/0,0.872577
   dense-256,numpy-out,1.288893,1.170424,1.229658,1.170424–1.288893,0.90808,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.740414,1.744859,1.742636,1.740414–1.744859,1.00255,65/131/0,65/131/0,0.87147
   dense-1024,native,18.200619,18.727834,18.464226,18.200619–18.727834,1.02897,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,18.962416,19.257212,19.109814,18.962416–19.257212,1.01555,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,21.626458,21.198346,21.412402,21.198346–21.626458,0.98020,69/139/0,69/139/0,0.857221
   dense-2048,native,265.156690,176.340029,220.748359,176.340029–265.156690,0.66504,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,214.930511,140.652107,177.791309,140.652107–214.930511,0.65441,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,222.062926,147.412578,184.737752,147.412578–222.062926,0.66383,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.076808,0.086728,0.081768,0.076808–0.086728,1.12915,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.363799,0.405274,0.384536,0.363799–0.405274,1.11401,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.598097,0.733355,0.665726,0.598097–0.733355,1.22615,49/99/0,49/99/0,0.794401
   dense-256,native,0.823718,0.940097,0.881907,0.823718–0.940097,1.14128,65/131/0,65/131/0,0.872577
   dense-256,numpy-out,1.153471,1.179446,1.166458,1.153471–1.179446,1.02252,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.579496,1.730778,1.655137,1.579496–1.730778,1.09578,65/131/0,65/131/0,0.87147
   dense-1024,native,17.626695,17.833135,17.729915,17.626695–17.833135,1.01171,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,18.206387,18.357799,18.282093,18.206387–18.357799,1.00832,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,20.672598,20.897218,20.784908,20.672598–20.897218,1.01087,69/139/0,69/139/0,0.857221
   dense-2048,native,238.225234,187.436127,212.830680,187.436127–238.225234,0.78680,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,201.765538,157.789767,179.777652,157.789767–201.765538,0.78205,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,206.126297,161.241090,183.683694,161.241090–206.126297,0.78224,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.063572,0.063107,0.063340,0.063107–0.063572,0.99269,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.407278,0.344618,0.375948,0.344618–0.407278,0.84615,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.745959,0.624228,0.685093,0.624228–0.745959,0.83681,49/99/0,49/99/0,0.794401
   dense-256,native,0.899914,0.813919,0.856917,0.813919–0.899914,0.90444,65/131/0,65/131/0,0.872577
   dense-256,numpy-out,1.215186,0.992349,1.103768,0.992349–1.215186,0.81662,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.707940,1.407105,1.557523,1.407105–1.707940,0.82386,65/131/0,65/131/0,0.87147
   dense-1024,native,23.129680,22.385730,22.757705,22.385730–23.129680,0.96784,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,22.677536,22.412682,22.545109,22.412682–22.677536,0.98832,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,25.659066,25.170625,25.414846,25.170625–25.659066,0.98096,69/139/0,69/139/0,0.857221
   dense-2048,native,288.415259,275.239150,281.827204,275.239150–288.415259,0.95432,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,208.238326,163.781269,186.009797,163.781269–208.238326,0.78651,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,256.666399,233.128121,244.897260,233.128121–256.666399,0.90829,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.069525,0.072135,0.070830,0.069525–0.072135,1.03753,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.454870,0.380735,0.417803,0.380735–0.454870,0.83702,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.753866,0.666941,0.710404,0.666941–0.753866,0.88470,49/99/0,49/99/0,0.794401
   dense-256,native,0.967390,0.925697,0.946543,0.925697–0.967390,0.95690,65/131/0,65/131/0,0.872577
   dense-256,numpy-out,1.276153,1.144662,1.210408,1.144662–1.276153,0.89696,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.815807,1.687615,1.751711,1.687615–1.815807,0.92940,65/131/0,65/131/0,0.87147
   dense-1024,native,25.146526,23.275253,24.210890,23.275253–25.146526,0.92559,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,25.341439,21.586662,23.464051,21.586662–25.341439,0.85183,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,27.263365,25.032059,26.147712,25.032059–27.263365,0.91816,69/139/0,69/139/0,0.857221
   dense-2048,native,312.613569,271.085460,291.849515,271.085460–312.613569,0.86716,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,188.092073,165.303695,176.697884,165.303695–188.092073,0.87884,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,257.133709,226.047132,241.590420,226.047132–257.133709,0.87910,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.025888,0.037793,0.031841,0.025888–0.037793,1.45987,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.282025,0.326251,0.304138,0.282025–0.326251,1.15681,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.384593,0.447713,0.416152,0.384593–0.447713,1.16412,26/53/0,26/53/0,0.579621
   poisson-32,native,0.571566,0.595373,0.583470,0.571566–0.595373,1.04165,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.814388,1.872279,1.843333,1.814388–1.872279,1.03191,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,1.807854,1.926121,1.866988,1.807854–1.926121,1.06542,92/185/0,92/185/0,0.838111
   poisson-64,native,4.030408,3.723670,3.877039,3.723670–4.030408,0.92389,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,7.887348,7.252446,7.569897,7.252446–7.887348,0.91950,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,6.375985,5.799221,6.087603,5.799221–6.375985,0.90954,174/349/0,174/349/0,0.912222
   poisson-128,native,28.876676,29.077000,28.976838,28.876676–29.077000,1.00694,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,47.889901,47.669632,47.779767,47.669632–47.889901,0.99540,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,37.182795,39.554601,38.368698,37.182795–39.554601,1.06379,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.034314,0.029652,0.031983,0.029652–0.034314,0.86415,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.278006,0.288453,0.283230,0.278006–0.288453,1.03758,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.374628,0.419604,0.397116,0.374628–0.419604,1.12006,26/53/0,26/53/0,0.579621
   poisson-32,native,0.630104,0.604568,0.617336,0.604568–0.630104,0.95947,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.752155,1.674101,1.713128,1.674101–1.752155,0.95545,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,1.692020,1.663523,1.677771,1.663523–1.692020,0.98316,92/185/0,92/185/0,0.838111
   poisson-64,native,4.455230,4.509729,4.482480,4.455230–4.509729,1.01223,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,7.873838,8.135143,8.004491,7.873838–8.135143,1.03319,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,6.145550,6.486226,6.315888,6.145550–6.486226,1.05543,174/349/0,174/349/0,0.912222
   poisson-128,native,35.073333,34.235278,34.654306,34.235278–35.073333,0.97611,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,50.696345,50.212910,50.454627,50.212910–50.696345,0.99046,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,38.566743,39.043943,38.805343,38.566743–39.043943,1.01237,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.018556,0.021101,0.019829,0.018556–0.021101,1.13718,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.271947,0.309154,0.290550,0.271947–0.309154,1.13682,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.396383,0.408562,0.402472,0.396383–0.408562,1.03073,26/53/0,26/53/0,0.579621
   poisson-32,native,0.562887,0.569357,0.566122,0.562887–0.569357,1.01149,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,2.098988,1.845131,1.972060,1.845131–2.098988,0.87906,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,2.012640,1.882794,1.947717,1.882794–2.012640,0.93548,92/185/0,92/185/0,0.838111
   poisson-64,native,3.843795,3.932588,3.888192,3.843795–3.932588,1.02310,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,7.670508,8.414089,8.042298,7.670508–8.414089,1.09694,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,6.797152,6.871657,6.834404,6.797152–6.871657,1.01096,174/349/0,174/349/0,0.912222
   poisson-128,native,32.556805,28.066398,30.311602,28.066398–32.556805,0.86207,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,53.504492,47.084447,50.294469,47.084447–53.504492,0.88001,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,42.605267,37.309407,39.957337,37.309407–42.605267,0.87570,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.025553,0.023662,0.024608,0.023662–0.025553,0.92600,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.328512,0.282376,0.305444,0.282376–0.328512,0.85956,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.458403,0.427167,0.442785,0.427167–0.458403,0.93186,26/53/0,26/53/0,0.579621
   poisson-32,native,0.660416,0.626426,0.643421,0.626426–0.660416,0.94853,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.969391,1.939089,1.954240,1.939089–1.969391,0.98461,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,2.082945,1.900871,1.991908,1.900871–2.082945,0.91259,92/185/0,92/185/0,0.838111
   poisson-64,native,4.700235,5.020810,4.860523,4.700235–5.020810,1.06820,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,8.841933,8.867041,8.854487,8.841933–8.867041,1.00284,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,7.041728,7.295486,7.168608,7.041728–7.295486,1.03604,174/349/0,174/349/0,0.912222
   poisson-128,native,34.665914,35.423585,35.044749,34.665914–35.423585,1.02186,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,48.897387,51.657071,50.277229,48.897387–51.657071,1.05644,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,38.665901,40.359221,39.512561,38.665901–40.359221,1.04379,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.106555,0.104682,0.105619,0.104682–0.106555,0.98242,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.276993,0.268226,0.272610,0.268226–0.276993,0.96835,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.463264,0.412183,0.437723,0.412183–0.463264,0.88974,49/99/0,49/99/0,0.794401
   dense-256,native,0.647366,0.603059,0.625212,0.603059–0.647366,0.93156,65/131/0,65/131/0,0.875657
   dense-256,numpy-out,1.046222,0.994287,1.020255,0.994287–1.046222,0.95036,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.208373,1.152715,1.180544,1.152715–1.208373,0.95394,65/131/0,65/131/0,0.87147
   dense-1024,native,20.885707,19.737507,20.311607,19.737507–20.885707,0.94502,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,20.906468,19.276001,20.091234,19.276001–20.906468,0.92201,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,24.601861,25.843764,25.222812,24.601861–25.843764,1.05048,69/139/0,69/139/0,0.857221
   dense-2048,native,145.675795,132.043921,138.859858,132.043921–145.675795,0.90642,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,119.951571,116.954195,118.452883,116.954195–119.951571,0.97501,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,139.357955,126.295916,132.826935,126.295916–139.357955,0.90627,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.141164,0.136286,0.138725,0.136286–0.141164,0.96544,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.285448,0.283592,0.284520,0.283592–0.285448,0.99350,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.463036,0.458545,0.460791,0.458545–0.463036,0.99030,49/99/0,49/99/0,0.794401
   dense-256,native,0.774850,0.716848,0.745849,0.716848–0.774850,0.92514,65/131/0,65/131/0,0.872577
   dense-256,numpy-out,0.870070,0.870321,0.870195,0.870070–0.870321,1.00029,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.216668,1.188022,1.202345,1.188022–1.216668,0.97645,65/131/0,65/131/0,0.87147
   dense-1024,native,21.307633,20.379420,20.843526,20.379420–21.307633,0.95644,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,20.847168,19.503888,20.175528,19.503888–20.847168,0.93557,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,24.960784,22.581891,23.771337,22.581891–24.960784,0.90469,69/139/0,69/139/0,0.857221
   dense-2048,native,147.755527,135.291704,141.523616,135.291704–147.755527,0.91565,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,131.607557,122.287662,126.947609,122.287662–131.607557,0.92918,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,131.807414,127.833111,129.820262,127.833111–131.807414,0.96985,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.101908,0.103187,0.102548,0.101908–0.103187,1.01255,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.282961,0.289876,0.286418,0.282961–0.289876,1.02444,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.461990,0.451074,0.456532,0.451074–0.461990,0.97637,49/99/0,49/99/0,0.794401
   dense-256,native,0.597027,0.678601,0.637814,0.597027–0.678601,1.13663,65/131/0,65/131/0,0.875657
   dense-256,numpy-out,0.848877,1.181155,1.015016,0.848877–1.181155,1.39143,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.028296,1.529133,1.278714,1.028296–1.529133,1.48706,65/131/0,65/131/0,0.87147
   dense-1024,native,21.665662,24.285124,22.975393,21.665662–24.285124,1.12090,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,21.765492,23.334813,22.550152,21.765492–23.334813,1.07210,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,25.206027,32.879615,29.042821,25.206027–32.879615,1.30443,69/139/0,69/139/0,0.857221
   dense-2048,native,187.905637,171.298384,179.602010,171.298384–187.905637,0.91162,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,134.177952,114.793787,124.485870,114.793787–134.177952,0.85553,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,181.789633,150.543193,166.166413,150.543193–181.789633,0.82812,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.126258,0.126804,0.126531,0.126258–0.126804,1.00432,49/99/0,49/99/0,0.806601
   dense-64,numpy-out,0.282203,0.290247,0.286225,0.282203–0.290247,1.02850,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.426530,0.447125,0.436828,0.426530–0.447125,1.04828,49/99/0,49/99/0,0.794401
   dense-256,native,0.659135,0.646157,0.652646,0.646157–0.659135,0.98031,65/131/0,65/131/0,0.872577
   dense-256,numpy-out,0.817076,1.014750,0.915913,0.817076–1.014750,1.24193,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.058981,1.091930,1.075455,1.058981–1.091930,1.03111,65/131/0,65/131/0,0.87147
   dense-1024,native,23.685686,23.746323,23.716005,23.685686–23.746323,1.00256,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,21.696721,22.130430,21.913575,21.696721–22.130430,1.01999,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,24.858833,31.724505,28.291669,24.858833–31.724505,1.27619,69/139/0,69/139/0,0.857221
   dense-2048,native,208.961754,164.436978,186.699366,164.436978–208.961754,0.78692,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,132.730755,120.535183,126.632969,120.535183–132.730755,0.90812,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,190.258056,152.663284,171.460670,152.663284–190.258056,0.80240,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.060501,0.068961,0.064731,0.060501–0.068961,1.13984,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.280463,0.313980,0.297222,0.280463–0.313980,1.11951,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.395143,0.468979,0.432061,0.395143–0.468979,1.18686,26/53/0,26/53/0,0.579621
   poisson-32,native,0.711302,0.830935,0.771119,0.711302–0.830935,1.16819,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.897998,2.203572,2.050785,1.897998–2.203572,1.16100,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,1.827983,2.377642,2.102813,1.827983–2.377642,1.30069,92/185/0,92/185/0,0.838111
   poisson-64,native,3.714224,3.957639,3.835932,3.714224–3.957639,1.06554,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,8.101031,9.041324,8.571178,8.101031–9.041324,1.11607,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,6.310825,7.074743,6.692784,6.310825–7.074743,1.12105,174/349/0,174/349/0,0.912222
   poisson-128,native,32.290009,35.095881,33.692945,32.290009–35.095881,1.08690,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,59.123309,62.784655,60.953983,59.123309–62.784655,1.06193,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,42.760896,41.358357,42.059627,41.358357–42.760896,0.96720,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.073763,0.071368,0.072566,0.071368–0.073763,0.96752,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.282311,0.265580,0.273945,0.265580–0.282311,0.94074,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.409848,0.378178,0.394013,0.378178–0.409848,0.92273,26/53/0,26/53/0,0.579621
   poisson-32,native,0.896954,0.899621,0.898288,0.896954–0.899621,1.00297,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.850270,1.941185,1.895727,1.850270–1.941185,1.04914,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,1.784280,1.775955,1.780117,1.775955–1.784280,0.99533,92/185/0,92/185/0,0.838111
   poisson-64,native,4.713963,4.675987,4.694975,4.675987–4.713963,0.99194,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,7.695140,7.721579,7.708360,7.695140–7.721579,1.00344,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,6.065793,6.459500,6.262646,6.065793–6.459500,1.06491,174/349/0,174/349/0,0.912222
   poisson-128,native,35.997430,35.632951,35.815190,35.632951–35.997430,0.98987,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,54.374912,61.321514,57.848213,54.374912–61.321514,1.12775,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,34.883325,39.587093,37.235209,34.883325–39.587093,1.13484,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.047780,0.055833,0.051806,0.047780–0.055833,1.16852,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.269908,0.307151,0.288529,0.269908–0.307151,1.13798,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.376026,0.456054,0.416040,0.376026–0.456054,1.21282,26/53/0,26/53/0,0.579621
   poisson-32,native,0.665965,0.673810,0.669887,0.665965–0.673810,1.01178,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.760188,1.770795,1.765491,1.760188–1.770795,1.00603,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,1.777549,1.823466,1.800508,1.777549–1.823466,1.02583,92/185/0,92/185/0,0.838111
   poisson-64,native,3.581929,4.022958,3.802443,3.581929–4.022958,1.12313,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,7.526415,8.122395,7.824406,7.526415–8.122395,1.07919,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,6.418216,6.610921,6.514569,6.418216–6.610921,1.03002,174/349/0,174/349/0,0.912222
   poisson-128,native,30.683875,34.273578,32.478727,30.683875–34.273578,1.11699,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,62.201408,58.675067,60.438238,58.675067–62.201408,0.94331,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,40.431697,41.700128,41.065913,40.431697–41.700128,1.03137,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,native,0.050162,0.064960,0.057561,0.050162–0.064960,1.29499,26/53/0,26/53/0,0.579621
   poisson-8,scipy-out,0.208882,0.301239,0.255060,0.208882–0.301239,1.44215,26/53/0,26/53/0,0.579621
   poisson-8,torch-eager-out,0.312964,0.449761,0.381363,0.312964–0.449761,1.43710,26/53/0,26/53/0,0.579621
   poisson-32,native,0.727168,0.817976,0.772572,0.727168–0.817976,1.12488,92/185/0,92/185/0,0.838111
   poisson-32,scipy-out,1.414322,1.746889,1.580605,1.414322–1.746889,1.23514,92/185/0,92/185/0,0.838111
   poisson-32,torch-eager-out,1.486195,1.789219,1.637706,1.486195–1.789219,1.20389,92/185/0,92/185/0,0.838111
   poisson-64,native,4.466442,4.578518,4.522480,4.466442–4.578518,1.02509,174/349/0,174/349/0,0.912222
   poisson-64,scipy-out,6.151206,8.083353,7.117280,6.151206–8.083353,1.31411,174/349/0,174/349/0,0.912222
   poisson-64,torch-eager-out,5.208306,6.481008,5.844658,5.208306–6.481008,1.24436,174/349/0,174/349/0,0.912222
   poisson-128,native,36.931201,38.085974,37.508587,36.931201–38.085974,1.03127,325/651/0,325/651/0,0.986874
   poisson-128,scipy-out,60.830439,48.937785,54.884113,48.937785–60.830439,0.80450,325/651/0,325/651/0,0.986874
   poisson-128,torch-eager-out,38.965885,35.017191,36.991538,35.017191–38.965885,0.89866,325/651/0,325/651/0,0.986874

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.066237,0.082927,0.074582,0.066237–0.082927,1.25198,49/99/0,49/99/0,0.806601
   dense-64,native-blas,0.057178,0.070082,0.063630,0.057178–0.070082,1.22568,50/101/0,50/101/0,0.578069
   dense-64,numpy-out,0.356946,0.420038,0.388493,0.356946–0.420038,1.17675,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.579338,0.780411,0.679875,0.579338–0.780411,1.34707,49/99/0,49/99/0,0.794401
   dense-256,native,0.839750,0.991093,0.915422,0.839750–0.991093,1.18022,65/131/0,65/131/0,0.872577
   dense-256,native-blas,0.818117,0.868839,0.843478,0.818117–0.868839,1.06200,65/131/0,65/131/0,0.870467
   dense-256,numpy-out,1.328439,1.497816,1.413128,1.328439–1.497816,1.12750,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.889175,2.318170,2.103672,1.889175–2.318170,1.22708,65/131/0,65/131/0,0.87147
   dense-1024,native,18.690009,17.276741,17.983375,17.276741–18.690009,0.92438,69/139/0,69/139/0,0.857221
   dense-1024,native-blas,17.378975,16.091919,16.735447,16.091919–17.378975,0.92594,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,18.990537,17.890016,18.440276,17.890016–18.990537,0.94205,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,21.538256,19.916393,20.727324,19.916393–21.538256,0.92470,69/139/0,69/139/0,0.857221
   dense-2048,native,226.347728,184.240322,205.294025,184.240322–226.347728,0.81397,68/137/0,68/137/0,0.981309
   dense-2048,native-blas,175.145237,146.414788,160.780013,146.414788–175.145237,0.83596,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,188.931041,154.413829,171.672435,154.413829–188.931041,0.81730,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,200.710647,164.475394,182.593021,164.475394–200.710647,0.81947,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.082996,0.083138,0.083067,0.082996–0.083138,1.00171,49/99/0,49/99/0,0.806601
   dense-64,native-blas,0.066146,0.059950,0.063048,0.059950–0.066146,0.90632,50/101/0,50/101/0,0.578069
   dense-64,numpy-out,0.394457,0.398174,0.396315,0.394457–0.398174,1.00942,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.667841,0.684760,0.676301,0.667841–0.684760,1.02533,49/99/0,49/99/0,0.794401
   dense-256,native,0.908231,1.044434,0.976333,0.908231–1.044434,1.14997,65/131/0,65/131/0,0.872577
   dense-256,native-blas,0.892911,0.645661,0.769286,0.645661–0.892911,0.72310,65/131/0,65/131/0,0.870467
   dense-256,numpy-out,1.418907,1.203728,1.311318,1.203728–1.418907,0.84835,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,2.033543,1.817264,1.925404,1.817264–2.033543,0.89364,65/131/0,65/131/0,0.87147
   dense-1024,native,19.710224,19.087878,19.399051,19.087878–19.710224,0.96843,69/139/0,69/139/0,0.857221
   dense-1024,native-blas,18.077519,17.367569,17.722544,17.367569–18.077519,0.96073,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,20.150509,19.667928,19.909219,19.667928–20.150509,0.97605,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,22.970860,21.943963,22.457411,21.943963–22.970860,0.95530,69/139/0,69/139/0,0.857221
   dense-2048,native,231.979275,178.604193,205.291733,178.604193–231.979275,0.76991,68/137/0,68/137/0,0.981309
   dense-2048,native-blas,182.271105,138.532464,160.401784,138.532464–182.271105,0.76004,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,189.258790,145.987689,167.623239,145.987689–189.258790,0.77137,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,201.119620,152.592215,176.855918,152.592215–201.119620,0.75871,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.058942,0.049920,0.054431,0.049920–0.058942,0.84693,49/99/0,49/99/0,0.806601
   dense-64,native-blas,0.036133,0.035500,0.035816,0.035500–0.036133,0.98247,50/101/0,50/101/0,0.578069
   dense-64,numpy-out,0.352485,0.305742,0.329114,0.305742–0.352485,0.86739,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.565728,0.470637,0.518183,0.470637–0.565728,0.83191,49/99/0,49/99/0,0.794401
   dense-256,native,0.843877,0.791088,0.817482,0.791088–0.843877,0.93744,65/131/0,65/131/0,0.872577
   dense-256,native-blas,0.587288,0.584564,0.585926,0.584564–0.587288,0.99536,65/131/0,65/131/0,0.870467
   dense-256,numpy-out,1.263447,1.091951,1.177699,1.091951–1.263447,0.86426,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.491798,1.227426,1.359612,1.227426–1.491798,0.82278,65/131/0,65/131/0,0.87147
   dense-1024,native,23.907758,24.052882,23.980320,23.907758–24.052882,1.00607,69/139/0,69/139/0,0.857221
   dense-1024,native-blas,21.215549,21.229451,21.222500,21.215549–21.229451,1.00066,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,23.466624,22.703269,23.084946,22.703269–23.466624,0.96747,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,26.116602,26.112040,26.114321,26.112040–26.116602,0.99983,69/139/0,69/139/0,0.857221
   dense-2048,native,260.734385,241.650778,251.192581,241.650778–260.734385,0.92681,68/137/0,68/137/0,0.981309
   dense-2048,native-blas,246.602751,190.613286,218.608018,190.613286–246.602751,0.77296,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,186.058916,148.603872,167.331394,148.603872–186.058916,0.79869,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,258.514980,197.551168,228.033074,197.551168–258.514980,0.76418,68/137/0,68/137/0,0.981309

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,native,0.066429,0.066008,0.066218,0.066008–0.066429,0.99367,49/99/0,49/99/0,0.806601
   dense-64,native-blas,0.045704,0.049938,0.047821,0.045704–0.049938,1.09264,50/101/0,50/101/0,0.578069
   dense-64,numpy-out,0.351453,0.409189,0.380321,0.351453–0.409189,1.16428,50/101/0,50/101/0,0.58563
   dense-64,torch-eager-out,0.596379,0.751849,0.674114,0.596379–0.751849,1.26069,49/99/0,49/99/0,0.794401
   dense-256,native,0.901215,0.904001,0.902608,0.901215–0.904001,1.00309,65/131/0,65/131/0,0.872577
   dense-256,native-blas,0.595709,0.618910,0.607309,0.595709–0.618910,1.03895,65/131/0,65/131/0,0.870467
   dense-256,numpy-out,1.325352,1.401946,1.363649,1.325352–1.401946,1.05779,65/131/0,65/131/0,0.87045
   dense-256,torch-eager-out,1.495086,1.830811,1.662948,1.495086–1.830811,1.22455,65/131/0,65/131/0,0.87147
   dense-1024,native,25.752859,23.263423,24.508141,23.263423–25.752859,0.90333,69/139/0,69/139/0,0.857221
   dense-1024,native-blas,23.592505,21.815080,22.703792,21.815080–23.592505,0.92466,69/139/0,69/139/0,0.857221
   dense-1024,numpy-out,24.532580,22.936529,23.734554,22.936529–24.532580,0.93494,69/139/0,69/139/0,0.857221
   dense-1024,torch-eager-out,28.309359,26.118820,27.214089,26.118820–28.309359,0.92262,69/139/0,69/139/0,0.857221
   dense-2048,native,307.638187,239.299276,273.468731,239.299276–307.638187,0.77786,68/137/0,68/137/0,0.981309
   dense-2048,native-blas,233.933849,171.870033,202.901941,171.870033–233.933849,0.73470,68/137/0,68/137/0,0.981309
   dense-2048,numpy-out,183.735955,140.425669,162.080812,140.425669–183.735955,0.76428,68/137/0,68/137/0,0.981309
   dense-2048,torch-eager-out,257.604451,185.630472,221.617462,185.630472–257.604451,0.72060,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,14.058542,14.650977,14.354760,14.058542–14.650977,1.04214,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,15.973721,16.907642,16.440681,15.973721–16.907642,1.05847,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,27.724572,29.410492,28.567532,27.724572–29.410492,1.06081,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,18.252636,17.852459,18.052547,17.852459–18.252636,0.97808,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,21.091321,20.006684,20.549003,20.006684–21.091321,0.94857,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,35.397672,35.838543,35.618108,35.397672–35.838543,1.01245,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,25.107782,23.149987,24.128884,23.149987–25.107782,0.92202,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,24.590349,22.155276,23.372812,22.155276–24.590349,0.90097,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,37.550641,32.973305,35.261973,32.973305–37.550641,0.87810,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,38.515605,39.208888,38.862247,38.515605–39.208888,1.01800,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,37.364341,37.014359,37.189350,37.014359–37.364341,0.99063,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,50.095515,47.882407,48.988961,47.882407–50.095515,0.95582,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,15.934628,14.438716,15.186672,14.438716–15.934628,0.90612,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,16.950553,16.006556,16.478555,16.006556–16.950553,0.94431,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,31.780228,26.629165,29.204697,26.629165–31.780228,0.83792,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,20.679695,20.375328,20.527511,20.375328–20.679695,0.98528,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,21.368254,20.274052,20.821153,20.274052–21.368254,0.94879,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,35.905850,35.827219,35.866535,35.827219–35.905850,0.99781,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,24.143333,25.761767,24.952550,24.143333–25.761767,1.06703,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,24.325644,25.354189,24.839917,24.325644–25.354189,1.04228,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,36.034475,38.220263,37.127369,36.034475–38.220263,1.06066,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,40.417402,37.991625,39.204513,37.991625–40.417402,0.93998,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,39.736722,37.014387,38.375554,37.014387–39.736722,0.93149,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,49.981307,47.231041,48.606174,47.231041–49.981307,0.94497,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,8.885770,9.068651,8.977211,8.885770–9.068651,1.02058,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,11.420613,11.457804,11.439208,11.420613–11.457804,1.00326,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,25.868190,29.107190,27.487690,25.868190–29.107190,1.12521,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,12.853011,13.902739,13.377875,12.853011–13.902739,1.08167,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,14.669080,17.378665,16.023872,14.669080–17.378665,1.18471,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,36.748581,46.074005,41.411293,36.748581–46.074005,1.25376,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,17.188703,27.031591,22.110147,17.188703–27.031591,1.57264,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,18.636920,27.424858,23.030889,18.636920–27.424858,1.47153,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,36.110300,58.244229,47.177264,36.110300–58.244229,1.61295,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,32.428401,31.327055,31.877728,31.327055–32.428401,0.96604,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,32.154882,30.574742,31.364812,30.574742–32.154882,0.95086,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,49.355117,47.130859,48.242988,47.130859–49.355117,0.95493,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,9.544292,9.690995,9.617643,9.544292–9.690995,1.01537,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,11.638036,11.543977,11.591006,11.543977–11.638036,0.99192,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,26.583936,27.715077,27.149507,26.583936–27.715077,1.04255,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,13.101762,12.832271,12.967016,12.832271–13.101762,0.97943,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,14.936086,15.011349,14.973717,14.936086–15.011349,1.00504,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,35.780320,37.294940,36.537630,35.780320–37.294940,1.04233,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,18.826749,20.121223,19.473986,18.826749–20.121223,1.06876,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,19.009367,19.694104,19.351736,19.009367–19.694104,1.03602,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,37.571370,39.834435,38.702902,37.571370–39.834435,1.06023,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,32.716302,35.359534,34.037918,32.716302–35.359534,1.08079,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,32.323718,34.976872,33.650295,32.323718–34.976872,1.08208,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,50.256729,52.804490,51.530609,50.256729–52.804490,1.05069,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,15.040579,15.806244,15.423412,15.040579–15.806244,1.05091,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,14.640560,16.283231,15.461896,14.640560–16.283231,1.11220,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,27.732084,29.803040,28.767561,27.732084–29.803040,1.07468,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,20.472446,20.068572,20.270509,20.068572–20.472446,0.98027,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,20.153305,20.097064,20.125184,20.097064–20.153305,0.99721,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,39.148637,38.022934,38.585786,38.022934–39.148637,0.97125,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,24.226235,25.514924,24.870580,24.226235–25.514924,1.05319,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,21.840394,23.091439,22.465917,21.840394–23.091439,1.05728,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,35.355460,38.719870,37.037664,35.355460–38.719870,1.09516,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,41.528026,41.870287,41.699157,41.528026–41.870287,1.00824,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,38.726129,38.587018,38.656574,38.587018–38.726129,0.99641,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,53.218734,51.576124,52.397429,51.576124–53.218734,0.96913,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,14.511916,15.695087,15.103501,14.511916–15.695087,1.08153,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,13.724120,15.682381,14.703251,13.724120–15.682381,1.14269,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,26.131397,29.410949,27.771173,26.131397–29.410949,1.12550,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,20.972230,21.873060,21.422645,20.972230–21.873060,1.04295,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,19.571218,20.479109,20.025164,19.571218–20.479109,1.04639,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,36.142337,38.422765,37.282551,36.142337–38.422765,1.06310,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,25.759374,26.825474,26.292424,25.759374–26.825474,1.04139,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,24.028932,24.957244,24.493088,24.028932–24.957244,1.03863,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,38.266934,39.312187,38.789560,38.266934–39.312187,1.02731,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,38.199442,41.883786,40.041614,38.199442–41.883786,1.09645,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,35.801462,38.785523,37.293493,35.801462–38.785523,1.08335,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,47.701013,52.551553,50.126283,47.701013–52.551553,1.10169,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,8.935327,9.325896,9.130612,8.935327–9.325896,1.04371,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,9.384726,10.000414,9.692569,9.384726–10.000414,1.06561,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,28.031965,30.131658,29.081812,28.031965–30.131658,1.07490,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,12.009194,11.821022,11.915109,11.821022–12.009194,0.98433,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,11.656955,12.452276,12.054616,11.656955–12.452276,1.06823,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,33.435250,36.914456,35.174853,33.435250–36.914456,1.10406,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,17.546061,17.660893,17.603477,17.546061–17.660893,1.00654,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,15.627963,15.611591,15.619777,15.611591–15.627963,0.99895,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,37.942134,36.569020,37.255577,36.569020–37.942134,0.96381,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,34.361909,33.125050,33.743479,33.125050–34.361909,0.96400,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,31.576791,30.412778,30.994784,30.412778–31.576791,0.96314,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,52.863123,51.002815,51.932969,51.002815–52.863123,0.96481,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,9.029711,10.080713,9.555212,9.029711–10.080713,1.11639,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,9.091935,10.371395,9.731665,9.091935–10.371395,1.14072,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,25.710005,29.264785,27.487395,25.710005–29.264785,1.13826,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,12.547605,14.074013,13.310809,12.547605–14.074013,1.12165,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,12.763061,14.092705,13.427883,12.763061–14.092705,1.10418,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,37.537414,37.528423,37.532919,37.528423–37.537414,0.99976,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,17.975721,20.548696,19.262209,17.975721–20.548696,1.14314,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,16.164727,18.056027,17.110377,16.164727–18.056027,1.11700,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,37.602561,41.153249,39.377905,37.602561–41.153249,1.09443,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,31.180636,34.877497,33.029067,31.180636–34.877497,1.11856,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,29.073847,32.415203,30.744525,29.073847–32.415203,1.11493,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,46.464176,53.741716,50.102946,46.464176–53.741716,1.15663,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,14.299883,15.302511,14.801197,14.299883–15.302511,1.07011,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,14.203869,15.750359,14.977114,14.203869–15.750359,1.10888,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,27.845617,30.502713,29.174165,27.845617–30.502713,1.09542,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,18.073747,21.083376,19.578562,18.073747–21.083376,1.16652,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,17.436299,21.289552,19.362925,17.436299–21.289552,1.22099,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,33.865271,39.367385,36.616327,33.865271–39.367385,1.16247,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,24.560104,24.845338,24.702721,24.560104–24.845338,1.01161,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,22.365848,23.327780,22.846814,22.365848–23.327780,1.04301,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,39.152139,36.956045,38.054092,36.956045–39.152139,0.94391,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,38.335634,41.416897,39.876266,38.335634–41.416897,1.08038,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,35.248866,39.393879,37.321373,35.248866–39.393879,1.11759,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,49.619523,52.995767,51.307645,49.619523–52.995767,1.06804,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,14.551293,16.423010,15.487152,14.551293–16.423010,1.12863,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,13.795271,16.294100,15.044686,13.795271–16.294100,1.18114,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,28.034751,29.752530,28.893641,28.034751–29.752530,1.06127,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,20.871927,21.371777,21.121852,20.871927–21.371777,1.02395,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,20.329078,20.779891,20.554485,20.329078–20.779891,1.02218,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,40.462180,39.719243,40.090711,39.719243–40.462180,0.98164,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,23.472830,27.126693,25.299761,23.472830–27.126693,1.15566,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,21.707798,24.905714,23.306756,21.707798–24.905714,1.14732,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,34.804864,39.947921,37.376393,34.804864–39.947921,1.14777,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,40.309208,41.797734,41.053471,40.309208–41.797734,1.03693,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,37.679487,38.888484,38.283986,37.679487–38.888484,1.03209,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,52.026528,52.061948,52.044238,52.026528–52.061948,1.00068,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,8.396119,9.441619,8.918869,8.396119–9.441619,1.12452,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,9.574396,9.940203,9.757300,9.574396–9.940203,1.03821,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,26.699912,29.287105,27.993509,26.699912–29.287105,1.09690,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,13.115176,11.927406,12.521291,11.927406–13.115176,0.90944,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,13.626630,12.679125,13.152878,12.679125–13.626630,0.93047,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,37.362689,37.162357,37.262523,37.162357–37.362689,0.99464,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,18.344732,18.385200,18.364966,18.344732–18.385200,1.00221,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,16.791155,16.559469,16.675312,16.559469–16.791155,0.98620,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,40.553453,37.759778,39.156616,37.759778–40.553453,0.93111,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,32.040557,34.456790,33.248674,32.040557–34.456790,1.07541,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,30.006324,31.511948,30.759136,30.006324–31.511948,1.05018,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,49.019630,51.850870,50.435250,49.019630–51.850870,1.05776,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   dense-64,cuda-native,11.719551,10.589122,11.154337,10.589122–11.719551,0.90354,50/101/0,50/101/0,0.575554
   dense-64,cuda-cublas,11.841250,10.595624,11.218437,10.595624–11.841250,0.89481,49/99/0,49/99/0,0.92839
   dense-64,torch-cuda-eager,33.224536,29.075913,31.150225,29.075913–33.224536,0.87513,49/99/0,49/99/0,0.92839
   dense-256,cuda-native,12.963257,14.239102,13.601180,12.963257–14.239102,1.09842,65/131/0,65/131/0,0.869377
   dense-256,cuda-cublas,13.410215,13.901572,13.655894,13.410215–13.901572,1.03664,65/131/0,65/131/0,0.870897
   dense-256,torch-cuda-eager,36.101774,38.067123,37.084449,36.101774–38.067123,1.05444,65/131/0,65/131/0,0.873989
   dense-1024,cuda-native,18.824558,19.898542,19.361550,18.824558–19.898542,1.05705,69/139/0,69/139/0,0.857221
   dense-1024,cuda-cublas,16.523961,17.662537,17.093249,16.523961–17.662537,1.06890,69/139/0,69/139/0,0.857221
   dense-1024,torch-cuda-eager,35.183495,38.798175,36.990834,35.183495–38.798175,1.10274,69/139/0,69/139/0,0.857221
   dense-2048,cuda-native,31.937415,34.633390,33.285402,31.937415–34.633390,1.08441,68/137/0,68/137/0,0.981309
   dense-2048,cuda-cublas,29.927633,32.048972,30.988302,29.927633–32.048972,1.07088,68/137/0,68/137/0,0.981309
   dense-2048,torch-cuda-eager,49.739182,51.563669,50.651426,49.739182–51.563669,1.03668,68/137/0,68/137/0,0.981309

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.879004,8.548049,7.213527,5.879004–8.548049,1.45400,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,11.467764,16.474370,13.971067,11.467764–16.474370,1.43658,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,24.139472,29.028234,26.583853,24.139472–29.028234,1.20252,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,43.533017,58.340545,50.936781,43.533017–58.340545,1.34014,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,50.457859,56.297558,53.377708,50.457859–56.297558,1.11573,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,93.536737,112.822401,103.179569,93.536737–112.822401,1.20618,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,89.770535,100.670998,95.220767,89.770535–100.670998,1.12143,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,166.566773,191.324288,178.945531,166.566773–191.324288,1.14863,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,6.960807,7.954951,7.457879,6.960807–7.954951,1.14282,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,12.113958,14.709400,13.411679,12.113958–14.709400,1.21425,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,25.130271,28.526132,26.828202,25.130271–28.526132,1.13513,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,45.828468,49.324291,47.576380,45.828468–49.324291,1.07628,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,47.595344,51.636685,49.616015,47.595344–51.636685,1.08491,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,81.457309,89.527990,85.492650,81.457309–89.527990,1.09908,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,102.159974,117.150515,109.655244,102.159974–117.150515,1.14674,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,167.519925,222.184447,194.852186,167.519925–222.184447,1.32632,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,4.858759,5.117767,4.988263,4.858759–5.117767,1.05331,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,14.966836,14.864387,14.915612,14.864387–14.966836,0.99315,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,15.721713,17.532452,16.627083,15.721713–17.532452,1.11517,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,43.710394,49.330731,46.520563,43.710394–49.330731,1.12858,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,28.243506,31.532888,29.888198,28.243506–31.532888,1.11647,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,82.391092,92.078803,87.234948,82.391092–92.078803,1.11758,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,56.007689,58.191772,57.099731,56.007689–58.191772,1.03900,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,163.477279,174.195028,168.836153,163.477279–174.195028,1.06556,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.192437,5.510792,5.351614,5.192437–5.510792,1.06131,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,13.924474,14.529575,14.227024,13.924474–14.529575,1.04346,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,17.661645,18.329395,17.995520,17.661645–18.329395,1.03781,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,46.422719,50.952549,48.687634,46.422719–50.952549,1.09758,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,32.670181,35.643480,34.156830,32.670181–35.643480,1.09101,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,83.094542,98.071303,90.582922,83.094542–98.071303,1.18024,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,57.972616,60.603213,59.287914,57.972616–60.603213,1.04538,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,149.301809,165.020114,157.160962,149.301809–165.020114,1.10528,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,7.276061,8.444583,7.860322,7.276061–8.444583,1.16060,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,12.528159,15.674777,14.101468,12.528159–15.674777,1.25116,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,24.222421,28.981627,26.602024,24.222421–28.981627,1.19648,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,43.839297,52.340033,48.089665,43.839297–52.340033,1.19391,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,47.276028,54.707105,50.991567,47.276028–54.707105,1.15718,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,85.249878,96.512682,90.881281,85.249878–96.512682,1.13212,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,94.103325,96.971793,95.537559,94.103325–96.971793,1.03048,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,178.475608,174.237715,176.356662,174.237715–178.475608,0.97626,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,6.279138,7.631652,6.955395,6.279138–7.631652,1.21540,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,12.008890,14.340486,13.174688,12.008890–14.340486,1.19416,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,24.162759,28.907253,26.535007,24.162759–28.907253,1.19636,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,41.429459,50.802450,46.115955,41.429459–50.802450,1.22624,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,51.382472,54.465471,52.923971,51.382472–54.465471,1.06000,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,90.123593,95.810919,92.967256,90.123593–95.810919,1.06311,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,98.324614,96.690998,97.507806,96.690998–98.324614,0.98339,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,169.381823,171.530391,170.456107,169.381823–171.530391,1.01268,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.154875,4.906061,5.030468,4.906061–5.154875,0.95173,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,13.079520,14.760066,13.919793,13.079520–14.760066,1.12849,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,16.882137,16.998911,16.940524,16.882137–16.998911,1.00692,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,50.851836,48.827045,49.839441,48.827045–50.851836,0.96018,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,30.500945,30.777047,30.638996,30.500945–30.777047,1.00905,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,84.031425,92.496674,88.264050,84.031425–92.496674,1.10074,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,54.815125,57.201445,56.008285,54.815125–57.201445,1.04353,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,159.639062,173.042655,166.340858,159.639062–173.042655,1.08396,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.282017,6.474045,5.878031,5.282017–6.474045,1.22568,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,13.489336,18.112920,15.801128,13.489336–18.112920,1.34276,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,20.518727,20.885578,20.702153,20.518727–20.885578,1.01788,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,66.324190,53.365164,59.844677,53.365164–66.324190,0.80461,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,36.609571,43.279972,39.944772,36.609571–43.279972,1.18220,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,111.101586,113.767469,112.434527,111.101586–113.767469,1.02399,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,59.916047,61.247252,60.581649,59.916047–61.247252,1.02222,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,159.344275,160.692440,160.018357,159.344275–160.692440,1.00846,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,9.567745,7.843083,8.705414,7.843083–9.567745,0.81974,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,18.341172,14.662223,16.501697,14.662223–18.341172,0.79942,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,30.822299,29.813685,30.317992,29.813685–30.822299,0.96728,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,55.862072,56.257110,56.059591,55.862072–56.257110,1.00707,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,52.713267,55.303879,54.008572,52.713267–55.303879,1.04915,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,98.327063,99.524478,98.925770,98.327063–99.524478,1.01218,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,103.219183,108.506063,105.862623,103.219183–108.506063,1.05122,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,181.383203,199.480446,190.431825,181.383203–199.480446,1.09977,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,9.752802,8.589344,9.171073,8.589344–9.752802,0.88071,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,17.192640,16.411524,16.802082,16.411524–17.192640,0.95457,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,31.457576,28.372770,29.915173,28.372770–31.457576,0.90194,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,57.374919,49.944764,53.659841,49.944764–57.374919,0.87050,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,51.954075,52.287552,52.120814,51.954075–52.287552,1.00642,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,92.985037,94.766645,93.875841,92.985037–94.766645,1.01916,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,103.437962,95.156243,99.297102,95.156243–103.437962,0.91994,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,184.264555,171.389467,177.827011,171.389467–184.264555,0.93013,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,5.107865,4.812090,4.959978,4.812090–5.107865,0.94209,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,13.957503,14.085888,14.021695,13.957503–14.085888,1.00920,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,17.082856,16.826434,16.954645,16.826434–17.082856,0.98499,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,48.310257,50.544469,49.427363,48.310257–50.544469,1.04625,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,31.537039,30.951393,31.244216,30.951393–31.537039,0.98143,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,91.534428,93.194845,92.364637,91.534428–93.194845,1.01814,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,56.328971,55.343103,55.836037,55.343103–56.328971,0.98250,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,165.103872,165.510604,165.307239,165.103872–165.510604,1.00246,325/651/0,325/651/0,0.986874

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,Forward ms,Reverse ms,Median ms,Process range ms,Reverse/forward,Forward I/M/R,Reverse I/M/R,Max TrueR/tol
   poisson-8,cuda-native,6.063617,7.684184,6.873901,6.063617–7.684184,1.26726,26/53/0,26/53/0,0.579621
   poisson-8,torch-cuda-eager,15.573287,23.394591,19.483939,15.573287–23.394591,1.50223,26/53/0,26/53/0,0.579621
   poisson-32,cuda-native,19.714223,17.581281,18.647752,17.581281–19.714223,0.89181,92/185/0,92/185/0,0.838111
   poisson-32,torch-cuda-eager,53.330266,45.859043,49.594654,45.859043–53.330266,0.85991,92/185/0,92/185/0,0.838111
   poisson-64,cuda-native,33.359356,33.905105,33.632231,33.359356–33.905105,1.01636,174/349/0,174/349/0,0.912222
   poisson-64,torch-cuda-eager,91.458936,90.371815,90.915376,90.371815–91.458936,0.98811,174/349/0,174/349/0,0.912222
   poisson-128,cuda-native,61.370146,59.762320,60.566233,59.762320–61.370146,0.97380,325/651/0,325/651/0,0.986874
   poisson-128,torch-cuda-eager,165.184386,161.071602,163.127994,161.071602–165.184386,0.97510,325/651/0,325/651/0,0.986874

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

   git fetch origin 445275a3994e342b06de4093889f3f814c8706c6
   git switch --detach 445275a3994e342b06de4093889f3f814c8706c6
   PROJECT="$PWD"
   export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   unset CUDA_LAUNCH_BLOCKING

The following 268 commands preserve the complete forward/reverse order and
unique outputs. Install the recorded local dependencies and toolkit first.
Each command records its thread variables, compiler flags and affinity;
Make may build before the worker, so finish all builds before a quiet rerun.
Equivalent cgbench CLI argv and loaded-stage paths remain in the plan.

.. code-block:: sh

   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=sparse-solvers PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-sparse-solvers-none-api-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=sparse-solvers PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-sparse-solvers-none-prepared-fused1-host-standard-r01.json 'PYPROF_ARGS=--implementations native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit-r01.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm explicit --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=sparse-solvers PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-sparse-solvers-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=sparse-solvers PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-sparse-solvers-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=session PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode device' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cuda-release PYTHON="$PROJECT/.venv-cuda/bin/python" BUILD_DIR=build/reproduce-cuda-release PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations cuda-native,cuda-cublas,torch-cuda-eager --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3 --device cuda --cuda-device 0 --cuda-execution-mode host' CUDA_ROOT="$PROJECT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"' -DCMAKE_CUDA_FLAGS= '"'"'-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-blas PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,native-blas,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BLIS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 taskset -c 0,2 make pyprof PRESET=cpu-openmp PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-openmp PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=2 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-csr-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,scipy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner jacobi --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=prepared PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --no-fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 make pyprof PRESET=cpu-release PYTHON="$PROJECT/.venv/bin/python" BUILD_DIR=build/reproduce-cpu-release PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_SCOPE=api PYPROF_THREADS=1 PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 PYPROF_OUTPUT=build/main-reproduction/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-r02.json 'PYPROF_ARGS=--implementations native,numpy-out,torch-eager-out --seed 20261017 --algorithm standard --fused --preconditioner none --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --warmups 3' 'CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_CXX_FLAGS= '"'"'-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'"'"''

Manifest SHA256: ``6d3e5f6ae08b6e09c3a7b3fd6269385571b4330b52664525403c5005eb381652``.
Plan SHA256: ``425faa634e2cd385f8fed674e4cd53a4db06788fdee4c11fa0ffa6aa7d7403df``.
Characterization SHA256: ``601c891661180b0ffbd228f89aab28584155e1bc07281b911fa8403165fa15ff``.
Renderer SHA256: ``99b13bbec938865ed17780535ea6090bb889470f052406f48d46539b46192073``.

.. csv-table:: Frozen audit and worker source provenance
   :class: benchmark-results
   :header-rows: 1

   Path,SHA256
   build/main-performance-20261017/run_baseline.py,8134d1bf62ae6f35b6adce30df3ad7f071eaf65626975e2477b310452d99ecf3
   build/main-performance-20261017/protocol.py,e9d8ee8e95b06c5a41416f807615b5f23020775702f4cb838dce7dd687ebcb0a
   build/main-performance-20261017/worker.py,af09787bf523283aa1c4853a1ac51f10aa8487181ce2c70a67b07eed954bb840
   build/main-performance-20261017/freeze_stage.py,6e4e06991f095c31d073df67ac2418d2eee52acc4d1226b4e47f108c1d20a2b4
   build/main-performance-20261017/README.md,22d72a2255083e5a8c3597d225493e129a0e5ce452505205508627a889931ec4

.. csv-table:: All retained raw/import sidecars
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Imports SHA256,Log SHA256
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-r01.json,d91af12fda77ccf0be0a4b5a5cfa4b156d325e7b06c4102df65c74e560b122c6,6101c3520b8336444aea41891fd7c29a365813a8f198a3dae16d02ed0a1e7f06,ecb13bc1946d5ca63c88904a1e817ba49b023bc90954ba8ba8309641d2e640cc
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard-r01.json,05cd26fd72facba35264f1ff1fbbdc17ba0faeefaee02431a7b2a569228b964a,0d7facddf5f0ef56d8223f4530175099ea806bcb717e44addc3dc318821ee253,3f1abb0c9cad1ec2c25d9e2507a4d7be58e569abdb9cceff5b4de5cb75b5286a
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01.json,0ac912b5c486def2a6f502283352ffb2b8bcc436cdf1fc3348d95f4e6cde9c81,b1560480c499ca7fe6d9cc68bfda603d5daeb7f5216bace110a807e8846a7c10,0d470c2a6e29e862a2d808d45b19ae53b5d3fe03176657988398981fb76b2564
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01.json,93d0d89f2b29e9edcc721044a079e77b12d61a3556add88a52ade9243df175c3,9f4b25fc239ba9c15e620326ee93cabddbbe7b3640a174aae26229ee377e55cc,644594d3adbfccc8cafc648c2a5bbec564e4d3cf2effde528991066b6fe48623
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json,3a62a64a8f40db138b9ac1b6c8d7fe267eb220f3b313fafd7a60bd2bae0cbd6e,cb980ec48878d773274d4193e14f2f0f57cf031cf1fd12a818d1a71b2a038acd,150ded9cdbf2cc6e1faf1bf2337ceaf5927ca320f7a082e0fc5c33b912252188
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json,cfc16e4f956b2f50450d8ddaf8f87d76c258fe4055353afa38acb5ba9571a30a,06948040344f374b7406534231e632cbc05d98ac4ad4bc3db5af06f71ae12cbf,e02faf94ab1ce3d073829c65e83c1b61a58ad1c3cb3b4589540d717c1d681ac2
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,510d76dc5f3ab9bccc346c6ea0fc3e143680c8ffb8066044eb13f6db79a1214f,0a707f4213bef16f94bd3814212b9c7b67579e5a19f272f94c760323d8418d11,dc411294fdee7b68c30959e0c1ba5db57a63440a1cc8def07efa2750a50cee33
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,968c014e7b3f2d788be557efadf13cdbb8ac8374635460257597e10f0dd52b32,aed2aab1a4f374ee92d158ad5244baefb30c78f386f636e5a73747065996e6d4,cec53ad205d23a00755e247527b7b0afbc9ec00144702b3a6dfdc15ecdc50c53
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard-r01.json,4f4c114eb2d3569b3c69f614dedec7cdd3a427249bea6e39317685602a2b8c8c,1e4262e37ac05e379a5869679a9c0a6fc010495d0fdb34912f2bfdd776d3fc7c,122cf282eb9ed66945f642540010c0cc341bf0b60151ba65914e21359732b2ba
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard-r01.json,d7dbe70e6cf1e198991c9634077f1be23815d8c8756e44319f39e7c02aaeac93,9fa25891969be3db0bbe8f8354c84224491740d1e90e4b31cf8a961bcd2dfba4,b5dd882a05db9c1fbecc6b3ab6807adeaae4d4646a3e35bb56c64aff2fa08329
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r01.json,08f13097f0247cff1038b53a194ca33d5b222683945d7267f89871e6a185f4cf,2fcdbaf166adc5c49a70f36e7224368302f6ba71d6323c4fdecbea12e52d8060,055c3c664252be3e051b5e77d35eb40a73fe1a4c7254de385a591e2f38e48f4d
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r01.json,56a9f53c4e3144a9b562dca8e05db796af8bfce53eed476b3b7a72720bb33ac7,3b32f6f87c2eef230af49b47795eaa7b8a46b9248c71c320a06fa55fa66ec832,8ab4df7f4eb7d5a5e7333fc634af5ad4d8f45de9b3828bb8cea7028f05769589
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r01.json,231b460668d97a452052505ff383719371519d9e060d32c3b8719fe5ac033d6f,8880abefd865d37840981a875180b8c00c257f067f329b9c2edd33753fbca9c7,91e409f325315a687e575abc644167e4b71459d86a609efc6e66546d0d59ac73
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r01.json,4d8d9aaaa67bbe6049377e203bf63d2b90930ed1a2f820ab35f4ac654b8e0724,5a9c6ab308eba613a3d1cb84b15cc384cfbc1a2472d8846fa5f69673e08c5746,fada372d1906cc2e2bf7ec5d780d195bb0c6b66ea38c7b93a9fde1493263ad23
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,c9af8f473ee67bd4484a4ed5688a721f3ffaa8f43aed355d70974d7bfc5e98bb,9086e74cfcd72e35274a9b5ca001c55d3aa1917f56eefe31bab7b941566700a0,b43dbefa127b729874dd30622030fe24ca4b46042ddf56944161dcfb2c449882
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,4ab17837316dba36e911a30b9237431dcd92160a874704808efb258980168fa1,0ee7a2488f967fd7fc07b4bbb8327f56789329d471f5c5a336c047d592e89fed,f9cded5327af49c6ab8ee04e37b748f0c1cee53e15672426dc1b0d7472f6727f
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard-r01.json,bb127f4c2c396e6d942621bc81b6e9fa6fba56fcc0186e62f8a6cb1df632f598,a63eeba469c394b2f892639ff9986301bf92c9477203edb94bd6cd9f51bae9cc,73aea71ad87ea82ef4c5ef00e136c9b78a8e8e4e0db75ef491b8ccf8c57e13da
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard-r01.json,5e9de676bf09d2708b0e5004a80797d95a4c6b7fdbe22cb630c3ec0a7f1e85ac,6e991d1c740ef1e5720c7e27a90b05f9ff8b03e7ea43c6c3042dfeeb7248e903,6765b31ea2b90f3df11d5c062f90c8594a579fc8556ff8cd9254930ebf33c587
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01.json,fee5d8ccdfd46b8c2269cb95c007d766853cf66aeffaeb41b0c1c8507e3cc2ae,f0ea220e2563112ba4705ffc54acbf169ff471ca823ff555890b62914afc6f7a,8c2288dd43ea9dc0eaa197044061f7960653af941dc6a7aac780b9aa9ecd77b5
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01.json,c41cedba8798c540d7e1e9ef8203d5c8dc41c442f6a249ec2b6d8024023a0a85,37cbb95422a956d5383c11aa1f2f6ac2fd3fbf4a41e3a502762f9ad85873cb70,d66ebd612ff8d4a40d7093279df08eb08b12299ff6cd55a8c43423d17f90619d
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json,bfc9a4ccc7cedbbfd49f28cc9317b5ec8bb496005523b2242f035ddec0841633,12d096afefe5d698527910ce7326328603d896f0e31e3b702308b33a92f34e4f,f04bd78037b5c37f9b6cbd90c929a500932815c1641e831ef1d0d20bdf514a3e
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json,90857aea8bc8913c8c8d708807263fe071ddb1da4c1904894d29759f2708eaa4,87caab8e131e31e02f81a34a9304d1f8a432cef6575fbe14a47f4ec8c8f7dd9c,e48275bc77362542d83e445877d76006e965bb05db75a5b1708e2a1e3a670417
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,37f2380f3f0b9368a7355f5ef20fbd7425667ae5104a021b347e3e8da72b2d1d,9b2b0301dba6d3c77d5a175897551374a1878b05935b1566751a699032b88563,1b3e4a75ee985b27411a2d9cf953c17b4275a5e6f682e4aa812581d9d1bd7fd2
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,bf42c941fd135a5779ce4f5f2b45ddf240a4c2dc0d93a4441c6c182ff4bb8450,30c4b07b485120b59fdac350df84963bebee25e00c62060001e55cb3ff500fab,6a70f70fce8b670617208b7663962fde5dc8733967255cfb5df5112ae992d340
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard-r01.json,bf7aa0361a2525d6db34686c22ae128e61c5f790ad3db2ac630cb0d83acffb38,03f337378e879a3b37050905f8f537da5f9ff1e836f94f86b70580cae1d3c46e,1504cb8aef6fc05ed77438e6300c4564bfed40154639e6cc8610564ec751d68a
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard-r01.json,498f6dca0adc32dfdb4d13345b8b01f3e16c64c6e30d21412c431dca885a01ee,2d0ad873b14a11b389a3920185433e0419e80f0acd519d47471897a2320a20be,f2578ebd68699118b4474ecb8a0ca6d4a1c8fc3b0defe1e9cda11b7701308ddc
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r01.json,78bfbad2536e7286bec704290ab97a0aac5bda613dc226270fa39da86ff78bf1,b9c8810fd909c9c6213181d9263d3ac3f9ccda06273ae51a036db1a022ff737c,a2ea9e030003cc301472275a15e2fc415d0526e10dd14735aaf07ce1fd22f005
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r01.json,99ba964284767ac52684c36e3f7388b063b36cd6be304dde4b62f496ed7903b2,069fadb8f86c16da24f6babe44f3624223c40f1e03c20dbb35eaa51d17db0a55,2b29cc861d6fbabab8d5dadeb52ad863163e71deed2aa52ff12b8f152a57b088
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r01.json,c3dd8125b7b2fa70e226272a4675f321a9174c56a3f86fb5baf38b7fde29f0c5,67250594f292532cc2b30db512937a48af5996c02e3dd9366431de7fb2767f74,7346d7fb614d28238d08e00a57580090f688cddcc1ee1c09523b37c8ce88d423
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r01.json,2e75a702a117f7b5ee379b16f9a36af038b009714d15cec6ec89fef5f95fd169,0b92bb89d8fd3627d6a14cb5a070e841783881bff703ba3ccba997238537836a,6d61d1e5d3114e1be9cc7ed1c6e6d0bef2bb111662cc15e0e24b810ccea4d20c
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,9bb34adae8537f1a39bcba341b76b59a1acec3138f678a495872f5fef40466bd,10618616c583f5480f8894edb1556f05c8a5d29d8e008fd29dc50edb2b9e94d2,0f526fd84c20a17eaf9572af930ea27dfc633db8cb7bac98b05cff096d1c847c
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,4cbdd4721c8ea02d6f8cc9a279139b3ad2e87f781ef1167fae1382c3a245ee15,abbd51fd8a64698229bea78449c54ee2a393262e5e6ab40af5653f8d5ef4d501,9e69057d62630ef46e3739ad4e00cb3dee2084ba4e6475f0b1d4f68b51351dac
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard-r01.json,003185309e8fd91dcbc326aeda56bfb4f005a0f5cafc215134443681cd05d4e4,982b7fcb38e8a7db7b10955732472c3bc5fb80b254c311f34dc241165e821b40,5cdfa9302ce2dfb43100fd56349ba2cc09a6b8a650eaff8e9ca66e34a6297f0b
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard-r01.json,c216ccd76a5443e57bee5da1928f29b9a9e2ec8c75c8e74f5be5d92a5ec8b08e,72c196a96c43cf6d095c46059ac9b705f59ffed1982dcb616c5404e9df6f4993,6f4fc62c7936fa55c752aeef5cb09f32a2629b783896ede874255a9a62941ca7
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r01.json,6237d07cee30cdeee0c77fa0b119a7905eafca68f0e403e3188ef1c87d015456,3e7528070525d808f6e2a58c67ca3a853bfb600ddffe6fd961ea65c2a58ae326,fe039f659cdd6e7daa45ab23a041929e00f8649bad43964867db18c490a13813
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r01.json,a65f08eeaa6a85b40d689e6aff3f98858872b7ac4784c9aa51343b6073567616,06983f452c641be03b57c0cd567dc29a6ee5cec728c0d2479980541dd3b7915e,65ea0aae85726d09de8a268e516c012a15cf1156a254da0dc3f7b429f8a813d4
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json,5ef7f6801b4f3ac2c3b14f8ea1e28cff5e60a977232f6e63a4777a485ebda186,c09b59a90cd1dd09f003f63ce70345f35197e92f31804431c8b531a2ccbe80f7,2af306bde2ceee3e5e1dc6c33ff46ff568cab1701282f4a3be89f7d316fc968e
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json,6da4ff7be7873302693ea25ddbadbdce2072918aa7a8aa2d3f354072b0817226,5d45c9c3ee1070f1672ae83aec5b8291fb9c83fe5e7032040ca30d8cfedd1dbc,8d97776acf4f47b836dd4a0d667822267b53138e74330af23c2ac87b70559a13
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,9cd64571c793609ac760ae0d2f70cddeac9d0395a6d9c49a60162cc7b1cfe8ab,8ff740702569eea5ae46f957f2ca98b40d272879348acf55ab8d2d43eed6d333,d8940284175d92c73b8ef16d2a97c735a75a4342697e5fecac71358a0c7fc9a4
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,aca877faefc05501bfa069542dc382470f3792851bde153398250e2e7a37d09c,fd100cf0c2c03c58af9b39c479b613a9c4a076bdf7e7cc44df1056e9104174e0,ce50f15626df2cf6c242903916c961f4ca78b0a11f4bcf65d489fa98a434b636
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard-r01.json,17564dad6dbeee4da987ac0ca98f9d1ce4f18a51d64647a99d9ec37ec7495778,4b12566230b228bfc28192e9c5f68f17dffb6a7b7035e35367836a35f393b1e1,858de2e040e87756c11d8826921c01d290345b5d17bcfa059aaf9d4230b8a453
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard-r01.json,35b55be43ec761c309c673601a2ea95f24bd0f490fc2804d746dbc68036e1d92,eff4373acc8f0fa23100ccc9b17fc3ccc07a4ecf6cb890315f7109a6affa613e,bf915835d8b93bc1fe74776626b36234175b625ad45c6e8dd89615ca5b65c608
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard-r01.json,607bc97e7b34d458b9a3853a267e2b7afa6216a50135974fe0496f7c32c01b0c,252dedccac4d519bce5ef5cb59ea232ef5cf2722c50311000692164313ada3d4,204ebbcab423d09e34550920a00a6cd983800dcdd0fb39307a8a65f4489bbb39
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard-r01.json,14a194abe7f9e35a946c70a2d92b7a533e79992682657b1f772302d24e6924bd,db0bb2d88288a8ced1b29a8e2f0129370a1443a745ac5b0f071a7d26f7dd0bcb,a174dfdcfa49c294a5a281c69b2641cc0cc2b346fec360104054ba7190a8e87d
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard-r01.json,f34085b98773bb351366ef2e7994c9e3c283c97672371a3c10651dc4d293e027,f8dd0f975e24e14ab2bef81e1ef796ad5c80c2a50e2393e7597156b0d1536fdf,565deaf802308a35b864f4fdf8e0b27e64b75659ad48098fdc50219725c5798f
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard-r01.json,0c72ebf9db3dacf59b1dc448e482d447b227084372534e94e2269543a937a42f,e775b1d9b2df00ff0966bd7e2d3dbd77c9343e9ad094f430bceabec04acb9d1b,bc8d76d2ef70a7d72339c6636dfd3a5c0ad40717a19b552d26fcf8797289bde4
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard-r01.json,5d876d1cac50d5a2682133874ed25422cb6f09f1e6db6a6afd91280a8b68d6d4,f3f9b3b542e7b50320101ec56bf5bbfa3bc69b51748b6074bc0a15ab05ae8db1,623841ce86d0e502bb5bed22b19c9e7f1274fafcc82f19b0664435005bebb4bf
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard-r01.json,5b7ee7abca99787808440b177a870b211253eed72a091fe0f598ca05f0b846ff,a09972195a8c94799e50ad782d139d4b4bd6229a3ad2bb4cc67a6cc2913aaa8c,4bfa7f6fead7b850017ba10baf0490a11bfee7727101d4d1f7280ac431b183df
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard-r01.json,d8faf1d1a535eff42e6aa9f8502023c17510b1b3a6ef88376739b58f068f37a9,db5e2893d00a15e2778805bab68eee0fd37380a603cf00ba3f7a1ef560a170bb,e773414945d3c5d94ea240413a6ca0ff02b774477006da932288c29ebefd91f3
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard-r01.json,ee78371e103549ab9b272553d362dcad891130c9ad8c54bf577cbd2b5be848cf,3b6b17c684ad3f336ddba7791c6253f05c3509ea66034819d091376ef73d577a,1c450cb2014c93bbaa2e79a2c47922ae5db2c0453d2c47e341f20a03952562ed
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard-r01.json,0deae3c8546dc09fcca6ec1796e58d0210cbf08a88e68f10d0a3369d8875b63f,79cdfb8ce52c15196d60fa28285c5d6ace6a684131076a35be6312906b608c6b,294bea8f4cc3217540c7f63dc934881582a6c496e9f86ab32e1f95349ddb011f
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard-r01.json,187ae880c398ad66d7b177746e554b9ec6cc817c3c4990672fc8d85268fe57d2,f398268d826874ffb2b9bf7bc6fc2a7dcac5a08a49ff8d6fa390749893a286c0,b05177dd9feb00150768bae2dc937dc963e2aab549dbddb52845816e4cddc276
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard-r01.json,b900e709a86a46578d94d10236bf64c4fbe0fd92012a5bec802438c2bcbac1ae,23e79639d65bbccf46a398e2624e00e3e7a7514e0f7af2fd456054433d7029d4,2fe82ca359542768f9a88abb078c3260e6f075e3d4f5f03b31595a8a45e1fdda
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard-r01.json,23680b45f50591d471bb862ddbe81bea0457d7e089b8ad2ccc99228f3f420a28,b780abde7a83dec7c1d67ccc921a363086c3edeb30ac22ac470f9c47f98d05f1,889ba1742cbed64823f42389e61da5120880df52eaa6787c80da2c31ed56dc4f
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard-r01.json,dcf63d90bfb21d18ae23de645d646c9bb8ec354dd74de39fc3ca39bf2374854f,8ab2d8ff43f89540402c5f395454e23a32486153a9c9a5bf5edcf63ce3ba5a1c,1c207b17531658e2eef91cc489789e85a78b3a4f8ef2fd8c117b0d76306e16b6
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard-r01.json,ce08479cd3986405c85fd6c172e7da2b69d6736ce3a2984569cc41f42a6d21b7,40e873519d94cb405a4c05be4310b64e7e3dbeb132ddefe880918c73a116e873,1d5a2d2559e9b605994840d4f70ac7e249b68401a685fdb67fc4a39c688211a3
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard-r01.json,c3a643c63ff73304ee6c9c80ff972def90d61e6fea97df7d00bb554814ba4938,a76529b5c0c4f9c63f28cddd64a0af9f77a7413bb397df0977bb7d6003197092,6f89e09d0e34a00ceff9905cba46e8e01a020e61da8ad11bb315a9a941dce376
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard-r01.json,1dac902cf949000ad53cadeeb2c5f66093897ab9c55297d311eb1315fc283f56,d87b59da6ea8992645add15df4d5a872490df4268ceec5136334b4794d248451,a2c193bb517f7e1718365e141c4dfb15f845bc99c0d6f9be606925964744e5eb
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard-r01.json,915bc6fcc1ebdecea2ef4586cb5edeec75f4dd8f8c0d35160f17a0890e07ef36,4e8358653a9f36113cf9c42a19604d958bae22f94ccffb332e3bba68bd38de1b,f9257b99e03ec49c972e5efd814f1dfef693d3d9354ffa982e859bac9661a8d5
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard-r01.json,1f93fdac94ddb575fe3034e773025fea3e6f75e8a589a81d57bfb96c3e1ccce8,b6e9fe2124cede0968d3985f5625391e8bfd7ad0adaf3e0752863557d8fd6c9d,cb30b4d9d116d6d4c473f8b1039f3d46b23962b08e2b3683879bc3c4499978c1
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,b042f621871153fd3ff1dbfd15334c42581baf435cfb59547d3c886fd8aa4876,9da8228e7b15aca8826aeef87ca6dd6049b6536d53592b59f1bafb9f02ba120a,62eb1411984b5482b6d073fd8095fe6c61164fd654cb74ba8d5cd461236c674e
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,85bd982ed39b52d7d72502d2f4eb2af28af2201bcfb59c1fd31791c98bdfc5f8,7d15f14b105b82adb25613db842dcc77d78cca4b355f6e070eb81532b38da47b,3020aa38f45489be1cf8dd9601b2c0d402ad50af8dcc7779a68110700413100d
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard-r01.json,0aa66996925deaf713e1b1654724f53e30cebd4a699529e9235681e3e8022c5d,ec926572f50cdeef82705d459e83e99a9e80b889609a45523c2d684fe3ba95a1,6da17e582f2b30e0ed9716debf43184285e6094563ff2d60e277238f6792e1f4
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard-r01.json,9a987326cbe1570d05407cfeaa0bdfc80e9da8e5b5fffb4d5b998c8c1e194546,70d2f24aa657a7173fa00d29907ed399c53a36a1de3ad706db2be7ceff236204,46bd680e5a645b35692ad61e7b0c430a905401c15aef6c7ae9896366620bb5f8
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard-r01.json,399dc417dc470235f88e80d796529a54f5eecc1be56111799effaff3943fa710,0cb3857eb72d5be7b5b2530fb518e4fd99bbbb6bd4464db9191ef1a70f6f192d,6cb09f5ceb3e6fb35493981a5c147581aefb1889b9ddfac3dc18fe78c6a56a26
   cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard-r01.json,2bfe2ed5753cf28bdf4efb31bfba3c79c00dec13dbcea83a7ad03738d9cf16be,cef505db93326c3ce39ab4459ee45cb577538d19f64132cff39a7d3efa7e325c,22ab10c12a3812560d11f5a3f7bbe16341920a3dbc45e5a800c0dd24b58f1ed1
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard-r01.json,ee9eb0a6c3fff480fb0b198391e0d2dfd116dd833690c4d4693a8ffd365f732e,6e0a65a0bac4df1ba88f11e6feb939f0a00ccca71ae08811673706a934113f1d,3fc63b711470531f1d6bd63461b701a99f433ace4207e4584018196d74111c37
   cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard-r01.json,5cd8699fa61eb154359039d04140dd640c70f441aae128b50a86609b99c12980,8d53953f0514e125861ccd2e6a57ced7502fbda33aced8be8939279142219b0e,46d2f4f359990bdf4e758116331034e2dffca5bf2ad01e535c13231284b2dddf
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard-r01.json,f4c459ec0a6a412f2dcec1d2c7a8ceb395b561fd0ef2fde2dbbe1a7c8261bb4b,c375b21ee676d5ecc35216a2902bbf16f7437bc9733ff8aa6833b60ae532fd4e,d0ea3c5826dc408ce071e41facebb369c36b2c4ddaf8e2fedea1ba755fabe4c6
   cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard-r01.json,c2cd2c9c7154d21a0a3240ae4ffcd8408184e91b32ba5aa5c3f358937cade28f,0c50c73605519edba9ba56f46bcc2322b721ead151302e379eddefd49ad13ed8,08ad14aa5fa726283bdcdd1156e3c61bc0617dc7b434eaac68fb6ef78e9ade85
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard-r01.json,7bc134fc667b910a61f7c84225d0c2ffafa568bbbf7be0d789068e5d669947e3,91baf54ca40f289610414ff09d2da9c7831aa2dd9351cf645cfa3d0a8cf1a87b,b0acd5f0615527e3c9cfe13919d7a7e2a2e461ce04153e5b7bbf821c8c63a405
   cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard-r01.json,663060220d0bd60e268f3fc19bc50193c2382f4066b65d2df073893191fbf021,24a2775775d16d2f9884ddd5e15f644f28254601102274f1a93e7b25f035beb2,f05518d8c4135cd4acccf31b28c3f47a4037aeebfb1cb93aedaf87b350198326
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard-r01.json,5aa133911e9b460f88964f991b0c14030235f41ac465f5a3386b4ba26d528c16,6b8ecba3316cff61a6f11ba9b0644e10d6bcc5d7d6e33aff756d364f48728fb2,edafe0ba9d16c20292f82d0613f373087565b1d6dbd40c41b1899afc414d7dca
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard-r01.json,29f57e5c980149a0de86e0d0745d1c38336ef32cca54bb2cd7fd3764256702eb,e099b04053fd0c1176441bcd756bfae8fd0710d6f396f38211c094c04b51ce05,d07aca6464082b09b8d16769147a4f15c69e5857709db08c32104b171031701e
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard-r01.json,12675bd7d1bb94f940357d6f65d572a757d474936ccce63487525abab9b43c3f,d3c3b657cecaca5972669485e8b644fa5fbd37196f5f10bab94151353cbab99b,0f10d7e00c612a71acfc9e8df0701edd3c9c13af2d17fed1df563da145506de4
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard-r01.json,fad115e4f76390c33c860227e24d3d35a79b29cc957b7905b3ae79d39d7033a3,c14e8231e3c670945b371c950ec1f67379aa26c1c0b5c8d87acae2889f4523e0,15b6b1151f820ecf9febe2882eb0173effac4e3dc59a6ee13c57e075dc8b42ed
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard-r01.json,2b8c44f6a60d08f9c076d8dab7f4b997aeb95c602e071f37eff21d8c057261a1,498ab428cd6c57c78c94e08345b3a340319a9b020f0b28fcb927e0c62ce08d6a,cd6db915912e14c93db87304181f72e95f7e75738f074489f4e9b6d20bbb1b46
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard-r01.json,3518475c10dbed4c9ab9e0e04064c70181cf255659b841799b31fd3f3eaac64b,077ad69aef43930f0d56c184f6d9a179d2354c14a4a5c664ed5b9434f8fad80d,edbfad9bc55322955b020e97256eb9c391cf558cb4ac329e4536298313273c2e
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard-r01.json,e70aa21ad5da662d3b3f0a8f020fcd53e6334ebf345abc6a2cf41c1ab5378c99,8191a4c75134e5b363856eb89f61b336d4c3947bacdcc1ed0c42e1f95dc931fa,71280b7e9fff4ea6301fc595f00cfb9775c1b0e7ad479ab0bf8034f45a744a69
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard-r01.json,a9eb942df41821659f7794ba76dfe98b592f75516e62668a99bc527dd8fd0b0f,caffb348cfe059d437f4cd4889d50756ea5d1d026669bace187fe248830ddcea,8130e4767d25ad60f48f0f525997a63ef52e146477f0842ee9d3580c228a62d1
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard-r01.json,db77aade81336ef1cba3b23b0455c22f43add835bf8bd1bc0abba824c4cf1280,1b76eb2eeadbd2d14412a27959ccc11241c9de72d47f81a780837e65cfda2c0e,24b6b30cc60ebde37ba7a0495b05d85e20b744da341be986f546a15ccd8fa610
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard-r01.json,9c5c4e90688bd9a22b0cbdaf44ab7140d99f5595c012309f0033a34aa044f65d,1de7c56cee8eb4d8f491eade57039f19476fc28a20aaed68d5e69905ac9908ad,f472427dc978ed0ca97f7a9181e522886d5edef2edb87d13a546f0028ef04229
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard-r01.json,2b78a81917ee19fa5ca6dda7610d030f77698e2ce968bddd40b185bd4eea02ef,b14f97089e51172601089d0a901f7765fe17f97d7e3448e0382f4363b0244b6a,247a3482ee09bb91d03fb9a2a695410334a90c6a70a38a71fc8f4a89783a58ef
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard-r01.json,9a54b826527f60cb8ff96034862346b028da43b4340cc8b0ef1cef3d1d5ec706,9ac603d345b1b53306e7879c62403e90f30abea2ccd77e5734ff684586099027,0d3fa9c776599d0babea7845cc922dd0666220deadf51066c08952d41bd498ee
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard-r01.json,40f265d9ec4020e382dc82674f677fa3ed5d63f776b59a57a0c3da9be87e04c9,52ad35addfdc1486a07e53b910ec130838b8dd1445d7c9d934a4dd0fc3943a82,8efdf8b0c671bfb7fd78120604562f41a0f7463f91e0e5e1eb55707b60c245a4
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard-r01.json,f37c180110bfa0fd324afbe671bdde27cb77818b4ab479cf8dbe35b52c01556f,0eeab5217f8f5b0c28f7b5cca91187904e98d5f214c0e0f45bebaa32ff544543,3dd128a30aacfb5f74cdcd703b33f75a0047418d62868430e3143b8bfb19fc9c
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard-r01.json,630ff7c88bca434f88ec7ae7fcae058635c015ed0ac1212bdf03f368e6005d27,fa210909f2c4cb58e3e38885a9306b13910f477b3012d0354478d20b3f73aede,dadd90358d958004049542bc9dcb0b2878d2a039af1b091a2c361a96b4959525
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard-r01.json,1f53bfade31933dd30e1786d36738eb015d524644ff8e20c611c1840c2fd9263,87eb13c9f1bc1fad7446e2afd477f62326a2c2ecf46dad2dcabb4fa22a2493c3,1c86eb1e913ad3f9b6e21bfbb273a134192afd5713b03efb6adfbb1c211c8f09
   cpu-release-sparse-solvers-none-api-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-sparse-solvers-none-api-fused1-host-standard-r01.json,efecd6474f3c267ede9057f23115bd938e7733070c64a11d633a75facd5dae0d,761ce3c6f5fb5436d3f3888f45746306fed8d3a1bbe23206eb66f9eac01ead23,dc0b37e141ce4288673e51fa80322a4fa66701290d2977bd0546c6cb06e592a1
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard-r01,build/main-performance-20261017/baseline/raw/cpu-release-sparse-solvers-none-prepared-fused1-host-standard-r01.json,6600640db3300b3dfd5f0cc357597cedb992f66a6317065c3123a3e2cc7fefab,f7bdb3fc23e08e928965a8b1105ac7e6c717eba1e2e062f640f6d981676eb37c,69479a1b77c36143dcd025eb673381cd673fbb17ebaf2ac8c5f86ca0c9ffbc22
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01.json,bc111e94f9589d8da25eef8ebf7e90a4070cc943f86417287cd3170c283a51d3,f00e60262667adad888502a526fc25f91e1df8b8af5ff727ba7b61341f9c3cf6,9516111a725d2a2e190a88d1f049ae28accf0c08aae7247b6e815c23ce964679
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01.json,9a4c51c78f06f9959de1452419f243a6ec19e38b7425e0b249315925fa2bded0,86711260b937642d4407f96fd8ef57418ce50ec1a62a4ebd4e7192c4456f9fca,2f078ad27a44819f6a55a87516fc0620299ca2e0d0aa27e6e835d0fab15216b6
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json,cc39fe8c2994ec9e9b9109bd11b6c7ef0f8a5380694b17c57f815ab32249ab05,f17f2d5ac94efbe14425b6813a89fc30907d696f752f227cbd459f74d5a50e61,4365d2facb560af7c5b4a87432fa08b1f0a574b3ff8fa757d07ef358b6df9bbc
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json,128834c58b8016b46f8a9b6ba3a72facb16a95d478558f00396157f30b4f1bb3,d135abfdf70719efeebca3b13a46a88f5a01c35049cb1b81b6fa047d30200a51,b8e9ed4c16659641625a59e65d0f7c2792ddb6f7669fe4aca59ba8a2a27d54a5
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit-r01.json,1cdea9502ea5d97d3f59b959a639f50dc1a341c06e6f982a0231a597bebff806,80d2724165e5868d9ee680d56f1eb2419fc84ac04d745ea3e221bd401583f99f,4d657952b0f21d37b87e4320765ffe61d352fb0f92f953c69d6c5046527579c1
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit-r01.json,338ca4a31a5c54ee3ed239b1d715917e1f57f1ba42988addc5ad6afb17ee1d0c,2076a2c479f05358a2c6cd39412e647d40c796316efe0a241ecdbc0862f4fff2,4d6d33131296bbad0cb9e595a9187956cc8f84af3e52875944e536bf2d0fb03a
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r01.json,58d6f5769fff2c3d0316ad663bc23b822a79b65c4f3c87480b56b3593a6feca5,c487dd8074189616ef76a97e809c63b30069cae00cb2eca3887bcb127c1ee6f5,f1caf5b75e8ea7977e297f902932d6b1c910b87570bb5e52c9516f2a1ade5986
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r01.json,403f902f99bafcd0a32298a2608ef7b41d23152d0f173fabf143aafc15dc55c9,adaf6a5c43b74394631113f463d5c4196d9d5e9cd949152a31b2e95d0569d938,0741c3cba1949e8685b2118cb394ac6ada92cb6007a8da0b510a409ee6c718bd
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01.json,efd29a10afc619ba6c4d3b7476b2439b5c01da58658967a38118af6b65df6ee5,76bd3a09873d224a73dc437b8b540646137a5e2c5f75cd915ee8535184a8c6bc,77ca288b92e6c8ae394f41243ca3d811e71a2149c1cbedbdcd964b60284f2317
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01.json,b28eec6d0df0c2ccf015f60f7eb61fdab76fcacd6cfcdcf76bb688a631e29025,69629aef5b6fa8e7ee0574cf6e95656601112806c9a5ff57d71bc6a41c66cd21,30a2ad41f800a75228d1b98437867ed1e94d8bac519b2a98ae9ace711c162302
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json,8e0e1f2b96eefc091cec98c7aed808b95516116f0b1ce3e5128cdf54fe79e570,8b3a17ee074c927ac007705b25c20991af2610cbe905dded80af91f7d4a24531,ce8428d203aac54984aa2dd9d1b781718df794011ef2828f37d8c08ce0b0b3b2
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json,0ea1066234cfe3ae363149ffb73884e647bc225008871fb2c13129a601ed2519,9e6f750c225f7db625b9baa2996ddab215d00248ce9dea8fe2eb8a671f93d861,9b4b3d28d9045f5a8bb3f8759d13f56190206790d81693eef5dc36ddca915cd2
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit-r01.json,a52c5fa6deb7aafbc8f0547d0318c6ff5164848a781fa0b27f73f1c73eeb12d2,5475a9357a37123aea62cc95739ce6f97376e14064cba65cf9d2e690ed888230,93b713869281e25c0783aba48458b00e5daaa68d49fc2256c46245dce3af10c6
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit-r01.json,b2a35fe84b48ef52009d361467951851a45f5f2738fac21e6313967ec7eb09a9,21f9d1f7a90c19f61c69a2e180d3344ceadd8d92313a47f6067a26bbd01e12cb,34dcc303f3259db3a02fa0dcafd93755311ed98accb1d4d48192c5ef921b1ed0
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r01.json,2fcb9a7bdfbf5c66af9f2a6f54036618a96fc75e80c9e6f3e1b739418bfe1ea3,69accd9ccbc317a05044c503c884ed203e8f69ed596716dd1e97ec3d1df375a0,5d2b17ed0ef8cade97918517faa48812a325ef9b8aea6191b8dcf65038e9bc73
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r01.json,f86432d978d5d2253243246c3fa55a14674173751119f530dfb842a4ed59687f,5f970afbaf6b75db74b716060a9d1f6b719ac7e1eb2a798184e611b267205563,cb80794cd6f1b6360460e7be548ff97781e310410386272ee45828fee6ec6742
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit-r01.json,2f4e453735b2ff7d0d3019b991991500db3cc784d753377622565149ceaadd1e,2c49a98b018ff9fad7fc8ff83e8b19034b4b861d6c3e23d5a2fcf2e4dc57dfc9,9148f9a5e830aa44e6f5b1c45f0b453562cc6370b188c22483de8fb3d0aa6e3b
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit-r01.json,465a62f669ba08f942dfa5ae51c41d56787aeefb6b20c139a7311f192133f391,d93728cef17102b374cddc643e66f692905fd67ba31c0f9c6782e1a5d054dd30,a6bfa8ddaa1bdbdfa45f9f896a0ed547d2a9e7b66fcce0dcae5183d6ad24f27a
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json,baff59483c461b5dcbe5de035a689d82f92bdb8dd348cc7f36ecfab9c5707c49,095763e887a088b0db6cdaffb81753437ad59055b83a71dce22b0a3ff2a936ec,0c25816374e7c85b0005b6bd230da3a07d837e6b0bd559328a33d3e9b1cbefba
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json,9ab145860afd388b4b6ad0c5078683fcf432be7181ee4d62ea5a6aac8b28bc6d,e0019f1c818a1c8201ff2161d02b7aee492dd8e37883f4f8d048db5b0f5fbbda,2d1e30f455e9f6805eb092f78808629a297463bf4c74ff5d05a144b6b6c54532
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit-r01.json,c1dc95b9c9ee328a8fca0343fc4db48b543aded10667481c7e23bf323a7796c6,d96fba958f7608b7afc59b0000285acaa21053b7b44ebd30485efc42c39a411a,6152d3d900f45084bb100ebe02695e4c2ae16c1bf7604217dbcf4c3f1a36420e
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit-r01.json,38b23a69c3884f554741446089ddbdc0a26d7c62420026260788bcc4a311c6be,1d6053f0a91f3c7d04d8e4f6e80d565327fea4298ee9c48af6d385ac46275445,5121d8c6ee78133f6674cda5dad48154e9ce78f12f3c2fea51ddea2b2c6c3253
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit-r01.json,1d9dd09f53b50a72ed0c39cb8b42bb4eb1e111a0ea2cea974d5ec8e36afc06b8,8abfcb5773831d72451597c81d1ed96d6f6e96a92c7550fbfb950c612856ed5c,6f302342aec31ad5ec1384053b91cba9ee6d5086c7c3805fa233abd3f163a437
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit-r01.json,e567691746c6c338fba88f9136b28ba73563735b6b24bd030939851fd8a6395b,ca56374b767e996133b74ec3867ba5bb23d14c56115f129a8c4d928e9a5bab51,df3cad1f6d3f720000182d9360cf669495f1773b35713f1a831b3a977f88309d
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit-r01.json,c1605fbe63e8e57f18668739960fc258f5fb7122455ef5fbadbf12493e7e205e,ab9352fa94c8736b1bb350d8a1f7804efd16c5b16ef86a4c857d3475a575b9ff,0fa6bbb9d8ef77df63986f889e062837106a711f4ff60f8634be7d02082f82b3
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit-r01.json,817fa962214937d30e37ae18cf918e486ca1b9219b68c187c1f097bfc6b4b759,d45a0a448cc9b9ec660deda974d42d41796bb6dfee67f348c69251e81d378560,8e7fd01027321c51ce62dfd94cebc2564bc1a6fb211130ebe3fcb6db12b0a705
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit-r01.json,1c964c96d00dd844ae14206a7b6893cd4f12088b7c5dc48baca5736e29ce0641,0a889360a33b6ae155bf7bd0aceb6934f002663deb8ecb719299f9d85f190e77,42707651632e2774e7326c52346cc30d0088152581563ad70145a845eaa26d57
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit-r01.json,704bf9e95574f319ed71c1b338154e070ad86e7f56b8b27647b1356cb7305506,519ad79260e322b94cf19cb501f2279bef068f00279af456316667db74ffb0e2,5e435962039fe4f64b41788c07c6695710d4e98f9a25a56d39b54e776aa9f51f
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit-r01.json,bb303f4d68ec4ec595a68570e456129e54843683b4d81a0b5340430e88b5d307,1aa0d68a515e29763eefe5d5080539489133be043932674f0de9543d77aefd0a,a43f3be7944567e27470b729c702e23128cc42c828346c5821d8a391ef654d90
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit-r01.json,149faea7d3aa1f6b6ec197dc67823f11d2041adef506438bd7c88fd5f4850a90,9d2733202007ca548b26163a0c070e06414ed3c8d5637849e8d94dd22d02806c,b01afea4344c748fa698ecc8bd7d4650bc789e400d18520210dd18267d33eb42
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit-r01.json,c63ee9218aa7050c13360602da7cc69ce86aa6d35fdf6113fb7f6ddd75d1f5cc,24d110a2027d790c3e98eb91dc20c3d3f662e828662cdf515e69946650e9c183,ae12c98b522eafc8e55c57e7a9bf1abc026c773bb46c6ad1071ce37ddaad5498
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit-r01.json,c69622eced166ff2b4a064ef55bdb9c7c552056742b631386a138a0c8cc74fa1,ab9b330918db29c376d3ad8e8332a7885bb2b48750fd6e50eae8ee0c4a89225a,fa395eaaf3da75f1a0715eb711f0bcac2f6681bffc1dc07050960557f53bdd04
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit-r01.json,15cd4b40afffd1a24d1e935664d66de7e52f8a0728a04c64016fb4b8b23ffa4c,06ca765c84bf5eb4e3069b3fc0f41474c0433c891a81bf71fc94d5558968e5aa,bf7be411a8bf1be32125dd5df392033f4165588f54b5d3589d6cdafea729682b
   cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit-r01.json,172c5f17b5bb254d0b5ffbfb8b5ebe4f67ce4aa27e26ac0bb5b4b6459d7fe357,25d2c63118d5728b7791938d93ea7fe0e869ec910bb4b567bcb5f009e178a39c,5d7433f37cd7f03a91669ed43ca4945c13b15cd482d6e563e7cc4c0b25eb6ed1
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit-r01.json,bb81f2980947da1fd5fc817d79dcd6206f88b74d0ca93106673b20ebee23f061,ff82a971d02d20c7f655159e7d9eea90cf02409db29edb4c543b51d857a10210,8c46adacc374d6c3bdc514b6886192cbb7fbe0ab39e963c4da4cea38a9f58148
   cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit-r01.json,99f96a75ce99ae739aa87a0a64137623b145a945d330b4b4ee741ea853fdd3ba,60b33579e1358d36f8eb862c1bb162db20788bc576ac45d737e6907a2d34db1b,5b5896ac1f4de99a2e7a5aa33d6d5ddee93802069482381374622b50fe9d8df0
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit-r01.json,82053aaebb15519a2734239774864e66405271f735bf76698221f9ee0418918d,a734f32945f0f5a3977a5c0697942ea8f64882dfc2619adeefaa0f6f663429ec,b577c42b23aa2d6fb6465cf9a23f94b985bd2e244d5bb8ed24714b82b2ee7df7
   cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit-r01.json,119f58076a4a79b8f58d8948c76bc8150359382e26b4168d0f609b65897145f3,fdd26093a70c3866fdaee1f1b874f00f276f4af3d3ad0ee5db83344944d2d33c,23ca8e63fae2c0769deb55501fefcbf08a6692704723c24aa18be5f6b3601240
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit-r01.json,2aeb18d4d3c41ebbce47db6df29c9566fa7b423406da029f9489f124d707c41c,3f8e37c75a3ca9b5972401d05f1802a029eed1e10a06a43ee20c14f8c4cecbb3,50d45a3bf12bf33eda10fc22e32ff70414bdcdb4593f7ecd5cd34a7311f716be
   cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit-r01.json,362b9cbe4136bcd1d8cc9d57ec75eb474a19fe7d297e3c79af95a6bfb57c66a5,c072bf75aa7828766318e7aec771924fac8bd40355730b69f29fb4e62f380a7d,9898ed90aaafa37efaa655b20610ce2d06ac7c79e64da5e38f43587b3b1be14e
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit-r01.json,b180158b933dea8bc2033bdc148643e1d70a50e3eff84719a5225bb12479f24d,6c8e0d22d7d456e0f29b11b093f587a46fac8be94169baa4e88c810e3e1b8996,696e556111efba7795a662717075bf50953806984b619dcfa0371d0f4d765196
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit-r01.json,c657bc248705f98019ff4e4af9a14bea2c6c4b05109033095a1dcee7dd684e91,bcf1611b13ab612d9a3fca1f1d34ab3c934a02f2fbbb11b25804e33b05b49268,a28b95d1e199ba8d241c8c242a642e474253339df970bc3b39ce4a833001edf3
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit-r01.json,a0b36ad7ce527d79efb03e039d46c3eb196039ba15051da9a164c270296d305d,5a78f2ad242d43eb5f201e144ab1f2b0f63f7dcde7724fd1c3d49c496fdae05d,b2edf6ac1578c8da6a9e91a08676025ad9ee5e99989c8d4035bb132e1305ef14
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit-r01,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit-r01.json,dd2b9f8966a9458b29c236747270c09d0c29dd126172ffd1db779cd99e709f5d,437770859dd7cd180a8a70af3f7d4823336980288c2e80894c2b0ba5e44c24d3,85ab1cfb474960a804e3df5bf35d1debe93fc63a31b10329de5e4f250492722e
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-explicit-r02.json,20f295fff9fecf427c78931d4850e21595201ecf0ef721a52fdd331c40239a1b,e2722a04b1e939cbe8799d76d64c7a12d47bc0c68f0f267ab687d2008ac0dc27,66bf37121d9bfe35f647ee8257d578ad9ec90d5f2b39da09f404625616511953
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-explicit-r02.json,940ae1bb8252596d0b95b8f5f07d80e8fbdeabdd194edf053736d0d2277274f8,8c105b3f6b2fd55ba58a70ff34aa8f5fb1ee5f24f0466e7abb9cda2dfcee7af4,c394c41fc17a6cf2c23882aec7640b6505c12d9fa4795f47fd914572da01507d
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-explicit-r02.json,906a47b7477c229f60932d8fc6db93b9529d72da5b8bef0c0e121c2cad60f0e6,1deb7dbe739ce56b8b2e416c868667674ab03e72742df0b115a793facf40836f,11673fa07277c78e743a373a75daf373892093191977bbdb90566d558d28b6f7
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-explicit-r02.json,4bd5af09d18ee2f0473932a17fb935404cda699af3693e9f53a61590c4582d04,138281ff7c36a58786b346724e3026f3b96b0514cfed92ba062efc32c1ac5b0e,970c42cd64d6b149954614e504aa530b2ff161e84c691b3ac194eaa6f05f94ad
   cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-explicit-r02.json,959babef2b6ee025d34d7c6eca5636b92a2d5962ed3e51d5ac404efc5303d17a,e0dc8c554485419aa91b5e365681ddfb15d8a588ebb518e9ab9b4204391e6088,4bed8b8bac1bbe161219fc71f21c29f7e58c12422e0ef5d2b9518cb9cfd50483
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-explicit-r02.json,9d1928a40552948b5cbfd2dc3a2437188301a2c9e5c0c33e1dfb2a401776e58e,23ad6cffbbfbaecd48a3e906f01d1442e4f8f8f7acd2e4522a2a90208ccb2dba,04e478e0f25851b90f2d80f26a0c00849134a9c77993447c7417d7a95b1377bd
   cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-explicit-r02.json,13a17dcd05a7358286e981017539990a508fa607530023e15de04d216453ddf1,31ae92e119a1b2847ceefeb63c030d427e8e735614007c9055968bf6b44444f4,ac2789eed8abbc354a57c294077095aed835301ec4ed5ed2f73c618a658ba86a
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-explicit-r02.json,1fc5a70594cede44909bb142ce5b6925d013fa459c3cacec5f929b5b839dc2fd,1e99dcc2fdf0486a74d4ff31901d903c40068d08bfc43d9f7d4a363c31e934bc,871259280c27cf3c65bcef6a5cd4ec52dfbfc3200659244e7f9608992c22457b
   cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-explicit-r02.json,e2a26fcad2c61c272b2d80082f8e11c82a969ebc49b633e69d3b5258a1d17dd1,503526a66298cab245e6ee69212ddc31029f606aae4f7b674032be3be01c4b3e,75fb487c5eb1cf765fa9342f8f95c31aa185575bc33218a02a8e19ac45187f68
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-explicit-r02.json,adbf0270e56233baaf231227a5c1ecc25b9334f92fa9d8a9b869c000ff1d856e,228b0ab0107a6f5fef871b9f6aea0c3d9af9bc647c12bf0c088c13e3937b67b9,37b397ce208706d33f5c7b1af43a9bdcc75c511edc96ec4c2d87ab8f5cc9a7f4
   cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-explicit-r02.json,9cf9d633cc6c72951bc0a42188438d3f387f548da9552bbd86d86a687a6c0b85,210195205a22c4195c395d4b29da0cab0053084d965a638617083a63bbd4c2b1,c09e58a9320736774c7bd06a88a985c06f08ce61f842bf21e83f1b48d3e2a667
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-explicit-r02.json,7fceaf5c65274d7ed5425643a88c41810e08a4c58e9a97abb01b6e0f69a66204,2c0ebcc903405ddabf22dd863cecf446373bb42bed7e5f67585d5770d3cb8705,635b28b14cb669e16f5b5e47ffe4d0a8c5ec76b51e37ea63724513d689c5dc8a
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-explicit-r02.json,2db7ec04c5e9b1447777879b43dac639e3721fa8a3b5f5a4acb9cc292dcd5397,6f0f61a76496da3861cb7e10db304f65bfcc65625d627f38d417283864a85e6c,570a877dc0edf46a42ac3511fd4e140df4d8553487232a5bb44271f600591167
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-explicit-r02.json,063cff6a3ede8c68723e6a6510610ba398d1862826d578348006c8e3d8513d79,be8ffe1d5aabc1eb0e59ac35219e545c8f90fa498af1a4670f47eda2b8ca5925,d35bcc70d9362267951c98662c342408f1d1455d7800625ee122d7d77b694766
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json,43fef2b526b315f270e3760c643761717f55a1582f7424e9268f9ddefbf0d05e,f15d4614f2e3e80761745bdd17c49a0d3e3e990fb37476605b469398b2a28f44,9d17690cca512fe82a64cf5b1e5e84eaefcedb79b9c45a1567c341f14504f1ac
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json,6f1026a908ae6318cb245105b85edd4c66b6dea709ab08907c0c3a13998eb50d,601b8ffe4bb7c40197de38985349d0beefc9dc02899665f1c6b06240905cf58a,39bb98410d3dab19c74e9ac7c72234c2a4d75be876ff262dea454e62f4f1bce1
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-explicit-r02.json,9d3db1d0086ae8e34aed1c3e9f0baca1e83ec2b00f389dd545851cdd4070ae7f,688f5a7279202aa1a4361fbac6c6a05daac19ebd6838ca0faad26fa7c527ab57,af7c35b037cf5f4087494c125cd35842e72edecc914cd2d7cd462e71dcd8e764
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-explicit-r02.json,cdce14b9f81ec9dafec7ce7f5fc247da2e4c3e440b7229c51205b6b313b34432,bd98913c00191ee33aedcd779a98c0e1431d8b6ace5c0216035e66cec81295d3,03f29a94c9e73f8b5e7eeb44e9b1ab0073b261efe84f5ab366dba0e15ceeb50d
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-explicit-r02.json,bd01cff72604de73e79aa43210d60d2f87ee68afa76697f86bee74cd0acb451c,a0b2c0ea874a4db31d526310b10794a6f16f0220f7a71844f1c928fb482defb6,cc894b6f36d0d7d9ac4ee8b0c7b4fc97706672e7a01711da21fb54ed70125cb5
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-explicit-r02.json,9414f08060a614c1259e387715c9978cd4680172d827615fb34895cbd595f7b7,188c72c2017ea1a324ec71570da1d784e4ad654b9ca36baf558e282c8ccec97c,a7f0fff4545c97618e7ba36358851e3b257d44f00e017a440781b7b7b1d91699
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-explicit-r02.json,fcc2fd4dcdc0aefb1c4f34fb2e2349bad1c6a90f1702dbe14bad42d5e848fbe6,55d11e4e8f2d53d9d84467fc334fb8e4a5c14094e5a242283957f17dbf7ef4d5,933ef727f32635cb02c00d9fa71fbf875375cad462cabf124d6ef1962f3ef185
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-explicit-r02.json,82cff76a814df1fb7e291e9e62e78bd891f20e943bfe67e015aa9f272ddb1305,281bb40ae47a088aab5ca5e344bac5c4c7a42ce2e6e6295c3648ca06eb9bc98b,5f3945936d61d6a22c9463db5e7902ce1b0582f74a19001bf68db6b2ef830899
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-explicit-r02.json,d8850478b5e8e523752a187898f7fd336fc6d4610c1b238df46543425bd5982e,317be9924560fdfed127fd6983612f6046f677b1bf90291d0209300eacb1b421,c06d0b95278bd9ae8a8f9f48b7baceecd896cbf4ca43fdd41f3d90edc50c1cb7
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-explicit-r02.json,4e6a31c2f8a14c6563738ee0214bd001269b41f4e5d440c0abdd77d369d699db,42cb38f0b3a33178ad9899e2ce25eacc64745a9bb83106f9d9f42321563d4324,c6109b7ec53d6022e7db9b1bb7e4c324663456a3b9ae8895a2820c0784625410
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json,eb849c8ba05ec88a76816d3cf6cbfa6fa36a036be9a11c3b67f590ada589ed0d,2bb5e128ffb807c370b4a65e67ba7284dff3f6946f15005b105c33ee5e885e40,e71050dd2719794f7643a7db0433b5c6d659571d5979cb32c80844f4251048b5
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json,0fd1284615b74b792df358d750387aee37b6944b1bf361cf3d142e4a4d9938de,073c63d5fc2c68ce865940b8f566227cb33d52290d244bf07c0db9de9d090bcb,1df683079169e2a6ff84642877c0463aa5568cba1ce28ccfe1b8de9e9bdf8c38
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02.json,8f31a9682de296eb34fb97d838133bad2425a31db2726905e2e9fc731e74533b,01c5c4bb376bf7b7bbe695333f65aa550459d74bb598d97d21253f5ccdde4ba8,5fdd158fca3b54b9dbd0405f08ff7300e1e14b2e5fcb26432a25520b8f531b5f
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02.json,5ee554f274aa214cc0b759cb176653faaf6013f1607595b8a9c3a803378b53ad,cd3514340e6c3c03d072901650319939a40956d178f8aba17f1ba28acd5bb782,027bdf4bf4e7eff6a4bdff4a06605b2181425a8d8b96afd937bef7b4c1242a64
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r02.json,a902ea66ecbcba9007736547261d0344b8385949145773064afc55b261443d00,c69fb699d042ea411c1039e91b60bd824ac5c52753dcc7a72e16709de763596a,7b227ac82ba616176ae2759abf713f9196a65c6b9aeef1f12715744f9f8135c6
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r02.json,1eb8f8fa713f70faefc158fe023a9d376e2f47b75c78039f7252e374584c2d0c,f9fe6a1d347c6ff3ea6a93d8de2cd977621f4704f794292f3fa38488d81f1e0b,d409a795e30562886e9cedf45526138acdc4e0aea21ee046bf5de8ff632a7d3d
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-explicit-r02.json,ab888f3da917e003b653d960b9502e1bd17ed5ddf5050822fd058ec576fd7d5e,63bfe16106e2cd41fbeff5843799c4fc4a9bd7d34952ef0ecb4a51b3f25c13f4,8b240b71fe21074eceeeabebe1a94ec268530fb1db8be451172115c6eb5bacb8
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-explicit-r02.json,5e47a6a73006892d2cb7e983cec15be8986383783a2a7cefdeaa3b66a835fac6,7712a4ecbd4bee86a2bd290ae353ca749746704cbd17c1c8b213133e50d1b08a,452265dd5eba4fd31869e8700928609e3c86cbb9d941ddf5f0cac1caa17d5a76
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json,278993461fd2eab2c6d9aa650d9fa8fd728a6f9d9c5478f60d58b22033a261b2,52f38f3f46201a8a9d502cce1ad34a282b9ac1f8e444b2899961785273c84854,a5c6265e12c978cfd50d39889faa93f0674df598e5eac4306d3e1d8ed5bd70a4
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json,5d691e766c55c3ad16494b243aa6ffbfbfeffdad026dd7a4b197a575231b476e,b2257bc200a0414e5da1a077ec7aa9bea679d17f17109c3ba775f7d0b1eab858,0d5447b3b5acb293a81e2d46c8d89c1a04449578e0ff17e89b3e5a454b3215b0
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02.json,39c8bf01bfd35200be6845f79316d2e3a0aae1c0ee61e876c1cfcd6b399e65d0,e26df6028292f16cbaf134550895471a477a0c190ca9c0df67b95d9a64806302,4f6cc44b05bd1c5507bdeaf745b77c6557f3cafe0573e33fda2686647a22027f
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02.json,a6e3e49c27d018545b57f4de8036804b59714875b39ef480eacd268d85955fc5,743127a10b3dcbedca08919e959a1b10744fe18e408773698af9ab79186218fa,67c5551dc5e5d69c0cd56d322ca322fa18aec5daa07921f13d455b10987bbde8
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-explicit-r02.json,0f27e95fcce427c05d1df04cdde3caf92d6c263e935d1704243bd846fe7aa265,59722d29b60c29d3b563de67cc1207776af44fa23e91449d2ad7a46198961ac5,021f158701ac8c4f02111081d5a99160f2109108aed291f843ffc1d17c602658
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-explicit-r02.json,51fd25909a6b37fc88d35e7cdc9175f026f682fb5447286e6fbb90066a2a3c32,b5687f88d0bbf46c992225a25b2e6afce7b6c9ad85273b3d45f4a1a6a3cb9311,83cad3aa0639f8375afe724558d61ccab02018fcbaec844ea56106a06cda84b4
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-explicit-r02.json,1c8715ba236a36e9a8104b84ae3fb5a49b547d88adbccc851702a86e6ca2cc9a,e001a8596f74d6063c7aba515e52c86c93519f8a84e8802959f00632871c37c9,26e2c0cfbfde0cfd1778bc1748b2c93abf70da79135902b9396d1e6ace57a91c
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-explicit-r02.json,2d635324ae76d7c905eb104095c8058f3d9a3e2067e8dc8817ebd6cd166389a1,943ea979657afa0f76728a5afcd72052d86bebb99dc6cc1792aa8c54b997b00c,1f190e5231cf420e3ccfc8176cc3a5ad81c7c9c9971544bf46f9a6fc8e4b43f0
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-explicit-r02.json,213da52d2e60880f7a48f218bcf9dea6a6ff82f062f39dc1c90a8200ea2ac521,c512b9344a587e677d3d8b200fe0883f0d09fd13183942545f2ed65c8077439b,c685e3260270c4b279b597e61e4a7868cc4ead278d14c4852f1bf5aa4de05ee1
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-explicit-r02.json,519c606c540c825d24862f8b6208bd75bdcaaa1b9f4ddcccc37c450e6e0fd053,794ba402ba12dcda109d0947fdb24dc963a4ed0d648bf1a73d72fa3117981cec,4b06035e034990737a919f8d1a49f276d135b3627924257196e232603fda0c8e
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-explicit-r02.json,c5c5203f4f325ec798d678339146f81faf87f486413baa0499e8b44f2418d939,047d7732323a044dbf120ecaba8fdbb4f1c2ee45bc8262e5af24f3a9ccefeb22,bb219b58bcbb63ad704b26e368013c5f8afe8c20f0f559a95db747114f4eb049
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-explicit-r02.json,cb91f2cc31774ef9e106a144c19e84a4af4d421ca7520b50ea52cc98d1258964,d601aa628d14dc1dccd44ca4b766d98227b66be0947b8df42c2866c84b2ab0e8,a6a9713c4492b05bcfd32db5d7326baed0bc2e66d2e73c746d5c10a98617cab8
   cpu-release-sparse-solvers-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-sparse-solvers-none-prepared-fused1-host-standard-r02.json,dd4a3cf00aab1321e5709c5f6a004e5044b5dffe7a6a8d17cd7895fc1f31a8ea,c21655aed7c016aac08029f00c4d3b49cc007db795d8594203cea65b07fc31c7,0822507fe67ea96012aca339322752562a22279b3c251b3d9d0ff485adefcbb6
   cpu-release-sparse-solvers-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-sparse-solvers-none-api-fused1-host-standard-r02.json,1dc37d291ddcf2e92e6cb736c951f0bf7dea514e72791744e1b4dcfa32a8bfdc,a2d00b80f5a43536496907608f46942aa09443de2ea5c846967c84acce333b24,f5389c9014cd640a7177ddcbac07b41f8b9e33d21fe83c0481502bcc353c7d16
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-device-standard-r02.json,8ae4f02277c0d0f8727ae17f541a8b89b6a918940d95968a06c2b7ba34797cb8,70d866753e50e70c5029a09f1198b5258ef43df5e19fbdfadde9620696b72195,1782a14c65d6dbd4e7bd92aabc143aedb3c5eb4609238894ed39920a3cf6804e
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-standard-r02.json,77da184ff2934969f1cb795b9b0ded21efb936748575765be1ff01be89f45bd8,cef478f1b2533f48e7f5ec549f7f6950f90c573a18cf4ac594658421a3e4c2ed,95a22da9ea383422fd636c9b40e2c8cb7f39ca74e184a4b69b2fb0f55ff87c58
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,dbc3a39c0dad906a7ee70e9501430f9041374ce88773936b34bb29060f92f396,d2087217a2528ad3e5b8a10d0a04ac210da8ecc58f63e886710891791023b3ed,61036bbf4deaafddf46e929b2e0b02151261399e7c00dceb14e05aaabff125f6
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,9d4bb95a2f3e63cb958ec760bcb67c2f6100790848e868b0d5c38024e38dee26,94b4324db37a916d417464695dafd57717c1c37516f0609dc3f7b117d69f9ad6,2d0833e8f60e1604732ed516a39f5bf4c68d264cc64399d474cc2f5a311528ec
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-device-standard-r02.json,062308f56a6c962206a55d303b80a1214971b52f2b9937dc06c0ce9d992d6e6a,98b93e05d0423181745d9ac7ca86d0479213130d66d3b3e2e3b4cf172593881a,c7cd0c390321cebd2f31cfa337bed30535bc7c9dab6611cc3d3c8fb1da1de7ff
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-device-standard-r02.json,a2543e72f2557dfc9d0a567413dd44e5bc6c31f9cb95b9d45264322b84e0bf0f,87c1e6b970b3656e5b1d6b519c8834ccf9db729612c4af5c55d817614b78396a,81abc5fccdb1ea7963e36a9f3f22fce76c9cb17f31c2c332c2fa4a5bf37b154c
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused0-host-standard-r02.json,b9185048fb41059f932015b9bc0b8f6ae7dcf820a9bfa9dcc89aabfd47ea210c,2785ae4c225d0d2e8efa6864b25f2ae5247b16f9ee3910f23da21537c26c027d,4a79257e799be8ddab966c3bc641ffec9a1a2b267cf754184e89e1a7a7fbfa9e
   cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-session-fused1-host-standard-r02.json,e3f151359566e02f21208b41d1997ed8e2d86004a5bfed34f70afd1008910b8a,939e2714ad09a28f37a46e2b38d0f685c93ec4713d4cfa0e56515f250a3135ba,3d26a2f318febe2df8f68589426e744e512553e4773869bc76cdcccf5eea5f32
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-device-standard-r02.json,919492219e42b35f145a9c3cc5d2b153140e0f61d3d5eff8c59620de84acf671,accdf3acdc646fe6d1f8c23a6bb281d1ec1cabcdde2b3bedc64e962e8c7309ce,df50f265ef29137133c614160bb9d5e3439f64554568bf0c12dde980213cdba5
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-device-standard-r02.json,14f08c403feebef41a9a0d3adabbb21ca95845b3d2c9fa3c3609acbc08406637,0b58eb325c217b3ce070aa3e0355b379dde1f4b5f366d2dc10d88307516e834a,080b98ad8c08e4749049b149610fc757ee93105684739777137e2a0721af6a66
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused0-host-standard-r02.json,d825cfdb947c88e7786cd6ba72d13e25d5fe935a31ebfd458c2d8c0a6f16179e,eb3fa8af947d0c6c5b2278bcfb4f7eb6fff4b55e26e0fd69efebba1ca4170d13,af6eed9e842d1b65aec528d0ec39409a731b80dac51a28732ef116f490a622d9
   cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-jacobi-api-fused1-host-standard-r02.json,4a9b94912c30d9cd5672ca3b108fdbccf9f97b39d0739d0b42d41e796eb06ac2,e11093fd12653e39be8cab011eff32d0a0939dd1a6dfe5e67f2fe3478114f03b,f9387ba72ac624b787ec68a95c34925283a461e07c0128335d10371d8882a4c4
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-device-standard-r02.json,ca70f253fc2f323a7a072fcb106edcfc55925fdadfef68318078b707f6363951,ef09d8ea0c5cc9c0101032368b3d8a55db2dcffce5d5ad867a69b501f086821a,9bd8f8a8206ad7569cd09e130338ad648d16a88ccd3a1ade0b632bc025c34b65
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-standard-r02.json,01f1d6a47a9975d357bf9ccae4d09128aab569f705e5354492d4f429d102ea6f,1f2ea45a1ee5b4a4a1bbf49e9dd872d293531f20aa1482647f247e28432de880,40838b72da9d6165d260bf776babb351dfccbdd9503d9c9b62b2ead270a5e18e
   cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused0-host-standard-r02.json,60f3d61c41e559d202759b5663eb591c05d73829f63ae04dfa83f204ecf55f6b,f2efa30676085d41db9b3dbe7248fdbf0de82eabbe1261a658ff93a8cae98030,44b981cf503aeb4f5fb48f361777a6ae20d7550f7676f5fcc46653a012c9cf15
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-host-standard-r02.json,12027d2660e560abea7bc43a8931098de8293982a3df721673b5fcdc4d9a4856,9e7c544811e1e3d915e7c365cd582be91af5d89b349fbacbc132b61860ba3c45,ea2bab6c8f95c56a2f830ea2ef9a5ac030e69585165048f53b64d2a9dbf2ab22
   cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-device-standard-r02.json,bbefa715a7411d63def49e4a4eab5101914d00e763fa4cd4c6cdd4361fc9371a,2acf61509913d8505beed161950c9f8a22741beaf2f00d859a131bb095e0f1ed,007cbd71a5125ca6cd7a3496ee236e63bc1296c6f3241549ba8f06905fdff5cf
   cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-device-standard-r02.json,224886475cd60cf6f09005446ef949d37dd741661a7f857fd41f942f97f534c7,dd0f08dfa76c9718bc215a0436747d56ec0f343090c682c7227fce8e19d59903,84e65e81ef9b7383041178cc658af82887d733c4dc429509250cec7192b23db4
   cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused0-host-standard-r02.json,fd7edf8e3ca45230f69873c18075ef79670bf6d34c0f61350807992ece5176a3,d90e190951a3eeb29e358485a125ac73f9da23674865340d3afc97b045f686b4,e10ee3a2e18b2b5bf6ee31c97abbcb23419c5bec30af8086798f27a25847a0ce
   cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-session-fused1-host-standard-r02.json,be043da43ad402ca4bb66e345524f4c6ca5dad81606e1d4ba39c3d2cc01379e0,a5235afb1cb66e6d2d1dc27593758ecd52cf0002d2ee327cb7523324bc9a2140,cd45734b3f853dbb2d979bcd15f1904e7f30784bbb8177d9013f7520501df10b
   cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-device-standard-r02.json,2d60116ec5cdc27ab23918a8b7bba46095432a283548a28b2520478986877681,b2fc1865ff8f613fe51fda0719f918aa00b7f6e82cd2b6097a28dbc1a012b43a,62fa7fe8084b61deb3feefb7b5755ee112fbede18622ab052d3a6bb259f3e5c1
   cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-device-standard-r02.json,436858ea276ac02b5d22bded8ba99b86492a7d53352fa0eac41d069880840130,3323c644f693c0fd2fdc991f6b9ea2425033a1262ed3661992549e493f5989c1,bb76304ef2152bbd250b6318bb9902a1ce79523b645ca8497f5d8cf691fa0230
   cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused0-host-standard-r02.json,e3666130f92a9f1ebd7ce5474a4ae1079741e7da881a9eccb37d6f07171bf714,6072d15a137e993161f6961bc5ba020aa1051485908b51277d0ed59dda501486,d8cc53883f30f181b71d34a6feeb53bc5ed92d4340d32147ffc12464ec4055c1
   cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-csr-f64-v1-none-api-fused1-host-standard-r02.json,e138f019dbfa96464bccbca5e74071d5cacc290e901b626bbec56bdcdc19c34d,9d36883b3eb273b59eaaef990e1dccbb28e07300a6c5636e677f0e1592c29d28,f4948a5daf43c6063a043d85fc1ba772f3f401e21c804721d32d2560dbe2d54e
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-device-standard-r02.json,446518b553281e90468265758738bebe093b3ed0eed01b1b4751130537fccb4c,3263a909a3d5674d703f951f5e2a61dfd222b4a468af24985d7bc953f291a0e7,a701c59ac847708c8d4a5b8b35bca361cba8459fac650c6ae1ae4d4527241e50
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-standard-r02.json,3733e65538486c5965b9a224fb9e3616aca5f3a019003d3a72c67f9b73a8009c,1215ffba33040fc5075cff0cefd61045d709bdbc7d8ecb72adebee7262e69d2f,e6f584ca9d965533cb475a537d390cbbb280570dde61834ea705a3f9a3922052
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,1dee6ad30afe58a541d57956306d3a142ca3e5d19cbbf5211997e0d9c94954ae,cd6aaaa151933435a8d287f15bc761badf46cde6b209823caeb752975ec7126f,4669c8fa5b22ef7747943e0363a363a534ad9cea78dab12b19e0ec86a21132f1
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,a04f185b6e37a88c5f9e34aed1f05ba5102fea5c6e847df138943da269f2da5f,156e5cbad0903abf6daf22fa503ec13318627f42e3e7e01c16bafe9785e9894a,c2ab297104bb23c0873c7700fa6faa2928d11a910d5949e82d85e77d3e78545d
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-device-standard-r02.json,d29e7f0324be4bacfb72b50561cc020043699c9b6d0335c25b748f0edd8b0282,615223d558cafd4cd0f026189ab61811f32276b7641728ed4acc565f183fb810,496b8e68877181a9a69bb21ffa82eb06e38df9d1ffa84433079c3291e0b9ac66
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-standard-r02.json,5a5636ec410fde4cb84636be91169a5fd10d6ac6b6f2f857905fa341c676f3d8,9e997692a2224532bd0e9b541b2b2b6367ac672b2095977091e7ac63fffc5ebe,bf832e0a88763b199266d44f64585cbddba371a1c96921ab5fc06200057b1201
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused0-host-standard-r02.json,09139108cde331fca09c2a59e351d52658a53bcce416e5ad729dbe811438e09e,27453676926ab0b5eb62f16a362653992e899c2bca2bd9cd5823f022f2d96abe,2f09df4499cfd487a480a217652fc69aa2320097d1eac10edb8be9b8aaf6edd7
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-standard-r02.json,e7fc31069d18f0984e485c33a5feca41e81461dd703456231b1de4b04cb955d5,5b642382a689ea6646cc80ca30bae95693a49c8e254782df65a3b2d8cf3b44b4,9291660db954f51829e3b0d7d1a8ec057e84ed33592dfa5bcf989fcd4c1bf294
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-device-standard-r02.json,9323f52b37f282adfa61c6bf54ca3d883a554c7f7446a1cb3f5d1257ca6b85ad,3c8e68e2fc08afa748bc5ddb516cbd805ac3e571c0c987b319421ece9ad98bfa,f048f6b13341d1f146ed53ea1549e2f84f06873ea32a06c8cea71e6fe63c2ea8
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-standard-r02.json,3451e432fa57b0b52bac6e4a5b2840b1c184c3deb5f716bb615327c5dd581257,18709bb5dab62770d608f6f70e423999fb7b6aa9904a2f55d6ca0361ea0669f5,abb9466d36b01b9a822fb8d7c9a9232bbd1a5bbdf8fbe9d0241392a344c7bf67
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json,3f6d7468385e711371c7410a05889f13d0dd35d95601f27a0035411ef7321307,b663c92ea0f56da99a43855ac987690b1824fade9d7f98f4eb8ddc9d230c4766,e657592048ced5482473dbff174a5b9a2c209cdac3befc97c9f0cb2ebeeb316d
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json,de281d36ec0187d2a26eaf47b36a5eb91bcbd1d91d1767b1910f107f9fbb908e,4d38cb1c8a8991c7eb8730b40cd561e70fa115ce50f1335431fe98abc679b4d7,b672230c0a2ecbc6691eb206b93e916320efb07757b1d2c9e78ed8ca482f0dd3
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-device-standard-r02.json,859075eb5cbd92d4092258147920616c30f7386d12f77d119132d2fb9ada2060,9320748c8dcbf801db3dd57f915ccd9af81b803714954e5bddfe26de6549f22c,e7322339f2474fcf1b33022711da5184acd53887d1fd8e58051d557eb46c7431
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-standard-r02.json,fef3857c45710f7bfe3b09d7fee6f844c2e3f8d0943724e1dabe06b88565c09d,8201ea489839244f32449047c128d1582f286a1d3a883f26c0dcd357556e8851,8a1cf26ecbeb03b308b2dfca6c6e0ab443f2afff0004da5e369c201fb08e5860
   cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused0-host-standard-r02.json,7ae3b64075607e1addeb648c101809d2754910a8dce93071c1ba47258f331880,a3309ef3045c6236c88e5414cf2773b9087f82329b8aa82b960bcdf03aea14ab,e0decef7101ae5d3ba24780b1ad5af36849b4faf633605946b6434f581e21fdf
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-host-standard-r02.json,75d613906278925d9075ee3b16c3e3aa3d303920b5f4de3fd71c9439e05f3e0a,3be093aec8e1ccfb9f1a097a3f6421184bf292d60ca6c1cae1e8b3d547fa5e15,a06223702f07f0ed83e7c5247ee289a96b7c711fa59526c0efa052b738d49908
   cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-device-standard-r02.json,c22be0c1906d3b6dc264aa683854a871e7919456b6f71f71c1b89b269e816d3d,5ba15a5f4b419cbf68545417d7369f7c3a59714ad6928dfcfe9dfcc39248237d,d29434524cec885f10471791a3d4b324b6146655018d214e9f112b3be70ef140
   cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-device-standard-r02.json,3338f8e325086088df28dcad08ff7e1235b873ca5804c84644a4878f682c5433,3a3724bf1293809b7d9c597d49e15c61d7e917a3ca51efcf9799134b33680651,caf655a48e811b0f87b47fb893a971c521158c4621d6a7c3596c764079a27c19
   cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused0-host-standard-r02.json,a72b39be92f34d9647212a1a4165d5b66b99316aa20033330a277e2e0a99cb77,0879ed8df5f36d56785a8eff6412007b045e25c7c04f94f5c391d7daa4bb76ab,a5c26de6e58ddfca5f6c944f7cb71665eb7a283e46ca568dc49ad8d23d0720bd
   cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-session-fused1-host-standard-r02.json,2130131d0b96b3aea0d22865c01e9bdbfe1dd3c2d137f68f2af56ff176288339,6a634316181557b66e54b13dc4218191626266409522a634f9482ea8dcfc246d,31c1bcb8428007ab4bbcb7b3f24bc96c0663bdfb688bd443c5fd64edda9f36f7
   cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-device-standard-r02.json,443a3f4bdaf7ec4bead904ba966d03e8536aa64c40dad1b937174e5859cc5241,7c09b164bee03effa62a284236d7076735685a01551cd888d980362b4b3b26f7,a17d65dd01600dd840521ad60befe4940ada0e3512cd8fc7602370011093030c
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-standard-r02.json,c2eb629cadf92a858a34fc0bc4e890938f002a9f369284c574585faa551f7157,a27d0efde470871dab5203a8e3d09f98a76f1c02677e303ed4d80f80b6362afc,4589475801310ae944e73fb1d5b1b0aa9c30ba969dd54b7d854fcdb579ff26cb
   cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused0-host-standard-r02.json,64433aa622eacb75ef5ea1caa09257ff216f220bd9d4a120e4e9b1600c733c18,4f98fdf134017cc27161c9f4093a75d735325e26d5e33438551e7b9b55bb97e2,abd6f28c7ff998901bb96758a4d40464afc0a9c009fbf59ff7d1d02985eab440
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-standard-r02.json,407386133905bfc35bc248ae4bd4aa516f20ce962ed60dba0c2516c0288ef563,b23400ee809a60cc357e3744d1526d38d9510fa6c05e48ca9a0a37ed482839fa,c9183b09294abfc1aea5db462bbfeb0d9e094c43793725245f7f78b1668c264b
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,1fe8203dc8ebcc4665a23bc4e9fe4fdeb4b3711e86b09934c20c99592204f9e1,9dbed0c6598e16a72ad4eca70f93d820f2c3ac809191fbabdf9c946b241a6af2,c1f0b30a9c1fc9a440e91624ce0607db0d17dcd739b6b60b65e938add5e87247
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,95bfef37ec8f1f2f7d797da1c3755eb2f22a334d56719a6df10deb19fc783832,0f410c23891586f0b6cf2b8c4c0bd4397e0dbbffbc26620f3f0942a88b1308e4,29a2307307bdac279801c391e21daba146ac27a5c0ac51bf874a2f1ff2a01b83
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json,4f3fd033a6aa5101bafbb704591f1d4560600f461ddf61f5d5ed0b72b0a3ef1c,a825f90c08648b86ffc21fb7d9283a51e9323575755a1086c0475c8632e96dcf,2575875558c6ae1c6f21ea878aa7033ffdc482834c8b2d40260fe46d2c333c22
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json,1b49a241729e31de6cdbf9e266f7cdc5f93b99901a101841b6123aef1c085c29,54ddd6e9e2a09c1eb1c3b5e058659d8ccdec2b2b42d9f436a3a493b76e4dff31,82c795ee67cb72e4c1ba7abcf002fe83fabcc2936ca7571adef932faa8819983
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02.json,4bc3586617973240b3d689845c07df29bc53253952c3e73c180f903654ebd548,8ce1552e79bbd387d86c4c613a8a405fc0c756505dcb62c321fcf848f465c68c,986142d6dd80684a952bb52f8e1bdb4e244be0614f94aab3edca08648e7573d9
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02.json,c3ae01bd5d78a9992e489d52ad40acf2b45f5b09129a94e6400dab0537eb2f22,89e28ced7a61f69a9eebfd4458d167273f222dc3f8e018ca43134f33b57c95ae,7ef8d410ec041313e8b3f90ad1d937c44bb86f1457595a57448eafc375f810e0
   cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused0-host-standard-r02.json,8f8245a5d3e3ee6f9abb7bbda34201cf6c51602fa8065117af14777698655669,209b3ddfe40fdde67793d0c592cb258aa6e1e3c8583fd31f04d36e19488179ec,48274b54d3a57b0739e23d5f3323320f3e736dda591a3862ce596c6514f546ff
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-standard-r02.json,dc239173c28e3f8db6c2fc07c5c9cc31e63479a8833022345bd9fc7f2a97d9ed,6de98c91a0619014dc5f2657815b2589e0de371069b8c676cdf1f13951d83099,9ff5877d6d96365fcd9a3d79df86a899a2545527d69d1fe3b17a4ac9387fe901
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,2156e88e6fd87dcf2841e021c855a9d576403cc95462e831b370828f0bc0f794,37898f3da1ff7787322b6b19910f3d626fd6204f73eacd309a5fd00dba8f98b2,047800fde21f47e873f86e5ee5ef4f5751728a621496d0c319a315752ef92726
   cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,da08fa968616149c588687c03d588b7954ce6520e7d7362c83607053878ba734,37ee946a2c76ca6c69703a59a9fa738a641fa6c099acd559ebb0121ae565b7cc,e892b02cc57d827553b7e914070b5885ea888182846e2d24af75e8dbd4248805
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r02.json,bf7119ae0cbb8ac7b6d135f91230a4993f2d186acae1657dac886accb450c36c,ede1b455e5bdce41fe22eb78ea49c649b38e612d20688f4c3062a373c67ec4ed,4618f40fef6eefdc26199c6a659c8eba556dbae1f1b0848aeef3922e9e6e6eb0
   cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r02.json,848eab55004e26750c200ea52f6faf508a5ff401cbba5dba33721795224be282,0e40a2ce229b6d88c5a9451cfa91a54fbf4e8ec7143cd05def1eda2ab2a198a8,0998aa6c04f40d821d1fbafaf2cda30bf24575088343649bc14f637bc1457b82
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r02.json,a16e6bc82a5700d62973f15f74f2d80db37b969837a972a105530b0d134aaf4b,0b3635d11a8183a20b941488c10356286c0af42a706b71a38250ec04e8723b79,a4d6d22dfc2c13256f45226b0077a865ca9581f784a9a9cb3b6324199e8701e3
   cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r02.json,0e4736647da864d505d9c5dc03a2e22864294cec1adc2b23f2de1b6516cf08a6,c0920aa0a1df279d5b11752a5f231223c421acf184ceff5f5eadfb4728eafd8f,a7847f3d9987b517ced7e11605cb9e8e8708ce580ec1267799359b400e429a36
   cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused0-host-standard-r02.json,92dbda8e237487e2377f3b74c50dc9b31f910def83f4b0991df9c2ddd88ed003,7d6a0d877187173b77edc4858562739b87cc84f92669ab4e17f7887662c1f6d8,3e4d02f577d132919d9c40b0ef424fcbe04fd3b4efc075b35c7aa06cfc21eb09
   cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-csr-f64-v1-none-api-fused1-host-standard-r02.json,5903aacd176ac11e3b8f1b67069cf6afeb15895b9fcb7f90b3584b79e8bb9055,e779a8d097cdd7385abfb24b2789d633a355f4888cacf978503a73e9012db4b9,ff9e76910e605d66f2cc377e37e0c7b938cafd5991ebd52d5275c942e5b73bec
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,5dabaecaacba53f69c5c5d66617c79c3ef506cf5114eb73b04ccb8fb9cd9f89a,a1e9907b2f258693f28b6d54153427695e9b3ca25ac6fab2f34eb24e74e69535,e4ee429d572f0e290012ac7ecfc163112d6cb0ce4a909dc2e67943f8b7c58f40
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,043f61ac0937c68fac2104686465a918360d8c8e66acf3303661a1312bdb5d2c,0d49682aa5a2ab89ee09b6f31077421046463cfa9603f1955cab402ffa594bf1,d61d569a9558ef61ed0972013751ea17ff1edd3725338300a8c36e20381a4130
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json,893d909942c89b785df36a8920f9eaaf7b409c49cd5bec18141c5551ce23f822,92adec1e376e2e4d7709bf681042878fedd24ac348fa18bbff50fcb0fef9505f,1ffa639d18715bb49edd75bc90a1c640f01f9b9ac0bce82746999c861308849a
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json,87c371edfb34822fd18de2712fc03980d51ae14a0851034fb0ded1a7ed4a32b9,3508c59e676bb21e1644b9d4d945bdbab06c208ff9010dd7b264376939fee30f,fdcd5db9e63bf6a43cb5bcbfb69f51dc305636b104f0ff5e681ab52d348c6b08
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02.json,a36f6f3a7df067a06ef3deb74b16356d491345a25f0a859ab42faede5f70bbb2,9a80c953d8e0b535648e0116113e8465dd7ed1c7b5b3846145c550cb86a32f37,f632230e066b321dcdd5457f34aae8df31cfa00e22843d1bfbd8126144b46419
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02.json,ae441757c24262bbc9cbab622c186ea0c53b4c3f47cb7865ada4f86ef94b057a,883a593d04a29ea19d7907bb9c61ac28da289622a118f88de49eef7e3856168f,9443529b17c82115aee8e290cb4d553f80b49af319cb5aa1a069f03223a8341f
   cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused0-host-standard-r02.json,f3da2b2798216a632743c7cd362f0f66e1dc022873abb638acd74a9a14b60ea3,d3e572bd8ea8f14aa8e6d6d9600b058f440b3ca840ed97ac301187f221e8888e,0280baef94308244bc4ba69dd9da106eb055e375dbd112977aaeb02dfd0e7570
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-standard-r02.json,86082d124c24f3d38bf4ea5ac562e29f9ab2d5db28c5d5bcccd8ba6d3542e622,20928baf3039f095dc294398662d5b88a3b54291dc6a6ce6ca7e37b6b806f83d,34016e51354609fdfd1f4554dcf6169448afdbadb25dd8da06cb39067a4f33d8
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,c96aac66af587cbc52312853db8fa3bae7faca4c595ff56c49f681d88b7a5cc8,eec432299a5bb3461d935938589570492323f595a0e7e149b451e3af8c11a813,e67b38e0b3e8b620053bb47a2c6c979d51e0572dac5fad7ee709887882887572
   cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,1e25a88aaec2a169c61541b9fb459cdde704e1e8c9f8798226ce7f0ebd75a9f0,a71c3f8de27a5da61507c8a997f44b16b7fc83e4b3448afaa2f062f8381c843e,64eed5fa0be36b12de3fe08bbd8768c689b29162b0f7881683d297111d553278
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-api-fused0-host-standard-r02.json,b1bebad35da54eb43b9c05f67d9833c546ee4863217bcbb0ea8f15182e811bb5,d98fb6b27adaa537a5b04c9b0bca6e24d2d2ae71a6052af175ed4b4c57365d50,2aab155ab1cfca98154e5ddb34fd88cd53af6a2ea54e214ba64c07466eb9d075
   cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-jacobi-api-fused1-host-standard-r02.json,aabf60180a8710f583c8567192c2f393fde119b3a02f832910dc5c15ae842607,8e6f8f24217b1b47ca11117cf590d4a579f6c499b6250e71d25854e17c9f5acd,824dd58824bd0b484660450753067f98ef8e3aace4dbe808bddc7081de2e8abb
   cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused0-host-standard-r02.json,ca8d690f78881572a3ace7da0ef5866a2735afb4b4ec905ea559f3cff13c1b63,87fbac905d42a035cee2b29c69e2438d6f3d6262f81829dffb71a80ec78967d9,fb83f401b742ab1c4cd2f7117e06fec398b156c9838ed08b79b864b6f0bff47f
   cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-prepared-fused1-host-standard-r02.json,eb63a57a437093780651b538db530f74643530c5e51423ceeed7a09130957a24,9828124969a509951cbf07cf6c61a0c51391494736637731d1f8c1d98721a745,cf32a987c9b594b31a54c5a15cf8da61e780354642bbd62e389eaa86b9aae39b
   cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused0-host-standard-r02.json,17c4a40e17da00660879a9d792e8180ffa7402c383199277dc0585c171cd58b5,3c6d6e647f8083c812b23db5be16a033fa43113ef3a75d3b0cde0834921ad1e6,073d529f50aa785e85de9fbf5cfe47017468adbab26ba86f3341610fec702f08
   cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-csr-f64-v1-none-api-fused1-host-standard-r02.json,24576256777d7d327a822ca09607bb78491a92ea53cef5ec583ec906154545b8,460492d8dddb463fd8504b2fc15d546a5d22654fa943157fe428bd38517835a9,c812d8a5627beec88065ba9598b323a69a4e2abb04b74505c5aef86572d7a60d
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-standard-r02.json,dd0d15e9897021c50c082543c04011dbb37da3e7ced4880a664b3f4f4b119aa1,319ab63b039af60c431541a0c8f3121ab22f39e42887788d5bc0b51deca41f44,709e26a83d7636da0eb9d92685dd790789dca6767e274fca40c7eb7f6063bc6e
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-standard-r02.json,7afa21240de556ea50c587ae66a46af001e8875bba5de5e0d021a27ef150e60f,a4b88b586a4d6972a1cbd777f1b43567a224da27e225e09ac806ed5ba3532c85,ca0ff126f8838e587a43137f537699455a3d8e921e4e64ddbc3c456898c82db6
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-standard-r02.json,a9cbdc7dbc5785ce577e6c1b13827fe2ad53e90127c669404a488560dc129fbe,62c25ba070912a00eb9d0bf5d4a130e049defce325b6220cf71a697b30dee4e0,354d8a2dabd61be366c89c02d551b65fa2afc3e4685d11d91878682c227b84d5
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-standard-r02.json,0948f20149e68b320d728b562fe3b857d39e41b3d179d161913120abfa999003,889ee85e99df6e66adfc4870c1e7d3f2045041b9014f0c3b068952f059ff46ed,419268eeafc8efa9529832a5ffd67b438ab32fc36c3b67abe71dbeb7a05c81d7
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-standard-r02.json,0d39f447e5530afaee5f123c9ad567c13c5206b62f0d82bf67d5231578c11ab1,ea34fed209c6ab18af550cb1242b61b0e488d45fd5ec28d69f6fdacaf75ffc32,ee5c5d6edb2a1c4b1300d7db5473bb0f7f480b14dcd96da89dabc763fb8cea76
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-standard-r02.json,be6e2562e782299aa29f3f4deff3fd613adfde273ed160379f1502a5a1160d75,4fe645675ef7f4a2d2a3de3525191b6392404499436330f8570b05ccb8043b3f,491c52a60a4247083e4cd1868f2c57d594474f83539a16a35d9b2b80fc0325cc
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-standard-r02.json,32ca59b2d3ba7784faaa9b42876e85335f9e6c90c343be024c0df60810b40ebe,309ddc8172cbf19f293f21319e5bfb624ed3f49aa768c6ec005aef2f8ebc76ae,8306c6cc13a972ee8798d0eeae1cdbfcd7d3b25a169051e8a2e03afc28f531e2
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-r02,build/main-performance-20261017/baseline/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-r02.json,10e5a312e1f78e1a053252d1da68395db64eca14236427eacc3efea30aea42a5,7a873ed09b7d456ae99468ad1902ad0d9ddccac3b4391e369f240d0b3db57727,8bd4f7032a273f3002636f37d85488ead3d420c5389216948e37735126650bc6

