.. _main-cuda-phases-20261017:

Current-main CUDA phase diagnostics — 2026-10-17
================================================

Interpretation and recorded resource policy
-------------------------------------------

The baseline is compiled commit ``445275a3994e342b06de4093889f3f814c8706c6``.
All 16 captures and 160 original whole-call ranges are retained. Fresh/API,
Session, prepared and workspace-only probes have distinct resource contracts.
The frozen driver checks true residual and iterations/matvecs/restarts before
and after capture; the 1024-row cuBLAS Jacobi solves report ``4/6/0``.
Read these diagnostics with the :doc:`complete mode baseline
<main-complete-20261017>` and :doc:`methodology`.

Nonfencing Python wrappers add same-thread nested NVTX ranges around existing
output initialization, workspace construction and solve calls. Their overhead
is included in the instrumented wall samples. Native solve includes Jacobi
validation and CG together; teardown and prepared reset remain unsegmented.
Nested ranges overlap: do not sum them, subtract scope medians, or treat these
diagnostics as uninstrumented wall-time measurements or framework scores.

For fresh/API cuBLAS Jacobi at 1024 rows, each median uses ten observed ranges:

.. csv-table:: Directly observed nested NVTX durations, milliseconds
   :class: benchmark-results
   :header-rows: 1

   Range,Host mode,Device mode
   Whole call,3.568377,3.2849895
   Output allocation and initialization,0.157808,0.0856515
   Workspace construction,0.401005,0.380009
   Python solve_into,2.5415465,2.258763
   Native solve_into including Jacobi and CG,2.5317145,2.21888
   cublasCreate_v2 inside workspace construction,0.18822,0.1746205

The complete API inventory includes outside-call fences, every recorded thread
and driver-named extras. API durations can include pending GPU work; nested API
intervals can overlap and do not establish exclusive CPU time or bandwidth.

Within each 1024-row fresh solve, source destruction order identifies an opaque
provider-destruction free #2 before our arena free. Across ten calls its total
is 2.227249 ms in host mode and 1.775905 ms in device mode; our arena free totals
are 1.271835 ms and 1.574293 ms respectively. These labels are source-order
inferences: the trace has no allocation pointer or byte arguments. Identifying
the opaque provider free specifically as its default workspace pool is unproven.
The workspace-only probe's longer cleanup is not an additive solve cost.

The captured policy retains ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and the full,
256-byte-aligned 4 MiB explicit provider workspace. NVIDIA documents that the
default pool is created with the context and that setting an explicit workspace
prevents its use; this does not promise elimination of its allocation or
release. The 16 KiB sufficiency statement concerns workspace-size allocation
failures, while larger buffers can improve performance and 4 MiB is recommended
for this GPU class. See `cuBLAS 12.8.1 workspace documentation
<https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublassetworkspace>`_.
The `cuBLAS destruction contract
<https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasdestroy>`_
includes implicit device synchronization when releasing handle resources.

No capacity or environment change is evaluated here. This document makes no
candidate-benefit, adoption, or guaranteed wall-time-savings claim. The fixed
environment also preserves the Torch workspace/control contract.

Forty original whole-call free inventories
------------------------------------------

The following rows retain the exact per-call ``cudaFree_v3020`` count and
duration aggregates from ``fresh-phases-analysis.json`` for the four 1024-row
API/workspace captures. Each row combines every free within that original
whole-call range and host thread; it does not identify individual owner costs.
Source-order slot aggregates remain in ``fresh-phase-ordered-free-proof.json``.
All durations are instrumented API durations, with the waiting limits above.

.. csv-table:: Saved per-call free inventories
   :class: benchmark-results
   :header-rows: 1

   Contract and mode,Call index,Free calls,Total ms,Median us,Min us,Max us
   fresh API solve host,0,7,0.488600,3.2880,1.0930,371.6590
   fresh API solve host,1,7,0.351963,2.0540,1.0580,245.2740
   fresh API solve host,2,7,0.328034,3.6300,1.1960,188.4990
   fresh API solve host,3,7,0.336023,4.7120,0.5470,200.9040
   fresh API solve host,4,7,0.362224,5.1260,0.5620,214.3370
   fresh API solve host,5,7,0.346756,2.5540,0.6180,220.0250
   fresh API solve host,6,7,0.335511,4.1640,0.4950,202.8490
   fresh API solve host,7,7,0.392205,4.5410,0.6690,195.5200
   fresh API solve host,8,7,0.293281,2.1280,0.5590,170.7640
   fresh API solve host,9,7,0.407860,3.6690,0.8010,229.0650
   fresh API solve device,0,8,0.316050,2.3130,1.3080,185.0870
   fresh API solve device,1,8,0.554106,2.3880,0.7110,378.4720
   fresh API solve device,2,8,0.287174,2.6300,0.6770,162.1710
   fresh API solve device,3,8,0.490818,3.3280,0.6320,286.1100
   fresh API solve device,4,8,0.314654,2.9865,0.7750,189.3790
   fresh API solve device,5,8,0.276527,1.6655,0.7880,167.4630
   fresh API solve device,6,8,0.290053,2.5390,0.6750,167.5990
   fresh API solve device,7,8,0.335236,2.8635,0.5790,182.6890
   fresh API solve device,8,8,0.329358,3.0245,0.6050,198.5990
   fresh API solve device,9,8,0.291945,1.5725,0.9770,169.7950
   workspace-only probe host,0,5,1.052812,18.6790,1.5890,797.1560
   workspace-only probe host,1,5,1.406754,2.8710,0.9430,1225.5130
   workspace-only probe host,2,5,1.466075,3.4290,1.0110,1268.6870
   workspace-only probe host,3,5,1.290352,4.8640,0.8040,1108.5430
   workspace-only probe host,4,5,1.241237,4.7730,0.6150,1068.1600
   workspace-only probe host,5,5,1.262721,2.9630,0.6450,1078.6230
   workspace-only probe host,6,5,1.325860,5.1660,0.5380,1100.4670
   workspace-only probe host,7,5,1.269942,5.1190,0.6930,1078.6800
   workspace-only probe host,8,5,1.291578,4.7440,0.6490,1071.6140
   workspace-only probe host,9,5,1.257697,5.4630,0.5200,1112.9650
   workspace-only probe device,0,6,1.220330,4.8485,1.4920,999.7370
   workspace-only probe device,1,6,1.256560,3.6695,0.6920,1023.7900
   workspace-only probe device,2,6,1.172214,4.1690,0.6310,991.7910
   workspace-only probe device,3,6,1.219053,4.1155,0.6410,1029.8660
   workspace-only probe device,4,6,1.156652,3.8945,0.6550,955.9670
   workspace-only probe device,5,6,1.177314,3.8365,0.6540,951.5540
   workspace-only probe device,6,6,1.234863,3.7045,0.6150,1042.2230
   workspace-only probe device,7,6,1.272606,3.7915,0.6790,1091.6520
   workspace-only probe device,8,6,1.188020,3.1335,0.6540,1013.3360
   workspace-only probe device,9,6,1.285155,4.3520,0.8110,1084.3490

The subsequent ``fresh-phase-ordered-allocation-events.jsonl`` export contains
480 ordered malloc/free observations from these 40 calls. It is available
under ``build/main-performance-20261017/`` with the proof and analysis files;
its SHA256 is ``ce9555e993705252d3ce74dccc671ec6e9798a861129002d9d1ada592ca11741``.
The subsequent read-only audit checked all 26 source-order free slots and all
40 whole-call inventories against the exact ordered events; its record is
``ordered-allocation-audit.json``. Event order does not supply allocation
pointer/byte observations or upgrade owner inference into direct allocation
identity. The original capture artifacts and identities remain below.

.. _main-cuda-phases-20261017-complete:

Current-main same-call CUDA phase diagnostics
---------------------------------------------

All sixteen captures are retained. Each contains ten original whole-call ranges; the scopes remain distinct fresh/API, Session, prepared and workspace contracts.

Phase durations are directly observed same-thread NVTX intervals. Native solve includes Jacobi and CG without a direct boundary; workspace teardown and prepared reset remain unsegmented. No Jacobi segment inference is performed by this reader.

Inner wrappers/markers are included in instrumented wall samples. Nested intervals overlap: never sum them, subtract separate scope medians, infer bandwidth from waiting API time, or interpret these diagnostics as uninstrumented A/B/framework scores.

CUDA API inventories cover the entire capture, including fences outside whole-call ranges, every recorded thread and driver-named extras. Same-thread per-call inclusive rows overlap; deepest-owner rows partition API records only, not exclusive CPU time. Nested runtime/driver API intervals can overlap too. Nonzero API return codes are recorded, not automatically classified as solver failures.

Capture requests use --cuda-trace-all-apis=true and --trace=cuda,cublas,nvtx with CPU sampling/context-switch tracing disabled. The source, frozen driver and Python NVTX wrappers below identify this instrumented diagnostic; no uninstrumented wall-time or framework-score claim follows.

Frozen source and instrumentation provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1

   Identity,Path or value,SHA256 or identity kind
   Compiled source,445275a3994e342b06de4093889f3f814c8706c6,commit
   Compiled tree,5d7ba07832dc7576810ddd5d07d66e4b87ef8109,tree
   Stage identity,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/baseline-stages/cuda-release/identity.json,f4a79e809aa636e857d50b5dce39f9eb11f94910901ea18213cd395a5c066ae2
   Frozen source byte record,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/baseline-stages/cuda-release/source-files.json,2e20b3385012e600bf330957b5e38d1312df0c95b3952f15ecee95f764badf06
   Original profile driver,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/baseline-stages/cuda-release/tools/profile_solve.py,20b805f3bd95a9f0c1a5d23105c17888e4bd285b754575c1540661df87b7e0c4
   Instrumentation/audit source,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/run_fresh_phases.py,24293d962fee3593083e75f4ff822eb8ae68bb30b552db00d2980930cc17545f
   Instrumentation/audit source,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/profile_fresh_phases.py,7ab1cd41d6f8d44f07af5dc0c4c92765c833194a72fd898a0ef0b6088528fa69
   Instrumentation/audit source,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/protocol.py,e9d8ee8e95b06c5a41416f807615b5f23020775702f4cb838dce7dd687ebcb0a
   Instrumentation/audit source,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/fresh-phase-notes.md,d6297ed8796a6f21227d716975eed81c2ba1f51b13f24917b89f429b55046c67
   interpreter,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv-cuda/bin/python,e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f
   nsys,/usr/local/bin/nsys,dff002f06a6068fad71ea64ff84bde41278b5a93b0916d27989bc6c9bb84f535
   plan,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/profiles-main-phases/plan.json,eb3645e62c2b177cd3488a219076510f68b485af637e092e2ffb298a7b897af2
   manifest,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/profiles-main-phases/manifest.json,078c43623dbbb7f7b96db5dfe7382dc8771cffdd87a790d4b6eef841092df520
   execution_claim,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/profiles-main-phases/execution-claim.json,5e4db6ce4e0a6f9149241937ed44e36c01472e902ce2b823a58de66669b857b8
   analyzer,/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/main-performance-20261017/analyze_fresh_phases.py,a8b4a922a6276a522f27b3afbcd348d8b7f873db654edf524554796c70ea0bec

All sixteen separate capture contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1

   Capture,I/M/R,Outer median ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms,Create ranges,Create median ms,API records outside calls
   cuda-cublas-dense-64-api-host-solve-jacobi-standard,5/7/0,4.522307,1180,0.230101,1220,0.692461,10,0.171109,81
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,5/7/0,3.5231825,900,0.25773,950,0.664479,10,0.202281,81
   cuda-cublas-dense-256-api-host-solve-jacobi-standard,4/6/0,4.0038555,1040,0.240257,1080,0.749163,10,0.172046,81
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,4/6/0,3.746681,830,0.265672,880,0.591739,10,0.182597,81
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,4/6/0,3.568377,1040,0.27221,1080,0.789429,10,0.18822,81
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,4/6/0,3.2849895,830,0.190984,880,0.436627,10,0.1746205,81
   cuda-cublas-dense-2048-api-host-solve-jacobi-standard,3/5/0,4.189824,900,0.310686,940,0.53988,10,0.187777,81
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,3/5/0,4.3697625,760,0.327621,810,0.569685,10,0.1975165,81
   cuda-cublas-dense-1024-session-host-solve-jacobi-standard,4/6/0,2.0210045,950,0.221329,970,0.76076,0,not in this contract,81
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,4/6/0,1.4310715,720,0.188444,740,0.454237,0,not in this contract,81
   cuda-cublas-dense-1024-prepared-host-solve-jacobi-standard,4/6/0,2.0678405,870,0.200418,870,0.461513,0,not in this contract,81
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,4/6/0,2.009668,640,0.139237,640,0.420021,0,not in this contract,81
   cuda-native-dense-1024-api-host-solve-jacobi-standard,4/6/0,2.1423195,940,0.195381,980,0.56974,0,not in this contract,81
   cuda-native-dense-1024-api-device-solve-jacobi-standard,4/6/0,2.0250855,730,0.162631,780,0.41336,0,not in this contract,81
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,4/6/0,1.71907,130,0.034832,150,0.221765,10,0.1761225,81
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,4/6/0,1.706485,150,0.037624,180,0.29096,10,0.1917585,81

cuda-cublas-dense-64-api-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``b2e274e240704161b83bff5de50cf7b8ba9e3848b098f027c1ba8b5fccf82a29``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,3.492723,2.279036,4.84377
   cg.phase/output_allocate_initialize,10,0.0953425,0.071313,1.379368
   cg.phase/python_solve_into,10,3.5034845,2.286333,4.854728
   cg.phase/workspace_construct,10,0.4143505,0.307428,0.56477
   cublasCreate_v2,10,0.171109,0.130883,0.236453

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,3450,31.954733,960,0.182722,970,0.513134
   cg.phase/output_allocate_initialize,260,2.391918,80,0.011043,80,0.033493
   cg.phase/python_solve_into,3450,31.954733,960,0.182722,970,0.513134
   cg.phase/workspace_construct,360,3.820426,60,0.018052,50,0.056396
   cublasCreate_v2,210,1.631317,10,0.006681,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,7.988542,7.970795,1.379368,0.412839,3.466457,3.452277,not present,0.138528,116,120
   1,4.89863,4.887514,0.095587,0.341459,3.731661,3.718362,not present,0.135052,116,120
   2,5.621725,5.612962,0.095098,0.510935,4.578138,4.445276,not present,0.137409,116,120
   3,5.736294,5.729642,0.072623,0.430647,4.854728,4.84377,not present,0.21899,116,120
   4,4.666747,4.657104,0.148223,0.436173,3.649049,3.553474,not present,0.199951,116,120
   5,3.075286,3.068241,0.084122,0.307428,2.286333,2.279036,not present,0.130883,116,120
   6,3.414846,3.410294,0.326004,0.352332,2.365174,2.358058,not present,0.175337,116,120
   7,4.377867,4.371618,0.071313,0.371161,3.540512,3.533169,not present,0.166881,116,120
   8,4.026091,4.011806,0.077099,0.415862,2.964323,2.948424,not present,0.235385,116,120
   9,3.641636,3.632966,0.271354,0.56477,2.401243,2.389225,not present,0.236453,116,120

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.006133,0.539,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.012472,12.472,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,2.234398,2.0725,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.060358,0.216,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.04769,0.1855,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,70,3.752296,3.2055,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.00329,0.074,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,1180,0.230101,0.166,"{""0"": 1180}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,550,0.059947,0.085,"{""0"": 550}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,480,2.632331,4.0345,"{""0"": 480}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,60,3.841668,4.9785,"{""0"": 60}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,210,28.370832,41.617,"{""0"": 210}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.147969,13.9945,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.06237,0.223,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,1220,0.692461,0.3345,"{""0"": 1220}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.026002,0.307,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.0128,0.2235,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,230,2.088449,0.6015,"{""0"": 230}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-64-api-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``b2e274e240704161b83bff5de50cf7b8ba9e3848b098f027c1ba8b5fccf82a29``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,2.4526635,1.854034,3.842531
   cg.phase/output_allocate_initialize,10,0.078547,0.0721,1.401758
   cg.phase/python_solve_into,10,2.4603395,1.859549,3.85683
   cg.phase/workspace_construct,10,0.417171,0.384192,0.545274
   cublasCreate_v2,10,0.202281,0.144855,0.26694

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2950,24.097031,660,0.132651,670,0.438704
   cg.phase/output_allocate_initialize,260,3.385637,80,0.078289,80,0.048019
   cg.phase/python_solve_into,2950,24.097031,660,0.132651,670,0.438704
   cg.phase/workspace_construct,390,3.905197,70,0.022661,60,0.078883
   cublasCreate_v2,210,1.882505,10,0.006263,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,5.60136,5.578877,0.522782,0.495413,3.85683,3.842531,not present,0.144855,88,93
   1,5.032897,4.921813,0.688925,0.545274,3.144351,3.13115,not present,0.26694,88,93
   2,4.917076,4.910433,1.401758,0.475039,2.639567,2.630457,not present,0.209483,88,93
   3,3.187149,3.180999,0.075276,0.402274,2.306352,2.297513,not present,0.208833,88,93
   4,2.853173,2.848038,0.072836,0.423366,1.977309,1.970249,not present,0.216016,88,93
   5,3.208516,3.201153,0.0721,0.413937,2.285376,2.277593,not present,0.233697,88,93
   6,2.671224,2.665631,0.072698,0.384192,1.859549,1.854034,not present,0.184685,88,93
   7,3.948074,3.942702,0.559693,0.384362,2.614327,2.607814,not present,0.190752,88,93
   8,3.837849,3.829774,0.075074,0.39619,2.955056,2.946786,not present,0.195729,88,93
   9,2.896008,2.89039,0.081818,0.420405,2.042285,2.03474,not present,0.185426,88,93

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.005706,0.476,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.008412,8.412,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.058742,1.559,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.064402,0.2225,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.056724,0.188,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,80,4.256732,2.284,"{""0"": 80}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.004522,0.0895,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,900,0.25773,0.18,"{""0"": 900}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,690,0.070655,0.081,"{""0"": 690}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,620,4.444288,4.1335,"{""0"": 620}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,70,3.849144,3.5155,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,120,18.038877,45.377,"{""0"": 120}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.432938,16.7735,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.049188,0.2205,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,950,0.664479,0.3535,"{""0"": 950}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.213942,0.419,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.01453,0.27,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,140,2.810071,0.8105,"{""0"": 140}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-256-api-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``6a6622788ec7f29156523e562995e09db8ec35032185f8c76ae6fccf2165a4a0``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,2.782109,1.924795,4.180144
   cg.phase/output_allocate_initialize,10,0.0782175,0.07329,0.636563
   cg.phase/python_solve_into,10,2.8831275,1.932543,4.188176
   cg.phase/workspace_construct,10,0.383186,0.325296,0.755442
   cublasCreate_v2,10,0.172046,0.111021,0.370838

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2940,27.253281,820,0.172172,830,0.441586
   cg.phase/output_allocate_initialize,260,1.705538,80,0.031146,80,0.034725
   cg.phase/python_solve_into,2940,27.253281,820,0.172172,830,0.441586
   cg.phase/workspace_construct,360,4.255448,60,0.018485,50,0.08614
   cublasCreate_v2,210,1.859426,10,0.007011,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,5.909223,5.828726,0.470391,0.755442,3.853667,3.824515,not present,0.370838,102,106
   1,3.448574,3.440806,0.191196,0.745123,2.104411,2.094205,not present,0.241188,102,106
   2,3.338236,3.322649,0.077443,0.325296,2.20438,2.193472,not present,0.111021,102,106
   3,3.736126,3.729509,0.253648,0.482782,2.603437,2.581504,not present,0.170372,102,106
   4,4.274192,4.265252,0.073566,0.353366,3.474142,3.461689,not present,0.159967,102,106
   5,2.756292,2.750951,0.078992,0.363228,1.932543,1.924795,not present,0.146141,102,106
   6,4.713778,4.707514,0.636563,0.506349,3.162818,2.982714,not present,0.259088,102,106
   7,4.962008,4.954188,0.07329,0.338912,4.188176,4.180144,not present,0.155543,102,106
   8,4.271585,4.263883,0.075832,0.403144,3.464116,3.45212,not present,0.209854,102,106
   9,2.837434,2.83223,0.076659,0.35573,2.072447,2.06391,not present,0.17372,102,106

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.006588,0.5595,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.023616,23.616,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.281599,1.9115,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.06334,0.2365,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.046906,0.188,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,70,3.253205,3.1095,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.003508,0.079,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,1040,0.240257,0.167,"{""0"": 1040}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,460,0.056202,0.086,"{""0"": 460}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,400,3.04869,4.1515,"{""0"": 400}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,60,4.179555,5.799,"{""0"": 60}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,180,23.386385,42.8,"{""0"": 180}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.210105,17.0615,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.064188,0.238,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,1080,0.749163,0.346,"{""0"": 1080}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.03729,0.3005,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.013902,0.225,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,200,1.348756,0.7,"{""0"": 200}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-256-api-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``6a6622788ec7f29156523e562995e09db8ec35032185f8c76ae6fccf2165a4a0``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,2.7588685,1.795814,3.528342
   cg.phase/output_allocate_initialize,10,0.080838,0.065144,0.58961
   cg.phase/python_solve_into,10,2.8249095,1.803351,3.60498
   cg.phase/workspace_construct,10,0.383,0.331616,0.929731
   cublasCreate_v2,10,0.182597,0.096347,0.264084

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2580,25.981278,590,0.214732,600,0.257621
   cg.phase/output_allocate_initialize,260,1.266404,80,0.012583,80,0.182924
   cg.phase/python_solve_into,2580,25.981278,590,0.214732,600,0.257621
   cg.phase/workspace_construct,390,4.252705,70,0.018344,60,0.056762
   cublasCreate_v2,210,1.701765,10,0.006225,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,4.855274,4.839463,0.58961,0.602248,3.133101,3.119299,not present,0.264084,81,86
   1,4.026044,4.016452,0.078561,0.458556,2.947946,2.939627,not present,0.096347,81,86
   2,4.98215,4.974304,0.079548,0.929731,3.60498,3.528342,not present,0.238602,81,86
   3,2.70977,2.704011,0.082128,0.378427,1.872904,1.864294,not present,0.18236,81,86
   4,3.528118,3.523367,0.069061,0.331616,2.811192,2.804506,not present,0.15887,81,86
   5,4.229721,4.221488,0.089906,0.387573,3.338469,3.33116,not present,0.190888,81,86
   6,3.931099,3.921985,0.2559,0.406376,2.838627,2.688786,not present,0.182834,81,86
   7,3.562263,3.555566,0.084629,0.352472,2.721049,2.713231,not present,0.155074,81,86
   8,2.5664,2.561627,0.073512,0.34487,1.803351,1.795814,not present,0.174213,81,86
   9,3.176413,3.171161,0.065144,0.372712,2.40226,2.396046,not present,0.188251,81,86

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.005277,0.484,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.025092,25.092,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.053947,1.372,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.079584,0.224,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.043436,0.1835,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,80,3.185081,2.025,"{""0"": 80}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.004394,0.0765,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,830,0.265672,0.159,"{""0"": 830}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,580,0.055069,0.079,"{""0"": 580}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,520,2.649314,3.791,"{""0"": 520}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,70,4.167914,3.2975,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,110,22.766271,47.553,"{""0"": 110}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.192348,13.947,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.047086,0.222,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,880,0.591739,0.3315,"{""0"": 880}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.024796,0.311,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.016454,0.224,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,130,0.726812,0.705,"{""0"": 130}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-api-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,2.5317145,2.212854,4.446783
   cg.phase/output_allocate_initialize,10,0.157808,0.069797,1.948018
   cg.phase/python_solve_into,10,2.5415465,2.221416,4.459115
   cg.phase/workspace_construct,10,0.401005,0.328063,0.619794
   cublasCreate_v2,10,0.18822,0.174087,0.346621

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2940,27.658591,820,0.216385,830,0.54603
   cg.phase/output_allocate_initialize,260,3.365918,80,0.01336,80,0.051269
   cg.phase/python_solve_into,2940,27.658591,820,0.216385,830,0.54603
   cg.phase/workspace_construct,360,3.883987,60,0.018953,50,0.069242
   cublasCreate_v2,210,1.978741,10,0.006603,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,6.075203,6.037562,0.749801,0.619794,3.705718,3.544597,not present,0.279685,102,106
   1,7.279972,7.26113,1.948018,0.387211,4.459115,4.446783,not present,0.199483,102,106
   2,3.09476,3.089299,0.090614,0.381683,2.221416,2.212854,not present,0.177305,102,106
   3,3.271748,3.265377,0.069797,0.328063,2.433406,2.426737,not present,0.174087,102,106
   4,3.570242,3.562444,0.076265,0.397576,2.632671,2.62455,not present,0.209638,102,106
   5,3.566512,3.557289,0.217616,0.452265,2.450422,2.438879,not present,0.191289,102,106
   6,4.27232,4.263937,0.100803,0.408864,3.333366,3.228807,not present,0.184781,102,106
   7,3.375948,3.37098,0.092642,0.404434,2.407499,2.400102,not present,0.17957,102,106
   8,3.540388,3.534578,0.436983,0.379564,2.363176,2.356243,not present,0.185151,102,106
   9,5.089177,5.082272,0.214813,0.529144,3.847134,3.836087,not present,0.346621,102,106

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.005831,0.57,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.007894,7.894,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.338876,1.873,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.059883,0.2115,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.047381,0.186,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,70,3.642457,3.6495,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.004609,0.075,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,1040,0.27221,0.177,"{""0"": 1040}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,460,0.052249,0.085,"{""0"": 460}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,400,2.529514,4.3495,"{""0"": 400}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,60,3.901943,5.249,"{""0"": 60}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,180,24.037489,48.5995,"{""0"": 180}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.15388,15.3045,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.049159,0.2545,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,1080,0.789429,0.357,"{""0"": 1080}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.028984,0.319,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.015207,0.233,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,200,3.132949,0.871,"{""0"": 200}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-api-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,2.21888,1.877363,2.592533
   cg.phase/output_allocate_initialize,10,0.0856515,0.066094,0.498362
   cg.phase/python_solve_into,10,2.258763,1.885459,2.598954
   cg.phase/workspace_construct,10,0.380009,0.301966,0.515626
   cublasCreate_v2,10,0.1746205,0.125761,0.311135

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2580,21.489313,590,0.141068,600,0.242039
   cg.phase/output_allocate_initialize,260,1.581727,80,0.013749,80,0.029417
   cg.phase/python_solve_into,2580,21.489313,590,0.141068,600,0.242039
   cg.phase/workspace_construct,390,3.578924,70,0.01724,60,0.070409
   cublasCreate_v2,210,1.696737,10,0.005418,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,3.540794,3.527342,0.304635,0.462393,2.3395,2.326561,not present,0.172882,81,86
   1,2.945396,2.93874,0.07246,0.359244,1.885459,1.877363,not present,0.176359,81,86
   2,3.057847,3.05236,0.209087,0.376578,2.11829,2.111199,not present,0.18163,81,86
   3,3.088393,3.082886,0.073017,0.343277,2.109365,2.101839,not present,0.161146,81,86
   4,3.7766,3.769373,0.498362,0.356313,2.531836,2.525361,not present,0.148676,81,86
   5,3.349483,3.344389,0.082985,0.38344,2.537187,2.529775,not present,0.171748,81,86
   6,2.750905,2.746065,0.068859,0.393139,1.935261,1.929019,not present,0.186234,81,86
   7,3.220496,3.215068,0.066094,0.301966,2.451182,2.445015,not present,0.125761,81,86
   8,3.405836,3.399543,0.308058,0.515626,2.178026,2.096758,not present,0.311135,81,86
   9,3.427053,3.422617,0.088318,0.388469,2.598954,2.592533,not present,0.180573,81,86

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.004187,0.3655,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.008473,8.473,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.046676,1.1865,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.060077,0.2215,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.041218,0.185,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,80,3.485921,2.072,"{""0"": 80}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.00376,0.0755,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,830,0.190984,0.164,"{""0"": 830}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,580,0.053277,0.079,"{""0"": 580}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,520,2.422085,3.8785,"{""0"": 520}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,70,3.508591,3.128,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,110,18.55103,66.6645,"{""0"": 110}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.142163,13.436,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.046884,0.228,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,880,0.436627,0.34,"{""0"": 880}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.022157,0.2965,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.012159,0.2095,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,130,1.320603,0.628,"{""0"": 130}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-2048-api-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,3.089844,2.422037,4.434274
   cg.phase/output_allocate_initialize,10,0.089906,0.069678,0.518624
   cg.phase/python_solve_into,10,3.1014655,2.429369,4.443336
   cg.phase/workspace_construct,10,0.3992465,0.310898,0.896457
   cublasCreate_v2,10,0.187777,0.146637,0.61399

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2430,31.15393,680,0.128797,690,0.342312
   cg.phase/output_allocate_initialize,260,1.654998,80,0.120373,80,0.041068
   cg.phase/python_solve_into,2430,31.15393,680,0.128797,690,0.342312
   cg.phase/workspace_construct,360,4.666271,60,0.019339,50,0.05983
   cublasCreate_v2,210,2.504139,10,0.006678,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,6.641482,6.625324,0.518624,0.896457,4.329657,4.316288,not present,0.47825,88,92
   1,5.036526,5.02761,0.400116,0.864392,3.323019,3.31069,not present,0.61399,88,92
   2,3.336757,3.33002,0.087598,0.310898,2.448412,2.440279,not present,0.146637,88,92
   3,3.990948,3.983651,0.095794,0.633808,2.879912,2.868998,not present,0.329697,88,92
   4,4.872001,4.864514,0.071582,0.415123,3.96678,3.955827,not present,0.186434,88,92
   5,3.274707,3.26753,0.083751,0.330122,2.468275,2.461581,not present,0.151431,88,92
   6,3.85904,3.854066,0.40117,0.405099,2.602479,2.594919,not present,0.18912,88,92
   7,5.322915,5.315043,0.069678,0.368454,4.443336,4.434274,not present,0.172218,88,92
   8,4.3887,4.381527,0.092214,0.38448,3.485227,3.47759,not present,0.177569,88,92
   9,3.280786,3.275362,0.077519,0.393394,2.429369,2.422037,not present,0.194953,88,92

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.006047,0.466,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.007226,7.226,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.502174,1.7055,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.05889,0.216,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.048712,0.196,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,70,3.361304,3.1295,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.003664,0.0775,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,900,0.310686,0.166,"{""0"": 900}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,370,0.039805,0.085,"{""0"": 370}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,320,1.942186,4.0905,"{""0"": 320}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,60,4.601393,4.876,"{""0"": 60}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,150,28.790116,50.062,"{""0"": 150}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.142076,13.85,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.048259,0.2555,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,940,0.53988,0.348,"{""0"": 940}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.024757,0.3595,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.014663,0.2495,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,170,1.053241,0.7925,"{""0"": 170}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-2048-api-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,3.306422,2.232793,4.350023
   cg.phase/output_allocate_initialize,10,0.1821885,0.078344,0.534204
   cg.phase/python_solve_into,10,3.316471,2.240066,4.364595
   cg.phase/workspace_construct,10,0.4332715,0.367799,0.601505
   cublasCreate_v2,10,0.1975165,0.181708,0.2652

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2210,29.847618,520,0.264717,530,0.347084
   cg.phase/output_allocate_initialize,260,1.998371,80,0.011989,80,0.067781
   cg.phase/python_solve_into,2210,29.847618,520,0.264717,530,0.347084
   cg.phase/workspace_construct,390,4.095484,70,0.031078,60,0.058909
   cublasCreate_v2,210,1.892307,10,0.01829,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,6.289789,6.271505,0.42851,0.601505,4.364595,4.350023,not present,0.2652,74,79
   1,4.871898,4.863657,0.28936,0.367799,3.756461,3.746074,not present,0.190802,74,79
   2,3.452543,3.446674,0.080048,0.491395,2.423634,2.412068,not present,0.234037,74,79
   3,3.67105,3.659192,0.2727,0.397475,2.500485,2.490428,not present,0.215262,74,79
   4,4.465555,4.456246,0.255853,0.448209,3.308339,3.298011,not present,0.196266,74,79
   5,3.19535,3.188421,0.078344,0.418334,2.276803,2.26754,not present,0.191266,74,79
   6,5.332566,5.327489,0.534204,0.388045,4.074302,3.987667,not present,0.181708,74,79
   7,4.370479,4.362485,0.083903,0.495601,3.324603,3.314833,not present,0.198767,74,79
   8,4.369046,4.361725,0.108524,0.448309,3.358372,3.349913,not present,0.201482,74,79
   9,3.082678,3.076099,0.079778,0.405251,2.240066,2.232793,not present,0.18373,74,79

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.004877,0.497,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.009573,9.573,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.240712,1.5905,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.063382,0.235,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.058098,0.1895,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,80,3.464915,2.6725,"{""0"": 80}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.003953,0.075,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,760,0.327621,0.168,"{""0"": 760}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,470,0.065266,0.086,"{""0"": 470}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,420,3.652879,4.1135,"{""0"": 420}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,70,4.018893,3.964,"{""0"": 70}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,100,25.448795,152.929,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.143136,14.1735,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.058392,0.248,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,810,0.569685,0.328,"{""0"": 810}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.026634,0.316,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.013697,0.2385,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,120,1.633816,0.929,"{""0"": 120}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-session-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,1.8087865,1.132996,3.303384
   cg.phase/output_allocate_initialize,10,0.195053,0.046316,1.683449
   cg.phase/python_solve_into,10,1.818879,1.136255,3.370101

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2940,17.495641,820,0.177457,830,0.46945
   cg.phase/output_allocate_initialize,260,2.986135,80,0.028653,80,0.056575
   cg.phase/python_solve_into,2940,17.495641,820,0.177457,830,0.46945

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,3.997792,3.981293,0.488437,not present,3.370101,3.303384,not present,not present,93,95
   1,2.955313,2.949994,0.271235,not present,2.617587,2.609884,not present,not present,93,95
   2,3.883821,3.87919,1.683449,not present,2.132674,2.120934,not present,not present,93,95
   3,2.905075,2.899573,0.470888,not present,2.368475,2.359294,not present,not present,93,95
   4,1.52577,1.521663,0.285563,not present,1.197621,1.191514,not present,not present,93,95
   5,1.23429,1.232655,0.051582,not present,1.161464,1.157775,not present,not present,93,95
   6,1.219702,1.217999,0.063502,not present,1.136255,1.132996,not present,not present,93,95
   7,2.379733,2.376595,0.046316,not present,2.279486,2.274369,not present,not present,93,95
   8,1.662276,1.658486,0.118871,not present,1.505084,1.496639,not present,not present,93,95
   9,1.475588,1.471909,0.075555,not present,1.350737,1.344504,not present,not present,93,95

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.008297,8.297,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,20,0.099593,4.0055,"{""0"": 20}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,30,0.003356,0.077,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,950,0.221329,0.177,"{""0"": 950}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,460,0.048895,0.086,"{""0"": 460}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,400,3.03601,4.1945,"{""0"": 400}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,20,0.154759,4.318,"{""0"": 20}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,180,13.183329,45.8025,"{""0"": 180}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.481165,7.1425,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.151654,0.243,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,970,0.76076,0.364,"{""0"": 970}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,30,0.017846,0.4395,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,30,0.010548,0.281,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,200,2.639387,0.778,"{""0"": 200}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-session-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,1.145812,0.921883,3.290812
   cg.phase/output_allocate_initialize,10,0.081332,0.04951,0.645866
   cg.phase/python_solve_into,10,1.1530335,0.926386,3.297003

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2580,13.948014,590,0.148418,600,0.30093
   cg.phase/output_allocate_initialize,260,1.470809,80,0.028138,80,0.121932
   cg.phase/python_solve_into,2580,13.948014,590,0.148418,600,0.30093

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,1.521983,1.512184,0.298371,not present,1.112074,1.103229,not present,not present,70,72
   1,1.34016,1.336237,0.10687,not present,1.193993,1.188395,not present,not present,70,72
   2,2.545619,2.542191,0.060348,not present,2.394653,2.388554,not present,not present,70,72
   3,1.591555,1.588283,0.248481,not present,1.3059,1.299248,not present,not present,70,72
   4,1.037707,1.035184,0.084694,not present,0.926386,0.921883,not present,not present,70,72
   5,1.060266,1.05866,0.055714,not present,0.971686,0.968234,not present,not present,70,72
   6,3.392058,3.388248,0.04951,not present,3.297003,3.290812,not present,not present,70,72
   7,1.049391,1.047102,0.07797,not present,0.943845,0.93858,not present,not present,70,72
   8,1.077198,1.074553,0.056694,not present,0.984782,0.980753,not present,not present,70,72
   9,2.739592,2.736563,0.645866,not present,2.050122,2.041167,not present,not present,70,72

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.034621,34.621,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,20,0.068942,2.6275,"{""0"": 20}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,30,0.003152,0.0835,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,720,0.188444,0.187,"{""0"": 720}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,580,0.059606,0.0895,"{""0"": 580}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,520,2.922975,4.0205,"{""0"": 520}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,20,0.091926,3.328,"{""0"": 20}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,110,10.407012,59.6775,"{""0"": 110}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.131363,8.745,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.050621,0.2375,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,740,0.454237,0.421,"{""0"": 740}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,30,0.016971,0.385,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,30,0.009422,0.26,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,130,1.128774,0.8295,"{""0"": 130}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-prepared-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/prepared_solve_into,10,1.862622,1.182444,4.296079

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/prepared_solve_into,2750,18.322737,770,0.180507,770,0.395039

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,4.66663,4.656045,not present,not present,not present,not present,4.296079,not present,85,85
   1,1.447553,1.442368,not present,not present,not present,not present,1.331753,not present,85,85
   2,2.044943,2.035531,not present,not present,not present,not present,1.86765,not present,85,85
   3,2.684717,2.680488,not present,not present,not present,not present,2.473944,not present,85,85
   4,1.310531,1.295415,not present,not present,not present,not present,1.220992,not present,85,85
   5,2.090738,2.086631,not present,not present,not present,not present,2.001353,not present,85,85
   6,2.381514,2.378515,not present,not present,not present,not present,1.857594,not present,85,85
   7,1.249552,1.247753,not present,not present,not present,not present,1.182444,not present,85,85
   8,1.258713,1.256528,not present,not present,not present,not present,1.202501,not present,85,85
   9,2.344375,2.341175,not present,not present,not present,not present,2.274978,not present,85,85

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.255505,255.505,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,30,0.002944,0.074,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,870,0.200418,0.19,"{""0"": 870}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,450,0.074535,0.101,"{""0"": 450}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,390,2.68445,4.554,"{""0"": 390}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,170,14.965197,51.411,"{""0"": 170}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,80,0.0484,0.258,"{""0"": 80}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,870,0.461513,0.444,"{""0"": 870}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,30,0.019047,0.362,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,30,0.009701,0.257,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,190,1.236362,0.837,"{""0"": 190}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/prepared_solve_into,10,1.7664335,0.918439,2.86608

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/prepared_solve_into,2390,15.594897,540,0.119127,540,0.317406

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,2.520325,2.508834,not present,not present,not present,not present,1.948717,not present,62,62
   1,3.09417,3.083456,not present,not present,not present,not present,2.86608,not present,62,62
   2,2.330563,2.324358,not present,not present,not present,not present,2.220585,not present,62,62
   3,1.59312,1.589024,not present,not present,not present,not present,1.502969,not present,62,62
   4,3.270682,3.265386,not present,not present,not present,not present,2.810961,not present,62,62
   5,1.1944,1.190444,not present,not present,not present,not present,1.089612,not present,62,62
   6,2.915969,2.912328,not present,not present,not present,not present,2.819609,not present,62,62
   7,1.077734,1.073601,not present,not present,not present,not present,0.967555,not present,62,62
   8,1.002871,0.999889,not present,not present,not present,not present,0.918439,not present,62,62
   9,1.688773,1.682989,not present,not present,not present,not present,1.58415,not present,62,62

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.008033,8.033,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,30,0.003498,0.078,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,640,0.139237,0.181,"{""0"": 640}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,570,0.173991,0.089,"{""0"": 570}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,510,4.493437,4.106,"{""0"": 510}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,100,10.756175,68.2635,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,80,0.039866,0.233,"{""0"": 80}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,640,0.420021,0.438,"{""0"": 640}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,30,0.096831,0.2675,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,30,0.010299,0.2195,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,120,0.90565,0.8385,"{""0"": 120}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-native-dense-1024-api-host-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,1.793592,1.231758,3.886703
   cg.phase/output_allocate_initialize,10,0.0950765,0.046277,0.723928
   cg.phase/python_solve_into,10,1.7975265,1.234446,3.892139
   cg.phase/workspace_construct,10,0.0116125,0.007105,0.025255

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2640,20.439719,760,0.162707,770,0.344902
   cg.phase/output_allocate_initialize,260,2.139561,80,0.014455,80,0.148919
   cg.phase/python_solve_into,2640,20.439719,760,0.162707,770,0.344902
   cg.phase/workspace_construct,110,0.07094,30,0.00546,30,0.009815

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,3.750804,3.737954,0.499958,0.018337,3.099126,3.088387,not present,not present,92,96
   1,4.023547,4.017564,0.723928,0.025255,3.194353,3.183976,not present,not present,92,96
   2,1.414224,1.411217,0.069746,0.00986,1.300344,1.296049,not present,not present,92,96
   3,2.130012,2.128245,0.507091,0.011457,1.584733,1.581186,not present,not present,92,96
   4,1.317096,1.315167,0.052028,0.007211,1.234446,1.231758,not present,not present,92,96
   5,4.010752,4.006132,0.046277,0.007105,3.892139,3.886703,not present,not present,92,96
   6,1.639045,1.634046,0.094097,0.012468,1.47055,1.453048,not present,not present,92,96
   7,2.154627,2.151805,0.082161,0.011768,2.01032,2.005998,not present,not present,92,96
   8,2.646714,2.642366,0.185495,0.016799,2.382086,2.376195,not present,not present,92,96
   9,1.496113,1.493985,0.096056,0.011426,1.361158,1.357564,not present,not present,92,96

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.007972,7.972,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,40,0.087536,1.2745,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.004404,0.0795,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,940,0.195381,0.171,"{""0"": 940}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,340,0.032021,0.081,"{""0"": 340}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,340,2.154419,4.14,"{""0"": 340}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,40,0.130897,1.808,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,180,17.332004,49.4435,"{""0"": 180}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.35201,9.371,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.039464,0.2195,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,980,0.56974,0.3355,"{""0"": 980}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.036605,0.3305,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.013982,0.2475,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,200,1.863839,0.784,"{""0"": 200}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-native-dense-1024-api-device-solve-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/native_solve_into_jacobi_and_cg,10,1.740704,1.019456,5.08593
   cg.phase/output_allocate_initialize,10,0.081462,0.052324,0.473286
   cg.phase/python_solve_into,10,1.7491335,1.023301,5.099208
   cg.phase/workspace_construct,10,0.013616,0.009985,0.091077

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/native_solve_into_jacobi_and_cg,2280,19.998129,530,0.122713,540,0.257161
   cg.phase/output_allocate_initialize,260,1.192064,80,0.015424,80,0.092038
   cg.phase/python_solve_into,2280,19.998129,530,0.122713,540,0.257161
   cg.phase/workspace_construct,140,0.153966,40,0.00812,40,0.014279

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,5.772035,5.743229,0.473286,0.022852,5.099208,5.08593,not present,not present,71,76
   1,2.221925,2.215948,0.292488,0.091077,1.765339,1.755014,not present,not present,71,76
   2,2.996858,2.991546,0.126642,0.013617,2.505308,2.494877,not present,not present,71,76
   3,1.988205,1.984329,0.172659,0.017818,1.732928,1.726394,not present,not present,71,76
   4,1.589553,1.586908,0.076882,0.013158,1.465905,1.450237,not present,not present,71,76
   5,2.061966,2.059146,0.062938,0.012374,1.931311,1.92583,not present,not present,71,76
   6,1.534786,1.531674,0.0696,0.013103,1.41241,1.40795,not present,not present,71,76
   7,1.170727,1.168592,0.085166,0.030097,1.023301,1.019456,not present,not present,71,76
   8,3.250016,3.246558,0.052324,0.009985,3.13683,3.129927,not present,not present,71,76
   9,1.177648,1.160456,0.077758,0.013615,1.023997,1.019496,not present,not present,71,76

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.007176,7.176,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,50,0.131451,1.193,"{""0"": 50}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,40,0.074653,0.0915,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,730,0.162631,0.18,"{""0"": 730}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetLastError_v3020,460,0.045423,0.081,"{""0"": 460}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaLaunchKernel_v7000,460,4.580445,4.084,"{""0"": 460}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,50,0.268219,1.722,"{""0"": 50}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemcpyAsync_v3020,110,14.816669,93.14,"{""0"": 110}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMemsetAsync_v3020,10,0.174511,7.049,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaPointerGetAttributes_v4000,100,0.040964,0.2395,"{""0"": 100}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,780,0.41336,0.3375,"{""0"": 780}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,40,0.023172,0.427,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,40,0.013526,0.302,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,130,0.835711,0.8935,"{""0"": 130}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-api-host-workspace-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/workspace_construct,10,0.3874105,0.336147,0.738462
   cublasCreate_v2,10,0.1761225,0.126042,0.212686

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/workspace_construct,360,4.013738,60,0.019161,50,0.061785
   cublasCreate_v2,210,1.57906,10,0.007751,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,2.130496,2.109261,not present,0.738462,not present,not present,not present,0.200036,11,13
   1,1.941107,1.879134,not present,0.421243,not present,not present,not present,0.158943,11,13
   2,2.139624,2.127096,not present,0.60254,not present,not present,not present,0.212686,11,13
   3,1.868042,1.860987,not present,0.526159,not present,not present,not present,0.126042,11,13
   4,1.621006,1.616008,not present,0.336147,not present,not present,not present,0.173649,11,13
   5,1.687876,1.683181,not present,0.385932,not present,not present,not present,0.184414,11,13
   6,1.72929,1.72027,not present,0.343611,not present,not present,not present,0.18082,11,13
   7,1.70885,1.703036,not present,0.388889,not present,not present,not present,0.178596,11,13
   8,1.68641,1.68101,not present,0.350132,not present,not present,not present,0.154624,11,13
   9,1.683162,1.675306,not present,0.366308,not present,not present,not present,0.149882,11,13

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.005094,0.5175,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.009027,9.027,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.239399,1.787,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.055005,0.2105,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.041978,0.1765,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,50,12.865028,4.8185,"{""0"": 50}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,30,0.002664,0.062,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,130,0.034832,0.165,"{""0"": 130}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,40,3.858437,79.484,"{""0"": 40}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,150,0.221765,0.3195,"{""0"": 150}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,30,0.023746,0.2585,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,30,0.009221,0.2055,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,20,0.023444,1.226,"{""0"": 20}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

cuda-cublas-dense-1024-api-device-workspace-jacobi-standard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input SHA256: ``204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb``.

Observed nested NVTX ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Range,Count,Median ms,Min ms,Max ms
   cg.phase/workspace_construct,10,0.42654,0.364845,0.664829
   cublasCreate_v2,10,0.1917585,0.158146,0.300759

Same-thread phase API inventories (inclusive and overlapping)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Phase,CUDA API records,Sum recorded API ms,GetDevice calls,GetDevice total ms,SetDevice calls,SetDevice total ms
   cg.phase/workspace_construct,390,4.252708,70,0.018219,60,0.118817
   cublasCreate_v2,210,1.856708,10,0.00551,0,0.0

Every original call (overlapping phase durations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Call,Outer ms,Instrumented wall ms,initial_output,workspace,python_solve,native_solve,prepared_solve,Create ms,GetDevice calls,SetDevice calls
   0,1.968224,1.95164,not present,0.664829,not present,not present,not present,0.243591,13,16
   1,1.709255,1.701336,not present,0.391162,not present,not present,not present,0.180457,13,16
   2,1.746565,1.737743,not present,0.510492,not present,not present,not present,0.300759,13,16
   3,1.703715,1.697059,not present,0.435827,not present,not present,not present,0.180208,13,16
   4,1.654818,1.647077,not present,0.439044,not present,not present,not present,0.191946,13,16
   5,1.630381,1.624257,not present,0.390637,not present,not present,not present,0.191846,13,16
   6,1.702916,1.696844,not present,0.417253,not present,not present,not present,0.20151,13,16
   7,1.956388,1.949212,not present,0.629271,not present,not present,not present,0.191671,13,16
   8,1.615263,1.608616,not present,0.377234,not present,not present,not present,0.158146,13,16
   9,1.716107,1.706288,not present,0.364845,not present,not present,not present,0.16606,13,16

Complete global CUDA API inventory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header-rows: 1

   Activity table,Exact raw API,Calls,Total API ms,Median API us,Return values
   CUPTI_ACTIVITY_KIND_RUNTIME,cuCtxGetLimit,10,0.006097,0.426,"{""0"": 10}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cuProfilerStart,1,0.009321,9.321,"{""0"": 1}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaDeviceSynchronize_v3020,30,0.073945,1.38,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventCreateWithFlags_v3020,160,0.062879,0.218,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaEventDestroy_v3020,160,0.039611,0.172,"{""0"": 160}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaFree_v3020,60,12.182767,3.9595,"{""0"": 60}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDeviceCount_v3020,30,0.002527,0.064,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaGetDevice_v3020,150,0.037624,0.1435,"{""0"": 150}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaMalloc_v3020,50,4.032865,4.6695,"{""0"": 50}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaSetDevice_v3020,180,0.29096,0.3185,"{""0"": 180}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetDevice_v12080,30,0.022766,0.2605,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamGetFlags_v5050,30,0.009303,0.207,"{""0"": 30}"
   CUPTI_ACTIVITY_KIND_RUNTIME,cudaStreamSynchronize_v3020,20,0.025372,1.3095,"{""0"": 20}"

The analysis JSON retains every same-thread per-call inclusive/deepest-owner API inventory and every unassigned API observation. Original SQLite/raw/sidecar/log/CSV hashes remain attached; API totals are not exclusive wall time.

Analyzer SHA256: ``a8b4a922a6276a522f27b3afbcd348d8b7f873db654edf524554796c70ea0bec``.
