CPU and CUDA optimization evaluation — 2026-10-16
=================================================

Fused CPU native has lower medians than NumPy/Torch in
**10/16 and 12/16** using process medians; the strict tables' pooled estimator
gives **8/16 and 12/16**, respectively. Explicit device-mode native CUDA medians
are lower than the same-artifact eager Torch reference in **52/52** cases:
Torch/native ratios are
1.33–2.47× dense, 2.16–3.18× CSR and 2.12–3.10× scaled/neutral sparse.
Device cuBLAS has lower medians in **12/16**; all four Jacobi API losses remain
visible. Device medians are lower in **68/68** native host/device pairs,
a descriptive comparison of different mode options, without confidence
intervals. The original strict CPU/CUDA batch verdicts remain regression;
the focused six-block pooled follow-up remains inconclusive.

All **216 primary native rows** are shown: 32 CPU A/B rows including separate
operation controls, 48 CUDA API A/B rows, and 136 explicit CUDA host/device
rows. The primary evidence cohorts contain **112 fresh-process artifacts /
1,184 eligible candidate/case results**. A separately disclosed focused
confirmation adds **8 artifacts / 96 results**, for **120 artifacts / 1,280
results** overall, retaining all original observations. Instrumented profiles
and fixed-input probes are separate diagnostics, outside these scores.
Whole-batch strict verdicts, framework controls, remaining losses and changed
work are retained. CUDA ABBA combines four complete dense cohorts and four
separately launched complete CSR cohorts after two disclosed metadata-audit
stops; retained partial cohorts are excluded explicitly. No universal
improvement or complete strict-gate pass is inferred from selected medians.

CPU: the saved overall verdict is **regression**.
Native rows contain 25 within-budget,
7 inconclusive, 0 regression and
0 comparability-failed cases. Unchanged framework
controls contain 4 regression, 40
inconclusive and 0 comparability-failed cases.
Within budget permits up to 5% slowdown and is not a speedup claim.

CUDA API: the saved overall verdict is **regression**.
Native rows contain 20 within-budget,
26 inconclusive, 2 regression and
0 comparability-failed cases. Unchanged framework
controls contain 1 regression, 18
inconclusive and 0 comparability-failed cases.
Within budget permits up to 5% slowdown and is not a speedup claim.

Host, correctness and timing boundaries
---------------------------------------

Host: 13th Gen Intel(R) Core(TM) i7-13700K; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39;
CPython 3.12.3. CUDA runtime identity:

.. code-block:: json

   {
     "compute_capability": [
       7,
       5
     ],
     "device": 0,
     "driver_api_version": 13010,
     "name": "NVIDIA GeForce RTX 2060",
     "runtime_version": 12080,
     "total_memory": 6442123264,
     "uuid": "GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302"
   }

.. csv-table:: Actual compiler and installed package metadata
   :class: benchmark-results
   :header: "environment","CPU compiler","CUDA compiler / host","packages"

   "CPU ABBA","GNU 16.0.1","{}","{""cgpy"": ""0.4.0"", ""numpy"": ""2.2.6"", ""scipy"": ""1.15.3"", ""threadpoolctl"": ""3.6.0"", ""torch"": ""2.8.0+cpu""}"
   "CUDA API ABBA","GNU 13.3.0","{""architectures"": ""75"", ""build_type"": ""Release"", ""compiler"": ""NVIDIA 12.8.93"", ""enabled"": true, ""flags"": "" -O3 -DNDEBUG"", ""host_compiler"": ""GNU 13.3.0"", ""scalar_dtype"": ""float64"", ""toolkit_version"": ""12.8.93""}","{""cgpy"": ""0.4.0"", ""numpy"": ""2.2.6"", ""scipy"": ""1.15.3"", ""threadpoolctl"": ""3.6.0"", ""torch"": ""2.8.0+cu128""}"
   "CUDA modes","GNU 13.3.0","{""architectures"": ""75"", ""build_type"": ""Release"", ""compiler"": ""NVIDIA 12.8.93"", ""enabled"": true, ""flags"": "" -O3 -DNDEBUG"", ""host_compiler"": ""GNU 13.3.0"", ""scalar_dtype"": ""float64"", ""toolkit_version"": ""12.8.93""}","{""cgpy"": ""0.4.0"", ""numpy"": ""2.2.6"", ""scipy"": ""1.15.3"", ""threadpoolctl"": ""3.6.0"", ""torch"": ""2.8.0+cu128""}"
   "CUDA confirmation","GNU 13.3.0","{""architectures"": ""75"", ""build_type"": ""Release"", ""compiler"": ""NVIDIA 12.8.93"", ""enabled"": true, ""flags"": "" -O3 -DNDEBUG"", ""host_compiler"": ""GNU 13.3.0"", ""scalar_dtype"": ""float64"", ""toolkit_version"": ""12.8.93""}","{""cgpy"": ""0.4.0"", ""numpy"": ""2.2.6"", ""scipy"": ""1.15.3"", ""threadpoolctl"": ""3.6.0"", ""torch"": ""2.8.0+cu128""}"

All configurations pin logical CPU 0 and use one native/library/Torch
intraop and one Torch interop thread. GPU clocks, thermal/display activity,
Windows background load and physical-core isolation are uncontrolled.
Seed 20261015, three warmups, 30 complete Williams rounds, standard CG/PCG,
rtol=1e-8, atol=1e-12, maxiter=2000 and residual refresh/restart every 50
updates are fixed. Dense repeats three solves per timed round; CSR/scaled
sparse repeats one. CPU includes both fused=True and fused=False controls;
CUDA uses fused=True. No automatic kernel or execution-mode selection occurs.

All results pass the independent stable CPU true-residual gate before/after
timing. Primary timing tables preserve iterations/matvecs and residual/threshold
ranges over both processes. Different native/reference counts mean different
numerical work: timings remain time-to-required-tolerance observations,
without an equal-work kernel claim. The constant-diagonal Poisson CSR
fixtures have no Jacobi conditioning benefit; the scaled sparse suite adds
four nonuniformly scaled cases and a neutral case.

Prepared calls reuse snapshots, output, Jacobi and workspace, charging
every x0 reset. API calls charge per-call output/workspace/Jacobi resource
construction; resident input snapshots/upload and correctness downloads
are excluded. CUDA times completed wall time including CPU dispatch,
control/scalar transfers and completion fences, not GPU-event kernel-only
or host-to-host latency. Torch is this project's same-policy eager CG/PCG
using public operations, rather than an official Torch CG solver or
compiled/graph capture. See :doc:`methodology` for precise boundaries.
Do not subtract separately sampled API/prepared medians to isolate a phase.

ABBA uses baseline/current/current/baseline adjacent fresh processes per
configuration, two artifact blocks per side. Its A/B table centers are the
median of **60 pooled normalized round samples** per side; brackets beside
milliseconds are min/max of the two process medians, not confidence
intervals or individual-call tail latency. Saved 95% intervals resample
artifacts and rounds 10,000 times, seed 20261015. The comparison does not
infer paired covariance from ABBA order; independently resampled rounds
can understate serial dependence, two process blocks give limited variance
evidence, and coverage is per case without multiplicity adjustment.

Strict verdicts use the unrounded interval: upper bound <=1.05 means
within the 5% regression budget (not necessarily a speedup); lower bound
>1.05 means regression; overlap is inconclusive. Any incompatible input,
specification or same-candidate work count rejects the whole configuration.
The renderer uses its saved report without recomputing/bootstrap projection.

Explicit-mode characterization uses one complete forward configuration
pass then its exact reverse, two processes per mode. Those table centers
are the **median of the two process medians**, as predeclared by that
runner, rather than the ABBA pooled estimator. Mode ratios are descriptive
only; different options are never sent through or allowed around the
strict equal-work comparator. No cross-host or universal performance claim
is made.

For the fused CPU cross-framework outcome summary, median-of-process-medians
aggregation gives 10/16 lower native medians versus NumPy and 12/16 versus
Torch. The strict tables and remaining-loss inventory instead use pooled
normalized rounds: 8/16 versus NumPy and 12/16 versus Torch. The two
dense-1024 API rows (none and Jacobi) change native/NumPy point ranking
between estimators. Both are disclosed; neither aggregation establishes
statistical significance or changes the strict A/B regression budget.

The maximum independently measured residual/threshold across the main
cohorts and focused confirmation is **0.9986449053814517**.

Device-engine correctness evidence and sanitizer limits
-------------------------------------------------------

The completed memcheck includes the concurrent-stream test. The
first four-test racecheck attempt did not complete before its timeout,
and remains retained. The subsequent focused run and serialized
concurrent-stream run have different coverage. A serialized racecheck
pass does **not** establish race freedom between overlapping streams;
normal execution and memcheck exercised concurrency. Instrumented or
force-blocking runs are correctness evidence and are excluded from every
formal timing batch.

.. csv-table:: Local retained sanitizer logs
   :class: benchmark-results
   :header: "log","outcome / coverage","SHA256"

   "build/performance-optimization/cuda-device-memcheck.log","27 GPU tests; 0 errors, 0 leaked bytes","18c9479e39e571071e9f46d2270a3a9eb42c0964aa141d027fdfa4760b94e902"
   "build/performance-optimization/cuda-device-racecheck.log","Initial four-test attempt timed out during concurrent streams at 60 seconds; incomplete","51f718631e9c895d6157e2182ff9d5debe61a46282c537942398ff8a071f1110"
   "build/performance-optimization/cuda-device-racecheck-focused.log","Three focused tests, without forced serialization; 0 hazards","42eafe35d1d44d60957548bdc5cf9ca9a96d9d7b4208a0632a35f0d3654c9fbe"
   "build/performance-optimization/cuda-device-racecheck-serialized-streams.log","Concurrent-stream test with force-blocking launches; 0 hazards","45810dbc1b092e24a2c2041470cc2b311c96fc06bd9f963315e1121fd76b2456"

CPU paired-row implementation and separate-operation controls
-------------------------------------------------------------

The saved whole-batch verdict is **regression**. Native and
framework-control rows below retain the original complete configuration
comparisons; a reference regression is not silently removed from the gate.
Unchanged framework code also exhibits latency variation, limiting the
attribution of every measured difference to the native source edit.

.. csv-table:: Saved strict verdict counts, including unchanged framework controls
   :class: benchmark-results
   :header: "candidate","within 5%","inconclusive","regression","comparison failed"

   "native","25","7","0","0"
   "numpy-out","10","19","3","0"
   "torch-eager-out","10","21","1","0"

.. csv-table:: cpu-dense-f64-v1 / none / prepared / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0327 [0.0322–0.0331]","0.0325 [0.0304–0.0355]","0.9942 [0.9082–1.1330]","inconclusive","NP 0.2330 [0.2062–0.2851]; T 0.3292 [0.3158–0.4549]","NP 0.1395; T 0.0987","A 49/51; B 49/51; NP 49/51; T 49/51","A 2.0721e-06 / 2.7376e-06; B 2.0721e-06 / 2.7376e-06; NP 2.0466e-06 / 2.7376e-06; T 2.4465e-06 / 2.7376e-06"
   "dense-256/native","0.5170 [0.5100–0.5258]","0.4123 [0.3945–0.4433]","0.7975 [0.7456–0.8728]","within 5%","NP 0.6806 [0.6120–0.7937]; T 0.7511 [0.6679–1.0028]","NP 0.6058; T 0.5489","A 67/70; B 67/70; NP 67/70; T 67/70","A 5.7065e-06 / 6.3423e-06; B 5.7065e-06 / 6.3423e-06; NP 5.6473e-06 / 6.3423e-06; T 5.7260e-06 / 6.3423e-06"
   "dense-1024/native","9.5429 [9.4981–9.6317]","8.2317 [7.9056–8.6634]","0.8626 [0.8159–0.9394]","within 5%","NP 7.8980 [7.5478–8.6324]; T 8.2500 [7.7859–8.9990]","NP 1.0423; T 0.9978","A 70/73; B 70/73; NP 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; NP 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-2048/native","87.7368 [81.0046–93.3070]","62.4681 [60.3494–63.4716]","0.7120 [0.6438–0.7870]","within 5%","NP 49.6681 [49.5943–50.2464]; T 53.4744 [52.0416–54.2202]","NP 1.2577; T 1.1682","A 70/73; B 70/73; NP 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; NP 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / none / api / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0397 [0.0382–0.0404]","0.0315 [0.0309–0.0324]","0.7917 [0.7466–0.8604]","within 5%","NP 0.1854 [0.1785–0.1901]; T 0.2808 [0.2663–0.2865]","NP 0.1697; T 0.1120","A 49/51; B 49/51; NP 49/51; T 49/51","A 2.0721e-06 / 2.7376e-06; B 2.0721e-06 / 2.7376e-06; NP 2.0466e-06 / 2.7376e-06; T 2.4465e-06 / 2.7376e-06"
   "dense-256/native","0.5715 [0.5479–0.5818]","0.3797 [0.3746–0.3826]","0.6643 [0.6382–0.7054]","within 5%","NP 0.5563 [0.4991–0.5952]; T 0.6843 [0.6264–0.7230]","NP 0.6826; T 0.5548","A 67/70; B 67/70; NP 67/70; T 67/70","A 5.7065e-06 / 6.3423e-06; B 5.7065e-06 / 6.3423e-06; NP 5.6473e-06 / 6.3423e-06; T 5.7260e-06 / 6.3423e-06"
   "dense-1024/native","9.5102 [9.4436–9.5481]","7.3781 [7.3141–8.6230]","0.7758 [0.7611–0.9309]","within 5%","NP 7.2938 [7.2771–8.7181]; T 7.6804 [7.6481–9.8028]","NP 1.0116; T 0.9606","A 70/73; B 70/73; NP 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; NP 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-2048/native","80.6819 [78.8766–82.8562]","65.6108 [62.8639–68.3456]","0.8132 [0.7577–0.8724]","within 5%","NP 50.3512 [48.4694–52.9775]; T 53.2306 [51.1151–54.8778]","NP 1.3031; T 1.2326","A 70/73; B 70/73; NP 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; NP 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / jacobi / prepared / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0071 [0.0067–0.0078]","0.0066 [0.0066–0.0067]","0.9345 [0.8362–1.0235]","within 5%","NP 0.0294 [0.0287–0.0296]; T 0.0493 [0.0470–0.0511]","NP 0.2259; T 0.1346","A 5/7; B 5/7; NP 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; NP 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-256/native","0.0517 [0.0493–0.0550]","0.0386 [0.0384–0.0387]","0.7469 [0.6935–0.7884]","within 5%","NP 0.0535 [0.0488–0.0571]; T 0.0659 [0.0643–0.0686]","NP 0.7210; T 0.5856","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; NP 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-1024/native","0.9551 [0.8995–0.9692]","0.8215 [0.7925–0.8729]","0.8601 [0.7931–0.9790]","within 5%","NP 0.8020 [0.7506–0.8686]; T 0.8739 [0.8043–0.9292]","NP 1.0243; T 0.9400","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; NP 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-2048/native","5.7502 [5.5917–5.8611]","4.8315 [4.7832–4.8676]","0.8402 [0.8106–0.8741]","within 5%","NP 3.9625 [3.8590–3.9980]; T 4.2547 [4.1849–4.2712]","NP 1.2193; T 1.1356","A 3/5; B 3/5; NP 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; NP 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / jacobi / api / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0122 [0.0122–0.0122]","0.0135 [0.0125–0.0169]","1.1131 [0.9886–1.5647]","inconclusive","NP 0.0410 [0.0366–0.0581]; T 0.0699 [0.0624–0.1018]","NP 0.3306; T 0.1936","A 5/7; B 5/7; NP 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; NP 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-256/native","0.0561 [0.0558–0.0577]","0.0511 [0.0451–0.0586]","0.9108 [0.7763–1.0523]","inconclusive","NP 0.0852 [0.0704–0.0951]; T 0.1178 [0.1048–0.1563]","NP 0.5992; T 0.4336","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; NP 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-1024/native","0.7712 [0.7606–0.7781]","0.7048 [0.5912–0.8451]","0.9138 [0.7566–1.1167]","inconclusive","NP 0.6801 [0.6074–0.9054]; T 0.7864 [0.6976–0.9942]","NP 1.0362; T 0.8961","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; NP 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-2048/native","5.1561 [5.1375–5.2358]","4.4342 [4.3593–4.9282]","0.8600 [0.8249–0.9716]","within 5%","NP 3.5361 [3.4721–3.8382]; T 3.9190 [3.8155–4.3792]","NP 1.2540; T 1.1315","A 3/5; B 3/5; NP 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; NP 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / none / prepared / fused=False
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0523 [0.0521–0.0539]","0.0529 [0.0500–0.0586]","1.0117 [0.9280–1.1419]","inconclusive","NP 0.2120 [0.1886–0.2433]; T 0.3315 [0.2734–0.3710]","NP 0.2497; T 0.1597","A 49/51; B 49/51; NP 49/51; T 49/51","A 2.0794e-06 / 2.7376e-06; B 2.0794e-06 / 2.7376e-06; NP 2.0466e-06 / 2.7376e-06; T 2.4465e-06 / 2.7376e-06"
   "dense-256/native","1.2713 [1.2659–1.2832]","1.3672 [1.2818–1.4448]","1.0755 [0.9878–1.1490]","inconclusive","NP 0.5842 [0.5578–0.6114]; T 0.6851 [0.6248–0.8650]","NP 2.3402; T 1.9957","A 67/70; B 67/70; NP 67/70; T 67/70","A 6.0140e-06 / 6.3423e-06; B 6.0140e-06 / 6.3423e-06; NP 5.6473e-06 / 6.3423e-06; T 5.7260e-06 / 6.3423e-06"
   "dense-1024/native","26.0288 [25.8835–26.2499]","25.9516 [25.9209–25.9516]","0.9970 [0.9812–1.0110]","within 5%","NP 7.6209 [7.5662–7.6705]; T 7.8271 [7.7915–7.9288]","NP 3.4053; T 3.3156","A 70/73; B 70/73; NP 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; NP 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-2048/native","131.1314 [130.4426–131.3950]","130.7779 [128.4803–131.7893]","0.9973 [0.9725–1.0248]","within 5%","NP 52.5557 [51.1531–53.4887]; T 55.9766 [54.2695–56.3214]","NP 2.4884; T 2.3363","A 70/73; B 70/73; NP 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; NP 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / none / api / fused=False
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0614 [0.0601–0.0676]","0.0592 [0.0579–0.0607]","0.9646 [0.8383–1.0233]","within 5%","NP 0.1918 [0.1871–0.1945]; T 0.2835 [0.2825–0.2873]","NP 0.3088; T 0.2089","A 49/51; B 49/51; NP 49/51; T 49/51","A 2.0794e-06 / 2.7376e-06; B 2.0794e-06 / 2.7376e-06; NP 2.0466e-06 / 2.7376e-06; T 2.4465e-06 / 2.7376e-06"
   "dense-256/native","1.3529 [1.3342–1.3819]","1.3370 [1.2976–1.3724]","0.9883 [0.9393–1.0302]","within 5%","NP 0.5587 [0.5294–0.5744]; T 0.6819 [0.6550–0.6937]","NP 2.3932; T 1.9608","A 67/70; B 67/70; NP 67/70; T 67/70","A 6.0140e-06 / 6.3423e-06; B 6.0140e-06 / 6.3423e-06; NP 5.6473e-06 / 6.3423e-06; T 5.7260e-06 / 6.3423e-06"
   "dense-1024/native","26.8475 [26.8186–27.6731]","27.4127 [27.2627–27.5514]","1.0211 [0.9835–1.0297]","within 5%","NP 7.4051 [7.4018–7.4080]; T 7.8390 [7.8390–7.8393]","NP 3.7019; T 3.4969","A 70/73; B 70/73; NP 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; NP 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-2048/native","125.1625 [122.6889–125.7705]","126.9796 [126.9047–126.9926]","1.0145 [1.0072–1.0402]","within 5%","NP 51.4261 [50.7474–51.7814]; T 53.5212 [53.3084–53.7101]","NP 2.4692; T 2.3725","A 70/73; B 70/73; NP 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; NP 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / jacobi / prepared / fused=False
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0098 [0.0098–0.0098]","0.0099 [0.0098–0.0099]","1.0115 [0.9855–1.0313]","within 5%","NP 0.0279 [0.0277–0.0285]; T 0.0474 [0.0466–0.0477]","NP 0.3545; T 0.2085","A 5/7; B 5/7; NP 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; NP 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-256/native","0.1216 [0.1211–0.1219]","0.1240 [0.1223–0.1255]","1.0201 [0.9937–1.0423]","within 5%","NP 0.0552 [0.0484–0.0602]; T 0.0682 [0.0673–0.0700]","NP 2.2457; T 1.8173","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; NP 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-1024/native","2.2542 [2.2498–2.2548]","2.2796 [2.2610–2.2991]","1.0113 [0.9931–1.0296]","within 5%","NP 0.7699 [0.7661–0.7722]; T 0.8294 [0.8029–0.8677]","NP 2.9609; T 2.7485","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; NP 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-2048/native","8.9808 [8.9480–8.9956]","8.8821 [8.8142–8.8886]","0.9890 [0.9685–1.0025]","within 5%","NP 3.6816 [3.6122–3.8224]; T 4.0778 [4.0641–4.0778]","NP 2.4126; T 2.1782","A 3/5; B 3/5; NP 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; NP 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"

.. csv-table:: cpu-dense-f64-v1 / jacobi / api / fused=False
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/native","0.0158 [0.0155–0.0160]","0.0155 [0.0154–0.0158]","0.9790 [0.9483–1.0499]","within 5%","NP 0.0373 [0.0371–0.0380]; T 0.0626 [0.0622–0.0649]","NP 0.4148; T 0.2474","A 5/7; B 5/7; NP 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; NP 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-256/native","0.1266 [0.1237–0.1283]","0.1320 [0.1292–0.1356]","1.0426 [1.0018–1.0983]","inconclusive","NP 0.0595 [0.0573–0.0615]; T 0.0924 [0.0893–0.0983]","NP 2.2186; T 1.4291","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; NP 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-1024/native","2.2226 [2.2035–2.2341]","2.2544 [2.2511–2.2756]","1.0143 [0.9996–1.0378]","within 5%","NP 0.6460 [0.6460–0.6464]; T 0.7403 [0.7288–0.7461]","NP 3.4901; T 3.0452","A 4/6; B 4/6; NP 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; NP 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-2048/native","8.7548 [8.7357–8.7727]","8.7740 [8.6412–8.8208]","1.0022 [0.9810–1.0161]","within 5%","NP 3.6800 [3.6720–3.6857]; T 3.9554 [3.9411–3.9714]","NP 2.3843; T 2.2183","A 3/5; B 3/5; NP 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; NP 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"

.. csv-table:: Every framework control; the same frozen reference code ran on both sides
   :class: benchmark-results
   :header: "configuration","case/reference","A ms [range]","B ms [range]","B/A [95% interval]","strict verdict","iterations/matvecs"

   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-64/numpy-out","0.1858 [0.1845–0.1891]","0.2330 [0.2062–0.2851]","1.2540 [1.0896–1.5627]","regression","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-64/torch-eager-out","0.2924 [0.2840–0.2996]","0.3292 [0.3158–0.4549]","1.1262 [1.0445–1.6166]","inconclusive","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-256/numpy-out","0.5251 [0.4847–0.5621]","0.6806 [0.6120–0.7937]","1.2961 [1.0863–1.6499]","regression","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-256/torch-eager-out","0.6286 [0.6125–0.6319]","0.7511 [0.6679–1.0028]","1.1949 [1.0414–1.6614]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-1024/numpy-out","7.5507 [7.2795–7.8390]","7.8980 [7.5478–8.6324]","1.0460 [0.9600–1.2164]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-1024/torch-eager-out","7.8831 [7.8147–7.9698]","8.2500 [7.7859–8.9990]","1.0465 [0.9716–1.1736]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-2048/numpy-out","52.3570 [50.4166–52.7483]","49.6681 [49.5943–50.2464]","0.9486 [0.9231–1.0095]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-prepared","dense-2048/torch-eager-out","56.0756 [55.8504–56.2239]","53.4744 [52.0416–54.2202]","0.9536 [0.9169–1.0031]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-64/numpy-out","0.1955 [0.1929–0.2004]","0.1854 [0.1785–0.1901]","0.9482 [0.8889–0.9890]","within 5%","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-64/torch-eager-out","0.2892 [0.2816–0.2989]","0.2808 [0.2663–0.2865]","0.9708 [0.8895–1.0381]","within 5%","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-256/numpy-out","0.5567 [0.5333–0.5735]","0.5563 [0.4991–0.5952]","0.9992 [0.8677–1.1211]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-256/torch-eager-out","0.6996 [0.6780–0.7186]","0.6843 [0.6264–0.7230]","0.9781 [0.8687–1.0837]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-1024/numpy-out","7.1943 [7.1540–7.2721]","7.2938 [7.2771–8.7181]","1.0138 [0.9904–1.2731]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-1024/torch-eager-out","7.6504 [7.5834–7.7050]","7.6804 [7.6481–9.8028]","1.0039 [0.9845–1.3318]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-2048/numpy-out","51.7156 [51.3171–51.9428]","50.3512 [48.4694–52.9775]","0.9736 [0.9299–1.0437]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api","dense-2048/torch-eager-out","53.6676 [53.4139–53.9260]","53.2306 [51.1151–54.8778]","0.9919 [0.9465–1.0674]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-64/numpy-out","0.0285 [0.0264–0.0300]","0.0294 [0.0287–0.0296]","1.0334 [0.9466–1.1451]","inconclusive","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-64/torch-eager-out","0.0468 [0.0437–0.0487]","0.0493 [0.0470–0.0511]","1.0552 [0.9381–1.1707]","inconclusive","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-256/numpy-out","0.0511 [0.0480–0.0528]","0.0535 [0.0488–0.0571]","1.0473 [0.9112–1.1947]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-256/torch-eager-out","0.0668 [0.0617–0.0689]","0.0659 [0.0643–0.0686]","0.9856 [0.9157–1.1159]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-1024/numpy-out","0.7393 [0.6922–0.8058]","0.8020 [0.7506–0.8686]","1.0848 [0.9191–1.2635]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-1024/torch-eager-out","0.7680 [0.7261–0.8508]","0.8739 [0.8043–0.9292]","1.1379 [0.9424–1.3114]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-2048/numpy-out","3.5914 [3.4855–3.8181]","3.9625 [3.8590–3.9980]","1.1033 [1.0014–1.1602]","inconclusive","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared","dense-2048/torch-eager-out","3.9794 [3.7588–4.1661]","4.2547 [4.1849–4.2712]","1.0692 [0.9934–1.1501]","inconclusive","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-64/numpy-out","0.0357 [0.0356–0.0363]","0.0410 [0.0366–0.0581]","1.1466 [1.0042–1.6913]","inconclusive","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-64/torch-eager-out","0.0611 [0.0605–0.0616]","0.0699 [0.0624–0.1018]","1.1442 [0.9908–1.8452]","inconclusive","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-256/numpy-out","0.0549 [0.0544–0.0563]","0.0852 [0.0704–0.0951]","1.5540 [1.2134–1.7554]","regression","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-256/torch-eager-out","0.0859 [0.0844–0.0869]","0.1178 [0.1048–0.1563]","1.3712 [1.1410–1.8715]","regression","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-1024/numpy-out","0.6052 [0.5960–0.6116]","0.6801 [0.6074–0.9054]","1.1239 [0.9897–1.5430]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-1024/torch-eager-out","0.6773 [0.6746–0.6797]","0.7864 [0.6976–0.9942]","1.1612 [1.0060–1.5358]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-2048/numpy-out","3.5018 [3.4081–3.5256]","3.5361 [3.4721–3.8382]","1.0098 [0.9787–1.2037]","inconclusive","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-jacobi-api","dense-2048/torch-eager-out","3.7106 [3.6821–3.7187]","3.9190 [3.8155–4.3792]","1.0562 [1.0065–1.2090]","inconclusive","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-64/numpy-out","0.1861 [0.1847–0.1922]","0.2120 [0.1886–0.2433]","1.1394 [0.9561–1.3565]","inconclusive","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-64/torch-eager-out","0.2920 [0.2861–0.3057]","0.3315 [0.2734–0.3710]","1.1354 [0.8859–1.3120]","inconclusive","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-256/numpy-out","0.5366 [0.4953–0.5775]","0.5842 [0.5578–0.6114]","1.0887 [0.9562–1.2481]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-256/torch-eager-out","0.6422 [0.6122–0.6695]","0.6851 [0.6248–0.8650]","1.0668 [0.9298–1.4144]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-1024/numpy-out","7.4866 [7.4475–7.5273]","7.6209 [7.5662–7.6705]","1.0179 [0.9880–1.0542]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-1024/torch-eager-out","7.7954 [7.5805–7.8963]","7.8271 [7.7915–7.9288]","1.0041 [0.9660–1.0523]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-2048/numpy-out","51.5171 [51.3825–51.5779]","52.5557 [51.1531–53.4887]","1.0202 [0.9775–1.0618]","inconclusive","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-prepared-unfused","dense-2048/torch-eager-out","57.0952 [55.7363–57.6858]","55.9766 [54.2695–56.3214]","0.9804 [0.9379–1.0145]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-64/numpy-out","0.2061 [0.1946–0.2218]","0.1918 [0.1871–0.1945]","0.9303 [0.8384–1.0014]","within 5%","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-64/torch-eager-out","0.3003 [0.2941–0.3200]","0.2835 [0.2825–0.2873]","0.9441 [0.8650–0.9943]","within 5%","A 49/51; B 49/51"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-256/numpy-out","0.5579 [0.5001–0.6207]","0.5587 [0.5294–0.5744]","1.0014 [0.8527–1.1504]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-256/torch-eager-out","0.7464 [0.6431–0.7986]","0.6819 [0.6550–0.6937]","0.9135 [0.8172–1.0818]","inconclusive","A 67/70; B 67/70"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-1024/numpy-out","7.2624 [7.2247–7.3775]","7.4051 [7.4018–7.4080]","1.0196 [0.9939–1.0361]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-1024/torch-eager-out","7.7998 [7.6823–7.9530]","7.8390 [7.8390–7.8393]","1.0050 [0.9742–1.0341]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-2048/numpy-out","50.9226 [50.3658–51.0929]","51.4261 [50.7474–51.7814]","1.0099 [0.9907–1.0399]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-none-api-unfused","dense-2048/torch-eager-out","53.1453 [51.7448–53.3468]","53.5212 [53.3084–53.7101]","1.0071 [0.9924–1.0444]","within 5%","A 70/73; B 70/73"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-64/numpy-out","0.0282 [0.0282–0.0283]","0.0279 [0.0277–0.0285]","0.9883 [0.9571–1.0230]","within 5%","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-64/torch-eager-out","0.0459 [0.0454–0.0460]","0.0474 [0.0466–0.0477]","1.0341 [0.9790–1.0655]","inconclusive","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-256/numpy-out","0.0519 [0.0477–0.0553]","0.0552 [0.0484–0.0602]","1.0640 [0.8731–1.2829]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-256/torch-eager-out","0.0689 [0.0689–0.0690]","0.0682 [0.0673–0.0700]","0.9903 [0.9356–1.0613]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-1024/numpy-out","0.7843 [0.7469–0.8141]","0.7699 [0.7661–0.7722]","0.9816 [0.8789–1.1027]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-1024/torch-eager-out","0.8310 [0.7997–0.8482]","0.8294 [0.8029–0.8677]","0.9980 [0.8924–1.1396]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-2048/numpy-out","3.7663 [3.7032–3.7864]","3.6816 [3.6122–3.8224]","0.9775 [0.9378–1.0448]","within 5%","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused","dense-2048/torch-eager-out","4.1117 [3.9839–4.2209]","4.0778 [4.0641–4.0778]","0.9917 [0.9431–1.0487]","within 5%","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-64/numpy-out","0.0370 [0.0367–0.0377]","0.0373 [0.0371–0.0380]","1.0090 [0.9724–1.0503]","inconclusive","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-64/torch-eager-out","0.0660 [0.0651–0.0675]","0.0626 [0.0622–0.0649]","0.9489 [0.9016–1.0121]","within 5%","A 5/7; B 5/7"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-256/numpy-out","0.0607 [0.0573–0.0637]","0.0595 [0.0573–0.0615]","0.9798 [0.8921–1.0982]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-256/torch-eager-out","0.0940 [0.0914–0.0959]","0.0924 [0.0893–0.0983]","0.9831 [0.9222–1.0930]","inconclusive","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-1024/numpy-out","0.6415 [0.6379–0.6452]","0.6460 [0.6460–0.6464]","1.0070 [0.9679–1.0359]","within 5%","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-1024/torch-eager-out","0.7250 [0.7212–0.7291]","0.7403 [0.7288–0.7461]","1.0212 [0.9772–1.0479]","within 5%","A 4/6; B 4/6"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-2048/numpy-out","3.6499 [3.6462–3.6499]","3.6800 [3.6720–3.6857]","1.0082 [0.9828–1.0368]","within 5%","A 3/5; B 3/5"
   "cpu-release-cpu-dense-f64-v1-jacobi-api-unfused","dense-2048/torch-eager-out","3.9163 [3.9059–3.9165]","3.9554 [3.9411–3.9714]","1.0100 [0.9758–1.0304]","within 5%","A 3/5; B 3/5"

CUDA API resource-construction implementation, host engine
----------------------------------------------------------

The saved whole-batch verdict is **regression**. Native and
framework-control rows below retain the original complete configuration
comparisons; a reference regression is not silently removed from the gate.
Unchanged framework code also exhibits latency variation, limiting the
attribution of every measured difference to the native source edit.

.. csv-table:: Saved strict verdict counts, including unchanged framework controls
   :class: benchmark-results
   :header: "candidate","within 5%","inconclusive","regression","comparison failed"

   "cuda-cublas","8","7","1","0"
   "cuda-native","12","19","1","0"
   "torch-cuda-eager","13","18","1","0"

.. csv-table:: cuda-dense-f64-v1 / none / prepared / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/cuda-native","5.3275 [5.1936–5.4056]","5.7956 [5.7262–5.8537]","1.0879 [1.0512–1.1414]","regression","T 12.4804 [12.3923–12.5702]","T 0.4644","A 49/51; B 49/51; T 49/51","A 2.0447e-06 / 2.7376e-06; B 2.0447e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06"
   "dense-64/cuda-cublas","5.3793 [5.3335–5.3955]","6.0801 [6.0552–6.1037]","1.1303 [1.0899–1.1741]","regression","T 12.4804 [12.3923–12.5702]","T 0.4872","A 49/51; B 49/51; T 49/51","A 2.2061e-06 / 2.7376e-06; B 2.2061e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06"
   "dense-256/cuda-native","7.9117 [7.7021–8.0989]","7.7398 [7.7281–7.7404]","0.9783 [0.9451–1.0178]","within 5%","T 15.0042 [14.6427–15.4140]","T 0.5158","A 67/70; B 67/70; T 67/70","A 5.9926e-06 / 6.3423e-06; B 5.9926e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06"
   "dense-256/cuda-cublas","7.7622 [7.4573–8.1062]","7.6929 [7.6239–7.7223]","0.9911 [0.9351–1.0528]","inconclusive","T 15.0042 [14.6427–15.4140]","T 0.5127","A 67/70; B 67/70; T 67/70","A 5.9162e-06 / 6.3423e-06; B 5.9162e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06"
   "dense-1024/cuda-native","10.6424 [10.4657–11.0369]","11.0780 [10.9751–11.1758]","1.0409 [0.9870–1.0692]","inconclusive","T 15.4751 [15.3184–15.6047]","T 0.7159","A 70/73; B 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-1024/cuda-cublas","9.4492 [9.1342–9.6438]","9.6212 [9.4733–9.6474]","1.0182 [0.9696–1.0639]","inconclusive","T 15.4751 [15.3184–15.6047]","T 0.6217","A 70/73; B 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-2048/cuda-native","16.7144 [16.6822–16.8339]","16.7913 [16.7107–17.0599]","1.0046 [0.9881–1.0284]","within 5%","T 21.0314 [20.9972–21.0487]","T 0.7984","A 70/73; B 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"
   "dense-2048/cuda-cublas","15.4516 [15.2487–15.5483]","15.3438 [15.3162–15.3498]","0.9930 [0.9770–1.0123]","within 5%","T 21.0314 [20.9972–21.0487]","T 0.7296","A 70/73; B 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"

.. csv-table:: cuda-dense-f64-v1 / none / api / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/cuda-native","5.3352 [5.2971–5.3993]","5.3478 [5.3455–5.3519]","1.0024 [0.9674–1.0417]","within 5%","T 10.9165 [10.8496–10.9750]","T 0.4899","A 49/51; B 49/51; T 49/51","A 2.0447e-06 / 2.7376e-06; B 2.0447e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06"
   "dense-64/cuda-cublas","7.1183 [7.0932–7.1705]","7.2511 [7.2256–7.2543]","1.0187 [0.9873–1.0395]","within 5%","T 10.9165 [10.8496–10.9750]","T 0.6642","A 49/51; B 49/51; T 49/51","A 2.2061e-06 / 2.7376e-06; B 2.2061e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06"
   "dense-256/cuda-native","7.6616 [7.6196–7.7171]","7.7582 [7.6545–7.9039]","1.0126 [0.9818–1.0469]","within 5%","T 14.7493 [14.7050–14.8789]","T 0.5260","A 67/70; B 67/70; T 67/70","A 5.9926e-06 / 6.3423e-06; B 5.9926e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06"
   "dense-256/cuda-cublas","9.1317 [9.1174–9.3037]","9.3471 [9.2487–9.4867]","1.0236 [0.9816–1.0459]","within 5%","T 14.7493 [14.7050–14.8789]","T 0.6337","A 67/70; B 67/70; T 67/70","A 5.9162e-06 / 6.3423e-06; B 5.9162e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06"
   "dense-1024/cuda-native","10.2409 [10.2023–10.2701]","10.7192 [10.4481–10.9875]","1.0467 [1.0013–1.0955]","inconclusive","T 15.1441 [15.0693–15.3272]","T 0.7078","A 70/73; B 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-1024/cuda-cublas","10.5783 [10.4771–10.6568]","10.9441 [10.7203–11.1712]","1.0346 [0.9921–1.0903]","inconclusive","T 15.1441 [15.0693–15.3272]","T 0.7227","A 70/73; B 70/73; T 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05"
   "dense-2048/cuda-native","16.7747 [16.7585–16.7800]","16.6802 [16.5460–17.0059]","0.9944 [0.9829–1.0338]","within 5%","T 20.9983 [20.9125–21.3523]","T 0.7944","A 70/73; B 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"
   "dense-2048/cuda-cublas","16.7813 [16.7784–16.7880]","17.0279 [16.8211–17.4700]","1.0147 [0.9942–1.0462]","within 5%","T 20.9983 [20.9125–21.3523]","T 0.8109","A 70/73; B 70/73; T 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05"

.. csv-table:: cuda-dense-f64-v1 / jacobi / prepared / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/cuda-native","0.8190 [0.8096–0.8355]","0.8557 [0.8474–0.8722]","1.0448 [1.0000–1.1522]","inconclusive","T 1.7211 [1.7079–1.7741]","T 0.4972","A 5/7; B 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-64/cuda-cublas","0.8895 [0.8667–0.9120]","0.8925 [0.8895–0.9143]","1.0034 [0.9496–1.0768]","inconclusive","T 1.7211 [1.7079–1.7741]","T 0.5186","A 5/7; B 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-256/cuda-native","0.8125 [0.8097–0.8167]","0.8197 [0.8122–0.8234]","1.0088 [0.9659–1.0733]","inconclusive","T 1.4179 [1.3977–1.5071]","T 0.5781","A 4/6; B 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-256/cuda-cublas","0.8164 [0.8008–0.8265]","0.8073 [0.7973–0.8184]","0.9888 [0.9503–1.1157]","inconclusive","T 1.4179 [1.3977–1.5071]","T 0.5694","A 4/6; B 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-1024/cuda-native","1.1283 [1.1108–1.1473]","1.1406 [1.1353–1.1914]","1.0108 [0.9755–1.0941]","inconclusive","T 1.5313 [1.5075–1.5484]","T 0.7448","A 4/6; B 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-1024/cuda-cublas","0.9836 [0.9807–1.0149]","1.0275 [1.0100–1.0352]","1.0446 [0.9846–1.0654]","inconclusive","T 1.5313 [1.5075–1.5484]","T 0.6710","A 4/6; B 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-2048/cuda-native","1.3894 [1.3596–1.4233]","1.4530 [1.4509–1.4575]","1.0458 [0.9925–1.1146]","inconclusive","T 1.7261 [1.7072–1.8181]","T 0.8418","A 3/5; B 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"
   "dense-2048/cuda-cublas","1.2260 [1.2238–1.2314]","1.2829 [1.2799–1.2829]","1.0464 [0.9704–1.0960]","inconclusive","T 1.7261 [1.7072–1.8181]","T 0.7432","A 3/5; B 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"

.. csv-table:: cuda-dense-f64-v1 / jacobi / api / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "dense-64/cuda-native","1.0078 [0.9073–1.0389]","0.9241 [0.8984–0.9651]","0.9170 [0.8610–1.0657]","inconclusive","T 2.0047 [1.9835–2.0382]","T 0.4610","A 5/7; B 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-64/cuda-cublas","2.5151 [2.3432–2.6572]","2.3204 [2.3074–2.3657]","0.9226 [0.8536–1.0174]","within 5%","T 2.0047 [1.9835–2.0382]","T 1.1575","A 5/7; B 5/7; T 5/7","A 1.1410e-07 / 2.7376e-06; B 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06"
   "dense-256/cuda-native","0.9134 [0.9063–0.9242]","0.8963 [0.8907–0.8998]","0.9813 [0.9466–1.0114]","within 5%","T 1.7955 [1.7784–1.8018]","T 0.4992","A 4/6; B 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-256/cuda-cublas","2.5174 [2.5102–2.5415]","2.5351 [2.5176–2.5537]","1.0070 [0.9753–1.0254]","within 5%","T 1.7955 [1.7784–1.8018]","T 1.4119","A 4/6; B 4/6; T 4/6","A 1.0990e-06 / 6.3423e-06; B 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06"
   "dense-1024/cuda-native","1.2074 [1.1955–1.2159]","1.1723 [1.1635–1.1793]","0.9709 [0.9455–0.9977]","within 5%","T 1.8600 [1.8360–1.8600]","T 0.6303","A 4/6; B 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-1024/cuda-cublas","2.7412 [2.7055–2.7537]","2.6450 [2.6411–2.6450]","0.9649 [0.9482–0.9910]","within 5%","T 1.8600 [1.8360–1.8600]","T 1.4220","A 4/6; B 4/6; T 4/6","A 1.7223e-07 / 1.0791e-05; B 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05"
   "dense-2048/cuda-native","1.4541 [1.4480–1.4553]","1.4320 [1.3806–1.4356]","0.9848 [0.9254–1.0254]","within 5%","T 1.9232 [1.9163–1.9326]","T 0.7446","A 3/5; B 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"
   "dense-2048/cuda-cublas","2.7856 [2.7771–2.7882]","2.7171 [2.6537–2.7271]","0.9754 [0.9338–1.0158]","within 5%","T 1.9232 [1.9163–1.9326]","T 1.4128","A 3/5; B 3/5; T 3/5","A 1.2450e-05 / 1.4741e-05; B 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05"

.. csv-table:: cuda-csr-f64-v1 / none / prepared / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "poisson-8/cuda-native","2.8771 [2.4971–2.9184]","2.9883 [2.8309–3.3613]","1.0386 [0.9623–1.3497]","inconclusive","T 5.4799 [4.9214–6.9493]","T 0.5453","A 26/28; B 26/28; T 26/28","A 2.0123e-07 / 3.0549e-07; B 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07"
   "poisson-32/cuda-native","14.8408 [14.6708–14.8969]","13.9285 [13.6360–14.0982]","0.9385 [0.9000–0.9996]","within 5%","T 26.2652 [25.4634–27.6783]","T 0.5303","A 129/133; B 129/133; T 129/133","A 1.3276e-06 / 1.3672e-06; B 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06"
   "poisson-64/cuda-native","32.6278 [32.2141–33.7764]","33.1393 [31.7969–35.8600]","1.0157 [0.9176–1.1226]","inconclusive","T 64.8660 [58.8068–70.9232]","T 0.5109","A 290/297; B 290/297; T 290/297","A 2.7729e-06 / 2.8101e-06; B 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06"
   "poisson-128/cuda-native","43.3739 [43.0946–44.9791]","47.3986 [46.7024–48.0620]","1.0928 [1.0146–1.1447]","inconclusive","T 85.7055 [83.1203–87.6735]","T 0.5530","A 395/404; B 395/404; T 395/404","A 5.6614e-06 / 5.7493e-06; B 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06"

.. csv-table:: cuda-csr-f64-v1 / none / api / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "poisson-8/cuda-native","2.8504 [2.8443–2.9244]","2.9993 [2.8715–3.0659]","1.0522 [0.9591–1.1298]","inconclusive","T 5.4003 [5.2988–5.4561]","T 0.5554","A 26/28; B 26/28; T 26/28","A 2.0123e-07 / 3.0549e-07; B 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07"
   "poisson-32/cuda-native","14.0509 [13.1247–14.8000]","14.6823 [14.5144–14.8924]","1.0449 [0.9578–1.1367]","inconclusive","T 27.0275 [26.5483–27.4447]","T 0.5432","A 129/133; B 129/133; T 129/133","A 1.3276e-06 / 1.3672e-06; B 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06"
   "poisson-64/cuda-native","29.7124 [29.6069–29.7899]","31.9598 [31.8916–32.3745]","1.0756 [1.0494–1.1112]","inconclusive","T 58.5403 [57.3512–59.8139]","T 0.5459","A 290/297; B 290/297; T 290/297","A 2.7729e-06 / 2.8101e-06; B 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06"
   "poisson-128/cuda-native","48.0683 [47.3417–48.1772]","47.3022 [47.2951–47.3206]","0.9841 [0.9581–1.0518]","inconclusive","T 85.0552 [83.4058–86.8914]","T 0.5561","A 395/404; B 395/404; T 395/404","A 5.6614e-06 / 5.7493e-06; B 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06"

.. csv-table:: cuda-csr-f64-v1 / jacobi / prepared / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "poisson-8/cuda-native","4.1691 [4.1194–4.2762]","4.0288 [3.9862–4.0708]","0.9663 [0.9139–1.0016]","within 5%","T 7.3065 [7.1296–7.4173]","T 0.5514","A 26/28; B 26/28; T 26/28","A 2.0123e-07 / 3.0549e-07; B 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07"
   "poisson-32/cuda-native","20.0321 [19.4902–20.1906]","20.1078 [19.8534–20.4937]","1.0038 [0.9782–1.0566]","inconclusive","T 36.4020 [36.1515–37.0770]","T 0.5524","A 129/133; B 129/133; T 129/133","A 1.3276e-06 / 1.3672e-06; B 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06"
   "poisson-64/cuda-native","45.0302 [44.3900–45.7719]","45.4883 [45.4549–45.9671]","1.0102 [0.9845–1.0401]","within 5%","T 82.9120 [82.4620–84.0477]","T 0.5486","A 290/297; B 290/297; T 290/297","A 2.7729e-06 / 2.8101e-06; B 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06"
   "poisson-128/cuda-native","72.4515 [69.9811–76.4699]","71.7564 [63.4136–74.0164]","0.9904 [0.8291–1.1079]","inconclusive","T 124.0750 [117.2633–131.5399]","T 0.5783","A 395/404; B 395/404; T 395/404","A 5.6614e-06 / 5.7493e-06; B 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06"

.. csv-table:: cuda-csr-f64-v1 / jacobi / api / fused=True
   :class: benchmark-results
   :header: "case/native","A ms [process range]","B ms [process range]","B/A [95% interval]","strict verdict","current references ms [range]","native/reference","iterations/matvecs","true R / threshold"

   "poisson-8/cuda-native","4.2228 [4.0270–4.3574]","4.2807 [4.0719–4.3441]","1.0137 [0.9299–1.0814]","inconclusive","T 7.8629 [7.5721–7.9994]","T 0.5444","A 26/28; B 26/28; T 26/28","A 2.0123e-07 / 3.0549e-07; B 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07"
   "poisson-32/cuda-native","20.0684 [19.6605–20.2553]","20.2701 [19.7048–20.4399]","1.0101 [0.9692–1.0479]","within 5%","T 36.5274 [35.6266–37.7420]","T 0.5549","A 129/133; B 129/133; T 129/133","A 1.3276e-06 / 1.3672e-06; B 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06"
   "poisson-64/cuda-native","44.8170 [43.9330–45.7980]","46.7027 [46.6835–46.9227]","1.0421 [0.9999–1.0824]","inconclusive","T 81.6588 [80.7255–85.7150]","T 0.5719","A 290/297; B 290/297; T 290/297","A 2.7729e-06 / 2.8101e-06; B 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06"
   "poisson-128/cuda-native","69.8240 [67.3080–70.0023]","71.7507 [71.2823–74.1164]","1.0276 [0.9381–1.1100]","inconclusive","T 118.6615 [117.0465–119.9322]","T 0.6047","A 395/404; B 395/404; T 395/404","A 5.6614e-06 / 5.7493e-06; B 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06"

.. csv-table:: Every framework control; the same frozen reference code ran on both sides
   :class: benchmark-results
   :header: "configuration","case/reference","A ms [range]","B ms [range]","B/A [95% interval]","strict verdict","iterations/matvecs"

   "cuda-release-cuda-dense-f64-v1-none-prepared","dense-64/torch-cuda-eager","11.0086 [10.8001–11.2528]","12.4804 [12.3923–12.5702]","1.1337 [1.0955–1.1701]","regression","A 49/51; B 49/51"
   "cuda-release-cuda-dense-f64-v1-none-prepared","dense-256/torch-cuda-eager","16.0624 [15.4810–17.1373]","15.0042 [14.6427–15.4140]","0.9341 [0.8480–1.0104]","within 5%","A 67/70; B 67/70"
   "cuda-release-cuda-dense-f64-v1-none-prepared","dense-1024/torch-cuda-eager","15.5206 [15.1884–15.7052]","15.4751 [15.3184–15.6047]","0.9971 [0.9734–1.0339]","within 5%","A 70/73; B 70/73"
   "cuda-release-cuda-dense-f64-v1-none-prepared","dense-2048/torch-cuda-eager","21.5322 [21.3604–21.6613]","21.0314 [20.9972–21.0487]","0.9767 [0.9613–0.9942]","within 5%","A 70/73; B 70/73"
   "cuda-release-cuda-dense-f64-v1-none-api","dense-64/torch-cuda-eager","10.9194 [10.8313–11.0087]","10.9165 [10.8496–10.9750]","0.9997 [0.9640–1.0351]","within 5%","A 49/51; B 49/51"
   "cuda-release-cuda-dense-f64-v1-none-api","dense-256/torch-cuda-eager","14.5311 [14.3910–14.5628]","14.7493 [14.7050–14.8789]","1.0150 [0.9981–1.0480]","within 5%","A 67/70; B 67/70"
   "cuda-release-cuda-dense-f64-v1-none-api","dense-1024/torch-cuda-eager","14.9559 [14.8353–15.2302]","15.1441 [15.0693–15.3272]","1.0126 [0.9833–1.0453]","within 5%","A 70/73; B 70/73"
   "cuda-release-cuda-dense-f64-v1-none-api","dense-2048/torch-cuda-eager","21.2725 [21.1455–21.3197]","20.9983 [20.9125–21.3523]","0.9871 [0.9754–1.0197]","within 5%","A 70/73; B 70/73"
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared","dense-64/torch-cuda-eager","1.7082 [1.6965–1.7468]","1.7211 [1.7079–1.7741]","1.0076 [0.9526–1.0874]","inconclusive","A 5/7; B 5/7"
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared","dense-256/torch-cuda-eager","1.4823 [1.4733–1.4823]","1.4179 [1.3977–1.5071]","0.9566 [0.8869–1.0556]","inconclusive","A 4/6; B 4/6"
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared","dense-1024/torch-cuda-eager","1.4888 [1.4746–1.5515]","1.5313 [1.5075–1.5484]","1.0286 [0.9654–1.0710]","inconclusive","A 4/6; B 4/6"
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared","dense-2048/torch-cuda-eager","1.6506 [1.6460–1.6506]","1.7261 [1.7072–1.8181]","1.0457 [0.9968–1.1652]","inconclusive","A 3/5; B 3/5"
   "cuda-release-cuda-dense-f64-v1-jacobi-api","dense-64/torch-cuda-eager","2.1761 [1.9517–2.3584]","2.0047 [1.9835–2.0382]","0.9212 [0.8269–1.0606]","inconclusive","A 5/7; B 5/7"
   "cuda-release-cuda-dense-f64-v1-jacobi-api","dense-256/torch-cuda-eager","1.7515 [1.7515–1.7528]","1.7955 [1.7784–1.8018]","1.0252 [0.9811–1.0860]","inconclusive","A 4/6; B 4/6"
   "cuda-release-cuda-dense-f64-v1-jacobi-api","dense-1024/torch-cuda-eager","1.8859 [1.8766–1.8911]","1.8600 [1.8360–1.8600]","0.9863 [0.9363–1.0293]","within 5%","A 4/6; B 4/6"
   "cuda-release-cuda-dense-f64-v1-jacobi-api","dense-2048/torch-cuda-eager","1.9422 [1.9279–1.9806]","1.9232 [1.9163–1.9326]","0.9902 [0.9507–1.0263]","within 5%","A 3/5; B 3/5"
   "cuda-release-cuda-csr-f64-v1-none-prepared","poisson-8/torch-cuda-eager","5.3986 [5.1529–5.4193]","5.4799 [4.9214–6.9493]","1.0151 [0.9017–1.3651]","inconclusive","A 26/28; B 26/28"
   "cuda-release-cuda-csr-f64-v1-none-prepared","poisson-32/torch-cuda-eager","27.5076 [27.4445–28.1143]","26.2652 [25.4634–27.6783]","0.9548 [0.8850–1.0295]","within 5%","A 129/133; B 129/133"
   "cuda-release-cuda-csr-f64-v1-none-prepared","poisson-64/torch-cuda-eager","62.5844 [59.8516–64.5181]","64.8660 [58.8068–70.9232]","1.0365 [0.9068–1.1974]","inconclusive","A 290/297; B 290/297"
   "cuda-release-cuda-csr-f64-v1-none-prepared","poisson-128/torch-cuda-eager","79.0569 [78.8968–79.7457]","85.7055 [83.1203–87.6735]","1.0841 [1.0155–1.1229]","inconclusive","A 395/404; B 395/404"
   "cuda-release-cuda-csr-f64-v1-none-api","poisson-8/torch-cuda-eager","5.0333 [4.9863–5.0818]","5.4003 [5.2988–5.4561]","1.0729 [0.9680–1.1112]","inconclusive","A 26/28; B 26/28"
   "cuda-release-cuda-csr-f64-v1-none-api","poisson-32/torch-cuda-eager","26.2260 [24.7409–28.7561]","27.0275 [26.5483–27.4447]","1.0306 [0.9177–1.1182]","inconclusive","A 129/133; B 129/133"
   "cuda-release-cuda-csr-f64-v1-none-api","poisson-64/torch-cuda-eager","56.1735 [55.7762–56.7253]","58.5403 [57.3512–59.8139]","1.0421 [1.0025–1.0784]","inconclusive","A 290/297; B 290/297"
   "cuda-release-cuda-csr-f64-v1-none-api","poisson-128/torch-cuda-eager","83.9102 [83.4573–84.2329]","85.0552 [83.4058–86.8914]","1.0136 [0.9831–1.0736]","inconclusive","A 395/404; B 395/404"
   "cuda-release-cuda-csr-f64-v1-jacobi-prepared","poisson-8/torch-cuda-eager","7.6977 [7.4586–7.9091]","7.3065 [7.1296–7.4173]","0.9492 [0.8912–1.0100]","within 5%","A 26/28; B 26/28"
   "cuda-release-cuda-csr-f64-v1-jacobi-prepared","poisson-32/torch-cuda-eager","36.9251 [35.9981–37.8559]","36.4020 [36.1515–37.0770]","0.9858 [0.9470–1.0385]","within 5%","A 129/133; B 129/133"
   "cuda-release-cuda-csr-f64-v1-jacobi-prepared","poisson-64/torch-cuda-eager","83.3677 [82.2066–85.6605]","82.9120 [82.4620–84.0477]","0.9945 [0.9518–1.0295]","within 5%","A 290/297; B 290/297"
   "cuda-release-cuda-csr-f64-v1-jacobi-prepared","poisson-128/torch-cuda-eager","125.2048 [123.2266–130.3456]","124.0750 [117.2633–131.5399]","0.9910 [0.8901–1.0814]","inconclusive","A 395/404; B 395/404"
   "cuda-release-cuda-csr-f64-v1-jacobi-api","poisson-8/torch-cuda-eager","7.7510 [7.7066–7.7510]","7.8629 [7.5721–7.9994]","1.0144 [0.9379–1.0795]","inconclusive","A 26/28; B 26/28"
   "cuda-release-cuda-csr-f64-v1-jacobi-api","poisson-32/torch-cuda-eager","36.8730 [36.0119–37.4822]","36.5274 [35.6266–37.7420]","0.9906 [0.9490–1.0504]","inconclusive","A 129/133; B 129/133"
   "cuda-release-cuda-csr-f64-v1-jacobi-api","poisson-64/torch-cuda-eager","80.7063 [78.7506–82.7359]","81.6588 [80.7255–85.7150]","1.0118 [0.9672–1.1398]","inconclusive","A 290/297; B 290/297"
   "cuda-release-cuda-csr-f64-v1-jacobi-api","poisson-128/torch-cuda-eager","119.4077 [111.3101–126.2210]","118.6615 [117.0465–119.9322]","0.9938 [0.9191–1.0843]","inconclusive","A 395/404; B 395/404"

Focused CUDA follow-up: all original and confirmation blocks
------------------------------------------------------------

The original unpreconditioned prepared dense configuration retains both
dense-64 native/cuBLAS regressions in its two-block reports above. After
observing them, one additional eight-worker confirmation was preplanned:
A-B-B-A-B-A-A-B, four new blocks per side, retaining all four cases and
all three candidates (**8 artifacts / 96 eligible candidate/case results**).
This is a focused follow-up, rather than a replacement or repeat-until-pass
campaign. All original unfavorable observations remain included.

The standalone confirmation verdict is **inconclusive**;
the pooled all-six-blocks-per-side verdict is **inconclusive**. Neither
overwrites the original composite verdict or its case classifications.
The table shows original, confirmation-only and pooled results for every
candidate/case. Pooled medians use all 180 normalized rounds per side;
the original two blocks plus all four confirmation blocks enter the
unchanged strict comparator, with identical input/options/work counts.
Campaign timing and two-stage sampling remain uncertainty limitations.

.. csv-table:: All 12 candidate/case comparisons; original losses remain visible
   :class: benchmark-results
   :header: "case/candidate","original 2-block B/A [95% CI]","original verdict","confirmation 4-block B/A [95% CI]","confirmation verdict","pooled A ms [6-process range]","pooled B ms [6-process range]","pooled 6-block B/A [95% CI]","pooled verdict","iterations/matvecs","true R / threshold"

   "dense-64/cuda-native","1.0879 [1.0512–1.1414]","regression","1.0240 [0.9689–1.0753]","inconclusive","5.3003 [5.1936–5.7040]","5.6024 [5.1299–5.8537]","1.0570 [0.9930–1.0948]","inconclusive","A 49/51; B 49/51","A 2.0447e-06 / 2.7376e-06; B 2.0447e-06 / 2.7376e-06"
   "dense-64/cuda-cublas","1.1303 [1.0899–1.1741]","regression","1.0206 [0.9755–1.0646]","inconclusive","5.4779 [5.2995–5.6101]","5.7447 [5.4108–6.1037]","1.0487 [1.0062–1.1112]","inconclusive","A 49/51; B 49/51","A 2.2061e-06 / 2.7376e-06; B 2.2061e-06 / 2.7376e-06"
   "dense-64/torch-cuda-eager","1.1337 [1.0955–1.1701]","regression","1.0320 [0.9714–1.1050]","inconclusive","11.0559 [10.8001–11.7034]","11.9314 [10.9315–12.5702]","1.0792 [1.0012–1.1323]","inconclusive","A 49/51; B 49/51","A 2.2061e-06 / 2.7376e-06; B 2.2061e-06 / 2.7376e-06"
   "dense-256/cuda-native","0.9783 [0.9451–1.0178]","within 5%","1.0346 [1.0061–1.1177]","inconclusive","7.5779 [7.3068–8.0989]","7.7069 [7.5642–8.7228]","1.0170 [0.9833–1.0555]","inconclusive","A 67/70; B 67/70","A 5.9926e-06 / 6.3423e-06; B 5.9926e-06 / 6.3423e-06"
   "dense-256/cuda-cublas","0.9911 [0.9351–1.0528]","inconclusive","1.0288 [0.9839–1.1037]","inconclusive","7.5201 [7.1716–8.1062]","7.6596 [7.3449–8.5463]","1.0186 [0.9732–1.0657]","inconclusive","A 67/70; B 67/70","A 5.9162e-06 / 6.3423e-06; B 5.9162e-06 / 6.3423e-06"
   "dense-256/torch-cuda-eager","0.9341 [0.8480–1.0104]","within 5%","1.0274 [0.9501–1.1278]","inconclusive","15.2091 [13.9744–17.1373]","15.1653 [14.1651–17.4122]","0.9971 [0.9391–1.0732]","inconclusive","A 67/70; B 67/70","A 5.6508e-06 / 6.3423e-06; B 5.6508e-06 / 6.3423e-06"
   "dense-1024/cuda-native","1.0409 [0.9870–1.0692]","inconclusive","1.0409 [0.9869–1.1530]","inconclusive","10.7539 [10.4657–11.0708]","11.1477 [10.4355–12.7332]","1.0366 [1.0007–1.1072]","inconclusive","A 70/73; B 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05"
   "dense-1024/cuda-cublas","1.0182 [0.9696–1.0639]","inconclusive","1.0465 [0.9651–1.1967]","inconclusive","9.4229 [9.0668–10.0328]","9.7438 [9.1499–11.5553]","1.0341 [0.9790–1.1154]","inconclusive","A 70/73; B 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05"
   "dense-1024/torch-cuda-eager","0.9971 [0.9734–1.0339]","within 5%","1.0124 [0.9744–1.1043]","inconclusive","15.3863 [14.9303–15.7701]","15.5126 [15.1028–17.4171]","1.0082 [0.9803–1.0542]","inconclusive","A 70/73; B 70/73","A 9.5050e-06 / 1.0791e-05; B 9.5050e-06 / 1.0791e-05"
   "dense-2048/cuda-native","1.0046 [0.9881–1.0284]","within 5%","1.0431 [1.0169–1.0664]","inconclusive","16.4680 [16.2543–16.8339]","16.9772 [16.6495–17.5506]","1.0309 [1.0081–1.0512]","inconclusive","A 70/73; B 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05"
   "dense-2048/cuda-cublas","0.9930 [0.9770–1.0123]","within 5%","1.0525 [1.0154–1.0828]","inconclusive","15.2024 [14.9340–15.5483]","15.5924 [15.3162–16.4585]","1.0257 [1.0029–1.0657]","inconclusive","A 70/73; B 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05"
   "dense-2048/torch-cuda-eager","0.9767 [0.9613–0.9942]","within 5%","1.0592 [1.0197–1.0918]","inconclusive","21.1298 [20.8585–21.6613]","21.6999 [20.9972–22.8744]","1.0270 [0.9954–1.0732]","inconclusive","A 70/73; B 70/73","A 1.3229e-05 / 1.4741e-05; B 1.3229e-05 / 1.4741e-05"

Confirmation manifest SHA256: ``af329cb54bc05643ef63637fe3791c10e19aaba9bfa95c3eee8eaa053fb4af21``;
pooled report ``build/performance-optimization/cuda-api-confirmation/pooled-report.json``, SHA256 ``95dc150cf5bf91a4053b33961c69333580352973cb3a57f426c4e47587bd324e``.
The eight extra artifacts are included in the raw inventory below.

Explicit CUDA host/device execution modes
-----------------------------------------

All **136 native case/mode rows** are retained, including the existing
scaled/neutral sparse fixtures. There are **0 native paired rows with
different observed operation counts** across modes. Host/device specifications
differ, so none of these ratios is a strict A/B pass, confidence interval or
pure-kernel claim, even where counts happen to match. Torch executes the same
reference engine in both requested modes; the selector applies only to native
CUDA, as recorded in each artifact.

.. csv-table:: cuda-dense-f64-v1 / none / prepared / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","5.2297 [5.0615–5.3979]","T 10.1879 [9.7269–10.6489]","T 0.5133","N 49/51; T 49/51","N 2.0447e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.8863","host 49/51; device 49/51"
   "dense-64/cuda-cublas","5.1888 [5.0283–5.3493]","T 10.1879 [9.7269–10.6489]","T 0.5093","N 49/51; T 49/51","N 2.2061e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.9546","host 49/51; device 49/51"
   "dense-256/cuda-native","7.7520 [7.6805–7.8234]","T 15.9045 [15.8090–15.9999]","T 0.4874","N 67/70; T 67/70","N 5.9926e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.8331","host 67/70; device 67/70"
   "dense-256/cuda-cublas","7.5921 [7.5704–7.6137]","T 15.9045 [15.8090–15.9999]","T 0.4774","N 67/70; T 67/70","N 5.9162e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.9232","host 67/70; device 67/70"
   "dense-1024/cuda-native","10.6174 [10.1428–11.0920]","T 15.0459 [14.7692–15.3226]","T 0.7057","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.7808","host 70/73; device 70/73"
   "dense-1024/cuda-cublas","9.3404 [9.1032–9.5776]","T 15.0459 [14.7692–15.3226]","T 0.6208","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.8021","host 70/73; device 70/73"
   "dense-2048/cuda-native","17.8663 [17.6517–18.0809]","T 23.0976 [22.2838–23.9115]","T 0.7735","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8183","host 70/73; device 70/73"
   "dense-2048/cuda-cublas","16.3790 [16.2211–16.5369]","T 23.0976 [22.2838–23.9115]","T 0.7091","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8169","host 70/73; device 70/73"

.. csv-table:: cuda-dense-f64-v1 / none / prepared / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","4.6351 [4.0153–5.2548]","T 11.1370 [9.8507–12.4232]","T 0.4162","N 49/51; T 49/51","N 2.0447e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.8863","host 49/51; device 49/51"
   "dense-64/cuda-cublas","4.9534 [4.3227–5.5841]","T 11.1370 [9.8507–12.4232]","T 0.4448","N 49/51; T 49/51","N 2.2061e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.9546","host 49/51; device 49/51"
   "dense-256/cuda-native","6.4582 [5.6299–7.2865]","T 14.5579 [13.1988–15.9170]","T 0.4436","N 67/70; T 67/70","N 5.9926e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.8331","host 67/70; device 67/70"
   "dense-256/cuda-cublas","7.0090 [6.0920–7.9260]","T 14.5579 [13.1988–15.9170]","T 0.4815","N 67/70; T 67/70","N 5.9162e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.9232","host 67/70; device 67/70"
   "dense-1024/cuda-native","8.2899 [7.5810–8.9988]","T 15.8638 [13.8545–17.8730]","T 0.5226","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.7808","host 70/73; device 70/73"
   "dense-1024/cuda-cublas","7.4923 [6.7982–8.1864]","T 15.8638 [13.8545–17.8730]","T 0.4723","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.8021","host 70/73; device 70/73"
   "dense-2048/cuda-native","14.6193 [14.0363–15.2023]","T 21.6253 [20.0704–23.1802]","T 0.6760","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8183","host 70/73; device 70/73"
   "dense-2048/cuda-cublas","13.3800 [12.7557–14.0043]","T 21.6253 [20.0704–23.1802]","T 0.6187","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8169","host 70/73; device 70/73"

.. csv-table:: cuda-dense-f64-v1 / none / api / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","5.6120 [5.1284–6.0957]","T 11.7988 [10.1174–13.4802]","T 0.4756","N 49/51; T 49/51","N 2.0447e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.7559","host 49/51; device 49/51"
   "dense-64/cuda-cublas","7.2258 [6.7714–7.6802]","T 11.7988 [10.1174–13.4802]","T 0.6124","N 49/51; T 49/51","N 2.2061e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.8406","host 49/51; device 49/51"
   "dense-256/cuda-native","8.6237 [8.5016–8.7458]","T 17.5651 [17.3508–17.7795]","T 0.4910","N 67/70; T 67/70","N 5.9926e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.6538","host 67/70; device 67/70"
   "dense-256/cuda-cublas","10.2875 [10.1045–10.4705]","T 17.5651 [17.3508–17.7795]","T 0.5857","N 67/70; T 67/70","N 5.9162e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.7635","host 67/70; device 67/70"
   "dense-1024/cuda-native","10.9420 [10.7068–11.1771]","T 15.4147 [15.1728–15.6567]","T 0.7098","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.7137","host 70/73; device 70/73"
   "dense-1024/cuda-cublas","11.3650 [10.9859–11.7441]","T 15.4147 [15.1728–15.6567]","T 0.7373","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.7272","host 70/73; device 70/73"
   "dense-2048/cuda-native","16.3286 [16.0615–16.5957]","T 20.3853 [20.0484–20.7222]","T 0.8010","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8810","host 70/73; device 70/73"
   "dense-2048/cuda-cublas","16.4961 [16.2756–16.7166]","T 20.3853 [20.0484–20.7222]","T 0.8092","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8823","host 70/73; device 70/73"

.. csv-table:: cuda-dense-f64-v1 / none / api / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","4.2419 [4.0511–4.4327]","T 9.9959 [9.8419–10.1499]","T 0.4244","N 49/51; T 49/51","N 2.0447e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.7559","host 49/51; device 49/51"
   "dense-64/cuda-cublas","6.0741 [5.9928–6.1555]","T 9.9959 [9.8419–10.1499]","T 0.6077","N 49/51; T 49/51","N 2.2061e-06 / 2.7376e-06; T 2.2061e-06 / 2.7376e-06","0.8406","host 49/51; device 49/51"
   "dense-256/cuda-native","5.6380 [5.5885–5.6874]","T 13.6953 [13.2104–14.1802]","T 0.4117","N 67/70; T 67/70","N 5.9926e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.6538","host 67/70; device 67/70"
   "dense-256/cuda-cublas","7.8544 [7.7668–7.9420]","T 13.6953 [13.2104–14.1802]","T 0.5735","N 67/70; T 67/70","N 5.9162e-06 / 6.3423e-06; T 5.6508e-06 / 6.3423e-06","0.7635","host 67/70; device 67/70"
   "dense-1024/cuda-native","7.8097 [7.6377–7.9818]","T 14.2202 [13.9438–14.4966]","T 0.5492","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.7137","host 70/73; device 70/73"
   "dense-1024/cuda-cublas","8.2650 [8.0850–8.4450]","T 14.2202 [13.9438–14.4966]","T 0.5812","N 70/73; T 70/73","N 9.5050e-06 / 1.0791e-05; T 9.5050e-06 / 1.0791e-05","0.7272","host 70/73; device 70/73"
   "dense-2048/cuda-native","14.3859 [14.1940–14.5778]","T 20.5286 [20.3373–20.7199]","T 0.7008","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8810","host 70/73; device 70/73"
   "dense-2048/cuda-cublas","14.5538 [14.3774–14.7303]","T 20.5286 [20.3373–20.7199]","T 0.7090","N 70/73; T 70/73","N 1.3229e-05 / 1.4741e-05; T 1.3229e-05 / 1.4741e-05","0.8823","host 70/73; device 70/73"

.. csv-table:: cuda-dense-f64-v1 / jacobi / prepared / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","0.8282 [0.7897–0.8666]","T 1.6426 [1.6135–1.6717]","T 0.5042","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.7857","host 5/7; device 5/7"
   "dense-64/cuda-cublas","0.8797 [0.8506–0.9089]","T 1.6426 [1.6135–1.6717]","T 0.5356","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.7768","host 5/7; device 5/7"
   "dense-256/cuda-native","0.7953 [0.7761–0.8145]","T 1.3609 [1.3106–1.4113]","T 0.5844","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.7471","host 4/6; device 4/6"
   "dense-256/cuda-cublas","0.7979 [0.7780–0.8177]","T 1.3609 [1.3106–1.4113]","T 0.5862","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.7673","host 4/6; device 4/6"
   "dense-1024/cuda-native","1.0990 [1.0755–1.1224]","T 1.4610 [1.4195–1.5024]","T 0.7522","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.7940","host 4/6; device 4/6"
   "dense-1024/cuda-cublas","1.0130 [0.9836–1.0424]","T 1.4610 [1.4195–1.5024]","T 0.6934","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.7639","host 4/6; device 4/6"
   "dense-2048/cuda-native","1.3515 [1.3102–1.3927]","T 1.5775 [1.5752–1.5798]","T 0.8567","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.8670","host 3/5; device 3/5"
   "dense-2048/cuda-cublas","1.2317 [1.2141–1.2492]","T 1.5775 [1.5752–1.5798]","T 0.7808","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.8450","host 3/5; device 3/5"

.. csv-table:: cuda-dense-f64-v1 / jacobi / prepared / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","0.6507 [0.6502–0.6512]","T 1.6081 [1.6051–1.6112]","T 0.4046","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.7857","host 5/7; device 5/7"
   "dense-64/cuda-cublas","0.6834 [0.6808–0.6859]","T 1.6081 [1.6051–1.6112]","T 0.4249","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.7768","host 5/7; device 5/7"
   "dense-256/cuda-native","0.5942 [0.5731–0.6152]","T 1.3558 [1.3043–1.4073]","T 0.4382","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.7471","host 4/6; device 4/6"
   "dense-256/cuda-cublas","0.6122 [0.5678–0.6566]","T 1.3558 [1.3043–1.4073]","T 0.4515","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.7673","host 4/6; device 4/6"
   "dense-1024/cuda-native","0.8725 [0.8607–0.8844]","T 1.4265 [1.4014–1.4516]","T 0.6117","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.7940","host 4/6; device 4/6"
   "dense-1024/cuda-cublas","0.7738 [0.7672–0.7804]","T 1.4265 [1.4014–1.4516]","T 0.5425","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.7639","host 4/6; device 4/6"
   "dense-2048/cuda-native","1.1718 [1.1712–1.1723]","T 1.5557 [1.5396–1.5719]","T 0.7532","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.8670","host 3/5; device 3/5"
   "dense-2048/cuda-cublas","1.0408 [1.0323–1.0492]","T 1.5557 [1.5396–1.5719]","T 0.6690","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.8450","host 3/5; device 3/5"

.. csv-table:: cuda-dense-f64-v1 / jacobi / api / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","1.0705 [0.9634–1.1776]","T 2.4117 [2.1477–2.6758]","T 0.4439","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.8973","host 5/7; device 5/7"
   "dense-64/cuda-cublas","2.5376 [2.5289–2.5464]","T 2.4117 [2.1477–2.6758]","T 1.0522","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.9877","host 5/7; device 5/7"
   "dense-256/cuda-native","1.0469 [0.8820–1.2118]","T 2.0801 [1.7283–2.4320]","T 0.5033","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.7620","host 4/6; device 4/6"
   "dense-256/cuda-cublas","2.6084 [2.4740–2.7427]","T 2.0801 [1.7283–2.4320]","T 1.2539","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.9459","host 4/6; device 4/6"
   "dense-1024/cuda-native","1.2425 [1.1422–1.3428]","T 1.8466 [1.7763–1.9169]","T 0.6729","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.8498","host 4/6; device 4/6"
   "dense-1024/cuda-cublas","2.6970 [2.5803–2.8136]","T 1.8466 [1.7763–1.9169]","T 1.4605","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.9501","host 4/6; device 4/6"
   "dense-2048/cuda-native","1.4403 [1.3696–1.5111]","T 2.0289 [1.9684–2.0894]","T 0.7099","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.9118","host 3/5; device 3/5"
   "dense-2048/cuda-cublas","2.7927 [2.7621–2.8232]","T 2.0289 [1.9684–2.0894]","T 1.3764","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.9283","host 3/5; device 3/5"

.. csv-table:: cuda-dense-f64-v1 / jacobi / api / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "dense-64/cuda-native","0.9605 [0.8819–1.0391]","T 2.2898 [2.2732–2.3064]","T 0.4195","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.8973","host 5/7; device 5/7"
   "dense-64/cuda-cublas","2.5063 [2.3804–2.6323]","T 2.2898 [2.2732–2.3064]","T 1.0945","N 5/7; T 5/7","N 1.1410e-07 / 2.7376e-06; T 1.1410e-07 / 2.7376e-06","0.9877","host 5/7; device 5/7"
   "dense-256/cuda-native","0.7978 [0.6916–0.9040]","T 1.9435 [1.7456–2.1414]","T 0.4105","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.7620","host 4/6; device 4/6"
   "dense-256/cuda-cublas","2.4673 [2.3623–2.5722]","T 1.9435 [1.7456–2.1414]","T 1.2695","N 4/6; T 4/6","N 1.0990e-06 / 6.3423e-06; T 1.0990e-06 / 6.3423e-06","0.9459","host 4/6; device 4/6"
   "dense-1024/cuda-native","1.0559 [0.9996–1.1121]","T 2.0907 [2.0526–2.1287]","T 0.5050","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.8498","host 4/6; device 4/6"
   "dense-1024/cuda-cublas","2.5624 [2.5384–2.5863]","T 2.0907 [2.0526–2.1287]","T 1.2256","N 4/6; T 4/6","N 1.7223e-07 / 1.0791e-05; T 1.7223e-07 / 1.0791e-05","0.9501","host 4/6; device 4/6"
   "dense-2048/cuda-native","1.3134 [1.2695–1.3572]","T 2.1172 [2.0805–2.1538]","T 0.6203","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.9118","host 3/5; device 3/5"
   "dense-2048/cuda-cublas","2.5925 [2.5853–2.5997]","T 2.1172 [2.0805–2.1538]","T 1.2245","N 3/5; T 3/5","N 1.2450e-05 / 1.4741e-05; T 1.2450e-05 / 1.4741e-05","0.9283","host 3/5; device 3/5"

.. csv-table:: cuda-csr-f64-v1 / none / prepared / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","2.7399 [2.5110–2.9689]","T 5.3008 [5.2385–5.3631]","T 0.5169","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.9217","host 26/28; device 26/28"
   "poisson-32/cuda-native","14.0195 [13.8777–14.1614]","T 25.9178 [25.7227–26.1129]","T 0.5409","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.8858","host 129/133; device 129/133"
   "poisson-64/cuda-native","31.6945 [31.2449–32.1442]","T 58.8995 [57.2275–60.5714]","T 0.5381","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.7797","host 290/297; device 290/297"
   "poisson-128/cuda-native","45.2123 [43.6536–46.7711]","T 82.1802 [80.2220–84.1384]","T 0.5502","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.7743","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / none / prepared / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","2.5254 [2.1907–2.8601]","T 5.8979 [5.0851–6.7106]","T 0.4282","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.9217","host 26/28; device 26/28"
   "poisson-32/cuda-native","12.4191 [10.4004–14.4378]","T 29.4741 [24.9728–33.9753]","T 0.4214","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.8858","host 129/133; device 129/133"
   "poisson-64/cuda-native","24.7111 [22.8447–26.5775]","T 60.9672 [54.9134–67.0211]","T 0.4053","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.7797","host 290/297; device 290/297"
   "poisson-128/cuda-native","35.0083 [34.2098–35.8068]","T 79.4906 [78.4069–80.5744]","T 0.4404","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.7743","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / none / api / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","2.9020 [2.8806–2.9234]","T 5.2838 [5.2076–5.3599]","T 0.5492","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.7946","host 26/28; device 26/28"
   "poisson-32/cuda-native","14.0893 [13.9796–14.1989]","T 26.8912 [26.4586–27.3239]","T 0.5239","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.7885","host 129/133; device 129/133"
   "poisson-64/cuda-native","31.4019 [30.3970–32.4068]","T 57.9259 [56.8239–59.0279]","T 0.5421","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.8003","host 290/297; device 290/297"
   "poisson-128/cuda-native","47.8586 [46.9411–48.7761]","T 85.3807 [84.1134–86.6479]","T 0.5605","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.7888","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / none / api / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","2.3060 [2.2463–2.3657]","T 5.2951 [5.0866–5.5036]","T 0.4355","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.7946","host 26/28; device 26/28"
   "poisson-32/cuda-native","11.1092 [10.9412–11.2773]","T 25.5321 [25.3357–25.7285]","T 0.4351","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.7885","host 129/133; device 129/133"
   "poisson-64/cuda-native","25.1323 [24.6455–25.6191]","T 55.8848 [55.4353–56.3344]","T 0.4497","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.8003","host 290/297; device 290/297"
   "poisson-128/cuda-native","37.7499 [36.0882–39.4117]","T 81.3830 [78.6878–84.0782]","T 0.4639","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.7888","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / jacobi / prepared / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","4.1273 [3.9672–4.2875]","T 8.1443 [7.4191–8.8695]","T 0.5068","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.6880","host 26/28; device 26/28"
   "poisson-32/cuda-native","20.7258 [19.8771–21.5744]","T 39.0207 [35.7897–42.2517]","T 0.5311","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.6543","host 129/133; device 129/133"
   "poisson-64/cuda-native","46.7434 [46.0425–47.4442]","T 87.3234 [82.0373–92.6095]","T 0.5353","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.6746","host 290/297; device 290/297"
   "poisson-128/cuda-native","70.9723 [68.0638–73.8808]","T 123.7102 [121.0454–126.3750]","T 0.5737","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.5943","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / jacobi / prepared / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","2.8397 [2.6991–2.9803]","T 8.8733 [7.7897–9.9569]","T 0.3200","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.6880","host 26/28; device 26/28"
   "poisson-32/cuda-native","13.5609 [12.8713–14.2505]","T 43.1800 [38.4016–47.9584]","T 0.3141","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.6543","host 129/133; device 129/133"
   "poisson-64/cuda-native","31.5340 [29.8967–33.1713]","T 98.8009 [88.0906–109.5113]","T 0.3192","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.6746","host 290/297; device 290/297"
   "poisson-128/cuda-native","42.1770 [39.7207–44.6332]","T 125.0549 [114.7683–135.3416]","T 0.3373","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.5943","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / jacobi / api / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","4.4298 [4.0632–4.7964]","T 8.7426 [7.5553–9.9300]","T 0.5067","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.6022","host 26/28; device 26/28"
   "poisson-32/cuda-native","21.2212 [20.3019–22.1405]","T 41.2328 [37.2537–45.2119]","T 0.5147","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.6027","host 129/133; device 129/133"
   "poisson-64/cuda-native","50.3621 [45.9302–54.7940]","T 96.8903 [83.9771–109.8034]","T 0.5198","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.5831","host 290/297; device 290/297"
   "poisson-128/cuda-native","75.5428 [69.5799–81.5057]","T 131.9680 [119.7696–144.1663]","T 0.5724","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.6160","host 395/404; device 395/404"

.. csv-table:: cuda-csr-f64-v1 / jacobi / api / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "poisson-8/cuda-native","2.6678 [2.6585–2.6772]","T 7.7493 [7.5791–7.9194]","T 0.3443","N 26/28; T 26/28","N 2.0123e-07 / 3.0549e-07; T 2.0123e-07 / 3.0549e-07","0.6022","host 26/28; device 26/28"
   "poisson-32/cuda-native","12.7910 [12.7340–12.8480]","T 37.0022 [36.7818–37.2226]","T 0.3457","N 129/133; T 129/133","N 1.3276e-06 / 1.3672e-06; T 1.3276e-06 / 1.3672e-06","0.6027","host 129/133; device 129/133"
   "poisson-64/cuda-native","29.3676 [28.6178–30.1175]","T 90.0141 [89.1021–90.9260]","T 0.3263","N 290/297; T 290/297","N 2.7729e-06 / 2.8101e-06; T 2.7729e-06 / 2.8101e-06","0.5831","host 290/297; device 290/297"
   "poisson-128/cuda-native","46.5333 [43.2862–49.7804]","T 139.8668 [122.0252–157.7085]","T 0.3327","N 395/404; T 395/404","N 5.6614e-06 / 5.7493e-06; T 5.6614e-06 / 5.7493e-06","0.6160","host 395/404; device 395/404"

.. csv-table:: sparse-solvers / none / prepared / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","61.9568 [60.6663–63.2472]","T 114.4294 [111.8844–116.9745]","T 0.5414","N 575/588; T 574/587","N 6.0643e-05 / 6.3862e-05; T 5.7011e-05 / 6.3862e-05","0.8624","host 575/588; device 575/588"
   "scaled-poisson-2d-32/cuda-native","187.7263 [185.6451–189.8074]","T 344.4783 [343.0650–345.8917]","T 0.5450","N 1498/1529; T 1498/1529","N 1.4484e-04 / 1.4732e-04; T 1.4500e-04 / 1.4732e-04","0.6650","host 1498/1529; device 1498/1529"
   "scaled-poisson-3d-6/cuda-native","32.0446 [30.2091–33.8801]","T 56.8623 [55.0504–58.6741]","T 0.5635","N 261/268; T 249/255","N 8.8109e-05 / 8.9682e-05; T 8.9560e-05 / 8.9682e-05","0.7393","host 261/268; device 261/268"
   "scaled-poisson-3d-10/cuda-native","53.9478 [53.8026–54.0929]","T 98.2136 [94.7792–101.6480]","T 0.5493","N 444/454; T 444/454","N 2.0951e-04 / 2.1493e-04; T 2.0952e-04 / 2.1493e-04","0.7490","host 444/454; device 444/454"
   "poisson-2d-16-neutral/cuda-native","6.4635 [6.1838–6.7432]","T 12.0000 [11.5557–12.4443]","T 0.5386","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.6211","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / none / prepared / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","53.4337 [47.1753–59.6921]","T 121.6392 [109.8958–133.3825]","T 0.4393","N 575/588; T 574/587","N 6.0643e-05 / 6.3862e-05; T 5.7011e-05 / 6.3862e-05","0.8624","host 575/588; device 575/588"
   "scaled-poisson-2d-32/cuda-native","124.8434 [121.6877–127.9992]","T 292.7929 [288.3054–297.2804]","T 0.4264","N 1498/1529; T 1498/1529","N 1.4484e-04 / 1.4732e-04; T 1.4500e-04 / 1.4732e-04","0.6650","host 1498/1529; device 1498/1529"
   "scaled-poisson-3d-6/cuda-native","23.6919 [23.4651–23.9186]","T 50.9696 [48.9399–52.9994]","T 0.4648","N 261/268; T 249/255","N 8.8109e-05 / 8.9682e-05; T 8.9560e-05 / 8.9682e-05","0.7393","host 261/268; device 261/268"
   "scaled-poisson-3d-10/cuda-native","40.4085 [39.9678–40.8491]","T 91.0312 [88.8046–93.2578]","T 0.4439","N 444/454; T 444/454","N 2.0951e-04 / 2.1493e-04; T 2.0952e-04 / 2.1493e-04","0.7490","host 444/454; device 444/454"
   "poisson-2d-16-neutral/cuda-native","4.0148 [3.9710–4.0586]","T 9.3549 [9.2609–9.4490]","T 0.4292","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.6211","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / none / api / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","60.5632 [60.2573–60.8691]","T 110.2223 [110.1876–110.2570]","T 0.5495","N 575/588; T 574/587","N 6.0643e-05 / 6.3862e-05; T 5.7011e-05 / 6.3862e-05","0.8846","host 575/588; device 575/588"
   "scaled-poisson-2d-32/cuda-native","176.9493 [176.7551–177.1435]","T 314.5587 [314.1296–314.9878]","T 0.5625","N 1498/1529; T 1498/1529","N 1.4484e-04 / 1.4732e-04; T 1.4500e-04 / 1.4732e-04","0.8008","host 1498/1529; device 1498/1529"
   "scaled-poisson-3d-6/cuda-native","34.7799 [34.5721–34.9876]","T 57.6891 [55.4815–59.8968]","T 0.6029","N 261/268; T 249/255","N 8.8109e-05 / 8.9682e-05; T 8.9560e-05 / 8.9682e-05","0.7642","host 261/268; device 261/268"
   "scaled-poisson-3d-10/cuda-native","56.4389 [55.5672–57.3106]","T 97.6172 [93.7302–101.5043]","T 0.5782","N 444/454; T 444/454","N 2.0951e-04 / 2.1493e-04; T 2.0952e-04 / 2.1493e-04","0.7099","host 444/454; device 444/454"
   "poisson-2d-16-neutral/cuda-native","5.2355 [5.1913–5.2798]","T 9.5784 [9.3106–9.8462]","T 0.5466","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.8322","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / none / api / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","53.5765 [47.4015–59.7514]","T 119.6268 [110.5881–128.6655]","T 0.4479","N 575/588; T 574/587","N 6.0643e-05 / 6.3862e-05; T 5.7011e-05 / 6.3862e-05","0.8846","host 575/588; device 575/588"
   "scaled-poisson-2d-32/cuda-native","141.7072 [122.1136–161.3009]","T 330.8429 [282.7930–378.8928]","T 0.4283","N 1498/1529; T 1498/1529","N 1.4484e-04 / 1.4732e-04; T 1.4500e-04 / 1.4732e-04","0.8008","host 1498/1529; device 1498/1529"
   "scaled-poisson-3d-6/cuda-native","26.5804 [22.4621–30.6986]","T 56.3327 [50.9981–61.6674]","T 0.4718","N 261/268; T 249/255","N 8.8109e-05 / 8.9682e-05; T 8.9560e-05 / 8.9682e-05","0.7642","host 261/268; device 261/268"
   "scaled-poisson-3d-10/cuda-native","40.0675 [39.7418–40.3932]","T 90.2162 [88.8872–91.5452]","T 0.4441","N 444/454; T 444/454","N 2.0951e-04 / 2.1493e-04; T 2.0952e-04 / 2.1493e-04","0.7099","host 444/454; device 444/454"
   "poisson-2d-16-neutral/cuda-native","4.3572 [4.1610–4.5534]","T 10.0133 [9.8601–10.1665]","T 0.4351","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.8322","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / jacobi / prepared / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","7.5418 [7.3822–7.7014]","T 14.0710 [13.2091–14.9329]","T 0.5360","N 48/50; T 48/50","N 5.8194e-05 / 6.3862e-05; T 5.8194e-05 / 6.3862e-05","0.6266","host 48/50; device 48/50"
   "scaled-poisson-2d-32/cuda-native","18.0006 [17.7584–18.2427]","T 32.3252 [31.3937–33.2566]","T 0.5569","N 115/119; T 115/119","N 1.3686e-04 / 1.4732e-04; T 1.3686e-04 / 1.4732e-04","0.6211","host 115/119; device 115/119"
   "scaled-poisson-3d-6/cuda-native","3.9948 [3.8694–4.1203]","T 7.3139 [6.8884–7.7394]","T 0.5462","N 25/27; T 25/27","N 3.9457e-05 / 8.9682e-05; T 3.9457e-05 / 8.9682e-05","0.6378","host 25/27; device 25/27"
   "scaled-poisson-3d-10/cuda-native","5.6938 [5.5869–5.8006]","T 10.0378 [10.0031–10.0725]","T 0.5672","N 37/39; T 37/39","N 2.0828e-04 / 2.1493e-04; T 2.0828e-04 / 2.1493e-04","0.6640","host 37/39; device 37/39"
   "poisson-2d-16-neutral/cuda-native","7.5668 [7.4850–7.6487]","T 13.6058 [13.5353–13.6764]","T 0.5561","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.6309","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / jacobi / prepared / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","4.7258 [4.6256–4.8259]","T 13.2945 [12.9879–13.6012]","T 0.3555","N 48/50; T 48/50","N 5.8194e-05 / 6.3862e-05; T 5.8194e-05 / 6.3862e-05","0.6266","host 48/50; device 48/50"
   "scaled-poisson-2d-32/cuda-native","11.1809 [11.0801–11.2818]","T 31.0593 [30.6399–31.4787]","T 0.3600","N 115/119; T 115/119","N 1.3686e-04 / 1.4732e-04; T 1.3686e-04 / 1.4732e-04","0.6211","host 115/119; device 115/119"
   "scaled-poisson-3d-6/cuda-native","2.5477 [2.5142–2.5812]","T 6.6871 [6.6034–6.7708]","T 0.3810","N 25/27; T 25/27","N 3.9457e-05 / 8.9682e-05; T 3.9457e-05 / 8.9682e-05","0.6378","host 25/27; device 25/27"
   "scaled-poisson-3d-10/cuda-native","3.7806 [3.7074–3.8537]","T 10.3671 [10.1984–10.5358]","T 0.3647","N 37/39; T 37/39","N 2.0828e-04 / 2.1493e-04; T 2.0828e-04 / 2.1493e-04","0.6640","host 37/39; device 37/39"
   "poisson-2d-16-neutral/cuda-native","4.7736 [4.7668–4.7805]","T 14.7966 [14.0182–15.5749]","T 0.3226","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.6309","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / jacobi / api / execution=host
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","7.4919 [7.2259–7.7579]","T 13.6826 [12.9277–14.4376]","T 0.5476","N 48/50; T 48/50","N 5.8194e-05 / 6.3862e-05; T 5.8194e-05 / 6.3862e-05","0.6171","host 48/50; device 48/50"
   "scaled-poisson-2d-32/cuda-native","17.8911 [17.0860–18.6962]","T 32.3864 [30.6734–34.0993]","T 0.5524","N 115/119; T 115/119","N 1.3686e-04 / 1.4732e-04; T 1.3686e-04 / 1.4732e-04","0.6169","host 115/119; device 115/119"
   "scaled-poisson-3d-6/cuda-native","4.0610 [3.8913–4.2307]","T 7.3110 [6.8000–7.8220]","T 0.5555","N 25/27; T 25/27","N 3.9457e-05 / 8.9682e-05; T 3.9457e-05 / 8.9682e-05","0.6606","host 25/27; device 25/27"
   "scaled-poisson-3d-10/cuda-native","5.9193 [5.6871–6.1514]","T 10.7216 [10.1790–11.2641]","T 0.5521","N 37/39; T 37/39","N 2.0828e-04 / 2.1493e-04; T 2.0828e-04 / 2.1493e-04","0.6335","host 37/39; device 37/39"
   "poisson-2d-16-neutral/cuda-native","7.7049 [7.5925–7.8172]","T 14.2661 [13.5536–14.9786]","T 0.5401","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.6365","host 50/52; device 50/52"

.. csv-table:: sparse-solvers / jacobi / api / execution=device
   :class: benchmark-results
   :header: "case/native","native ms [process range]","Torch ms [range]","native/Torch","iterations/matvecs","true R / threshold","device/host (descriptive)","host/device work"

   "scaled-poisson-2d-16/cuda-native","4.6230 [4.5121–4.7338]","T 13.3024 [13.1554–13.4493]","T 0.3475","N 48/50; T 48/50","N 5.8194e-05 / 6.3862e-05; T 5.8194e-05 / 6.3862e-05","0.6171","host 48/50; device 48/50"
   "scaled-poisson-2d-32/cuda-native","11.0362 [10.9584–11.1140]","T 33.0431 [32.5678–33.5184]","T 0.3340","N 115/119; T 115/119","N 1.3686e-04 / 1.4732e-04; T 1.3686e-04 / 1.4732e-04","0.6169","host 115/119; device 115/119"
   "scaled-poisson-3d-6/cuda-native","2.6827 [2.6765–2.6889]","T 7.4435 [7.1963–7.6907]","T 0.3604","N 25/27; T 25/27","N 3.9457e-05 / 8.9682e-05; T 3.9457e-05 / 8.9682e-05","0.6606","host 25/27; device 25/27"
   "scaled-poisson-3d-10/cuda-native","3.7497 [3.6421–3.8574]","T 11.2794 [10.5953–11.9635]","T 0.3324","N 37/39; T 37/39","N 2.0828e-04 / 2.1493e-04; T 2.0828e-04 / 2.1493e-04","0.6335","host 37/39; device 37/39"
   "poisson-2d-16-neutral/cuda-native","4.9042 [4.6365–5.1719]","T 14.8631 [14.2578–15.4685]","T 0.3300","N 50/52; T 50/52","N 4.8293e-07 / 6.3194e-07; T 4.8293e-07 / 6.3194e-07","0.6365","host 50/52; device 50/52"

Current native/reference medians and remaining losses
-----------------------------------------------------

A lower measured median means native/reference < 1; this is a different
ratio from the A/B optimization gate. These descriptive within-artifact
rankings do not establish statistical significance. All ties or higher
native medians are listed, with no excluded controls or favorable-case filter.

.. csv-table:: Lower native median counts, with full native/reference ratio range
   :class: benchmark-results
   :header: "batch","suite","mode","fused","native","reference","lower/total","ties","ratio range"

   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","8/16","0","0.1395–1.3031"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","torch-eager-out","12/16","0","0.0987–1.2326"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","4/16","0","0.2497–3.7019"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","4/16","0","0.1597–3.4969"
   "CUDA API ABBA","cuda-dense-f64-v1","host","True","cuda-native","torch-cuda-eager","16/16","0","0.4610–0.8418"
   "CUDA API ABBA","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","12/16","0","0.4872–1.4220"
   "CUDA API ABBA","cuda-csr-f64-v1","host","True","cuda-native","torch-cuda-eager","16/16","0","0.5109–0.6047"
   "CUDA modes","cuda-dense-f64-v1","host","True","cuda-native","torch-cuda-eager","16/16","0","0.4439–0.8567"
   "CUDA modes","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","12/16","0","0.4774–1.4605"
   "CUDA modes","cuda-dense-f64-v1","device","True","cuda-native","torch-cuda-eager","16/16","0","0.4046–0.7532"
   "CUDA modes","cuda-dense-f64-v1","device","True","cuda-cublas","torch-cuda-eager","12/16","0","0.4249–1.2695"
   "CUDA modes","cuda-csr-f64-v1","host","True","cuda-native","torch-cuda-eager","16/16","0","0.5067–0.5737"
   "CUDA modes","cuda-csr-f64-v1","device","True","cuda-native","torch-cuda-eager","16/16","0","0.3141–0.4639"
   "CUDA modes","sparse-solvers","host","True","cuda-native","torch-cuda-eager","20/20","0","0.5360–0.6029"
   "CUDA modes","sparse-solvers","device","True","cuda-native","torch-cuda-eager","20/20","0","0.3226–0.4718"

.. csv-table:: Every remaining tie or higher native median
   :class: benchmark-results
   :header: "batch","suite","mode","fused","native","reference","M","scope","case","native/reference"

   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","none","prepared","dense-1024","1.0423"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","none","prepared","dense-2048","1.2577"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","none","api","dense-1024","1.0116"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","none","api","dense-2048","1.3031"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","jacobi","prepared","dense-1024","1.0243"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","jacobi","prepared","dense-2048","1.2193"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","jacobi","api","dense-1024","1.0362"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","numpy-out","jacobi","api","dense-2048","1.2540"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","torch-eager-out","none","prepared","dense-2048","1.1682"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","torch-eager-out","none","api","dense-2048","1.2326"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","torch-eager-out","jacobi","prepared","dense-2048","1.1356"
   "CPU ABBA","cpu-dense-f64-v1","host","True","native","torch-eager-out","jacobi","api","dense-2048","1.1315"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","none","prepared","dense-256","2.3402"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","none","prepared","dense-1024","3.4053"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","none","prepared","dense-2048","2.4884"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","none","api","dense-256","2.3932"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","none","api","dense-1024","3.7019"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","none","api","dense-2048","2.4692"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","jacobi","prepared","dense-256","2.2457"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","jacobi","prepared","dense-1024","2.9609"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","jacobi","prepared","dense-2048","2.4126"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","jacobi","api","dense-256","2.2186"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","jacobi","api","dense-1024","3.4901"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","numpy-out","jacobi","api","dense-2048","2.3843"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","none","prepared","dense-256","1.9957"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","none","prepared","dense-1024","3.3156"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","none","prepared","dense-2048","2.3363"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","none","api","dense-256","1.9608"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","none","api","dense-1024","3.4969"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","none","api","dense-2048","2.3725"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","jacobi","prepared","dense-256","1.8173"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","jacobi","prepared","dense-1024","2.7485"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","jacobi","prepared","dense-2048","2.1782"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","jacobi","api","dense-256","1.4291"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","jacobi","api","dense-1024","3.0452"
   "CPU ABBA","cpu-dense-f64-v1","host","False","native","torch-eager-out","jacobi","api","dense-2048","2.2183"
   "CUDA API ABBA","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-64","1.1575"
   "CUDA API ABBA","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-256","1.4119"
   "CUDA API ABBA","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-1024","1.4220"
   "CUDA API ABBA","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-2048","1.4128"
   "CUDA modes","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-64","1.0522"
   "CUDA modes","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-256","1.2539"
   "CUDA modes","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-1024","1.4605"
   "CUDA modes","cuda-dense-f64-v1","host","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-2048","1.3764"
   "CUDA modes","cuda-dense-f64-v1","device","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-64","1.0945"
   "CUDA modes","cuda-dense-f64-v1","device","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-256","1.2695"
   "CUDA modes","cuda-dense-f64-v1","device","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-1024","1.2256"
   "CUDA modes","cuda-dense-f64-v1","device","True","cuda-cublas","torch-cuda-eager","jacobi","api","dense-2048","1.2245"

Diagnostic profiles: control traffic and CPU operations
-------------------------------------------------------

These profiles are diagnostics, excluded from the formal framework scores,
strict-gate verdict counts and native/reference rankings above. Nsight
instrumentation changes timing, and the CPU probe fixes p=x=b instead of
recording evolving CG directions. Neither diagnostic supports additive
phase attribution or a universal solver speed claim.

The five CUDA captures use the same frozen candidate binary as the formal
host/device characterization, three warmups and ten prepared solves per
capture. The before/after independent residual gates remain valid.
Counts below are totals for ten solves: CUDA memcpy/synchronize calls
include initialization, status/norm traffic and completion boundaries,
rather than only recurrence packets. GPU kernel instances are not matvec
counts; one operation can require several kernels, especially cuBLAS.

For dense-1024 Jacobi, host and device both perform four iterations and
six matvecs per solve. Device execution reduces observed memcpy and sync
calls while increasing kernel launches; this tradeoff is retained rather
than treating fewer copies as proof of lower complete-solve latency.
Explicit device CG retains 69 iterations / 139 matvecs per solve and its
much larger launch/status traffic. The CSR capture retains 129 / 133.
Only the uninstrumented formal measurements support performance rankings.

.. csv-table:: Nsight API and GPU activity counts; ten solves per capture
   :class: benchmark-results
   :header: "mode/kernel/case/M/algorithm","solves","iterations/matvecs per solve","cudaMemcpyAsync calls","cudaStreamSynchronize calls","cudaLaunchKernel calls","GPU kernel instances","max true R / threshold"

   "device/cuda-cublas/dense-1024/jacobi/standard","10","4/6","100","120","630","630","1.722285e-07 / 1.079068e-05"
   "device/cuda-native/dense-1024/jacobi/standard","10","4/6","100","120","570","570","1.722285e-07 / 1.079068e-05"
   "device/cuda-native/dense-1024/none/explicit","10","69/139","1410","1440","10410","10410","8.844894e-06 / 1.079068e-05"
   "device/cuda-native/poisson-32/jacobi/standard","10","129/133","1430","1410","15670","15670","1.327602e-06 / 1.367232e-06"
   "host/cuda-native/dense-1024/jacobi/standard","10","4/6","170","190","330","330","1.722285e-07 / 1.079068e-05"

.. csv-table:: Every captured GPU kernel; extra launches are retained
   :class: benchmark-results
   :header: "capture","kernel","instances"

   "device/cuda-cublas/dense-1024/jacobi/standard","``void gemv2T_kernel_val<int, int, double, double, double, double, (int)128, (int)16, (int)2, (int)2, (bool)0, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const double>, cublasGemvTensorStridedBatched<const double>, cublasGemvTensorStridedBatched<double>, double>>(T13, T6, T6)``","60"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::finish_kernel(const double *, unsigned long, double *)``","160"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::dot_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, double *)``","50"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::multiply_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, double *, bool)``","50"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::guarded_update(cg::cuda::VectorView, cg::cuda::VectorView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, const cg::cuda::detail::DeviceState *, double *)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::check_alpha(cg::cuda::detail::DeviceState *, bool)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::stable_norm_kernel(cg::cuda::ConstVectorView, const double *, double *)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::check_beta(cg::cuda::detail::DeviceState *, bool, bool)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::residual_kernel(cg::cuda::ConstVectorView, cg::cuda::VectorView, bool, double *)``","20"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::check_residual(cg::cuda::detail::DeviceState *, bool)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::guarded_direction(cg::cuda::ConstVectorView, cg::cuda::VectorView, const cg::cuda::detail::DeviceState *, double)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::begin_iteration(cg::cuda::detail::DeviceState *, bool, bool)``","40"
   "device/cuda-cublas/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::initialize_state(cg::cuda::detail::DeviceState *, double, double, double)``","10"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::dense_kernel(cg::cuda::DenseMatrixView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, int, bool, double *)``","60"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::finish_kernel(const double *, unsigned long, double *)``","160"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::multiply_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, double *, bool)``","50"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::guarded_update(cg::cuda::VectorView, cg::cuda::VectorView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, const cg::cuda::detail::DeviceState *, double *)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::check_alpha(cg::cuda::detail::DeviceState *, bool)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::stable_norm_kernel(cg::cuda::ConstVectorView, const double *, double *)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::check_beta(cg::cuda::detail::DeviceState *, bool, bool)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::check_residual(cg::cuda::detail::DeviceState *, bool)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::guarded_direction(cg::cuda::ConstVectorView, cg::cuda::VectorView, const cg::cuda::detail::DeviceState *, double)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::begin_iteration(cg::cuda::detail::DeviceState *, bool, bool)``","40"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::dot_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, double *)``","10"
   "device/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::initialize_state(cg::cuda::detail::DeviceState *, double, double, double)``","10"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::dense_kernel(cg::cuda::DenseMatrixView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, int, bool, double *)``","1390"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::finish_kernel(const double *, unsigned long, double *)``","2780"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::dot_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, double *)``","1390"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::guarded_axpy(cg::cuda::ConstVectorView, cg::cuda::VectorView, const cg::cuda::detail::DeviceState *, bool)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::check_alpha(cg::cuda::detail::DeviceState *, bool)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::guarded_direction(cg::cuda::ConstVectorView, cg::cuda::VectorView, const cg::cuda::detail::DeviceState *, double)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::check_beta(cg::cuda::detail::DeviceState *, bool, bool)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::check_residual(cg::cuda::detail::DeviceState *, bool)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::stable_norm_kernel(cg::cuda::ConstVectorView, const double *, double *)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::begin_iteration(cg::cuda::detail::DeviceState *, bool, bool)``","690"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::linear_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, double, double)``","10"
   "device/cuda-native/dense-1024/none/explicit","``cg::cuda::detail::<unnamed>::initialize_state(cg::cuda::detail::DeviceState *, double, double, double)``","10"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::finish_kernel(const double *, unsigned long, double *)``","3950"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::csr_kernel(cg::cuda::CsrMatrixView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, int, bool, double *)``","1330"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::guarded_update(cg::cuda::VectorView, cg::cuda::VectorView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, const cg::cuda::detail::DeviceState *, double *)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::multiply_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, double *, bool)``","1320"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::check_beta(cg::cuda::detail::DeviceState *, bool, bool)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::check_alpha(cg::cuda::detail::DeviceState *, bool)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::check_residual(cg::cuda::detail::DeviceState *, bool)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::guarded_direction(cg::cuda::ConstVectorView, cg::cuda::VectorView, const cg::cuda::detail::DeviceState *, double)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::stable_norm_kernel(cg::cuda::ConstVectorView, const double *, double *)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::begin_iteration(cg::cuda::detail::DeviceState *, bool, bool)``","1290"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::initialize_state(cg::cuda::detail::DeviceState *, double, double, double)``","30"
   "device/cuda-native/poisson-32/jacobi/standard","``cg::cuda::detail::<unnamed>::dot_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, double *)``","10"
   "host/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::dense_kernel(cg::cuda::DenseMatrixView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, int, bool, double *)``","60"
   "host/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::finish_kernel(const double *, unsigned long, double *)``","150"
   "host/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::update_kernel(cg::cuda::VectorView, cg::cuda::VectorView, cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, double, double *)``","40"
   "host/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::multiply_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, double *, bool)``","40"
   "host/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::linear_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, cg::cuda::VectorView, double, double)``","30"
   "host/cuda-native/dense-1024/jacobi/standard","``cg::cuda::detail::<unnamed>::dot_kernel(cg::cuda::ConstVectorView, cg::cuda::ConstVectorView, double *)``","10"

.. csv-table:: Diagnostic JSON/CSV and actual compiled-stage identities
   :class: benchmark-results
   :header: "profile JSON","JSON SHA256","API CSV SHA256","kernel CSV SHA256","compiled revision","stage identity SHA256","CUDA module SHA256"

   "engine-device-cuda-cublas-dense-1024-jacobi-standard.json","f0b2c06bd145ab8453726af83d51dc6177d92d6b764258a3cf4482121473bfa6","4aa962d25ac3e42ccfcee879f0c31f47cb2b516d43d64ebd8bf1c6ee33965266","f6b70e676635fdca3df2bbad3c7c11699e1feff1fc05274085c55bfaf279e9d6","abffa451c26d96f2a4ee8014333a8581afde46fb","37849a4ee29c8a1755b9376633fba92c2a38ef2cc7932c5297f2396449ae1a31","d487e33440807c40cb789434ff40e5f3fbfe58831e3c655c384776b0f984f162"
   "engine-device-cuda-native-dense-1024-jacobi-standard.json","a399fa76241de146435b1e2df7f3080e6a9e42540146baaa7de8a78e01a10085","0277b6ae2a98964a015e07ffc513dec223b442f56f9b97b6281d871428beb2b9","ec93777a9d62b5e59635219747ced604707f40c487b907682d5b2393e1715fa5","abffa451c26d96f2a4ee8014333a8581afde46fb","37849a4ee29c8a1755b9376633fba92c2a38ef2cc7932c5297f2396449ae1a31","d487e33440807c40cb789434ff40e5f3fbfe58831e3c655c384776b0f984f162"
   "engine-device-cuda-native-dense-1024-none-explicit.json","800105b53c4874b7938d9c908a32f98ad46cdd7e6e73cc819bbcf14ca694adf7","50da0bd7141b95536ac1cb6c96e0fe06bad5ae4b9dadc4f9f266fe2ab5853703","6b4551aaf3447efdbddff6fcd660ce2e7fbbacc2948dbbce81114add74749d71","abffa451c26d96f2a4ee8014333a8581afde46fb","37849a4ee29c8a1755b9376633fba92c2a38ef2cc7932c5297f2396449ae1a31","d487e33440807c40cb789434ff40e5f3fbfe58831e3c655c384776b0f984f162"
   "engine-device-cuda-native-poisson-32-jacobi-standard.json","b217688346b0b4722697fc52f0d4cdecfea276a98fdffe31741a3e4769953599","28d54e1a4eb7ea704d5645ab2be19632a6466d7749d843e02a9faae610a5531f","75fc67ae89adb7112033a7b3d6a01788e4fe61088461819f305248a4657b4932","abffa451c26d96f2a4ee8014333a8581afde46fb","37849a4ee29c8a1755b9376633fba92c2a38ef2cc7932c5297f2396449ae1a31","d487e33440807c40cb789434ff40e5f3fbfe58831e3c655c384776b0f984f162"
   "engine-host-cuda-native-dense-1024-jacobi-standard.json","8a239ea2b1331bec1be67fab3ce030ccab5f34408b9e6a7e38fe11c66ff8ac99","6826bbe02dad56612ca8a633c4b0877eab6227d77b2ba0860470f7864f68868d","0e258920d75ab6f1327b0f4df404c2e983cb1a993f55269280b704cb6d06328d","abffa451c26d96f2a4ee8014333a8581afde46fb","37849a4ee29c8a1755b9376633fba92c2a38ef2cc7932c5297f2396449ae1a31","d487e33440807c40cb789434ff40e5f3fbfe58831e3c655c384776b0f984f162"

The CPU diagnostic links the two preserved production archives through
the same public CpuBackend methods, identical GNU 16 / -O3 -DNDEBUG C++20
flags and identical probe source. It uses the actual dense-1024/2048
cgbench fixtures, seed 20261015, one thread and fused=True. Each operation
has one precheck, three warmups and thirty timed calls in each process;
two fresh processes per side follow ABBA. Allocations, fixture file I/O,
finite scans, independent scalar long-double reference arithmetic and
before/after propagated FP64 rounding-bound gates are outside timing.
All eight process artifacts satisfy the recorded vector/scalar gates.

The center below is the median of two process medians with their min/max
range; no bootstrap interval or strict solver verdict is inferred.
Inputs are fixed x=p=b: apply computes A*b, apply_and_dot computes
b^T*A*b, and residual_squared_norm computes ||b-A*b||^2. These are not
evolving CG directions or independent samples of the full solver.
Do not sum these medians, subtract API/prepared scores, or use a fixed-p
ratio to account for a percentage of the measured whole-CG improvement.

.. csv-table:: Fixed-input public CPU operation diagnostics; milliseconds
   :class: benchmark-results
   :header: "case","operation","A ms [process range]","B ms [process range]","B/A descriptive","max vector error/bound","max scalar error/bound"

   "dense-1024","apply","0.1267 [0.1262–0.1272]","0.1157 [0.0953–0.1362]","0.9136","0.000740437","0"
   "dense-1024","apply_and_dot","0.1336 [0.1306–0.1365]","0.1153 [0.0944–0.1362]","0.8630","0.000740437","0.000706711"
   "dense-1024","residual_squared_norm","0.1358 [0.1282–0.1434]","0.1258 [0.1152–0.1365]","0.9268","0.000813839","1.05075e-05"
   "dense-2048","apply","1.1198 [1.1178–1.1218]","0.8608 [0.8536–0.8681]","0.7687","0.00105014","0"
   "dense-2048","apply_and_dot","1.0675 [1.0555–1.0794]","0.8565 [0.8492–0.8638]","0.8024","0.00105014","0.000403557"
   "dense-2048","residual_squared_norm","1.0817 [1.0672–1.0963]","0.8544 [0.8497–0.8591]","0.7899","0.00108405","0.00039402"

Diagnostic manifest SHA256: ``26906357ce437c216c3aa297e6ff56d49653c35a4f12cd43ecd511c4450773b8``;
archive-snapshot manifest SHA256: ``d059000ce4ad6ce05a1826b57858102a163c7b4f9447da5bfbdb39830297d6a6``;
probe source SHA256: ``018b5a18be29cb89a1febc9cc343da56e9fa5225931d4a35b93b90eec2ea3821``.
Compiler: ``c++ (Ubuntu 16-20260315-1ubuntu1~24~ppa1) 16.0.1 20260315 (experimental) [trunk r16-8100-g3aca3bae8ee]``.

.. csv-table:: Preserved diagnostic binaries, archive snapshots and exact commands
   :class: benchmark-results
   :header: "side","probe executable","executable SHA256","archive snapshot","archive SHA256","compile command"

   "baseline","build/performance-optimization/cpu/profile-dense-baseline","5c65deef8d804592f65c3011908f748c17d145cdebbe61396fb027596279d428","build/performance-optimization/cpu/archives/baseline.a","190fac6b8e35ab82104b8fb425eaf5cf6060636fe2222f7a58b58630483c3460","/usr/bin/c++ -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -Icpp /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/performance-optimization/cpu/profile_dense.cpp /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/cpu-release/libcg_core.a -o /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/performance-optimization/cpu/profile-dense-baseline"
   "candidate","build/performance-optimization/cpu/profile-dense-candidate","457658cacc52fb76cefe94bd374a9671078dbc5a3a9cd33c45a626fb4d258dff","build/performance-optimization/cpu/archives/candidate.a","20bc5a202c4860041479d79c6510e8a7815b84400fd683ecf3cc96decc7d8602","/usr/bin/c++ -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -Icpp /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/performance-optimization/cpu/profile_dense.cpp /home/threemonth03/.codex/worktrees/cpu-dense-optimization/Conjugate-Gradient-Method-library/build/cpu-release/libcg_core.a -o /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/performance-optimization/cpu/profile-dense-candidate"

.. csv-table:: All CPU diagnostic artifacts and formal input hashes
   :class: benchmark-results
   :header: "artifact","side","SHA256","input SHA256"

   "diagnostic-dense-1024-1-baseline.json","baseline","1e7673bf397f8e5b970b7accb85244b4f9a14b74a4f3b305e69af8d8a67efcf7","58a504a95c04274e80597b08bb8363241cf0c16ddbb5060b0ec3758065181329"
   "diagnostic-dense-1024-2-candidate.json","candidate","ca61ad5aff01d30f05a25a8ca8e3087dbf7bcb612849d7e76fe45c46c89601e9","58a504a95c04274e80597b08bb8363241cf0c16ddbb5060b0ec3758065181329"
   "diagnostic-dense-1024-3-candidate.json","candidate","3373e2749aa5b7839839c33c4227304edfce5083260b36568f7039ea26b63358","58a504a95c04274e80597b08bb8363241cf0c16ddbb5060b0ec3758065181329"
   "diagnostic-dense-1024-4-baseline.json","baseline","c2f79080a8ac3aa84a56ddbf1129866464f37e3e1718240e1c567102166f954c","58a504a95c04274e80597b08bb8363241cf0c16ddbb5060b0ec3758065181329"
   "diagnostic-dense-2048-1-baseline.json","baseline","cbd12c0a72ace304f1405a422d99fa57e84466faa905ac11020ea8a69d97ee83","074f888f022f56e8ef847862a4554a716ce5571c6da3859a06568e2657f22bc9"
   "diagnostic-dense-2048-2-candidate.json","candidate","b54cfd4a010ca999c8407e680ccf5cfb324fe693d91ba61c370defc20381cfeb","074f888f022f56e8ef847862a4554a716ce5571c6da3859a06568e2657f22bc9"
   "diagnostic-dense-2048-3-candidate.json","candidate","08bf52f3356f886cc4dff7d61f7d507d5b0545dc41117d9e87d77467feec3cf6","074f888f022f56e8ef847862a4554a716ce5571c6da3859a06568e2657f22bc9"
   "diagnostic-dense-2048-4-baseline.json","baseline","8ce80e9f5b8d9084f4f89b1bd770fb1e1ae759b6c9139cb0845c87b46c71174e","074f888f022f56e8ef847862a4554a716ce5571c6da3859a06568e2657f22bc9"

Inspection of the preserved portable Release objects already finds
SSE2 mulpd/addpd in the baseline; the starting kernel was not scalar-only.
The two-row candidate's bulk loop loads each x pair once, reuses it for
both matrix rows, and maintains four independent packed accumulation
chains. Four matrix loads remain. The inspected hot column loop has no
FP-accumulator spill; ordinary prologue/argument stack traffic and the
out-of-line pair-helper call remain. This confirms the generated mechanism,
not a DRAM-bandwidth bottleneck or a solver speedup. The object uses no
AVX/FMA instructions or architecture/fast-math flags.

Recorded evidence from ``cpu/codegen-review.txt``:

.. code-block:: text

   130:    66 0f 10 1c 01          movupd xmm3,XMMWORD PTR [rcx+rax*1]
   13f:    66 0f 10 4c 01 10       movupd xmm1,XMMWORD PTR [rcx+rax*1+0x10]
   145:    66 0f 59 c3             mulpd  xmm0,xmm3
   154:    66 0f 59 c1             mulpd  xmm0,xmm1
   161:    66 0f 59 c3             mulpd  xmm0,xmm3
   173:    66 0f 59 c1             mulpd  xmm0,xmm1

.. csv-table:: Inspected source/object identities; compiler commands are recorded in the codegen log
   :class: benchmark-results
   :header: "side","source SHA256","object SHA256"

   "baseline","41953efc6109753785defaf3e95a3cddc6467edca0b8cd82875306d6f7d966e0","e4fd8683420601c6d2baa0d89289d86e8ddaa07d1e02486f367b86cf84565314"
   "candidate","661a4f4a4164107d98dc739f8581f4e1f58851a3935d0e4fc546ad0278a8b906","c831ed8072ff1122edbfac4d21cebb30aa33423f6b0988053ea0df34b6f4cbb5"

Codegen-review log SHA256: ``bd316fad03ea6ca93bfec12ff7415bcacaa7332d9e4f6e3cec41a5252b38f574``.
These diagnostic evidence files remain local ignored artifacts.

Retained CUDA campaign: cuda-api-abba
-------------------------------------

Local manifest ``build/performance-optimization/cuda-api-abba/manifest.json``, SHA256
``39cfdadfca4b409d6806a78b52d5fd759acb72746fdc0a16b8d49c8fdc3a40e3``, stopped with 2 retained raw JSON/log pairs.
The original audit compared CMake cache declaration suffix
``UNINITIALIZED`` versus ``STRING`` despite the same compiler path.
The replacement frozen helper compares the key before ``:`` while still
requiring exact compiler paths, flags and runtime controls. Neither of
these two original files belongs to a completed four-worker ABBA cohort;
they remain retained outside the reported medians/intervals.

These metadata-audit stops and restarted campaigns are disclosed:
there is **no overall retry-free/no-discard campaign claim**. No completed
cohort was remeasured to select favorable timing samples.

.. csv-table:: All retained artifacts from this stopped campaign
   :class: benchmark-results
   :header: "artifact","audit status","cohort use","SHA256","failure"

   "cuda-release-cuda-dense-f64-v1-none-prepared-b01-baseline","measured","excluded incomplete ABBA","ae1a536fd0108976daa23024d4dc43569a4ae6c11e95f8af17af8796ef97cb13",""
   "cuda-release-cuda-dense-f64-v1-none-prepared-b02-current","failed","excluded incomplete ABBA","9bef771408a4a7d6911b0659caae82b25778fe1dd70bb26237f100296f981cb6","ValueError: Actual hardware/compiler/flags/runtime/thread environment changed across A/B"

Retained CUDA campaign: cuda-api-abba-cachetypes
------------------------------------------------

Local manifest ``build/performance-optimization/cuda-api-abba-cachetypes/manifest.json``, SHA256
``18457e203ff7656f21b4bd5f3b113a9fe165ea5db11a958b42fe9ce42eaf2dc2``, stopped with 17 retained raw JSON/log pairs.
This next audit stopped at the first CSR artifact because importing
SciPy adds its one-thread library pool, unlike the dense configurations.
Its 16 completed dense artifacts contain four full ABBA cohorts and are
retained in the composite results. The single incomplete CSR baseline is
not pooled; a separately preplanned CSR-only campaign supplies four
complete ABBA cohorts. Complete library-pool metadata remains exact
within configurations and host/device mode pairs; other environment
controls remain exact globally.

These metadata-audit stops and restarted campaigns are disclosed:
there is **no overall retry-free/no-discard campaign claim**. No completed
cohort was remeasured to select favorable timing samples.

.. csv-table:: All retained artifacts from this stopped campaign
   :class: benchmark-results
   :header: "artifact","audit status","cohort use","SHA256","failure"

   "cuda-release-cuda-dense-f64-v1-none-prepared-b01-baseline","measured","selected complete dense ABBA","fbe279fe11d992cd2f0b9eb376ef2edd3824b348cad1eb1c1eaa3d3f375bebb0",""
   "cuda-release-cuda-dense-f64-v1-none-prepared-b02-current","measured","selected complete dense ABBA","5ccd410ece6c7f14e7dd90c6e65b5db70d0493a6c6f127d2bed28f6ec7532f8e",""
   "cuda-release-cuda-dense-f64-v1-none-prepared-b03-current","measured","selected complete dense ABBA","17d52a6dbe631578b160a6afc25eac71b1312a5b0d5357e2fe4673304906e69e",""
   "cuda-release-cuda-dense-f64-v1-none-prepared-b04-baseline","measured","selected complete dense ABBA","386a1ec9d8c491ac340a8c776286824bad8ae72efdb70ead996f5491d28cb326",""
   "cuda-release-cuda-dense-f64-v1-none-api-b01-baseline","measured","selected complete dense ABBA","c80701930f848c20582f8f6f68ac3840c1443f90bb3967e7e03d21782957cbd5",""
   "cuda-release-cuda-dense-f64-v1-none-api-b02-current","measured","selected complete dense ABBA","72eff510d4f9606373d9ec628b2525765d09127db0fca6faa110c8d53df78145",""
   "cuda-release-cuda-dense-f64-v1-none-api-b03-current","measured","selected complete dense ABBA","588b9a6f79542f22147ae5e737406af983815d1c9cf6ccb4dc7c5a763c23aaa4",""
   "cuda-release-cuda-dense-f64-v1-none-api-b04-baseline","measured","selected complete dense ABBA","ac14f6fceb8ef3f66000701fa157576bc6ee87946d7cb6cf8eac6ff328b42f44",""
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared-b01-baseline","measured","selected complete dense ABBA","443700183d9a6f17f3a731a656808759a77e0ee061828bf0b84c3d8cafcc2f08",""
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared-b02-current","measured","selected complete dense ABBA","c4073e27d4f13257aaa3faa0c00637e40654c2137075fabbc87442bca6c4fbb7",""
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared-b03-current","measured","selected complete dense ABBA","61a81617593b6f0a82483c807591c4e0c0c119471612ca96e29bbcd2bb6f0063",""
   "cuda-release-cuda-dense-f64-v1-jacobi-prepared-b04-baseline","measured","selected complete dense ABBA","10a076a68d3511c52bb37949fbc1a094cef86f6ab878413137588764b0031003",""
   "cuda-release-cuda-dense-f64-v1-jacobi-api-b01-baseline","measured","selected complete dense ABBA","1f54e566f08027c813c6bad3a967722570f8ee33b730bab0423a0bec32ca3521",""
   "cuda-release-cuda-dense-f64-v1-jacobi-api-b02-current","measured","selected complete dense ABBA","30227826cdbf3746649cef7f1eb96f83820afb770195e0d4dfe1a19c8308d236",""
   "cuda-release-cuda-dense-f64-v1-jacobi-api-b03-current","measured","selected complete dense ABBA","81564bf4a6d2adfa10c784af35ebeb8409aadabc269dd7e7862444a7c387623c",""
   "cuda-release-cuda-dense-f64-v1-jacobi-api-b04-baseline","measured","selected complete dense ABBA","f2dfdf3708b6d25cf0efdc3f306b36887be90d7c01486bc557deae04d3e0b636",""
   "cuda-release-cuda-csr-f64-v1-none-prepared-b01-baseline","failed","excluded incomplete ABBA","9cc26cf1bc4e99289ca7db2c0d6ad1f5eee38ab64a656c6b8354f10a102d8fa8","ValueError: Actual hardware/compiler/flags/runtime/thread environment changed across A/B"

Source, binary and evidence identities
--------------------------------------

The compiled revision belongs to each frozen source/binary stage. The
collector's live checkout revision is recorded separately and is never used
to relabel the compiled extension. Stage identity SHA256 covers the declared
payload manifest; active extension hashes below identify what workers loaded.
Raw JSON, logs, manifests, stages and this renderer remain local ignored
artifacts under ``build/performance-optimization/``; these paths are not
public raw-data downloads and cannot be used for replay from a clean clone.

The measured candidate sources are publicly retained by exact tags; later
rebased or merged heads do not replace their compiled identities. Source
equivalence of selected GPU/Python files does not imply identical complete
source trees after CPU/base changes.

* ``benchmarks/cpu-dense-20261016`` → ``cd1f292cd404241647a32cb10deaa6ae187c3e8c``.
* ``benchmarks/cuda-api-20261016`` → ``0ade08d30ccdf7d7817cb34338bb73510b52cb2d``.
* ``benchmarks/cuda-device-20261016`` → ``abffa451c26d96f2a4ee8014333a8581afde46fb``.

* CPU ABBA: ``build/performance-optimization/cpu-abba/manifest.json``, SHA256
  ``12b036695c171b640630276f9ed8493e62cc6b00bc0ee7d52497d89f9bd1aa9b``; 32 artifacts /
  384 eligible candidate/case results.
* CUDA API ABBA: ``build/performance-optimization/cuda-api-abba-complete/manifest.json``, SHA256
  ``53af02da6a703670b3681edfa17deb164f5db220812ada1a768494564c12d156``; 32 artifacts /
  320 eligible candidate/case results.
* CUDA modes: ``build/performance-optimization/cuda-device-modes/manifest.json``, SHA256
  ``597b402fb5352f1c6479d1745f3bee61a1ac8e9159a305841cd7f950f628e52b``; 48 artifacts /
  480 eligible candidate/case results.
* CUDA confirmation: ``build/performance-optimization/cuda-api-confirmation/manifest.json``, SHA256
  ``af329cb54bc05643ef63637fe3791c10e19aaba9bfa95c3eee8eaa053fb4af21``; 8 artifacts /
  96 eligible candidate/case results.

.. csv-table:: Frozen compiled stages and active modules
   :class: benchmark-results
   :header: "batch","side","compiled source","stage identity SHA256","active module SHA256"

   "CPU ABBA","baseline","b96e455972b937641cec52fc7fc86e75a058f07c","641c23ef0781d8c419b32dd1c3f00e3c38ba5571a70cc4c27982c59981bb0546","_core fd4e3a62e057a70a18ad8dd60656d5fefb51bee559243171d0d84de45516746a"
   "CPU ABBA","current","cd1f292cd404241647a32cb10deaa6ae187c3e8c","408ba1ab032d484acf9450d3cd6d3e5c43cff06d0f94605d12d163994b88fea6","_core 3111f94040afdea2519c5e61023336b8a7ef869c77bf84d7d6c4974eee8cbbff"
   "CUDA API ABBA","baseline","b96e455972b937641cec52fc7fc86e75a058f07c","5bc918dfc424b36d994a383e2736de58942fb68795b092c530cd07d87d116364","_core 7e68ab3075f92801e2c2d53e58405f1cccf4f1dc8d2e3147b65c7159b2ecf853; _cuda 419b4718d7eed0b9544805fe8f89bf4675f29f000794a259a714d4198450454c"
   "CUDA API ABBA","current","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","edb120ea47f8af1e9ee6f91b9b0eb780199fac5c39994e8251a277b808cebeb1","_core 7e68ab3075f92801e2c2d53e58405f1cccf4f1dc8d2e3147b65c7159b2ecf853; _cuda 098d81cbbc4df934b30bf74024059de171d39f5f7ab3a77bb805a29c9e433f26"
   "CUDA modes","candidate","abffa451c26d96f2a4ee8014333a8581afde46fb","37849a4ee29c8a1755b9376633fba92c2a38ef2cc7932c5297f2396449ae1a31","_core 7e68ab3075f92801e2c2d53e58405f1cccf4f1dc8d2e3147b65c7159b2ecf853; _cuda d487e33440807c40cb789434ff40e5f3fbfe58831e3c655c384776b0f984f162"
   "CUDA confirmation","baseline","b96e455972b937641cec52fc7fc86e75a058f07c","5bc918dfc424b36d994a383e2736de58942fb68795b092c530cd07d87d116364","_core 7e68ab3075f92801e2c2d53e58405f1cccf4f1dc8d2e3147b65c7159b2ecf853; _cuda 419b4718d7eed0b9544805fe8f89bf4675f29f000794a259a714d4198450454c"
   "CUDA confirmation","current","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","edb120ea47f8af1e9ee6f91b9b0eb780199fac5c39994e8251a277b808cebeb1","_core 7e68ab3075f92801e2c2d53e58405f1cccf4f1dc8d2e3147b65c7159b2ecf853; _cuda 098d81cbbc4df934b30bf74024059de171d39f5f7ab3a77bb805a29c9e433f26"

.. csv-table:: Frozen runner, worker and comparison helper identities
   :class: benchmark-results
   :header: "batch","helper","SHA256"

   "CPU ABBA","build/performance-optimization/ab_runner.py","54668d5459fec4041f125d21d38a6fc73074c5b7dedc539a765a8bde254c7ace"
   "CPU ABBA","build/performance-optimization/worker.py","48d7a1e7f4cd36cb39b17e8f509450dd86e5b403edfd94afd03352d095f22571"
   "CPU ABBA","build/performance-optimization/run_baseline.py","213afaa21053d4f65dea506158dc65d60cc4587033bc388d0efccb362e76233f"
   "CPU ABBA","scripts/compare_performance.py","956d0fc65dc463bb2edfdc800da2dbcbfe3049bc4ae6996844fc0c0c804f6124"
   "CUDA API ABBA","build/performance-optimization/assemble_cuda_abba.py","d684bc77c59edad562c7b142cdaec6a0e8d6035aa2a52e42b2922ae0423fd68b"
   "CUDA API ABBA","build/performance-optimization/ab_runner_csr.py","d5f244785cdf1da014f3ca09a0c97de6d50a3f0a5abd84ec785346e60e314c0c"
   "CUDA API ABBA","scripts/compare_performance.py","956d0fc65dc463bb2edfdc800da2dbcbfe3049bc4ae6996844fc0c0c804f6124"
   "CUDA modes","build/performance-optimization/run_device_modes.py","84e56fc5b688fe6fa91e752dc3bf069cc304d3f771460db3e5bdf4f1d9a7481b"
   "CUDA modes","build/performance-optimization/worker.py","48d7a1e7f4cd36cb39b17e8f509450dd86e5b403edfd94afd03352d095f22571"
   "CUDA modes","build/performance-optimization/run_baseline.py","213afaa21053d4f65dea506158dc65d60cc4587033bc388d0efccb362e76233f"
   "CUDA modes","build/performance-optimization/ab_runner.py","54668d5459fec4041f125d21d38a6fc73074c5b7dedc539a765a8bde254c7ace"
   "CUDA modes","scripts/compare_performance.py","956d0fc65dc463bb2edfdc800da2dbcbfe3049bc4ae6996844fc0c0c804f6124"
   "CUDA confirmation","build/performance-optimization/ab_runner_confirmation.py","c3bde3d70c96cde0a3f407f0e340520a5ec7a2999375ca8a1c1fd4040662a3e4"
   "CUDA confirmation","build/performance-optimization/worker.py","48d7a1e7f4cd36cb39b17e8f509450dd86e5b403edfd94afd03352d095f22571"
   "CUDA confirmation","build/performance-optimization/run_baseline.py","213afaa21053d4f65dea506158dc65d60cc4587033bc388d0efccb362e76233f"
   "CUDA confirmation","scripts/compare_performance.py","956d0fc65dc463bb2edfdc800da2dbcbfe3049bc4ae6996844fc0c0c804f6124"

.. csv-table:: Full fixture specifications and shared input hashes
   :class: benchmark-results
   :header: "case specification","input SHA256"

   "{""condition_scale"": 100.0, ""diagonal_scale"": 20.0, ""dimensions"": 3, ""grid"": 10, ""name"": ""scaled-poisson-3d-10"", ""size"": 1000, ""storage"": ""csr""}","9ec41e8562ee6a9fd198f544bd57d73e4ab336d5fbfc0e325ac1a3af7dd94a14"
   "{""condition_scale"": 100.0, ""diagonal_scale"": 20.0, ""dimensions"": 3, ""grid"": 6, ""name"": ""scaled-poisson-3d-6"", ""size"": 216, ""storage"": ""csr""}","4e02a84a1bf5d5ca9405bb22428d7c4b17b9ae9a3e6cfca2b273dc8be2fd8b79"
   "{""condition_scale"": 100.0, ""diagonal_scale"": 20.0, ""grid"": 16, ""name"": ""scaled-poisson-2d-16"", ""size"": 256, ""storage"": ""csr""}","2c63599a62a84ed1a3a927453cf93e257911915078adde48692665ac04946bed"
   "{""condition_scale"": 100.0, ""diagonal_scale"": 20.0, ""grid"": 32, ""name"": ""scaled-poisson-2d-32"", ""size"": 1024, ""storage"": ""csr""}","3781fb80b92c567cb96f05581efc9496448112ddefab34d592a5a0e5a89f17a5"
   "{""condition_scale"": 100.0, ""grid"": 128, ""name"": ""poisson-128"", ""size"": 16384, ""storage"": ""csr""}","366658110507894737aae729348095b3103c569c950fba3bf3c69a3f05205149"
   "{""condition_scale"": 100.0, ""grid"": 16, ""name"": ""poisson-2d-16-neutral"", ""size"": 256, ""storage"": ""csr""}","97964de69d5bd0a4d355aca03f133609c86f1dd48213a14ab1bd558d305b4ff7"
   "{""condition_scale"": 100.0, ""grid"": 32, ""name"": ""poisson-32"", ""size"": 1024, ""storage"": ""csr""}","2be39617e771847e425d64a8d997d0eea8942b930852010d2ac2c83bb3e26ab7"
   "{""condition_scale"": 100.0, ""grid"": 64, ""name"": ""poisson-64"", ""size"": 4096, ""storage"": ""csr""}","366644870e9d921500fcf8ec3dc21b6daa83afdc9d3482e3846feb816df2483d"
   "{""condition_scale"": 100.0, ""grid"": 8, ""name"": ""poisson-8"", ""size"": 64, ""storage"": ""csr""}","c42223d3f75a5c3b4637a56319aec5e0c74ac30f9e81647f0008ae311c362c71"
   "{""condition_scale"": 100.0, ""name"": ""dense-1024"", ""size"": 1024}","58a504a95c04274e80597b08bb8363241cf0c16ddbb5060b0ec3758065181329"
   "{""condition_scale"": 100.0, ""name"": ""dense-2048"", ""size"": 2048}","074f888f022f56e8ef847862a4554a716ce5571c6da3859a06568e2657f22bc9"
   "{""condition_scale"": 100.0, ""name"": ""dense-256"", ""size"": 256}","96f633f1697698d7e55c0e06b38ab801460f1ba1aa3909deaae939aa1d2e6274"
   "{""condition_scale"": 100.0, ""name"": ""dense-64"", ""size"": 64}","e8985fd6c5ed371b92931d8ff1bbddf1a718eeeebd399f5ca2a62a45255e4cb6"

.. csv-table:: Complete raw artifact inventory; no numerical sample removed
   :class: benchmark-results
   :header: "batch","artifact","SHA256","bytes","compiled revision","collector live revision"

   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-b01-baseline","4d953940570b5ce79e849db2d9695516a42f047781c1a2a8728df30c6792ad26","41546","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-b02-current","d7814dc57a8f67ac6338e05ed34539af174646bd47a74aa7e301565ae1819afd","41542","cd1f292cd404241647a32cb10deaa6ae187c3e8c","cd1f292cd404241647a32cb10deaa6ae187c3e8c"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-b03-current","3b0c701a1bb1479cb15dff7dabc1b4a1e3601c998f8a0cc07b484cef2ef1d384","41572","cd1f292cd404241647a32cb10deaa6ae187c3e8c","cd1f292cd404241647a32cb10deaa6ae187c3e8c"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-b04-baseline","f41cb617a6ae456e9cfd3a594f582bdf25147a48cb951675e48afe176c2e0025","41538","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-b01-baseline","7dc60c418654fe7107e78a9c7297f93a8940144615b8fdac7a5b3f5a45418c44","41348","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-b02-current","042624eb7191e5022c7cae4a66cd469b0fff702aa6ddec87b3870d95e5e16670","41329","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-b03-current","99af435753f13e7c0ac36370f383fc24e27a2f8a57f830143bd9bf45420d3400","41326","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-b04-baseline","696a16750b2cccfdfc025dc6ae3cd365fc7219689f62414849508acb5eab0f3c","41348","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-b01-baseline","63b26b76c966b2bb68c8a04a4b09d932a026c6221a3ea27ea1021a7f203b2ca4","41340","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-b02-current","30390c0958217548778277ed4f5fcb3419c644e71a3494cad8381aab2662252c","41345","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-b03-current","bdc6879250f7bc61620b7a0f465219464db23298feb60233022502e6b2335e37","41345","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-b04-baseline","35f62590c46cb34de766a21008aa407f5436dd8da3ea5c1e01e835adc4759848","41329","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-b01-baseline","f3b751f1492510b9fa88f945cd7901b31002dd533af307d8af0a8a9570d7b1cd","41143","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-b02-current","69f36b1f2af1c2f757d8c5d7bd24421cfcde435bcd1157467f2e178a471504ff","41146","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-b03-current","918dcda190e8a434cef2bbdccc8b2306bc94312f9ab54ba9b217c77b0a2d889a","41154","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-b04-baseline","803f859ae7c8214bdf7c14d8ead5e4cd5812b165195330bfd0bb417fd39e0b54","41139","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-unfused-b01-baseline","232b343a42885fcdbab52f03d82839aa0b7cf8748a77af6896717816543f538f","41558","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-unfused-b02-current","8f97790fd5bbd0fa070c8211da84b826ab2f0339edcf3c2eed4d413abf9d6d56","41567","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-unfused-b03-current","f06d75995bc7dba54dd5491ce7bf601010f2b39ad8608d9119cfced28805a362","41586","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-prepared-unfused-b04-baseline","eab491de0922bc43a340d1fb26bc1f92624156fe70d84f046b9684a90326a291","41568","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-unfused-b01-baseline","2306b00c5d4346a503744c9471e3f5f2b6e9d7fc6169fea42c7300e5656c17f7","41357","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-unfused-b02-current","46ac6860a7b8984037ae9110c27bb8627a0b93ff3f0b9d749c0d7edd5ef69d07","41353","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-unfused-b03-current","2ee0c697d828665040c80bfd890699f8e0e812aa888932913117742d11070d28","41351","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-none-api-unfused-b04-baseline","6fba431381178306ab14db3567ad49d11f95e4e74f33fcb4d6a70535ae126879","41346","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused-b01-baseline","5edfc807a4393b84645dab46d732c376d3b29357a63890f6dfc38b8fe9409e7e","41337","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused-b02-current","9e5dfbf034a9c9a9d8730b4a0d8844b863eebbd8595a733a004e76bea0e5fabc","41340","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused-b03-current","fd0f6e153d2ff2ef80c987894b5832a953f94707507dd9a1de835d0063792549","41342","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused-b04-baseline","af4bf57a6c395fe05a521d7aff7295fcafd43ad345f94aec41562b78898e5f92","41336","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-unfused-b01-baseline","e36044bcf72462ee34c448240f83366501d9c54b99a8d3e127572ca6e8e2da03","41151","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-unfused-b02-current","01fb3b597f83a603de6fb949d44bb8cef65582d3cab8f7406c2f7924c7d8ce21","41158","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-unfused-b03-current","9d590a7abc74da0a37fddda4fc57c842782b622c228e570395a51ff28b46a448","41152","cd1f292cd404241647a32cb10deaa6ae187c3e8c","ae5449a128c5b14ce4ec7021bbfa29ef352bf521"
   "CPU ABBA","cpu-release-cpu-dense-f64-v1-jacobi-api-unfused-b04-baseline","4d9feb8cc692c41d920c4879de40423e7737691af40a8714e8a7cbf710cddbc9","41149","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-prepared-b01-baseline","fbe279fe11d992cd2f0b9eb376ef2edd3824b348cad1eb1c1eaa3d3f375bebb0","55228","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-prepared-b02-current","5ccd410ece6c7f14e7dd90c6e65b5db70d0493a6c6f127d2bed28f6ec7532f8e","55219","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-prepared-b03-current","17d52a6dbe631578b160a6afc25eac71b1312a5b0d5357e2fe4673304906e69e","55219","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-prepared-b04-baseline","386a1ec9d8c491ac340a8c776286824bad8ae72efdb70ead996f5491d28cb326","55229","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-api-b01-baseline","c80701930f848c20582f8f6f68ac3840c1443f90bb3967e7e03d21782957cbd5","55199","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-api-b02-current","72eff510d4f9606373d9ec628b2525765d09127db0fca6faa110c8d53df78145","55190","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-api-b03-current","588b9a6f79542f22147ae5e737406af983815d1c9cf6ccb4dc7c5a763c23aaa4","55190","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-none-api-b04-baseline","ac14f6fceb8ef3f66000701fa157576bc6ee87946d7cb6cf8eac6ff328b42f44","55199","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-prepared-b01-baseline","443700183d9a6f17f3a731a656808759a77e0ee061828bf0b84c3d8cafcc2f08","54984","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-prepared-b02-current","c4073e27d4f13257aaa3faa0c00637e40654c2137075fabbc87442bca6c4fbb7","54972","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-prepared-b03-current","61a81617593b6f0a82483c807591c4e0c0c119471612ca96e29bbcd2bb6f0063","54972","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-prepared-b04-baseline","10a076a68d3511c52bb37949fbc1a094cef86f6ab878413137588764b0031003","54983","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-api-b01-baseline","1f54e566f08027c813c6bad3a967722570f8ee33b730bab0423a0bec32ca3521","54961","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-api-b02-current","30227826cdbf3746649cef7f1eb96f83820afb770195e0d4dfe1a19c8308d236","54949","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-api-b03-current","81564bf4a6d2adfa10c784af35ebeb8409aadabc269dd7e7862444a7c387623c","54951","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-dense-f64-v1-jacobi-api-b04-baseline","f2dfdf3708b6d25cf0efdc3f306b36887be90d7c01486bc557deae04d3e0b636","54961","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-prepared-b01-baseline","692406810c131ae2a256961d5c2235b301505fba973c312af2236a440208c9e8","43634","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-prepared-b02-current","a906bb452dbc0f51b2a41a1be96ca19922a936010c4784d293669d5fa2047dad","43624","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-prepared-b03-current","40c934c27c4a57d9baa210f19e6794a4a87a39a47e99624dfd539160b246accc","43623","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-prepared-b04-baseline","0c4cffb8b9108ed78896b9b8302826bf7da25055a971c5650b76a06fc835687a","43634","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-api-b01-baseline","a6feaf2080309a292c3c309b82cbf7172f4d406b47c21ec482f68cce30a9bfe5","43614","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-api-b02-current","1c5824f91775b359877b7028c4876e82de5aaf70ee321e35c8e0f773b65a580d","43606","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-api-b03-current","aacd54a87719efcd50c4a1a35c8cb31098d19053c50d85a52f5558294b7c0778","43607","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-none-api-b04-baseline","721cc7b6c226bfc3eb8e16aa20d946999258ea4ada28f121dac20795b075a072","43614","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-prepared-b01-baseline","815b6bad95d9bb5582bbd94de290509da3f2782baec12ba4b9edfc0775244bc6","43764","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-prepared-b02-current","fab1f41368f781a4d7b6058a39750019353630e0ae597de21c9f002082d79865","43748","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-prepared-b03-current","ea6c73311a1523d922772fdb0f0c6ea739ee1be61155417ea5ab061118f0d340","43746","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-prepared-b04-baseline","f117e0b88285bda61b5e1ec4355b8945ad32415c43ed2f9fddc920777d43a0ad","43756","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-api-b01-baseline","054dbd9127a27551b36d557e9e08c48105f027e66eac5a0f4674c38272c5b13a","43744","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-api-b02-current","ff6ab76545a0233e25459d979ec868bd8e197b57d7baf9c0a7fc39f5b486829d","43737","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-api-b03-current","e78a643eab67d5caa92156fac14200e9a38098ac88402bab007ade865926d62e","43736","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA API ABBA","cuda-release-cuda-csr-f64-v1-jacobi-api-b04-baseline","88f5e1dbd6ede4e58ad82e2c5896e15a3e397ea77f960cd34ecfcbe4e878b648","43743","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-prepared-host-r01","7ff8f7faf9ec5476039f5367c50a6e71006b78aac2ec5e6537bac602983a4f30","55929","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-prepared-device-r01","ebb6cb2a9c81b7c2f757dec7be75cde57209557956d7dc5f54fb447add277bea","56348","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-api-host-r01","5964c4b6cb2c5703afba02f4c6ea0c6a0a52a9657451f0fa382edca3248429b0","55901","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-api-device-r01","8e5ac380b8bf33ae9de0cb8e863a7418194559995d09376617dce28f19e313f6","56318","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-prepared-host-r01","431c52d93556d26efa44a9fbf59c91ff4f66e96bf52a726c2d81dbb81d1664a0","55682","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-prepared-device-r01","92a249419cea07d72d7eb1db90a4bd5e1cce7dc0aa364d5711006cba27726cad","56100","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-api-host-r01","0cadce59496f28b4e641ab32e59792c22795809c7d08d5a8ff93eba30ecfe99e","55662","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-api-device-r01","452a3600a7744f3d2798695db59a3d83fcb53582881e0fb46e84337d99e75bbc","56080","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-prepared-host-r01","851e93a83b8ceb8a8cb698ee3e5cb96082e07405ca886786af12cbf62d7477c1","44183","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-prepared-device-r01","6fdbf61d5e5d39b47b7354a2f0c2d716cb24ea9aec9fc4e82ee5201c019ea0a1","44412","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-api-host-r01","76bceeceff8993d50547e517d6e3f95846bac2354e17a4ce5fba4edd6b3b8085","44162","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-api-device-r01","633bad627d068c4d6eb5910ae3b6869bbbe3d858c992d6c4b20c185d16f60e2f","44393","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-prepared-host-r01","284ff70d4b4e21d27e8cb2cd9001d85d5ef13b900ba92cf75dae1d096b327aab","44309","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-prepared-device-r01","dd6e597eb61c1c1d6ca081fa90671e97043fd9d6f7b1f8448016035efe59bf5f","44537","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-api-host-r01","06e748ad2a07b7f861ae4acfe69a36ae30d1115a11eebece5060c7ad70fcca20","44295","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-api-device-r01","3724756048d99d1e8dc8100eade8328c8b1ae997f5bb61dcd6ed195c02ab0b92","44522","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-prepared-host-r01","30f25cc0a486fb5fb27c16f23ae02f57982bfcbeb004f17bc67f81bc1f77b5e4","52843","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-prepared-device-r01","178a5c5aaf3d902494436585303fb4a19d99ee4655b64bd906fd803c73222076","53090","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-api-host-r01","c7a2d3ad3ae14fd47c2ce6c42edb010ea2217cd11d45ca4c95de7f2003500abc","52799","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-api-device-r01","2b8a86686ab47cf8c9d82963573c80ecf72c13245853b970d0980213f7db9d21","53073","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-prepared-host-r01","fe7c621591aa5993981f6929aac9c8ef109b798de3aa345b4bbb9920dd1bc91c","52702","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-prepared-device-r01","698bcdfb67912d34fb41d912f8c5f896715fbd1ac64d40e61386e038b6802dc3","52986","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-api-host-r01","97764f18714985821c22da30b38c6a7820bcd4f31ebad2bdfaa3c59d1f361c61","52691","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-api-device-r01","cee644c6757fb1b0f985ce38628e715ca4f54c533c50a11137b9d3023ad16a37","52978","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-api-device-r02","8b62eecac1879f98b7d93ae5f236d1b4bcd569e757c846ceaea408bbccc870fd","52981","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-api-host-r02","7019be9694d0a8c66c8c311799ae8a495bcada113115f1564b4137144cd3a0a7","52705","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-prepared-device-r02","e48a38140484d46ee10225a7f0da24cc40389d24c3ba740dc79ea2ea88bd6bd0","52984","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-jacobi-prepared-host-r02","8de09dced24ff16f59ace5960f2eb72425d76e0911f170e46bc87d76a96ffef1","52705","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-api-device-r02","47b72ac0cf29db691163a68a3eaad7ba87288b1ac0d156c0836cf39c9dd71451","53077","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-api-host-r02","24504f64759d40e68fdb09e42f1f37a7a9bdd9dd45cf9261fd348fa0033c9a87","52809","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-prepared-device-r02","6802aa2c5a8cc83d5a59a2648df913053180e193b1f790ab0eca58f40c38378f","53090","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-sparse-solvers-none-prepared-host-r02","7faf9fe86225ccad9f5d4bfb3f484b22feded0454795d25362f2e9376a36d7c5","52848","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-api-device-r02","c257e97783e669083e0ab7d171ae16f614581efdc122c7928a11d6cf1e4bccc4","44523","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-api-host-r02","65fdd7bad7d0d00791ba07e1e69391689626c061b2152ad5782cd58b4a0c713d","44335","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-prepared-device-r02","9e4ec5db4867c613bf577f012c95548600188384db0c454db3ccd0700878592d","44580","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-jacobi-prepared-host-r02","089d9ab3411891381a1c967e671866c59e134556a598fb721733c8aff64afc36","44314","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-api-device-r02","1ef474711f95c2781ac26142c7ddf2d02ea12bcf0530bef1e6332d318f835f54","44392","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-api-host-r02","e757715436a4d27806a151f40d2a9916147b61fb772220ffbe5a6abed031a161","44163","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-prepared-device-r02","8c358f8996b305b5a12ec25c6fc9fbd00ac7d3c757b24dbaeef1c33c0e45d613","44400","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-csr-f64-v1-none-prepared-host-r02","84505f65567e97cbd0c446fe5c6ab6f22b61164a12e7697ff7d8184fd9c92ee3","44181","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-api-device-r02","005b751e05fbc30d7c1697e2b17ed3d6ce47e3688d295ca495e642d4e7ebf2f9","56078","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-api-host-r02","5dd4eba388eff7c74c027fe49d01527121f19aeb54dfb5c19c6fb94b1139ef7c","55660","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-prepared-device-r02","f6f094d3c1f86b3a79ea94dd648f64771f2edbb9e5c5fcb2c77e54264bb8ccbb","56100","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-jacobi-prepared-host-r02","5259af75bac099bf80ee9585e61f03f7637a098f79d671edd363d0e6596bd5ed","55681","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-api-device-r02","e043e3498fe5c615b23ff3e01d00e82f6d2e8b4b205b2f93abf5ecdc64a5d9d3","56318","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-api-host-r02","9861aceadb76f842877503196dd9512212a0f4e0b50a069ee09f78d067c931ad","55900","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-prepared-device-r02","5d78dd5d0a172e024278e1c3d695f92615eeb4df4f1f9f5299b4b765b571acfd","56346","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA modes","cuda-release-cuda-dense-f64-v1-none-prepared-host-r02","3a6c8548bd6d16a4969274f7b2fc82cff7e412f109833ecc7c179d1262bf837b","55928","abffa451c26d96f2a4ee8014333a8581afde46fb","7369578026bc69095ea249a8d52896e4dcccb6c0"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b01-baseline","2edb13894a3f1934f008956fc0ea849a2695309b639bb7247f05d236e55b9be5","55230","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b02-current","a18926ffbf787fbe68b65217007e7d3a35c27d2738fc24a291808addbbb62124","55219","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b03-current","0085a48f5fdfcc60507044afb9dd5be443a950024296ab9c724365af585b7082","55221","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b04-baseline","b1d9131f70e852b90f070e93cc5efeb49a02069113c19decb7ca6b75f213343b","55229","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b05-current","017bb391a0f7b6f86c63c3fd7fe1a3e7e6984efb1e27b8fe83379c0ef4f72f69","55218","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b06-baseline","7083966fb37dbeded596072ecd1d33c9d2e3403fce22a9070333c9394cc5f0f3","55230","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b07-baseline","50520b192abbe08d2208026e238200fa53385582cb5a349b3f8314056dcf0a59","55229","b96e455972b937641cec52fc7fc86e75a058f07c","a27e010b7297fbb7e9143c888ea09efff815af94"
   "CUDA confirmation","cuda-release-cuda-dense-f64-v1-none-prepared-b08-current","8111196d9274b5c22b78f11ee5f1f23b37662170b4fa3fb435a764d30159af19","55220","0ade08d30ccdf7d7817cb34338bb73510b52cb2d","ac1ff6e1956c7aec33a7d828a28a29c18a27a57b"

Reproduce the configurations with public commands
-------------------------------------------------

First follow :doc:`/build` for the project-local CPU/CUDA virtual
environments and pinned toolkit. Keep the same package locks, compiler
paths, CUDA 12.8 toolkit, sm75 target and visible physical GPU for both
sources; build and test all sources before quiet measurements. The
commands below use only the public Make/cgbench interfaces. Rebuilding
a revision produces a new binary identity and timing evidence, rather
than reproducing the original binary hashes. The local ignored stages,
runner, manifests and raw replay files are not installed or public APIs.

Use detached source checkouts at the exact compiled revisions in the
identity table above. Keep separate absolute ``BUILD_DIR`` paths for each
source, and use the same root-local environments/toolkit. For example:

.. code-block:: bash

   ROOT=$PWD
   OUT="$ROOT/build/reproduce-optimization-20261016"
   mkdir -p "$OUT"
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
   export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
   export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   unset CUDA_LAUNCH_BLOCKING
   git fetch origin refs/tags/benchmarks/cpu-dense-20261016:refs/tags/benchmarks/cpu-dense-20261016 refs/tags/benchmarks/cuda-api-20261016:refs/tags/benchmarks/cuda-api-20261016 refs/tags/benchmarks/cuda-device-20261016:refs/tags/benchmarks/cuda-device-20261016
   test "$(git rev-parse refs/tags/benchmarks/cpu-dense-20261016)" = cd1f292cd404241647a32cb10deaa6ae187c3e8c
   test "$(git rev-parse refs/tags/benchmarks/cuda-api-20261016)" = 0ade08d30ccdf7d7817cb34338bb73510b52cb2d
   test "$(git rev-parse refs/tags/benchmarks/cuda-device-20261016)" = abffa451c26d96f2a4ee8014333a8581afde46fb
   git worktree add --detach "$OUT/cpu-abba-baseline-source" b96e455972b937641cec52fc7fc86e75a058f07c
   git worktree add --detach "$OUT/cpu-abba-current-source" cd1f292cd404241647a32cb10deaa6ae187c3e8c
   git worktree add --detach "$OUT/cuda-api-abba-baseline-source" b96e455972b937641cec52fc7fc86e75a058f07c
   git worktree add --detach "$OUT/cuda-api-abba-current-source" 0ade08d30ccdf7d7817cb34338bb73510b52cb2d
   git worktree add --detach "$OUT/cuda-modes-candidate-source" abffa451c26d96f2a4ee8014333a8581afde46fb

Build each source once with ``make build`` and the same variables shown
below before starting the ABBA or forward/reverse sequence. Each subsequent
``make pyprof`` launches one fresh numerical Python worker and performs an
up-to-date build check; stop if it would compile during measurement. For
CUDA, the C++ and nvcc host compiler are explicitly GNU 13; CPU uses the
recorded GNU 16 compiler. Each configuration template is listed below;
use a fresh output file for every launch.

For each ABBA configuration, run its template four times with source/build
selector A, B, B, A, sequentially. ``SRC_A``/``BUILD_A`` and
``SRC_B``/``BUILD_B`` point to its baseline/current detached sources and
separate builds above; choose ``SRC``/``BUILD`` for that launch. Set
``RUN=1``, then 2/3/4, so all raw artifacts survive. For explicit modes,
``SRC``/``BUILD`` select the one candidate stage: execute all 24 templates
in listed order for ``RUN=1``, then their exact reverse for ``RUN=2``.
This reproduces 32 CPU + 32 CUDA ABBA + 48 mode workers, excluding the
separately retained stopped CUDA attempts.

CPU ABBA command templates
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-none-prepared-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-none-api-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-jacobi-prepared-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-jacobi-api-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-none-prepared-unfused-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --no-fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-none-api-unfused-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --no-fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-jacobi-prepared-unfused-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --no-fused --implementations native,numpy-out,torch-eager-out'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cpu-release \
       PYTHON="$ROOT/.venv/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
       PYPROF_SUITE=cpu-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cpu-release-cpu-dense-f64-v1-jacobi-api-unfused-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --no-fused --implementations native,numpy-out,torch-eager-out'

CUDA API ABBA command templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-none-prepared-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-none-api-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-jacobi-prepared-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-jacobi-api-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-none-prepared-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-none-api-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-jacobi-prepared-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-jacobi-api-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

CUDA modes command templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-none-prepared-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-none-prepared-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-none-api-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-none-api-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-jacobi-prepared-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-jacobi-prepared-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-jacobi-api-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=3 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-dense-f64-v1-jacobi-api-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,cuda-cublas,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-none-prepared-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-none-prepared-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-none-api-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-none-api-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-jacobi-prepared-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-jacobi-prepared-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-jacobi-api-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=cuda-csr-f64-v1 PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-cuda-csr-f64-v1-jacobi-api-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-none-prepared-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-none-prepared-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-none-api-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-none-api-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner none --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-jacobi-prepared-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-jacobi-prepared-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-jacobi-api-host-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0'

   taskset -c 0 make -C "$SRC" pyprof PRESET=cuda-release \
       PYTHON="$ROOT/.venv-cuda/bin/python" BUILD_DIR="$BUILD" JOBS=2 \
       CMAKE_ARGS=-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
       CUDA_ROOT="$ROOT/.deps/cuda-12.8" CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75 \
       PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=api \
       PYPROF_ROUNDS=30 PYPROF_REPETITIONS=1 \
       PYPROF_OUTPUT="$OUT/cuda-release-sparse-solvers-jacobi-api-device-r$RUN.json" \
       PYPROF_ARGS='--warmups 3 --seed 20261015 --algorithm standard --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50 --preconditioner jacobi --fused --implementations cuda-native,torch-cuda-eager --device cuda --cuda-device 0 --cuda-execution-mode device'

For each completed ABBA configuration, supply both distinct baseline
files and both distinct current files to the unchanged strict comparator:

.. code-block:: bash

   PYTHONPATH="$BUILD_B/python" "$ROOT/.venv/bin/python" \
       "$SRC_B/scripts/compare_performance.py" \
       --baseline "$OUT/config-r1.json" "$OUT/config-r4.json" \
       --current "$OUT/config-r2.json" "$OUT/config-r3.json" \
       --max-regression 0.05 --bootstrap-samples 10000 --seed 20261015 \
       --output "$OUT/config-comparison.json"

Replace ``config`` with its exact configuration name. A nonzero exit
status is expected for regression, inconclusive or incompatible-work
reports; retain the report and raw data. Never compare host/device modes
with this command, suppress a changed-count row, or divide timing by work
counts to manufacture a pass. Existing raw replay requires the local raw
file plus its matching compiled source and runtime; a clean clone must
generate fresh measurements with the commands above.

The separately disclosed focused confirmation uses the same unpreconditioned
prepared dense CUDA template and unchanged compiled source pairs. Execute
it once in A-B-B-A-B-A-A-B order, with fresh outputs named
``confirmation-r1.json`` through ``confirmation-r8.json``. Retain all four
cases and all three candidates in every artifact. Its standalone four-block
report is diagnostic; the pooled strict comparison must also include both
original blocks per side, never replace them. For this one configuration:

.. code-block:: bash

   PYTHONPATH="$BUILD_B/python" "$ROOT/.venv/bin/python" \
       "$SRC_B/scripts/compare_performance.py" \
       --baseline "$OUT/config-r1.json" "$OUT/config-r4.json" \
           "$OUT/confirmation-r1.json" "$OUT/confirmation-r4.json" \
           "$OUT/confirmation-r6.json" "$OUT/confirmation-r7.json" \
       --current "$OUT/config-r2.json" "$OUT/config-r3.json" \
           "$OUT/confirmation-r2.json" "$OUT/confirmation-r3.json" \
           "$OUT/confirmation-r5.json" "$OUT/confirmation-r8.json" \
       --max-regression 0.05 --bootstrap-samples 10000 --seed 20261015 \
       --output "$OUT/config-pooled-comparison.json"

