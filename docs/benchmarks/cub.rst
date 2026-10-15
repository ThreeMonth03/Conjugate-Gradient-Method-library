CUB block-reduction evaluation
==============================

Keep the original reduction implementation. The CUB candidate passes the
correctness checks, but it does not satisfy the predeclared complete 5%
regression-budget gate: **45 matched-work rows are supported, 47 are
inconclusive, and 4 rows change numerical work**. No matched row establishes
a greater-than-5% regression under this interval rule. This decision does not
establish that CUB is universally slower; the three new boundary tests are
retained with the original implementation.

All **128 fresh-process artifacts / 768 candidate-case results** are valid.
The 96 rows below include every configuration, loss, inconclusive result and
changed-work case. Among the 92 matched-work rows, 75 point estimates are
lower with CUB and 17 are higher; point estimates alone do not meet the gate.


Candidate and correctness
-------------------------

* A, original reduction: ``a1302db7c076e102dd18ee6e2dd2f1a094b3510c``.
* B, CUB candidate: ``aec53a08628d5c3147c8a6d99e0a169b74821894``.

B replaces the block sum with CUB 2.7.0 (version code 200700),
``BLOCK_REDUCE_WARP_REDUCTIONS`` and 256 threads. The stable-norm fallback
and solver policy remain; the Python harness/wrappers and active CPU core
binary are identical between A and B. CUB can change floating-point summation
order and therefore the number of CG iterations.

Before measurement, the candidate passed 22 native GPU tests and the pytest
target. Three added boundary tests cover warp/block/grid-stride tails,
non-finite values outside the leading lane, and padded-row storage reuse.
The candidate CTest configuration completed 106 registered checks with zero
failures and the expected unavailable-mode skips. Compute Sanitizer memcheck
and synccheck reported zero errors across the original 19 and new 3 GPU tests.
Serialized racecheck reported zero hazards for the original 19 tests;
racecheck without forced serialization also reported zero hazards for dense
row reuse and the three new tests. Serialized checks do not establish freedom
from races between overlapping streams.
Sanitizer instrumentation/serialization was not used in formal timing.
After rejecting adoption, the original reduction with those tests again
completed 106 registered CTest checks; installed CPU/CUDA wheel/device checks
also passed. Correctness success alone is not performance evidence.


Host, timing and decision protocol
----------------------------------

NVIDIA GeForce RTX 2060, compute capability 7.5, device 0,
6,442,123,264 reported bytes; UUID ``GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302``.
NVIDIA display driver 591.86; runtime API 12080 and driver API 13010.
CUDA toolkit/compiler 12.8.93, architecture 75, host GNU 13.3.0, Release
``-O3 -DNDEBUG``; CPU BLAS/OpenMP disabled. Host: Intel i7-13700K,
WSL2/Linux 6.6.87.2, CPython 3.12.3; NumPy 2.2.6 and SciPy 1.15.3.
CUDA/native and library threads are one, with logical CPU 0 affinity.
GPU clocks/thermal/display activity and Windows background load were not
controlled; logical affinity does not establish physical-core isolation.

There are 16 fixed dense/CSR × fused/separate × none/Jacobi × prepared/API
configurations. Dense uses ``cuda-native,cuda-cublas``; CSR uses
``cuda-native``. For each configuration, eight adjacent fresh numerical
processes run **A-B-B-A-B-A-A-B**, four per source. Every process uses seed
20261015, three untimed warmups, 30 complete Williams rounds, dense three
solves per round / CSR one, standard CG, rtol=1e-8, atol=1e-12, maxiter=2000
and residual refresh/restart every 50 updates. Backend and fusion are explicit;
there is no automatic crossover or sampling reduction.

Completed device-resident wall time includes host/control/scalar
synchronization, every output reset and completion fences. Prepared reuses
matrix/Jacobi/workspace/output; API includes per-call output/workspace/Jacobi
resource construction. Input uploads/snapshots and correctness downloads are
excluded. ``CUDA_LAUNCH_BLOCKING`` is unset and
``CUBLAS_WORKSPACE_CONFIG=:4096:8``. These are whole-solve times, not isolated
block-sum kernel timings or host-to-host latency. No builds ran during sampling.
CSR fixtures have constant diagonal 4; Jacobi has no conditioning benefit here.
See :doc:`/benchmarks/methodology` for the full boundaries.

All solutions pass independent stable CPU true-residual checks before and
after timing; maximum residual/threshold is **0.9867878398498536**.
A/B input hashes, options, hardware, compiler and active binary identities
were checked. The only excluded common-build metadata difference is the
explicitly validated B-only CUB provider/version/algorithm/block-size field;
full per-run build dictionaries remain in the evidence.

A/B milliseconds are medians of normalized round-average timings pooled
across four process artifacts per side (120 rounds each side). They are not
medians of the four process medians or individual-solve tail latency.
Ratios are **B/A**; above one means higher measured latency. The unmodified
``scripts/compare_performance.py`` comparability gate checks exact input,
options and iteration/matvec counts for every row. Its bootstrap resamples
process artifacts and rounds, using 10,000 resamples and a 95% percentile
interval. It does not infer paired covariance from ABBA order; independently
resampled rounds can understate serial dependence, and four process blocks
provide limited variance evidence. Coverage is per row, without 96-row
simultaneous adjustment, and does not establish behavior on another host.

* **Within 5%:** the unrounded interval upper bound is at most 1.05.
  This permits a small slowdown and does not establish a speedup.
* **Inconclusive:** the interval crosses 1.05.
* **Regression:** the interval lower bound exceeds 1.05; none were observed
  among matched-work rows.
* **TTS only:** counts differ, so the strict equal-work gate rejects the row.
  Its time-to-required-tolerance interval is retained separately, and never
  becomes a pure-kernel or equal-work pass.

Under the predeclared conservative rule, any inconclusive, regression or
changed-work row means retaining the original reduction. Classification uses
full precision; displayed intervals are rounded.


Changed-work rows
-----------------

All four are dense-64 cuBLAS without preconditioning. A uses **49 updates /
51 matvecs**, B **48 / 50** in every process. The two fused intervals are
inconclusive; the separate-operation TTS intervals are within budget. Fewer
updates contribute to their elapsed times, so even an interval below one
cannot establish a reduction-kernel speedup. All four remain rejected by the
strict work gate and are visible in the full tables.


All 96 rows
-----------

``fused1`` selects fused vector operations; ``fused0`` selects separate
operations. A/B work columns are iterations/matvecs; differing values are
retained explicitly. Timings are milliseconds.

.. csv-table:: dense-fused1-none-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","7.2408","7.1918","0.99324 [0.93376, 1.04512]","49/51","49/51","within 5%"
   "dense-64/cublas","7.8413","7.8838","1.00542 [0.91181, 1.10448]","49/51","48/50","TTS only: inconclusive"
   "dense-256/native","10.7363","10.9230","1.01739 [0.96977, 1.07411]","67/70","67/70","inconclusive"
   "dense-256/cublas","11.1275","11.1878","1.00542 [0.94504, 1.06381]","67/70","67/70","inconclusive"
   "dense-1024/native","15.0897","15.0645","0.99833 [0.96439, 1.04902]","70/73","70/73","within 5%"
   "dense-1024/cublas","13.8477","14.1514","1.02193 [0.97596, 1.05198]","70/73","70/73","inconclusive"
   "dense-2048/native","24.6049","24.6043","0.99997 [0.95210, 1.03141]","70/73","70/73","within 5%"
   "dense-2048/cublas","22.9644","23.1915","1.00989 [0.95610, 1.04781]","70/73","70/73","within 5%"

.. csv-table:: dense-fused1-none-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","9.8033","9.4725","0.96626 [0.90248, 1.11760]","49/51","49/51","inconclusive"
   "dense-64/cublas","12.0698","11.9874","0.99317 [0.90312, 1.10387]","49/51","48/50","TTS only: inconclusive"
   "dense-256/native","12.8938","13.2380","1.02669 [0.90410, 1.15033]","67/70","67/70","inconclusive"
   "dense-256/cublas","14.4627","14.7520","1.02000 [0.89421, 1.16146]","67/70","67/70","inconclusive"
   "dense-1024/native","17.7134","17.0072","0.96013 [0.88526, 1.07716]","70/73","70/73","inconclusive"
   "dense-1024/cublas","17.8178","17.2161","0.96624 [0.89211, 1.07655]","70/73","70/73","inconclusive"
   "dense-2048/native","28.1334","26.6408","0.94694 [0.85468, 0.99763]","70/73","70/73","within 5%"
   "dense-2048/cublas","28.0597","26.5821","0.94734 [0.86169, 1.00706]","70/73","70/73","within 5%"

.. csv-table:: dense-fused1-jacobi-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","1.2534","1.2344","0.98482 [0.85748, 1.11605]","5/7","5/7","inconclusive"
   "dense-64/cublas","1.3322","1.3125","0.98525 [0.83120, 1.11141]","5/7","5/7","inconclusive"
   "dense-256/native","1.0962","1.0546","0.96201 [0.79279, 1.13340]","4/6","4/6","inconclusive"
   "dense-256/cublas","1.1187","1.1731","1.04858 [0.88917, 1.14742]","4/6","4/6","inconclusive"
   "dense-1024/native","1.4308","1.4039","0.98116 [0.87945, 1.07875]","4/6","4/6","inconclusive"
   "dense-1024/cublas","1.3240","1.3412","1.01304 [0.91989, 1.07923]","4/6","4/6","inconclusive"
   "dense-2048/native","1.8155","1.7654","0.97241 [0.84344, 1.15439]","3/5","3/5","inconclusive"
   "dense-2048/cublas","1.7329","1.8340","1.05834 [0.89431, 1.22313]","3/5","3/5","inconclusive"

.. csv-table:: dense-fused1-jacobi-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","3.8492","3.6455","0.94710 [0.77600, 1.20637]","5/7","5/7","inconclusive"
   "dense-64/cublas","5.9784","5.3270","0.89104 [0.76594, 1.14516]","5/7","5/7","inconclusive"
   "dense-256/native","3.5449","3.4052","0.96059 [0.77355, 1.18797]","4/6","4/6","inconclusive"
   "dense-256/cublas","5.0896","5.3145","1.04420 [0.86810, 1.22790]","4/6","4/6","inconclusive"
   "dense-1024/native","4.1650","3.8702","0.92922 [0.76117, 1.20261]","4/6","4/6","inconclusive"
   "dense-1024/cublas","5.8736","5.7923","0.98616 [0.77591, 1.19415]","4/6","4/6","inconclusive"
   "dense-2048/native","4.6247","4.2094","0.91021 [0.74055, 1.16352]","3/5","3/5","inconclusive"
   "dense-2048/cublas","6.2362","5.7073","0.91520 [0.78353, 1.08357]","3/5","3/5","inconclusive"

.. csv-table:: dense-fused0-none-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","8.6080","7.9593","0.92464 [0.87476, 1.00497]","49/51","49/51","within 5%"
   "dense-64/cublas","8.6331","8.0757","0.93543 [0.89851, 0.99869]","49/51","48/50","TTS only: within 5%"
   "dense-256/native","11.7651","10.8770","0.92451 [0.82422, 1.01019]","67/70","67/70","within 5%"
   "dense-256/cublas","11.4628","10.9631","0.95641 [0.82122, 1.01149]","67/70","67/70","within 5%"
   "dense-1024/native","16.1408","15.0605","0.93307 [0.87898, 0.98519]","70/73","70/73","within 5%"
   "dense-1024/cublas","14.7237","13.9305","0.94613 [0.89506, 1.00034]","70/73","70/73","within 5%"
   "dense-2048/native","25.6228","24.1444","0.94230 [0.90867, 1.01228]","70/73","70/73","within 5%"
   "dense-2048/cublas","24.0015","22.9630","0.95673 [0.92098, 1.03348]","70/73","70/73","within 5%"

.. csv-table:: dense-fused0-none-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","9.9703","9.7419","0.97709 [0.93512, 1.02552]","49/51","49/51","within 5%"
   "dense-64/cublas","11.5222","11.1199","0.96508 [0.92759, 1.01004]","49/51","48/50","TTS only: within 5%"
   "dense-256/native","13.1313","12.6749","0.96525 [0.90845, 1.01772]","67/70","67/70","within 5%"
   "dense-256/cublas","14.8562","14.3396","0.96522 [0.92980, 1.01911]","67/70","67/70","within 5%"
   "dense-1024/native","17.7178","17.6941","0.99867 [0.92117, 1.05283]","70/73","70/73","inconclusive"
   "dense-1024/cublas","18.1614","17.9968","0.99094 [0.92734, 1.05096]","70/73","70/73","inconclusive"
   "dense-2048/native","27.1550","27.1077","0.99826 [0.95353, 1.04675]","70/73","70/73","within 5%"
   "dense-2048/cublas","26.7606","26.8616","1.00377 [0.95237, 1.04377]","70/73","70/73","within 5%"

.. csv-table:: dense-fused0-jacobi-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","1.6188","1.3427","0.82947 [0.67816, 1.10565]","5/7","5/7","inconclusive"
   "dense-64/cublas","1.6826","1.3620","0.80948 [0.66771, 1.03668]","5/7","5/7","within 5%"
   "dense-256/native","1.3378","1.1133","0.83224 [0.65789, 1.02500]","4/6","4/6","within 5%"
   "dense-256/cublas","1.3949","1.1206","0.80339 [0.65763, 1.00152]","4/6","4/6","within 5%"
   "dense-1024/native","1.5703","1.4792","0.94198 [0.75045, 1.22876]","4/6","4/6","inconclusive"
   "dense-1024/cublas","1.4840","1.4521","0.97848 [0.76137, 1.25254]","4/6","4/6","inconclusive"
   "dense-2048/native","1.8427","2.1856","1.18611 [0.88807, 1.45952]","3/5","3/5","inconclusive"
   "dense-2048/cublas","1.8596","1.9533","1.05038 [0.81411, 1.38640]","3/5","3/5","inconclusive"

.. csv-table:: dense-fused0-jacobi-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "dense-64/native","2.3595","2.1562","0.91384 [0.74720, 1.06149]","5/7","5/7","inconclusive"
   "dense-64/cublas","3.9559","3.6127","0.91326 [0.82762, 1.06167]","5/7","5/7","inconclusive"
   "dense-256/native","2.0419","2.0647","1.01118 [0.92084, 1.06664]","4/6","4/6","inconclusive"
   "dense-256/cublas","3.5900","3.5452","0.98753 [0.88618, 1.05275]","4/6","4/6","inconclusive"
   "dense-1024/native","2.6107","2.5350","0.97102 [0.85461, 1.13268]","4/6","4/6","inconclusive"
   "dense-1024/cublas","4.2348","4.0572","0.95805 [0.85384, 1.05875]","4/6","4/6","inconclusive"
   "dense-2048/native","2.8345","2.7818","0.98142 [0.89737, 1.17415]","3/5","3/5","inconclusive"
   "dense-2048/cublas","4.4726","4.2658","0.95376 [0.87387, 1.09890]","3/5","3/5","inconclusive"

.. csv-table:: csr-fused1-none-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","2.9345","2.6506","0.90324 [0.86715, 0.95242]","26/28","26/28","within 5%"
   "poisson-32/native","14.5325","13.6639","0.94023 [0.82426, 1.09530]","129/133","129/133","inconclusive"
   "poisson-64/native","34.9017","33.3565","0.95573 [0.89791, 1.00418]","290/297","290/297","within 5%"
   "poisson-128/native","48.8453","48.6772","0.99656 [0.96850, 1.02400]","395/404","395/404","within 5%"

.. csv-table:: csr-fused1-none-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","4.2586","4.2063","0.98771 [0.87852, 1.08011]","26/28","26/28","inconclusive"
   "poisson-32/native","17.9022","17.1078","0.95563 [0.89611, 1.03646]","129/133","129/133","within 5%"
   "poisson-64/native","36.4431","34.2855","0.94079 [0.91193, 0.96641]","290/297","290/297","within 5%"
   "poisson-128/native","55.0629","51.8445","0.94155 [0.84761, 1.05637]","395/404","395/404","inconclusive"

.. csv-table:: csr-fused1-jacobi-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","4.6159","4.2546","0.92172 [0.77295, 1.16885]","26/28","26/28","inconclusive"
   "poisson-32/native","23.2222","22.6378","0.97483 [0.92589, 1.11199]","129/133","129/133","inconclusive"
   "poisson-64/native","52.0523","51.9870","0.99875 [0.86453, 1.04281]","290/297","290/297","within 5%"
   "poisson-128/native","71.1289","70.6627","0.99345 [0.96652, 1.01784]","395/404","395/404","within 5%"

.. csv-table:: csr-fused1-jacobi-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","5.8099","5.3595","0.92249 [0.87818, 0.98600]","26/28","26/28","within 5%"
   "poisson-32/native","24.8993","24.3238","0.97689 [0.93771, 0.99745]","129/133","129/133","within 5%"
   "poisson-64/native","52.8291","51.9504","0.98337 [0.96972, 0.99701]","290/297","290/297","within 5%"
   "poisson-128/native","72.3470","73.4062","1.01464 [0.98875, 1.05430]","395/404","395/404","inconclusive"

.. csv-table:: csr-fused0-none-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","3.3197","3.0487","0.91834 [0.83675, 1.01495]","26/28","26/28","within 5%"
   "poisson-32/native","17.5733","16.2221","0.92311 [0.87503, 1.03388]","129/133","129/133","within 5%"
   "poisson-64/native","37.8152","37.8552","1.00106 [0.97000, 1.04690]","290/297","290/297","within 5%"
   "poisson-128/native","56.0809","52.2982","0.93255 [0.89202, 0.96770]","395/404","395/404","within 5%"

.. csv-table:: csr-fused0-none-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","4.5709","4.4286","0.96887 [0.88871, 1.05666]","26/28","26/28","inconclusive"
   "poisson-32/native","19.5020","19.2956","0.98942 [0.95152, 1.03170]","129/133","129/133","within 5%"
   "poisson-64/native","39.9900","39.0062","0.97540 [0.95239, 1.00852]","290/297","290/297","within 5%"
   "poisson-128/native","57.9792","54.0702","0.93258 [0.90630, 0.95392]","395/404","395/404","within 5%"

.. csv-table:: csr-fused0-jacobi-prepared
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","4.6035","4.3630","0.94776 [0.88765, 1.02270]","26/28","26/28","within 5%"
   "poisson-32/native","24.5634","23.7792","0.96808 [0.94060, 1.00606]","129/133","129/133","within 5%"
   "poisson-64/native","53.9303","54.4694","1.01000 [0.97698, 1.06698]","290/297","290/297","inconclusive"
   "poisson-128/native","79.6393","75.0152","0.94194 [0.91601, 0.97047]","395/404","395/404","within 5%"

.. csv-table:: csr-fused0-jacobi-api
   :class: benchmark-results
   :header: "case/kernel","A ms","B ms","B/A [95% interval]","A work","B work","decision"

   "poisson-8/native","5.9081","5.6372","0.95415 [0.89054, 1.01435]","26/28","26/28","within 5%"
   "poisson-32/native","25.9829","25.2192","0.97061 [0.94321, 0.99575]","129/133","129/133","within 5%"
   "poisson-64/native","55.8281","54.2957","0.97255 [0.94014, 1.01952]","290/297","290/297","within 5%"
   "poisson-128/native","82.6627","77.0066","0.93158 [0.90388, 0.95405]","395/404","395/404","within 5%"


Local evidence and compiled identity
------------------------------------

Raw local evidence is under ``build/performance-cleanup/cub/results/raw/``:
**128 JSON files, 4,210,851 bytes**. The original stage identities, predeclared
commands, raw SHA-256/bytes and complete per-run build dictionaries remain in
``build/performance-cleanup/cub/results/manifest.json``. Full-precision row
statistics/counts/residuals and every loss remain in
``build/performance-cleanup/cub/results/analysis.json``. These raw files and
local tools are ignored, not stored in Git.

* ``manifest.json`` SHA-256:
  ``43ae12a7c2abe85524e87b066eeb61b58c9f74873138ddb1bc6d34f5905bb6f5``.
* ``analysis.json`` SHA-256:
  ``187221700e9a829f23d2ba3299ae7b0adbd69960379134834fcb44221da35024``.

Actual loaded CUDA extension SHA-256:

* A: ``419b4718d7eed0b9544805fe8f89bf4675f29f000794a259a714d4198450454c``.
* B: ``fdd2300a06728b39dd21d7931f2062ace33d70c45710110db61a632b9bf03737``.

Both active CPU extensions have SHA-256
``7e68ab3075f92801e2c2d53e58405f1cccf4f1dc8d2e3147b65c7159b2ecf853``.
The immutable staged packages are ``build/performance-cleanup/cub/A/python``
and ``.../B/python``; their adjacent identity manifests supply compiled source
provenance. The collector reports live checkout revision
``8c0fd49749c8ce5b9867e5708a0c188f29a8984a`` separately; that revision adds
tests and is not substituted for either compiled extension revision.
Summed worker wall time was 846.686 seconds (about 14.1 minutes), including
untimed setup/auditing; it is not summed solve-only time. No failed runs were
retried or selectively discarded.

Candidate source remains publicly fetchable on
``experiment/cub-block-reductions`` even though it is not adopted. The exact
compiled commits above, rather than a moving branch head, define reproduction.


Reproduce the CUB A/B comparison
--------------------------------

Use the project-local CPython 3.12 CUDA environment and pinned toolkit setup
from :doc:`/benchmarks/current`, under **Reproduce**. The following Bash
commands run from that repository root. They build two detached sources with
the same interpreter, toolkit, compiler, architecture and Release preset:

* A, before CUB: ``a1302db7c076e102dd18ee6e2dd2f1a094b3510c``.
* B, CUB candidate: ``aec53a08628d5c3147c8a6d99e0a169b74821894``.

The public ``experiment/cub-block-reductions`` branch retains B even if the
candidate is rejected. Fetch it to obtain the objects, then select the exact
commits below; do not substitute the branch head for B. Choose compatible
affinity, architecture and compiler values on another host and retain that
changed environment in the new JSON. Use a fresh output directory, keep both
sources unchanged, and finish compilation before sampling.

.. code-block:: bash

   set -euo pipefail
   repo="$PWD"
   work="$repo/build/cub-reproduction"
   python="$repo/.venv-cuda/bin/python"
   toolkit="$repo/.deps/cuda-12.8"
   host_compiler=/usr/bin/g++-13
   architecture=75
   affinity=0
   test ! -e "$work"
   git fetch origin experiment/cub-block-reductions
   mkdir -p "$work/sources" "$work/raw" "$work/comparisons"
   git worktree add --detach "$work/sources/A" a1302db7c076e102dd18ee6e2dd2f1a094b3510c
   git worktree add --detach "$work/sources/B" aec53a08628d5c3147c8a6d99e0a169b74821894
   for variant in A B; do
       make -C "$work/sources/$variant" build PRESET=cuda-release \
           PYTHON="$python" BUILD_DIR="$work/build/$variant" JOBS=2 \
           CUDA_ROOT="$toolkit" CUDA_HOST_COMPILER="$host_compiler" \
           CUDA_ARCHITECTURES="$architecture"
   done
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
   export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONNOUSERSITE=1
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   unset CUDA_LAUNCH_BLOCKING

Continue in the same Bash session. There are 16 fixed configurations:
dense/CSR × fused/separate × CG/Jacobi PCG × prepared/API. Dense compares
``cuda-native,cuda-cublas`` and repeats three solves per round; CSR uses
``cuda-native`` and one solve. Each configuration runs eight adjacent fresh
numerical worker processes in **ABBABAAB** order, four per source, for 128
artifacts total. Each worker uses seed 20261015, three warmups, 30 rounds,
one thread and device 0. Native vector fusion is independent of cuBLAS GEMV.

.. code-block:: bash

   : > "$work/configs.txt"
   for storage in dense csr; do
       if [ "$storage" = dense ]; then
           candidates=cuda-native,cuda-cublas
           repetitions=3
       else
           candidates=cuda-native
           repetitions=1
       fi
       for fused in 1 0; do
           if [ "$fused" -eq 1 ]; then
               fusion=--fused
           else
               fusion=--no-fused
           fi
           for preconditioner in none jacobi; do
               for scope in prepared api; do
                   config="$storage-fused$fused-$preconditioner-$scope"
                   block=0
                   for variant in A B B A B A A B; do
                       block=$((block + 1))
                       printf -v block_id '%02d' "$block"
                       name="$config-block$block_id-$variant"
                       flags="--device cuda --cuda-device 0 --implementations $candidates"
                       flags+=" --seed 20261015 --warmups 3 --algorithm standard $fusion"
                       flags+=" --preconditioner $preconditioner --rtol 1e-8 --atol 1e-12"
                       flags+=" --maxiter 2000 --residual-refresh 50"
                       taskset -c "$affinity" make -C "$work/sources/$variant" pyprof \
                           PRESET=cuda-release PYTHON="$python" JOBS=2 \
                           BUILD_DIR="$work/build/$variant" CUDA_ROOT="$toolkit" \
                           CUDA_HOST_COMPILER="$host_compiler" CUDA_ARCHITECTURES="$architecture" \
                           PYPROF_SUITE="cuda-$storage-f64-v1" PYPROF_THREADS=1 \
                           PYPROF_SCOPE="$scope" PYPROF_ROUNDS=30 \
                           PYPROF_REPETITIONS="$repetitions" PYPROF_ARGS="$flags" \
                           PYPROF_OUTPUT="$work/raw/$name.json" > "$work/raw/$name.log" 2>&1
                   done
                   printf '%s\n' "$config" >> "$work/configs.txt"
               done
           done
       done
   done

Sampling stops on a failed command; do not selectively retry or discard a
block. The harness checks the true residual before and after timing. Inspect
every artifact's candidate status, input hash, counts, loaded module hashes,
source revision and environment before using its timings. These are completed
device-resident wall times, including output reset and completion. Prepared
reuse and per-call API setup follow :doc:`/benchmarks/methodology`; uploads
and correctness downloads are excluded. The original audit driver, binary
snapshots and raw manifest remain local ignored evidence; these commands need
none of those files and generate a separate set of results. Rebuilt binary
bytes and measured latency need not match the original host.

Compare all four A files against all four B files for each configuration,
using the existing strict comparison script and the predeclared 5% budget:

.. code-block:: bash

   while IFS= read -r config; do
       before=( "$work/raw/$config"-block??-A.json )
       after=( "$work/raw/$config"-block??-B.json )
       test "${#before[@]}" -eq 4
       test "${#after[@]}" -eq 4
       if PYTHONPATH="$work/build/B/python" "$python" \
           "$work/sources/B/scripts/compare_performance.py" \
           --baseline "${before[@]}" --current "${after[@]}" \
           --max-regression 0.05 --bootstrap-samples 10000 --seed 20261015 \
           --output "$work/comparisons/$config.json" \
           > "$work/comparisons/$config.log" 2>&1; then
           printf '%s: supported within budget\n' "$config"
       else
           printf '%s: inspect rejection/regression/inconclusive log\n' "$config"
       fi
   done < "$work/configs.txt"

Ratios are B/A: values above one mean higher measured latency. The script
resamples process blocks and rounds and reports a 95% bootstrap interval; it
does not infer paired covariance from ABBABAAB filenames. An interval crossing
1.05 is inconclusive for the chosen regression budget. A supported budget
result does not by itself establish a speedup or universal performance.

The comparison requires matching fixtures, candidate sets, options and
iteration/matvec counts. CUB can change floating-point reduction order and
therefore counts. Keep the strict gate intact: a count mismatch is rejected,
not a pass, and must be reported separately as time to an accepted solution,
with its counts and true residuals. It cannot establish a pure-kernel speedup.
Keep all 16 configurations and every loss or inconclusive result visible.
