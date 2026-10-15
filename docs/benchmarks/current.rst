Cleanup baseline — 2026-10-15
=============================

Measured compiled source: ``d34036f5668559612766cb5d8872a41f3097ffdc``. These are
the baseline measurements before the subsequent conversion/API, ownership
and CUB evaluation changes. They are not measurements of every later checkout.
The report commit follows the measured revision. All 52 runs were completed
before tracked files changed; the two pre-existing untracked research files
were preserved. No failed runs were retried or discarded.

All **680 candidate/case results** across 26 configurations × two fresh
processes passed independent stable CPU true-residual checks before and after
timing. Full input bytes are represented by recorded hashes.

On these samples, native CPU dense remains slower than NumPy/Torch at larger
sizes. The explicit CPU BLAS median beats Torch in all 16 rows and NumPy in
15; unpreconditioned prepared n=2048 loses to NumPy (79.46 versus 75.61 ms),
with overlapping process ranges. Native CPU CSR medians are lower than both
references in all listed rows.

Native CUDA medians are below eager Torch in every listed row, including
dense Jacobi API; the older native API loss is not reproduced at this revision.
Explicit cuBLAS still loses all four dense Jacobi API comparisons. These
observations identify resource construction as a phase to profile; subtracting
API/prepared medians would not establish its isolated cost.

On the four scaled sparse GPU inputs, Jacobi reduces native updates from
575/1498/261/444 to 48/115/25/37, with 7.42–8.72× lower prepared medians.
The neutral case stays at 50 updates and adds work. Some unpreconditioned
GPU CG counts differ between native and Torch, as explicitly retained below.
CPU LU reuse remains faster than native PCG on the five default strategy
fixtures after its prior factorization. All these rankings are descriptive,
with the two-process uncertainty limits given below.


Hardware, builds and sampling
-----------------------------

Host: 13th Gen Intel(R) Core(TM) i7-13700K; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; CPython 3.12.3.
GPU: NVIDIA GeForce RTX 2060, compute capability 7.5,
device 0, 6,442,123,264 reported bytes; UUID ``GPU-725be96d-2b67-c8ec-fa10-5d9b23a2c302``.
NVIDIA display driver 591.86; CUDA toolkit/compiler 12.8.93, runtime API 12080,
driver API 13010, architecture 75. GPU clocks, thermal/display activity and
physical-core isolation from the Windows host were not controlled.

All runs use ``taskset -c 0``, one native/library/Torch intraop thread and
one Torch interop thread. CPU builds use GNU 16.0.1; CUDA uses GNU 13.3.0.
All are Release ``-O3 -DNDEBUG`` with OpenMP disabled. CPU BLAS is the
private sequential LP64 OpenBLAS 0.3.30 provider with locking. No automatic
kernel selection is used. NumPy 2.2.6, SciPy 1.15.3 and Torch 2.8.0+cpu /
2.8.0+cu128 use separate project-local CPU/CUDA environments.

Every configuration uses seed 20261015 and three warmups. Implementation
candidates use standard CG, fused native operations, rtol=1e-8, atol=1e-12,
maxiter=2000 and true-residual
refresh/restart every 50 updates. Dense repeats three solves per round;
CSR repeats one. Thirty requested Williams rounds remain 30 for two, three
or six candidates; the four-candidate BLAS configuration completes 32 rounds.
Pass one executes the configurations forward and pass two reverses them.

Each cell is the **median of the two process medians**, in milliseconds,
followed by **[minimum–maximum process median]**. Each process median uses
round-average completed wall times. The bracket is neither a confidence
interval nor individual-call tail latency. Two launches are limited evidence
of process variance; no statistical significance or universal speedup is claimed.

Native and matching-policy prepared candidates reuse output/workspace/Jacobi
and charge every x0 reset. API charges per-call output/workspace/Jacobi
allocation and setup. Official SciPy CG allocates iterative workspace in both
scopes and follows its own residual policy. CUDA inputs
are resident, with completion fences; uploads/snapshots, canonical CSR
conversion and correctness downloads are excluded. Torch uses eager public
operations implementing this project's matching-policy CG, not an official
Torch CG solver, ``torch.compile`` or CUDA Graphs. SuperLU total/reuse and
official SciPy CG/PCG remain strategy comparisons with their own stopping
and allocation policies. LU reuse excludes factorization in both scopes.
Do not subtract separately measured API/prepared medians to isolate a phase.
See :doc:`methodology` for the full timing, storage and accuracy contracts.

The work column gives iteration/matvec counts. A single value is shared by
all candidates and both processes; otherwise labels identify the differences.
N=native, B=BLAS/cuBLAS, NP=NumPy, S=SciPy-out, T=Torch; NCG/NPCG and
SCG/SPCG denote native and official SciPy CG/PCG; LUT/LUR denote LU total/reuse.
``max R/tol`` is the largest independent residual/threshold ratio among
that row's candidates and both processes. Different counts mean different
numerical work despite shared recurrence; those are times to accepted
solutions, not equal-work kernel speedups.


CPU release — dense
-------------------

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","N","NP","T","iter/matvec","max R/tol"

   "none","prepared","dense-64","0.0323 [0.0319–0.0326]","0.1860 [0.1853–0.1868]","0.2736 [0.2735–0.2738]","49/51","0.893697"
   "none","prepared","dense-256","0.5733 [0.5714–0.5752]","0.5693 [0.5134–0.6251]","0.6769 [0.6727–0.6810]","67/70","0.902831"
   "none","prepared","dense-1024","10.7700 [10.6922–10.8478]","9.0658 [8.9677–9.1639]","9.6995 [9.5407–9.8583]","70/73","0.880853"
   "none","prepared","dense-2048","122.2988 [116.5445–128.0530]","76.2861 [71.1337–81.4385]","84.6573 [74.1511–95.1635]","70/73","0.897447"
   "none","api","dense-64","0.0414 [0.0408–0.0419]","0.1933 [0.1931–0.1935]","0.2952 [0.2922–0.2982]","49/51","0.893697"
   "none","api","dense-256","0.5826 [0.5642–0.6009]","0.5818 [0.5565–0.6071]","0.7433 [0.7332–0.7535]","67/70","0.902831"
   "none","api","dense-1024","9.8246 [9.7512–9.8981]","7.5968 [7.4708–7.7229]","8.0648 [7.9393–8.1902]","70/73","0.880853"
   "none","api","dense-2048","133.5519 [116.9792–150.1246]","87.7914 [74.1477–101.4352]","94.1355 [83.7485–104.5224]","70/73","0.897447"
   "jacobi","prepared","dense-64","0.0071 [0.0071–0.0072]","0.0286 [0.0285–0.0286]","0.0444 [0.0435–0.0452]","5/7","0.041680"
   "jacobi","prepared","dense-256","0.0522 [0.0521–0.0524]","0.0510 [0.0468–0.0551]","0.0654 [0.0643–0.0666]","4/6","0.173276"
   "jacobi","prepared","dense-1024","1.0910 [1.0868–1.0952]","0.9460 [0.8960–0.9959]","1.0780 [1.0612–1.0947]","4/6","0.015961"
   "jacobi","prepared","dense-2048","8.7216 [7.7434–9.6999]","6.4917 [5.4934–7.4901]","6.6167 [5.7315–7.5019]","3/5","0.844587"
   "jacobi","api","dense-64","0.0156 [0.0150–0.0162]","0.0423 [0.0390–0.0456]","0.0715 [0.0681–0.0749]","5/7","0.041680"
   "jacobi","api","dense-256","0.0578 [0.0561–0.0594]","0.0646 [0.0637–0.0654]","0.0969 [0.0960–0.0978]","4/6","0.173276"
   "jacobi","api","dense-1024","0.8833 [0.8463–0.9203]","0.7713 [0.7200–0.8227]","0.8919 [0.8670–0.9168]","4/6","0.015961"
   "jacobi","api","dense-2048","11.2840 [9.4532–13.1149]","7.5827 [6.4733–8.6920]","8.6846 [7.0566–10.3125]","3/5","0.844587"


CPU release — CSR
-----------------

The Poisson diagonal is constant 4: Jacobi is scalar scaling and
has no conditioning benefit on this suite.

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","N","S","T","iter/matvec","max R/tol"

   "none","prepared","poisson-8","0.0115 [0.0102–0.0128]","0.1575 [0.1322–0.1828]","0.2076 [0.1691–0.2461]","26/28","0.658698"
   "none","prepared","poisson-32","0.3764 [0.3382–0.4147]","1.4021 [1.1457–1.6584]","1.4455 [1.1062–1.7848]","129/133","0.971015"
   "none","prepared","poisson-64","3.0041 [2.7202–3.2880]","6.7633 [5.6183–7.9083]","6.0249 [4.4421–7.6076]","290/297","0.986788"
   "none","prepared","poisson-128","14.9528 [14.8224–15.0833]","26.3708 [26.0029–26.7387]","18.0927 [17.6816–18.5039]","395/404","0.984698"
   "none","api","poisson-8","0.0191 [0.0174–0.0208]","0.1405 [0.1338–0.1472]","0.1839 [0.1759–0.1918]","26/28","0.658698"
   "none","api","poisson-32","0.3732 [0.3690–0.3774]","1.2372 [1.2229–1.2515]","1.2289 [1.2160–1.2419]","129/133","0.971015"
   "none","api","poisson-64","2.8820 [2.8635–2.9006]","5.7967 [5.5908–6.0025]","4.9247 [4.7519–5.0974]","290/297","0.986788"
   "none","api","poisson-128","15.8451 [15.6788–16.0114]","27.7452 [27.2014–28.2889]","18.0600 [17.6772–18.4428]","395/404","0.984698"
   "jacobi","prepared","poisson-8","0.0125 [0.0118–0.0132]","0.1606 [0.1555–0.1658]","0.2188 [0.2116–0.2259]","26/28","0.658698"
   "jacobi","prepared","poisson-32","0.4092 [0.4017–0.4167]","1.3667 [1.3375–1.3958]","1.4121 [1.3559–1.4682]","129/133","0.971015"
   "jacobi","prepared","poisson-64","3.2763 [3.2444–3.3081]","6.4279 [6.4046–6.4511]","5.5259 [5.2834–5.7683]","290/297","0.986788"
   "jacobi","prepared","poisson-128","17.3979 [17.0547–17.7410]","28.6252 [28.3872–28.8633]","20.1415 [19.5747–20.7083]","395/404","0.984698"
   "jacobi","api","poisson-8","0.0240 [0.0217–0.0262]","0.1952 [0.1753–0.2152]","0.2870 [0.2547–0.3193]","26/28","0.658698"
   "jacobi","api","poisson-32","0.4739 [0.4365–0.5114]","1.5453 [1.3556–1.7349]","1.7329 [1.5109–1.9548]","129/133","0.971015"
   "jacobi","api","poisson-64","3.6963 [3.4352–3.9574]","7.4631 [6.4656–8.4607]","6.7202 [5.7233–7.7171]","290/297","0.986788"
   "jacobi","api","poisson-128","17.9828 [17.5230–18.4427]","29.9931 [29.3265–30.6596]","21.1601 [21.1576–21.1625]","395/404","0.984698"


CPU BLAS build — dense
----------------------

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","N","B","NP","T","iter/matvec","max R/tol"

   "none","prepared","dense-64","0.0344 [0.0342–0.0347]","0.0223 [0.0219–0.0226]","0.2036 [0.2027–0.2045]","0.3083 [0.2988–0.3178]","49/51","0.893697"
   "none","prepared","dense-256","0.5618 [0.5492–0.5745]","0.3224 [0.2774–0.3674]","0.5754 [0.5273–0.6234]","0.6720 [0.6470–0.6970]","67/70","0.949804"
   "none","prepared","dense-1024","10.8858 [10.8637–10.9080]","8.6246 [8.2998–8.9495]","9.5691 [9.4619–9.6763]","9.7125 [9.6011–9.8239]","70/73","0.880853"
   "none","prepared","dense-2048","125.6368 [111.9356–139.3380]","79.4552 [72.3696–86.5407]","75.6052 [69.3558–81.8547]","87.5200 [76.2406–98.7994]","70/73","0.897447"
   "none","api","dense-64","0.0387 [0.0377–0.0396]","0.0275 [0.0274–0.0277]","0.1938 [0.1919–0.1958]","0.2849 [0.2809–0.2890]","49/51","0.893697"
   "none","api","dense-256","0.5435 [0.5378–0.5492]","0.3092 [0.2619–0.3566]","0.5524 [0.5246–0.5802]","0.7044 [0.6611–0.7477]","67/70","0.949804"
   "none","api","dense-1024","9.4791 [9.1492–9.8089]","6.5780 [6.4167–6.7392]","7.1725 [6.9727–7.3723]","7.6103 [7.4793–7.7413]","70/73","0.880853"
   "none","api","dense-2048","124.3239 [123.2999–125.3480]","78.6205 [76.3474–80.8935]","80.2853 [79.4052–81.1654]","81.8049 [81.2807–82.3291]","70/73","0.897447"
   "jacobi","prepared","dense-64","0.0072 [0.0069–0.0074]","0.0058 [0.0056–0.0060]","0.0309 [0.0302–0.0316]","0.0486 [0.0466–0.0505]","5/7","0.041680"
   "jacobi","prepared","dense-256","0.0512 [0.0492–0.0531]","0.0269 [0.0258–0.0280]","0.0572 [0.0551–0.0594]","0.0680 [0.0644–0.0716]","4/6","0.173276"
   "jacobi","prepared","dense-1024","1.1496 [1.1215–1.1778]","0.9352 [0.9118–0.9585]","0.9681 [0.9402–0.9961]","1.0800 [1.0759–1.0842]","4/6","0.015961"
   "jacobi","prepared","dense-2048","9.1624 [8.3778–9.9469]","5.7542 [5.5466–5.9618]","6.5701 [6.0389–7.1014]","6.6466 [6.0483–7.2449]","3/5","0.844587"
   "jacobi","api","dense-64","0.0130 [0.0129–0.0131]","0.0126 [0.0125–0.0127]","0.0409 [0.0405–0.0414]","0.0738 [0.0721–0.0755]","5/7","0.041680"
   "jacobi","api","dense-256","0.0575 [0.0554–0.0596]","0.0375 [0.0341–0.0409]","0.0633 [0.0613–0.0653]","0.0973 [0.0942–0.1004]","4/6","0.173276"
   "jacobi","api","dense-1024","0.8729 [0.8489–0.8969]","0.6624 [0.6376–0.6872]","0.7592 [0.7467–0.7718]","0.8882 [0.8595–0.9170]","4/6","0.015961"
   "jacobi","api","dense-2048","8.4539 [7.6997–9.2080]","5.0473 [4.7342–5.3603]","5.4693 [5.2213–5.7174]","6.3584 [5.2693–7.4476]","3/5","0.844587"


CUDA — dense
------------

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","N","B","T","iter/matvec","max R/tol"

   "none","prepared","dense-64","6.0612 [5.9523–6.1701]","6.2410 [6.0727–6.4094]","12.0772 [11.7963–12.3581]","49/51","0.805850"
   "none","prepared","dense-256","9.2882 [9.0993–9.4771]","9.0463 [8.8700–9.2226]","18.6506 [18.4365–18.8647]","67/70","0.944865"
   "none","prepared","dense-1024","10.9658 [10.4475–11.4841]","10.0428 [9.5056–10.5801]","17.5017 [16.1631–18.8403]","70/73","0.880853"
   "none","prepared","dense-2048","19.9544 [19.8562–20.0525]","18.2762 [17.9892–18.5632]","25.6436 [25.1801–26.1071]","70/73","0.897447"
   "none","api","dense-64","6.6795 [6.3204–7.0386]","8.4697 [8.3047–8.6346]","12.9628 [12.4572–13.4684]","49/51","0.805850"
   "none","api","dense-256","9.0779 [8.8774–9.2784]","11.0377 [10.7450–11.3303]","18.0043 [17.7393–18.2692]","67/70","0.944865"
   "none","api","dense-1024","11.7241 [11.6856–11.7626]","12.4333 [12.4001–12.4666]","17.9361 [17.1552–18.7170]","70/73","0.880853"
   "none","api","dense-2048","19.8588 [19.3429–20.3747]","20.2758 [20.2266–20.3250]","25.3478 [25.0289–25.6667]","70/73","0.897447"
   "jacobi","prepared","dense-64","1.0608 [0.9910–1.1306]","1.0496 [0.9559–1.1432]","2.0045 [1.8566–2.1524]","5/7","0.041680"
   "jacobi","prepared","dense-256","0.8901 [0.7850–0.9951]","0.9478 [0.8182–1.0773]","1.6860 [1.4406–1.9315]","4/6","0.173276"
   "jacobi","prepared","dense-1024","1.2610 [1.0965–1.4254]","1.0804 [1.0031–1.1576]","1.6880 [1.4839–1.8920]","4/6","0.015961"
   "jacobi","prepared","dense-2048","1.4435 [1.3705–1.5165]","1.3054 [1.1914–1.4195]","1.9233 [1.6815–2.1652]","3/5","0.844587"
   "jacobi","api","dense-64","1.1687 [1.1327–1.2046]","3.2810 [3.2184–3.3436]","2.5633 [2.4624–2.6642]","5/7","0.041680"
   "jacobi","api","dense-256","1.0022 [0.9803–1.0240]","3.0166 [2.9945–3.0387]","2.1523 [2.1482–2.1563]","4/6","0.173276"
   "jacobi","api","dense-1024","1.3311 [1.3228–1.3395]","2.9725 [2.9211–3.0238]","2.1146 [2.0652–2.1639]","4/6","0.015961"
   "jacobi","api","dense-2048","1.5593 [1.4566–1.6621]","3.2110 [3.1193–3.3027]","2.3082 [2.2479–2.3684]","3/5","0.844587"


CUDA — constant-diagonal CSR
----------------------------

The Poisson diagonal is constant 4: Jacobi is scalar scaling and
has no conditioning benefit on this suite.

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","N","T","iter/matvec","max R/tol"

   "none","prepared","poisson-8","2.9719 [2.9612–2.9826]","6.1768 [5.7531–6.6005]","26/28","0.658698"
   "none","prepared","poisson-32","16.2430 [15.9087–16.5774]","31.4184 [30.5676–32.2692]","129/133","0.971015"
   "none","prepared","poisson-64","35.5777 [33.6075–37.5479]","68.2418 [64.5363–71.9474]","290/297","0.986788"
   "none","prepared","poisson-128","50.2899 [48.8504–51.7293]","95.6425 [89.5245–101.7606]","395/404","0.984698"
   "none","api","poisson-8","3.2958 [3.0347–3.5570]","6.1661 [5.3882–6.9440]","26/28","0.658698"
   "none","api","poisson-32","15.5457 [15.1345–15.9569]","30.5565 [28.5029–32.6101]","129/133","0.971015"
   "none","api","poisson-64","35.3437 [32.7193–37.9681]","69.6143 [67.5248–71.7039]","290/297","0.986788"
   "none","api","poisson-128","49.5050 [49.4595–49.5505]","91.4707 [90.3118–92.6296]","395/404","0.984698"
   "jacobi","prepared","poisson-8","4.9214 [4.2910–5.5519]","8.8884 [8.6003–9.1765]","26/28","0.658698"
   "jacobi","prepared","poisson-32","24.0287 [22.6674–25.3899]","43.2289 [39.5559–46.9020]","129/133","0.971015"
   "jacobi","prepared","poisson-64","50.2152 [49.3427–51.0878]","95.5511 [93.1101–97.9922]","290/297","0.986788"
   "jacobi","prepared","poisson-128","73.6294 [72.9193–74.3394]","139.0250 [136.8366–141.2133]","395/404","0.984698"
   "jacobi","api","poisson-8","4.4023 [4.3984–4.4062]","8.4792 [8.4125–8.5458]","26/28","0.658698"
   "jacobi","api","poisson-32","24.7613 [22.9605–26.5621]","45.0635 [42.2730–47.8540]","129/133","0.971015"
   "jacobi","api","poisson-64","51.6640 [47.3984–55.9295]","98.9125 [93.9397–103.8854]","290/297","0.986788"
   "jacobi","api","poisson-128","72.5927 [72.1054–73.0801]","138.8207 [138.5700–139.0713]","395/404","0.984698"


CUDA — scaled sparse CG/PCG
---------------------------

Scaled cases use ``A = D P D`` with positive D spanning 1..20,
plus a constant-diagonal neutral case. These parameters are not exact
condition numbers. Native/Torch CG have different counts on two inputs,
while PCG counts match; both pass the independent true-residual gate.

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","N","T","iter/matvec","max R/tol"

   "none","prepared","scaled-poisson-2d-16","69.5534 [68.4905–70.6164]","137.2447 [130.1899–144.2994]","N:575/588; T:574/587","0.949605"
   "none","prepared","scaled-poisson-2d-32","185.0965 [182.2070–187.9860]","370.3119 [358.9713–381.6525]","1498/1529","0.984252"
   "none","prepared","scaled-poisson-3d-6","33.0213 [31.2609–34.7816]","60.2981 [58.4771–62.1191]","N:261/268; T:249/255","0.998645"
   "none","prepared","scaled-poisson-3d-10","54.4952 [54.4154–54.5750]","104.8242 [102.4973–107.1510]","444/454","0.974858"
   "none","prepared","poisson-2d-16-neutral","6.2013 [5.8387–6.5639]","11.9717 [11.1624–12.7809]","50/52","0.764204"
   "none","api","scaled-poisson-2d-16","69.9795 [66.5822–73.3768]","131.0953 [122.9758–139.2149]","N:575/588; T:574/587","0.949605"
   "none","api","scaled-poisson-2d-32","190.0993 [184.5412–195.6574]","371.9122 [370.8897–372.9348]","1498/1529","0.984252"
   "none","api","scaled-poisson-3d-6","32.2357 [30.6563–33.8150]","62.2525 [61.5884–62.9166]","N:261/268; T:249/255","0.998645"
   "none","api","scaled-poisson-3d-10","51.9963 [51.9131–52.0795]","103.4289 [102.4915–104.3663]","444/454","0.974858"
   "none","api","poisson-2d-16-neutral","6.3670 [5.6834–7.0505]","13.4360 [12.2513–14.6208]","50/52","0.764204"
   "jacobi","prepared","scaled-poisson-2d-16","9.0347 [8.8500–9.2195]","18.8391 [17.0609–20.6173]","48/50","0.911256"
   "jacobi","prepared","scaled-poisson-2d-32","21.2359 [21.0164–21.4555]","39.0692 [38.0208–40.1176]","115/119","0.929037"
   "jacobi","prepared","scaled-poisson-3d-6","4.4503 [4.4227–4.4779]","9.2636 [8.8980–9.6292]","25/27","0.439964"
   "jacobi","prepared","scaled-poisson-3d-10","6.6303 [6.4405–6.8200]","12.6303 [12.0156–13.2451]","37/39","0.969066"
   "jacobi","prepared","poisson-2d-16-neutral","9.0027 [8.5240–9.4814]","18.0454 [17.4063–18.6845]","50/52","0.764204"
   "jacobi","api","scaled-poisson-2d-16","8.6395 [7.9783–9.3007]","17.1381 [16.5369–17.7393]","48/50","0.911256"
   "jacobi","api","scaled-poisson-2d-32","20.9024 [20.5442–21.2607]","40.3409 [37.8065–42.8754]","115/119","0.929037"
   "jacobi","api","scaled-poisson-3d-6","4.4296 [4.2419–4.6173]","9.0008 [8.6184–9.3831]","25/27","0.439964"
   "jacobi","api","scaled-poisson-3d-10","6.9896 [6.0050–7.9743]","13.9987 [11.3864–16.6111]","37/39","0.969066"
   "jacobi","api","poisson-2d-16-neutral","9.7720 [9.7092–9.8347]","18.8995 [16.9554–20.8436]","50/52","0.764204"


CPU — official sparse solver strategies
---------------------------------------

Scaled cases use ``A = D P D`` with positive D spanning 1..20,
plus a constant-diagonal neutral case. These parameters are not exact
condition numbers. Official SciPy and direct strategies can have different
counts/accepted errors; LU can achieve substantially tighter residuals.
For scaled-poisson-2d-32 prepared, the first process records native CG/PCG
residuals 1.45605e-4 / 1.36862e-4 versus LU's 3.44175e-12, all below the
requested 1.47316e-4 threshold. This is not an equal-error comparison.

.. csv-table:: Completed solve milliseconds; process-median range
   :class: benchmark-results
   :header: "M","scope","case","NCG","NPCG","SCG","SPCG","LUT","LUR","iter/matvec","max R/tol"

   "by candidate","prepared","scaled-poisson-2d-16","0.4460 [0.4031–0.4890]","0.0594 [0.0527–0.0662]","2.6482 [2.1932–3.1031]","0.5342 [0.4350–0.6334]","0.3948 [0.3661–0.4236]","0.0379 [0.0333–0.0426]","NCG:555/568; NPCG:48/50; SCG:282/283; SPCG:48/49; LUT:0/1; LUR:0/1","0.980265"
   "by candidate","prepared","scaled-poisson-2d-32","4.1633 [3.9357–4.3908]","0.4109 [0.4022–0.4196]","8.2819 [6.6399–9.9239]","1.3170 [1.2415–1.3925]","1.6149 [1.5384–1.6914]","0.1349 [0.1228–0.1471]","NCG:1498/1529; NPCG:115/119; SCG:609/610; SPCG:92/93; LUT:0/1; LUR:0/1","0.988386"
   "by candidate","prepared","scaled-poisson-3d-6","0.1944 [0.1727–0.2160]","0.0367 [0.0298–0.0436]","1.8500 [1.4986–2.2015]","0.3190 [0.2628–0.3752]","0.4183 [0.3671–0.4695]","0.0377 [0.0263–0.0491]","NCG:252/259; NPCG:25/27; SCG:185/186; SPCG:25/26; LUT:0/1; LUR:0/1","0.978981"
   "by candidate","prepared","scaled-poisson-3d-10","1.4514 [1.3790–1.5238]","0.1794 [0.1641–0.1947]","3.8982 [3.5285–4.2679]","0.6182 [0.5373–0.6991]","3.5389 [3.2384–3.8394]","0.2174 [0.1683–0.2666]","NCG:444/454; NPCG:37/39; SCG:304/305; SPCG:37/38; LUT:0/1; LUR:0/1","0.974337"
   "by candidate","prepared","poisson-2d-16-neutral","0.0446 [0.0427–0.0465]","0.0523 [0.0495–0.0551]","0.4662 [0.4241–0.5083]","0.4915 [0.4309–0.5521]","0.2875 [0.2621–0.3130]","0.0256 [0.0233–0.0280]","NCG:50/52; NPCG:50/52; SCG:50/51; SPCG:50/51; LUT:0/1; LUR:0/1","0.764204"
   "by candidate","api","scaled-poisson-2d-16","0.4252 [0.4200–0.4304]","0.0799 [0.0744–0.0853]","2.2640 [2.2230–2.3051]","0.5028 [0.4730–0.5326]","0.3992 [0.3490–0.4494]","0.0404 [0.0364–0.0444]","NCG:555/568; NPCG:48/50; SCG:282/283; SPCG:48/49; LUT:0/1; LUR:0/1","0.980265"
   "by candidate","api","scaled-poisson-2d-32","3.8981 [3.8690–3.9272]","0.4147 [0.4072–0.4222]","6.8638 [6.7902–6.9373]","1.2605 [1.2374–1.2835]","1.4512 [1.4183–1.4842]","0.1331 [0.1317–0.1345]","NCG:1498/1529; NPCG:115/119; SCG:609/610; SPCG:92/93; LUT:0/1; LUR:0/1","0.988386"
   "by candidate","api","scaled-poisson-3d-6","0.1875 [0.1858–0.1892]","0.0452 [0.0440–0.0464]","1.4909 [1.4623–1.5195]","0.2844 [0.2765–0.2924]","0.3711 [0.3676–0.3747]","0.0295 [0.0293–0.0297]","NCG:252/259; NPCG:25/27; SCG:185/186; SPCG:25/26; LUT:0/1; LUR:0/1","0.978981"
   "by candidate","api","scaled-poisson-3d-10","1.4325 [1.4170–1.4480]","0.2108 [0.1984–0.2233]","3.4814 [3.4745–3.4884]","0.5910 [0.5834–0.5987]","3.4231 [3.3453–3.5008]","0.2587 [0.2238–0.2937]","NCG:444/454; NPCG:37/39; SCG:304/305; SPCG:37/38; LUT:0/1; LUR:0/1","0.974337"
   "by candidate","api","poisson-2d-16-neutral","0.0657 [0.0540–0.0775]","0.0785 [0.0662–0.0909]","0.5676 [0.4295–0.7058]","0.6250 [0.4830–0.7671]","0.3192 [0.2752–0.3631]","0.0335 [0.0260–0.0410]","NCG:50/52; NPCG:50/52; SCG:50/51; SPCG:50/51; LUT:0/1; LUR:0/1","0.764204"


Raw evidence and immutable compiled identity
--------------------------------------------

All raw JSON, logs and actual argv remain local under
``build/performance-cleanup/baseline/``. They are not stored in Git.
The complete 52-command forward/reverse declaration and per-file SHA-256/bytes
are in ``build/performance-cleanup/manifest.json``. Its SHA-256 is:

``b6168caa597ea4a1f15901adbf281b4c738b002bbaf6056ecb9bdbe0800d5c21``.

* ``cpu-release``:
  ``_core.cpython-312-x86_64-linux-gnu.so`` → ``420801aebf8c68bce67f7e343387294159676740ccf5a4036eae7b4ecda1f959``.
* ``cpu-blas``:
  ``_core.cpython-312-x86_64-linux-gnu.so`` → ``af43dccbc20d96c6e98a19ca51717b1d824f3d38029d9d91281986e1f9e2a0db``.
* ``cuda-release``:
  ``_core.cpython-312-x86_64-linux-gnu.so`` → ``f668f6f4ef3671ba656a3e5c175e37506218098ddeea9bcb611c229da34260d4``.
  ``_cuda.cpython-312-x86_64-linux-gnu.so`` → ``2917f821463d3aecdb50c5548f6e122458d52b1180f8d9f3875d8fa53c995820``.

The original staged cgpy/cgbench packages, binaries and CMake caches were
copied before sampling to ``build/performance-cleanup/binaries/<preset>/``.
Copied CMake caches still name the live checkout: when rerunning a saved
binary after source changes, use the manifest's immutable compiled revision,
rather than relabeling it with the live checkout HEAD. Binary hashes identify
what was actually loaded; rebuilds on another host need not have identical bytes.


Reproduce
---------

Use CPython 3.12, a compatible NVIDIA driver and existing GNU C++20 toolchains.
For the original baseline, check out the full measured source revision above
before the builds. These commands use project-local dependencies and toolkit.
Choose compatible architecture/affinity/compiler values on another host; keep
that changed environment visible in its JSON.

.. code-block:: bash

   python3.12 -m venv .venv
   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
   python3.12 -m venv .venv-cuda
   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip -r requirements/cuda-benchmark.lock
   .venv-cuda/bin/python scripts/setup_cuda.py --prefix .deps/cuda-12.8
   make build PRESET=cpu-release JOBS=2
   make build PRESET=cpu-blas JOBS=2
   make build PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" JOBS=2 \
       CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75

The following standalone driver reproduces all 52 fixed profiles. It stops
on a failed command. Inspect every saved candidate status, true residual,
count, input hash and binary identity before using a performance comparison.

.. code-block:: python

   from pathlib import Path
   import subprocess

   root = Path.cwd()
   groups = [
       ("cpu-release", "cpu-dense-f64-v1", "native,numpy-out,torch-eager-out", "cpu", 3),
       ("cpu-release", "cpu-csr-f64-v1", "native,scipy-out,torch-eager-out", "cpu", 1),
       ("cpu-blas", "cpu-dense-f64-v1", "native,native-blas,numpy-out,torch-eager-out", "cpu", 3),
       ("cuda-release", "cuda-dense-f64-v1", "cuda-native,cuda-cublas,torch-cuda-eager", "cuda", 3),
       ("cuda-release", "cuda-csr-f64-v1", "cuda-native,torch-cuda-eager", "cuda", 1),
       ("cuda-release", "sparse-solvers", "cuda-native,torch-cuda-eager", "cuda", 1),
       ("cpu-release", "sparse-solvers", "native-cg,native-pcg,scipy-cg,scipy-pcg,scipy-lu-total,scipy-lu-reuse", "cpu", 1),
   ]
   configs = [
       (*group, m, scope)
       for group in groups
       for m in (("none",) if group[0] == "cpu-release" and group[1] == "sparse-solvers" else ("none", "jacobi"))
       for scope in ("prepared", "api")
   ]
   for pass_id in (1, 2):
       for preset, suite, candidates, device, repetitions, m, scope in (configs if pass_id == 1 else reversed(configs)):
           python = root / (".venv-cuda" if device == "cuda" else ".venv") / "bin/python"
           name = f"{preset}-{suite}-{m}-{scope}-r{pass_id:02}"
           output = root / "build/performance-cleanup/reproduced" / f"{name}.json"
           flags = (
               f"--implementations {candidates} --seed 20261015 --warmups 3 "
               f"--algorithm standard --fused --preconditioner {m} "
               "--rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50"
           )
           if device == "cuda":
               flags += " --device cuda --cuda-device 0"
           command = [
               "taskset", "-c", "0", "make", "pyprof", f"PRESET={preset}",
               f"PYTHON={python}", "JOBS=2", f"PYPROF_SUITE={suite}",
               "PYPROF_THREADS=1", f"PYPROF_SCOPE={scope}", "PYPROF_ROUNDS=30",
               f"PYPROF_REPETITIONS={repetitions}", f"PYPROF_OUTPUT={output}",
               f"PYPROF_ARGS={flags}",
           ]
           if device == "cuda":
               command += ["CUDA_HOST_COMPILER=/usr/bin/g++-13", "CUDA_ARCHITECTURES=75"]
           subprocess.run(command, check=True)

Replay an original local artifact with its saved fixture hashes, candidate
set and actual sampling schedule. The new environment/binary identity is
recorded; identical latency is not expected. Replay is a separate validation
and is not pooled into baseline timings.

.. code-block:: bash

   taskset -c 0 env PYTHONPATH="$PWD/build/cuda-release/python" \
       .venv-cuda/bin/python -m cgbench replay \
       build/performance-cleanup/baseline/cuda-release-sparse-solvers-jacobi-prepared-r01.json \
       --output build/performance-cleanup/replayed-scaled-pcg.json

For later cleanup regression, keep both original process files and compare
matching new files with ``scripts/compare_performance.py``; its strict
input/options/count gates must remain intact. A change in reduction order
can legitimately change iterations: disclose those time-to-tolerance cases
separately rather than weakening the equal-work gate. CUB adoption needs its
own immediate pre/post-change baseline, correctness evidence and predeclared
uncertainty/regression budget. No automatic crossover or backend mode is inferred.
