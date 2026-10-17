Current-main CPU GEMV diagnostics — 2026-10-17
==============================================

Both measured component passes passed their independent numerical checks. The
CPU scheduling candidate was rejected by the separate whole-solve gate; these
descriptive operation timings do not describe an adopted production change.
See :doc:`main-optimization-20261017` for the decision and complete controls.

Across all 78 descriptive pairs, the candidate has 32 lower, 36 higher and
10 equal medians. The 24 official-fixture pairs split 12 lower and 12 higher.
These single-process comparisons do not establish an overall improvement.

Portable CPU paired-row code generation
---------------------------------------

Baseline source: ``445275a3994e342b06de4093889f3f814c8706c6``.
Candidate source: ``5d0367f7ed06fbfe3f7ebabc28d6b110b2392eb2``.
These are existing GNU 16.0.1 CPU-release diagnostic objects, not timing results.
Portable flags: ``-O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic``.
Original evidence under ``build/main-performance-20261017/proposals/``:
``dense-baseline.asm``, ``dense-vectorization.txt``, ``dense-vectorization.o``,
``dense-vectorization-command.json``;
``dense-candidate.asm``, ``dense-candidate-vectorization.log``, ``dense-candidate-vectorization.o``.

.. csv-table:: First row_pair_dot hot-loop bodies
   :header-rows: 1

   Observation,Baseline,Candidate
   Address range,0x40 through 0x8e,0x60 through 0xf0
   Columns per backedge,4,8
   Instructions per backedge,18,34
   Instructions normalized to eight columns,36,34
   Packed loads per eight columns,12,12
   Packed multiply/add instructions per eight columns,8/8,8/8
   SIMD width,128-bit SSE2,128-bit SSE2
   Independent packed accumulator chains,4,4
   XMM registers used within this hot body,7,9
   Floating-point spills within this hot body,None,None

The candidate has one compare/conditional backedge per eight columns versus two.
Both already share x loads between the same two rows; bytes loaded per column
and arithmetic counts do not decrease. Neither body introduces AVX or FMA.
Candidate setup adds an unconditional 24-byte frame saving/restoring rbx/rbp;
these are callee-saved general registers, not floating-point accumulator spills.
Its four-column remainder conditionally saves r12; power-of-two fixtures bypass
that remainder. The paired function's span to the next symbol, including padding,
grows from 0x1e0 to 0x2e0. Its final horizontal reduction uses packed shuffles/adds.
The compiler reports 16-byte vectors and main-loop unroll factor one for both:
baseline source line65, candidate line68. Eight columns come from source unrolling;
candidate line83 is the four-column remainder and line94 is a separately vectorized
tail (compiler factor two). Do not mistake that tail report for the main loop.

Reproduction: root must first select separate checkouts at the exact commits above
and the same GNU16 compiler. Set the two source variables below to those directories.
Use fresh output paths; these commands must not replace the original evidence.

.. code-block:: bash

   CG_BASE_SOURCE=/absolute/baseline445/source-checkout
   CG_CAND_SOURCE=/absolute/candidate5d0367/source-checkout
   CG_CODEGEN_OUT="$PWD/build/main-performance-20261017/proposals"
   /usr/bin/c++ -I"$CG_BASE_SOURCE/cpp" -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -fopt-info-vec-all="$CG_CODEGEN_OUT/baseline-reproduce-vectorization.log" -c "$CG_BASE_SOURCE/cpp/cg/backend/cpu/dense.cpp" -o "$CG_CODEGEN_OUT/baseline-reproduce.o"
   /usr/bin/objdump -d -C -w "$CG_CODEGEN_OUT/baseline-reproduce.o" > "$CG_CODEGEN_OUT/baseline-reproduce.asm"
   /usr/bin/c++ -I"$CG_CAND_SOURCE/cpp" -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -fopt-info-vec-all="$CG_CODEGEN_OUT/candidate-reproduce-vectorization.log" -c "$CG_CAND_SOURCE/cpp/cg/backend/cpu/dense.cpp" -o "$CG_CODEGEN_OUT/candidate-reproduce.o"
   /usr/bin/objdump -d -C "$CG_CODEGEN_OUT/candidate-reproduce.o" > "$CG_CODEGEN_OUT/candidate-reproduce.asm"

The flags and observations cover these diagnostic objects; other compilers,
OpenMP outlining and final linked binaries require their own inspection.
Fewer backedges coexist with changed register use, setup and code size: no
causal wall-time, bandwidth, cache-residency or whole-CG speedup follows.

CPU component observations
--------------------------

All 78 matched pairs and 156 observations are retained. Baseline and candidate are separate sequential passes, with three warmups and 36 samples per process. Official fixtures use the fixed vector b; padded integer probes are separate operation diagnostics. These are single-process descriptive results, not framework scores, confidence intervals, hardware bandwidth proofs, or additive whole-CG phase attribution. Higher medians remain visible. Adoption requires separate whole-solve evidence.

baseline: compiled revision ``445275a3994e342b06de4093889f3f814c8706c6``, tree ``5d7ba07832dc7576810ddd5d07d66e4b87ef8109``.

candidate: compiled revision ``5d0367f7ed06fbfe3f7ebabc28d6b110b2392eb2``, tree ``896ff8ca497514087550eeefc0be320906297334``.

Each successful frozen C++ source checks independent long-double vector/scalar bounds before and after timing; JSON error fields are postcheck only; renderer audits those fields and process/source evidence, not a recomputed oracle.

All matched medians and sample ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1

   Case,Fused,Operation,Baseline median ns,Baseline min ns,Baseline max ns,Candidate median ns,Candidate min ns,Candidate max ns,Candidate/baseline
   dense-64-fused0-apply,False,apply,406,376,594,364,360,499,0.896551724
   dense-64-fused0-apply_and_dot,False,apply_and_dot,429,425,625,380,378,467,0.885780886
   dense-64-fused0-residual_squared_norm,False,residual_squared_norm,440,436,552,400,397,11684,0.909090909
   dense-64-fused1-apply,True,apply,350.5,346,440,361,353,495,1.0299572
   dense-64-fused1-apply_and_dot,True,apply_and_dot,392,385,487,349,346,450,0.890306122
   dense-64-fused1-residual_squared_norm,True,residual_squared_norm,388.5,383,482,380,374,526,0.978120978
   dense-256-fused0-apply,False,apply,4624,4603,36360,4622,4589,14808,0.999567474
   dense-256-fused0-apply_and_dot,False,apply_and_dot,4715.5,4697,33886,4735,4680,81531,1.0041353
   dense-256-fused0-residual_squared_norm,False,residual_squared_norm,4763,4744,40826,4787,4730,21142,1.00503884
   dense-256-fused1-apply,True,apply,5438.5,4610,20501,4622,4588,21628,0.849866691
   dense-256-fused1-apply_and_dot,True,apply_and_dot,4684,4649,30229,4723.5,4707,5081,1.00843296
   dense-256-fused1-residual_squared_norm,True,residual_squared_norm,4738,4699,23494,4792,4770,8469,1.01139721
   dense-1024-fused0-apply,False,apply,169378.5,141085,193970,173292.5,133422,215233,1.02310801
   dense-1024-fused0-apply_and_dot,False,apply_and_dot,164687.5,150587,200365,136754,96798,222152,0.83038482
   dense-1024-fused0-residual_squared_norm,False,residual_squared_norm,162899,138695,244195,180872.5,136983,216693,1.11033524
   dense-1024-fused1-apply,True,apply,187490,162578,221519,176141.5,137590,208321,0.939471438
   dense-1024-fused1-apply_and_dot,True,apply_and_dot,165487,137533,187219,182191,139442,214221,1.10093844
   dense-1024-fused1-residual_squared_norm,True,residual_squared_norm,156972,139532,189261,175534.5,137953,217153,1.11825357
   dense-2048-fused0-apply,False,apply,1129076,1016634,1414130,1276765,986300,1555526,1.13080519
   dense-2048-fused0-apply_and_dot,False,apply_and_dot,1761777,1114477,5154043,1321025.5,991971,1809622,0.749825602
   dense-2048-fused0-residual_squared_norm,False,residual_squared_norm,1359908,955583,1759305,1268780.5,911372,1573365,0.932989952
   dense-2048-fused1-apply,True,apply,1247716,918952,1399385,1332255,936435,1692714,1.067755
   dense-2048-fused1-apply_and_dot,True,apply_and_dot,1211743,956570,1381300,1138600.5,921178,1300627,0.939638603
   dense-2048-fused1-residual_squared_norm,True,residual_squared_norm,1053714.5,810277,1165593,1324562.5,1089189,1484014,1.25704116
   integer-tail-7-fused0-apply,False,apply,43,40,147,42,41,113,0.976744186
   integer-tail-7-fused0-apply_and_dot,False,apply_and_dot,59,57,131,58.5,57,128,0.991525424
   integer-tail-7-fused0-residual_squared_norm,False,residual_squared_norm,69.5,68,153,68,67,154,0.978417266
   integer-tail-7-fused1-apply,True,apply,41,40,112,41,40,103,1
   integer-tail-7-fused1-apply_and_dot,True,apply_and_dot,44,43,112,44,43,198,1
   integer-tail-7-fused1-residual_squared_norm,True,residual_squared_norm,49,47,11202,49,48,116,1
   integer-tail-8-fused0-apply,False,apply,39,38,95,40,39,101,1.02564103
   integer-tail-8-fused0-apply_and_dot,False,apply_and_dot,57,56,144,57,55,131,1
   integer-tail-8-fused0-residual_squared_norm,False,residual_squared_norm,71.5,68,10242,68.5,66,152,0.958041958
   integer-tail-8-fused1-apply,True,apply,40,39,93,39,38,103,0.975
   integer-tail-8-fused1-apply_and_dot,True,apply_and_dot,43,42,105,43,42,123,1
   integer-tail-8-fused1-residual_squared_norm,True,residual_squared_norm,47,46,118,47,46,112,1
   integer-tail-9-fused0-apply,False,apply,42,41,108,43,42,154,1.02380952
   integer-tail-9-fused0-apply_and_dot,False,apply_and_dot,61,59,130,59,58,151,0.967213115
   integer-tail-9-fused0-residual_squared_norm,False,residual_squared_norm,72,71,151,71,70,10174,0.986111111
   integer-tail-9-fused1-apply,True,apply,43,42,106,42,41,104,0.976744186
   integer-tail-9-fused1-apply_and_dot,True,apply_and_dot,46,45,125,46,44,108,1
   integer-tail-9-fused1-residual_squared_norm,True,residual_squared_norm,51,50,115,51,50,172,1
   integer-tail-15-fused0-apply,False,apply,62,61,133,66,65,143,1.06451613
   integer-tail-15-fused0-apply_and_dot,False,apply_and_dot,81,80,167,87,84,168,1.07407407
   integer-tail-15-fused0-residual_squared_norm,False,residual_squared_norm,97,95,198,97,96,196,1
   integer-tail-15-fused1-apply,True,apply,62,61,129,67,66,141,1.08064516
   integer-tail-15-fused1-apply_and_dot,True,apply_and_dot,66,65,128,70,69,144,1.06060606
   integer-tail-15-fused1-residual_squared_norm,True,residual_squared_norm,71,69,142,75.5,74,196,1.06338028
   integer-tail-16-fused0-apply,False,apply,58,57,138,85,64,229,1.46551724
   integer-tail-16-fused0-apply_and_dot,False,apply_and_dot,80,77,164,79.5,77,201,0.99375
   integer-tail-16-fused0-residual_squared_norm,False,residual_squared_norm,99,94,240,89.5,88,179,0.904040404
   integer-tail-16-fused1-apply,True,apply,57,56,137,61,59,154,1.07017544
   integer-tail-16-fused1-apply_and_dot,True,apply_and_dot,61,59,132,64,62,136,1.04918033
   integer-tail-16-fused1-residual_squared_norm,True,residual_squared_norm,67,65,140,69,68,149,1.02985075
   integer-tail-17-fused0-apply,False,apply,62,60,135,64,63,151,1.03225806
   integer-tail-17-fused0-apply_and_dot,False,apply_and_dot,85,82,11360,91,88,186,1.07058824
   integer-tail-17-fused0-residual_squared_norm,False,residual_squared_norm,101,99,214,98,94,199,0.97029703
   integer-tail-17-fused1-apply,True,apply,65,63,143,67,65,150,1.03076923
   integer-tail-17-fused1-apply_and_dot,True,apply_and_dot,68,66,147,68,66,141,1
   integer-tail-17-fused1-residual_squared_norm,True,residual_squared_norm,71,70,9550,74,73,151,1.04225352
   integer-tail-129-fused0-apply,False,apply,1437,1423,1550,1287.5,1282,1365,0.895963814
   integer-tail-129-fused0-apply_and_dot,False,apply_and_dot,1302,1295,11666,1344,1335,1432,1.03225806
   integer-tail-129-fused0-residual_squared_norm,False,residual_squared_norm,1574,1565,21894,1626,1372,2522,1.03303685
   integer-tail-129-fused1-apply,True,apply,1461,1450,1555,1290,1282,2436,0.882956879
   integer-tail-129-fused1-apply_and_dot,True,apply_and_dot,1522,1512,1619,1579,1332,2559,1.03745072
   integer-tail-129-fused1-residual_squared_norm,True,residual_squared_norm,1314.5,1303,1395,1381.5,1373,1456,1.05096995
   integer-tail-130-fused0-apply,False,apply,1349,1330,11543,1385,1371,2676,1.02668643
   integer-tail-130-fused0-apply_and_dot,False,apply_and_dot,1694.5,1678,1776,1461.5,1450,11697,0.862496312
   integer-tail-130-fused0-residual_squared_norm,False,residual_squared_norm,1724.5,1460,12049,1487,1473,11692,0.862278921
   integer-tail-130-fused1-apply,True,apply,1322.5,1308,11394,1392.5,1380,23581,1.05293006
   integer-tail-130-fused1-apply_and_dot,True,apply_and_dot,1636.5,1621,13411,1449.5,1424,14462,0.885731745
   integer-tail-130-fused1-residual_squared_norm,True,residual_squared_norm,1664,1645,14357,1472.5,1462,1554,0.884915865
   integer-tail-131-fused0-apply,False,apply,1534.5,1296,12480,1615,1584,13336,1.05246008
   integer-tail-131-fused0-apply_and_dot,False,apply_and_dot,1608.5,1592,1823,1688,1672,1771,1.04942493
   integer-tail-131-fused0-residual_squared_norm,False,residual_squared_norm,1632.5,1620,1803,1722.5,1448,13274,1.05513017
   integer-tail-131-fused1-apply,True,apply,1535,1295,12167,1277,1264,11233,0.831921824
   integer-tail-131-fused1-apply_and_dot,True,apply_and_dot,1404,1393,10978,1340,1320,10590,0.954415954
   integer-tail-131-fused1-residual_squared_norm,True,residual_squared_norm,1702,1441,12573,1635,1620,12328,0.960634548

All postcheck oracle observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1

   Side,Case,Vector error,Vector bound,Scalar error,Scalar bound,Raw SHA256,Execution sidecar SHA256
   baseline,dense-64-fused0-apply,1.0196288258157438e-12,2.0618619623662891e-10,0,0,2e241b88706f7bffe08f6fb295d0689d0461a0b922598854142f6b7e6c91a7cf,abe98cdd9d97c7b11cb8f17eb1e34e2adb4565efbe24eeb297ebb212711ef820
   candidate,dense-64-fused0-apply,1.0196288258157438e-12,2.0618619623662891e-10,0,0,4652b769c67b3797e7608c8edd56bd3593aac4402e4d1b85f5681f1098c56f14,5f57a358569a0e7d574fa83c9c707d9c0334aacb65fdd2507a505794efa82040
   baseline,dense-64-fused0-apply_and_dot,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,b7a050a6c8e00c83e294566efd4ba0138bdd74f95040ff95a521e7cbb122332d,be002b3ba0f49bf666a1beaf979104f6653c00329f8a8d5d26641f13b231d87c
   candidate,dense-64-fused0-apply_and_dot,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,e0c25b7b1e0b9eed5c963347d4f726dccf973370034e22fb95b8d01f7802fa50,55b784d383bd86f2d12ead72d69625ae9a97cdd3830a0c77430418616a2b0767
   baseline,dense-64-fused0-residual_squared_norm,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,9f5a2f8734fffdd7418fcd7e27b2d24ee3add332120db8067b581fa6699bec50,e7927914294055055ba341b252331caa7ccd578fd46ecc2fad9997245031a9fe
   candidate,dense-64-fused0-residual_squared_norm,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,4239e8d7767735a5feb17569b32006695806e09dfb2ce72c020c080906a19bb1,78d266a90c2929ab25891dab40c6bcff9f7604479ab24d3d51c65f3ae1acdebc
   baseline,dense-64-fused1-apply,1.0196288258157438e-12,2.0618619623662891e-10,0,0,e8904e3bad1d1da6d48f322a1b28ef6905fe1cb6ff7e1e53a06b5442cc382158,875f84a07b561a083d93fc61dbf000c66056a20b7d47710b25308cbad8021e9b
   candidate,dense-64-fused1-apply,1.0196288258157438e-12,2.0618619623662891e-10,0,0,c5f9dd592bfba0919fd78739ad62a935b7a0acfd6bb3e553b108a422261d67ba,8a55142dba1b3ef197a1f7b3c7c957c25c266b6d0f6a29d22a09b4803e1b6f7e
   baseline,dense-64-fused1-apply_and_dot,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,2a983bcc5c445e9d9dd85a1c925f2f6b7435c90e7f14c2e584cfaf3bf2765f13,005651ee41e62de36b1fe284720750094746cb625b0e6fbf2dbaa4a597a145a6
   candidate,dense-64-fused1-apply_and_dot,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,5ec3577cdcdcbab4c7bbc8e099550021a86026a0fb1965a762cce8195e76c935,85cd5e658cf23836613beda8c2622c4f8b52c005f0a53b2aa58787ab7f425831
   baseline,dense-64-fused1-residual_squared_norm,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,89fa84b04cc305bbf097831c6dba1711ce3b5211bc4fd59685c232fd2c715014,f5956153364ea12d7630ec3e0fb96cbab8b51e789dd00661ebf604255f3b7f0d
   candidate,dense-64-fused1-residual_squared_norm,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,738e0965b54baad3ffe2d1eade2f93af3070ae078a0021808f20e6c78daff542,cd6d8d2c52306119f93cd68af1f274a028f7274864a1d734a13b2ed342ee71db
   baseline,dense-256-fused0-apply,2.1898038937706588e-12,8.7769074042863557e-10,0,0,6224d1684341d58a448bbaac897d3f5bf4e1417b79555241d98f589efa3bf7af,4ca120116cf6dc2bdcb7a475291127b3204f908ce54f60d2efbc1736756ef509
   candidate,dense-256-fused0-apply,2.1898038937706588e-12,8.7769074042863557e-10,0,0,2aaa97341a7387a3fdaf2e157d88d357cd99017d17d39ef5e8e96f7755a4147f,3caa1f14afc46a8e667170dc1dd838fdf26be58ef569b221a5ddc69d38923b2b
   baseline,dense-256-fused0-apply_and_dot,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,fa5c285f81af05c603693862fffba5f0cb2d6b84e184affcd26a21aa0a919497,1be69f5903af0fe2fba27b8f7a1cef3aafead84a3d7dc11cbe76e223458234ab
   candidate,dense-256-fused0-apply_and_dot,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,d06335f6e403ebc1c10e572b27b25a07b44d7f82165b7a1174a02fd83cf27a1f,c6b37c81171ac44154b84764b538682bde25b9ae79664bda739d68aed9cacf4c
   baseline,dense-256-fused0-residual_squared_norm,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,c92853afbe9263e4e6aa08e552fa9bc0dc19e8e3ba0432b1cf3630fe8bda2cb3,0ffbf79336832de895f693834af3d35878d50af5ccbc51c6ca49eabb624e5349
   candidate,dense-256-fused0-residual_squared_norm,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,4bb1daf583033a579e6b4eaa32fd51c6d6468759185178a44134fb1b1bf16f37,cc9b797369f20314b46205a445c709d5bd438ceb53cf5fd171c84d3e10b16a3b
   baseline,dense-256-fused1-apply,2.1898038937706588e-12,8.7769074042863557e-10,0,0,41288a798fb82eaf0d8d2fdb9ccefa6752eda941353b27dd2dfec06888c359e7,82be3d640ef44fe1e87f6cc724823f8d18f1a7f5a0a3966d768b9caac29fa53e
   candidate,dense-256-fused1-apply,2.1898038937706588e-12,8.7769074042863557e-10,0,0,0c8a8b6476e86f057954ac27c81515ebb9b6c17833a8f108d81a266ddd24d3ee,8f8cf6a2443fa90a1b9d7016f22d39bb8366de22251c671655546763078d9ee9
   baseline,dense-256-fused1-apply_and_dot,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,53595014a2a2783e823b47cf93cf7509f1ead85113a5e81c9dcefa2410fb6d1f,34166c0cb12f5ab1ccf772fe7854708d1108160c07feefa455501d2100b4dea3
   candidate,dense-256-fused1-apply_and_dot,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,5f8d9593c467d4cf02f2b3ef54d428cc0638af7f2200d97b3e45dee727482c72,60416a9ae7302926cc228d9f2d0add2971c52b490702d22703e5b8cc7234bfa2
   baseline,dense-256-fused1-residual_squared_norm,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,9a310fffdb46f2d17ad12c0ab665da4650f237a04e86d32a8381c0360088533e,45866ffc88ff81fe78b7c44afc5e74c0277b25a6a6388f1d8af2554b32be71a4
   candidate,dense-256-fused1-residual_squared_norm,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,867d7a6ec7f4f2390c96351eb00f81b92b437abc6043a4985470090dcae1dc87,a6b89010c4ec9c8044ce95d1eaccede56cc8d24aee9c2383cbd5ce90b685ef27
   baseline,dense-1024-fused0-apply,4.0416558988454199e-12,4.1708565652771602e-09,0,0,aba0df45a4afbe4485472edf0be608397b7518b9d15a577607a70c41f928889d,401de46bf9de2f451815e3d887de0119aa3b98c1e69f0a53dfd839c6f4ada099
   candidate,dense-1024-fused0-apply,4.0416558988454199e-12,4.1708565652771602e-09,0,0,352d7e3b646f91930300d9b68f5f00e69af524ff24c7a134675fec5e411d617e,7feff1930c70b66153920c8b99032ad3e26ceb7f22ff86bf67e26e37fe108533
   baseline,dense-1024-fused0-apply_and_dot,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,853f6e8c28d93b58d08ce8c64e79faa8525c03994348a812def9636a6f2a0939,de9fd4abe78fe8821fd1ab13f9a0b142fd70c68c0f5bdaed5a86eab3f4ff81e1
   candidate,dense-1024-fused0-apply_and_dot,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,3eeb2052e8d41eed172d034562aaa59518abbbb5df35529959e7a5ffca2b40c1,5ec6f7f51f0e00532fdac359da987eb02d0c6bcbc81efe6eaee476c740badcad
   baseline,dense-1024-fused0-residual_squared_norm,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,904fac36ac95a0a032e5164010d8df686de5c4af7b4afc401ff26ab8e8d4cd2b,7a4be32f3ff2dee7c1eee30e5891d255bac8738e997b251793669f0a1b40e077
   candidate,dense-1024-fused0-residual_squared_norm,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,d2542ac1b652920a3209f516a3ed83f718b0382dd99baf2744f3e1efe8c03c96,f8cf4819d9a327145b173c62817f48d6b408c9d8da57132039355b5a45493217
   baseline,dense-1024-fused1-apply,4.0416558988454199e-12,4.1708565652771602e-09,0,0,084732f55edfbe063de52d54448c9f981505c57cc9feaa5fe6e31204c8c25fb4,b44339b6159e46d6e120a89af937de4d8fa360ab77288cbb998ad3d697fa6b8c
   candidate,dense-1024-fused1-apply,4.0416558988454199e-12,4.1708565652771602e-09,0,0,a96aece7cb11c1fc9784f788e1f852c6ff31c47185fb69d628257952280908b7,bf895afc256d6b007af9fbf672f3ca1f00aa517ba50afdbd3f43ed869a07bd5b
   baseline,dense-1024-fused1-apply_and_dot,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,c3ebbc3ccd600fa69e5268e27c7aaca3fe96a664e1a0b44851ce54520deb14c2,abd055fe409d4c5977a2fd86606fcff892d4455bc2324f6cd655dea15251e277
   candidate,dense-1024-fused1-apply_and_dot,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,abf278f5e5028ebb0aa0fec1b57031987980ae99ad42295baaf17874c9907d5b,8c44dec2ba8620f4ce270d2c62aa549d587961c28d95339d61d6f321117d019e
   baseline,dense-1024-fused1-residual_squared_norm,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,ee2c42a33513d902a22045e8a91c18486450e01b740c5d6840499f7dcf7e1e5e,9364e2ede5e2a9b4fe12c4ab5a6f16d61fd57ba308c639678e186ea7da109694
   candidate,dense-1024-fused1-residual_squared_norm,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,7f6a45bce3043342a7d9431b811264dcfd2ac91fdac591d4e2f4e3aab8e7eac1,8e667b55e4057490ce0d463287e55dc1fdde5f1dac54b168a34143044a01b3c8
   baseline,dense-2048-fused0-apply,6.0809135504769074e-12,1.1096243113407423e-08,0,0,0b174f016fb8e7a7d4e1dfff41538bac7ce8e6ce00ebbec0aa70e285af82ae9f,68c32303d5da0fb797dbe042c19d3b96391fcd9e4c75f8b32e1decebb590077f
   candidate,dense-2048-fused0-apply,6.0809135504769074e-12,1.1096243113407423e-08,0,0,a36aee6965b91b9e729769de947960c892fce5846375287b2dfedfc67f7f018a,ef50fb0ade76a5363260af02049ca347bfd8d00cdd4b3e5ad66f335dd5da0dba
   baseline,dense-2048-fused0-apply_and_dot,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,ceeb24225d4afd73e024730af44e6dab5b80cbf3f1957569d00390cb59542da4,28916b907b71eacc6a020e68f89a0c477992a5fea01a0ee5c1d5f6ab4f87c1ee
   candidate,dense-2048-fused0-apply_and_dot,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,39f2fc0613d5ac2dd76fc7071962107019e42c53bceb1b0cf918eb30c5c4605e,0e2a4acd4efe385ab5c036f1c1cbf6318f5c1f4e170a60f8780951e3ab121b51
   baseline,dense-2048-fused0-residual_squared_norm,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,0f8665f727ca0a454b89c0a7eedb68780e711025c0fc73bf0aab8acfd7d0693c,47f487fc02b855b650e6c100eb2d3e1b4e86a6a0d8f5964eff0631ee41ddb58a
   candidate,dense-2048-fused0-residual_squared_norm,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,8fe18473cb9cc01455edc9df119f7d68552cb4ef9c87da47aab48037d1c84be9,653be97c70b008be1ca9d44f8168efd37ee23d044761e0fc4330750ed465659d
   baseline,dense-2048-fused1-apply,6.0809135504769074e-12,1.1096243113407423e-08,0,0,4e75add623d055ce49f5c23ff69a07f147ae0bff394f268e425f4dab0bf0db12,9e465ad66e6a9241fdef6602149a0cb6e85322f11bc15035b6684ee8a48e459c
   candidate,dense-2048-fused1-apply,6.0809135504769074e-12,1.1096243113407423e-08,0,0,88538a9b566580418329e4134c4239b00b685b71def9f8c49192a84685aec121,c344b92a25ea92485d93cbc7cef95387c00b74661190bb908a2e3adb03538a61
   baseline,dense-2048-fused1-apply_and_dot,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,79cda406dc8260d32d9187d2d69d1b926b75b3246f42db59045b82cdf4802459,35cbee53ced843a87d4fda9d9ef6d2d1695f2cf3dcb1244ffa10d003a7d45065
   candidate,dense-2048-fused1-apply_and_dot,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,c604ef224dd91ca8d38660278167927e1f091054320fd4e5a19e4af918bad79f,67f5833e1e422ee5f0fa50cdbc0521e74046619d5f9f006b940179e767aae65d
   baseline,dense-2048-fused1-residual_squared_norm,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,f0147ee3fadd8083948dd358ec8764044da40c679177857199c92f9982cb86d7,a851aeb91477583718427f95da0eb597153b3bb24d56296254590f0afc88d523
   candidate,dense-2048-fused1-residual_squared_norm,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,878fded8479de342fabef81d0a5839bf99418f5a4a3b9e509826f86e967ebbc1,2a8cefcc6fdb3fe8b77d742be20f16c382171e2ac1926f7bce350d43826f5dad
   baseline,integer-tail-7-fused0-apply,0,5.6621374255883173e-14,0,0,253435b5baac37f71ce6ae24abf24368cb79004e92e15d1d4351eee129f1db65,a083184f8500e2e2fa964373565dc4eb669e8ca1e48c0c4b170340a467a72a33
   candidate,integer-tail-7-fused0-apply,0,5.6621374255883173e-14,0,0,cc4109997d632feed7410a6c26792a73c0fea0b818ef2c9d876e8252453deea1,456263373a1652278b51565c0232f8cd13ac207be34486735fbe81609cfd414a
   baseline,integer-tail-7-fused0-apply_and_dot,0,5.6621374255883173e-14,0,6.6613381477509776e-13,bdf98da3892195f6194052778756e5812916084d88a974bd8b19b958b52bb3cb,f50e82e5f57005c9fcceb5f59167fa5138a30bfd936de1c6872f082a84c725c2
   candidate,integer-tail-7-fused0-apply_and_dot,0,5.6621374255883173e-14,0,6.6613381477509776e-13,67bc5931ef219f4e040e5746096457f535f039ffa4323eb86d6c0052733bdfb5,08b181dca70a2d87ed18a371959efd6730fba17ebcea805f174b3deb9adbfcab
   baseline,integer-tail-7-fused0-residual_squared_norm,0,5.7509552675583311e-14,0,7.4575901010121645e-12,89cc665fdc3cacdfdbc538c07ac37b2767927b7bf39ca045ebd68856361ab47b,edc04674587fc552736ed125e891c334a7e60db3120bb3237ab9050a58109977
   candidate,integer-tail-7-fused0-residual_squared_norm,0,5.7509552675583311e-14,0,7.4575901010121645e-12,75a23a4d0d7f6a1049209be7212599d9e59d464db6668fef741bd8534f7c811c,428e0fa7d7be7019454f4fc297eca1b26dd604f1047a50d248bb96a5787d276a
   baseline,integer-tail-7-fused1-apply,0,5.6621374255883173e-14,0,0,cd33ff31502524ff5d6f86b70f9a22a50a53e906e6edcd2ea6df8a08362193b0,ae00dea42459b087333786eb201b486b65ba4ff8511e7f3a6f2be401b25206ae
   candidate,integer-tail-7-fused1-apply,0,5.6621374255883173e-14,0,0,7a8ac6ae51dd704f5d9241db227cba7017c98e47eeb789507e63e75bde0cc0cb,4e001726607fb658551174dd55742c7da29bbaffd757926b490def4e69cc6168
   baseline,integer-tail-7-fused1-apply_and_dot,0,5.6621374255883173e-14,0,6.6613381477509776e-13,d5706fdfb37dea9da267aeaf732d0f85c39007c91f529e6d17d760baf2dc280b,75999de1e0944196ea3793dda26ca64e36c56256970109e1d10781879138af08
   candidate,integer-tail-7-fused1-apply_and_dot,0,5.6621374255883173e-14,0,6.6613381477509776e-13,579442eaa39ffed158dafc0db33f5ddc149a46d5902e9eac4064f33eb7489e38,efcbfefd70bc32a7b39eb990f136642b4af84bd3bf97f9e1adcd08e6e319b20e
   baseline,integer-tail-7-fused1-residual_squared_norm,0,5.7509552675583311e-14,0,7.4575901010121645e-12,16d585104f11db7b8be67aeb4a0c6882a685f6fc460d2b3c19269bbf7302fdbf,ecb85b68f0370246034125ffec9ddc5ae1ae38edf78fc8eca06dedded571388b
   candidate,integer-tail-7-fused1-residual_squared_norm,0,5.7509552675583311e-14,0,7.4575901010121645e-12,aab932fe481c667116a903b7c1290221e81a33f2e4be95faffcb804b1c2c8ad0,4b4ba94e608ea9a541b55e8938b4ae6383b8970976b14f0ed3dd03db422480ff
   baseline,integer-tail-8-fused0-apply,0,6.039613253960873e-14,0,0,5fe5b18bf4f7aa7e6acc5855a343899cc882d0738606677bb47031afaae8e3b6,95bcb218ce98a45856557519ff72aa2882d2bb8f1414fe6619ae96fd9f636731
   candidate,integer-tail-8-fused0-apply,0,6.039613253960873e-14,0,0,e7c9ae73c50517fc2f01e234ce31c41b570dc32cac248e95308769e406fbcb98,d7f2c5c289bebe40b0ca9084b63dcf58d2244db08acb65a66e074f30c4a743c5
   baseline,integer-tail-8-fused0-apply_and_dot,0,6.039613253960873e-14,0,7.1054273576010453e-13,5299ce9229f6190595654c4dd5e571ccf6cd145449f2522c762254b6c04b8176,5dcc3a1724e13b4c4e95a58cacecd5f83c5514f052e79b2f712245d858875c3f
   candidate,integer-tail-8-fused0-apply_and_dot,0,6.039613253960873e-14,0,7.1054273576010453e-13,67a8b6c82ee2a2043403ff80c94938afe9b68ab60afd406c1811574abc72b0d5,d3e76b9ac8a6cb5ca538b86d7cb3c1beae45c12096d0e9a041a73c0f9beb6d71
   baseline,integer-tail-8-fused0-residual_squared_norm,0,6.1284310959308868e-14,0,1.0542233752630729e-11,911fefb7fd26459372a82a1c0937a8bccd96c64f5276ceef2ce8bc0111ba9e17,6a76425c6038378b7d4b9c038ef2f451d54bb08680904a6146bea010541b5738
   candidate,integer-tail-8-fused0-residual_squared_norm,0,6.1284310959308868e-14,0,1.0542233752630729e-11,41958c9eea2f0d8a0ec41cbe94370e2c5e79014fb78fe0649d5afe50da6dbf0f,3d41f6faec33547b4e604fae3619e2871c8d6202489a890f5466aa387a5c0980
   baseline,integer-tail-8-fused1-apply,0,6.039613253960873e-14,0,0,cd079726befc01d254690b39a01fa7bd945a740475457ffb963f38dcf74ae3c6,38621a924b99c605b597412948a173b92568bf015fb6b35a5e9212e1d97895f8
   candidate,integer-tail-8-fused1-apply,0,6.039613253960873e-14,0,0,2e83c09484a2c5e05cce7b57798be2a737cf84b9c54a1fb8d93fe0b0185a33d1,3795c62c9329878e1417bd6b03e1fbd0dc633e9b8376309017abdd3b05122540
   baseline,integer-tail-8-fused1-apply_and_dot,0,6.039613253960873e-14,0,7.1054273576010453e-13,6a2efe0e63fa6bfb6db90ce853c04983dc30014a2ec9843464044700412eb320,fe72c0d250e230b81e63507233b2bcec89bcade7a27b024eb42f69d56aacd99b
   candidate,integer-tail-8-fused1-apply_and_dot,0,6.039613253960873e-14,0,7.1054273576010453e-13,4e1408cce0f1595083008ca6f81c64d0b8005a9aa2796246b4f5834d025b0347,f5ff7142ac79f7249d6970b704280a2a52f6ed846295d8e08dd9dc6a83534edf
   baseline,integer-tail-8-fused1-residual_squared_norm,0,6.1284310959308868e-14,0,1.0542233752630729e-11,59f71ceb5686ba51b39834cdc3dc35b46a2b13aeafa64fcc5f102ba9a91b969c,abffefc4f7f0bd4781f436f91e1826759e9caea7ed1a73f9f0d02ab1d6b0c5d4
   candidate,integer-tail-8-fused1-residual_squared_norm,0,6.1284310959308868e-14,0,1.0542233752630729e-11,e659e9ab1d69ba9ac4d49cc9c4e77a5bcf37aa5dcb162bf40a3a63e8c936f8a5,e0fc60017cda297ea5c1b3dbefedb580e519d5ef908127d7e12a0ee145fae90a
   baseline,integer-tail-9-fused0-apply,0,7.1720407390785378e-14,0,0,cb3747a229b10e175424a7904a2a6dbd36f1e73744a2eb81a46f1c32ae31cd95,e0596f1bf7898cc1440ee676208d14037f4e2f16e737c4a73c8af07522e7f6dd
   candidate,integer-tail-9-fused0-apply,0,7.1720407390785378e-14,0,0,14708b8ff7a698858001d03f1707d353f01bc157ff4d25e9f808b4ede5124ba2,2f7e4fe700831bfb92f7ced1be94d8b3b83c92d19c45e2383815bc42f83ac773
   baseline,integer-tail-9-fused0-apply_and_dot,0,7.1720407390785378e-14,0,9.3614005436393805e-13,a89a91f9e320d3b6920690634e30fe37ea51e1a4fbf903e9431ed5fcd7126c5d,9a33e3055c4c97afae275a93a21d34daa68afec925591186a842f21e23920f13
   candidate,integer-tail-9-fused0-apply_and_dot,0,7.1720407390785378e-14,0,9.3614005436393805e-13,ab5f0e4b82388374e5aaac119fd1f67bbbfd43d5ae3ada34e74c671619342667,517115b76939393a8a8433f5b255a350aa595d16fe27576946d8063526e8d974
   baseline,integer-tail-9-fused0-residual_squared_norm,0,7.2608585810485528e-14,0,1.4299006423357375e-11,110c5ad2d8de9c0b5e3d721d9273ce188aa65f05d4f6a1e577bd6241541c4c2c,ea8654f6942991b39a880947a3486273457143e84fc46dd8093944cf027c2850
   candidate,integer-tail-9-fused0-residual_squared_norm,0,7.2608585810485528e-14,0,1.4299006423357375e-11,122c3766bdcba438f0fc7543fbce745e2c32715788124b7f4fc0585272fab7f6,08d955cc2f53009c751396eccc1cbbce91885c9d8e8919904c6ae99f149f51ad
   baseline,integer-tail-9-fused1-apply,0,7.1720407390785378e-14,0,0,68b613abb4b984c7de6d2ef1b442f3dfa89ce94b222a313ff517754c835e8c51,ccbdf28dbb632838153fc64fa00ed19a9fb7773d5fd35e45cfc5de0fcd860f94
   candidate,integer-tail-9-fused1-apply,0,7.1720407390785378e-14,0,0,89684fde86afcf34058c5665829dbd6f60e0609232feaadd0b52c8c3e97dffe6,8a674c9efa02eed6db5045ba607ebd056a680fcf16cd184e892d13c0c1f7031a
   baseline,integer-tail-9-fused1-apply_and_dot,0,7.1720407390785378e-14,0,9.3614005436393805e-13,46def3637c755fcebfd7fcc28c0939958ce0924a39f0e6a8b065666547e3d239,8e616b0ce6c3d3925bdf78e884d21f1561427ed20b71b7084d61fc0f842aa2a9
   candidate,integer-tail-9-fused1-apply_and_dot,0,7.1720407390785378e-14,0,9.3614005436393805e-13,a003d4f1bc3cafafe0f35b2de0f5a4f8dde19e9f15466e1c12675ea35d4dc78b,f2520599c634c833afbb2d639a95ae9bd7309fe3d42ec0f5be328997b5ebe57b
   baseline,integer-tail-9-fused1-residual_squared_norm,0,7.2608585810485528e-14,0,1.4299006423357375e-11,9d32a8d3cf53609c0dc5e4361a4d44aaeb727bbbf2d5cd8944f2f2bd3bc7b792,d6b151d93a7d2e73f8c0b15627ecb2d00d30dad5e400212b9974e1ea9fb1e723
   candidate,integer-tail-9-fused1-residual_squared_norm,0,7.2608585810485528e-14,0,1.4299006423357375e-11,23737a15770aa9c2d75fa5381883433af3cab6d3be7d119b467962c49f046b44,effe4cccc76c224824efc5f7a6e83d128b2581483544ad47ab112a2aaeddbf04
   baseline,integer-tail-15-fused0-apply,0,1.6853185513809962e-13,0,0,3cbb69d4504d5d4c9bb609e6f5cb9424bab220cd53af93e900c59f4060c2a01d,36e704d3e02b810f9506ebece0b0084cdc56bb140c3fb6c55184ccbe8a959182
   candidate,integer-tail-15-fused0-apply,0,1.6853185513809962e-13,0,0,d33e09e22be0a9d967e76370b58930ef4c3e84316b6c22c59ab39cf047daad5b,6d7c8a0848c3fea1ee9254a66a34316757864f832817142bf60e2a28d4ec8442
   baseline,integer-tail-15-fused0-apply_and_dot,0,1.6853185513809962e-13,0,3.5851321911195882e-12,bb827bfef70540b8be78aa7f0aa4f1fb50aa7cc1e81a9309ef37da39bd976f68,1dcc75ff92c5d61b1a56bb31ea2814b8b8f618af3bc3477c04b2586a52c0d9ab
   candidate,integer-tail-15-fused0-apply_and_dot,0,1.6853185513809962e-13,0,3.5851321911195882e-12,116f28637c7fd92165048c33749eed33c6d73c703e48ef7b973bc8d966d308e3,f0876396f0b8bfa3dd776c4539c8b588f23156fff3eba93c820477c01aa77935
   baseline,integer-tail-15-fused0-residual_squared_norm,0,1.7208456881690017e-13,0,4.7540638092869743e-11,053569176a93ca7c1447f6af105f6b23fd980ca89a7032deb3a4115939188fc2,8157a921f015ad4c4b784d411728999ccf87073b5b0cc4ced39ea6b349cd259a
   candidate,integer-tail-15-fused0-residual_squared_norm,0,1.7208456881690017e-13,0,4.7540638092869743e-11,9d22ad5d7877882b30d26a841a16f0cd3ea2454e328b5b3d6f61c2d502c5b3f4,ea5973931f598957458f0964fcdf71a213caf17007ea7efeeed2c8d91085d2d3
   baseline,integer-tail-15-fused1-apply,0,1.6853185513809962e-13,0,0,af7a22be8e3e82ee309c1758c6adb61c5c1e2cc2184868b463bebc7e181292e8,d54b586c8182bb32c4d95a0dfa515dee37487546c7af6c033ff1a964bff86882
   candidate,integer-tail-15-fused1-apply,0,1.6853185513809962e-13,0,0,5b8c3f543d9fe3c8614debebd2a4adeffdcf567eff393f747c70df76c43f272e,45eea097f2285e85dca2d940519e842c31ff5fb2cb88ef11e5290e4c5fcf2208
   baseline,integer-tail-15-fused1-apply_and_dot,0,1.6853185513809962e-13,0,3.5851321911195882e-12,5a5df061aabd6585ae0f176520b82f7d5867c5615041cd111f9c5403548171c4,7af76fe28c7b27824b9edfb9044d5a656b304fabc7b42644f2d4ab4d16d95942
   candidate,integer-tail-15-fused1-apply_and_dot,0,1.6853185513809962e-13,0,3.5851321911195882e-12,77163a5ca9e6df7e909197b6b23580bcfec6a0c5d6fa2946de3d58625a5bd68e,3ac3861dc22a389713ab9227f1c0438fffb6ac50ef417c3c527191e191e9ff39
   baseline,integer-tail-15-fused1-residual_squared_norm,0,1.7208456881690017e-13,0,4.7540638092869743e-11,faefc05ee1656e0cfcbf1178fda2e907fca6f57e5ef3acda5899ff0fdafafcf9,5d7bfb1e11a6ebb46cdf0ac5be69e236cff39d2be9261ca3b390e1e3c23fcfb0
   candidate,integer-tail-15-fused1-residual_squared_norm,0,1.7208456881690017e-13,0,4.7540638092869743e-11,f5a22b72e7c0fe088bcfae752399746fa0fbb0b27ed10243780a86a0add7d277,ef7f8dad558654b3d98919c85394c7eab69bd6e5da3ac33d8b39ebe3ef91fc53
   baseline,integer-tail-16-fused0-apply,0,1.9184653865522806e-13,0,0,cdf0284cfbe0ebb03f7c09684cefa53adfc25fc5a6dea82db069d55ad3cb06b6,d22555d537ada08d3201e96535af702b54ad070acafa7bef69596d4355c0e461
   candidate,integer-tail-16-fused0-apply,0,1.9184653865522806e-13,0,0,104989129e4629460c4149b20fc456119f9dcf55965b839b6b5b207929ef8ebb,222050e1f0bf89de51f5c926797fa48633d82ca303a82312fcb6fc862f028ce0
   baseline,integer-tail-16-fused0-apply_and_dot,0,1.9184653865522806e-13,0,4.5510262225434853e-12,5290411e1a20be30dbc21247cbfe14777be76970ccdfc48908a49e5ac5765302,4de0d66dde939315838ab6d8a991a58d0ecf95feaeacf9879323d74102e7e2d7
   candidate,integer-tail-16-fused0-apply_and_dot,0,1.9184653865522806e-13,0,4.5510262225434853e-12,749ab291a711d87006ad7965fb0401c33ff58df5dfe68714b96dd68d5680cd41,8d745c625856b0b64fa024d7e9acf4cec11cb5b1c1b8e093a1c31442590df120
   baseline,integer-tail-16-fused0-residual_squared_norm,0,1.925126724700032e-13,0,5.8039351102935351e-11,63979c3a4ec5879112e1311828b327035b9ead5d33d03f09df8b800bb6ccc8fa,04cccb207f6f4f41a764afe6b864663e41b4dd37b37171049131b8d41fe85da4
   candidate,integer-tail-16-fused0-residual_squared_norm,0,1.925126724700032e-13,0,5.8039351102935351e-11,c58be4f937d89748f064fc16ee57f16e0526be196366db60a88afa93a21adf94,4062669f648c493a5ae8f5461723d9dfc53007fd5436ff61f9530ecb349c7642
   baseline,integer-tail-16-fused1-apply,0,1.9184653865522806e-13,0,0,70d6581364b11d755753a2fee709a131944ab61c50e7cb431f00415451bb8bfd,93b086f0bac10f04c93cf9427e767def61cb4e405d8ff70ce1b0664f93eb24b7
   candidate,integer-tail-16-fused1-apply,0,1.9184653865522806e-13,0,0,6f3c10aede17dfad61cc9f64a57645891a4146bbc8c974285cb7631903f8add0,6e7f45ba874629c928ac41c93f17e04287437b966187a2d47720f732de4e22d8
   baseline,integer-tail-16-fused1-apply_and_dot,0,1.9184653865522806e-13,0,4.5510262225434853e-12,e89ec238eaed69877f48f615c3dba3fd1c2650e826d644e1af4fe8da7bc869a3,07e0d25ef5fe28ba4ada75d9a767909492b6a00c097386b4059db6c5f28bee4e
   candidate,integer-tail-16-fused1-apply_and_dot,0,1.9184653865522806e-13,0,4.5510262225434853e-12,6427d40eb480d0b1c5ed3fa245961fe4bb0c8fe5f3e92b3620ee0aa3ca4a81f1,1797d13bef8ff33d5efb02aaa26892840ede0e5517deb1c40371f0b746872323
   baseline,integer-tail-16-fused1-residual_squared_norm,0,1.925126724700032e-13,0,5.8039351102935351e-11,99d2ffacb1a34ca66b65a54a4dc18aa02ed87c4629becd94e9b1ad7bef41e1c5,337d8779654a65837537b891239004c855bb1da71f88fc071cc447984f537280
   candidate,integer-tail-16-fused1-residual_squared_norm,0,1.925126724700032e-13,0,5.8039351102935351e-11,e622576af781b32447dea6c770c9af5889af3bf1ec278a9944415a12ca63e946,253cd398319aa0cd84e989678037a9b81a14eb108d7a9a453a11afd5534b341e
   baseline,integer-tail-17-fused0-apply,0,2.053912595556551e-13,0,0,83b819e768772c58dd157e84c77ab06f2c1c3d5370670d6f09495f111a6870c2,bb19f131e486cd710a721c60a62e538164919e116451d2478506cbfa301fe51a
   candidate,integer-tail-17-fused0-apply,0,2.053912595556551e-13,0,0,59dbe199877645dabb7ba36c9fa04e1da25c00a72fe899c038d42f70a7078ff8,f5409e3c89c47cdc947d54299f414d4cff133c6faeb0c96a01e0d572caaccdf4
   baseline,integer-tail-17-fused0-apply_and_dot,0,2.053912595556551e-13,0,4.9960036108132553e-12,e15e594dbf945d10e6a4e373a9be60f33552cd59ddea924f86de273e0a8516e0,27a69de95ca18343ff637b2055b9ea44055d8d626e529c6e5cf5b17eaea26dc9
   candidate,integer-tail-17-fused0-apply_and_dot,0,2.053912595556551e-13,0,4.9960036108132553e-12,3d87cb2f31ad8919c944d5a7245344bcb180db052d6af236017dd6f5e06c6a5a,9afdd81a27665fb614dfa061626c34fbe9980353bd3cd7ce60d5e65fd9d5ac3a
   baseline,integer-tail-17-fused0-residual_squared_norm,0,2.0894397323445565e-13,0,5.7472693271166853e-11,573caa0b1c05adc1d0e55772f2916d15d77b420f9894d1da844bcddd5b89a78d,20b2650840d07a965bc647b3c1a440854fad0c09857fef119beda46f98696b0c
   candidate,integer-tail-17-fused0-residual_squared_norm,0,2.0894397323445565e-13,0,5.7472693271166853e-11,160105073b90d813afb53cc4b0a2bbc766ec9c4d341f3eae19e2b60e5860be66,37d1053c6cb2215f26d149a509361c6497c3b27c5f580710d0ca6e8676c71e74
   baseline,integer-tail-17-fused1-apply,0,2.053912595556551e-13,0,0,513063b313b96e328436aa95487d8649526b439d0d27f26bc628d64e1856f0e3,db4914d4b2c7ef153577b67ea94faec2d911c2ae3037c3b4f5110e76d88ed128
   candidate,integer-tail-17-fused1-apply,0,2.053912595556551e-13,0,0,8045ca752deda0988e9ee61ffd8447b769fba960b18fe8ba84bf8d01c0edd13e,45ca487b9a5b058f706fd8194d144d2b44065adab17b8429b3ac9768577c4bee
   baseline,integer-tail-17-fused1-apply_and_dot,0,2.053912595556551e-13,0,4.9960036108132553e-12,a7455bfecd3edaa7b565ec3b3bb10fa016bc730e50427d285883e2066694a10a,e75a23e10987178da1b7e8635ecd40845d4428dead13828736faafc9f14f863c
   candidate,integer-tail-17-fused1-apply_and_dot,0,2.053912595556551e-13,0,4.9960036108132553e-12,72d2bb74181a6dfb4c0238cad634906e3a3e10cbc73d32878b65dc9cace1820d,8c82b316f1309d07e8c9ac08dc27bb7812d4c29cd234fb8ef0acde3fb82577ae
   baseline,integer-tail-17-fused1-residual_squared_norm,0,2.0894397323445565e-13,0,5.7472693271166853e-11,551be07809a8ce807b7c8ccbc551daea86be7dfbac902327d935232f2e77f297,a6b3f44a5cf2072674a7edcaf1532db3931e7d83330875e599c7077c55864a36
   candidate,integer-tail-17-fused1-residual_squared_norm,0,2.0894397323445565e-13,0,5.7472693271166853e-11,422e42cbc191ed67b2039d1c2751c14ac5e07911469d0a5a38116004071f3c3c,2ff973f78958aae9aedfc36f8da12b5e450e4c947280ddaf91e4c57c609041f9
   baseline,integer-tail-129-fused0-apply,0,8.0917494926782365e-12,0,0,52e587680dd2802b44d5acfc9e1bfd931ab7f0806c5ed199beddfe541d254697,3af581d302158e172fb506d2dd1759ae2daf700f2e1d029d779e69c0def572cf
   candidate,integer-tail-129-fused0-apply,0,8.0917494926782365e-12,0,0,46802276a39800b12aa365efdc05618093b04e709c4f5946026fe30860605ce8,4302a227e45a7acaebfa044ff764bf0a80e561a1d048a0ab7f08368ed443d219
   baseline,integer-tail-129-fused0-apply_and_dot,0,8.0917494926782365e-12,0,1.2798349047216872e-09,e878b764c241bfc3ac8702b113ed6830c6716ac186d23653519d42f9bea97521,547fc239ebc3dd651af88057dc65cdb7418800a3f1c448b69329a550316dceb3
   candidate,integer-tail-129-fused0-apply_and_dot,0,8.0917494926782365e-12,0,1.2798349047216872e-09,503a57f006b031a4592801a136f2c8f12f01e26e38555a091cfd1e843e7f9a0a,e0afa5f78dc33651953cfd8f385d818b34456aedb0994ab3d3c72bd6ae5855f1
   baseline,integer-tail-129-fused0-residual_squared_norm,0,8.0950801617521136e-12,0,1.7782194827740646e-08,b167511b5074f5f711afbe640d680a807291059cdc41387b40ad3f03b1931a71,802802d3aababfdbcddb68476f8526fa9bd0e84b8497fc0cefb98045e0ad5759
   candidate,integer-tail-129-fused0-residual_squared_norm,0,8.0950801617521136e-12,0,1.7782194827740646e-08,7011116b36b589f3e1997aa13220804054e4e06c7190c47acbb88084776535a2,9d41c46c519664a08a8a895d6f2bd3d29c82928f90ab349859d7ad769d3ac584
   baseline,integer-tail-129-fused1-apply,0,8.0917494926782365e-12,0,0,75d82654d0594d660250b988f7b7512a269cc2ad8144f4603edca4185a85359f,8e258caba06923ed8e57c8a5ef7c92dc59a2bf56651402568559b2c2ee10df49
   candidate,integer-tail-129-fused1-apply,0,8.0917494926782365e-12,0,0,d006a6520469f81546b8b81db77b4ec5a8a985a7ac6d831e1e3afa56cfc069ca,0421cf8400b91f400f7deaba0fe46429b4c76fe2c6f812fff3091ae03e22a596
   baseline,integer-tail-129-fused1-apply_and_dot,0,8.0917494926782365e-12,0,1.2798349047216872e-09,3020f18155b8b0dfbe6ad8a556509a7ca0fbf9766e40f71eb1ae8e6955e752ec,84092a2b179e2bfa7243397123ab08c375463d504928a7edf9ab1859b5a9653b
   candidate,integer-tail-129-fused1-apply_and_dot,0,8.0917494926782365e-12,0,1.2798349047216872e-09,9691703efdaef5d127e57f3e57661fd92b6fdf9039f15cb75c56b9e863fe8340,7e7d6e9a87aa1272bb53c22fb225bdbf69868515bba6c516f6050b866b8a1649
   baseline,integer-tail-129-fused1-residual_squared_norm,0,8.0950801617521136e-12,0,1.7782194827740646e-08,e5ea3db87db3749279eff1d9961d221437522cb7a44e1aad9d77830418696c4d,99dd05f0e78c5738ae729dc50281ff9c2532695296fa89909ef142356feb5192
   candidate,integer-tail-129-fused1-residual_squared_norm,0,8.0950801617521136e-12,0,1.7782194827740646e-08,e85f12925d7a24640cf4f01530840d682c31f493986af5cc621bde17b1e14295,474b4bd7fa754e9bd9120f0d9a1a37003f25ba568a80397453be305be69a3673
   baseline,integer-tail-130-fused0-apply,0,8.2427398240272643e-12,0,0,16cf3e28162073272b81e9cc206f59d5d6d3f1774cfa4f2ca9b788ad6cd3adca,56def145beab4b23d226684a6ca750c5c35a93d10a0e8c6376707e598fae0f58
   candidate,integer-tail-130-fused0-apply,0,8.2427398240272643e-12,0,0,e98eb691e979781c55275337a1199b0f8e9af7b17a6f60abf07764ae860b0ac2,65d816869fb7fb4b8f08ea4c1dbb309fec7d43c61717b102fc6e690d62fd29e9
   baseline,integer-tail-130-fused0-apply_and_dot,0,8.2427398240272643e-12,0,1.3138130583457381e-09,76d328ea905b1006ea3ed4ad87013dedd4156825df346598384fca30c31d106d,622c1957927b60565d546fe4cd3e119b76d6f421c6cf3e029a31a47412de1176
   candidate,integer-tail-130-fused0-apply_and_dot,0,8.2427398240272643e-12,0,1.3138130583457381e-09,cefc3bf60cca5f68a4962e436035a27dbf648d2c28d26393bd3abe6ffb96f5da,0515e5af0f0993266bab0f7f7cbe9ba7361180fc468507d3df94d20f97997201
   baseline,integer-tail-130-fused0-residual_squared_norm,0,8.2440720916568161e-12,0,1.4466784215026626e-08,e82e1a46b2136a61de055fc4476f3dd6ff34c29682360b5673eb27b7ba86d0aa,d75265790025939e05fa301851c065542bc2436a029d0654ff2c8c79b2baec71
   candidate,integer-tail-130-fused0-residual_squared_norm,0,8.2440720916568161e-12,0,1.4466784215026626e-08,7bee45f64f79b7c95e455310ee10aeecf6fa432611f9b63194b30ea5035101c3,588c4f5ff9bcffc2ca18127f2ae1f2c4b1c2d204dc0992d892ef59b0d40bfe7a
   baseline,integer-tail-130-fused1-apply,0,8.2427398240272643e-12,0,0,0742d37b41f4b10beb1d36819524bda8e70d70f8b0ef561a80fbbe0427bfb2f7,71e46af64ca205bb6c05fa34f7dd6dfc9b5f7fa338a29a3bf0a19425a873cb41
   candidate,integer-tail-130-fused1-apply,0,8.2427398240272643e-12,0,0,0a99f568583e3a6faebad281ada79f28035781f5485d5cd8ec0664f348a7c58d,48f9f5a34b8f963356617f68f99cbea1d79bf9767f50ba956ea35c8461a0c9bc
   baseline,integer-tail-130-fused1-apply_and_dot,0,8.2427398240272643e-12,0,1.3138130583457381e-09,3078e029c5243cd41d512f2ead2602e00f461fceadbe8896407704e5d8a9b2dc,99a7bcd06788c11d03555c89831512f7ea876dc5f4dac80b20f899f0903b3de5
   candidate,integer-tail-130-fused1-apply_and_dot,0,8.2427398240272643e-12,0,1.3138130583457381e-09,d5ead19a62b416d15853cd8024cc0438d8f4d1e34e78591c75fe20ff464fcde0,72d8d0b697821b8b62286ae8e5a86fe910c6cf38f77e0018e7ccec513d56e0f6
   baseline,integer-tail-130-fused1-residual_squared_norm,0,8.2440720916568161e-12,0,1.4466784215026626e-08,51bd690a56c82e08f840ea2a69fdc9554320818c72d94429351f2124eae04d3f,681ad56b2265e2b21daf2b48ae574040fa3249b3a7e082fc5f418f2b7fa2ecbd
   candidate,integer-tail-130-fused1-residual_squared_norm,0,8.2440720916568161e-12,0,1.4466784215026626e-08,c1b355db5de057f4f5aa625ae2e48ace743c1c1ff31ab6edd287e4dd9fe9c260,bfb58ccfaa258dde6da487f291622d50a3fd3482a13dc2ebf5422cd5ec8d8931
   baseline,integer-tail-131-fused0-apply,0,8.3950624230058422e-12,0,0,eb29017b7d3bf79b127d549e7de3e79719d1289c8a2091f013c29974671bdf57,3b39d965150a4608fac93c687f01ed08b0b7856171155d8af2f4e595bf8cc1d7
   candidate,integer-tail-131-fused0-apply,0,8.3950624230058422e-12,0,0,444cefadfeff843257da252c4a3d2817d588e99509762f1781235e29453ed088,0e8f18d652a351eae0076d81294922f37dea32216c9fdda13d0143c6d1b7ba84
   baseline,integer-tail-131-fused0-apply_and_dot,0,8.3950624230058422e-12,0,1.3516050501039815e-09,88a480dce33ee7d516a87567bfba2f84da4b16c33b57eb22cf338bc970752174,e50534e29210e2906449340e91a8d7f1091331635c327bb293bbf962bf999c36
   candidate,integer-tail-131-fused0-apply_and_dot,0,8.3950624230058422e-12,0,1.3516050501039815e-09,b570c20c6f181ab30f1800ed8d19cf9545ee66340c74964ffe912b3fc6753592,ab200859eddb87d384b72154eca9a10b3929622227a8f8907084861a699fc13a
   baseline,integer-tail-131-fused0-residual_squared_norm,0,8.3979490028698708e-12,0,1.3550524036186618e-08,fac24f39fec449bf4443ff03e90395fc86f6ccb2dfc302836b87eb481e49c48b,3bf56834591d76a69a8b0d5ecc008d1fb78a92d39b043e7fc1ba937934b0c22f
   candidate,integer-tail-131-fused0-residual_squared_norm,0,8.3979490028698708e-12,0,1.3550524036186618e-08,9076139d2405dc66bd238837697428ef1222e4622850ab180568826057e91a72,d3525878cfe7f8af04a2d7d698d0d1a1e548f9e3cc99a7a5312f85e0c5795420
   baseline,integer-tail-131-fused1-apply,0,8.3950624230058422e-12,0,0,282d136b6e5d60598408b53d276b97888f6786ab68a4848040b96bc67825e513,413a87f504162e6ed10fb4ce85dd94b355b10c2b54638a36ee51b0471f141b0c
   candidate,integer-tail-131-fused1-apply,0,8.3950624230058422e-12,0,0,5c0e20eb16d7e0b9738e5730c3fe8eb475c4b063350f7811364baf75be8151d4,ff3c24aa7cd9c797be8a91dfccf6dcaf108615b073e2cd52dad630cea06122a1
   baseline,integer-tail-131-fused1-apply_and_dot,0,8.3950624230058422e-12,0,1.3516050501039815e-09,a34c07e2ad3eb16a7fad982b47e1f24f23d4cc7afdd2ec95ef8ed725587a092e,8a3c688767f7a89bdbaf99ddce8dbdc16ca49ad90e347959c73284cb67286356
   candidate,integer-tail-131-fused1-apply_and_dot,0,8.3950624230058422e-12,0,1.3516050501039815e-09,828e17bd82b615d8a24dce33f7fda462366c6f14769ff9f6e63cd248d6f87b9c,05c0836cd2e18a0e04c3e5409980b474e2a423628e01c47c902a430ea3349b1b
   baseline,integer-tail-131-fused1-residual_squared_norm,0,8.3979490028698708e-12,0,1.3550524036186618e-08,5c11a7946659f6868cc8305942dd3f74f7e69cd5e050b77a957acd38d20f6500,5037c542cb2e5046838011b79c6d9db3012ccc8680e0e3f3348c52431ceb8217
   candidate,integer-tail-131-fused1-residual_squared_norm,0,8.3979490028698708e-12,0,1.3550524036186618e-08,4510660a3cb6f88ddea4ed2b39d9c36e1adef43c7d69b4c960d4b66db10f63a8,7d829a045da27a487c83bd9c771e86bec62c1e708e781e249b7e17410810f1c3
