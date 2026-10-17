CPU dense decision and diagnostic evidence
==========================================

The eight-accumulator CPU candidate was rejected at the fixed-work preflight. Its first configuration converged with different iterations and matvecs even though both revisions passed true-residual checks. No candidate CPU A/B timing campaign was started, so this is neither a measured slowdown nor a demonstrated speedup. Keep the baseline production arithmetic; retain useful independent boundary, cancellation and non-finite coverage. CPU 2048 prepared optimization remains open.

This report records diagnostics from baseline ``84c88c98c895103395e1b6d6e98f7618848a14d3`` and rejected candidate ``abd655a20a6835b6b95b2e6c3aeae2b69b9957aa``. It is not a current-main NumPy/PyTorch scorecard. The 5% adoption threshold is unchanged. Fresh API, Session and prepared scopes remain separate; no backend, fusion or size AUTO policy is introduced.

Fixed-work rejection
--------------------

The refined source was frozen before candidate performance outcomes. The first configuration is CPU-release, dense float64, native kernel, standard CG without preconditioner, fused operations, fresh API, one CPU thread. For dense-64, two checks each reproduce baseline 49/51/0 versus candidate 48/50/0 (iterations/matvecs/restarts). Both true residuals pass the original tolerance. Rounding-group changes are numerically valid here but ineligible for this declared same-work optimization trial. There is no work normalization, selective size exception, timing comparison, or relaxed tolerance.

.. csv-table:: All checks from the two completed first-configuration workers
   :header: "Revision","Case","Implementation","I/M/R","Checks","Maximum true residual","Minimum threshold"

   "baseline","dense-64","native","49/51/0","2","1.8582645e-06","3.0576161e-06"
   "baseline","dense-64","numpy-out","49/51/0","2","1.8734617e-06","3.0576161e-06"
   "baseline","dense-64","torch-eager-out","49/51/0","2","1.9437482e-06","3.0576161e-06"
   "baseline","dense-256","native","67/70/1","2","4.8815545e-06","5.0262292e-06"
   "baseline","dense-256","numpy-out","67/70/1","2","4.8752604e-06","5.0262292e-06"
   "baseline","dense-256","torch-eager-out","67/70/1","2","4.8872893e-06","5.0262292e-06"
   "baseline","dense-1024","native","70/73/1","2","9.6567151e-06","1.0405785e-05"
   "baseline","dense-1024","numpy-out","70/73/1","2","9.6567151e-06","1.0405785e-05"
   "baseline","dense-1024","torch-eager-out","70/73/1","2","9.6567151e-06","1.0405785e-05"
   "baseline","dense-2048","native","70/73/1","2","1.2911386e-05","1.4810685e-05"
   "baseline","dense-2048","numpy-out","70/73/1","2","1.2911386e-05","1.4810685e-05"
   "baseline","dense-2048","torch-eager-out","70/73/1","2","1.2911385e-05","1.4810685e-05"
   "current","dense-64","native","48/50/0","2","2.307155e-06","3.0576161e-06"
   "current","dense-64","numpy-out","49/51/0","2","1.8734617e-06","3.0576161e-06"
   "current","dense-64","torch-eager-out","49/51/0","2","1.9437482e-06","3.0576161e-06"
   "current","dense-256","native","67/70/1","2","4.9381732e-06","5.0262292e-06"
   "current","dense-256","numpy-out","67/70/1","2","4.8752604e-06","5.0262292e-06"
   "current","dense-256","torch-eager-out","67/70/1","2","4.8872893e-06","5.0262292e-06"
   "current","dense-1024","native","70/73/1","2","9.6567151e-06","1.0405785e-05"
   "current","dense-1024","numpy-out","70/73/1","2","9.6567151e-06","1.0405785e-05"
   "current","dense-1024","torch-eager-out","70/73/1","2","9.6567151e-06","1.0405785e-05"
   "current","dense-2048","native","70/73/1","2","1.2911385e-05","1.4810685e-05"
   "current","dense-2048","numpy-out","70/73/1","2","1.2911386e-05","1.4810685e-05"
   "current","dense-2048","torch-eager-out","70/73/1","2","1.2911385e-05","1.4810685e-05"

The enclosing preflight manifest remains failed; later declared configurations were not completed by this aborted CPU trial. The full per-check values and source identities are preserved in the accompanying structured audit.

Actual code-generation inspection
---------------------------------

GNU C++ 16.0.1 used ``-O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic``. All inspected helpers use 128-bit SSE2, without AVX or FMA. The refined low/high four-element arrays preserve the first eight-accumulator prototype's per-lane recurrence and balanced reduction, but change association relative to the four-accumulator baseline. Static instruction and frame counts do not establish runtime performance, cache behavior or bandwidth.

.. csv-table:: Actual serial two-row helper observations
   :header: "Observation","Baseline","713c8ae","abd655a"

   "Packed accumulation chains","4","8","8"
   "Distinct bulk-loop XMM register names","7","13","13"
   "Bulk-loop FP stack references","0","0","0"
   "Instructions per eight columns","36","34","34"
   "Unconditional frame bytes","0 on multiple-of-four path","168","104"
   "Entry array-zeroing bytes","0","128","64"
   "Post-bulk FP stores before optional four-column tail (bytes)","0","64","64"
   "Two-row helper symbol bytes","475","918","835"
   "Whole dense object .text bytes","3774","4667","4539"

The refinement partially reduces setup and footprint; its unconditional frame, canary and post-loop stores remain. No further source variant was selected from performance outcomes. Earlier rejected four-row and four-accumulator unroll experiments are separate evidence and are not repeated or relabelled here.

.. csv-table:: Previously preserved actual object SHA256 identities
   :header: "Representation","Preserved object","SHA256"

   "Baseline","codegen-halves/baseline.o","9315d269ab22598ddd4227a0fa6080681569b9fdf5c0b2baff07a4b43cbfdf95"
   "713c8ae","codegen/current-713-stage.o","778f5045420972e87161c5dd291a1f269ad6b5da807273ade4734c0563454b6b"
   "abd655a","codegen-halves/current.o","5b82b88229d0dcfd998974b5bc926413f08aae25dc4d3e1b7589cda18c7d7284"

These object hashes were checked in the earlier static audit; report rendering does not rehash binaries. The full frame/instruction audit remains preserved as ``cpu-codegen-halves-note.rst``.

Baseline same-call engine attribution
-------------------------------------

The exact shared solver engine and actual CPU backends are instrumented on the same dense-2048 fixture (seed 20261017). There are 24 ordinary captures and six additional provider-wrapped captures, each in a fresh process with three warmups and ten measured engine calls. Every warmup and sample has a converged status, stable I/M/R within its capture, and an independent wider-long-double true residual check outside timing. Jacobi preparation, workspace allocation, x reset, Python dispatch and public solve validation are outside these engine intervals.

The matvec-entry percentage is the median of each call's summed ``apply``, ``apply_and_dot`` and ``residual_squared_norm`` times divided by its engine time. Composite entries include vector arithmetic. The nested private CBLAS timer isolates actual provider-entry elapsed time inside those composites; it includes provider checks/dispatch and timer overhead, rather than exclusive microkernel CPU time. Provider time must not be added to entry sums. All percentages below come from same-call ratios; separate capture medians are not subtracted.

.. csv-table:: All engine captures; times are medians in milliseconds
   :header: "Capture","I/M/R","Engine ms","Entry sum ms","Remainder ms","Matvec-entry %","Nested provider %","Maximum true-residual/threshold"

   "cpu-release-native-standard-none-fused0","70/73/1","100.8635","100.83195","0.0281545","99.720549","n/a","0.87176155"
   "cpu-release-native-standard-none-fused1","70/73/1","103.91455","103.88041","0.026821","99.790351","n/a","0.87176155"
   "cpu-release-native-standard-jacobi-fused0","3/5/0","6.1359045","6.133435","0.002723","99.655595","n/a","0.83641225"
   "cpu-release-native-standard-jacobi-fused1","3/5/0","5.8770945","5.875022","0.0021835","99.728347","n/a","0.83641225"
   "cpu-release-native-explicit-none-fused0","68/137/0","158.36342","158.27195","0.083896","99.752594","n/a","0.98130942"
   "cpu-release-native-explicit-none-fused1","68/137/0","159.26138","159.21603","0.046001","99.72865","n/a","0.98130942"
   "cpu-openmp-native-standard-none-fused0","70/73/1","48.546378","48.520315","0.0257545","99.248994","n/a","0.87176155"
   "cpu-openmp-native-standard-none-fused1","70/73/1","52.606431","52.584942","0.02666","99.224025","n/a","0.87176155"
   "cpu-openmp-native-standard-jacobi-fused0","3/5/0","3.450463","3.448252","0.002453","99.151357","n/a","0.83641225"
   "cpu-openmp-native-standard-jacobi-fused1","3/5/0","3.456848","3.454399","0.002546","99.104059","n/a","0.83641225"
   "cpu-openmp-native-explicit-none-fused0","68/137/0","101.59005","101.48882","0.046475","99.519448","n/a","0.98130942"
   "cpu-openmp-native-explicit-none-fused1","68/137/0","93.244557","93.199611","0.0442705","99.289699","n/a","0.98130942"
   "cpu-blas-native-standard-none-fused0","70/73/1","90.747881","90.725233","0.023825","99.697779","n/a","0.87176155"
   "cpu-blas-native-standard-none-fused1","70/73/1","90.203702","90.170068","0.028774","99.783773","n/a","0.87176155"
   "cpu-blas-native-standard-jacobi-fused0","3/5/0","6.6076875","6.6001355","0.0017715","99.715629","n/a","0.83641225"
   "cpu-blas-native-standard-jacobi-fused1","3/5/0","6.198311","6.1967285","0.0015495","99.640093","n/a","0.83641225"
   "cpu-blas-native-explicit-none-fused0","68/137/0","171.74281","171.68967","0.053137","99.801063","n/a","0.98130942"
   "cpu-blas-native-explicit-none-fused1","68/137/0","192.42125","192.38192","0.045734","99.794583","n/a","0.98130942"
   "cpu-blas-blas-standard-none-fused0","70/73/1","84.548345","84.516131","0.0312445","99.656482","n/a","0.87176155"
   "cpu-blas-blas-standard-none-fused1","70/73/1","79.859586","79.823885","0.0326855","99.726432","n/a","0.87176155"
   "cpu-blas-blas-standard-jacobi-fused0","3/5/0","5.456052","5.453814","0.0020045","99.623192","n/a","0.83641225"
   "cpu-blas-blas-standard-jacobi-fused1","3/5/0","5.3576465","5.356163","0.002013","99.718472","n/a","0.83641225"
   "cpu-blas-blas-explicit-none-fused0","68/137/0","150.38634","150.36122","0.0264545","99.759554","n/a","0.98130942"
   "cpu-blas-blas-explicit-none-fused1","68/137/0","141.27486","141.24558","0.025433","99.734617","n/a","0.98130942"
   "provider-standard-none-fused0","70/73/1","76.042181","76.010869","0.02529","99.664691","99.514132","0.87176155"
   "provider-standard-none-fused1","70/73/1","73.564548","73.541472","0.033333","99.741645","99.587221","0.87176155"
   "provider-standard-jacobi-fused0","3/5/0","5.433049","5.4304385","0.0025535","99.543501","99.32157","0.83641225"
   "provider-standard-jacobi-fused1","3/5/0","5.3692645","5.366724","0.002565","99.692583","99.522189","0.83641225"
   "provider-explicit-none-fused0","68/137/0","136.8543","136.6764","0.0458755","99.735184","99.521313","0.98130942"
   "provider-explicit-none-fused1","68/137/0","135.29772","135.25119","0.053275","99.732051","99.572332","0.98130942"

Across the six provider-wrapped captures, the median fraction of an engine interval spent inside the actual private GEMV entry ranges from 99.322% to 99.587%. This supports investigating actual GEMV/provider/storage behavior before optimizing unrelated vector operations. It does not determine the cause of an earlier full-solver NumPy gap. Oracle matrix reads between calls and instrumentation overhead limit generalization to uninstrumented solves.

Actual private provider identity after a checked GEMV:

.. code-block:: json

   {
     "blasint_bytes": 4,
     "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
     "environment_OPENBLAS_CORETYPE": null,
     "framework_score": false,
     "identity_dgemv_check_passes": true,
     "metric": "private_openblas_identity",
     "parallel_kind": 0,
     "reported_num_threads": 1,
     "schema_version": 1,
     "selected_core_after_dgemv": "Haswell"
   }

All per-entry call/time medians are retained in the structured audit. Their separate medians are descriptive and need not sum to the median engine time.

Same-buffer provider and storage probe
--------------------------------------

A separate component diagnostic crosses two providers with two storage owners: actual baseline private BLAS through ``CpuBackend::apply`` and NumPy ``dot(out=)``; the exact fixture's NumPy allocation and an actual owning ``cg::DenseMatrix`` snapshot made with ``std::vector<double>``. Both providers use the same matrix, vector and output addresses for a given storage choice. The fixture's rhs b is the fixed GEMV vector; these are not CG search directions. Native snapshot construction and all identity/oracle checks are outside timing. The NumPy borrowed allocation is not a claim about the full NumPy prepared adapter's separate copied matrix.

For each size, all four fresh processes are retained in A/B/B/A label order. A and B reverse treatment enumeration before the same Williams scheduler; they use the same source and bridge, not different optimization builds. Each process uses CPU 0 and one thread in all providers, three warmups per treatment and 36 complete four-treatment Williams rounds with one call per treatment. No global THP setting or NumPy environment variable is changed.

.. csv-table:: Every fresh-process component median in milliseconds
   :header: "Process","Size","Private / NumPy A","NumPy / NumPy A","Private / C++ A","NumPy / C++ A"

   "dense2048-p1-A","2048","1.22442","1.2408155","1.268276","1.2350245"
   "dense2048-p2-B","2048","1.372966","1.439262","1.415382","1.42037"
   "dense2048-p3-B","2048","1.0745405","1.100359","0.979196","1.025904"
   "dense2048-p4-A","2048","1.1087915","1.0762205","1.182623","1.165265"
   "dense1024-p1-A","1024","0.1485445","0.1442195","0.1485685","0.1483445"
   "dense1024-p2-B","1024","0.146474","0.1546845","0.1486875","0.151545"
   "dense1024-p3-B","1024","0.155286","0.1519175","0.1584395","0.15542"
   "dense1024-p4-A","1024","0.146485","0.148404","0.1498585","0.15461"

.. csv-table:: Within-process ratios; values above one mean the numerator is slower
   :header: "Process","Private/NumPy on NumPy A","Private/NumPy on C++ A","C++/NumPy A private","C++/NumPy A NumPy"

   "dense2048-p1-A","0.98678651","1.0269238","1.0358178","0.99533291"
   "dense2048-p2-B","0.9539375","0.99648824","1.0308937","0.98687383"
   "dense2048-p3-B","0.97653629","0.95447137","0.91126951","0.93233572"
   "dense2048-p4-A","1.0302642","1.0148962","1.0665874","1.0827382"
   "dense1024-p1-A","1.029989","1.00151","1.0001616","1.0286022"
   "dense1024-p2-B","0.94692099","0.98114421","1.0151119","0.97970385"
   "dense1024-p3-B","1.0221732","1.019428","1.0203077","1.0230553"
   "dense1024-p4-A","0.98706908","0.96926784","1.0230297","1.0418183"

The observed 2048 workers do not show a consistent large provider or ownership penalty. These component medians are not a full-solver score, a 5% adoption result, or a calibrated confidence interval. Between-process variation and treatment-dependent cache state remain visible. No aggregate winner is selected. The probe does not reproduce all prepared-adapter allocations, CG operands, or real solver memory-access sequences. Provider calls include their different Python/ctypes dispatch boundaries; this is not exclusive kernel timing.

Product correctness and ownership evidence
------------------------------------------

Each worker performs four prechecks, twelve warmup checks and 144 timed-call checks, all outside timing. The oracle uses wider-long-double row products and sums, with a double forward-error bound plus a long-double oracle bound. All 160 checks per process pass. The maximum observed error and maximum rowwise bound below summarize different rows/calls; their ratio is not the per-row gate. Input and both matrix byte hashes remain unchanged.

.. csv-table:: Every worker's independent product checks
   :header: "Process","Precision bits","Passing checks","Maximum absolute error","Maximum row bound","Canonical fixture SHA256"

   "dense2048-p1-A","64","160","6.0822458e-12","2.2138527e-08","8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c"
   "dense2048-p2-B","64","160","6.0822458e-12","2.2138527e-08","8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c"
   "dense2048-p3-B","64","160","6.0822458e-12","2.2138527e-08","8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c"
   "dense2048-p4-A","64","160","6.0822458e-12","2.2138527e-08","8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c"
   "dense1024-p1-A","64","160","4.5670134e-12","8.2972642e-09","204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb"
   "dense1024-p2-B","64","160","4.5670134e-12","8.2972642e-09","204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb"
   "dense1024-p3-B","64","160","4.5670134e-12","8.2972642e-09","204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb"
   "dense1024-p4-A","64","160","4.5670134e-12","8.2972642e-09","204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb"

THP observations and limits
---------------------------

The allocator policy was a source-grounded hypothesis: NumPy 2.2.6 can request ``MADV_HUGEPAGE`` for sufficiently large allocations, whereas the native owning snapshot uses ordinary vector storage. See the `NumPy global-state documentation <https://numpy.org/doc/2.2/reference/global_state.html>`_ and `versioned allocator source <https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/multiarray/alloc.c>`_. The following records describe actual overlapping VMAs, not a guessed allocator policy. ``AnonHugePages`` is VMA residency; ``THPeligible`` and ``hg`` are eligibility/advice observations. A VMA may include unrelated allocations, so its huge-page bytes cannot be assigned wholly to a matrix. ``KernelPageSize`` alone is not a huge-page-residency test.

At 2048, the distinct VMA-wide AnonHugePages values are 0 kB, 30720 kB for the NumPy matrix and 0 kB for the C++ matrix. The table retains every overlapping matrix mapping in both phases. These residency/advice/eligibility fields remain unchanged across the probe. The component timings do not establish that this policy difference caused the historical prepared-solver gap. No allocator change follows from this evidence. All exact VMA records, including matrix/vector/output mappings before and after every worker, are retained below and in the structured audit.

.. csv-table:: Every overlapping matrix VMA; AnonHugePages applies to the entire mapping
   :header: "Process","Phase","Buffer","VMA address range","AnonHugePages","THPeligible","VmFlags","VMA exceeds buffer"

   "dense2048-p1-A","before","borrowed-numpy","7d56eaa1c000-7d56eaa1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p1-A","before","borrowed-numpy","7d56eaa1d000-7d56eca1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p1-A","before","owned-cpp","7d56e6a1b000-7d56e8a1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p1-A","after","borrowed-numpy","7d56eaa1c000-7d56eaa1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p1-A","after","borrowed-numpy","7d56eaa1d000-7d56eca1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p1-A","after","owned-cpp","7d56e6a1b000-7d56e8a1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p2-B","before","borrowed-numpy","7b9f36d1c000-7b9f36d1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p2-B","before","borrowed-numpy","7b9f36d1d000-7b9f38d1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p2-B","before","owned-cpp","7b9f32d1b000-7b9f34d1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p2-B","after","borrowed-numpy","7b9f36d1c000-7b9f36d1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p2-B","after","borrowed-numpy","7b9f36d1d000-7b9f38d1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p2-B","after","owned-cpp","7b9f32d1b000-7b9f34d1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p3-B","before","borrowed-numpy","76ae53a1c000-76ae53a1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p3-B","before","borrowed-numpy","76ae53a1d000-76ae55a1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p3-B","before","owned-cpp","76ae4fa1b000-76ae51a1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p3-B","after","borrowed-numpy","76ae53a1c000-76ae53a1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p3-B","after","borrowed-numpy","76ae53a1d000-76ae55a1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p3-B","after","owned-cpp","76ae4fa1b000-76ae51a1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p4-A","before","borrowed-numpy","717aabc1c000-717aabc1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p4-A","before","borrowed-numpy","717aabc1d000-717aadc1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p4-A","before","owned-cpp","717aa7c1b000-717aa9c1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p4-A","after","borrowed-numpy","717aabc1c000-717aabc1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense2048-p4-A","after","borrowed-numpy","717aabc1d000-717aadc1d000","30720 kB","1","rd wr mr mw me ac sd hg","True"
   "dense2048-p4-A","after","owned-cpp","717aa7c1b000-717aa9c1c000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p1-A","before","borrowed-numpy","7b3d56d1c000-7b3d56d1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p1-A","before","borrowed-numpy","7b3d56d1d000-7b3d5751d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p1-A","before","owned-cpp","12796000-13e2e000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p1-A","after","borrowed-numpy","7b3d56d1c000-7b3d56d1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p1-A","after","borrowed-numpy","7b3d56d1d000-7b3d5751d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p1-A","after","owned-cpp","12796000-13e2e000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p2-B","before","borrowed-numpy","7ab58191c000-7ab58191d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p2-B","before","borrowed-numpy","7ab58191d000-7ab58211d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p2-B","before","owned-cpp","30516000-31bae000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p2-B","after","borrowed-numpy","7ab58191c000-7ab58191d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p2-B","after","borrowed-numpy","7ab58191d000-7ab58211d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p2-B","after","owned-cpp","30516000-31bae000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p3-B","before","borrowed-numpy","78d372a1c000-78d372a1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p3-B","before","borrowed-numpy","78d372a1d000-78d37321d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p3-B","before","owned-cpp","1ce4d000-1e4e5000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p3-B","after","borrowed-numpy","78d372a1c000-78d372a1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p3-B","after","borrowed-numpy","78d372a1d000-78d37321d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p3-B","after","owned-cpp","1ce4d000-1e4e5000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p4-A","before","borrowed-numpy","7586eed1c000-7586eed1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p4-A","before","borrowed-numpy","7586eed1d000-7586ef51d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p4-A","before","owned-cpp","224ab000-23b43000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p4-A","after","borrowed-numpy","7586eed1c000-7586eed1d000","0 kB","0","rd wr mr mw me ac sd","True"
   "dense1024-p4-A","after","borrowed-numpy","7586eed1d000-7586ef51d000","6144 kB","1","rd wr mr mw me ac sd hg","True"
   "dense1024-p4-A","after","owned-cpp","224ab000-23b43000","0 kB","0","rd wr mr mw me ac sd","True"

dense2048-p1-A: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 137812257128464,
         "address_hex": "0x7d56eaa1c010",
         "address_mod_2MiB": 114704,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 870531888,
         "address_hex": "0x33e34330",
         "address_mod_2MiB": 213808,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 137812190015504,
         "address_hex": "0x7d56e6a1b010",
         "address_mod_2MiB": 110608,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 870432352,
         "address_hex": "0x33e1be60",
         "address_mod_2MiB": 114272,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d",
       "owned-cpp": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 137812257132544,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7d56eaa1c000-7d56eaa1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 137812257132544,
         "overlap_start": 137812257128464,
         "phase": "before",
         "start": 137812257128448
       },
       {
         "buffer": "borrowed-numpy",
         "end": 137812290686976,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7d56eaa1d000-7d56eca1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 137812290682896,
         "overlap_start": 137812257132544,
         "phase": "before",
         "start": 137812257132544
       },
       {
         "buffer": "owned-cpp",
         "end": 137812223574016,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7d56e6a1b000-7d56e8a1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 137812223569936,
         "overlap_start": 137812190015504,
         "phase": "before",
         "start": 137812190015488
       },
       {
         "buffer": "vector",
         "end": 870928384,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "337bd000-33e95000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 870448736,
         "overlap_start": 870432352,
         "phase": "before",
         "start": 863752192
       },
       {
         "buffer": "output",
         "end": 870928384,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "337bd000-33e95000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 870548272,
         "overlap_start": 870531888,
         "phase": "before",
         "start": 863752192
       },
       {
         "buffer": "borrowed-numpy",
         "end": 137812257132544,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7d56eaa1c000-7d56eaa1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 137812257132544,
         "overlap_start": 137812257128464,
         "phase": "after",
         "start": 137812257128448
       },
       {
         "buffer": "borrowed-numpy",
         "end": 137812290686976,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7d56eaa1d000-7d56eca1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 137812290682896,
         "overlap_start": 137812257132544,
         "phase": "after",
         "start": 137812257132544
       },
       {
         "buffer": "owned-cpp",
         "end": 137812223574016,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7d56e6a1b000-7d56e8a1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 137812223569936,
         "overlap_start": 137812190015504,
         "phase": "after",
         "start": 137812190015488
       },
       {
         "buffer": "vector",
         "end": 870928384,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6992 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6992 kB",
           "Rss": "6992 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "337bd000-33e95000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 870448736,
         "overlap_start": 870432352,
         "phase": "after",
         "start": 863752192
       },
       {
         "buffer": "output",
         "end": 870928384,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6992 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6992 kB",
           "Rss": "6992 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "337bd000-33e95000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 870548272,
         "overlap_start": 870531888,
         "phase": "after",
         "start": 863752192
       }
     ]
   }

dense2048-p2-B: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 135923749732368,
         "address_hex": "0x7b9f36d1c010",
         "address_mod_2MiB": 1163280,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 191288112,
         "address_hex": "0xb66d330",
         "address_mod_2MiB": 447280,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 135923682619408,
         "address_hex": "0x7b9f32d1b010",
         "address_mod_2MiB": 1159184,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 191188576,
         "address_hex": "0xb654e60",
         "address_mod_2MiB": 347744,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d",
       "owned-cpp": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 135923749736448,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7b9f36d1c000-7b9f36d1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135923749736448,
         "overlap_start": 135923749732368,
         "phase": "before",
         "start": 135923749732352
       },
       {
         "buffer": "borrowed-numpy",
         "end": 135923783290880,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7b9f36d1d000-7b9f38d1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135923783286800,
         "overlap_start": 135923749736448,
         "phase": "before",
         "start": 135923749736448
       },
       {
         "buffer": "owned-cpp",
         "end": 135923716177920,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7b9f32d1b000-7b9f34d1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135923716173840,
         "overlap_start": 135923682619408,
         "phase": "before",
         "start": 135923682619392
       },
       {
         "buffer": "vector",
         "end": 191684608,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0aff6000-0b6ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 191204960,
         "overlap_start": 191188576,
         "phase": "before",
         "start": 184508416
       },
       {
         "buffer": "output",
         "end": 191684608,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0aff6000-0b6ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 191304496,
         "overlap_start": 191288112,
         "phase": "before",
         "start": 184508416
       },
       {
         "buffer": "borrowed-numpy",
         "end": 135923749736448,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7b9f36d1c000-7b9f36d1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135923749736448,
         "overlap_start": 135923749732368,
         "phase": "after",
         "start": 135923749732352
       },
       {
         "buffer": "borrowed-numpy",
         "end": 135923783290880,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7b9f36d1d000-7b9f38d1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135923783286800,
         "overlap_start": 135923749736448,
         "phase": "after",
         "start": 135923749736448
       },
       {
         "buffer": "owned-cpp",
         "end": 135923716177920,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7b9f32d1b000-7b9f34d1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135923716173840,
         "overlap_start": 135923682619408,
         "phase": "after",
         "start": 135923682619392
       },
       {
         "buffer": "vector",
         "end": 191684608,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6988 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6988 kB",
           "Rss": "6988 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0aff6000-0b6ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 191204960,
         "overlap_start": 191188576,
         "phase": "after",
         "start": 184508416
       },
       {
         "buffer": "output",
         "end": 191684608,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6988 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6988 kB",
           "Rss": "6988 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0aff6000-0b6ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 191304496,
         "overlap_start": 191288112,
         "phase": "after",
         "start": 184508416
       }
     ]
   }

dense2048-p3-B: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 130491099496464,
         "address_hex": "0x76ae53a1c010",
         "address_mod_2MiB": 114704,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 77218608,
         "address_hex": "0x49a4330",
         "address_mod_2MiB": 1721136,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 130491032383504,
         "address_hex": "0x76ae4fa1b010",
         "address_mod_2MiB": 110608,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 77119072,
         "address_hex": "0x498be60",
         "address_mod_2MiB": 1621600,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d",
       "owned-cpp": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 130491099500544,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "76ae53a1c000-76ae53a1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 130491099500544,
         "overlap_start": 130491099496464,
         "phase": "before",
         "start": 130491099496448
       },
       {
         "buffer": "borrowed-numpy",
         "end": 130491133054976,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "76ae53a1d000-76ae55a1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 130491133050896,
         "overlap_start": 130491099500544,
         "phase": "before",
         "start": 130491099500544
       },
       {
         "buffer": "owned-cpp",
         "end": 130491065942016,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "76ae4fa1b000-76ae51a1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 130491065937936,
         "overlap_start": 130491032383504,
         "phase": "before",
         "start": 130491032383488
       },
       {
         "buffer": "vector",
         "end": 77615104,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0432d000-04a05000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 77135456,
         "overlap_start": 77119072,
         "phase": "before",
         "start": 70438912
       },
       {
         "buffer": "output",
         "end": 77615104,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0432d000-04a05000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 77234992,
         "overlap_start": 77218608,
         "phase": "before",
         "start": 70438912
       },
       {
         "buffer": "borrowed-numpy",
         "end": 130491099500544,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "76ae53a1c000-76ae53a1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 130491099500544,
         "overlap_start": 130491099496464,
         "phase": "after",
         "start": 130491099496448
       },
       {
         "buffer": "borrowed-numpy",
         "end": 130491133054976,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "76ae53a1d000-76ae55a1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 130491133050896,
         "overlap_start": 130491099500544,
         "phase": "after",
         "start": 130491099500544
       },
       {
         "buffer": "owned-cpp",
         "end": 130491065942016,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "76ae4fa1b000-76ae51a1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 130491065937936,
         "overlap_start": 130491032383504,
         "phase": "after",
         "start": 130491032383488
       },
       {
         "buffer": "vector",
         "end": 77615104,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6992 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6992 kB",
           "Rss": "6992 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0432d000-04a05000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 77135456,
         "overlap_start": 77119072,
         "phase": "after",
         "start": 70438912
       },
       {
         "buffer": "output",
         "end": 77615104,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6992 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6992 kB",
           "Rss": "6992 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "0432d000-04a05000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 77234992,
         "overlap_start": 77218608,
         "phase": "after",
         "start": 70438912
       }
     ]
   }

dense2048-p4-A: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 124771681550352,
         "address_hex": "0x717aabc1c010",
         "address_mod_2MiB": 114704,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 655807280,
         "address_hex": "0x2716d330",
         "address_mod_2MiB": 1495856,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 124771614437392,
         "address_hex": "0x717aa7c1b010",
         "address_mod_2MiB": 110608,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 33554432,
         "shape": [
           2048,
           2048
         ],
         "strides": [
           16384,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 655707744,
         "address_hex": "0x27154e60",
         "address_mod_2MiB": 1396320,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 16384,
         "shape": [
           2048
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d",
       "owned-cpp": "d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 124771681554432,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "717aabc1c000-717aabc1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 124771681554432,
         "overlap_start": 124771681550352,
         "phase": "before",
         "start": 124771681550336
       },
       {
         "buffer": "borrowed-numpy",
         "end": 124771715108864,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "717aabc1d000-717aadc1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 124771715104784,
         "overlap_start": 124771681554432,
         "phase": "before",
         "start": 124771681554432
       },
       {
         "buffer": "owned-cpp",
         "end": 124771647995904,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "717aa7c1b000-717aa9c1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 124771647991824,
         "overlap_start": 124771614437392,
         "phase": "before",
         "start": 124771614437376
       },
       {
         "buffer": "vector",
         "end": 656203776,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "26af6000-271ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 655724128,
         "overlap_start": 655707744,
         "phase": "before",
         "start": 649027584
       },
       {
         "buffer": "output",
         "end": 656203776,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6880 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6880 kB",
           "Rss": "6880 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "26af6000-271ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 655823664,
         "overlap_start": 655807280,
         "phase": "before",
         "start": 649027584
       },
       {
         "buffer": "borrowed-numpy",
         "end": 124771681554432,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "717aabc1c000-717aabc1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 124771681554432,
         "overlap_start": 124771681550352,
         "phase": "after",
         "start": 124771681550336
       },
       {
         "buffer": "borrowed-numpy",
         "end": 124771715108864,
         "fields": {
           "AnonHugePages": "30720 kB",
           "Anonymous": "32768 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32768 kB",
           "Rss": "32768 kB",
           "Size": "32768 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "717aabc1d000-717aadc1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 124771715104784,
         "overlap_start": 124771681554432,
         "phase": "after",
         "start": 124771681554432
       },
       {
         "buffer": "owned-cpp",
         "end": 124771647995904,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "32772 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "32772 kB",
           "Rss": "32772 kB",
           "Size": "32772 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "717aa7c1b000-717aa9c1c000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 124771647991824,
         "overlap_start": 124771614437392,
         "phase": "after",
         "start": 124771614437376
       },
       {
         "buffer": "vector",
         "end": 656203776,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6988 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6988 kB",
           "Rss": "6988 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "26af6000-271ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 655724128,
         "overlap_start": 655707744,
         "phase": "after",
         "start": 649027584
       },
       {
         "buffer": "output",
         "end": 656203776,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "6988 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "6988 kB",
           "Rss": "6988 kB",
           "Size": "7008 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "26af6000-271ce000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 655823664,
         "overlap_start": 655807280,
         "phase": "after",
         "start": 649027584
       }
     ]
   }

dense1024-p1-A: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 135503379808272,
         "address_hex": "0x7b3d56d1c010",
         "address_mod_2MiB": 1163280,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 316664288,
         "address_hex": "0x12dfe9e0",
         "address_mod_2MiB": 2091488,
         "address_mod_4096": 2528,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 316724016,
         "address_hex": "0x12e0d330",
         "address_mod_2MiB": 54064,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 316608096,
         "address_hex": "0x12df0e60",
         "address_mod_2MiB": 2035296,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5",
       "owned-cpp": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 135503379812352,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7b3d56d1c000-7b3d56d1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135503379812352,
         "overlap_start": 135503379808272,
         "phase": "before",
         "start": 135503379808256
       },
       {
         "buffer": "borrowed-numpy",
         "end": 135503388200960,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7b3d56d1d000-7b3d5751d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135503388196880,
         "overlap_start": 135503379812352,
         "phase": "before",
         "start": 135503379812352
       },
       {
         "buffer": "owned-cpp",
         "end": 333635584,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "12796000-13e2e000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 325112624,
         "overlap_start": 316724016,
         "phase": "before",
         "start": 309944320
       },
       {
         "buffer": "vector",
         "end": 333635584,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "12796000-13e2e000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 316616288,
         "overlap_start": 316608096,
         "phase": "before",
         "start": 309944320
       },
       {
         "buffer": "output",
         "end": 333635584,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "12796000-13e2e000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 316672480,
         "overlap_start": 316664288,
         "phase": "before",
         "start": 309944320
       },
       {
         "buffer": "borrowed-numpy",
         "end": 135503379812352,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7b3d56d1c000-7b3d56d1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135503379812352,
         "overlap_start": 135503379808272,
         "phase": "after",
         "start": 135503379808256
       },
       {
         "buffer": "borrowed-numpy",
         "end": 135503388200960,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7b3d56d1d000-7b3d5751d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 135503388196880,
         "overlap_start": 135503379812352,
         "phase": "after",
         "start": 135503379812352
       },
       {
         "buffer": "owned-cpp",
         "end": 333635584,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "12796000-13e2e000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 325112624,
         "overlap_start": 316724016,
         "phase": "after",
         "start": 309944320
       },
       {
         "buffer": "vector",
         "end": 333635584,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "12796000-13e2e000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 316616288,
         "overlap_start": 316608096,
         "phase": "after",
         "start": 309944320
       },
       {
         "buffer": "output",
         "end": 333635584,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "12796000-13e2e000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 316672480,
         "overlap_start": 316664288,
         "phase": "after",
         "start": 309944320
       }
     ]
   }

dense1024-p2-B: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 134919981482000,
         "address_hex": "0x7ab58191c010",
         "address_mod_2MiB": 1163280,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 817359328,
         "address_hex": "0x30b7e9e0",
         "address_mod_2MiB": 1567200,
         "address_mod_4096": 2528,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 817419056,
         "address_hex": "0x30b8d330",
         "address_mod_2MiB": 1626928,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 817303136,
         "address_hex": "0x30b70e60",
         "address_mod_2MiB": 1511008,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5",
       "owned-cpp": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 134919981486080,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7ab58191c000-7ab58191d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 134919981486080,
         "overlap_start": 134919981482000,
         "phase": "before",
         "start": 134919981481984
       },
       {
         "buffer": "borrowed-numpy",
         "end": 134919989874688,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7ab58191d000-7ab58211d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 134919989870608,
         "overlap_start": 134919981486080,
         "phase": "before",
         "start": 134919981486080
       },
       {
         "buffer": "owned-cpp",
         "end": 834330624,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "30516000-31bae000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 825807664,
         "overlap_start": 817419056,
         "phase": "before",
         "start": 810639360
       },
       {
         "buffer": "vector",
         "end": 834330624,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "30516000-31bae000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 817311328,
         "overlap_start": 817303136,
         "phase": "before",
         "start": 810639360
       },
       {
         "buffer": "output",
         "end": 834330624,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "30516000-31bae000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 817367520,
         "overlap_start": 817359328,
         "phase": "before",
         "start": 810639360
       },
       {
         "buffer": "borrowed-numpy",
         "end": 134919981486080,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7ab58191c000-7ab58191d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 134919981486080,
         "overlap_start": 134919981482000,
         "phase": "after",
         "start": 134919981481984
       },
       {
         "buffer": "borrowed-numpy",
         "end": 134919989874688,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7ab58191d000-7ab58211d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 134919989870608,
         "overlap_start": 134919981486080,
         "phase": "after",
         "start": 134919981486080
       },
       {
         "buffer": "owned-cpp",
         "end": 834330624,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "30516000-31bae000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 825807664,
         "overlap_start": 817419056,
         "phase": "after",
         "start": 810639360
       },
       {
         "buffer": "vector",
         "end": 834330624,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "30516000-31bae000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 817311328,
         "overlap_start": 817303136,
         "phase": "after",
         "start": 810639360
       },
       {
         "buffer": "output",
         "end": 834330624,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "30516000-31bae000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 817367520,
         "overlap_start": 817359328,
         "phase": "after",
         "start": 810639360
       }
     ]
   }

dense1024-p3-B: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 132849556635664,
         "address_hex": "0x78d372a1c010",
         "address_mod_2MiB": 114704,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 491477472,
         "address_hex": "0x1d4b59e0",
         "address_mod_2MiB": 743904,
         "address_mod_4096": 2528,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 491537200,
         "address_hex": "0x1d4c4330",
         "address_mod_2MiB": 803632,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 491421280,
         "address_hex": "0x1d4a7e60",
         "address_mod_2MiB": 687712,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5",
       "owned-cpp": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 132849556639744,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "78d372a1c000-78d372a1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 132849556639744,
         "overlap_start": 132849556635664,
         "phase": "before",
         "start": 132849556635648
       },
       {
         "buffer": "borrowed-numpy",
         "end": 132849565028352,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "78d372a1d000-78d37321d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 132849565024272,
         "overlap_start": 132849556639744,
         "phase": "before",
         "start": 132849556639744
       },
       {
         "buffer": "owned-cpp",
         "end": 508448768,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "1ce4d000-1e4e5000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 499925808,
         "overlap_start": 491537200,
         "phase": "before",
         "start": 484757504
       },
       {
         "buffer": "vector",
         "end": 508448768,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "1ce4d000-1e4e5000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 491429472,
         "overlap_start": 491421280,
         "phase": "before",
         "start": 484757504
       },
       {
         "buffer": "output",
         "end": 508448768,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "1ce4d000-1e4e5000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 491485664,
         "overlap_start": 491477472,
         "phase": "before",
         "start": 484757504
       },
       {
         "buffer": "borrowed-numpy",
         "end": 132849556639744,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "78d372a1c000-78d372a1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 132849556639744,
         "overlap_start": 132849556635664,
         "phase": "after",
         "start": 132849556635648
       },
       {
         "buffer": "borrowed-numpy",
         "end": 132849565028352,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "78d372a1d000-78d37321d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 132849565024272,
         "overlap_start": 132849556639744,
         "phase": "after",
         "start": 132849556639744
       },
       {
         "buffer": "owned-cpp",
         "end": 508448768,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "1ce4d000-1e4e5000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 499925808,
         "overlap_start": 491537200,
         "phase": "after",
         "start": 484757504
       },
       {
         "buffer": "vector",
         "end": 508448768,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "1ce4d000-1e4e5000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 491429472,
         "overlap_start": 491421280,
         "phase": "after",
         "start": 484757504
       },
       {
         "buffer": "output",
         "end": 508448768,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "1ce4d000-1e4e5000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 491485664,
         "overlap_start": 491477472,
         "phase": "after",
         "start": 484757504
       }
     ]
   }

dense1024-p4-A: observed buffers, policies and every overlapping VMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "buffers": {
       "borrowed-numpy": {
         "address": 129222392791056,
         "address_hex": "0x7586eed1c010",
         "address_mod_2MiB": 1163280,
         "address_mod_4096": 16,
         "address_mod_64": 16,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": true
       },
       "output": {
         "address": 582040032,
         "address_hex": "0x22b139e0",
         "address_mod_2MiB": 1128928,
         "address_mod_4096": 2528,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       },
       "owned-cpp": {
         "address": 582099760,
         "address_hex": "0x22b22330",
         "address_mod_2MiB": 1188656,
         "address_mod_4096": 816,
         "address_mod_64": 48,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8388608,
         "shape": [
           1024,
           1024
         ],
         "strides": [
           8192,
           8
         ],
         "writeable": false
       },
       "vector": {
         "address": 581983840,
         "address_hex": "0x22b05e60",
         "address_mod_2MiB": 1072736,
         "address_mod_4096": 3680,
         "address_mod_64": 32,
         "aligned_for_dtype": true,
         "c_contiguous": true,
         "dtype": "<f8",
         "nbytes": 8192,
         "shape": [
           1024
         ],
         "strides": [
           8
         ],
         "writeable": true
       }
     },
     "environment": {
       "BLIS_NUM_THREADS": "1",
       "MKL_DYNAMIC": "FALSE",
       "MKL_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1",
       "NUMPY_MADVISE_HUGEPAGE": null,
       "OMP_DYNAMIC": "FALSE",
       "OMP_NUM_THREADS": "1",
       "OPENBLAS_CORETYPE": null,
       "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"
     },
     "matrix_sha256": {
       "borrowed-numpy": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5",
       "owned-cpp": "ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5"
     },
     "numpy_madvise_hugepage": {
       "available": true,
       "source": "read-only numpy._core.multiarray._get_madvise_hugepage",
       "value": true
     },
     "private_provider": {
       "config": "OpenBLAS 0.3.30 NO_LAPACK NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY USE_LOCKING Haswell SINGLE_THREADED",
       "core": "Haswell",
       "parallel_kind": 0,
       "threads": 1
     },
     "thp_sysfs": {
       "defrag": {
         "error": null,
         "value": "always defer defer+madvise [madvise] never"
       },
       "enabled": {
         "error": null,
         "value": "always [madvise] never"
       },
       "hpage_pmd_size": {
         "error": null,
         "value": "2097152"
       }
     },
     "threadpools": [
       {
         "architecture": "Haswell",
         "filepath": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so",
         "internal_api": "openblas",
         "num_threads": 1,
         "prefix": "libscipy_openblas",
         "threading_layer": "pthreads",
         "user_api": "blas",
         "version": "0.3.29"
       }
     ],
     "vmas": [
       {
         "buffer": "borrowed-numpy",
         "end": 129222392795136,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7586eed1c000-7586eed1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 129222392795136,
         "overlap_start": 129222392791056,
         "phase": "before",
         "start": 129222392791040
       },
       {
         "buffer": "borrowed-numpy",
         "end": 129222401183744,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7586eed1d000-7586ef51d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 129222401179664,
         "overlap_start": 129222392795136,
         "phase": "before",
         "start": 129222392795136
       },
       {
         "buffer": "owned-cpp",
         "end": 599011328,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "224ab000-23b43000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 590488368,
         "overlap_start": 582099760,
         "phase": "before",
         "start": 575320064
       },
       {
         "buffer": "vector",
         "end": 599011328,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "224ab000-23b43000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 581992032,
         "overlap_start": 581983840,
         "phase": "before",
         "start": 575320064
       },
       {
         "buffer": "output",
         "end": 599011328,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "224ab000-23b43000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 582048224,
         "overlap_start": 582040032,
         "phase": "before",
         "start": 575320064
       },
       {
         "buffer": "borrowed-numpy",
         "end": 129222392795136,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "4 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "4 kB",
           "Rss": "4 kB",
           "Size": "4 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "7586eed1c000-7586eed1d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 129222392795136,
         "overlap_start": 129222392791056,
         "phase": "after",
         "start": 129222392791040
       },
       {
         "buffer": "borrowed-numpy",
         "end": 129222401183744,
         "fields": {
           "AnonHugePages": "6144 kB",
           "Anonymous": "8192 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "8192 kB",
           "Rss": "8192 kB",
           "Size": "8192 kB",
           "THPeligible": "1",
           "VmFlags": "rd wr mr mw me ac sd hg"
         },
         "header": "7586eed1d000-7586ef51d000 rw-p 00000000 00:00 0 ",
         "mapping_exceeds_buffer": true,
         "overlap_end": 129222401179664,
         "overlap_start": 129222392795136,
         "phase": "after",
         "start": 129222392795136
       },
       {
         "buffer": "owned-cpp",
         "end": 599011328,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "224ab000-23b43000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 590488368,
         "overlap_start": 582099760,
         "phase": "after",
         "start": 575320064
       },
       {
         "buffer": "vector",
         "end": 599011328,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "224ab000-23b43000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 581992032,
         "overlap_start": 581983840,
         "phase": "after",
         "start": 575320064
       },
       {
         "buffer": "output",
         "end": 599011328,
         "fields": {
           "AnonHugePages": "0 kB",
           "Anonymous": "23008 kB",
           "KernelPageSize": "4 kB",
           "MMUPageSize": "4 kB",
           "Pss": "23008 kB",
           "Rss": "23008 kB",
           "Size": "23136 kB",
           "THPeligible": "0",
           "VmFlags": "rd wr mr mw me ac sd"
         },
         "header": "224ab000-23b43000 rw-p 00000000 00:00 0                                  [heap]",
         "mapping_exceeds_buffer": true,
         "overlap_end": 582048224,
         "overlap_start": 582040032,
         "phase": "after",
         "start": 575320064
       }
     ]
   }

Decision and remaining CPU work
-------------------------------

Retain baseline production arithmetic and explicit native/BLAS and fusion choices. The eight-accumulator candidate is ineligible under the fixed-work gate. Preserve meaningful independent tests and these negative/diagnostic results instead of adopting a size exception, changing tolerances, or retrying arithmetic variants against observed results.

The next justified CPU investigation is attribution inside actual prepared adapter calls: observe the actual matrix pointer/owner, GEMV operand sequence and provider calls before proposing another implementation. This probe shows that a large private-provider or native-ownership penalty is not universal for the shared rhs GEMV; it does not explain the complete solver gap. Such a new diagnostic must preserve numerical work and allocation/provider policies and be declared before measurement. No further production variant is justified by the present component evidence.

The full A/A calibration is separate evidence: inconclusive controls must remain inconclusive, and a finite set of controls neither proves future host stability nor estimates a universal false-positive rate. No control or started campaign is replaced, pooled across scopes, or omitted from an adoption decision. No CPU speedup or new win over NumPy/PyTorch is claimed by this report.

Artifact identity and regeneration
----------------------------------

The structured audit records all source JSON hashes, exact per-process argv, baseline/core/provider/bridge identities, full work and numerical gates, per-entry engine medians, all worker medians and all VMA observations. The renderer verifies each CPU diagnostic stdout hash against its execution manifest and each storage worker against completion.json. It checks frozen identity equality and complete sampling, but does not rehash binaries or rerun benchmarks. The archive/binary hashes below are the already verified worker identity records, not new hashing performed during report rendering.

.. code-block:: json

   {
     "bridge": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/storage_gemv_bridge.so",
     "compile_command": [
       "/usr/bin/c++",
       "-O3",
       "-DNDEBUG",
       "-std=c++20",
       "-fPIC",
       "-Wall",
       "-Wextra",
       "-Wpedantic",
       "-I/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers",
       "-shared",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/storage_gemv_bridge.cpp",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/libcg_core.a",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/builds/baseline/cpu-blas/_deps/cg-openblas/private/lib/libcgpy_openblas.a",
       "-Wl,--exclude-libs,libcgpy_openblas.a",
       "-Wl,--no-undefined",
       "-pthread",
       "-lm",
       "-ldl",
       "-o",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/storage_gemv_bridge.so"
     ],
     "compiled_revision": "84c88c98c895103395e1b6d6e98f7618848a14d3",
     "compiled_tree": "57a755d537e516b91c70c1b555f2803753389862",
     "core_archive": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/libcg_core.a",
     "files": {
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so": "0bd815d04b6b54990e3cccc7528fbb696456d09569f533d0390c13f0cdc4dd4a",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1": "e4b79735f816cbb906c8bf35b31847c5ce667cfbdbe76bd18aaf6e7baad2686e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/CMakeCache.txt": "9a1dc6e556b6dad28dc1d49491297e5c6a00015b44df41c435644c61c734e6ae",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/compile_commands.json": "7cd5720a683e3378b3048dce648138a042012ca42eb21d59b4e7ab68abffff8c",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/identity.json": "76c69c450baed44d866848639ed8f8f9da3dc4f72f0764d62a9d255a9538bbda",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/libcg_core.a": "fe9159eb82deff9837978f28fbd51b57ad0aeb097bf68fecb3995e60d4ce5352",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/__init__.py": "bc4222c581ef5482130ac60e0732f8b343db48a074a820c18a62b21f016a9c71",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/__main__.py": "892345399d0053c363014a82654c3c9214ca001eb4ba914b8f27b1fd590f6d11",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/adapters/__init__.py": "9bd17a1753b0e2780f859620285847aefa18db4f1d1fe9535c140685edf93315",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/adapters/cuda.py": "322775fceac5a8ed2c375e85fb305b707c0e8cc0f2a6eea261151ade7e922a47",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/adapters/native.py": "0d10c130398260317a57779620515b16a3f8fa7d36e672ed7ea3bd70c7cd67cb",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/adapters/scipy.py": "f23733b88457b1c13e76693913c2c9aaab26fe9ae624f487e5bcea2344dfe5c9",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/artifacts.py": "92647f9ac365a6ae967fce0de7a63418d6cd5a4d761180705faa743b2228ae8b",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/cases.py": "6b0b9941cca645dbd155675292eb010c1930d8140c7ceecc057cfdd6c493d7d2",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/cli.py": "55697b379e4bde4489c0bb8c9b7afb48b39dcd3514986e098b2877f82c643140",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/collector.py": "ecf527e5da8dc3eb96132f51700bb0ec34e36c2edffd5ceb32cab4cce5f2324e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/cuda_metadata.py": "aac9c0f3e8611617a1422fb0b72207533141ccf729ed211e6c495573b828b7fa",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/cuda_runtime.py": "b6498e14bc96f58ebba8eefb7892c60962c29b4d0876440a894e7ecd0daf0742",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/execution.py": "757e046b81b11185105a22031d0e6ac1511ac5dc084c37facfb291158d10f7f7",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/json_io.py": "abc0f80185ec57a46718e09f570bc7ba17f87677495e9d776e5e10064b396a4e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/numerics.py": "5f6362150929fb3b22bd7c933a2867424341640e5a5efa4c1d2382a7706e1046",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/__init__.py": "94dc33f0b15c64058ef7c129695aca98b1de3a591cf73a2c7ff8da274f54d592",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/engine.py": "8655689f7e1a682e6f7369a931bed37ce33e0c9fd7446fc5edcf4e6b8b56dd73",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/numpy.py": "051db413a8948c43fae6c4613797e13e08cedad425656afafa0ec6ecd804018e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/operations.py": "8f6e97d8783dbde3d772281bf95fe78d56fa114ff97bab172b5a5acd0af66a07",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/scipy.py": "47366e93fe1a4dcdbb64bad6de206ef3363cc81de88de9c08f7ffc5a8efced4e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/torch.py": "31d1f93bbd88b16617342b2c1a46dafccd2709ac5798d76fd367fa0445477870",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/reference/torch_cuda.py": "a4cea1db2e0c30c33cea4d64a632dd2b0cf37242707af65091f0ad0b53fdd2ae",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgbench/validation.py": "f785e07bc39bd2025c82bb053b2227d16d65e30b3ce362f85d2e11740d36a29d",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/__init__.py": "9e217f2851fa2cbf76ecac2f94b06acd69b4bb247f9402e56aaf69c4420b838b",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_core.cpython-312-x86_64-linux-gnu.so": "81f5e99ede9ebf0d62e2ffcef1b4529183ce335a9d2e713d6a2d5373c7f63702",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_core.pyi": "e883ac42f577625812728e8869474972d9a2d09725098c9a1ff858240d3663d8",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_cuda.pyi": "6d16441c0732c2df61ae5e76a0be858dd1a7f845d48511ced25428929210aa15",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_interop/__init__.py": "1073944d645396dd672b8a0985dbce02d8f9b6d6c323c5f61ea0d9b36a6f08c9",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_interop/cuda.py": "31d35ae7794836a7478831e6ec9ab2fc0f1561320372ecb82b0e47a03dbcd614",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_interop/host.py": "0975e8877dc50fede372ef56e76f2753d0d50ab3f5ada4e3d777a0dea4df111d",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_runtime/__init__.py": "c5989ff1781cf1f36785abb29d034adbc69f9f23333346965c8c0fb572f172ec",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_runtime/cuda.py": "f49c477bae3fb97caa9bd9631a0456242a5ebf7a98e7732dc0b98686767bc670",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/_typing.py": "52baa05a23d8485a5fcd914e533104d1e3eab6c4c5245fc6d13c88fda9f2fd74",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/cuda/__init__.py": "cd5e84a5d355dfed74c1e8c6db41d69e836f0a9554574380e2c63f92940c22a6",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/matrix/__init__.py": "590054fc4f2f1276b5e36ffb91f92b56e6a629ae76f95b3285db64f6eafd1c1a",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/matrix/csr.py": "676ac49ef0b5652cb5c7e85a375bdd224f6ea4faffa5d2f8d1c168632bdcb730",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/matrix/cuda.py": "679ff441c676c215360727dd6a5756f8838423a2fc2d29ac3008ca099f29c284",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/__init__.py": "624bc31b7377ea9412d21d96d4f9a2d2f8621df06d6fc74a55df784bc4989c12",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/_validation.py": "d71660178ef8a73c734dd56cd81d97c9c97dbd58e1b41226428a2cefb3a4f07f",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/cpu.py": "579a866fd34fcf1b7f17f73fc731ac52d0bcda17eae818c3dc8adc9b59813953",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/cuda/__init__.py": "dc4c4079fd45e10519d82be01798011e0e5cccba661b35635113beaf52030b03",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/cuda/api.py": "91b83f3db21d94ef6cc898d52373a80c3986eac5d2c26e0dd3c93787117f9a93",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/cuda/prepared.py": "d0105e268289c79479fb7212bc8602727b32f500c8c10de9a8682c310aadf015",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/cuda/session.py": "cc4470ef44114d0bb18f61d6b8db5c1b84eb25e88fd2ad495090f5dbba60cc7c",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/cuda/workspace.py": "2e5dfd824d7bfd7e4dfd6e51f48060d8bde07fbfba100a78ed920bbc4f1ba200",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/result.py": "4f0a5a741d6e5469849ed2a628d7ae9c4390bf0a8316d6dee42e50e4826e5f53",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/solver/workspace.py": "ff1bab4209342c3569c88ad051ee4f706413284a61bdcbede1116446b870051e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/storage/__init__.py": "02c0c344773a6eb71e9fac5c9dea3379823f97cacf70bf382ffe2f7bb42d9a43",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/baseline-stages/cpu-blas/python/cgpy/storage/cuda.py": "5fd60bda9eb52f370c34c2915e7c9a08e3195357c0a230ee511b5cdb50680d55",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/builds/baseline/cpu-blas/_deps/cg-openblas/private/lib/libcgpy_openblas.a": "ab29acd4627d2872fc6d4d3a7fa735afbc5983a8b6fd99dcdcf636fcdccceff7",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cpu/backend.hpp": "265479a84eb2313d23b821ecb04048a93a725370a6278b102a4caf0414a05409",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cpu/detail/blas.hpp": "2550dba71feebf9338755e53124cf0a322fbd35c18479c8a20f17a1c19598ff7",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cpu/detail/parallel_thresholds.hpp": "5a9f814670e756c52370de49a3458fc6f6df72491b33d2f8e56fedf560339191",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cpu/detail/reduction.hpp": "f4ff634b596462b7bf847c37491e9d25de8222d370490372350730904b26a442",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cpu/detail/validation.hpp": "9a4a19ccd137f05b8230bf37eac949bcb2d1d0096d234d5b6c639733a25b5244",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cuda/dense_kernel.hpp": "c93207a8b78569e8e43690779ce863dfd2887e3f1355bd7671bb856dff14fb7e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cuda/detail/backend.hpp": "f167306a693fc673c18b8950848de34a0897c4d37cb07028def9e25905775857",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cuda/detail/blas_handle.hpp": "db8a16b1b141c164846ba1dd4de5c6b13b40ed009fa545ef1cfd1b6a63428beb",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cuda/detail/execution_resources.hpp": "95cc529b523923727d9ae0bafc731470e41cc445ce6c94bcf3d789dfd6602b71",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cuda/detail/reduction.cuh": "b92599be04bedae16a5c7150fe9da7e736cd8c9bc7fa1ad25dc5b6a28bf0c891",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/cuda/detail/validation.hpp": "6b65a251c0383ab882317e91c3e113803997856fc1efe605f90a73d34f4b038c",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/backend/residual_sign.hpp": "6d10cde48e611fb016f0a083e2ba6a59758a4b5ca1875e7948f2c72db4c2c3d0",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/cg.hpp": "a3255b617fb5034ffa142e91c164d1ba253ea34d85b404cbcbe5e790fe283310",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/cuda.hpp": "5fc07eb16f20885527da4fca5c0160b53390f6f7534f7e49d870ab3515c50b23",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/csr_matrix.hpp": "6a91a0798962933b85c15921c5a9511a3cb5b637dec5415e4f201af519afc022",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/cuda/csr_matrix.hpp": "5d54a44cacd8ea0c372616a201d2b5a52aef42d1aba1d95d9031baec21f39888",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/cuda/csr_matrix_view.hpp": "452abcee5c6cd226c7111deabd5a77b7816d4544256a9f93e8063cc3433ffe0a",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/cuda/dense_matrix.hpp": "6e58badbbea4ca08ed9104113780e8811eeb6cb7aeb124f6cda2243b4119a1e4",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/cuda/dense_matrix_view.hpp": "29a96eb6173d3d9c3a105142e484a9c42d96c7da02a94c65bf7ef7f26c079aee",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/cuda/detail/validation.hpp": "69f62fab7b24190f980d07b763e0b29973bae898d301ba660bc195e9ea42e614",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/dense_matrix.hpp": "fa602ba210a68ce2e74b75d4f50c1d3d517296c9de46ebbe2007b732f0b4a211",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/dense_matrix_view.hpp": "0869ab54e368c0dd4dcf91e2a877d48a0490147df5fe27af5052e425e29c282d",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/detail/csr_validation.hpp": "c908c81f73a199fde2e34a1739c300258d39fb903ce3eb33199d2565e3cc2168",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/detail/storage_extent.hpp": "64ac2d1b3f97fe41f9cceabbf44e623e8241f75ad573cb567b9bfd16c52e05b1",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/matrix/detail/validation.hpp": "a5593cf84c0e8f5d67dd14e215d3f677c82806d5938b9379b6b85d9a01e40bcf",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/preconditioner/cuda/detail/kernels.hpp": "53cd02810393334b960eeef986f0ba6c7623264ef0d233e09c8c3188d6ebc960",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/preconditioner/cuda/jacobi.hpp": "077a07d3d8fe990d10e9a30fe0e7e4132f111b636825d78001e756e274b85df7",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/preconditioner/jacobi.hpp": "850d8224fb1d16dcadb061cd676c4ef1e45cfb5e48a6d0b04432e32853d3b239",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/preconditioner/kind.hpp": "419d65ab025c9d87a78ccde2b8116fb7cce6d4263a0891efb1ed81929bc9e575",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/python/array_conversion.hpp": "1a62cbac4c23096b7c5081ca48f94946de99664a5c64b71f1ea9833b4940b66b",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/python/cuda_array_conversion.hpp": "b31bb7fff0907d68f96bc915610cf6075dcf4aec82f885796a32908163f9360e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/python/module.hpp": "8060885b993da1a46170db52a3908ee63dcad7db5ec8682c6dc81dadcbde5a9e",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/runtime/cuda/context.hpp": "8e50646c61186fcba57927d1cd8d557d038961218ab1fcfc81c3d6a85ff1245f",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/runtime/cuda/detail/event.hpp": "0a46f9a4e81ddfa285bdf31155c6e77f8cbc7cebdb96295b6ed04e6c946b3b7b",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/runtime/cuda/detail/launch.hpp": "c25bb420865c690700d3463da522e272c866f17eaba01f17f59d53e6431fe931",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/runtime/cuda/detail/runtime.hpp": "ae49ae02c985675947985c85274945f52a3611ada0b5603ef00aac459aed892c",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/runtime/cuda/host_device.hpp": "4c7a6cfd83f2f18fd5c8bbb50f086840e813b51ab16fea6db78d8aa437160e20",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/runtime/detail/checked_size.hpp": "0ff643ef82dc0986a25aa60985d4ea2b6968b7cf3f843c221dd697a939c75aa8",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/conjugate_gradient.hpp": "cb87970f2de4e2fb80a165240b22c62e0e012bcb0a3c29edc76eec19ba098961",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/cuda/conjugate_gradient.hpp": "e3a0b4abe1bd1662adfea6312a812f258c81335dee43320f7798e5d8136dcbab",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/cuda/detail/device_execution.hpp": "d223a1b2ba0dcdc5595a117c74efb673e694a10230e587a053998288c31be6f0",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/cuda/detail/workspace.hpp": "ee272266c8d54130772f9221145d3ef5bebb642086910f671987932537836f65",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/cuda/detail/workspace_use_guard.hpp": "d865f3030d1686b17903620aea48a226d1960c2837ff43c6e3c89c751682fdea",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/cuda/prepared_solver.hpp": "39d438a711a8e59e41616978807c6304f78ca09e99ef00c3bf89a63b5f4c05a5",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/cuda/workspace.hpp": "dd976e57d56d88ce1b647f7e57361e86bc05474625eb5468c2f4d4a627cc4602",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/detail/engine.hpp": "bd4e6846e73954488c54d05a514c6be30443cc7eb1aaeef60972673059b44a99",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/detail/numerical_policy.hpp": "3c9f3568495650a0ea0061c09ec254188d39f22d57d335779da63b98f476e349",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/detail/validation.hpp": "83bfd2ec0181e307bb76e75c8a9464ee5ce674faa5487dee91f4c55d869a8135",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/options.hpp": "cc39d5a585595cb7189e1a84e96cb999a2bfbd1ffdd1bbcd7be0a486293efa15",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/prepared_solver.hpp": "d674843b42f9580ef1492842d8e7bdd8bafa337c0437ef217885febd84988ad6",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/pymod/options.hpp": "08c6e9b9a112186b136c0ee275c0687959e1b69b5e584ec9ae28521e3b981ff0",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/solve_info.hpp": "1953f0ffa1209e6a67cdbdfb6ab2e95d185ec5760d39604a40c99390f0705b82",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/solver/workspace.hpp": "5018a796a366defef9c184b93c9db2507f91dc80ac170d27c87df124c5a71c57",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/cuda/detail/fill.hpp": "2b06e8ba220b070f1d745dd484c038428855655f5af51fc283ac7d5c128d4f30",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/cuda/detail/validation.hpp": "aad6152bd2eb8b91baa79098b024768c74668bc6178fdda70801da03a3fba5cf",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/cuda/device_memory.hpp": "b7dbb2f83cc509b3be5103ac2030fd2a18e3d381b1437b85944f3293725c8301",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/cuda/vector.hpp": "f5846f417461ea250d48cbbb0f894b03e8fef950733b0ef77c7985969e06f94b",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/cuda/vector_view.hpp": "eb378341b123e6c9726252da8e20ad8403cc86d31f3165138e66d0e6547b9890",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/detail/validation.hpp": "54bf0c30bfb601a31f43083110bcd1e1c9e9ce7bdaccfd1f6f534c8fed680c2c",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/cpu-profiles/headers/cg/storage/vector_view.hpp": "dc49a8ef869421d50c8184a4b82dd750fe4d954b67252900edceb0bdac93d889",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/setup_storage_probe.py": "465131b06258f9309787f0594ea33a42ccb4edd61f8c9b81098225adc43a23d0",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/storage_gemv_bridge.cpp": "c1693ef3b1885c82b1a5e9c587592b03ad1e859ed97ac7b4a54e68b595a4edbd",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/storage_gemv_bridge.so": "5cefdd9e0ef9b290560e6e5a66719affbf131957ec08c1c90f8f35132899975d",
       "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/storage_gemv_probe.py": "64b5e41ad23373476b5f328cacda497d8206907253604e3a5b7c2e74d16dae0c",
       "/usr/bin/python3.12": "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f",
       "/usr/bin/x86_64-linux-gnu-g++-16": "546ddb2e5c2aadbdbcf2aaab30b8ac4f72abb8bb8f5fd51dc4bf6864a29491d7"
     },
     "fixture_hashes": {
       "dense-1024": "204aa61929e5e8d34102ac2d0bda5714b6f0d17f8bf4796a7f71b535a21053bb",
       "dense-2048": "8bda02fed9ccafa8c12f857ad51962b02d8c860ee96a006078aed872c40cae6c",
       "dense-256": "6a6622788ec7f29156523e562995e09db8ec35032185f8c76ae6fccf2165a4a0",
       "dense-64": "b2e274e240704161b83bff5de50cf7b8ba9e3848b098f027c1ba8b5fccf82a29"
     },
     "private_blas_archive": "/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/reliable-performance-20261018/builds/baseline/cpu-blas/_deps/cg-openblas/private/lib/libcgpy_openblas.a"
   }

Regenerate only from the preserved local evidence after quiet measurements finish. This is artifact rendering, not a command that reconstructs missing baseline binaries or reruns performance samples:

.. code-block:: console

   python3 build/reliable-performance-20261018/render_cpu_decision.py \
     --output build/reliable-performance-20261018/cpu-decision-report.rst \
     --analysis build/reliable-performance-20261018/cpu-decision-audit.json

All source-artifact SHA256 values read by this rendering:

.. csv-table:: Small source/JSON artifacts only
   :header: "Relative artifact","SHA256"

   "cpu-candidate-decision.json","aa270020eb558f4be85b539a0ec459856940cdf2fe2f366e9a910a49315ed687"
   "cpu-codegen-halves-note.rst","83f78bb7061fcefd65c1809ee68c5a8931aa5e1e5c16b0ffa8392b775c013af5"
   "cpu-profiles/compile-manifest.json","2b22fe31909a04d6edab729893281917ffacd2016ba09b90e443ebb900969636"
   "cpu-profiles/cpu-blas-blas-explicit-none-fused0.json","6525880bcb8cbef523d9dabfaac23bb003a846435320dcba1506223edeffec70"
   "cpu-profiles/cpu-blas-blas-explicit-none-fused1.json","82d3cddb3c7013713377d4aed7445e573e18f88715f3edc89a655dea74bba8cd"
   "cpu-profiles/cpu-blas-blas-standard-jacobi-fused0.json","4ae9abc2920582dd01f7194958ddb173b8f62c6e67df36d6ea092df93a828d0c"
   "cpu-profiles/cpu-blas-blas-standard-jacobi-fused1.json","a834ede091e6672e4599e744baef837e8c1a47dd3cc4c1425ede9c5ca0157d11"
   "cpu-profiles/cpu-blas-blas-standard-none-fused0.json","c7e395524af754e4aa04e1e08f5977491ba1d5d15a5c7d3e2f6400f1e365eaed"
   "cpu-profiles/cpu-blas-blas-standard-none-fused1.json","9530f642642e06110793f8e19f2d685736d6604811411cef407e6bd7770f3113"
   "cpu-profiles/cpu-blas-native-explicit-none-fused0.json","4c1ff1c79b544a3cb1e1fdfe3e532bedde0102f7cbde3b4b4bd7fcbf038501cd"
   "cpu-profiles/cpu-blas-native-explicit-none-fused1.json","ed24b4c43abac1cdfc2bfe85270f35985e8edd4f4aef0a6999dd41eeee99d496"
   "cpu-profiles/cpu-blas-native-standard-jacobi-fused0.json","c1a05c23ea8e13aa36183085da80e27263de4cdbbb3e1b704d8d0935736939da"
   "cpu-profiles/cpu-blas-native-standard-jacobi-fused1.json","9b087011053a8ae17cd6c441576b6afe5550c7c00db61d4ad8a9868ce3c01695"
   "cpu-profiles/cpu-blas-native-standard-none-fused0.json","c93066e839e2c913f02a149eb7c45ec1cc345a82cc441191c5fea0779a61b281"
   "cpu-profiles/cpu-blas-native-standard-none-fused1.json","bf27114f3b1dc8655dfb5097129d12958cce984be10c9f2beb0dde73847406ac"
   "cpu-profiles/cpu-openmp-native-explicit-none-fused0.json","0cd0b3910933c709be5ab4f441f4b36f7b1840c1244bf4f9799053995b2b8944"
   "cpu-profiles/cpu-openmp-native-explicit-none-fused1.json","2915bf78588a4d26d2f22b4c8283e4aecab90421f9e64e4bf8ed95f1a1d2c38d"
   "cpu-profiles/cpu-openmp-native-standard-jacobi-fused0.json","251bcc74aec8629c3b3712f096763dc2a999766aaf780d3ffec61b2cc979ed09"
   "cpu-profiles/cpu-openmp-native-standard-jacobi-fused1.json","bd8060305e360e2084fc2f9131afdc3b4c638f9fac1054b87c9d78e98a6fc059"
   "cpu-profiles/cpu-openmp-native-standard-none-fused0.json","98d1fab8e3ea22539c5ae8af08cc5dec2b9cad5fa35ad574e525846f783371ff"
   "cpu-profiles/cpu-openmp-native-standard-none-fused1.json","c2f1eaf67c1700add284cc3d5ea9fed563e87f42fdf93a2a1cdc5b1d06fcf502"
   "cpu-profiles/cpu-release-native-explicit-none-fused0.json","f1d262f19fc2c5205d65420d79dc71fced02852cc3b6d833a2f98f1be4ed8a47"
   "cpu-profiles/cpu-release-native-explicit-none-fused1.json","ff8b886ba0148a7a412671ee865f77ffcda8b807ed4702a42b99724e1a8e3d6f"
   "cpu-profiles/cpu-release-native-standard-jacobi-fused0.json","39373eb3557acd5a0d45e93f04e71b41159d7a3b90d84c5c03313c8230297d50"
   "cpu-profiles/cpu-release-native-standard-jacobi-fused1.json","54eb1b071326f2be86e95f2bf75d761d5e702e807e4574f25c80d8663af8ae3c"
   "cpu-profiles/cpu-release-native-standard-none-fused0.json","eb3865a8621b3e086b4ba0a54367a3c70c119346aa2c6601795788f6e91b6bdb"
   "cpu-profiles/cpu-release-native-standard-none-fused1.json","ad932d9d5dc285b90acff3d0fbf3cd0415b6137c3144ec14d6334916e334dc13"
   "cpu-profiles/execution-manifest.json","322797b7bd79d2b91408c15c78d3c1cb691f9d2cecba7dbc3e20dfd270752aa6"
   "cpu-profiles/execution-plan.json","416ed10a1392e0a4f998d72242a1dc13063cc81ec00c46779443eba9b0eeab16"
   "cpu-profiles/fixtures.json","3f6d71680c9048e623efed5a7bdec6fb5f1535cb33f04c4af5fe3e9b38c1b742"
   "cpu-profiles/private-blas-identity.json","9c8ae849cedcea6f62d901762fad541fc2ebd5f67809526c0cb2378ee395a34c"
   "cpu-profiles/provider-compile-manifest.json","4cc0ecd7a647c0c1229f5be45d24f758d61154c6d993c266f8734437c302aa14"
   "cpu-profiles/provider-explicit-none-fused0.json","082f6d814e06332308624cc5e942fa9d1c4a3e54608e9036a51f1d0fa581fe77"
   "cpu-profiles/provider-explicit-none-fused1.json","bbd5ad08a6338fd60cfd6d2e0f82adee5c33a924937940a76cd71a1c1016b810"
   "cpu-profiles/provider-standard-jacobi-fused0.json","50bdc6f0abfa1e09490ac9e0fcc8652a785525128cfa54dd393b3898f517feee"
   "cpu-profiles/provider-standard-jacobi-fused1.json","4c475f305148d413932aa88188a1e800021263aa9e9747c616dadaa13f1fdcd0"
   "cpu-profiles/provider-standard-none-fused0.json","9d2c36db348e4123cd20a702f0bd8471b797306f0b39379c81d4d228e014fdea"
   "cpu-profiles/provider-standard-none-fused1.json","58e2988f5312ba268c5f8f04f6f8df1b0fd17a712caaf8750613a22136b9d744"
   "preflight-v2/manifest.json","b48fe10ee4254bd09432e46e79a3e7ebf82045f99bfe91371ac0cc949d986ff9"
   "preflight-v2/plan.json","e691e7e10bbda5c0353a339e5b844d77343748b7721f1297ac1903b492d1e737"
   "preflight-v2/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-baseline.json","4dcbd4b8af91cfd1a569c476759da307cc855c79eb991445d7b3ea637244fbf8"
   "preflight-v2/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-standard-current.json","b98a394eecc7d60a450f1ac38d6a8dca91a8fd1dd72207119af04dfc2e5a9d4e"
   "render_cpu_decision.py","136d3af7cd6c51982bc94323e106b5e88d393f7de35f392ea98553c4da1ef404"
   "storage-gemv-identity.json","de99db06b7a0eee4cfbb63aabbc2553d97beca07753bbe9a72c11cfff4073594"
   "storage-gemv-results/completion.json","1d1a1635c9f633b7b0157d5ac6584e877b6c3cc461e073b730c702b722cc8aed"
   "storage-gemv-results/dense1024-p1-A.json","c76d7397d882206c7fb4514cd2b6d526074c9adab33e28498c6b34a81ba4e961"
   "storage-gemv-results/dense1024-p2-B.json","a4cb0e39144f26d40d6847c2224a8b899759ba77ed3a31b4d7e390d53436211b"
   "storage-gemv-results/dense1024-p3-B.json","aa2113bc39fc1c7a197d6bfdc9cc2dbb0e0d4d3138c883a32227761c2f090cff"
   "storage-gemv-results/dense1024-p4-A.json","6070a2569e00a83d10ad5a46976639ad95066fa3a86ff28bcf87dc26f6ca9465"
   "storage-gemv-results/dense2048-p1-A.json","c9906002f12175f69e2681c7c6a620ea568d6a4553699ef6e259004f02f08413"
   "storage-gemv-results/dense2048-p2-B.json","4b30bf5eadb0a8c63b987e4ee117ae6ed04579780c2e6cbd2db6e562f5081c53"
   "storage-gemv-results/dense2048-p3-B.json","0255427bd90e1709cee0b8e79a50850e08a19aebf22bbd8849bb757d2834982f"
   "storage-gemv-results/dense2048-p4-A.json","a451b3bf9b112b6905093698c46344249c57b92ec1618b8fd0ab335c854f0e51"
   "storage-gemv-results/plan.json","9417a92035b66d6380cf7071e210aa0cc122d9b8b2fafe93be7cb29ac7376b48"

