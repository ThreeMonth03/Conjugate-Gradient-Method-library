Restart-aware optimization decisions, 2026-10-17
================================================

The CUDA ownership consolidation and stable-norm/control launch change are
retained; the CPU four-row GEMV proposal is rejected, leaving the original
two-row kernel. The initial CUDA resource variant is also rejected and remains
separately reported. No complete wall-time study establishes a universal
speedup or a pass for every row's 5% regression budget.

The final supplemental CUDA study has no confirmed regression, but its whole
verdict is inconclusive. Its 64 native/cuBLAS-to-Torch comparisons retain
56 lower and eight higher measured medians. All eight higher medians are
cuBLAS Jacobi fresh API: all four sizes in host and device execution, with
ratios 1.108486–1.463973. Native CUDA's 32 comparisons and all measured
Session/prepared comparisons are lower descriptively. These observations
cover the declared fixtures and scopes, not statistical or universal wins.

All five A/B campaigns retain **176 fresh-process artifacts, 2,144 eligible
outcomes and 536 strict candidate/case rows**, including both rejected
variants and every framework control. The 57 instrumented CUDA captures and
84 CPU component observations are separate diagnostics. The preceding
:doc:`restart-aware baseline <restart-baseline-20261017>` retains its own
108 artifacts/1,336 results and 668 characterization rows; none is silently
pooled into this evaluation. Its unremeasured modes remain historical context.

Completed ownership decisions
-----------------------------

The baseline was compiled from
``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``. Its stable public source tag is
`benchmarks/restart-baseline-20261017
<https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/tree/benchmarks/restart-baseline-20261017>`__.
The rejected resource-layout variant was compiled from
``e5c587889906511b0f9e74f42d88d8e857dea3d0``. The separate borrowed-span
follow-up was compiled from ``ce45ea9cc6d38cea05ef04cff8a321507e9684d5``.
Final merge heads do not replace those actual compiled identities. The
complete tables retain actual source, stage, module and manifest identities;
they are not inferred from a later checkout.

Each ownership campaign contains eight configurations, 32 fresh-process
artifacts, 384 eligible results and 96 strict candidate/case rows. Both retain
all same-stage I/M/R gates and all native/cuBLAS/Torch observations.

.. csv-table:: Completed ownership campaigns
   :class: benchmark-results
   :header: "Variant", "Whole verdict", "Within 5% budget", "Inconclusive", "Regression"

   "Initial resource layout, rejected", "regression", "12", "83", "1"
   "Borrowed-span follow-up, adopted for ownership", "inconclusive", "34", "62", "0"

.. csv-table:: Every implementation's strict row inventory
   :class: benchmark-results
   :header: "Variant", "Implementation", "Within 5% budget", "Inconclusive", "Regression"

   "Initial", "cuda-native", "6", "25", "1"
   "Initial", "cuda-cublas", "2", "30", "0"
   "Initial", "torch-cuda-eager", "4", "28", "0"
   "Follow-up", "cuda-native", "13", "19", "0"
   "Follow-up", "cuda-cublas", "14", "18", "0"
   "Follow-up", "torch-cuda-eager", "7", "25", "0"

The original regression is native dense-256, Jacobi, prepared scope and host
execution: current/baseline ratio **1.16509**, 95% interval
**[1.05821, 1.25940]**, with unchanged **4/6/0** I/M/R. It remains in the final
report even though the later variant has no confirmed regression. All other
initial rows, including higher medians and inconclusive controls, remain too.

The follow-up was selected after observing the initial failure. Its 95%
intervals are unadjusted for this selection and for evaluating another
variant. It has its own fixed complete ABBA campaign, retains the same budget,
and does not replace or pool with the rejected campaign. The observation does
not establish that host-object layout caused the original regression. No
control drift is subtracted, and no timing result is normalized by iterations.

The ownership change combines reduction/scalar storage and the optional full
4 MiB cuBLAS scratch prefix in one RAII allocation. An explicit borrowed byte
span supplies the provider's scratch; its handle is destroyed before that
allocation. Fresh solve still owns fresh output/workspace/handle and optional
Jacobi setup. Session and prepared contracts are unchanged.

Recorded diagnostics support one fewer cudaMalloc/cudaFree per fresh cuBLAS
solve: ten fewer of each in the matching ten-call capture. Complete API
inventories retain the extra single driver ``cuCtxSynchronize`` observed in a
native-API diagnostic; not every synchronization count is unchanged.
Allocation counts establish an allocation change, not an additive
latency contribution or a general completed-wall improvement.

CPU and CUDA launch decisions
-----------------------------

CPU four-row GEMV
~~~~~~~~~~~~~~~~~

The complete ``cpu-quad-abba/manifest.json`` retains 16 configurations,
64 artifacts and 832 eligible candidate/case results. Release includes both
fusion settings; BLAS and two-thread OpenMP retain their declared controls.
All 208 strict rows remain: 64 native-kernel rows, 16 private-BLAS rows and
128 NumPy/Torch controls. Every known I/M/R gate matches. The whole verdict is
**regression**, and the four-row variant is **not adopted**.

The measured four-row source is
``2d81b3d1409b0b428572cbf26fc7418e03bd270c``. Tables labelled B/current describe
that rejected measured source, not the production checkout. Only expanded
correctness tests are retained from this CPU proposal; production keeps the
original two-row dense kernel. No crossover threshold or automatic selection
was added. The complete source/stage/binary identities and remaining-loss
inventory below belong to the measured variant and must not be relabelled as
latest-main performance.

.. csv-table:: Complete CPU strict inventory
   :class: benchmark-results
   :header: "Implementation", "Rows", "Within 5% budget", "Inconclusive", "Regression"

   "native", "64", "15", "45", "4"
   "native-blas", "16", "5", "9", "2"
   "numpy-out", "64", "16", "41", "7"
   "torch-eager-out", "64", "12", "47", "5"

The unchanged BLAS provider and NumPy/Torch controls also retain regressions.
These observations limit causal attribution; they neither justify subtracting
control drift nor excuse the proposal's failed adoption gate. All cases,
including improvements and inconclusive results, remain in the tables.

The separate 84 fixed-vector component probes are retained below. They are one
descriptive baseline pass and one candidate pass, not a confidence study.
They use the four official seeded dense fixtures and three padded integer
tails, with the vector fixed to b rather than a changing CG direction. Matvec,
fused reduction and whole-solve observations answer different questions;
component medians cannot be summed into CG latency attribution or used to
overrule the complete rejection. Odd-shape OpenMP rounding still requires the
retained correctness tests; the component probes themselves use one thread.

CUDA stable-norm/control launch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete prepared/device study retains four configurations, 16 artifacts,
160 eligible results and 40 strict rows. Actual A is adopted-span source
``ce45ea9cc6d38cea05ef04cff8a321507e9684d5``; actual B is
``14c37277de16182fdc7f4cfcf9193e0b352791d1``. Its whole verdict is
**regression**, with the sole strict regression in the Torch control. No
native/cuBLAS row is a confirmed regression; ten meet the 5% budget and
fourteen remain inconclusive. This is not a complete-gate pass.

.. csv-table:: Complete prepared/device launch inventory
   :class: benchmark-results
   :header: "Implementation", "Rows", "Within 5% budget", "Inconclusive", "Regression"

   "cuda-native", "16", "7", "9", "0"
   "cuda-cublas", "8", "3", "5", "0"
   "torch-cuda-eager", "16", "7", "8", "1"

The Torch dense-64 Jacobi prepared/device row has current/baseline ratio
**1.25757**, 95% interval **[1.12458, 1.42341]**. It remains in the complete
batch; no control drift is subtracted and no unrelated native ratio overrides
it. All 24 native/cuBLAS medians are lower than their same-artifact Torch
controls descriptively, without a confidence claim for those framework
comparisons. The full candidate/control I/M/R and true-residual gates repeat.

The kernel runs the existing stable norm collective before residual/beta
policy within the same 256-thread block. Initial and true-residual checks
retain separate norm calls, while the ordinary post-update norm launch is
merged. Arithmetic, residual priority and completed packet checks are retained.
Explicit invalid-alpha exits save no launch. The six completed diagnostic
captures are retained below as instrumentation, not completed-wall scores.

A separate supplemental scope study was fixed before any launch ABBA results
were inspected. ``launch-scope-preregistration.json`` records UTC
``2026-10-16T18:33:53.770121+00:00`` and plan SHA256
``3e9f579275d06bb4936f96e4759aeb120e2d7a96344b17750c1728b264887de8``.
All eight configurations and 32 processes completed, retaining 384 eligible
outcomes and 96 strict rows. None/API is included in host and device modes;
Jacobi is included in API, Session and prepared scopes in both modes. Every
artifact includes native, cuBLAS and eager Torch on all four dense sizes.

.. csv-table:: Complete supplemental scope inventory
   :class: benchmark-results
   :header: "Implementation", "Rows", "Within 5% budget", "Inconclusive", "Regression"

   "cuda-native", "32", "22", "10", "0"
   "cuda-cublas", "32", "22", "10", "0"
   "torch-cuda-eager", "32", "13", "19", "0"

The whole verdict is **inconclusive**, not a complete-gate pass. Its observed
work repeats across all four blocks for every candidate. All eight remaining
higher framework medians are cuBLAS Jacobi fresh API; the complete per-size,
host/device timing and I/M/R inventory below retains them. Session and
prepared scope savings must not replace that fresh-call contract.

The study compares adopted-span source
``ce45ea9cc6d38cea05ef04cff8a321507e9684d5`` against launch source
``14c37277de16182fdc7f4cfcf9193e0b352791d1``. The frozen launch-stage identity is
``daf34907780c5b6fd366347ff81161904289228e9cf5960e85c0c669887e6466``;
its CPU binary is unchanged from the adopted span stage. The initial prepared
study and supplemental study remain separate, including their repeated
prepared configurations. Neither replaces or pools with the other. The
prepared study's Torch regression is retained. Supplemental intervals are
unadjusted for the exploratory sequence and multiple studies; preregistration
before inspecting launch results does not make the whole sequence independent
confirmatory evidence.

Current-source coverage
------------------------

The initial resource and CPU four-row sources are rejected experiments, not
production performance. Production retains the two-row CPU kernel and adopts
the narrower borrowed-span ownership consolidation and stable-norm/control
launch change. Launch observations describe actual frozen B source
``14c37277de16182fdc7f4cfcf9193e0b352791d1``. CUDA CSR is measured in
device/prepared scope with standard CG and Jacobi PCG. Dense explicit-residual
prepared execution has diagnostic captures only, not a framework timing
campaign. Unmeasured CSR API/Session scopes and current BLAS/OpenMP scope
characterizations remain historical; they do not inherit a win claim from a
different source or from the preceding baseline.

All framework medians belong to the recorded frozen Python harness hashes.
The later reference-boundary correction makes unusable recurrence squared
norms fail before refresh/iteration-limit recovery, matching native priority.
It has focused regression coverage but is absent from these timed harnesses.
That edge-case fix and later merges do not turn old bytecode into a new
measurement, even when normal fixtures retain their numerical work. The final
checkout is not silently substituted for an actual compiled source. Source
identity and same-spec I/M/R parity establish the boundaries of the evidence,
not universal performance equivalence for unmeasured cases.

Method, estimators and environment
----------------------------------

All fixed A/B runs use seed 20261017, three warmups, 36 complete Williams
rounds and one solve per timed round. Each configuration launches A/B/B/A
once, yielding two fresh-process artifact blocks per side. All declared
candidates execute every round. Strict estimates pool the 72 recorded round
averages per side; the hierarchical percentile bootstrap resamples artifact
blocks and then rounds, with 10,000 resamples, seed 20261017 and 95% intervals.
The budget is fixed at current/baseline ratio 1.05. Meeting this budget is
evidence against a material regression under this policy, not proof of a
speedup. Inconclusive is not a passing result. Framework controls receive the
same strict analysis and remain visible without drift subtraction.

The characterization baseline instead reports the median of two process
medians and their two-value range. It does not use the pooled-72 estimator or
bootstrap. Do not mix its point ratios with strict A/B ratios, reuse historical
processes as new A/B blocks, or attach a new interval to an old estimate.
Cross-framework and cross-scope point comparisons are descriptive; strict
comparison requires matching complete specs, input bytes and known I/M/R.

Restarts count actual successfully installed directions after true-residual
replacement. Initial setup and failed/converged/last-iteration exits before
replacement do not count. The explicit algorithm reports zero. Counts reset
on every solve. Official SciPy CG/PCG and SuperLU restart counts remain
unknown, never a fabricated zero. Unknown external work prevents a claim of
full native numerical-work parity.

The hardware context is Intel Core i7-13700K, WSL2/Linux 6.6.87.2 and RTX 2060
sm75. The measured driver is **610.88**, driver API **13030**, CUDA runtime
**12080**, and NVCC **12.8.93**. CPU compilation uses GNU 16.0.1; CUDA host
compilation uses GNU 13.3.0, with Release ``-O3 -DNDEBUG`` and no fast-math.
CPU OpenMP uses two threads and logical affinity [0,2]; other modes use one
thread and affinity [0]. These logical constraints do not isolate the physical
Windows host. GPU clocks, thermal state, Windows display activity and host
background load are uncontrolled. Historical driver-591.86 evidence is not
pooled with this driver-610.88 campaign.

Actual compiler flags, public runtime libraries, Python/packages, full
threadpool inventories and loaded module hashes are audited. Pools are exact
within the applicable preset/suite/configuration; CPU/GPU providers and
dense/CSR import inventories need not be identical. Native mode selection
does not change Torch's project reference engine. These frozen sources keep
the Python harness and comparator identical for each C++-only A/B study.

Lineage and diagnostics
-----------------------

The original characterization stopped after numerical worker 29 succeeded,
because its parent audit mishandled deliberately absent Torch metadata in the
official solver-strategy cohort. A separate repair reaudited those 29 without
rerunning them and executed only positions 30–108 in the original order.
The failed original manifest, claims, tools and all raw/import/log evidence
remain unchanged. The continuation has a wall-time gap; it is not an
uninterrupted campaign. No numerical worker was retried or discarded.

Every A/B campaign has its own frozen plan, claim, source stages, manifest,
worker/import/log records and complete saved comparator results. The first
ownership trial remains rejected; the post-selection follow-up is a separate
exploratory evaluation. CPU and both launch campaigns retain that same
lineage discipline. Raw JSON, binaries and Nsight traces remain ignored local
evidence; public commands regenerate new observations, not historical times.

All 57 audited baseline/initial/follow-up/launch profile captures are retained
below. Each retains its actual compiled revision/tree,
driver path/SHA and separate orchestration-runner path/SHA. Resource probes
time construction and destruction; solve probes retain their declared scope.
Before/after independent true-residual gates retain identical known I/M/R.

CUDA API counts cover the full ten-call capture, including pre-NVTX/inter-call
completion fences. They are not solely solver-internal counts. Outer NVTX
phase wall times have a separate boundary. API, GPU and NVTX durations can
overlap; do not add them into percentages or subtract independently sampled
scope medians. Preserve every driver call and distinguish launch API counts
from GPU kernel instances. A missing kernel CSV remains unreported; known
zero requires the retained no-event export/log evidence.

Validation
----------

Central validation of the merged implementation passes 92 CTest entries in
each CPU release/BLAS/OpenMP preset and 127 in the CUDA preset, covering C++
and Python with declared optional-backend skips. CUDA pytest reports
737 passed and nine skipped; those skips concern unavailable BLAS/OpenMP or
non-CUDA requirements, rather than skipping required GPU tests. Compute
Sanitizer memcheck covers all 35 GPU tests with zero errors and zero leaked
bytes; synccheck also passes. Relocated native CPU, BLAS and CUDA consumers
pass, with the CUDA consumer requiring actual device execution. These checks
establish tested correctness and ownership behavior, not a performance gate.
Python, typing, documentation and distribution validation use the workflows
described in :doc:`../build`.

Public reproduction
-------------------

Follow :doc:`../build`, :doc:`../install` and :doc:`../cuda` for CPython 3.12,
the local environments, pinned requirements and project-local CUDA 12.8.
Use the exact public measured sources in separate checkouts and build
directories. Do not silently substitute the final merge head. The
source identities above and complete tables name the actual measured versions.
Ignored orchestration helpers are not a required benchmark dependency.

The following is the public CUDA eight-configuration ownership matrix. It
keeps all three scopes and both native execution modes explicit; it does not
compare different scopes with a strict gate. Set an absolute PYTHON and local
CUDA_ROOT consistently when running from different source checkouts.

.. code-block:: bash

   PROJECT="${PROJECT:-$PWD}"
   for preconditioner in none jacobi; do
       if [ "$preconditioner" = none ]; then
           scopes=api
       else
           scopes='api session prepared'
       fi
       for scope in $scopes; do
           for mode in host device; do
               taskset -c 0 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
                   make pyprof PRESET=cuda-release \
                   PYTHON="$PROJECT/.venv-cuda/bin/python" \
                   CUDA_ROOT="$PROJECT/.deps/cuda-12.8" \
                   PYPROF_SUITE=cuda-dense-f64-v1 PYPROF_THREADS=1 \
                   PYPROF_SCOPE="$scope" PYPROF_ROUNDS=36 \
                   PYPROF_REPETITIONS=1 \
                   PYPROF_OUTPUT="build/reproduce/cuda-${preconditioner}-${scope}-${mode}.json" \
                   PYPROF_ARGS="--device cuda --cuda-device 0 --cuda-execution-mode $mode --implementations cuda-native,cuda-cublas,torch-cuda-eager --preconditioner $preconditioner --algorithm standard --fused --seed 20261017 --warmups 3 --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50"
           done
       done
   done

This loop makes one fresh observation per configuration. To reproduce ABBA,
run the same complete configuration from sources A/B/B/A with four distinct
output paths, retaining both A files and both B files. Never overwrite a
previous artifact. Compare corresponding configurations with the public
``scripts/compare_performance.py`` from the matching frozen current source,
``--max-regression 0.05 --bootstrap-samples 10000 --seed 20261017``. Do not pool
the rejected and selected ownership variants. Keep both selected intervals
and the original failure when reporting the exploration.

For the completed CPU matrix, run the following separately at the exact A and
rejected B sources. It creates one process observation per configuration;
use distinct paths in the same A/B/B/A order for a strict reproduction.

.. code-block:: bash

   PROJECT="${PROJECT:-$PWD}"
   for preset in cpu-release cpu-blas cpu-openmp; do
       case "$preset" in
           cpu-release)
               fusions='--fused --no-fused'
               candidates=native,numpy-out,torch-eager-out
               threads=1
               affinity=0
               ;;
           cpu-blas)
               fusions=--fused
               candidates=native,native-blas,numpy-out,torch-eager-out
               threads=1
               affinity=0
               ;;
           cpu-openmp)
               fusions=--fused
               candidates=native,numpy-out,torch-eager-out
               threads=2
               affinity=0,2
               ;;
       esac
       for preconditioner in none jacobi; do
           for scope in api prepared; do
               for fusion in $fusions; do
                   taskset -c "$affinity" make pyprof PRESET="$preset" \
                       PYTHON="$PROJECT/.venv/bin/python" \
                       PYPROF_SUITE=cpu-dense-f64-v1 \
                       PYPROF_THREADS="$threads" PYPROF_SCOPE="$scope" \
                       PYPROF_ROUNDS=36 PYPROF_REPETITIONS=1 \
                       PYPROF_OUTPUT="build/reproduce/${preset}-${preconditioner}-${scope}-${fusion#--}.json" \
                       PYPROF_ARGS="--implementations $candidates --preconditioner $preconditioner --algorithm standard $fusion --seed 20261017 --warmups 3 --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50"
               done
           done
       done
   done

Set ``PROJECT`` once to the absolute checkout holding the pinned environments
and toolkit. Keep it fixed when changing between measured source checkouts;
the fallback assignment above retains an existing value. Build each exact
source in its own directory, and inspect the recorded compiler/runtime and
loaded extension identity before comparing new artifacts.

The prepared/device launch matrix is dense/CSR times none/Jacobi:

.. code-block:: bash

   for suite in cuda-dense-f64-v1 cuda-csr-f64-v1; do
       if [ "$suite" = cuda-dense-f64-v1 ]; then
           candidates=cuda-native,cuda-cublas,torch-cuda-eager
       else
           candidates=cuda-native,torch-cuda-eager
       fi
       for preconditioner in none jacobi; do
           taskset -c 0 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
               make pyprof PRESET=cuda-release \
               PYTHON="$PROJECT/.venv-cuda/bin/python" \
               CUDA_ROOT="$PROJECT/.deps/cuda-12.8" \
               PYPROF_SUITE="$suite" PYPROF_THREADS=1 \
               PYPROF_SCOPE=prepared PYPROF_ROUNDS=36 \
               PYPROF_REPETITIONS=1 \
               PYPROF_OUTPUT="build/reproduce/launch-${suite}-${preconditioner}-prepared-device.json" \
               PYPROF_ARGS="--device cuda --cuda-device 0 --cuda-execution-mode device --implementations $candidates --preconditioner $preconditioner --algorithm standard --fused --seed 20261017 --warmups 3 --rtol 1e-8 --atol 1e-12 --maxiter 2000 --residual-refresh 50"
       done
   done

Run this matrix independently at the recorded adopted-span and launch sources
in A/B/B/A order, retaining four distinct files per configuration. The dense
eight-configuration loop above also describes the preregistered supplemental
study when run at these sources. It is a separate campaign; do not combine
its prepared observations with this four-configuration study. Keep independent
processes separate rather than treating 36 rounds as 36 independent runs.

CUDA API allocates fresh output/workspace and optional Jacobi per call.
Session reuses iterative execution resources but creates fresh output and
optional Jacobi per call. Prepared reuses output/workspace/operator/Jacobi;
every call resets its initial guess. Torch uses matching scope policies with
its documented caching allocator. All CUDA inputs are resident, and input
uploads and independent correctness downloads are outside solve timing.
Dispatch, required scalar/status synchronization and final completion stay
inside. None of these scopes is host-to-host end-to-end latency.

For the explicit dense prepared diagnostic, use an existing Nsight Systems
installation and the selected source's staged package:

.. code-block:: bash

   mkdir -p build/profiles
   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
       --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
       --output build/profiles/native-explicit-device-dense1024 \
       taskset -c 0 env PYTHONPATH="$PWD/build/cuda-release/python" \
       CUBLAS_WORKSPACE_CONFIG=:4096:8 \
       "$PROJECT/.venv-cuda/bin/python" scripts/profile_solve.py \
       --suite cuda-dense-f64-v1 --case dense-1024 \
       --implementation cuda-native --device cuda --cuda-device 0 \
       --scope prepared --probe solve --cuda-execution-mode device \
       --algorithm explicit --preconditioner none --fused --threads 1 \
       --seed 20261017 --rtol 1e-8 --atol 1e-12 --maxiter 2000 \
       --residual-refresh 50 --warmups 3 --repetitions 10 --cuda-profiler \
       --output build/profiles/native-explicit-device-dense1024.json

This capture is diagnostic only. Nsight availability and WSL permissions are
external prerequisites. It includes full-capture completion fences and cannot
be combined with unprofiled scores or used as an additive phase breakdown.

CPU component diagnostic appendix — rejected four-row variant
-------------------------------------------------------------

All **84 observations / 42 matched pairs** are retained, including
higher candidate medians and every long-double oracle gate. The
four-row candidate was rejected by the separate complete CG/PCG
A/B campaign; production restored the original two-row kernel. These
component observations do not replace that decision.

Each side has one fresh process per fixed operation, three warmups,
36 samples and one operation per sample, native CPU mode, one thread
and affinity [0]. The entire baseline pass preceded the candidate
pass. No ABBA interleaving or independent process repetitions were
collected for these components. Ranges below are sample minima/maxima,
not confidence intervals. No strict regression or winner is inferred.

Official sizes64/256/1024/2048 use the actual seed20261017 dense
fixtures, whose full A/b/x0/truth hashes match the saved baseline.
The fixed diagnostic vector is b; it does not represent every CG
direction. Integer sizes129/130/131 have padded row stride n+5:
A[i,j]=((i+3j)%7)-3, vector[i]=(i%5)-2, b[i]=(i%7)-3, with NaN
padding excluded from arithmetic. Both sides reuse identical bytes.

File reads, allocations, finite scans, independent reference arithmetic,
validation, warmups and output are outside operation timing. The
source-pinned C++ probe checks every vector entry and applicable scalar
against scalar long-double arithmetic and propagated FP64 bounds
before and after measurement. Successful exit proves both checks;
the stored error/bound fields below are the post-measurement checks.
No oracle is recomputed by this stdlib renderer.

These fixed-vector wall times are diagnostics, not framework scores,
hardware bandwidth measurements or additive whole-CG phase attribution.
Do not add operation medians or subtract separate scope timings.
WSL affinity does not isolate physical host load, clocks or thermals.

.. csv-table:: All matched diagnostic timings
   :class: benchmark-results
   :header-rows: 1

   Case,Stride,Fused,Operation,A median ms,A min–max ms,B median ms,B min–max ms,B/A
   dense-64,64,False,apply,0.000361000,0.000359000–0.000452000,0.000389000,0.000381000–0.048573000,1.077562
   dense-64,64,False,apply_and_dot,0.000402000,0.000401000–0.000542000,0.000379000,0.000375000–0.000547000,0.942786
   dense-64,64,False,residual_squared_norm,0.000425000,0.000422000–0.011278000,0.000400000,0.000394000–0.000524000,0.941176
   dense-64,64,True,apply,0.000360500,0.000358000–0.011217000,0.000339000,0.000333000–0.010987000,0.940361
   dense-64,64,True,apply_and_dot,0.000368000,0.000366000–0.000470000,0.000413000,0.000408000–0.000517000,1.122283
   dense-64,64,True,residual_squared_norm,0.000407000,0.000399000–0.001641000,0.000360000,0.000357000–0.000451000,0.884521
   dense-256,256,False,apply,0.004889500,0.004800000–0.022465000,0.004773000,0.004605000–0.029226000,0.976173
   dense-256,256,False,apply_and_dot,0.004941500,0.004891000–0.035683000,0.004737000,0.004700000–0.063968000,0.958616
   dense-256,256,False,residual_squared_norm,0.005623000,0.004953000–0.034259000,0.004957500,0.004745000–0.007224000,0.881647
   dense-256,256,True,apply,0.004820000,0.004792000–0.018282000,0.004624500,0.004602000–0.025039000,0.959440
   dense-256,256,True,apply_and_dot,0.004906000,0.004879000–0.027261000,0.004663500,0.004628000–0.026027000,0.950571
   dense-256,256,True,residual_squared_norm,0.004941000,0.004899000–0.015916000,0.004665000,0.004632000–0.004826000,0.944141
   dense-1024,1024,False,apply,0.161656500,0.137956000–0.182225000,0.152456000,0.129621000–0.196575000,0.943086
   dense-1024,1024,False,apply_and_dot,0.173882000,0.139432000–0.233091000,0.170992000,0.137561000–0.270517000,0.983380
   dense-1024,1024,False,residual_squared_norm,0.190878500,0.144089000–0.263680000,0.179298500,0.149665000–1.051264000,0.939333
   dense-1024,1024,True,apply,0.156799000,0.139120000–0.190669000,0.136624500,0.134449000–0.217335000,0.871335
   dense-1024,1024,True,apply_and_dot,0.181145500,0.140580000–0.230790000,0.199004000,0.140301000–0.449935000,1.098586
   dense-1024,1024,True,residual_squared_norm,0.183166500,0.160023000–0.225488000,0.152435000,0.135637000–0.212076000,0.832221
   dense-2048,2048,False,apply,1.916531000,0.955828000–4.313184000,1.671557500,0.896646000–3.751114000,0.872179
   dense-2048,2048,False,apply_and_dot,1.619994500,1.115293000–3.410844000,1.513830000,0.797859000–4.204348000,0.934466
   dense-2048,2048,False,residual_squared_norm,1.428512000,1.096583000–3.487011000,1.733059000,0.827660000–3.478580000,1.213192
   dense-2048,2048,True,apply,1.380356500,0.859954000–3.118661000,1.386002500,0.732642000–3.352031000,1.004090
   dense-2048,2048,True,apply_and_dot,1.826700500,0.915059000–4.993389000,1.220367000,0.861965000–2.941810000,0.668072
   dense-2048,2048,True,residual_squared_norm,1.506064500,1.098146000–3.521191000,1.667759000,1.157505000–3.443238000,1.107362
   integer-tail-129,134,False,apply,0.001307000,0.001297000–0.012591000,0.001496500,0.001295000–0.023596000,1.144989
   integer-tail-129,134,False,apply_and_dot,0.001370000,0.001360000–0.007264000,0.001558500,0.001551000–0.001696000,1.137591
   integer-tail-129,134,False,residual_squared_norm,0.001394000,0.001385000–0.001478000,0.001433000,0.001383000–0.012618000,1.027977
   integer-tail-129,134,True,apply,0.001514000,0.001498000–0.017745000,0.001265500,0.001252000–0.038629000,0.835865
   integer-tail-129,134,True,apply_and_dot,0.001586500,0.001569000–0.044206000,0.001363000,0.001280000–0.013667000,0.859124
   integer-tail-129,134,True,residual_squared_norm,0.001348000,0.001341000–0.011256000,0.001334000,0.001276000–0.012161000,0.989614
   integer-tail-130,135,False,apply,0.001418000,0.001406000–0.016137000,0.001356000,0.001346000–0.001452000,0.956276
   integer-tail-130,135,False,apply_and_dot,0.001738500,0.001729000–0.002060000,0.001431000,0.001413000–0.013826000,0.823123
   integer-tail-130,135,False,residual_squared_norm,0.001483000,0.001431000–0.012640000,0.001699000,0.001688000–0.001815000,1.145651
   integer-tail-130,135,True,apply,0.001636500,0.001383000–0.097503000,0.001356500,0.001340000–0.036614000,0.828903
   integer-tail-130,135,True,apply_and_dot,0.001420500,0.001407000–0.012487000,0.001373500,0.001358000–0.001466000,0.966913
   integer-tail-130,135,True,residual_squared_norm,0.001424500,0.001408000–0.040305000,0.001389000,0.001372000–0.014208000,0.975079
   integer-tail-131,136,False,apply,0.001580500,0.001563000–0.001677000,0.001545500,0.001524000–0.013927000,0.977855
   integer-tail-131,136,False,apply_and_dot,0.001413000,0.001388000–0.011180000,0.001394000,0.001366000–0.102640000,0.986553
   integer-tail-131,136,False,residual_squared_norm,0.001428500,0.001407000–0.011633000,0.001646500,0.001633000–0.013531000,1.152608
   integer-tail-131,136,True,apply,0.001333500,0.001306000–0.012329000,0.001312000,0.001301000–0.011341000,0.983877
   integer-tail-131,136,True,apply_and_dot,0.001420000,0.001410000–0.027486000,0.001565500,0.001555000–0.015704000,1.102465
   integer-tail-131,136,True,residual_squared_norm,0.001480500,0.001462000–0.011378000,0.001353000,0.001343000–0.001451000,0.913880

.. csv-table:: All 84 finite post-measurement oracle records
   :class: benchmark-results
   :header-rows: 1

   Side,Run,Long-double bits,Vector max error,Vector max bound,Scalar error,Scalar bound,Finite
   baseline,dense-64-fused0-apply,64,1.0196288258157438e-12,2.0618619623662891e-10,0,0,True
   baseline,dense-64-fused0-apply_and_dot,64,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,True
   baseline,dense-64-fused0-residual_squared_norm,64,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,True
   baseline,dense-64-fused1-apply,64,1.0196288258157438e-12,2.0618619623662891e-10,0,0,True
   baseline,dense-64-fused1-apply_and_dot,64,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,True
   baseline,dense-64-fused1-residual_squared_norm,64,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,True
   baseline,dense-256-fused0-apply,64,2.1898038937706588e-12,8.7769074042863557e-10,0,0,True
   baseline,dense-256-fused0-apply_and_dot,64,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,True
   baseline,dense-256-fused0-residual_squared_norm,64,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,True
   baseline,dense-256-fused1-apply,64,2.1898038937706588e-12,8.7769074042863557e-10,0,0,True
   baseline,dense-256-fused1-apply_and_dot,64,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,True
   baseline,dense-256-fused1-residual_squared_norm,64,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,True
   baseline,dense-1024-fused0-apply,64,4.0416558988454199e-12,4.1708565652771602e-09,0,0,True
   baseline,dense-1024-fused0-apply_and_dot,64,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,True
   baseline,dense-1024-fused0-residual_squared_norm,64,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,True
   baseline,dense-1024-fused1-apply,64,4.0416558988454199e-12,4.1708565652771602e-09,0,0,True
   baseline,dense-1024-fused1-apply_and_dot,64,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,True
   baseline,dense-1024-fused1-residual_squared_norm,64,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,True
   baseline,dense-2048-fused0-apply,64,6.0809135504769074e-12,1.1096243113407423e-08,0,0,True
   baseline,dense-2048-fused0-apply_and_dot,64,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,True
   baseline,dense-2048-fused0-residual_squared_norm,64,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,True
   baseline,dense-2048-fused1-apply,64,6.0809135504769074e-12,1.1096243113407423e-08,0,0,True
   baseline,dense-2048-fused1-apply_and_dot,64,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,True
   baseline,dense-2048-fused1-residual_squared_norm,64,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,True
   baseline,integer-tail-129-fused0-apply,64,0,8.0917494926782365e-12,0,0,True
   baseline,integer-tail-129-fused0-apply_and_dot,64,0,8.0917494926782365e-12,0,1.2798349047216872e-09,True
   baseline,integer-tail-129-fused0-residual_squared_norm,64,0,8.0950801617521136e-12,0,1.7782194827740646e-08,True
   baseline,integer-tail-129-fused1-apply,64,0,8.0917494926782365e-12,0,0,True
   baseline,integer-tail-129-fused1-apply_and_dot,64,0,8.0917494926782365e-12,0,1.2798349047216872e-09,True
   baseline,integer-tail-129-fused1-residual_squared_norm,64,0,8.0950801617521136e-12,0,1.7782194827740646e-08,True
   baseline,integer-tail-130-fused0-apply,64,0,8.2427398240272643e-12,0,0,True
   baseline,integer-tail-130-fused0-apply_and_dot,64,0,8.2427398240272643e-12,0,1.3138130583457381e-09,True
   baseline,integer-tail-130-fused0-residual_squared_norm,64,0,8.2440720916568161e-12,0,1.4466784215026626e-08,True
   baseline,integer-tail-130-fused1-apply,64,0,8.2427398240272643e-12,0,0,True
   baseline,integer-tail-130-fused1-apply_and_dot,64,0,8.2427398240272643e-12,0,1.3138130583457381e-09,True
   baseline,integer-tail-130-fused1-residual_squared_norm,64,0,8.2440720916568161e-12,0,1.4466784215026626e-08,True
   baseline,integer-tail-131-fused0-apply,64,0,8.3950624230058422e-12,0,0,True
   baseline,integer-tail-131-fused0-apply_and_dot,64,0,8.3950624230058422e-12,0,1.3516050501039815e-09,True
   baseline,integer-tail-131-fused0-residual_squared_norm,64,0,8.3979490028698708e-12,0,1.3550524036186618e-08,True
   baseline,integer-tail-131-fused1-apply,64,0,8.3950624230058422e-12,0,0,True
   baseline,integer-tail-131-fused1-apply_and_dot,64,0,8.3950624230058422e-12,0,1.3516050501039815e-09,True
   baseline,integer-tail-131-fused1-residual_squared_norm,64,0,8.3979490028698708e-12,0,1.3550524036186618e-08,True
   candidate,dense-64-fused0-apply,64,1.0196288258157438e-12,2.0618619623662891e-10,0,0,True
   candidate,dense-64-fused0-apply_and_dot,64,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,True
   candidate,dense-64-fused0-residual_squared_norm,64,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,True
   candidate,dense-64-fused1-apply,64,1.0196288258157438e-12,2.0618619623662891e-10,0,0,True
   candidate,dense-64-fused1-apply_and_dot,64,1.0196288258157438e-12,2.0618619623662891e-10,1.6279955161735415e-10,2.0302254959166928e-07,True
   candidate,dense-64-fused1-residual_squared_norm,64,7.5539574595495651e-13,2.0908316183909403e-10,8.7311491370201111e-08,2.2454420314339871e-05,True
   candidate,dense-256-fused0-apply,64,2.1898038937706588e-12,8.7769074042863557e-10,0,0,True
   candidate,dense-256-fused0-apply_and_dot,64,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,True
   candidate,dense-256-fused0-residual_squared_norm,64,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,True
   candidate,dense-256-fused1-apply,64,2.1898038937706588e-12,8.7769074042863557e-10,0,0,True
   candidate,dense-256-fused1-apply_and_dot,64,2.1898038937706588e-12,8.7769074042863557e-10,2.2537278709933162e-09,1.9603692385893142e-06,True
   candidate,dense-256-fused1-residual_squared_norm,64,2.2466473126314668e-12,8.8104926014635646e-10,2.046581357717514e-07,0.00021547992368137688,True
   candidate,dense-1024-fused0-apply,64,4.0416558988454199e-12,4.1708565652771602e-09,0,0,True
   candidate,dense-1024-fused0-apply_and_dot,64,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,True
   candidate,dense-1024-fused0-residual_squared_norm,64,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,True
   candidate,dense-1024-fused1-apply,64,4.0416558988454199e-12,4.1708565652771602e-09,0,0,True
   candidate,dense-1024-fused1-apply_and_dot,64,4.0416558988454199e-12,4.1708565652771602e-09,3.328023012727499e-08,3.2850246876764954e-05,True
   candidate,dense-1024-fused1-residual_squared_norm,64,4.1269210271366319e-12,4.1749363516710137e-09,1.2945383787155151e-07,0.0036328106130226736,True
   candidate,dense-2048-fused0-apply,64,6.0809135504769074e-12,1.1096243113407423e-08,0,0,True
   candidate,dense-2048-fused0-apply_and_dot,64,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,True
   candidate,dense-2048-fused0-residual_squared_norm,64,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,True
   candidate,dense-2048-fused1-apply,64,6.0809135504769074e-12,1.1096243113407423e-08,0,0,True
   candidate,dense-2048-fused1-apply_and_dot,64,6.0809135504769074e-12,1.1096243113407423e-08,2.0736479200422764e-08,0.00013626509015883027,True
   candidate,dense-2048-fused1-residual_squared_norm,64,5.9530158580400894e-12,1.1101691666334374e-08,5.0524249672889709e-06,0.015375788579093783,True
   candidate,integer-tail-129-fused0-apply,64,0,8.0917494926782365e-12,0,0,True
   candidate,integer-tail-129-fused0-apply_and_dot,64,0,8.0917494926782365e-12,0,1.2798349047216872e-09,True
   candidate,integer-tail-129-fused0-residual_squared_norm,64,0,8.0950801617521136e-12,0,1.7782194827740646e-08,True
   candidate,integer-tail-129-fused1-apply,64,0,8.0917494926782365e-12,0,0,True
   candidate,integer-tail-129-fused1-apply_and_dot,64,0,8.0917494926782365e-12,0,1.2798349047216872e-09,True
   candidate,integer-tail-129-fused1-residual_squared_norm,64,0,8.0950801617521136e-12,0,1.7782194827740646e-08,True
   candidate,integer-tail-130-fused0-apply,64,0,8.2427398240272643e-12,0,0,True
   candidate,integer-tail-130-fused0-apply_and_dot,64,0,8.2427398240272643e-12,0,1.3138130583457381e-09,True
   candidate,integer-tail-130-fused0-residual_squared_norm,64,0,8.2440720916568161e-12,0,1.4466784215026626e-08,True
   candidate,integer-tail-130-fused1-apply,64,0,8.2427398240272643e-12,0,0,True
   candidate,integer-tail-130-fused1-apply_and_dot,64,0,8.2427398240272643e-12,0,1.3138130583457381e-09,True
   candidate,integer-tail-130-fused1-residual_squared_norm,64,0,8.2440720916568161e-12,0,1.4466784215026626e-08,True
   candidate,integer-tail-131-fused0-apply,64,0,8.3950624230058422e-12,0,0,True
   candidate,integer-tail-131-fused0-apply_and_dot,64,0,8.3950624230058422e-12,0,1.3516050501039815e-09,True
   candidate,integer-tail-131-fused0-residual_squared_norm,64,0,8.3979490028698708e-12,0,1.3550524036186618e-08,True
   candidate,integer-tail-131-fused1-apply,64,0,8.3950624230058422e-12,0,0,True
   candidate,integer-tail-131-fused1-apply_and_dot,64,0,8.3950624230058422e-12,0,1.3516050501039815e-09,True
   candidate,integer-tail-131-fused1-residual_squared_norm,64,0,8.3979490028698708e-12,0,1.3550524036186618e-08,True

Baseline compiled revision: ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``.

Actual portable compiler argv: ``/usr/bin/c++ -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -I/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-four-row/headers-baseline /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-four-row/profile_gemv.cpp /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/baseline-stages/cpu-release/libcg_core.a -o /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-four-row/baseline-components/probe``.

Candidate compiled revision: ``2d81b3d1409b0b428572cbf26fc7418e03bd270c``.

Actual portable compiler argv: ``/usr/bin/c++ -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -I/home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-four-row/headers-baseline /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-four-row/profile_gemv.cpp /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-quad-stages/cpu-release/libcg_core.a -o /home/threemonth03/Downloads/Conjugate-Gradient-Method-library/build/restart-optimization-20261017/cpu-four-row/candidate-components/probe``.

.. csv-table:: Source, archive, executable and public-header identities
   :class: benchmark-results
   :header-rows: 1

   Role,Path,SHA256
   renderer,build/restart-optimization-20261017/render_components_report.py,31a10e28d5d7422142542a356c646863a0dc13934797375e7e8cf6d13d6e356c
   baseline plan,build/restart-optimization-20261017/cpu-four-row/baseline-components/plan.json,09f15a68b80705b942c57ed4633ebd44b2d8e28e4972c4df90ca09be7f11974b
   baseline manifest,build/restart-optimization-20261017/cpu-four-row/baseline-components/manifest.json,5b10527756a3cc7c6f204c5f787a1d325d29877b92835bbb0e196e972541f069
   baseline archive,build/restart-optimization-20261017/baseline-stages/cpu-release/libcg_core.a,e47471ca51084098a11eee8f0b45bb1b06b8a78b27f0b868408d8e709de6f3da
   baseline claim,build/restart-optimization-20261017/cpu-four-row/baseline-components/execution-claim.json,fdb8d82238597bffaef0539d280c0713cbe3c2e0bb2de4e82c1d2c0eab0cdde9
   baseline export,build/restart-optimization-20261017/cpu-four-row/baseline-components/exports.json,6552201a8072542783850a48af86aebe5f13a869c38fded7c0db299ec2f4aea2
   baseline binary,build/restart-optimization-20261017/cpu-four-row/baseline-components/probe,2553ed19dfe90515e2684d8fec9dedeffa34e45c307f23ca4d7db7eaf608ee3a
   baseline log,build/restart-optimization-20261017/cpu-four-row/baseline-components/compile.log,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline header 0,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/backend/cpu/backend.hpp,265479a84eb2313d23b821ecb04048a93a725370a6278b102a4caf0414a05409
   baseline header 1,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/backend/residual_sign.hpp,6d10cde48e611fb016f0a083e2ba6a59758a4b5ca1875e7948f2c72db4c2c3d0
   baseline header 2,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/matrix/dense_matrix_view.hpp,0869ab54e368c0dd4dcf91e2a877d48a0490147df5fe27af5052e425e29c282d
   baseline header 3,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/storage/vector_view.hpp,dc49a8ef869421d50c8184a4b82dd750fe4d954b67252900edceb0bdac93d889
   baseline compiler,/usr/bin/c++,546ddb2e5c2aadbdbcf2aaab30b8ac4f72abb8bb8f5fd51dc4bf6864a29491d7
   baseline interpreter,.venv/bin/python,e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f
   baseline stage identity,build/restart-optimization-20261017/baseline-stages/cpu-release/identity.json,82f66c8a54d2c5a029ab6b62941d37650c1d2f07441ad6dd5e852b052e567b38
   baseline source snapshot,build/restart-optimization-20261017/baseline-stages/cpu-release/source-files.json,ec6553bb49b270edfabdad0b64f6090b3bc614def58cdce5cb02836d837f48dd
   candidate plan,build/restart-optimization-20261017/cpu-four-row/candidate-components/plan.json,31ca10eb08575f365c46119c3a567edec1bf372dc3b05d7cc039147ef31777fb
   candidate manifest,build/restart-optimization-20261017/cpu-four-row/candidate-components/manifest.json,6aea12827f7e3be167b27fdc528ae14248a9b96593a188d0b792ba5e25083e2a
   candidate archive,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/libcg_core.a,3ec4a15d19cba01a53955be4eb16e4ba8b600346257b15bb7b207d31f6be35dd
   candidate claim,build/restart-optimization-20261017/cpu-four-row/candidate-components/execution-claim.json,c61d4f2f67db39e80448a2d3536862de96352101b6fcf988cbb8e880a132a0a7
   candidate export,build/restart-optimization-20261017/cpu-four-row/baseline-components/exports.json,6552201a8072542783850a48af86aebe5f13a869c38fded7c0db299ec2f4aea2
   candidate binary,build/restart-optimization-20261017/cpu-four-row/candidate-components/probe,bcf1910d426b94738f4ee566b40a423d1fc1fb049c535736dd64cfa1095697ff
   candidate log,build/restart-optimization-20261017/cpu-four-row/candidate-components/compile.log,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate header 0,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/backend/cpu/backend.hpp,265479a84eb2313d23b821ecb04048a93a725370a6278b102a4caf0414a05409
   candidate header 1,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/backend/residual_sign.hpp,6d10cde48e611fb016f0a083e2ba6a59758a4b5ca1875e7948f2c72db4c2c3d0
   candidate header 2,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/matrix/dense_matrix_view.hpp,0869ab54e368c0dd4dcf91e2a877d48a0490147df5fe27af5052e425e29c282d
   candidate header 3,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/storage/vector_view.hpp,dc49a8ef869421d50c8184a4b82dd750fe4d954b67252900edceb0bdac93d889
   candidate compiler,/usr/bin/c++,546ddb2e5c2aadbdbcf2aaab30b8ac4f72abb8bb8f5fd51dc4bf6864a29491d7
   candidate interpreter,.venv/bin/python,e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f
   candidate stage identity,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/identity.json,bd3a934ae3d0b8e43055634a6ba22fbc8870a405bce2a2f4082ef7aecdf2c5da
   candidate source snapshot,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/source-files.json,8b5b51b4ddebd5428c2d95d2516e0f6c94cb1c9dd42e61a31c281f650254517c
   frozen helper/oracle,build/restart-optimization-20261017/run_baseline.py,b368f781a3aa6516e366770b2d1f8c57df04da0f21f1c0274331c913ecc1ba51
   frozen helper/oracle,build/restart-optimization-20261017/protocol.py,8814adcd9e90fa800e3654c248b1554fbc19e74a4a773dfcbe61e60187afdc0c
   frozen helper/oracle,build/restart-optimization-20261017/worker.py,038fa8530e6ed876f52dcea09497e804840f74ebabb008d4adaa99db24f9456c
   frozen helper/oracle,build/restart-optimization-20261017/freeze_stage.py,ea90c787125e9bb2f7977f40a25e5d4fe4e94afbe2a237b2dc3893ca678b68ab
   frozen helper/oracle,build/restart-optimization-20261017/README.md,3f308246cf5540f79b24b43bd492cd57b35b6fe7e1f8c1e7a2977f6165cbea14
   frozen helper/oracle,build/performance-optimization/ab_runner.py,54668d5459fec4041f125d21d38a6fc73074c5b7dedc539a765a8bde254c7ace
   frozen helper/oracle,build/performance-optimization/run_baseline.py,213afaa21053d4f65dea506158dc65d60cc4587033bc388d0efccb362e76233f
   frozen helper/oracle,build/restart-optimization-20261017/baseline-audit-repair/run_baseline.py,7173f06e038d6de74596ec06c506136705d3be7a8a346f6d3cb9d235c77d7cff
   frozen helper/oracle,build/restart-optimization-20261017/baseline-audit-repair/protocol.py,5276f6ddface2b5011df30c4feea845e4ccd91e90b00d9d8c0bf092315f9dced
   frozen helper/oracle,build/restart-optimization-20261017/baseline-audit-repair/worker.py,038fa8530e6ed876f52dcea09497e804840f74ebabb008d4adaa99db24f9456c
   frozen helper/oracle,build/restart-optimization-20261017/baseline-audit-repair/freeze_stage.py,ea90c787125e9bb2f7977f40a25e5d4fe4e94afbe2a237b2dc3893ca678b68ab
   frozen helper/oracle,build/restart-optimization-20261017/baseline-audit-repair/README.md,7ba0ff2cc21ce90529945aa3c84b13b9a40a4a8d90ab00ad5de9a18e73e2cc55
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/run_components.py,d8f711b1910205a8da2ba6e170bd3996c797fa1b2e5d6d3648031e20ea6ca2bd
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/component_common.py,e4acc393194643a5f2538e8354b544dcdb7f6270442d381d34d1da939e4be9ee
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/component_worker.py,6a52b987fa8521c8ca9067703440a248ecc6390c2ad491a829c32db055e993e1
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/profile_gemv.cpp,670fa8067d49ad5eb325bcd550768afa8e733934b9292fc009ddd3510255dedc
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/headers-baseline.json,cd96fb1ff29d0dda3077904daf4c508656be3751abd2e9dfc4ab4c48c02a7e4b
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/driver.md,d0eb6675f50952902333f60b1788c1bac2a9108de06ca620ed1023d52e58fdcc
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/backend/cpu/backend.hpp,265479a84eb2313d23b821ecb04048a93a725370a6278b102a4caf0414a05409
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/backend/residual_sign.hpp,6d10cde48e611fb016f0a083e2ba6a59758a4b5ca1875e7948f2c72db4c2c3d0
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/matrix/dense_matrix_view.hpp,0869ab54e368c0dd4dcf91e2a877d48a0490147df5fe27af5052e425e29c282d
   frozen helper/oracle,build/restart-optimization-20261017/cpu-four-row/headers-baseline/cg/storage/vector_view.hpp,dc49a8ef869421d50c8184a4b82dd750fe4d954b67252900edceb0bdac93d889

.. csv-table:: Shared exact input identities
   :class: benchmark-results
   :header-rows: 1

   Case,Role,Path,SHA256
   dense-64,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-64.A.f64,2ae2227d37228e9b4f5c5d9060961ed9c7a27c53135f51bcbfcf0c621f04b9f2
   dense-64,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-64.b.f64,06946998cb419a52b0b367fc7b27c4941d1552abaa46d25552b8693f34d81fe4
   dense-64,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-64.vector.f64,06946998cb419a52b0b367fc7b27c4941d1552abaa46d25552b8693f34d81fe4
   dense-256,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-256.A.f64,4453bfe9013b7aa2538e1d4e5f1aca18336ec1ce54ddf3d15c6cfa4643abac14
   dense-256,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-256.b.f64,73d3bf719a3bc3266180bfee2a4de8840e9eb2ef8e425277453464134e76fc53
   dense-256,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-256.vector.f64,73d3bf719a3bc3266180bfee2a4de8840e9eb2ef8e425277453464134e76fc53
   dense-1024,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-1024.A.f64,ce8175bcf567ab4f16594849c5c48510b7947f249be6c4216f4a5c68998d5bf5
   dense-1024,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-1024.b.f64,1891b597da63bf9d4823dac02fa7633c06cedba3e4468e18fef568e56574a9f4
   dense-1024,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-1024.vector.f64,1891b597da63bf9d4823dac02fa7633c06cedba3e4468e18fef568e56574a9f4
   dense-2048,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-2048.A.f64,d97407fb8fab562a10fb8cc40d8dca819aefca248681c12e2f10f5cd304db56d
   dense-2048,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-2048.b.f64,1ac142af51538f14b47212b9ef4235467330dab06c45f80592fbdb7c09018371
   dense-2048,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/dense-2048.vector.f64,1ac142af51538f14b47212b9ef4235467330dab06c45f80592fbdb7c09018371
   integer-tail-129,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-129.A.f64,538a2b692d7136266f3d8d289723076a175cf4355995ff85942aa76065dbd39d
   integer-tail-129,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-129.b.f64,db811c05c5cc79a296ec9f7210975e9bb6891ec95836a5bc0cd09675ea68ee84
   integer-tail-129,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-129.vector.f64,32b1129fca2a5e1db81e2029494845bf1192c51adce81b47d8b5512f7c82bf39
   integer-tail-130,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-130.A.f64,9766b0037e754753aee00e0437bd4197a7455b2cfad6f52ea011e409a6a3caac
   integer-tail-130,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-130.b.f64,d6f72dd9601107715902f4aa1aa28956d148e0b5bb37e79d49874095cfc50223
   integer-tail-130,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-130.vector.f64,759dcf000800c7efc749756c153884b62f5ccdbb8cc2eabbc4bde05b8693ee61
   integer-tail-131,matrix,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-131.A.f64,205582384beb5635dffd8e0310ff0087f899466f2c7c6659edcaccbb30a86ee4
   integer-tail-131,rhs,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-131.b.f64,2a3e8c826093ad0769ebd0783ba20bc069803b23b2eea206071749cb4c36fca1
   integer-tail-131,vector,build/restart-optimization-20261017/cpu-four-row/baseline-components/inputs/integer-tail-131.vector.f64,c8e913873a76341110dc19bf10e00fb3d931ed7961cb51a54a5c064bb2e3c272

.. csv-table:: All original raw, execution-sidecar and log identities
   :class: benchmark-results
   :header-rows: 1

   Side,Run,Raw path,Raw SHA256,Sidecar SHA256,Log SHA256
   baseline,dense-64-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-64-fused0-apply.json,04b3164197bb2f8863117de1e02e32bfcd2c21b25f782b20ce8aa3025a4901c5,20ada4fc625491f6f97e248c8eab4fc7ec7761c48239cfa6ff1d55df3b55f75a,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-64-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-64-fused0-apply_and_dot.json,11098867feb66d67859a5777a7d1a9219e9a6d2e29177f8cbad65fc22306f513,0fde73fbee19f1f83f9512dc019f7026cd1b4ffe5776b1936e9ce2845bc472b0,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-64-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-64-fused0-residual_squared_norm.json,d1e6ca038c074de23302ab08cc3aa7c77fdd8ae2b13632835276141ad5021e06,af7a650bcb44d5809ce36eb1db59cc37bdd13b7b5dacd243804dc4268cd2f804,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-64-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-64-fused1-apply.json,98f5e55c639c86287d6ca92dbf345818a12bd597a8fd2f7ddd609ef839b2c450,65003f8bef8d613a9fbc0093e1231d7402a08fc92f2092342e31c6766d06e666,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-64-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-64-fused1-apply_and_dot.json,1f9767c305b0b8dc8b99672b2ba02f0f25ac20901e7c47271172952fe50b02c6,6cd0bb0bed0967f7500ddbdd4dcf90399d7dee4da4aea5b8d9ed6c6e18715abc,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-64-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-64-fused1-residual_squared_norm.json,b98f628a1da3878b7ab8215fb52a419db0348e9a18a9deaa74a5e7cbc4406178,91195d400c6bfa9b69d586299a22f29a0ec111b6248a3aa88456f8c4abedb7ac,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-256-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-256-fused0-apply.json,2585e321daa89b5613e9e1ca5f2e3af5cbe9bedacece24c2aa0caa2a0ae7d814,a14c7f3c418b56860bbdf7f35c745c3c627d4e58e9f2f4977025922b68ec823c,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-256-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-256-fused0-apply_and_dot.json,2462f55b47a371f7986b4c6c35d9388614bb9ee9d68f3ad3f8b0af5773bc54ca,a257b4ddca678cc1e4eec804861fa045cfa0871252b2b06dc4e91f7da2188ed7,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-256-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-256-fused0-residual_squared_norm.json,05190c313db8d501c9452532a39e61ff1d0677d77a1a0591fe5ce2d69b34f554,207663390de87826dd8b0af5514974c837d3383a9f179897d0c52f3b8f26fa42,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-256-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-256-fused1-apply.json,860b0fbac5541378e74d8706ce47bf4d672d4603b4b7834954f373bb638918e5,8698fb1ce933e4f3014b001854cf16d027c7956096d8e56cae08ec5a3c6b78ec,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-256-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-256-fused1-apply_and_dot.json,6cc7e78b9e9881e6d8c4142682933fb016247e1a83a3152466b27a4feb95fcc0,574cea21d7e13854cfed5405245c161a2174530fcd70446411ace33bdfab4c21,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-256-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-256-fused1-residual_squared_norm.json,9466d336e966da0233f450c1c52d862e3bb1c829ac2564b9fc8b3b1e89ae1dab,3e0ddeb22642377656a29ed69ba105e7dd1fa1b3d33c2760f1f6e28c6f398bb9,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-1024-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-1024-fused0-apply.json,ee5947c0a101a90e5a1b5d3cce7da136841774c3953f27230c9ad08aaa759618,2c56f16e6303ccb5a82c0e3ec3f125c5109d4334fec86485ab63cbd6c8462e67,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-1024-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-1024-fused0-apply_and_dot.json,b0fa8f5e261df4bd08fa5a7823c16c2c3cf6981cdd3f8c06cd205b5744d294fa,74004498a9b27a9936c17174add4fb008045ecc00596bfb879ea720d8e049971,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-1024-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-1024-fused0-residual_squared_norm.json,5737d4501cefbcabef919f45f03516fedd706934650a02bae89d908a279d09b2,494ae78be7924eb660c074c4949fd7fadd42b4c241db289c34ff707a2f554dd8,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-1024-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-1024-fused1-apply.json,84a41ef638f192df389d8c545307df7de8d36a92c2c7b24d2723ea65d3891531,6e4042cdf6fb56e39e0f7941f95d3a19fdf365d81ca511551d117223f17f1c06,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-1024-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-1024-fused1-apply_and_dot.json,a82639605949938214557fd3bccd92bca295a75fbe65c7f17040510610d07def,27f4e044118837347e5825bc4c1f77db58ffb1c73b229de11237829dcdf50e31,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-1024-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-1024-fused1-residual_squared_norm.json,cff5a2bb1c5969fae7e845beecc3c67c5326e72dd7339e25449e8543aae21012,eff2c432cfd18af11b20c2f82f93faf62dc67354ce2f2097f9846b671e17950a,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-2048-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-2048-fused0-apply.json,47b6d40c3ae8f2cc30223f6c8fe643e860dc1826b8ef45e32c5700a8959dae16,32eb27cb86966f691e8d76362077b38f9fe34adef4595952bd827bd65178581d,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-2048-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-2048-fused0-apply_and_dot.json,26e7a080508a20c4dfbfb4d88c9c6622685cb8a604964731b1b0d81e6b05aa78,9d39aa9a84454ec956f10cf7ea9854bc7083c4a16bf98213d1ed5bfd58f60e91,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-2048-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-2048-fused0-residual_squared_norm.json,770b8312049cf78e394795b827098a9e7c4b295998241f701d811f508670e52f,7c2aeb74e548d70b06b01c27d0e9689b47570392306ad53cfab77fa277068c1f,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-2048-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-2048-fused1-apply.json,7f9d40e7939f94a5807a5a859b45474e099c1e0f6b89a6c2973e373d5b8f4b91,4d2a2a4c2f6c3a285879cd123944b9e67c29a6e1e0054a5267a4a69f92d6190a,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-2048-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-2048-fused1-apply_and_dot.json,726cbaf5dd9186fff23b4077392853011943cf9ad79ab7beb5686423a2c03c10,93e700ffbb32e6b69257f3788ec1b3f0a9d5df1bd60224698c5a607899172d94,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,dense-2048-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/dense-2048-fused1-residual_squared_norm.json,81a2ca14b92533b714e24f8871573425be195ae0c1d9157d425aa8f44ea435a8,1e4345f077fcdbc3d3b70010395e84e4387350a4c926cc63cb93e500632e93ae,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-129-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-129-fused0-apply.json,9ad86c354af6fb3efc647707c641505617cfaa2ae0b03d570a3af246b87f4ef8,2136ee3ba51e4515bd039a16a08d8f5a2602c1c549f1f48fd722276909f5c549,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-129-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-129-fused0-apply_and_dot.json,6379164fab7f7643cbc3ba60138e15adcde47438cb8c68a661d2078df1698d4b,cf9f36dbf6c6c3312039b303e77c79b29807e98633dd7b58fc99e641b6d572cc,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-129-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-129-fused0-residual_squared_norm.json,8be92776da8dad11bcc012b2d41c44785c763b64d79302a1d2058257d76a1ff4,d5525b28a6db52e2114c014538576ee480dc8f68326221cb18468b2c5924829b,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-129-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-129-fused1-apply.json,b762508cda20f73786cbf846696d9e9a451ccb1211c2be51bc9aa557996667f4,dbeb18c4f2a54c244bce59ed3eceb65be84a4c69d1521b6d8494b6516c772c0b,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-129-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-129-fused1-apply_and_dot.json,e05cc07afa20c8870eaec5ea05cfb0f23a996583c12c71467f298f7b92e61362,646fd0df4e8c632fe2370b75882e14a4bef4a63aa64db9e762cdc1521b1db4ea,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-129-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-129-fused1-residual_squared_norm.json,d427e38bd73c8d5c6f9922494dfb3254bc11406f354765de041d99be36693763,64b0c610540e83a9ecb1eccfd47d69bf60e0427b25dff2d068490a0779571a66,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-130-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-130-fused0-apply.json,f103f599d40950ce040bc5780c12c26ca9174e881405e3c0485ee7d65c7ba331,93ec28b46e4dd482aefb4af06a286dc3b4cae6cba4a009cf91c8731724556b48,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-130-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-130-fused0-apply_and_dot.json,35c5886d0fb453bf903858d63dbb2b95ca34abb0fa931b84be462298f808890c,785de9d4e6b00011abcf3766897dc2a4b28f6e7193c76ef3627628b5286a3f7c,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-130-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-130-fused0-residual_squared_norm.json,e59afe7a04beb4c59ed3947fa3db477f61c50d48d77f99a59f1b1f4abe70cd42,98f5ecd447fc98663bcb8631be5a8b49f39f7ef7535b62e9952554f8d7db4bb9,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-130-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-130-fused1-apply.json,3c29e45e3c8715f84b6ce7cc2b3c4003c5f883afa1f47c24d61d858d0fec4d80,b043a00dd3e75023073ac1ee7f178d4eab0837353ab8d4dd2f82a95db1e932dd,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-130-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-130-fused1-apply_and_dot.json,d9818516313a1d4e101ffce2b73e5b5ff2853d1669b597a5df0f7b939ae538d5,d7a6245dff80e31ec2f87ddfde981b975a2fe6e5bcfa7e5b9a1e8b3b1c7ae104,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-130-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-130-fused1-residual_squared_norm.json,7b973d7aab4bc577563b5fb56fbaf21c879045c8887cce5beb4c083a3657125a,7588918cd02600f180dc130977ac4899d118f74318e335b60d4f927dcff85d24,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-131-fused0-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-131-fused0-apply.json,de8ad637d93ada010cb0a4df737621348070066429bd2b4bb770430ea6b4ec28,0665d3e555b1faf0e81e36ed4f9d76a97101f1a87b51357321ef7ea58f3c7d93,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-131-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-131-fused0-apply_and_dot.json,98d7abc92bd21125ce52c37ef20f763d9de569a0b0dda7dc42574e99983751bd,2fd40dc5880e2b96939d26f46b8b500d1116a8cd38cb419d5fdfe5e96bb7daf9,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-131-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-131-fused0-residual_squared_norm.json,ed5724dc3d412d44e1861e92b18b842f60b1a158865af337d37987e3939689c5,2484c7ed00b0950f350f3173d961867e6422e74a42d1ce75ac9615ab217aa55e,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-131-fused1-apply,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-131-fused1-apply.json,cfa12d73421d3340413cfff52c41c98890e060a93d4110170f61d6127674434f,345d9d6646c3c93dd5004769164d407fb2c08aa4cbf88259c79ecc1627576170,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-131-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-131-fused1-apply_and_dot.json,d4c83a42da53c34c186157aa180826af58acc3566bcd5db7c6f61632a5f76ac3,ff7b70792c0ab071b346bca3247776eeace330b6e50893ebc8f50aedc56f5794,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   baseline,integer-tail-131-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/baseline-components/raw/integer-tail-131-fused1-residual_squared_norm.json,3c489126c52afc100c454b2ecc4f39f86901a656412f5a1ec4d875c5ef2bcf30,8edeea1cee2cd93b95b03312c1d46a381689e90c3d33a08d85bc54f68dd51607,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-64-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-64-fused0-apply.json,db831ce2e53d13a1cbf8e4ebf78b5dea58dc45c9c004b4f9ccb7686331093f4a,025ccd7d9fbe536f8666b2fd093bcd9928c303ea5301ddc292257c78a9868c79,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-64-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-64-fused0-apply_and_dot.json,53792a37d04beebd3ddbecbec1bc511d2a7d67e1bd364554332440b742abc915,cc78103362a5e1af8b56643b246ebbf881af7f796c0e0700006149d534554200,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-64-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-64-fused0-residual_squared_norm.json,339417b9538e004e34f4db25a6604fb78126c14a384e6dad9eef2a67a98ad0ad,848400e51815ffa4a753ac7ee3b0c6b5400bef5f686b6464e094719bd9badf5e,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-64-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-64-fused1-apply.json,1fd3d5526d0ebcf9c2d0972f63895bb406a154dd498559e7c1f7e6b1113215d0,363604d5ed10b6c01ace8c6c0444f0474d9057a8333b7b0836f6238e7ecca31e,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-64-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-64-fused1-apply_and_dot.json,2dc2dd559e90a290586af8419695a26f8228c6274ff7a88c4383f5260303ed36,cea74713f7ba0822cd02b7f904934d7f09e773bd4058d2e2776f53b6d9e7d9f4,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-64-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-64-fused1-residual_squared_norm.json,7c775b8c4310c810b398a6270f9220a9f311933465406b6961cf3f30f73e4d8e,e7c3e18c3b14dc21e96077158918b7ea2d9a992393d94363f7b52931ce36f798,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-256-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-256-fused0-apply.json,664945fda5670ecdd58670ffd4846431a37059dc1aba7d186b5be7f76e6ea115,e6056efd5aa9af81cd28050731ab2d5d5519ad03cbc5b33f9220cda7e441ee10,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-256-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-256-fused0-apply_and_dot.json,6ec3879c7227fadc122650ce7c9acbfff9467c966757300c3478cbdc2eb31d3e,58ed21d043e4580f7e27144f0ae580cdf63168057a7f2ef99fa7a10dc7b3234f,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-256-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-256-fused0-residual_squared_norm.json,a732b5ef44f4a2ac5c0c42867decc1413b9ef5d108862bff6e4c718ff52dec2c,3fcb43eb09edbfa3061b6a08dd391e47bd5b31aa9c29bf564a4cef77fb727dc1,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-256-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-256-fused1-apply.json,e2498b7b494c01db34a8b5fb19f2d06f404c9e0b97c93e58e8bc2c1d75eda0da,bffa9c45602d2502963ad4b17b98f8ddeb5626b970a9e36de3c762c73954eb91,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-256-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-256-fused1-apply_and_dot.json,bd0d9bbd043e2923625a7781dd87cd4732dae8c43aa1b23b48761ab09b06cf29,eb621ae733f2e3c5819b1a7bc9d78a4f065d74eb1f21b50981b0f25bddc40fe6,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-256-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-256-fused1-residual_squared_norm.json,2ae5a24d9a6720f12d89510133a838800a1be22534a63f87dd68ad6667061fa8,d204c88828028677cc78a0a064a8b2675793f1f34b0492940f9f5d7a38050c1d,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-1024-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-1024-fused0-apply.json,c9641aa172af7aafccf171f3da4fdb8d94b22bc5687c1779da80203070d62d89,b1c07b700723a5779fbe2892068df17467e2d68702eda50f4e3fbca61628d405,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-1024-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-1024-fused0-apply_and_dot.json,5fd380b8ce90eb2f7028cfa949aebc2a55e1248966df908a1a211c8a0c4c4ae5,519e7b6ad9c7d89b2c877e9ed0588ef36456d04ec36bbbb0ae5d379908a4bf2d,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-1024-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-1024-fused0-residual_squared_norm.json,fe5d1d80828b4b9bcb3027307a7031539af183c8c677d2e02e03d550c1a53fed,7c3824027fb56080a21a33bce01aa586054c818f208de86c1dbddfbde4c10a5c,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-1024-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-1024-fused1-apply.json,68241de7f2d6897989abdf3186dda11904fecca3e5ea2137aa40760cc3dd47c6,fc9eccb77e6360bdd6a6d1b8d0f088e1fb5efddedd96c00d5abce49f816865ed,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-1024-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-1024-fused1-apply_and_dot.json,6ed07e1e70094b2d7edba8d8086f7c9027bc223e947f2728845134f1bd08c5bd,8da1152e5e3b189203c3ab56ca00bc69c9d24fda2cf8da8900fcfefbb1bd67e2,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-1024-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-1024-fused1-residual_squared_norm.json,b29349474f679128d7f955e5c2804fa0b93b22adef212729e81b25a0c71e4318,aab8a9fdabd07dbd13c491d9751b187f86f070205287f9f553e8561368acb763,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-2048-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-2048-fused0-apply.json,495988025cde6b146c5a3c22e32eba70d9159dae7e64346431e8fa0113f30766,0a839dd47862f5e8cc7cdcdf422ebcbb912c8a6f89fa047b6df04cbce37e6084,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-2048-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-2048-fused0-apply_and_dot.json,7bb373bba5b0e3036a6cbde4b2b8e981b72eac304fe9e6caa659caa35d273038,90d83455c1437ef7362a6f9a4838bd3169584f46c2d233c6cfd5f4fa44e7313f,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-2048-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-2048-fused0-residual_squared_norm.json,470e7ce6bafdf459b11a9b36014f5827ee9dd5f70bf9495d96639b8d215734d6,47d1ca3138a0ed864e82ba43ee9ace0ad4f660ac39171fd375df7bed6fc8dec9,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-2048-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-2048-fused1-apply.json,3f837ffa397269e90e4f3572080123c19af3cce5c9a7d050fbaa3119eca536d0,faed75e3705144ace31b5d785170fb0e6638cb290f95ace2f8d636c8f5096df0,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-2048-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-2048-fused1-apply_and_dot.json,e10544a50c352c411cb48b69f2dc3c1e8cfed0f8ede1963b79ca0a6e9bb01a55,47a7ddaf0644f6b7b7ddc729d8fed064bb3b63054ee9ee5735a8399ab23b2ce0,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,dense-2048-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/dense-2048-fused1-residual_squared_norm.json,5eae0121bdd51c61b118ab3712b1addc54ec979f72c104dacde2190fd9872253,e4f4bf6e712c357ecbc4771de16f83c116b2bf1a15dca11f172fa9e3e1195165,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-129-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-129-fused0-apply.json,931f4fe666b486a9934593db20246455ee1ecd8a7e2c2e1c2068d5aa824230fc,77c34d20754290ddd7fb7aa651f796688bb12ac1223b195fd15ea044cf5e6447,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-129-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-129-fused0-apply_and_dot.json,7d7542450496afe89a2154a16713bcabe0599d9879e037692380a462f048a59c,444eaf7e6bb2b91af7ec5ca48e05d2c82b21791afb4fa547873d7582e5f614a7,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-129-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-129-fused0-residual_squared_norm.json,01175084cf836cca1730ef322d2c28b99aca48dc4db92aed6f0f545c21cdc608,d0401c63d4f6cc8abc7c1a41b251d5377c0622fca0e869d97cbd09300a7bef45,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-129-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-129-fused1-apply.json,6eed34208e857485da200f08a44769f65039bc462cf6c42ec4a76b4d1b8eff70,d5468461bf2cd20be4487b324b0aef897cd7df0b0030018d8f3ecd7021da623c,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-129-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-129-fused1-apply_and_dot.json,aa443d02e206838bfc9856999bd3a9b377969b9990e073c2e49d823cf3c05ef7,03de9c24c801e33b30918a9dad73ec35bf7d584697219958d5baccde0f61bc7c,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-129-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-129-fused1-residual_squared_norm.json,627dbe9608750a7bd14f9b0944d19cfb3838c6c156a4456ca7ec08613e7dfec4,04e6571098e88a5166f206f8d410bb6ad5b7975d157dd3ca252efd01fa7650c9,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-130-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-130-fused0-apply.json,19ce044e8b8fdb780d844a396b2f6baf778a9fd7e09cce0c17f69dee483bed32,160c69499e8185544084edc25cbd77dd0bd013efe904facd6c19e182a2a492c8,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-130-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-130-fused0-apply_and_dot.json,6707db2e279124565a7ba498aee2e2413c4ec219030c32a581f742d8d2859658,6bcd05f637ec78001e07ea0688b21881c15c7f05ac39eab57ac287ea36d84e6b,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-130-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-130-fused0-residual_squared_norm.json,d065a75ad75defa014e4994d4c7bdc0075c215e26877b01339d2f7dde19d4b94,470d8c7f57b4cc5cc9cbe7e940c244212abbbce60a87c471cf8775b7b030f101,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-130-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-130-fused1-apply.json,652961529d96df843e573bec8678723e588463886a986ff5be577c3498c9d333,68c007f67f3d6c2359b8db803b2034d2640f3d674873b8527e68090e79734398,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-130-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-130-fused1-apply_and_dot.json,cdd1befc51ce2ef91f91f50e710a439f3b25b936f40790044e0229bd74f8fcff,6ae25b38c12ffedaaf2eb30c3173d7d943d8a97c2e57095b5296da0856a1c18d,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-130-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-130-fused1-residual_squared_norm.json,3529227a9e2f601b75df75f7774b66af3c1f1f33a9d5e00ceaa93d4efb7bfd7b,575d874178869972374beb41a2e34c0104ce54934b5098832c13545f851f4db7,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-131-fused0-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-131-fused0-apply.json,815c5c596ec9064502df86cbee0bb421bda3ec3b20dac7e98bad5805f5ae6129,ea13c76ff548cf8f8e7c85b489040ebd0870e40a96ba8c3e07207ce7bf85eddb,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-131-fused0-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-131-fused0-apply_and_dot.json,35908abe5c7efd11b71b881b39e0d2bf73df7c192ca2edb5f09a59f85b3e9e71,26d194d2038f1a2eeaa7d4e7e3fa93c59f305e2d5d5f84fbe730c4edbffae6fd,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-131-fused0-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-131-fused0-residual_squared_norm.json,f6daf9850d88f45c5521dd4acb7f26c78c3e917214161b98fc500cb60f1fcef3,0972c197c55c9ab541e9a30429d035d4c18ba04708fb526849ebad522fe12e89,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-131-fused1-apply,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-131-fused1-apply.json,cd47797b1d9672dd5c37c82786be4f0b693af3003de60a7b9eb57564f2437dc9,7231db445af80919a1f4e219d62d554f5bfe44ffa22e4dd20847a6057d15d382,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-131-fused1-apply_and_dot,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-131-fused1-apply_and_dot.json,2a068b776916b95fa9f55434f76c77d7b19ca978bfcd55d66962bd856021b005,08853273d04ed91a461e4e36539266b1c18a746bad9239a2f6b93467ff314b85,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   candidate,integer-tail-131-fused1-residual_squared_norm,build/restart-optimization-20261017/cpu-four-row/candidate-components/raw/integer-tail-131-fused1-residual_squared_norm.json,e161373d46650ae5a23e2451ef3ec0b55363c7c9077d770bdf409cc10e1f7047,17f09547dc0c2b0febb939f7212ff35efd3ff68375840d1f516159c8e9655a67,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

All36 raw samples, exact input/compiler/worker argv, public-header
closure and stage/archive identities are retained in
``component-analysis.json`` and the original immutable manifests.
Paths under build/ are local ignored evidence, not public downloads.
To reproduce, build the two recorded source revisions with the exact
portable flags and public-header closure, export the named fixtures
with seed20261017, and use the probe/worker argv recorded per row.
Run all42 operations per side, preserving padding/vector/control
policies and every result; new execution creates new timings.

Complete retained A/B tables and provenance
-------------------------------------------

All536 strict rows below retain the five complete campaigns. B/current
means the specific measured candidate, not a later production checkout.
Rejected experiments remain visible alongside adopted changes. Cross-scope
framework ratios are descriptive and cannot override any strict verdict.

Every campaign retains its complete preplanned A,B,B,A order, all native
candidates and framework controls, two fresh-process blocks per side and
every declared fixture. No failed comparison or regression is discarded.

Each block contains 36 complete Williams rounds, one solve per round, three
warmups and seed 20261017. Strict A and B summaries pool their 72 normalized
round samples. This estimator differs from the median of two process medians
used in the earlier descriptive baseline. The A samples here are new runs in
each ABBA campaign; historical baseline samples are not pooled into them.

The frozen production comparator uses the original 5% budget, 95% hierarchical
percentile intervals, 10,000 resamples and seed 20261017. These saved intervals
are audited without recalculation. Per-case intervals are not simultaneous
coverage, serial round dependence can narrow them, and two artifact blocks
give limited evidence of process variance. Control drift is not subtracted.

Known iterations/matvecs/restarts must match all four observations for each
same candidate/configuration. Different frameworks/scopes can perform different
accepted-tolerance work. No pure-kernel normalization or cross-scope strict
comparison bypasses a failed gate. Resident CUDA scopes keep transfers outside
timing; Session still creates fresh output and optional Jacobi while reusing
only iterative workspace. Device execution remains explicit and does not
change the Torch reference implementation.

CPU OpenMP uses two threads and logical affinity [0,2]; other presets use one
thread/[0]. Compiler, runtime, package and full threadpool controls are audited
per preset/suite/configuration. WSL affinity is not physical-host isolation.
Thermals, clocks, Windows display and host load remain uncontrolled. Campaigns
with different actual sources stay separate; no universal speedup is claimed.

fresh-abba (fresh) fixed A/B campaign
-------------------------------------

Whole-campaign verdict: **regression**.
All 32 artifacts/384 candidate/case outcomes are retained.

.. csv-table:: Every native and framework strict result
   :class: benchmark-results
   :header-rows: 1

   Candidate,Role,Rows,Within 5%,Inconclusive,Regression,Comparison failed
   cuda-native,native,32,6,25,1,0
   cuda-cublas,native,32,2,30,0,0
   torch-cuda-eager,framework control,32,4,28,0,0

All candidate rows
~~~~~~~~~~~~~~~~~~

A/B and current/reference ratios below one indicate a lower measured numerator.
Ranges bracket the two process medians and are not confidence intervals.
Saved strict intervals compare current against A/B baseline; they do not give
confidence for the separate current/native-to-framework comparison.

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,5.990799,5.566319–6.075101,6.878946,6.361309–8.367543,1.14825,1.0158–1.5252,inconclusive,49/51/0,0.648887
   dense-64,cuda-cublas,8.027313,7.900297–8.490600,8.586097,8.076489–10.819370,1.06961,0.9172–1.4040,inconclusive,49/51/0,0.611009
   dense-64,torch-cuda-eager,15.572653,15.572653–15.582584,16.431311,15.620182–17.442043,1.05514,0.9666–1.2370,inconclusive,49/51/0,0.611009
   dense-256,cuda-native,8.727754,8.082854–9.732482,8.868001,8.330844–9.609753,1.01607,0.8340–1.2008,inconclusive,67/70/1,0.979365
   dense-256,cuda-cublas,11.162451,9.925811–12.249381,12.028764,10.432799–13.320502,1.07761,0.8459–1.3350,inconclusive,67/70/1,0.994421
   dense-256,torch-cuda-eager,19.945405,18.923392–20.439822,21.938080,20.491255–23.595915,1.09991,0.9901–1.2841,inconclusive,67/70/1,0.977359
   dense-1024,cuda-native,12.565022,12.020011–13.108141,13.930637,12.606834–15.409537,1.10868,0.9274–1.2981,inconclusive,70/73/1,0.928014
   dense-1024,cuda-cublas,12.969179,12.619217–13.061041,14.666550,14.006196–14.776200,1.13088,1.0004–1.2527,inconclusive,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.420195,20.310178–22.154037,23.441527,23.217088–23.735183,1.09437,1.0293–1.1914,inconclusive,70/73/1,0.928014
   dense-2048,cuda-native,20.723479,20.651594–21.392236,21.500293,21.134262–21.735993,1.03748,0.9645–1.0732,inconclusive,70/73/1,0.871762
   dense-2048,cuda-cublas,21.298671,20.981434–21.933403,21.685540,21.274891–22.638927,1.01816,0.9596–1.0950,inconclusive,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.267417,27.856112–28.820904,29.355132,28.935983–30.218388,1.03848,0.9942–1.0880,inconclusive,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.958106,4.511310–5.414434,4.480462,4.367547–4.551558,0.90366,0.7701–1.0458,within 5%,49/51/0,0.648887
   dense-64,cuda-cublas,7.238997,6.952524–7.439982,6.980843,6.976151–6.990018,0.96434,0.8320–1.1065,inconclusive,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.615526,14.945384–18.101284,15.322932,15.241425–15.335229,0.92221,0.8082–1.0833,inconclusive,49/51/0,0.611009
   dense-256,cuda-native,7.318467,6.535211–9.310261,6.303477,6.072243–6.489874,0.86131,0.6276–1.0154,within 5%,67/70/1,0.979365
   dense-256,cuda-cublas,9.420893,8.703219–9.973629,8.559380,7.517732–9.830570,0.90855,0.7192–1.1554,inconclusive,67/70/1,0.994421
   dense-256,torch-cuda-eager,22.043427,21.068174–26.013520,20.136776,19.305942–20.988536,0.91350,0.7380–1.0044,within 5%,67/70/1,0.977359
   dense-1024,cuda-native,9.356555,9.213931–9.416039,8.329847,8.205682–8.381606,0.89027,0.7577–0.9628,within 5%,70/73/1,0.928014
   dense-1024,cuda-cublas,10.120622,9.675468–10.695447,9.713313,9.243750–10.045236,0.95975,0.8277–1.0643,inconclusive,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.322642,21.201767–22.084372,19.749612,19.421816–20.349317,0.92623,0.8419–0.9730,within 5%,70/73/1,0.928014
   dense-2048,cuda-native,18.239783,17.909893–19.024649,17.593325,16.732059–18.638730,0.96456,0.8556–1.0741,inconclusive,70/73/1,0.871762
   dense-2048,cuda-cublas,19.247432,18.169121–20.327390,18.834974,17.273602–19.482138,0.97857,0.8437–1.0913,inconclusive,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.938409,29.788967–30.445128,27.898975,26.573736–30.453212,0.93188,0.8628–1.0408,within 5%,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.148835,1.067803–1.279684,1.293035,1.252372–1.329819,1.12552,0.9022–1.3085,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,3.175622,2.973723–3.575760,3.383411,3.341808–3.398331,1.06543,0.9041–1.1703,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.672486,2.404158–3.215788,3.078105,3.049989–3.113431,1.15178,0.8974–1.3509,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,1.025435,0.927322–1.047999,1.157195,1.072326–1.336868,1.12849,0.9767–1.4642,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,2.913485,2.726187–3.158002,3.376632,3.346917–3.442177,1.15897,1.0385–1.3284,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.312146,2.149276–2.520730,2.843663,2.745783–3.013510,1.22988,1.0232–1.4664,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.327343,1.292107–1.363278,1.486171,1.464773–1.516747,1.11966,0.9925–1.2398,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,3.088715,2.952999–3.317144,3.378528,3.370707–3.387824,1.09383,0.9927–1.1771,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.172534,2.023207–2.407300,2.456437,2.342446–2.532166,1.13068,0.9246–1.3485,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.668646,1.608481–1.691979,1.606190,1.489635–1.809052,0.96257,0.8471–1.1429,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,3.433789,3.304135–3.466320,3.214390,3.150298–3.428475,0.93611,0.8755–1.0668,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.703196,2.646180–2.744915,2.462680,2.347752–2.593837,0.91103,0.8058–1.0562,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.822068,0.811028–0.829899,0.785447,0.782057–0.803102,0.95545,0.8942–1.0508,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,2.848774,2.847453–2.879440,2.631548,2.586017–2.674647,0.92375,0.8335–0.9858,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.659801,2.530358–2.768668,2.575579,2.493769–2.792286,0.96834,0.8390–1.1219,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.744645,0.744501–0.744645,0.723645,0.719113–0.724900,0.97180,0.9136–1.0141,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,2.683412,2.579143–2.748382,2.570172,2.530750–2.664811,0.95780,0.9060–1.0646,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.198082,2.129034–2.250818,2.158510,2.053362–2.325639,0.98200,0.8523–1.1202,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.026048,1.001931–1.056528,1.059681,1.014325–1.089045,1.03278,0.9488–1.0969,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,2.712432,2.688716–2.744487,2.711186,2.633013–2.814614,0.99954,0.9087–1.1058,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.176731,2.109250–2.211806,2.282760,2.101776–2.385665,1.04871,0.8889–1.1532,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.344855,1.333973–1.348355,1.350888,1.329449–1.448594,1.00449,0.9696–1.1458,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,2.952529,2.918608–3.137034,3.028773,2.944456–3.232934,1.02582,0.9283–1.1207,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.303138,2.136286–2.471726,2.504496,2.458140–2.557809,1.08743,0.9435–1.2188,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.211960,1.169873–1.230563,1.246935,1.228885–1.254141,1.02886,0.9250–1.1015,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,1.242849,1.148566–1.438000,1.279424,1.244671–1.297194,1.02943,0.8305–1.1520,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,3.020078,2.699664–3.464084,3.123741,2.997441–3.304105,1.03432,0.8218–1.2449,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,1.065230,1.032461–1.126694,1.148883,1.090025–1.155936,1.07853,0.9579–1.1520,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,1.062603,1.031986–1.084192,1.105408,1.095670–1.113050,1.04028,0.9763–1.1587,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.523412,2.440196–2.568271,2.665855,2.646980–2.829159,1.05645,0.9691–1.2112,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.371506,1.336181–1.479844,1.574534,1.569454–1.574534,1.14803,1.0121–1.2606,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,1.224973,1.189229–1.250009,1.379453,1.339786–1.398062,1.12611,1.0463–1.2202,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.569501,2.299796–2.718920,2.942484,2.791991–3.258278,1.14516,1.0006–1.4770,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.784979,1.717823–1.926554,1.857889,1.799575–1.885850,1.04085,0.9096–1.1380,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.640162,1.523467–1.867833,1.640367,1.601130–1.685409,1.00012,0.7927–1.1252,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.792294,2.502795–3.175930,2.757448,2.720937–2.816527,0.98752,0.8291–1.1550,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.877521,0.838837–1.069317,0.836047,0.816065–0.862156,0.95274,0.7466–1.0432,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,0.972186,0.917690–1.013689,0.922022,0.908374–0.943342,0.94840,0.7857–1.0499,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.844071,2.603395–3.128595,2.745232,2.588211–2.842109,0.96525,0.8006–1.1376,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.807076,0.796100–0.812168,0.753803,0.750744–0.771586,0.93399,0.8914–1.0136,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,0.869463,0.868022–0.869463,0.839700,0.757115–0.908761,0.96577,0.8347–1.1050,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.772206,2.592509–2.994291,2.370183,2.370183–2.423693,0.85498,0.7455–1.0209,within 5%,4/6/0,0.209854
   dense-1024,cuda-native,1.110392,1.099149–1.161992,1.078235,1.051075–1.094998,0.97104,0.8908–1.0711,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,0.958198,0.956453–0.962375,0.955894,0.904855–1.091875,0.99760,0.9039–1.2691,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.536409,2.477205–2.609453,2.437807,2.251127–2.786450,0.96113,0.8432–1.2604,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.486492,1.464488–1.560732,1.566050,1.535071–1.648108,1.05352,0.9515–1.1502,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.271726,1.250188–1.284863,1.345154,1.332597–1.361391,1.05774,0.9843–1.1244,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.605779,2.576443–2.644598,2.555153,2.522720–2.697633,0.98057,0.8924–1.1080,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.115659,1.077836–1.173229,1.290857,1.078509–1.720284,1.15704,0.9144–1.7486,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,1.203971,1.108972–1.363744,1.412484,1.132160–2.064938,1.17319,0.7996–1.9184,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.450992,2.265464–2.775006,3.124087,2.321696–4.076408,1.27462,0.8324–1.8110,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.906042,0.884392–0.926696,1.055621,1.001681–1.083035,1.16509,1.0582–1.2594,regression,4/6/0,0.209854
   dense-256,cuda-cublas,0.929852,0.873166–0.988967,0.993488,0.956736–1.036269,1.06844,0.9585–1.2023,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.855552,1.782391–2.146524,2.149329,2.109222–2.151924,1.15832,0.8631–1.2412,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.225636,1.205553–1.234992,1.318239,1.274443–1.373106,1.07556,1.0112–1.1645,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,1.098774,1.073492–1.115592,1.136211,1.112354–1.140335,1.03407,0.9863–1.1030,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,1.922899,1.892256–1.955464,2.122151,1.987136–2.367614,1.10362,0.9851–1.3291,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.591410,1.590284–1.594009,1.677698,1.660426–1.690355,1.05422,0.9965–1.1038,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.429221,1.418883–1.435116,1.477210,1.459999–1.555427,1.03358,0.9718–1.1338,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.134995,2.105147–2.134995,2.189638,2.073052–2.303970,1.02559,0.9286–1.1570,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.744911,0.727204–0.766778,0.792945,0.769998–0.796427,1.06448,0.9741–1.1862,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,0.785900,0.773809–0.812098,0.811352,0.755027–0.848996,1.03239,0.9107–1.1452,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.159779,2.098845–2.220444,2.375952,2.199975–2.511372,1.10009,0.9363–1.2424,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.712705,0.702860–0.733469,0.763787,0.759822–0.767930,1.07167,0.9877–1.1832,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,0.726788,0.708832–0.731567,0.725878,0.700228–0.730787,0.99875,0.9222–1.0802,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.044892,2.000097–2.099070,2.056516,1.980679–2.111378,1.00568,0.8884–1.1201,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,0.998027,0.948211–1.044831,1.044334,1.010714–1.063390,1.04640,0.9592–1.1320,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,0.863951,0.825384–0.924487,0.890622,0.832925–0.919002,1.03087,0.9001–1.1313,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.107824,1.704477–2.252203,2.091747,2.015678–2.314008,0.99237,0.8538–1.3935,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.352891,1.294159–1.418786,1.439126,1.318542–1.818169,1.06374,0.9214–1.4335,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.230195,1.133638–1.299477,1.418946,1.168682–1.772522,1.15343,0.8916–1.5763,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.078502,1.938821–2.221036,2.274089,1.948582–3.035519,1.09410,0.8745–1.5811,inconclusive,3/5/0,0.836412

Current-source framework comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are descriptive ratios of current pooled-round medians from the same
matched configuration, not strict comparisons across frameworks or scopes.
Every higher current-native median remains listed below.

.. csv-table:: All current matched scopes
   :class: benchmark-results
   :header-rows: 1

   Configuration,Native,Reference,Lower/higher/equal,Current ratio range,Same known I/M/R
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4042–0.7324,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.5225–0.7387,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2924–0.6306,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.4251–0.6751,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4069–0.6522,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-cublas,torch-cuda-eager,0/4/0,1.0992–1.3754,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3050–0.5394,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-cublas,torch-cuda-eager,0/4/0,1.0217–1.2093,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.3992–0.6738,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4096–0.5949,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3045–0.6129,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3359–0.5264,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4132–0.7662,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4521–0.6746,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3337–0.6328,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3415–0.6240,True

.. csv-table:: Complete current higher-median inventory
   :class: benchmark-results
   :header-rows: 1

   Configuration,Case,Native/reference,Native pooled ms,Reference pooled ms,Ratio,Native I/M/R,Reference I/M/R
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-64,cuda-cublas/torch-cuda-eager,3.383411,3.078105,1.09919,5/7/0; 5/7/0,5/7/0; 5/7/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-256,cuda-cublas/torch-cuda-eager,3.376632,2.843663,1.18742,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-1024,cuda-cublas/torch-cuda-eager,3.378528,2.456437,1.37538,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-2048,cuda-cublas/torch-cuda-eager,3.214390,2.462680,1.30524,3/5/0; 3/5/0,3/5/0; 3/5/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-64,cuda-cublas/torch-cuda-eager,2.631548,2.575579,1.02173,5/7/0; 5/7/0,5/7/0; 5/7/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-256,cuda-cublas/torch-cuda-eager,2.570172,2.158510,1.19072,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-1024,cuda-cublas/torch-cuda-eager,2.711186,2.282760,1.18768,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-2048,cuda-cublas/torch-cuda-eager,3.028773,2.504496,1.20933,3/5/0; 3/5/0,3/5/0; 3/5/0

Source, controls and raw evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* baseline/cuda-release: compiled ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``, tree ``06f324529806d3edf005e4ecbb6a3c27f048f946``, identity SHA256 ``7a956531880eaee793f22d75c4cc748f53c20ee8d5569fc0f2fe4cf67411d416``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``760af83cc19b2e0c6b06f9373166d7bd7022d4a2a779677f5a3d141dec774240``.
* current/cuda-release: compiled ``e5c587889906511b0f9e74f42d88d8e857dea3d0``, tree ``241772b9eb7a0ed25bfbf8e82bbe4118386991ea``, identity SHA256 ``aba137408ee38d8d66ed6e90c092ad5cbe0afe31193999fd19cd936fa0d23d77``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``f66428197ece464a22112fd0f6aee8e922f92fb462a7e38d9b204b194a4d2213``.

Manifest SHA256: ``f7d874200fee86d850deb96e962257fbe057dfe5f4e945740cd54b34bb24498f``.
Plan SHA256: ``a3b43a52d9cb01a262df97f064627f64729c25d7efb10a1e367854d682fbaf97``.

.. csv-table:: Every full strict report and actual comparator
   :class: benchmark-results
   :header-rows: 1

   Configuration,Verdict,Report SHA256,Comparator SHA256,Comparator path
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,inconclusive,45aa248dd5adb06814f515099c0fc21a2964bd026370194db8d2e81cf1d43cc3,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,inconclusive,237a9c526d6895f9dbb2e3b987aafc932c99853fcf20d6023977f0ae0e7d6387,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,inconclusive,6d3b96e65c37f14d33db55f7c75e966d73687e7a6ad37968dccd3fab6e08537e,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,inconclusive,00d4ba3a0c05aad38d26233711ac9cdefece1d067178f0b7d820e9ffde73b31a,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,inconclusive,b1d543830fdd97e6fd51dd783000d078e7d14aeb375525096b3a5b477979d252,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,inconclusive,0035674076b58c384418c3440342656b350b5f75d29d7c3bd0281fc93877d8a5,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,regression,da560e993cef119a690924a5c8a040b93ddd9cbdfdd86ca9539dcaede37b5a3e,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,inconclusive,85ae1ff9ba60043eeff33bd255a9ffc58491e66d4e730cb2b1cd64f97b725611,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/compare_performance.py

.. csv-table:: All original raw/import/log identities
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Import SHA256,Log SHA256
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b01-baseline.json,19dc286b5d6217ac233ef41653123f8093e7d0cfc181b7291397c90e288b34d7,b67c6ef8ee1427535a4e45cd1a53a6887918a2d82e2faa2ba67a600b7ea22559,c5ab8b88ea093a402532ebf7cce8fb9d1145bbaea60623d9886c590834009376
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b02-current.json,3319ad1c3e63f1518b2e69c10d01ecfb36b59702f17cfd3b6350b08a2c560fa1,e2d854eebdf85d12cdd607620aa1f4531661e1153e6a2590bacfcfde7f92194a,67a3188d10c26c700a5789ef541ed5d9d654e40b15dc48dc05f62eedbeb40a39
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b03-current.json,3da761940a0cac147fb0730521d73915de4055681d16d04da85f55690c500941,fb0be82f8f5a549e62b756afa0d87bcccf93d4500900f83b6eb47b25c99420fc,9e17f98ee11f95cc58e2025054b99a673e4ec17b7057e90d259a9abb7ac8f3e6
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b04-baseline.json,81a1e1b7b7fab5bc4f43d186549dc529f3d14a535e8b879752f34fc65bab1fa9,f1676f49404eea2f55379215c3adf2a00210876ad66d91e0de42e36477622160,d8c2f78202ba17c74c458affe1cf8bc9b5092555ab95aa750bce2168a65bdb91
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b01-baseline.json,b7153f1bfd694668cd43922bdbfa6ce208b5ff4e6ad34c2a9e8ec7be132b00b8,ac7c258386833a5191be0dbb0d6109fe76e20c116856d5abf08e2530cdb9268f,a76bcba96710ce2371e3d9e176d4250fa2eeab6fe2054cac17fb74154e91ac18
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b02-current.json,dadddebad67d99d3106b01db77d2c2a3e5d13aa8abc4715d2dfc06952c0902c9,c03920ea2a29c19f7e1a71872ed45177d5d967f38770ea0ed75dad15125ef6e6,6f2feaf712fc7c2909348abd559e997282e46c39a14b1aad4500f2296dd3092a
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b03-current.json,40304371589048188515f1a38b76b63f6fc0e71a8972cacf8e875a46bc169273,1cf215018dafce90e34cb716ca02eaedf3145272a406125e5f922dff6313599e,ed855b719b516cc44cd127967b8a869c4697b3fb2d5dc30de86e1d0b377cb215
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b04-baseline.json,b3f21f8a30ddf726836b0325a7eb7854e061de469007d41f3a355887a0d04f5f,96a9bbb8cb605b867c550d562ae93cad1b9079c6bd04c3aa794c4bd4e2790db7,1c768b631315f760e478232c7a1ba487c3b0713fd6730ca5bb663608d1297de1
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b01-baseline.json,5a3b4bb509b5e43ce68a12ad683b3c28cd4384f565dd77421631ed14e963d296,220713e7da8c00b1a71321d93e89035612544bcfcfc53dc221c70906d4f8b820,1cf8041c11d64c5f926994dfc79b7d465a2a99be2f31df401d2d6cdab4044968
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b02-current.json,960fbdabbaf07d05e120680bcdedc737e8ec4596c8d0e230c16c74112b35609a,d1a9d062282dfb17475d768db159e65a2d362a8f3535e815052b6588063a6dd4,8eb422dc89cb69b1e9de4e479b9bf99f841300ebe6be983c293bf0a9af918884
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b03-current.json,71737878bff6f9feae27ffbb6a683448f9c18604994c0af3f237ffe64cd2b7a1,38b9504d076b6ec45e8502787d15c113e193040841cc9620d360d14b5fc84e8a,6257e0f2035d7faf0c92312ebd4cbfd9d948b2acfcaacb70d80bc024484a654d
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b04-baseline.json,f972ca9d0a771daf35523d8523ecd7626765fde7d8bacd6a8c5b473ee2daf126,80a8c64fe3a2bb70fd5fb0e705c71df23816a34f6e762777d0b104d782ff7523,ee239d883675f9fc2e59200cda7ef769e4639eca8208b132263d5f793c60fd9b
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b01-baseline.json,3b5e7a52ef2ddfcedd4aa173f3577b46d3f3f0e65a8d36ef69533e4995a390c1,1498af8fac317680d3f4763cd0c25853ea5f8e605ce97cb9205a7ffe5819a22b,c240abe20841efa9b882a9bc2cc7682ac29198884d399e4ae8b322c5fdfd86bb
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b02-current.json,c58db327968b9d165ee14e196501f0e5ea92b78ed03ea04a93978ab702dcac70,6c02d14db8fcff17831f3d6d0a1809000413508bb45639134c4272777fca4213,74d8113df84a2cfa762c6538b3141bc27208ac94bd04072904b7603b09a03dde
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b03-current.json,9424dbe675ae821f025c91f0220d3de92bea66be8556120efe5df8a5e3cdd5ad,f93075ad51ca15d4d51d16f000075cff56c62d0d0f88630d3daf6fb10eebed8d,c1b524d0f72c95f706020f2a5621faddae701a79b80afd154c8b02cc5f155d7c
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b04-baseline.json,2e1366c87e77d9d8ece4f8b7bf202ae7da222a9d3babc6b54a9b2b6b8651d443,93e868405e27457dae68fbf5573d04eb401a2136254a08ad971ad45c4963d09e,0f00250a30be3b1705059565d99cde9d79f71a9a095b7d9150d274f8813c4ac3
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b01-baseline.json,116ef578940cb4bbbd145e89896bfc5ecbaeaeac05e0970a123638a9a9606ef0,d270e83b27ffdc6901d6290993c0743424409e3b12bd6c511e81bb8717dbaa4c,d314242a548164c8bf03ead5ee5462aa3824b7e3d9e65b5772011e8049d9d3bf
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b02-current.json,9fe83ca97e24e80564e97a269600651a2a2ae6598ff6897c7b0ae7f6ad657cab,c6e44f7be5f55233da79c01643ef4b27729e46a848df900d5215bc66d60c8fc2,78348ae51bad33805db7e3953f5319a997367eb3881583c80828bf6e6a658960
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b03-current.json,1b933c32294f59ae906c72d67cd88efe37bd61e096971a13082d06ebad59bf41,14294477812994237e85088eace88dc6673d3dd8031464909305e8442db93698,7df6be65ffeee2073ffa6eb19d665a8d668bc143a8fc5db5c38f59296aa8bab0
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b04-baseline.json,97d02180723d4a6397d941102d839ac543f0b32362a446812f6a9763ae2421ba,b95c3ea0a2009f9d52d09e71571bb824ce8ff0e6e16c17161b9d839f273807d3,f0c29ffa876f3fcebfaf6ee6ac2cb0a7c14236ccd5edc3ed9eafd07795d96cda
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b01-baseline.json,58b008b8fc40ba611c6c16aca6409d274e22a0cfad7838c78e2ebddff93a4bc8,9d208d4b6c4a65eec07c54c662d17cfcb7703de89ecaa610c27e037e84be87f6,75f6727813d4dfe40b4a24061833cf9b256549e75571dcc62250a7f9b971e3e3
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b02-current.json,33192be8317b07ce61440295e6bdaac48faac0382c48532b63dface06fe6df79,517f3e5c3e69732a73ca8924ddfcc2db524ff097a636f0b8e15162a125f061ca,a7c9165d95e684c4dc43b7f3ffc39c97183c2dc93319c72383e6bc5afd352294
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b03-current.json,02c8094acd1c0db1f8f681b2822955c656a43ba99bd6f11b174ae8bbd48d38c2,9d15ebb775aff094c7eace30d2bae242b7d2c52425d11f1b8b2ea5188ac26473,0533cb38b40bae5127002036b4f3fa4d46baf64088346e940b666fecd5552f77
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b04-baseline.json,d5154ca50752e560cf9df30cbb7be34eb774795f72c7a19fd7a2aafe383760f6,419b0c2845a35cb74e7ffc22692b00c5800dc4a9904a5267ee1c3854a05db229,6d8f67339cb664b4d96724ead6d8c5e106f70991d59beb02ebfd582636489693
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline.json,808a1fa271aa73543c381ac9bce55d3487dfc9060c8575d635fbe2afb8b51fa4,125683e26f3e6b0d9465b8722340bbcd52dbb0e7e8e574904b293b6b987d9218,d9429750188b64acf38c5d366f5329661dc0568e00b6540caaf8256044f47d06
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b02-current.json,842d90db8836356548927b9ed3a0738dbf59179b0cced8d36e86babd969d7b79,87721dd422d20fcb16f6f717b4f14d67510afb7045d81a2a224550206ef2314c,47fb6a3bc345c45799495cef94a506e31b4b49649ce28879b26bcd613b2eff4d
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b03-current.json,b9f7ba548ffcb83ecc0f39b5889f64f856a055a1742ec08bb8c8b383d54d07c5,5e7035b5fb77574621e3b0f820043b09c0fb29fb8049bd75d18c9f4d861046d1,ba2475e1a2dd05084b7f9999b5b9ef77b7347e1aa9342d4714183532fa74a8e2
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline.json,90854a50fbcb3063244c7303ed0038606d13adac78cbe91344a1ce1ccc7190bb,c396a0326e39d33c6612b7cf0b172ba06cc6993ee05bdffae10734d406cf8db0,0bb6092ef35c0b92f0bd1a09e4c893566a6dac5e7da389a532d85dce34a7d95b
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline.json,e8933534f1c6f38fba8fc653cfd19ec9f71f8fd2e8b3bcda008783e199c24b84,832ae606ca2d19af98140b5e6736def20e15704355976004d4e09bd241b13ff0,ab8057eaf6ca7412486d5e80262f712d2973336fe2d3de58106d0081ce6c6082
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current.json,79a870966d582ac6aabf93432860f2597677ebfb65d7f0cceca4987afeac19f7,38607aca1978e15d16c2618aecc72e917fda7b6c6b986d797bda12bbc2e697d0,daace775613c54f18c5a4e82d35d7b3b655fcdf7b9bbca84142a0194e900ca2f
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current.json,a7cd755e7a8eaa91518d36bb42f52a566f90940d0cd591324465c3755213c3e5,4988cea683b7492c08787eef557a1688eb06614bb8e9bb45e1c65f8a94ec1ffd,d017a1813afb3322539af5920b622775b9b638ff938116479bff2d614d649e2f
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline.json,31c5f7cdea34dd98a6a3d6f6971ca5985730d3485cddf0dd6723a1b9c3d90342,1aeddf7ece3c07d8cab1809cd96d5333e8260629dfb88418966a5ad93a0e3a31,b3692bffbc3c6cf081555f87245cc092cebdefc6762c1146412f8f94f808d48e


fresh-span-abba (fresh) fixed A/B campaign
------------------------------------------

Whole-campaign verdict: **inconclusive**.
All 32 artifacts/384 candidate/case outcomes are retained.

This follow-up variant was selected after observing the rejected initial
fresh-variant trial. Its intervals are unadjusted for that selection and
multiple variants; they are exploratory rather than independent confirmatory
evidence. The original rejected trial remains separately reported, without
replacement, pooling or a claim that resource layout caused its regression.

.. csv-table:: Every native and framework strict result
   :class: benchmark-results
   :header-rows: 1

   Candidate,Role,Rows,Within 5%,Inconclusive,Regression,Comparison failed
   cuda-native,native,32,13,19,0,0
   cuda-cublas,native,32,14,18,0,0
   torch-cuda-eager,framework control,32,7,25,0,0

All candidate rows
~~~~~~~~~~~~~~~~~~

A/B and current/reference ratios below one indicate a lower measured numerator.
Ranges bracket the two process medians and are not confidence intervals.
Saved strict intervals compare current against A/B baseline; they do not give
confidence for the separate current/native-to-framework comparison.

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,6.492710,6.475779–6.879581,6.262711,6.242748–6.298239,0.96458,0.8169–0.9973,within 5%,49/51/0,0.648887
   dense-64,cuda-cublas,8.596882,8.520694–8.639883,8.368508,8.240198–8.483339,0.97344,0.9156–1.0275,within 5%,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.055709,15.897201–16.472330,14.753621,14.633412–14.846294,0.91890,0.8348–1.0165,within 5%,49/51/0,0.611009
   dense-256,cuda-native,9.564447,9.007661–10.216734,9.341464,9.315929–9.341464,0.97669,0.8505–1.0778,inconclusive,67/70/1,0.979365
   dense-256,cuda-cublas,12.509035,12.184772–12.627019,10.911715,10.894336–11.191428,0.87231,0.8108–0.9683,within 5%,67/70/1,0.994421
   dense-256,torch-cuda-eager,22.280898,21.522447–23.254687,20.682583,20.562271–20.735802,0.92827,0.8578–0.9761,within 5%,67/70/1,0.977359
   dense-1024,cuda-native,12.915876,12.310626–13.792145,13.977778,12.603044–15.037432,1.08222,0.9129–1.2496,inconclusive,70/73/1,0.928014
   dense-1024,cuda-cublas,13.594882,13.260993–14.199153,15.020853,13.835985–16.652614,1.10489,0.9131–1.2848,inconclusive,70/73/1,0.928014
   dense-1024,torch-cuda-eager,21.166487,20.042276–22.461407,22.082067,21.658887–24.694976,1.04326,0.9472–1.2710,inconclusive,70/73/1,0.928014
   dense-2048,cuda-native,21.044254,20.518545–21.894064,21.927488,21.678189–22.469369,1.04197,0.9836–1.1211,inconclusive,70/73/1,0.871762
   dense-2048,cuda-cublas,22.227856,22.177400–22.259943,21.424021,21.275544–21.524456,0.96384,0.9331–1.0006,within 5%,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.526084,28.012313–29.235084,29.681604,28.045909–30.358908,1.04051,0.9575–1.0866,inconclusive,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.707842,4.658784–4.727042,4.568682,4.527483–4.618936,0.97044,0.9186–1.0344,within 5%,49/51/0,0.648887
   dense-64,cuda-cublas,6.975551,6.952021–7.443373,6.706884,6.391678–6.808772,0.96148,0.8265–1.0318,within 5%,49/51/0,0.611009
   dense-64,torch-cuda-eager,15.121043,13.807603–16.062929,16.048172,15.898145–16.378561,1.06131,0.9517–1.2083,inconclusive,49/51/0,0.611009
   dense-256,cuda-native,6.901218,6.894364–6.924187,6.776741,6.616523–6.792236,0.98196,0.9075–1.0186,within 5%,67/70/1,0.979365
   dense-256,cuda-cublas,9.829019,9.178750–10.566930,8.874627,8.845365–9.104415,0.90290,0.7991–1.0106,within 5%,67/70/1,0.994421
   dense-256,torch-cuda-eager,22.021948,20.911348–22.174436,21.938269,20.460935–23.026325,0.99620,0.8862–1.1095,inconclusive,67/70/1,0.977359
   dense-1024,cuda-native,9.439955,9.343283–9.710426,9.420049,9.395634–9.420049,0.99789,0.8964–1.0743,inconclusive,70/73/1,0.928014
   dense-1024,cuda-cublas,11.111570,11.111570–11.261017,10.973718,10.482556–11.124723,0.98759,0.8330–1.1361,inconclusive,70/73/1,0.928014
   dense-1024,torch-cuda-eager,22.104254,22.104254–22.237153,21.454560,20.919110–22.013459,0.97061,0.9174–1.0228,within 5%,70/73/1,0.928014
   dense-2048,cuda-native,18.151483,18.116229–18.252569,18.125956,17.965605–18.248862,0.99859,0.8950–1.0858,inconclusive,70/73/1,0.871762
   dense-2048,cuda-cublas,19.073935,19.057133–19.111118,18.927781,18.822575–18.958759,0.99234,0.9337–1.0444,within 5%,70/73/1,0.871762
   dense-2048,torch-cuda-eager,29.569926,29.234782–30.014788,27.701878,27.397935–27.991996,0.93683,0.9077–0.9637,within 5%,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.266555,1.229469–1.394025,1.247036,1.233672–1.296671,0.98459,0.8000–1.2006,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,3.397706,3.389849–3.399818,3.378935,3.230551–3.688777,0.99448,0.9001–1.1367,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.982594,2.855068–3.156010,2.653832,2.598919–2.845048,0.88977,0.8074–1.0559,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.981962,0.981962–0.986020,1.077880,0.992350–1.130281,1.09768,0.9465–1.2368,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,2.928189,2.844587–3.117306,3.007370,2.944055–3.212563,1.02704,0.9398–1.1728,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.328559,2.303974–2.335638,2.410013,2.314339–2.580392,1.03498,0.9317–1.2260,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.372162,1.362731–1.372162,1.407542,1.307038–1.497481,1.02578,0.9335–1.1632,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,3.185598,3.114729–3.200362,3.396927,3.117334–3.509448,1.06634,0.9454–1.1476,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.337518,2.329017–2.367221,2.468707,2.467648–2.468707,1.05612,0.9767–1.2512,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.843573,1.829899–1.861125,1.618324,1.536388–1.846562,0.87782,0.7821–1.0562,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,3.872453,3.628766–4.234067,3.370900,3.198328–3.497922,0.87048,0.7416–0.9904,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,3.039229,3.039229–3.060577,2.440940,2.257434–2.512784,0.80314,0.6975–0.8870,within 5%,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.798799,0.795222–0.805222,0.941199,0.886746–1.003705,1.17827,1.0426–1.3068,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,2.798882,2.797344–2.811113,3.063679,2.856313–3.338705,1.09461,0.9679–1.3223,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.589501,2.517878–2.595079,2.919957,2.701872–3.143229,1.12761,0.9979–1.3239,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.788459,0.779026–0.792480,0.903934,0.733626–1.259531,1.14646,0.9002–1.7935,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,2.690084,2.688299–2.705843,3.097362,2.585812–4.159072,1.15140,0.9368–1.6266,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.356784,2.356416–2.356784,2.682766,2.060376–4.148799,1.13832,0.8326–1.9135,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.130004,1.121617–1.192038,1.208567,0.994286–1.425980,1.06952,0.8184–1.3366,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,3.087725,3.081171–3.090010,3.320677,2.902990–3.996428,1.07544,0.9021–1.4187,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.606009,2.344104–2.735156,2.706859,2.341058–3.457313,1.03870,0.8359–1.5683,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.489747,1.419120–1.573346,1.382803,1.377790–1.407878,0.92821,0.8274–1.0195,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,3.144896,3.069691–3.231789,3.201679,3.024430–3.383576,1.01806,0.9043–1.1430,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.488376,2.473180–2.579085,2.457333,2.271940–2.766853,0.98752,0.8585–1.1903,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.265761,1.264669–1.265761,1.222193,1.147981–1.247621,0.96558,0.8870–1.0235,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,1.256360,1.252603–1.274698,1.266875,1.227461–1.333627,1.00837,0.8976–1.1571,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,3.115334,3.069008–3.147870,3.161504,2.937587–3.543308,1.01482,0.8942–1.1930,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,1.136534,1.066345–1.230172,1.141242,1.113314–1.160324,1.00414,0.8275–1.2458,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,1.111992,1.087347–1.188699,1.161641,1.132564–1.165538,1.04465,0.9154–1.1326,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.795138,2.582121–3.229781,2.852864,2.726445–3.150254,1.02065,0.8011–1.2506,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.544483,1.322973–1.906103,1.641867,1.619741–1.647288,1.06305,0.8125–1.3614,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,1.289686,1.200869–1.685232,1.351320,1.330807–1.378114,1.04779,0.7590–1.2389,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.914446,2.354602–4.245139,3.254523,3.145407–3.254523,1.11669,0.6762–1.5223,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.825524,1.710330–2.102473,1.745843,1.628840–1.817692,0.95635,0.7619–1.1199,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.718138,1.482538–2.218884,1.605112,1.490426–1.737405,0.93422,0.6715–1.1768,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,3.003433,2.458655–4.278227,2.972180,2.485482–3.723283,0.98959,0.5720–1.5427,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.845392,0.809280–0.868799,0.880169,0.840543–0.989590,1.04114,0.9491–1.3235,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,0.910698,0.886803–0.962322,0.931658,0.918962–0.962815,1.02302,0.9172–1.1400,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.748053,2.741615–2.884665,2.791490,2.707608–2.984863,1.01581,0.9142–1.2219,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.786393,0.757095–0.886089,0.853531,0.807536–0.879088,1.08538,0.8853–1.2080,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,0.864039,0.821983–0.901191,0.864681,0.792915–1.096519,1.00074,0.8750–1.3637,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.626563,2.376130–2.908967,2.876340,2.620608–2.985366,1.09510,0.8733–1.2952,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.067304,1.063821–1.076678,1.079902,1.069503–1.086544,1.01180,0.9622–1.0354,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,0.964318,0.963151–0.965573,0.955630,0.950603–0.957287,0.99099,0.9444–1.0333,within 5%,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.308142,2.300397–2.342414,2.381022,2.381022–2.393305,1.03158,0.9463–1.1206,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.507021,1.460899–1.527416,1.519600,1.489032–1.534117,1.00835,0.9504–1.0609,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.309915,1.299411–1.340023,1.326377,1.320107–1.331659,1.01257,0.9462–1.0412,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.442630,2.387434–2.536990,2.550007,2.494976–2.560583,1.04396,0.9432–1.1190,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.086500,1.073859–1.103111,1.045322,1.039050–1.059414,0.96210,0.9202–1.0223,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,1.114066,1.111733–1.114643,1.082314,1.082314–1.093741,0.97150,0.9181–1.0603,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.411312,2.366190–2.445047,2.266996,2.259046–2.266996,0.94015,0.8789–1.0091,within 5%,5/7/0,0.0796191
   dense-256,cuda-native,0.952408,0.949527–0.957789,0.916674,0.911605–0.923790,0.96248,0.8712–1.0279,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,0.959397,0.957405–0.961314,0.940688,0.904094–0.979206,0.98050,0.9144–1.0492,within 5%,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.931015,1.897104–2.005447,1.853854,1.846626–1.956182,0.96004,0.8915–1.0847,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.260374,1.195303–1.312348,1.271768,1.243469–1.324194,1.00904,0.9366–1.1187,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,1.177188,1.135690–1.186303,1.144365,1.107617–1.148024,0.97212,0.9047–1.0606,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.013682,1.777284–2.118470,1.922793,1.917353–1.924268,0.95486,0.8488–1.1198,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.614268,1.592365–1.742449,1.579406,1.527241–1.590879,0.97840,0.8574–1.0160,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,1.457574,1.454629–1.479019,1.375128,1.362593–1.378348,0.94344,0.8779–1.0072,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.191901,2.118638–2.202979,2.061099,2.030724–2.142581,0.94033,0.8639–1.0552,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.784088,0.732989–0.832766,0.729788,0.726796–0.731922,0.93075,0.7949–1.0236,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,0.890722,0.834290–0.965932,0.798679,0.788022–0.808387,0.89666,0.7797–1.0090,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.475205,2.214082–2.704721,2.253088,2.130445–2.330832,0.91026,0.7617–1.0816,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.714356,0.699291–0.744426,0.698982,0.698913–0.702105,0.97848,0.8944–1.0422,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,0.798995,0.744861–0.826347,0.740572,0.719957–0.753350,0.92688,0.8240–1.0414,within 5%,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.102325,2.087372–2.131005,1.864931,1.859509–1.871943,0.88708,0.8034–0.9827,within 5%,4/6/0,0.209854
   dense-1024,cuda-native,1.047702,1.024793–1.093608,1.016387,0.994820–1.042295,0.97011,0.8507–1.0283,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,0.912474,0.853364–1.013491,0.885826,0.883447–0.890259,0.97080,0.8124–1.0615,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.273217,1.931328–2.503678,1.904384,1.779768–2.280957,0.83775,0.7005–1.1865,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.420087,1.404957–1.422676,1.420836,1.400667–1.432044,1.00053,0.9473–1.0393,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,1.250657,1.244795–1.285658,1.224201,1.215957–1.243715,0.97885,0.9299–1.0229,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.085856,1.943342–2.194827,2.062934,2.054965–2.088296,0.98901,0.8998–1.0926,inconclusive,3/5/0,0.836412

Current-source framework comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are descriptive ratios of current pooled-round medians from the same
matched configuration, not strict comparisons across frameworks or scopes.
Every higher current-native median remains listed below.

.. csv-table:: All current matched scopes
   :class: benchmark-results
   :header-rows: 1

   Configuration,Native,Reference,Lower/higher/equal,Current ratio range,Same known I/M/R
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4245–0.7388,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.5276–0.7218,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2847–0.6543,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.4045–0.6833,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4473–0.6630,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-cublas,torch-cuda-eager,0/4/0,1.2479–1.3810,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3223–0.5627,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-cublas,torch-cuda-eager,0/4/0,1.0492–1.3029,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.3866–0.5874,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4007–0.5400,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2967–0.5959,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3006–0.5201,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4611–0.7663,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4774–0.6672,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3239–0.6887,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3545–0.5934,True

.. csv-table:: Complete current higher-median inventory
   :class: benchmark-results
   :header-rows: 1

   Configuration,Case,Native/reference,Native pooled ms,Reference pooled ms,Ratio,Native I/M/R,Reference I/M/R
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-64,cuda-cublas/torch-cuda-eager,3.378935,2.653832,1.27323,5/7/0; 5/7/0,5/7/0; 5/7/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-256,cuda-cublas/torch-cuda-eager,3.007370,2.410013,1.24787,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-1024,cuda-cublas/torch-cuda-eager,3.396927,2.468707,1.37599,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-2048,cuda-cublas/torch-cuda-eager,3.370900,2.440940,1.38098,3/5/0; 3/5/0,3/5/0; 3/5/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-64,cuda-cublas/torch-cuda-eager,3.063679,2.919957,1.04922,5/7/0; 5/7/0,5/7/0; 5/7/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-256,cuda-cublas/torch-cuda-eager,3.097362,2.682766,1.15454,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-1024,cuda-cublas/torch-cuda-eager,3.320677,2.706859,1.22676,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-2048,cuda-cublas/torch-cuda-eager,3.201679,2.457333,1.30291,3/5/0; 3/5/0,3/5/0; 3/5/0

Source, controls and raw evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* baseline/cuda-release: compiled ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``, tree ``06f324529806d3edf005e4ecbb6a3c27f048f946``, identity SHA256 ``7a956531880eaee793f22d75c4cc748f53c20ee8d5569fc0f2fe4cf67411d416``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``760af83cc19b2e0c6b06f9373166d7bd7022d4a2a779677f5a3d141dec774240``.
* current/cuda-release: compiled ``ce45ea9cc6d38cea05ef04cff8a321507e9684d5``, tree ``6cf00b82bac6f75e89093d1eb627f36d4fcc7d9b``, identity SHA256 ``e7232ead6d14f51f9f6081b982c63c3fc3c9c553876d2f4daa052ba5c8fc3d47``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``f5ae10070dd259e06e629113ac3c101d8c0e2248a2b5810239d3779cba11de25``.

Manifest SHA256: ``81cef076a347463044875f94f61748628e415182c63862426a32667553731b9a``.
Plan SHA256: ``64813c89d17e146815643d1e907a7ca489b13c4592eb31f031a474c801d72d21``.

.. csv-table:: Every full strict report and actual comparator
   :class: benchmark-results
   :header-rows: 1

   Configuration,Verdict,Report SHA256,Comparator SHA256,Comparator path
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,inconclusive,a66bf49d4492f69b67417b860266e4027163cba18e470191e85232ecf5e5796a,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,inconclusive,41505e326b15bb1e07a8c6da3397954cc1cd0c05d859abee258fcd10f18f76b0,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,inconclusive,7d0d0e0ff2325964d4567c345094db98aad1ce63506cd241450246080a90ac3a,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,inconclusive,139f2cf9a95260deb1863e67d68263437fcc6901eb3d679d97af378c5d827a03,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,inconclusive,9eca7380972c803754fd05f29c5e1d19e8fc2faef0d74f2375a12bdef40d3792,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,inconclusive,86a06f3c93323000fe099c801ef136586c8cc11d7b219a45ea1a2a54a4af8a69,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,inconclusive,79a2bb2b1da2f4f1d1edaf079f7acab75ec745a843c3b915fffeec25d335221b,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,inconclusive,fac1daea85ccd4602dd3481e1f7f843c8bd9f8c4235f96f3a51edff1b16f4545,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/compare_performance.py

.. csv-table:: All original raw/import/log identities
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Import SHA256,Log SHA256
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b01-baseline.json,c6b400c87a95f6d8dfafca285dcb2f0a93d15fdd4837d48465b521587ed0d066,67723d89710a992f438388095c5b0c3e877c18ba0da082430283e9a0c532d629,2f639791e6904266c89a514c1fc05b437bb756858b25a4388be8239c3883b8e9
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b02-current.json,38bbdefd8db8ce26bf54e0788a10e945d98ae9e5ad289374784b7afe009209ac,aa77909c2320b54828d2be46ebad906f04b3d7cd8b4647eb28537560209b075e,1a0197963b49f08222843859dd86f41076784aa5408823b616ac5bd290849b26
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b03-current.json,3cebff6a21c00e05011b7f9ee036d3aa9d408cf428f270a4136e95cab9d79131,0aacaeed4b11a7c08f5304659da5506319d3fee4eee35981b07cdd48d6a05dde,d8869c69a6c6734632a82ee82d5ff6cdef5378a36511d11b64ec060afa2956f1
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b04-baseline.json,6f1e3463127defbc40ca75e6f42c89ac326bbaf1145ef4f2f69929879d4e2a41,5d17b384e3c21933d1ab608db89b9052e134dc0f07460f105774edb97127d70f,fea5a2bca3be12a8f46f01504c04c21fbb1c65e9d064f38936046e69dd3fa25d
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b01-baseline.json,c95cf4c0cf56952e39a51735093db77c7ec6ea2a5b513d3dd760c25e326c0cad,857b8bb5771859da054154f014bbce8c8e275e6e4f9a21cb8c76f1ee3d911240,3c9d39285e37859c5ec9ebeb0c93da33f28cedee2b18932ac43086ae0301c981
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b02-current.json,884e085f724adc8ca03baed6542e445a08e4034ee2c14edbf781c82d6a0834ec,b6342cf6ef4fec2a8e195c1932f1865ed3ed6b4c7cb2b46977436248b2dc9a8a,952118457d7656b35e83c177103c2bc148f3f3253235b83d250304456be4a084
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b03-current.json,91be7f0f6e80375c3aed1c5f852fbba7d3c076d88f0e1b2452b46f75e22aa6f6,b7e7c15b275cbe20d2a24b2dbc355e4e8c61e89c2eff0c275fc5163963ccf93d,185c1e0496d20a92dffa6dc529b3f52f09af5d292a5b41263338b76e5b2fd798
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b04-baseline.json,904b8b3bcab9a496b6488d6fd5eeb77c5a94b7dd7ee1769d7a10d05612c7e35c,f0ce5bd5e1804d3bd0506916eee8d588c732222dfef3156b03069cadb63e68ef,a7615e6f7ad8ccb3228b609c6c69e6d5305793e53d4c729d6969d8a9055b9ef6
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b01-baseline.json,272c9d7064b839a8907f75b4e8a5af082a50dce4d69da5ba1501c8a63919f2f5,52b40da52dc445e24bb60e3af10d4fe08c17c562170156e0f94b90383e6b79b7,d0688f54e3d2c30ad189df584d42a02079465a1c435df99ed83c5a8bc7c804e2
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b02-current.json,34f2959d5d982da366bca2b7e3d61c2f26d3395d0c4a3ce1b9295519160ff7b5,06fb238a0256878611bffacde8fb6e600dc977c64c70e76a72ca13f2652878b2,8644006411534ba538a6bd9d76f09a73cfec7d4536c783df21e821111e207ece
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b03-current.json,e9bf22c5b7ce570bba0af26b643fdd77cefeff3e189882854de6ecf7c1af26f8,815674b0cad91fd6073a6b833c5264751cbf88ae82733b378cdaeefdc6d2bc61,7157dc3717dd04916bc164c9313cd106fcea9236bd4746eee33a3174cf1b2787
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b04-baseline.json,5ca87533ada3c689c603ef17ebb2025a1297e45e609456658b6d8226eec20a44,43c5a16487a55e9c5b8a383af953107162097e1340648032c2729f8d0ef718ed,f54a22ba8d6ff18ce88bef58fb264983a858ef1a4ef67da0b1dede793955ffa7
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b01-baseline.json,c87eb3a21b22787b2bea3e604899ac523d91d7f8b77fe830c63262fe6ede30eb,b46bd35ce92e888044d4d5e6018d1def57578bc171e1659f5226a0504ec4c87d,d0e14ca072b311b5f3a09c15585cd8c5fe2513868f71f8a44a3325d0367dbe89
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b02-current.json,6e6f905353dcd7f59faf56f570c7aadefafb2d360d63a3c1241e74b7610da9d4,9b96f3dcfecdad52fec7946486c6ca481d15cc718f55b950424bcff6bce6d680,434d2920c9ca1312ae45441c75f7c570bcceea49dd5bf02ccac8f6c3027dd21c
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b03-current.json,5aa194539fc162368dca41b44f6e80a33eda5af45ffbc883863d45aa42ba0c53,e923f43ecd1948706c669fa0d76d37412ba47d608f1a082a0c462187be04037c,0c3346cc9f09f3c12b72e783a810eab528b8f433b95bbd259f69937b72115dcc
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b04-baseline.json,a240d4490c960a787ced80121bfa41880989b6033a691d8790158ec112fcf60e,6adf96da1656b347251b97ba764c05b469c7be5f284e81d779e85483e15c5ca7,a85c34c2fbf85f3385f15e93af9baf3dba08266c6c9bc2dc813b046cc207c6fc
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b01-baseline.json,68ec064da128e2957bdbb8ade4ac4f5eed81f133241607408f315bf0db8dc39a,7e1053b06c1f67df915262021ac45cddf118957bfafcb9815e6266864280a843,687f6504f9c19904c05242f4cd9790d42cfed004b7cb4213306f4248bc294824
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b02-current.json,4695162abeaa7fd1493217d6c949e71fb2339b4091b5ec32e47b106639d3e315,4fd4ae8199d727b6a8f7db990550c250396336c37577dafbd83e2610bcc8a7c3,5a194fa5ddff982513fbe91b51f8bee1a8af1cf23615e8a89dc86567288fdc4a
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b03-current.json,2303d8777a448c8a86a0f8dc30ef13f129a0de5f0db046b4f31c08e265c9edef,47108681ffde1b287bac517f1c9c34a89137c09ae70e472756688afaf6c7cc5d,bf0fa244ece34b59ddd75bbaa7c45e5ff4bff3a286b9d84ffb8559a52119a9f9
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b04-baseline.json,e5b6568e446700a389c971dc4941c85110325c1c3e9177d08c6e9c862f7c9dd7,486959e54c70b524eaedb2ea211de8e627c93a524599f1b2645460edc4059c59,23a0abfcbe4fa0d2ad68294c17556241f19428ac40365db4a7e7dd200b222aab
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b01-baseline.json,1662a130bb70262d64a9380f9c7b27c541b90b6198a885bf27d21bfb2548c68d,ed4cc9b237cf2a1521be0e665f50a9a5c8a6b19dab5ddd604384acc4de417367,0b0bfdf6faf5f9e71177a22bacc30c36fb39eebf260dc9540a4515bab16f2414
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b02-current.json,a5db30505215400c5f9ac3bb7618d038147c4ef4b35ec3e1d8f31255ebf9cdc4,241600c901471558a970530eda6c3f97a7a6dc51f0a19c92f306fb6ae8f89718,fdddab3cc1f7d4f4091a9deeade048fff7e4b2c46544144af6b937624c5e5c11
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b03-current.json,c12d2774da83657630798f6cdc96c45b161ae78e7fd0fec69a4157cb60db75a6,cd6ac5f08339ce9ea715f3a3ffd8cf040500a52b34cd3cba688626bc9b16aa0a,b81ae318170331b53f9084675e593ac0bb32dbd16c0fa2ca2612856ac11981da
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b04-baseline.json,b5b7d49bc31f892adacb35380111fd28da4259a3717c956b553f4ac09a69e366,53f6458c93c8831db163e485ea85cd89f4a8c95bc33797fa0cca3ac556cd7b0d,698a856714307da3b62a553d874ff2f32d6cc2d5d14c2df9ee7c788ed1d4789b
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline.json,57a6093752f488d2eab0b2555878c7b75f0b9301026279b92bbe2e94684925a2,084f2129c97e53bf00455eb2a09437c59ac37f6c4de2223f69f82b433be8d8d6,3455a0d8ba5a7bfbc194ec930746d1f49d1fc75f71b79a62947726bf7eeb3b3f
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b02-current.json,bacfd6d390a9e8ea92a6e80b56971f3c34060a52881764775cd917910e05e46f,2163bd9076e9628c96545d835b20f841fbccb4a0e7c2a03b71a62e82e0fd2215,cccc5e71270c617a972943f00cecc502dd38a38b0d36be509dd4c4b4b4fde8b0
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b03-current.json,f5f3d84a98d157f71888ae99df100c212e6e7129b71eb715305aaa0344e3d209,a88f0688a16f111195926b801d860b53eb7aad481bc9ca4aa868a49367f6ded1,28dbbeb4c0a71d06c1a56830e79f159bfb1e1f65435e18209b058a2ecda0d744
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline.json,10bf2f05eafe33ac064d7f553d9fd84962be53d8465fce2010e3c48c9d46b6da,cb6f4f11f815a027df4bbd7eaf7f4a0d0a199d695e7981aa860cb0ac30b3977d,6b1aceee01497393aed19f230595069875893c4b6ea49d01215bb77def53c390
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline.json,f450bb046f0f065d03b48b5518caa478a5db9405c56c8e2ea36486d40b9c63bd,32791041f4e29d2d84c58bf7508c85da6775964142af8dfd18cc4846ac4bad88,4dfc872d25aa3a99f6e89db4ebfb45816c0729cc01e052d449955995055348be
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current.json,6f666bb30a5040d591c21f296d40df3f41f456fdf56df00b1302526cdd8fdd5c,f99dd6df6ee812180a3867ca75f5386b03a417b63b36394bbbf17e94adbf6e6f,949bccddddd2fe89463178333d99ab61e52d7a558a824cb6e7ae0386110195e0
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current.json,33f1142823afb9d44a600400909e163e59e85413b8a42749d338ec035c8c07f7,2bccb25c3719c26a73e749d4604f9bca6a32eaf615379f95eeea614059f20963,833cc846b04061ac1a0835ee17bef12eba141c0517f4ac67e7a315e35039dc76
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/fresh-span-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline.json,f651010cdb1c358a4f7dbc1715c8b75cdac815a5196f00be45ab74cdfa121b82,c56d445a28c9dc972c8804da8905a812356cd8f36eb83695e318ed9f8d06163e,5f8fb89f46b677e9e921d559611e63241194cf8871baf4426492549f9ab24398


cpu-quad-abba (cpu) fixed A/B campaign
--------------------------------------

Whole-campaign verdict: **regression**.
All 64 artifacts/832 candidate/case outcomes are retained.

.. csv-table:: Every native and framework strict result
   :class: benchmark-results
   :header-rows: 1

   Candidate,Role,Rows,Within 5%,Inconclusive,Regression,Comparison failed
   native,native,64,15,45,4,0
   numpy-out,framework control,64,16,41,7,0
   torch-eager-out,framework control,64,12,47,5,0
   native-blas,native,16,5,9,2,0

All candidate rows
~~~~~~~~~~~~~~~~~~

A/B and current/reference ratios below one indicate a lower measured numerator.
Ranges bracket the two process medians and are not confidence intervals.
Saved strict intervals compare current against A/B baseline; they do not give
confidence for the separate current/native-to-framework comparison.

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.042038,0.038106–0.050751,0.049491,0.048831–0.051365,1.17728,0.8556–1.3839,inconclusive,49/51/0,0.607749
   dense-64,numpy-out,0.216004,0.202672–0.281401,0.285015,0.273992–0.295851,1.31949,0.9056–1.4646,inconclusive,49/51/0,0.61272
   dense-64,torch-eager-out,0.349208,0.330628–0.447311,0.447082,0.427642–0.456652,1.28028,0.8212–1.4385,inconclusive,49/51/0,0.635707
   dense-256,native,0.484480,0.477195–0.504274,0.498094,0.478643–0.508410,1.02810,0.9102–1.1178,inconclusive,67/70/1,0.971216
   dense-256,numpy-out,0.770166,0.758203–0.802014,0.815955,0.713938–0.909397,1.05945,0.8368–1.2363,inconclusive,67/70/1,0.969964
   dense-256,torch-eager-out,0.953616,0.897260–1.041708,1.156014,0.995309–1.283745,1.21224,0.9337–1.5040,inconclusive,67/70/1,0.972357
   dense-1024,native,10.457527,10.387518–10.631519,9.524603,9.346824–9.629594,0.91079,0.8624–0.9462,within 5%,70/73/1,0.928014
   dense-1024,numpy-out,11.039568,10.871386–11.102237,10.950899,10.752448–11.021052,0.99197,0.9367–1.0348,within 5%,70/73/1,0.928014
   dense-1024,torch-eager-out,12.570985,12.399534–12.684242,12.355033,12.275438–12.510957,0.98282,0.9408–1.0310,within 5%,70/73/1,0.928014
   dense-2048,native,119.740431,119.740431–119.972992,107.810716,106.164976–110.135357,0.90037,0.8584–0.9434,within 5%,70/73/1,0.871762
   dense-2048,numpy-out,100.697592,100.697592–101.255465,100.093237,97.741563–109.928444,0.99400,0.9529–1.1086,inconclusive,70/73/1,0.871762
   dense-2048,torch-eager-out,104.664960,102.070537–105.384689,104.753747,101.140635–112.854564,1.00085,0.9356–1.1143,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-api-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.043701,0.039690–0.050471,0.056100,0.055732–0.056546,1.28371,1.0628–1.4743,regression,49/51/0,0.607749
   dense-64,numpy-out,0.227460,0.210323–0.245940,0.282634,0.276720–0.289480,1.24257,1.1026–1.4001,regression,49/51/0,0.61272
   dense-64,torch-eager-out,0.356376,0.309134–0.368166,0.474930,0.467484–0.487223,1.33267,1.2425–1.5929,regression,49/51/0,0.635707
   dense-256,native,0.468238,0.439736–0.499845,0.543973,0.514590–0.577717,1.16175,1.0068–1.3218,inconclusive,67/70/1,0.971216
   dense-256,numpy-out,0.653466,0.566266–0.744281,0.795365,0.794628–0.802897,1.21715,1.0358–1.4971,inconclusive,67/70/1,0.969964
   dense-256,torch-eager-out,0.823513,0.733635–1.035921,1.139810,1.124154–1.155962,1.38408,1.0387–1.6379,inconclusive,67/70/1,0.972357
   dense-1024,native,10.071204,10.044321–10.112115,10.085772,9.335278–12.462783,1.00145,0.9015–1.2896,inconclusive,70/73/1,0.928014
   dense-1024,numpy-out,10.558459,10.410975–10.758313,11.141518,10.589441–13.842702,1.05522,0.9630–1.4850,inconclusive,70/73/1,0.928014
   dense-1024,torch-eager-out,11.759586,11.635023–11.910004,12.897099,11.769146–14.003522,1.09673,0.9551–1.2706,inconclusive,70/73/1,0.928014
   dense-2048,native,121.653656,120.672128–125.237603,126.676158,118.025280–132.960107,1.04129,0.9309–1.1342,inconclusive,70/73/1,0.871762
   dense-2048,numpy-out,100.415560,99.108128–101.349485,124.629762,115.835740–140.956800,1.24114,1.0831–1.4541,regression,70/73/1,0.871762
   dense-2048,torch-eager-out,104.527313,104.310374–104.938545,124.295817,115.321748–133.386662,1.18912,1.0496–1.3138,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.045463,0.045366–0.045784,0.037671,0.031343–0.044030,0.82861,0.6312–1.1517,inconclusive,49/51/0,0.607749
   dense-64,numpy-out,0.327890,0.326711–0.340554,0.247629,0.240754–0.255865,0.75522,0.6312–0.9312,within 5%,49/51/0,0.61272
   dense-64,torch-eager-out,0.498279,0.467086–0.597500,0.394910,0.390265–0.403298,0.79255,0.5890–0.9342,within 5%,49/51/0,0.635707
   dense-256,native,0.528154,0.523107–0.567871,0.472382,0.454319–0.490812,0.89440,0.7881–0.9677,within 5%,67/70/1,0.971216
   dense-256,numpy-out,0.922941,0.907277–1.092906,0.748051,0.697568–0.840042,0.81051,0.6234–0.9491,within 5%,67/70/1,0.969964
   dense-256,torch-eager-out,1.294853,1.251946–1.346327,0.992642,0.938895–1.006969,0.76661,0.6644–0.9430,within 5%,67/70/1,0.972357
   dense-1024,native,12.838892,12.178964–13.983548,12.077255,12.030092–12.260506,0.94068,0.8419–1.0322,within 5%,70/73/1,0.928014
   dense-1024,numpy-out,12.638963,12.230132–13.341102,13.211063,12.988356–13.958096,1.04526,0.9599–1.1665,inconclusive,70/73/1,0.928014
   dense-1024,torch-eager-out,14.139568,13.556372–14.994195,14.601103,14.035477–15.166621,1.03264,0.9107–1.1328,inconclusive,70/73/1,0.928014
   dense-2048,native,127.292558,125.456946–131.899644,136.774067,136.708186–137.852006,1.07449,1.0063–1.1304,inconclusive,70/73/1,0.871762
   dense-2048,numpy-out,93.683631,93.655204–93.760593,112.315505,104.391305–123.373615,1.19888,1.0800–1.3547,regression,70/73/1,0.871762
   dense-2048,torch-eager-out,106.899927,98.194232–117.437431,125.381565,121.470311–129.998575,1.17289,1.0313–1.3421,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.044189,0.044015–0.044189,0.043073,0.042543–0.044735,0.97474,0.8533–1.2089,inconclusive,49/51/0,0.607749
   dense-64,numpy-out,0.272817,0.250077–0.290573,0.288411,0.269136–0.333082,1.05716,0.9070–1.3412,inconclusive,49/51/0,0.61272
   dense-64,torch-eager-out,0.452307,0.417258–0.477467,0.491169,0.472341–0.548358,1.08592,0.9773–1.3500,inconclusive,49/51/0,0.635707
   dense-256,native,0.489382,0.464647–0.538067,0.488546,0.466158–0.507574,0.99829,0.8481–1.0967,inconclusive,67/70/1,0.971216
   dense-256,numpy-out,0.779444,0.764717–0.790898,0.881588,0.665701–0.936291,1.13105,0.8010–1.2492,inconclusive,67/70/1,0.969964
   dense-256,torch-eager-out,0.962159,0.916911–1.047567,1.051166,0.911379–1.144756,1.09251,0.7830–1.2693,inconclusive,67/70/1,0.972357
   dense-1024,native,12.382461,11.866444–13.120945,11.921061,11.712153–12.117420,0.96274,0.8793–1.0455,within 5%,70/73/1,0.928014
   dense-1024,numpy-out,12.204106,12.077063–12.351791,12.829164,12.662799–13.057472,1.05122,0.9583–1.1238,inconclusive,70/73/1,0.928014
   dense-1024,torch-eager-out,13.587476,13.060068–13.878958,14.239138,14.059055–14.829439,1.04796,0.9832–1.1491,inconclusive,70/73/1,0.928014
   dense-2048,native,145.815435,136.375575–156.510558,127.130635,125.981415–130.231523,0.87186,0.7525–0.9649,within 5%,70/73/1,0.871762
   dense-2048,numpy-out,101.841864,96.127741–114.278188,94.764996,94.632415–96.747348,0.93051,0.7903–1.0170,within 5%,70/73/1,0.871762
   dense-2048,torch-eager-out,125.922307,123.735653–132.946276,113.048113,108.448492–118.731271,0.89776,0.8060–0.9825,within 5%,70/73/1,0.871762

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.015482,0.014863–0.017413,0.018815,0.017754–0.019886,1.21528,0.9796–1.5646,inconclusive,5/7/0,0.0796191
   dense-64,numpy-out,0.042320,0.041096–0.044341,0.060068,0.050837–0.063053,1.41936,1.0602–1.5659,regression,5/7/0,0.0796191
   dense-64,torch-eager-out,0.073302,0.070135–0.088708,0.113624,0.099184–0.118639,1.55008,1.0727–1.7043,regression,5/7/0,0.0796191
   dense-256,native,0.056544,0.047074–0.073399,0.061164,0.050009–0.064848,1.08170,0.6681–1.3943,inconclusive,4/6/0,0.209854
   dense-256,numpy-out,0.076975,0.062198–0.096728,0.097557,0.084087–0.107437,1.26739,0.8499–1.7395,inconclusive,4/6/0,0.209854
   dense-256,torch-eager-out,0.136835,0.096295–0.186451,0.169849,0.120029–0.180671,1.24127,0.6211–1.8953,inconclusive,4/6/0,0.209854
   dense-1024,native,0.945860,0.934688–0.971317,0.930875,0.923060–0.952358,0.98416,0.9270–1.0346,within 5%,4/6/0,0.013684
   dense-1024,numpy-out,1.073007,1.067175–1.073007,1.072718,1.072718–1.080943,0.99973,0.9443–1.0792,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,1.218091,1.191566–1.252444,1.293532,1.282047–1.298506,1.06193,0.9908–1.1173,inconclusive,4/6/0,0.013684
   dense-2048,native,7.578267,7.475696–7.671149,6.310298,5.954938–7.206009,0.83268,0.7238–1.1381,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,6.197669,6.147465–6.220285,6.030121,5.595532–6.529207,0.97297,0.7832–1.1295,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,6.268376,6.021684–6.792315,6.551185,6.018681–7.112641,1.04512,0.7884–1.2038,inconclusive,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.018055,0.016443–0.020211,0.026266,0.023141–0.031273,1.45478,1.0569–1.9263,regression,5/7/0,0.0796191
   dense-64,numpy-out,0.044401,0.043075–0.049570,0.069025,0.067081–0.069351,1.55458,1.0411–1.7027,inconclusive,5/7/0,0.0796191
   dense-64,torch-eager-out,0.077837,0.073405–0.089402,0.133079,0.131712–0.133870,1.70971,1.3192–1.8631,regression,5/7/0,0.0796191
   dense-256,native,0.071806,0.070410–0.072629,0.068281,0.063453–0.070042,0.95092,0.8520–1.0385,within 5%,4/6/0,0.209854
   dense-256,numpy-out,0.105774,0.103247–0.114004,0.102357,0.102357–0.103022,0.96770,0.8428–1.0588,inconclusive,4/6/0,0.209854
   dense-256,torch-eager-out,0.192489,0.174966–0.218127,0.178041,0.178041–0.179604,0.92494,0.7600–1.0596,inconclusive,4/6/0,0.209854
   dense-1024,native,0.988382,0.893336–1.130740,0.970795,0.908275–1.023026,0.98221,0.7956–1.1509,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.054271,0.967160–1.215243,1.108482,1.095954–1.132640,1.05142,0.8548–1.1794,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,1.241844,1.178023–1.516934,1.316274,1.234677–1.401963,1.05993,0.8121–1.1986,inconclusive,4/6/0,0.013684
   dense-2048,native,7.820111,6.628355–8.300432,6.418099,6.185906–7.941310,0.82072,0.7322–1.2210,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,6.890287,6.111979–7.544366,5.811723,5.578374–6.531288,0.84347,0.7145–1.1303,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,7.229095,6.420790–7.605319,6.444130,6.038529–6.731510,0.89142,0.7491–1.0884,inconclusive,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.008655,0.007138–0.009232,0.009581,0.009089–0.009911,1.10711,0.9807–1.4163,inconclusive,5/7/0,0.0796191
   dense-64,numpy-out,0.044408,0.029879–0.061859,0.049668,0.048962–0.051452,1.11845,0.7714–1.7424,inconclusive,5/7/0,0.0796191
   dense-64,torch-eager-out,0.080779,0.046984–0.112021,0.093152,0.091557–0.094792,1.15318,0.7770–2.0244,inconclusive,5/7/0,0.0796191
   dense-256,native,0.053611,0.047677–0.069941,0.050781,0.044278–0.057374,0.94720,0.6241–1.2077,inconclusive,4/6/0,0.209854
   dense-256,numpy-out,0.086014,0.076028–0.118541,0.087477,0.072407–0.092425,1.01700,0.6140–1.2599,inconclusive,4/6/0,0.209854
   dense-256,torch-eager-out,0.128025,0.115958–0.153790,0.128892,0.091393–0.142948,1.00678,0.5765–1.2676,inconclusive,4/6/0,0.209854
   dense-1024,native,1.540671,1.540671–1.543071,1.506552,1.481142–1.566455,0.97785,0.8061–1.1950,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.266905,1.265508–1.295896,1.369044,1.338928–1.391082,1.08062,0.9194–1.1694,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,1.625737,1.355144–1.672821,1.602343,1.563997–1.630384,0.98561,0.8224–1.2700,inconclusive,4/6/0,0.013684
   dense-2048,native,9.262180,9.130049–9.298321,9.171795,8.272897–9.371507,0.99024,0.8407–1.1187,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,7.071290,6.493213–7.470228,7.947388,7.598876–8.372616,1.12390,0.8922–1.3583,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,7.970667,7.752719–8.335994,7.689222,7.677315–7.693801,0.96469,0.8748–1.0397,within 5%,3/5/0,0.836412

.. csv-table:: cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.008649,0.008121–0.009994,0.009292,0.009088–0.009628,1.07434,0.8695–1.2092,inconclusive,5/7/0,0.0796191
   dense-64,numpy-out,0.031178,0.030160–0.035552,0.047380,0.046147–0.048661,1.51970,1.1342–1.6384,regression,5/7/0,0.0796191
   dense-64,torch-eager-out,0.052431,0.049261–0.059672,0.083480,0.073039–0.090913,1.59220,1.0445–1.8486,inconclusive,5/7/0,0.0796191
   dense-256,native,0.045408,0.043624–0.048582,0.059101,0.057912–0.059875,1.30156,1.1131–1.4361,regression,4/6/0,0.209854
   dense-256,numpy-out,0.061679,0.060687–0.061847,0.093931,0.088812–0.098358,1.52289,1.2777–1.6840,regression,4/6/0,0.209854
   dense-256,torch-eager-out,0.089441,0.072542–0.118157,0.135785,0.118618–0.140973,1.51814,0.9764–1.9804,inconclusive,4/6/0,0.209854
   dense-1024,native,1.340797,1.242754–1.457267,1.475731,1.411318–1.494826,1.10064,0.8743–1.2591,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.303882,1.199370–1.381997,1.447552,1.363893–1.509780,1.11019,0.8695–1.3175,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,1.429414,1.357732–1.525151,1.608480,1.608480–1.628892,1.12527,0.8984–1.3639,inconclusive,4/6/0,0.013684
   dense-2048,native,9.887118,9.368645–11.046350,8.704638,8.451689–8.994224,0.88040,0.7250–1.0989,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,6.467257,6.123685–7.784951,6.423397,6.288486–6.519978,0.99322,0.7681–1.0990,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,7.605853,7.340197–8.581200,8.045276,7.441446–8.959401,1.05777,0.8298–1.3102,inconclusive,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.052203,0.050818–0.058204,0.048942,0.045938–0.050340,0.93753,0.7330–1.0512,inconclusive,49/51/0,0.607749
   dense-64,native-blas,0.045922,0.045256–0.055877,0.042667,0.040752–0.046194,0.92911,0.6641–1.0448,within 5%,49/51/0,0.612054
   dense-64,numpy-out,0.301914,0.299313–0.331434,0.278184,0.259766–0.290822,0.92140,0.7427–1.0029,within 5%,49/51/0,0.61272
   dense-64,torch-eager-out,0.513254,0.478587–0.584865,0.449313,0.410478–0.484636,0.87542,0.6716–1.0658,inconclusive,49/51/0,0.635707
   dense-256,native,0.485963,0.474413–0.493465,0.481976,0.468123–0.488113,0.99180,0.9390–1.0736,inconclusive,67/70/1,0.971216
   dense-256,native-blas,0.474730,0.470595–0.476241,0.335533,0.328861–0.346406,0.70679,0.6620–0.7755,within 5%,67/70/1,0.979189
   dense-256,numpy-out,0.847515,0.847515–0.849541,0.707725,0.698615–0.721323,0.83506,0.7764–0.8846,within 5%,67/70/1,0.969964
   dense-256,torch-eager-out,1.238738,1.177280–1.268535,0.991865,0.945593–1.002665,0.80071,0.7274–0.8891,within 5%,67/70/1,0.972357
   dense-1024,native,9.789756,9.729269–9.966482,9.172807,9.167500–9.216453,0.93698,0.8885–0.9825,within 5%,70/73/1,0.928014
   dense-1024,native-blas,9.323401,8.800693–9.377906,8.974327,8.730684–9.119744,0.96256,0.9130–1.0547,inconclusive,70/73/1,0.928014
   dense-1024,numpy-out,10.328694,10.263386–10.421463,10.348399,10.335130–10.373466,1.00191,0.9504–1.0614,inconclusive,70/73/1,0.928014
   dense-1024,torch-eager-out,12.118351,11.943413–12.273738,11.932993,11.814045–12.045082,0.98470,0.9409–1.0299,within 5%,70/73/1,0.928014
   dense-2048,native,110.671547,109.987273–111.026158,123.736721,116.248153–125.338402,1.11805,1.0145–1.1688,inconclusive,70/73/1,0.871762
   dense-2048,native-blas,99.501520,97.522853–101.044513,103.599949,98.018648–111.902693,1.04119,0.9477–1.1643,inconclusive,70/73/1,0.871762
   dense-2048,numpy-out,106.337981,105.077995–107.749245,112.036158,103.604195–117.647297,1.05359,0.9377–1.1743,inconclusive,70/73/1,0.871762
   dense-2048,torch-eager-out,107.797856,105.469099–111.969113,117.203422,116.471084–119.352217,1.08725,0.9841–1.1644,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.042750,0.042500–0.044368,0.039211,0.036607–0.040938,0.91722,0.7414–1.0045,within 5%,49/51/0,0.607749
   dense-64,native-blas,0.035370,0.031116–0.041829,0.035631,0.030812–0.037082,1.00735,0.7080–1.4083,inconclusive,49/51/0,0.612054
   dense-64,numpy-out,0.315423,0.309986–0.358909,0.289305,0.287069–0.295054,0.91720,0.7653–0.9933,within 5%,49/51/0,0.61272
   dense-64,torch-eager-out,0.535139,0.514803–0.588260,0.479405,0.475867–0.485261,0.89585,0.7372–0.9856,within 5%,49/51/0,0.635707
   dense-256,native,0.515799,0.515799–0.516112,0.469185,0.469185–0.476060,0.90963,0.8662–0.9806,within 5%,67/70/1,0.971216
   dense-256,native-blas,0.493503,0.487303–0.499342,0.472686,0.467697–0.482542,0.95782,0.9162–1.0132,within 5%,67/70/1,0.979189
   dense-256,numpy-out,0.804721,0.797051–0.806351,0.689586,0.683435–0.706800,0.85693,0.8199–0.9606,within 5%,67/70/1,0.969964
   dense-256,torch-eager-out,1.160993,1.155217–1.173793,0.968673,0.966001–0.982948,0.83435,0.7804–0.9033,within 5%,67/70/1,0.972357
   dense-1024,native,13.499565,12.709122–14.318356,12.587576,12.572033–12.631638,0.93244,0.8608–1.0176,within 5%,70/73/1,0.928014
   dense-1024,native-blas,12.561340,12.239774–14.190210,11.810638,11.726218–12.150629,0.94024,0.8006–1.0021,within 5%,70/73/1,0.928014
   dense-1024,numpy-out,14.059589,13.411640–15.053088,13.707803,13.653808–13.852920,0.97498,0.8633–1.0539,inconclusive,70/73/1,0.928014
   dense-1024,torch-eager-out,15.232900,14.280247–16.860312,14.936497,14.800920–15.031305,0.98054,0.8434–1.0609,inconclusive,70/73/1,0.928014
   dense-2048,native,143.625237,130.627575–155.832783,133.696443,133.594631–133.755485,0.93087,0.8364–1.0575,inconclusive,70/73/1,0.871762
   dense-2048,native-blas,108.001609,104.311105–109.439793,102.202950,101.193609–102.789967,0.94631,0.8705–1.0045,within 5%,70/73/1,0.871762
   dense-2048,numpy-out,109.609177,103.320093–112.790036,104.550442,104.293179–106.001856,0.95385,0.9019–1.0343,within 5%,70/73/1,0.871762
   dense-2048,torch-eager-out,107.946269,100.618544–116.453821,112.504743,112.278622–113.357761,1.04223,0.9334–1.1629,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.014847,0.013881–0.015772,0.014541,0.014461–0.016442,0.97939,0.7870–1.2103,inconclusive,5/7/0,0.0796191
   dense-64,native-blas,0.017086,0.016679–0.017636,0.017209,0.016875–0.017470,1.00717,0.7632–1.2407,inconclusive,5/7/0,0.0796191
   dense-64,numpy-out,0.060934,0.056723–0.067666,0.056890,0.053209–0.061480,0.93364,0.6998–1.1937,inconclusive,5/7/0,0.0796191
   dense-64,torch-eager-out,0.112548,0.108313–0.118320,0.110706,0.108995–0.110908,0.98363,0.7180–1.1367,inconclusive,5/7/0,0.0796191
   dense-256,native,0.057264,0.052848–0.059606,0.059688,0.052565–0.062391,1.04234,0.8627–1.2402,inconclusive,4/6/0,0.209854
   dense-256,native-blas,0.055284,0.047932–0.062789,0.057804,0.047782–0.061405,1.04557,0.7466–1.3129,inconclusive,4/6/0,0.209854
   dense-256,numpy-out,0.097103,0.096566–0.098763,0.100959,0.089750–0.108750,1.03971,0.8777–1.1834,inconclusive,4/6/0,0.209854
   dense-256,torch-eager-out,0.161055,0.153261–0.167513,0.183249,0.154486–0.201778,1.13780,0.8702–1.3616,inconclusive,4/6/0,0.209854
   dense-1024,native,1.067340,1.026217–1.081611,0.936248,0.917486–0.955413,0.87718,0.7897–0.9557,within 5%,4/6/0,0.013684
   dense-1024,native-blas,0.978814,0.945925–1.008271,0.942434,0.922166–0.958318,0.96283,0.8960–1.1212,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.161325,1.137908–1.229516,1.147377,1.127992–1.215865,0.98799,0.8781–1.0963,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,1.428410,1.337514–1.668497,1.439147,1.327090–1.599923,1.00752,0.7814–1.2015,inconclusive,4/6/0,0.013684
   dense-2048,native,7.299926,6.774863–7.713626,7.031703,6.988801–7.241459,0.96326,0.7658–1.2149,inconclusive,3/5/0,0.836412
   dense-2048,native-blas,6.587444,6.587444–6.788151,5.788287,5.470110–6.665693,0.87868,0.7032–1.2306,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,6.113086,5.953421–6.484996,6.337937,6.109414–6.454701,1.03678,0.8394–1.1293,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,6.767993,6.690172–7.052211,7.209759,6.531131–7.397428,1.06527,0.8234–1.1824,inconclusive,3/5/0,0.836412

.. csv-table:: cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.007047,0.006861–0.007057,0.008824,0.008393–0.009202,1.25208,1.0984–1.4183,regression,5/7/0,0.0796191
   dense-64,native-blas,0.006944,0.006944–0.007032,0.008604,0.008192–0.009921,1.23907,1.0791–1.5032,regression,5/7/0,0.0796191
   dense-64,numpy-out,0.032341,0.032283–0.034788,0.050469,0.044636–0.056384,1.56055,1.1280–1.7997,regression,5/7/0,0.0796191
   dense-64,torch-eager-out,0.051829,0.050334–0.054172,0.090479,0.088161–0.101347,1.74572,1.4553–2.0363,regression,5/7/0,0.0796191
   dense-256,native,0.057734,0.055585–0.062919,0.058722,0.057654–0.059852,1.01712,0.8787–1.2278,inconclusive,4/6/0,0.209854
   dense-256,native-blas,0.043659,0.039216–0.044876,0.056848,0.055963–0.062053,1.30209,1.0880–1.6689,regression,4/6/0,0.209854
   dense-256,numpy-out,0.095132,0.085542–0.105620,0.092943,0.082989–0.095230,0.97699,0.7734–1.1202,inconclusive,4/6/0,0.209854
   dense-256,torch-eager-out,0.139974,0.120566–0.160095,0.145616,0.142889–0.154712,1.04031,0.8298–1.3136,inconclusive,4/6/0,0.209854
   dense-1024,native,1.889512,1.697399–2.029024,1.728854,1.692831–1.743683,0.91497,0.7930–1.0839,inconclusive,4/6/0,0.013684
   dense-1024,native-blas,1.581378,1.448908–1.795345,1.516158,1.488968–1.567935,0.95876,0.7948–1.1165,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.683308,1.555769–1.842799,1.693000,1.531552–1.859689,1.00576,0.8017–1.2105,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,2.010719,1.941064–2.286630,2.210337,2.010807–2.497941,1.09928,0.8737–1.3293,inconclusive,4/6/0,0.013684
   dense-2048,native,10.465901,9.743443–11.323823,11.146144,9.757210–12.855193,1.06500,0.7962–1.4271,inconclusive,3/5/0,0.836412
   dense-2048,native-blas,7.867612,7.731559–7.924334,9.403459,8.300538–10.828259,1.19521,0.9958–1.4636,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,7.502494,7.222422–7.526651,9.116241,8.475718–9.500371,1.21509,0.9793–1.4016,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,10.291389,9.514822–11.283906,9.008320,8.155505–9.867854,0.87533,0.7053–1.1179,inconclusive,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.069964,0.068458–0.070649,0.069852,0.069852–0.069875,0.99841,0.9319–1.1218,inconclusive,49/51/0,0.607749
   dense-64,numpy-out,0.208315,0.204798–0.212701,0.215673,0.215558–0.216716,1.03532,0.9766–1.0821,inconclusive,49/51/0,0.61272
   dense-64,torch-eager-out,0.333528,0.333102–0.333589,0.332829,0.326782–0.343045,0.99790,0.9624–1.0917,inconclusive,49/51/0,0.635707
   dense-256,native,0.358784,0.338190–0.364019,0.373569,0.367424–0.381783,1.04121,0.9802–1.1643,inconclusive,67/70/1,0.993806
   dense-256,numpy-out,0.847326,0.794802–0.869617,0.579588,0.537435–0.632835,0.68402,0.6108–0.8021,within 5%,67/70/1,0.969964
   dense-256,torch-eager-out,1.005043,0.885763–1.060450,0.791250,0.742814–0.841740,0.78728,0.6957–0.9616,within 5%,67/70/1,0.972357
   dense-1024,native,12.250766,12.162724–12.328628,10.817203,10.478015–11.305290,0.88298,0.7617–1.0325,within 5%,70/73/1,0.928014
   dense-1024,numpy-out,13.088453,12.613152–14.200572,11.908826,11.781162–11.908826,0.90987,0.7850–1.0171,within 5%,70/73/1,0.928014
   dense-1024,torch-eager-out,16.836122,16.436981–16.963508,13.578562,13.508268–14.270492,0.80651,0.6936–0.9190,within 5%,70/73/1,0.928014
   dense-2048,native,73.455928,71.248919–73.841260,68.010165,66.082171–68.345557,0.92586,0.8208–1.1301,inconclusive,70/73/1,0.871762
   dense-2048,numpy-out,69.592870,69.375901–70.299243,66.694926,65.495268–68.728290,0.95836,0.9098–1.0138,within 5%,70/73/1,0.871762
   dense-2048,torch-eager-out,68.273240,67.451418–73.251066,68.532064,62.516466–71.412288,1.00379,0.7707–1.1581,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.052873,0.052781–0.053293,0.052191,0.051741–0.053314,0.98710,0.9367–1.0622,inconclusive,49/51/0,0.607749
   dense-64,numpy-out,0.203425,0.197418–0.204746,0.198362,0.196043–0.200947,0.97511,0.9454–1.0437,within 5%,49/51/0,0.61272
   dense-64,torch-eager-out,0.296554,0.295398–0.298700,0.295524,0.291297–0.297707,0.99653,0.9312–1.0604,inconclusive,49/51/0,0.635707
   dense-256,native,0.352040,0.324651–0.378273,0.387538,0.370636–0.418844,1.10084,0.9738–1.2902,inconclusive,67/70/1,0.993806
   dense-256,numpy-out,0.670878,0.645710–0.679493,0.752991,0.666762–0.858802,1.12240,0.9669–1.3389,inconclusive,67/70/1,0.969964
   dense-256,torch-eager-out,0.805469,0.783343–0.812639,0.932033,0.874379–1.010854,1.15713,1.0520–1.3257,regression,67/70/1,0.972357
   dense-1024,native,14.235617,13.997974–15.551402,14.656363,13.866990–15.530318,1.02956,0.8691–1.1396,inconclusive,70/73/1,0.928014
   dense-1024,numpy-out,13.915693,13.149484–14.206482,12.655776,12.444251–13.444104,0.90946,0.8305–1.2358,inconclusive,70/73/1,0.928014
   dense-1024,torch-eager-out,16.948152,15.352688–17.226251,19.039321,17.513781–20.755053,1.12339,0.9750–1.3707,inconclusive,70/73/1,0.928014
   dense-2048,native,88.003933,85.039827–88.974012,104.817928,93.166456–121.083851,1.19106,0.9472–1.5189,inconclusive,70/73/1,0.871762
   dense-2048,numpy-out,69.109776,67.298872–71.240920,71.412644,66.189376–74.469460,1.03332,0.9079–1.1486,inconclusive,70/73/1,0.871762
   dense-2048,torch-eager-out,89.510291,87.756258–93.114704,82.778267,73.905640–88.191157,0.92479,0.7699–1.2826,inconclusive,70/73/1,0.871762

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.020064,0.019838–0.020608,0.019353,0.018535–0.023335,0.96459,0.8891–1.2299,inconclusive,5/7/0,0.0796191
   dense-64,numpy-out,0.042443,0.042125–0.042527,0.043236,0.041378–0.051187,1.01871,0.9457–1.2786,inconclusive,5/7/0,0.0796191
   dense-64,torch-eager-out,0.072801,0.071981–0.073389,0.072694,0.069410–0.093528,0.99854,0.9314–1.3434,inconclusive,5/7/0,0.0796191
   dense-256,native,0.050425,0.045469–0.055061,0.059419,0.044521–0.084257,1.17835,0.7904–1.9746,inconclusive,4/6/0,0.209854
   dense-256,numpy-out,0.084556,0.071368–0.098680,0.067947,0.065181–0.071235,0.80357,0.6418–1.0113,within 5%,4/6/0,0.209854
   dense-256,torch-eager-out,0.143461,0.110938–0.157150,0.146321,0.109553–0.201478,1.01994,0.6818–1.8696,inconclusive,4/6/0,0.209854
   dense-1024,native,1.028299,0.599132–1.085761,0.979938,0.970577–1.021329,0.95297,0.8683–1.7514,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.237694,1.132184–2.147800,1.041165,0.996602–1.112097,0.84121,0.4559–1.0221,within 5%,4/6/0,0.013684
   dense-1024,torch-eager-out,1.242117,1.042675–1.277740,1.239275,1.194782–1.257710,0.99771,0.8720–1.3148,inconclusive,4/6/0,0.013684
   dense-2048,native,9.441247,8.869248–10.619672,8.575426,7.893645–9.268961,0.90829,0.7041–1.1720,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,7.223762,6.884082–8.432111,8.142355,7.895445–8.336089,1.12716,0.8448–1.4780,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,9.626836,9.100214–10.111403,9.197698,8.896773–9.510724,0.95542,0.7923–1.2398,inconclusive,3/5/0,0.836412

.. csv-table:: cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,native,0.011095,0.011041–0.011154,0.010927,0.010731–0.011123,0.98490,0.9572–1.0315,within 5%,5/7/0,0.0796191
   dense-64,numpy-out,0.030765,0.030210–0.031339,0.029743,0.029639–0.029792,0.96678,0.9048–1.0010,within 5%,5/7/0,0.0796191
   dense-64,torch-eager-out,0.048301,0.048056–0.048601,0.046539,0.046490–0.046539,0.96353,0.9362–0.9873,within 5%,5/7/0,0.0796191
   dense-256,native,0.033208,0.030194–0.044512,0.034356,0.032236–0.037651,1.03457,0.7090–1.2725,inconclusive,4/6/0,0.209854
   dense-256,numpy-out,0.065110,0.049401–0.093093,0.054981,0.050080–0.060639,0.84442,0.5343–1.2423,inconclusive,4/6/0,0.209854
   dense-256,torch-eager-out,0.082896,0.061449–0.118122,0.068386,0.067404–0.068710,0.82495,0.5532–1.1589,inconclusive,4/6/0,0.209854
   dense-1024,native,0.959715,0.942902–0.993173,1.441454,1.049789–1.690936,1.50196,0.4872–2.0268,inconclusive,4/6/0,0.013684
   dense-1024,numpy-out,1.839719,1.825372–1.839719,1.922732,1.590231–2.030394,1.04512,0.7251–1.6261,inconclusive,4/6/0,0.013684
   dense-1024,torch-eager-out,1.399378,1.300302–1.592701,1.725651,1.336782–2.351413,1.23316,0.4319–2.0907,inconclusive,4/6/0,0.013684
   dense-2048,native,12.106832,11.156087–12.522657,12.848515,10.737441–13.488679,1.06126,0.7930–1.3025,inconclusive,3/5/0,0.836412
   dense-2048,numpy-out,9.912330,9.806442–9.990468,9.816973,9.095509–9.816973,0.99038,0.7649–1.1819,inconclusive,3/5/0,0.836412
   dense-2048,torch-eager-out,12.684347,10.877647–13.940584,13.091261,12.016633–14.441135,1.03208,0.8367–1.3580,inconclusive,3/5/0,0.836412

Current-source framework comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are descriptive ratios of current pooled-round medians from the same
matched configuration, not strict comparisons across frameworks or scopes.
Every higher current-native median remains listed below.

.. csv-table:: All current matched scopes
   :class: benchmark-results
   :header-rows: 1

   Configuration,Native,Reference,Lower/higher/equal,Current ratio range,Same known I/M/R
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,native,numpy-out,3/1/0,0.1736–1.0771,True
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,native,torch-eager-out,3/1/0,0.1107–1.0292,True
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,native,numpy-out,3/1/0,0.1985–1.0164,True
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,native,torch-eager-out,3/1/0,0.1181–1.0192,True
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,native,numpy-out,3/1/0,0.1521–1.2178,True
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,native,torch-eager-out,3/1/0,0.0954–1.0909,True
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,native,numpy-out,3/1/0,0.1493–1.3415,True
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,native,torch-eager-out,3/1/0,0.0877–1.1246,True
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,native,numpy-out,3/1/0,0.3132–1.0465,True
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,4/0/0,0.1656–0.9632,True
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,native,numpy-out,3/1/0,0.3805–1.1043,True
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,native,torch-eager-out,4/0/0,0.1974–0.9960,True
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,numpy-out,2/2/0,0.1929–1.1541,True
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,3/1/0,0.1029–1.1928,True
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native,numpy-out,2/2/0,0.1961–1.3551,True
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,native,torch-eager-out,3/1/0,0.1113–1.0820,True
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native,numpy-out,3/1/0,0.1759–1.1044,True
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native,torch-eager-out,3/1/0,0.1089–1.0557,True
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native-blas,numpy-out,4/0/0,0.1534–0.9247,True
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,native-blas,torch-eager-out,4/0/0,0.0950–0.8839,True
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native,numpy-out,3/1/0,0.1355–1.2788,True
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native,torch-eager-out,3/1/0,0.0818–1.1884,True
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native-blas,numpy-out,4/0/0,0.1232–0.9775,True
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,native-blas,torch-eager-out,4/0/0,0.0743–0.9084,True
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native,numpy-out,3/1/0,0.2556–1.1095,True
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,4/0/0,0.1313–0.9753,True
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native-blas,numpy-out,4/0/0,0.3025–0.9133,True
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,native-blas,torch-eager-out,4/0/0,0.1554–0.8028,True
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,numpy-out,2/2/0,0.1748–1.2227,True
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,3/1/0,0.0975–1.2373,True
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native-blas,numpy-out,3/1/0,0.1705–1.0315,True
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native-blas,torch-eager-out,3/1/0,0.0951–1.0439,True
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,native,numpy-out,3/1/0,0.3239–1.0197,True
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,native,torch-eager-out,4/0/0,0.2099–0.9924,True
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,native,numpy-out,2/2/0,0.2631–1.4678,True
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,native,torch-eager-out,3/1/0,0.1766–1.2662,True
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,native,numpy-out,3/1/0,0.4476–1.0532,True
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,native,torch-eager-out,4/0/0,0.2662–0.9323,True
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,numpy-out,3/1/0,0.3674–1.3088,True
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,native,torch-eager-out,4/0/0,0.2348–0.9815,True

.. csv-table:: Complete current higher-median inventory
   :class: benchmark-results
   :header-rows: 1

   Configuration,Case,Native/reference,Native pooled ms,Reference pooled ms,Ratio,Native I/M/R,Reference I/M/R
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/numpy-out,107.810716,100.093237,1.07710,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/torch-eager-out,107.810716,104.753747,1.02918,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,dense-2048,native/numpy-out,126.676158,124.629762,1.01642,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,dense-2048,native/torch-eager-out,126.676158,124.295817,1.01915,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/numpy-out,136.774067,112.315505,1.21777,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/torch-eager-out,136.774067,125.381565,1.09086,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,dense-2048,native/numpy-out,127.130635,94.764996,1.34154,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,dense-2048,native/torch-eager-out,127.130635,113.048113,1.12457,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/numpy-out,6.310298,6.030121,1.04646,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,dense-2048,native/numpy-out,6.418099,5.811723,1.10434,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-1024,native/numpy-out,1.506552,1.369044,1.10044,4/6/0; 4/6/0,4/6/0; 4/6/0
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/numpy-out,9.171795,7.947388,1.15406,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/torch-eager-out,9.171795,7.689222,1.19281,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-1024,native/numpy-out,1.475731,1.447552,1.01947,4/6/0; 4/6/0,4/6/0; 4/6/0
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native/numpy-out,8.704638,6.423397,1.35515,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,dense-2048,native/torch-eager-out,8.704638,8.045276,1.08196,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/numpy-out,123.736721,112.036158,1.10444,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/torch-eager-out,123.736721,117.203422,1.05574,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/numpy-out,133.696443,104.550442,1.27877,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/torch-eager-out,133.696443,112.504743,1.18836,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/numpy-out,7.031703,6.337937,1.10946,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-1024,native/numpy-out,1.728854,1.693000,1.02118,4/6/0; 4/6/0,4/6/0; 4/6/0
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/numpy-out,11.146144,9.116241,1.22267,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/torch-eager-out,11.146144,9.008320,1.23732,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native-blas/numpy-out,9.403459,9.116241,1.03151,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native-blas/torch-eager-out,9.403459,9.008320,1.04386,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,dense-2048,native/numpy-out,68.010165,66.694926,1.01972,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,dense-1024,native/numpy-out,14.656363,12.655776,1.15808,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/numpy-out,104.817928,71.412644,1.46778,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,dense-2048,native/torch-eager-out,104.817928,82.778267,1.26625,70/73/1; 70/73/1,70/73/1; 70/73/1
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,dense-2048,native/numpy-out,8.575426,8.142355,1.05319,3/5/0; 3/5/0,3/5/0; 3/5/0
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,dense-2048,native/numpy-out,12.848515,9.816973,1.30881,3/5/0; 3/5/0,3/5/0; 3/5/0

Source, controls and raw evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* baseline/cpu-release: compiled ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``, tree ``06f324529806d3edf005e4ecbb6a3c27f048f946``, identity SHA256 ``82f66c8a54d2c5a029ab6b62941d37650c1d2f07441ad6dd5e852b052e567b38``.
  Loaded ``_core`` SHA256: ``76168203038c854eec161fb28eac73b832ec0d0341d76e0266dfdc3514318611``.
* baseline/cpu-blas: compiled ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``, tree ``06f324529806d3edf005e4ecbb6a3c27f048f946``, identity SHA256 ``c2ee4d3b54cb2f013799122cfb6bb254f102582931ebe16f341bbb3698ace8a9``.
  Loaded ``_core`` SHA256: ``81f5e99ede9ebf0d62e2ffcef1b4529183ce335a9d2e713d6a2d5373c7f63702``.
* baseline/cpu-openmp: compiled ``1aae5d2d8c07243ce5e15b2fe04784ed24b12ad7``, tree ``06f324529806d3edf005e4ecbb6a3c27f048f946``, identity SHA256 ``425d959e2b85fc58d2bdfcd3bc1f41f95845478f6989adb1ff723d237920375d``.
  Loaded ``_core`` SHA256: ``17d44c0b297d33607aef6c51084304e1955b3e5961fec29409e4a13a57cb1ea1``.
* current/cpu-release: compiled ``2d81b3d1409b0b428572cbf26fc7418e03bd270c``, tree ``90f513866f9f2aa349b92c8958b1aaaebe24f1e0``, identity SHA256 ``bd3a934ae3d0b8e43055634a6ba22fbc8870a405bce2a2f4082ef7aecdf2c5da``.
  Loaded ``_core`` SHA256: ``26a23a8dcbfa77ef6be4cfa34aa5c001f5a684edf94d77d403742afcd15b3fca``.
* current/cpu-blas: compiled ``2d81b3d1409b0b428572cbf26fc7418e03bd270c``, tree ``90f513866f9f2aa349b92c8958b1aaaebe24f1e0``, identity SHA256 ``5f6ad045349cf3e70689876c9011497479502e32f9472a37e763be9ff8092467``.
  Loaded ``_core`` SHA256: ``fba22b9264cbb2f4ef0109143883ab7bd475b85b92ce42b0b93b30d7483bf261``.
* current/cpu-openmp: compiled ``2d81b3d1409b0b428572cbf26fc7418e03bd270c``, tree ``90f513866f9f2aa349b92c8958b1aaaebe24f1e0``, identity SHA256 ``3ab4d7e1b7d6f6ad63e6cde68708e5a075565a94d2a0cec0501832af4f82692e``.
  Loaded ``_core`` SHA256: ``2d58e607d8562879f6fbaaa8c0a6dc5643b3c3927b5e7a5e1bbf1be4c4aa3d31``.

Manifest SHA256: ``a5ec759c657e79f7cd22547fb36794c328855f7a270f31720c13b079e4dce591``.
Plan SHA256: ``7e0661254f1159b132252a445b93c8bd867da77bd952362f8d9b1a4940ea6418``.

.. csv-table:: Every full strict report and actual comparator
   :class: benchmark-results
   :header-rows: 1

   Configuration,Verdict,Report SHA256,Comparator SHA256,Comparator path
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host,inconclusive,3e67a6348345165654564a23be227abff1118ed45ed429e7baa216624e9aef28,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host,regression,3d96f293b271cbb959025bf99cbb3f2d81f2943656ed3a422959e21e820e9de9,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host,regression,b8d566ef0f4ef96baf0847830a9a8fc08feae4afb663e4a0d263f6472f305ba1,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host,inconclusive,14b0f0ae955461e0aefcf2ab44d10fcf53a97cb45d26ede9d709a5fb6646c56a,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host,regression,e6020d59604ce6c331c1f18caea531223b7494b5a1acf6c59443c471318e4c17,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host,regression,bf44631d3ba9cc979a2bc9291f9127ad669c6aa853c2c5afee150c81d1d45b02,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host,inconclusive,ab3e9f053242e98810f880409ed235ca6bd07bee485ab4cd521ea7afa16d2740,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host,regression,57d76f3fad69758da3790309170da00c69eed55e012140598bf53933f9ce5510,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-release/tools/compare_performance.py
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host,inconclusive,c02763e66be880d3258388a45e602b2e4edc9408236136b3e476f79d57b828c0,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-blas/tools/compare_performance.py
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host,inconclusive,00bfc0956ec4e4a28217710bbc9e1fcc35b447cf7cc04684cca6f7c90c601f5c,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-blas/tools/compare_performance.py
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host,inconclusive,a3fbcc75f93dc6c2cd533fb3d6744b0ee065e456808a0555d2445c1d7f5612ee,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-blas/tools/compare_performance.py
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host,regression,68528bed32601a75b33465ecf2953706b6c4bde1e1034e5aa2aa8aee41d5f5fa,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-blas/tools/compare_performance.py
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host,inconclusive,e01dd4b27285aaf91e18befbcf9868ffd8e757321480ad9582c97b6b3624c4c8,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-openmp/tools/compare_performance.py
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host,regression,3c4638a66be3c7dadc44ba997ed2c27bf6bdffcfff48e1bd3fee9c6664376e74,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-openmp/tools/compare_performance.py
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host,inconclusive,85a4c0d54740cf21e6711e12ac251b802321ebccba6665ba1300a64629824cd8,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-openmp/tools/compare_performance.py
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host,inconclusive,e7736ccc6415925ad97d5517ebf6f7583d4ce28c857e1e4e340586e2b5e7a81f,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/cpu-quad-stages/cpu-openmp/tools/compare_performance.py

.. csv-table:: All original raw/import/log identities
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Import SHA256,Log SHA256
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b01-baseline.json,87338e369b6a4839c414a5d57657c53dec80bd1a66587429edf319ea7c368c6b,57d4335ea815368e616a27d53b9bc603abb9adfc574c8d5d11501229fceb9724,a8ba3a85b0d9386235bbd2f1d34ba29f7c761a9e83adfe53bcc8c2ff6d35c255
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b02-current.json,0ac19d9a812ae435b66d0ea7d0331b587b08324b0f2fa50bfa6b05d76ce27f85,774093c654a8c3a06eb52694543110f3bec5f72a727e4f0f62dfda9cdf658ad2,bdbc907e6b8045d237197b4569b4303b392338576b0c7ce3b462a921ae1b6852
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b03-current.json,157f9477a0b99eda693bae3ff72740dbd162e2e98f46ac825477e62dedb7c1ce,486ae209d90937ff863f494074bb11efb9624e71d38d6552282c57574485a223,b8a97fa61bd7baa4b1219b53c81868a41db92238ccaa321e4e2056b599112eb6
   cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused1-host-b04-baseline.json,9841601b1dc2511c6c4c3e8de09bb6c21ab37bd0adb53bafe60551919b195cfc,6625af419557c0dd017466e59097a899a5ecd36c2a4133c9ebc5bbb261a0933d,8c10829e0f34650870a853c2c124ba1e4a8e82be3ffcf9f82412076d3aa953e6
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b01-baseline.json,beb2478b2186e60128c8027a834d7ca6903121e9031db70124fb4c4cef2550bf,b7b4b5c7a19e2495b129050dbf09e4a05109a81277325c18e6ff72e38cbeb99f,36175de7f4029aa8038488be0caccd216c4d661723f562c0ab7186b633910657
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b02-current.json,d452f9139e5a8cf62a94dedbe02074c837efb43bd0d6c3593910860579f270fa,57af0119ac585576d04176f4eca2c7734d8f25f75577d44c86be0ded6c33cd71,63a3131897610391c6c78f44d46354583cb76b43e3dca5eab5d7fd30fec42980
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b03-current.json,0fbea281bb6ec73cf842b0e9edbbf67126b106fab09ef7fc171d731f71a0d522,ef46b3fa2950d8754e182d2c89b741758663df9e12336593f994be091bb20007,b024fee7e37bf880d7acabf0926a530333cbdc3e3934b89f1290de8678480ba7
   cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-api-fused0-host-b04-baseline.json,a942b2afb8128da93964cb2149a870f2330cc4ae27f6f39adf60d9afaf334efe,fb3eb7ad9b8535c74d7ecd9cd5f46eea998451002fdfd07acc832d4123620d75,0d583a5ec8509c544820eaad316dbd66ca473ff50701bb229365d2d36c2b0ae8
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b01-baseline.json,322a4fef90ce9492640b27356844c442c0a1efb5f9cc971a7f652839fa9775e7,ed45901afa37c265a360dba66ea3e9bf11f7903f70bc2315388223112f66965f,532c15b54d654beb5b99cbc6400cc5201d7370fd34b0d7f992ea509cb59ed980
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b02-current.json,deb7aa8919400a6e3a4af4af681a76aa79d61e465d56abf26d4a283e0d7a6a74,ad7ad589b9b7455928b792eba5fd74faefd99a7dd63455bf69df89fba12aff03,9f9928adef823340d8d612a8e5c5deb152322979cc77b01c6d23f8a5f315c72b
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b03-current.json,113766534c8d308a98091f5a74593b4c509e87f2582803ad4769fca048e8d590,20112e902b50f710b32638c913804637a9267eac39985a11b4af10d2f891ab6d,354174aebc75fd34eab3a44beb5a192d0ddf0957497e45510b9ecc65ca39e229
   cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused1-host-b04-baseline.json,43d6482ac311b9c6e2ed95460fa123e69ce453c5d71b37bae8834a472e7d79da,016c14c963d6ca1e922893012fefb8c9812a24c128c52103721c6f569cd80440,abfa7e8ba5a6d744e5b592a679331081ccd50d883acb86165b78eee7b32f4bbe
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b01-baseline.json,dc4e9fda08611e63dd26ed6e58a6f2914d4cacad0990382ed39996653d129860,3cf9b6e8bccfb4cfdbfb60103e75f5eae802c1c76f28fed84a13003cd2fbac5c,6679883aae79cc23f7b434f0a26a97c4b6d2e7a0b3f52de719f6e271945f0401
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b02-current.json,e6fe5f1b16bf23bdaa6a18ea9a584f05a9c5bd37b8a35dbd80c9138f927c3c84,23a94ea04ce9fd19525540ab8d9a7fac062515f2bc4496193058ac951b930028,0a20f03cc5a449aa109cb1b180eb8f4963bd42085d6b2e15e0696ee0a3dea2f0
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b03-current.json,0fd74d3a9249fbed9d15f663c21f8f47c07fc636105aab19a1687ed1825d1bc5,66e28331e14182e7d681f70f40dd6ffe61ce158469ea817a1881a4d737c30704,18541d45747db5693af9720c22eff50052a7ccf9c34c480dad3844cb29a8c660
   cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-none-prepared-fused0-host-b04-baseline.json,38eb9591e38f2eed7478a68f2e8be8b7d223b198fd0738e22bd738d1eab86211,2aa88675193d95db46def7bfbbf6c0942f75e48f02a1ce261ebdabee15ea8a9b,8faf8790f29586842b8035115cde3b7ec9e8d0a8ab1866ea7ce6ce2dd9a43314
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b01-baseline.json,05f6de441e576f7d62476b2fa465a33e9b8c1bdab4cc2ac617b1452e9c29547a,89fc5f284d9c4dd73208decea1330ffd01eac7f47747749ab335e7d02c045b41,3ad38009fe0bea027945006b065bac51f45d3353a025eff1f330a6ad1d999125
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b02-current.json,d18d0c68c316b6648c2dc5a34a392908987430e0dcd1d3ba3b8f2d3cd612d159,a057f538ff1effb3b268905675d49b81c02080b5e36f4d3c42b6d1b10aebfcd8,252187d69908be0529979002ea3a4439004d12ff175cac8ac9f79fde943105fb
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b03-current.json,40b1852bcda59ccd05f5a3432ed78c6937a6da278ce5baaf85a156334c202ead,f4c860dac51bff6bf9dd59cf684edafc5f4b1cd52a854dfab1dbfa34f26fd7fe,3e8875d27e92b0afd562a1a3025b3c448def01317b558a948b51ee63a19c4c8e
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused1-host-b04-baseline.json,54708e6a02e61aa7fe4c317459d8b69f4ef6296edfefe3fa5ff98a7c6281ffde,a1db7be8d26d86f4ab1cad386c88065d208d5e4ec0c178ef55db3ad66edd2b87,bb70a2c74fadb4d0ff1b3c9a05a47aec90f779050b1ccf909bc7c22fc908e266
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b01-baseline.json,4782cba580fd844c703bfb429fadd6ff40fb8a3ff7da59ba29704fa3a311dee7,13563f8fda9d17e8200113640e51cbd2b472ff1340a52fa6218826ad66206b43,65d8b09571385fa3730fd8b03e81ccef49ee62df2dce6a150291efa9be120a7b
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b02-current.json,1b1e600257c498a70076ccc17ff79b224577244ed04b13cdf4e422f60cb0879c,9dafcc13c0309c5d80d30e93e13cca80cb7802f9c4fee73918c440e58c3f7bd5,3f0d873aac4390360373971bf3e3db415403c215ac42b6fff27d5fba3723e0da
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b03-current.json,9cf803081601a4dd49ed693b0f1085fecc7c1ca30d3df47aa74e7a6cffacf523,42bde8b33ca6fb2b3331297d2960165998c8cce14302fa6bace5a4a900bf00a6,24d656ef7ea5fa592fff27a3f55374ffa87f6d3e5abc83d14e4a1c100c357cdb
   cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-api-fused0-host-b04-baseline.json,e4ddde3e4cedcbda765a54bb1a01aa2fbcb625a3ced1a095811322e6d84c909e,6b5290e9cecf53e976b767c09ac6950fb13ffffabe6b75b2ac1f515b6bc52524,9ebbdae703797527cda719cf8a89f6f34036d458ff6605bbb6364f6d3d4472f9
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline.json,e3b3cb481d2b591264098e908eb99a16cbc4c90cd5387a43604a9b7cec1dfbd8,e6e321c9e1f5fb88eff27a34f6fd389fcb710324ac03d97efddeee5cbf495b66,5dbfbb0f950fef01b215367ae5e4869858fecf24dccacd430037c883109e1351
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b02-current.json,d6fae0c4ecd0cd705c39be343e4a9c0e493c0f8f13ab8eed0e341771c2672de4,8862fb89be47ce36a4317abe9f09c7d30ad5c0f2dbfd615632854b9614b5d822,a930a07d30b6cdf93c32e31afa9d03df7cbec65b01415e148918ca9f613ad120
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b03-current.json,b13dc2bccf1858366445e5416e59e5236b653911ae2532bf7a75781694c7960b,9ad835904a9afd1e009e271add5b4f5e740909e76500fb60cd55218c5aa6dcd7,1733302ebfa8287559e040d7e094cbddaa81f75e11923b66459f8b2b5efceff5
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline.json,a981799aa437e51ecabaee10c8a93626d63bde7fc61824645e37ec8ea9596685,2e571c1050d14c21d7c913a55f73df67b857c50c5c76eef89ee9ee34a431b9e9,5c0c4f0b0740ffb1349df6eabbf8df2dacd6f3fcaa917446dfb27e0c33e6a171
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b01-baseline.json,e94541d3bb94dfc6a77d0b1e4cab2befba488a703ec7b081f08402aa61a539d7,338c033f4614bd7d7f4dc6bd2f9c4b0f947502447cd2394c16ea912c1ecc1cdc,e36c98d02779b77735d9b27cf47428e4f0a00032bb5c239c217cad0b5453699d
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b02-current.json,419de2dc289ff85f64358515e46b299c959714e42000c71aeabb8aa4aaf2d738,7c34ff8b4ed5b84e1673a2aa6c12fe03cb4d4e1f3d68233410d0fa2e8cbe949b,7c717c7d3f5eb9c978065a4c7a2221e4e1060b2fcec1bf81aaa7f57cb9037a30
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b03-current.json,6351d9ebaf0e228c127ecd5fd97fdc7237728f9603e71deff69730b8a769ae93,eaaf54828086b6d22d4a4335c043e5e81c27e88992f0ed8810e75c249a0b6785,da3abb9d8a1086f4e567d40f13a22ccf7858b6134e085173b57f0acc5de8b27a
   cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-release-cpu-dense-f64-v1-jacobi-prepared-fused0-host-b04-baseline.json,25578d56b0e35ffe903226f1d5b74da2f7872e7146d59646e6c098a2b19811ed,26f562699876835efa8d8fb758b0503374c0bc32d49695975558ab674a5dbb72,768bdab273f7fa21489a4bb6e4acf5bd9ebd2ef1a2a1dbea6cae60fa0d051cca
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b01-baseline.json,231f8a7c00990528b539f8dfedd28c09b7a5029905234f3582e17903a03882bf,cb6bcfebadb597558ed7695f2f94bf4f22a05c4ef9afb22c25556bd4b548a12d,536e78bcd056117280571f922cb4295eee28402dfcda21ba91fb9a02f9bddc3e
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b02-current.json,6fc5019782e5f678c9e5350c2cbb3bb7bed36f0521aab83872f836f1bc2095ab,f2c2f4c17cd1b69b65da99ff4ba6eefeee6dc12162af3e1e94ec19850f3a3ed7,50dba0663c0b3bc8d195329322a7263e382a63bfd007487690d57d81d4c4578f
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b03-current.json,ef235b127d5b73556f627896e2b535a3665cca20d435d5c3e8d8f571f649f0d6,086cfa88938b8f7ae0b8ff1d70c438981629b3b1476d398324542056e482f7b8,4e9692615bceafd63fee30ae65a6e9b974f2db38be8ca7883f42afe52919f59f
   cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-api-fused1-host-b04-baseline.json,e36ecc9787a3f9757b38656f22f2344b0d0bdfa1d23c96c67688decf6237f2a6,fef2fa44930b71e845afdddd57feda28d8d98b4596509ad7926bcacf7ee7d81f,3d5e2377d1c2f85444084d9c2eca31b91ca8d2ddb36f4d3cee976a3877798fd5
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b01-baseline.json,fbf4e2b3cdd46d0ea13dac1d0fc0901931e02db2b0cf62b72c30fb0eb531ca7c,067b0eacd3dafcc5f2ab46e18fc46756d18b6ff9c5003a5fdd2bf7148c1d08ed,a53b236113e87c127268703544cda3249197a04929c97f4ff2e00e6253a6392a
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b02-current.json,fa1e3fcfd01cad90bfb2ec2d3bd9c332c4d7483b3e68b503a175f53c517280ea,08ac9afc66162a180a6ae9c1ad2d87b86dc2c410aa31be66a0e5880fe4bd0b48,a31aac1cfc83b0177d674e6608a501609bbc16ba618e7d973e533d54835be9a1
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b03-current.json,d3f1801ae87cdd2e10dcd077d1f41db8c7f71f9dd1eb500416c93f2e7a958154,4ec3544167836adc3e6cc7b10065d5a347f62d10f4c81c4fc707f9038fe21b4c,6fb656c9927a310ca2434a490dde767a72a7bf92ab5e7172380c67c4851f1e66
   cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-none-prepared-fused1-host-b04-baseline.json,5b3daa55b91f6b33f8a72c5be0043508245a3d9e40d9916cde4864dcabb3db1a,89ae0a6b55f7a9feed266565a1799b164cd6083546eb3f6bdce1057cd76c8ec1,2f709b7b5042062956d0feeef4c9414b625daa9f963bcc268e52eacbddeb3a5e
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b01-baseline.json,d9f118d337623fe2fda735264a454aa0d83064c07464a355dcca030f38920cd0,eb3a60ad3149babc7c3f0406c4553ff5b919b34815809f5ab79061b79e21d9d3,416eaf2a3a507006068c2b1bb8eb31f4ee217ce3cb482a8e1eb8a6839b104ce3
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b02-current.json,aaa2817f2714b6fb872e007b9637d3225fc5351467c08312ebf9e5a5b3680ee7,7847b579f531edcf5b7636699b7ef24feae2383e30e797da69f625c64154b0dd,3fb6e2a8ac7d0517b4f96758d797f63329fd9d1dfa42e34f2c9ab0dcbc63299c
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b03-current.json,79b36f82b48c02477896f8d9d81ce30605b84a9d2791e9137c183f4d60ed444f,10e90bcceef699a7e1c34254e664ea61a0a8eac7219ed8f773b9f33779c188b1,43d568fa00427b30f77b712a67250c669e5e10511f393eb257cf8bd7f8c1564f
   cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-api-fused1-host-b04-baseline.json,2039fd2b9ecc6d165bcb262a3c29da908924c8caec9a0e467b663b8c7cade3c2,8aeb4cdf0be47fe11ff2dacff9cec6d0e3ef02064bf53a20f1a86c87c903ff29,b13b8759eff38999b49b6478bcebd92160119ea0955cd0f891ea4c6a5f4b2654
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline.json,193461c9569d7bd77d54416126f047adddf63f54f3678d27584b3e0a0de4d7d4,a096a7482da30a735fbf808918a2d4ddcfa9f8d3d2e0671253dba2b3477f6610,e2e403f46aeb51c80e71f856729c461854114571e42de911fdc488d00b4c6085
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b02-current.json,a6c2db7ccab86ce0e796eed141cfa6773d35389345bb14db22a88ab46b87858e,bb7f6ca14f4e0b0f8a42249c14574d82374fcab2ff97291313ae373d77f005fc,00b15a232939c55da0bfdcd515e80a835ab83f59bab18eb0c4f108e392606e6d
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b03-current.json,713f303ef4aaacd36f59aee74c0be87b9d9718ec275ba847d7bca88918a3828e,4867f72317e3531cbd5898ce043a1a355d6f40e4eeb13f5d811071fd551c116b,44c583fb97cf2c6765b6b315c00eb626e098271d49f9318e0c36d499e86da026
   cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-blas-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline.json,bc9c31fe613b106cf5e569882862539201607ff6c3a484467d09986dc38c3b83,387d456bc1a7db5d338a63bd6a0df71792c627fdc1d2d6a98846eaff2ec4aa73,6568cddb4157d8cf719eac0138bbeeb321cb08a2f7d588d7828cb8f7c2fa6fc5
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b01-baseline.json,e49ec9d453bb315b07b8050271198428f93b9062fadd785e2a0273bc0443c92d,79e3cfc4f7a743fedc1be951eb1f724651a5083d57f22cb759cb9c01cc444e58,782356b3bb1a3a7f07e52d86228a0ada663d0ce67b9a5d5456c791a7e4025baf
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b02-current.json,42c151026883abb2793f0dc77f01ceff0f9c966d2a5b61854e6f4ed0b3128d9e,d364810b1bdd81699d0f7d84d6556a753873efe2dea722e53c32239b8bb66ebc,afcc7029798bf828e99bc19167379f01e2a22553c4d245a2800e3be64a519a07
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b03-current.json,e09a657d8166c80e32bf5d022e4f3fc348173f016dc3392b85683c61df471c09,aeaceb33d31063a1e58a290036448b7d80e4a6ffee046f3f3e6a3f92b1ec6886,307141be5cd4cf8607576f53902da517a97e8e086b2344ad5b9856e1a7e7d46c
   cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-api-fused1-host-b04-baseline.json,87b3584d7b8161ce7e95e066e017bdcf9d60d34dd58fd366b165786a0d147758,43c682a76c265764fbcfe905af75ed8c8230310730fff1365b9fe5525b14e7de,5e3407e1bcec7076c5e4fb97cc3a704b815db90e385f8543f7ec52792664c277
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b01-baseline.json,fde3dc6c3a87af2d849d3b477df60cefe7ff3f9a922db6650a9c62277698f946,8fc70404ba771f8c826b75247fae0f3340ceee4c829f45a6a48cb72bfd4e8001,1fee172758d54a724135e4bbc32e43edfafd6f099dc6733fe50bb71bd30ea228
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b02-current.json,27fb3fc27d7102319c600f85f4e961c0e46b4a578f5250c5284f30c614630440,6221722326c14faa15f8d162bca7d31e65f0b92b5c370fb2115341bf0c738d26,d35d782ac392baf3eb29bee295024747cd36c601fc0033a06a0995d9e9f25ac2
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b03-current.json,10855a1dd8097548d43fe8b990e853aab0fa12032ed0fd80ef52dda5383f533c,12d9162844942c40469d3334b0f096b0b07dcffab383f76bb7b546ad6d599654,6986a79ceb48f9e76576f3771b0ef6b54f03a8b9abf3eff6191d0c5304eeb7c0
   cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-none-prepared-fused1-host-b04-baseline.json,912700f596ebe3442b044b70bac5ee22bc768124db8ad520382331f2b4831fdc,07bfc8505b5a6313a0243bd719ac6333f262b66c1725ae3042069aeb2792308e,8a2871dc43421da4ee973b626bc01551c6c2d95aff603c07832f71f4677b9814
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b01-baseline.json,3be161702bfd043750af70a9d62dd6c56a3ce6aaf8f1d6d426734edc01d606fc,cc1c8263a09c66f8524cba614dfb0fc8fd67bba2ba8a0421d5dbd4c2dda353f4,fda09f24414892d80082947d968e3b1afb22f91d5d3ad94749c8fa86f58eb05d
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b02-current.json,eb2f92cdbffa93c30b0dad400c4c1842627eacc7cfd89b027f83021299259af4,728c6d98ffeab4dd63d85af0b21f20a061208a12089343eb77b37abe4696ae02,b0e1f4b283b0867e29f10f3f7fe5745ee4552adbbf3a7d66913321764df8d32c
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b03-current.json,60acb7967326b2181c98e9b8872b75af8f91090bb38f80194b7938aa5d4e8ab0,e6dfae4e995bb36dd6b1a243f5093ecf8a9700d06b1a10f4abf9641d5d238d9d,f095633a3b6112fd85ab5cbfc080b4b9a1d1708115d6507d133772f731853cb2
   cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-api-fused1-host-b04-baseline.json,cdb156e48c8ae81d568f7c055b73d81149c1e8c515caf6a2c02e5376bf87739d,5145e3aaf6ed238a03694d166b18deda525b640025570ccb831b500d463ea00c,abc9249c2c6c4c12f3d17cc44e6f30d1ee3d3242413947736c067a1f1df71a1d
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline.json,58d3cf8c6899924b2d8066db7d659ed975b290914672ac45a8f5a5f7681cdbb4,8ebbf8eced9c45eb0e1bc647caafdd5ad9474f345bfdabf78fe8170b48e256c2,9f373e470db3f3f67945b1fc50d22560489e7efb40ab8b187e681a410a9d06fa
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b02-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b02-current.json,5d7e1a15e3667ef45acb05ee27be76c9f8b6365552d2a478799698ba0d76ceb5,a77c2846942c988b6cbc7136c8350847b658715bcb8d854d6b23d568faee87e6,40c39d93eccfef811a9de74add3728efcae1b909acd9fe900ec2c697d0e0b4bc
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b03-current,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b03-current.json,bfd07e1cfa3806456c022342c325522b36924963ca2554a1e668bc84c2a9b900,a309a6df3bb0608ce9a47ac0029c5ffa95ceae540df3da9eaf3b29d35d942972,f9669c4256c71d489c0f6597a1d36d9e0d3ae1eba2c1faf23983ddc64fb88332
   cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/cpu-quad-abba/raw/cpu-openmp-cpu-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline.json,314f1be3aa5e3086c8f44529356b3d0618d9fe47ce2ea6b1418c0e551b00fec8,43a2ac5df703cef12136ca782d097c0115166b7ef35b7748f45d8d46dda35bae,e6a617d62f625a13b985a0046de69c0e9a683a3e6d7c513eca97e1913cbc164d


launch-abba (launch) fixed A/B campaign
---------------------------------------

Whole-campaign verdict: **regression**.
All 16 artifacts/160 candidate/case outcomes are retained.

.. csv-table:: Every native and framework strict result
   :class: benchmark-results
   :header-rows: 1

   Candidate,Role,Rows,Within 5%,Inconclusive,Regression,Comparison failed
   cuda-native,native,16,7,9,0,0
   cuda-cublas,native,8,3,5,0,0
   torch-cuda-eager,framework control,16,7,8,1,0

All candidate rows
~~~~~~~~~~~~~~~~~~

A/B and current/reference ratios below one indicate a lower measured numerator.
Ranges bracket the two process medians and are not confidence intervals.
Saved strict intervals compare current against A/B baseline; they do not give
confidence for the separate current/native-to-framework comparison.

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,4.813830,4.556268–5.550495,4.675959,4.664255–4.675959,0.97136,0.8148–1.0666,inconclusive,49/51/0,0.648887
   dense-64,cuda-cublas,5.244673,5.126512–5.356832,5.042561,4.869444–5.580466,0.96146,0.8839–1.1368,inconclusive,49/51/0,0.611009
   dense-64,torch-cuda-eager,16.846000,16.743127–16.893090,16.146093,15.254762–17.171689,0.95845,0.8559–1.0860,inconclusive,49/51/0,0.611009
   dense-256,cuda-native,6.553387,6.523076–6.589938,7.002052,6.592493–7.248671,1.06846,0.9546–1.2605,inconclusive,67/70/1,0.979365
   dense-256,cuda-cublas,7.117926,6.572613–7.163162,6.782412,6.387729–7.027467,0.95286,0.8460–1.0843,inconclusive,67/70/1,0.994421
   dense-256,torch-cuda-eager,20.814728,20.379113–21.335915,21.279331,20.428297–23.146691,1.02232,0.9468–1.1632,inconclusive,67/70/1,0.977359
   dense-1024,cuda-native,9.972689,9.891568–10.082052,9.296761,8.969531–9.324780,0.93222,0.8313–1.0360,within 5%,70/73/1,0.928014
   dense-1024,cuda-cublas,9.121081,9.003308–9.400038,8.356147,8.272370–8.810037,0.91614,0.8272–1.0241,within 5%,70/73/1,0.928014
   dense-1024,torch-cuda-eager,22.006749,21.843463–22.198815,21.489071,20.914811–22.545226,0.97648,0.9214–1.0432,within 5%,70/73/1,0.928014
   dense-2048,cuda-native,17.643095,17.643095–17.645789,17.394528,16.077773–19.172180,0.98591,0.8752–1.1296,inconclusive,70/73/1,0.871762
   dense-2048,cuda-cublas,16.857435,15.434188–16.938054,16.039076,15.595882–16.795307,0.95145,0.8600–1.1303,inconclusive,70/73/1,0.871762
   dense-2048,torch-cuda-eager,28.044739,28.039097–28.074096,28.948305,27.654878–30.322626,1.03222,0.9677–1.0992,inconclusive,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.728329,0.718538–0.731125,0.793611,0.793382–0.794507,1.08963,1.0057–1.3034,inconclusive,5/7/0,0.0796191
   dense-64,cuda-cublas,0.815262,0.811589–0.819505,0.924808,0.886712–0.936325,1.13437,1.0261–1.3675,inconclusive,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.231371,2.210525–2.240413,2.806099,2.601447–3.035518,1.25757,1.1246–1.4234,regression,5/7/0,0.0796191
   dense-256,cuda-native,0.682779,0.672356–0.692455,0.717371,0.665252–0.746328,1.05066,0.9396–1.1419,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,0.710753,0.659617–0.734714,0.734561,0.706212–0.740878,1.03350,0.9404–1.1624,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.776912,1.577338–1.978453,1.925737,1.868912–2.232284,1.08375,0.9258–1.5043,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.024139,1.024139–1.033605,0.983904,0.980726–0.986058,0.96071,0.9201–1.0124,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,0.890312,0.890312–0.900717,0.865443,0.861279–0.868336,0.97207,0.9068–1.0204,within 5%,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.063054,2.003967–2.079198,1.861300,1.799389–1.909934,0.90221,0.8432–1.0289,within 5%,4/6/0,0.013684
   dense-2048,cuda-native,1.493251,1.452932–1.517471,1.354861,1.319800–1.377295,0.90732,0.8475–0.9595,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,1.340388,1.274400–1.389114,1.226325,1.187407–1.238725,0.91490,0.8511–0.9851,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.326382,2.130661–2.472691,2.049732,2.005732–2.136675,0.88108,0.8010–1.0166,within 5%,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   poisson-8,cuda-native,2.726065,2.608788–2.744769,2.271615,2.203330–2.314783,0.83329,0.7679–0.9020,within 5%,26/28/0,0.579621
   poisson-8,torch-cuda-eager,8.167160,8.123282–8.167160,7.141832,6.724729–8.084623,0.87446,0.7745–1.0480,within 5%,26/28/0,0.579621
   poisson-32,cuda-native,12.351859,12.351859–12.501883,10.142781,10.141766–10.204629,0.82115,0.7616–0.9193,within 5%,118/122/2,0.840189
   poisson-32,torch-cuda-eager,35.556039,35.338629–35.750465,32.857698,32.658737–32.958426,0.92411,0.8894–0.9582,within 5%,118/122/2,0.840189
   poisson-64,cuda-native,21.039181,20.199127–21.737904,19.249882,18.542243–20.657129,0.91495,0.8337–1.0268,within 5%,201/207/4,0.907427
   poisson-64,torch-cuda-eager,62.768426,59.110305–65.638706,56.278432,53.656345–60.147633,0.89660,0.8158–1.0201,within 5%,201/207/4,0.907427
   poisson-128,cuda-native,61.233210,60.160044–63.369062,60.074118,58.036546–61.842773,0.98107,0.9133–1.0719,inconclusive,627/641/12,0.989568
   poisson-128,torch-cuda-eager,179.504109,169.993008–189.327686,175.607445,172.820148–182.200422,0.97829,0.9048–1.0778,inconclusive,627/641/12,0.989568

.. csv-table:: cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   poisson-8,cuda-native,3.209334,2.885346–3.536908,3.244278,3.180382–3.284078,1.01089,0.8233–1.1758,inconclusive,26/28/0,0.579621
   poisson-8,torch-cuda-eager,12.119196,9.662614–13.498572,12.371728,12.250894–12.673979,1.02084,0.8101–1.3510,inconclusive,26/28/0,0.579621
   poisson-32,cuda-native,13.737134,13.084474–14.073719,12.920089,12.648853–13.527864,0.94052,0.8255–1.1026,inconclusive,118/122/2,0.840189
   poisson-32,torch-cuda-eager,47.675364,46.249704–49.681671,49.659800,47.601290–51.178409,1.04162,0.9559–1.1082,inconclusive,118/122/2,0.840189
   poisson-64,cuda-native,23.085894,22.834246–23.554558,22.461419,20.958669–24.395005,0.97295,0.8839–1.0869,inconclusive,201/207/4,0.907427
   poisson-64,torch-cuda-eager,78.673308,78.450570–78.916816,84.202037,74.901499–92.845909,1.07027,0.9377–1.1933,inconclusive,201/207/4,0.907427
   poisson-128,cuda-native,67.216064,66.933736–67.486862,62.867447,62.646747–63.872352,0.93530,0.8927–0.9740,within 5%,627/641/12,0.989568
   poisson-128,torch-cuda-eager,257.307359,256.046632–260.969516,240.192646,240.192646–240.737775,0.93349,0.8959–0.9854,within 5%,627/641/12,0.989568

Current-source framework comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are descriptive ratios of current pooled-round medians from the same
matched configuration, not strict comparisons across frameworks or scopes.
Every higher current-native median remains listed below.

.. csv-table:: All current matched scopes
   :class: benchmark-results
   :header-rows: 1

   Configuration,Native,Reference,Lower/higher/equal,Current ratio range,Same known I/M/R
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2896–0.6009,True
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3123–0.5541,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2828–0.6610,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3296–0.5983,True
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3087–0.3421,True
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.2602–0.2668,True

No current higher native median in this fixed matched inventory.

Source, controls and raw evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* baseline/cuda-release: compiled ``ce45ea9cc6d38cea05ef04cff8a321507e9684d5``, tree ``6cf00b82bac6f75e89093d1eb627f36d4fcc7d9b``, identity SHA256 ``e7232ead6d14f51f9f6081b982c63c3fc3c9c553876d2f4daa052ba5c8fc3d47``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``f5ae10070dd259e06e629113ac3c101d8c0e2248a2b5810239d3779cba11de25``.
* current/cuda-release: compiled ``14c37277de16182fdc7f4cfcf9193e0b352791d1``, tree ``26fcec1817f59f8dd680980fffcdf8fe0907e628``, identity SHA256 ``daf34907780c5b6fd366347ff81161904289228e9cf5960e85c0c669887e6466``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``2b95332f5e7cfa622e5de9eeb109ed8d6b2798f7e9eb9f8ee7d686a9c1f71fbe``.

Manifest SHA256: ``e01983ecfbb268ada98843af502a3b650a53e2ba7653c51a1f8195ab95848d2d``.
Plan SHA256: ``d3e77ce3b1aa79ad4a10df4a595702a227b180318dd3603cc60d22dfd98f3ac2``.

.. csv-table:: Every full strict report and actual comparator
   :class: benchmark-results
   :header-rows: 1

   Configuration,Verdict,Report SHA256,Comparator SHA256,Comparator path
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device,inconclusive,5508004489fade1785405ab6d12fea7f50d7acc09942e0d755c9fb7841051ac4,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,regression,e83a95ad83860bd5a7f48887b0cc82db8ac81e02f64fadf1294f8b66842ca661,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device,inconclusive,1b8ee57cbc7fd19b5570acfa24cf23951199ac95c9336d896302fc0d6dc429e8,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device,inconclusive,12282675d9c6d8959e724e03c9a56be03d6af0d71b6f506280b1312e1d1c4928,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py

.. csv-table:: All original raw/import/log identities
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Import SHA256,Log SHA256
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b01-baseline.json,6b8e219f2cdb1d3e6b5641fd241f75882c1e154252457ea1299f32b804a48fb5,316bd75d37ec2bca60fd3534582918db407ab4e2c28e10066f5d1dddee8cc77a,86742513986bbd3d3d84d7eedc8122585103f665c81f0ed3966cf24f28109309
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b02-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b02-current.json,0fb1e8787f0ebb5fc7bb6dc0fef0c689e8659f6084f9b1f758e6cb4c38f5ab9c,5adb7be9b90a2bab5026815a695f523a4a96564857bd5f35025252ec90d4f11d,5c7e0044c08c551ea7a85714ebbc5272ed386e30cee9619573d9d598fd36dc7c
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b03-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b03-current.json,917a535abcb0436d1f58df11bdd66c3c563f6dfcb7eff5602656e351a4e3b737,97668c610740b82b57f5396a7a474f76ccf08e713bd468ac145ae06ff58edd0a,7c3c3d5be3e7dd40602c09f2f6d35f87333c5aa2d819a5ce915738fb797b1332
   cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-none-prepared-fused1-device-b04-baseline.json,dbf158f4d671a01aed9fd4c08f85ffa5b08b3604cf7c5b69de31d270dbbedac3,e7459a963961f43b55aca146326b49e065e4ec08f64d9c4b0d91ebe8ee1dea30,3d1393e53420dabaf5cbe35653ec0229bd3bbecf042d59cad94f3d7bcaa3d890
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline.json,8973810f23384690cae2d8c30669de93a28f594477c4a2a650acd61c4aba4c26,0fb273f04ee4c1c357642c87a49bf589257f6047805b1fbf27bc6dac37c85888,f2aecf87e7cb6d1fea3f6461df6856ca2ce84bbfc67cb2aa55224a94deaf2069
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current.json,5e555be2c6ae9375cd276a7a6145b2ccfe6377a1004a70bc720ca6e1c3fb7600,a7c5c5136d1c6255c23d695228e0615b2d36e23a1a94a8566a0e80cc724fbeee,c62f3eea585ded27ffa93a723e00c6995d6c38a938a9d47729052eaf62c27776
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current.json,89d56029f5cad57bfd31c773799fd63da12595aaf2d165142f9a19200f9a1228,f8610f5b873978a180fd973581d989cc325bbc7c6e1bbf9c7d9a953a3e840dd9,eeccc04d10c19553f669879de376441fe51f8b2a51d7914ed138f72ec51be273
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline.json,b9c29957b5c7c708c4252713c5e0e55608aafaff1d014b39e20d9ad4dce340e0,67f0fbee39a83b5fcc03942c055504575d9e820c8995f4c9bb3f262936369c15,357f939d3642b9c46b93ab94d767a59882d56d3a5101458ccc8d1121a2fd8b2b
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b01-baseline.json,bd06d6676b8eedfb87dfbcd7ff7e52c126bca238ddcc3f1c60b53335fd31ecea,d04faa50713239c2663a87adf495e43aff459f70f90ab79b6d04272d885dff31,1b29354e4c90874e7939a1ae722dfd613df8c314b8377a821de1110983e0425a
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b02-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b02-current.json,75f6931fe6a18553d0d6455a14fd88c40b4712ef50555e006f7b4bc4357df0ed,5b328bb5462c6a8d0c652f024b7822ce9b8d18f593ea9cee48eb8724e8e32d19,cdb52911f60174325f3a5a55fed20997e03111bc28ba217578fdcf816baa191a
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b03-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b03-current.json,9afc6f2a05096285e30e4aff5aca898ccf32e9b9c5f8e69a21b8fa3fe4a47f96,e7606c28a11ddd6c805bb14eb038f7db2f9bae7876268e9e23aeca340ef65ccb,a173333301db54338357df1592295c2fd9c8d37228f60cb8d8e264451bea2a34
   cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-none-prepared-fused1-device-b04-baseline.json,d7647f84b1d089dcef6ce13b1f51d74f22768654be417d90a3145f43c5c558d7,12245323d8ad5782ebfb2d51b6cb60dc21a7c9b4bbca234dc657051e952f4384,649b2058ea8c6353251c085c2cbaaf58610c04d8253b7384033a4c094b5c1b69
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b01-baseline.json,49663fe9f18fd0ee506b11f7537feaefabe477462dd2f5c2685080b7f1185a6a,e0bf7422c9e2c28858e72955419b96207d66915786acec77cde9acb1de8d8742,5c9f151f308f086bb752e0d7d9c513756bccfd9d0d02df11331c588ef027ba1c
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b02-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b02-current.json,0dd9c5ffa622af3ac668165bfe9a0b6a1ad280bb89b243599b08aa072f3347dd,d208395ffd58ca62ab21bbc15bbf7dd6e20dca11fb6947b8627e09c581aa3e60,c15a6e5a4524d181b80c6ea783457e7215ed45f237516664b93a070c6cb30084
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b03-current,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b03-current.json,a5af15b1fec8a3d262edfb66456f630f8b2146537f7efaac639664745213719a,d90097e23b4bf47ffae87dabcee7e653f90bae74a86ad0b94954a32e52db2884,5118039dc393e32c61f9b7c8fa7090839c326a56e154af6d3f405342690ff253
   cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-abba/raw/cuda-release-cuda-csr-f64-v1-jacobi-prepared-fused1-device-b04-baseline.json,4c5c703a7ee9924a33cd655ee56443414ff7b8b5f6aa0d54d6ff260c1f5f0c91,4e9c87d757fc0823bf139a270a62a3efc3d2a1753208e0ea90048bee989bc9ae,95db05f771691988468398dae0f338555ea1ce7581a528b828f962fa4cfe68fa


launch-scope-abba (fresh) fixed A/B campaign
--------------------------------------------

Whole-campaign verdict: **inconclusive**.
All 32 artifacts/384 candidate/case outcomes are retained.

This supplemental scope study was preregistered before any launch A/B
results were inspected, following the earlier resource-variant exploration.
Intervals remain unadjusted for that exploratory sequence and multiple
studies. The separate prepared study and rejected resource/CPU trials remain
reported without replacement, pooling or causal layout attribution.

.. csv-table:: Every native and framework strict result
   :class: benchmark-results
   :header-rows: 1

   Candidate,Role,Rows,Within 5%,Inconclusive,Regression,Comparison failed
   cuda-native,native,32,22,10,0,0
   cuda-cublas,native,32,22,10,0,0
   torch-cuda-eager,framework control,32,13,19,0,0

All candidate rows
~~~~~~~~~~~~~~~~~~

A/B and current/reference ratios below one indicate a lower measured numerator.
Ranges bracket the two process medians and are not confidence intervals.
Saved strict intervals compare current against A/B baseline; they do not give
confidence for the separate current/native-to-framework comparison.

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,5.558584,5.509592–5.668073,5.318354,5.317839–5.325962,0.95678,0.9255–0.9759,within 5%,49/51/0,0.648887
   dense-64,cuda-cublas,7.624638,7.498715–7.903087,7.080199,7.077820–7.086291,0.92859,0.8815–0.9574,within 5%,49/51/0,0.611009
   dense-64,torch-cuda-eager,12.122936,11.717379–13.200377,11.954493,11.684459–12.036627,0.98611,0.8839–1.0403,within 5%,49/51/0,0.611009
   dense-256,cuda-native,7.904368,7.780297–8.094331,7.804919,7.748337–7.874125,0.98742,0.9495–1.0227,within 5%,67/70/1,0.979365
   dense-256,cuda-cublas,9.700395,9.627548–9.766035,9.457528,9.367525–9.568084,0.97496,0.9365–1.0136,within 5%,67/70/1,0.994421
   dense-256,torch-cuda-eager,17.199876,17.199876–17.242945,16.625322,16.547660–16.940137,0.96660,0.9242–1.0109,within 5%,67/70/1,0.977359
   dense-1024,cuda-native,11.756305,11.693552–11.780366,12.222746,11.193752–12.662413,1.03968,0.9434–1.0931,inconclusive,70/73/1,0.928014
   dense-1024,cuda-cublas,11.668278,11.583510–11.687221,12.423585,11.587998–12.561685,1.06473,0.9848–1.0896,inconclusive,70/73/1,0.928014
   dense-1024,torch-cuda-eager,17.357174,17.326778–17.488304,18.983285,17.066225–20.092351,1.09369,0.9661–1.1665,inconclusive,70/73/1,0.928014
   dense-2048,cuda-native,17.156406,17.025800–17.392353,17.457335,17.284956–17.558044,1.01754,0.9921–1.0369,within 5%,70/73/1,0.871762
   dense-2048,cuda-cublas,17.443869,17.330283–17.531420,17.684451,17.539849–18.099760,1.01379,0.9970–1.0473,within 5%,70/73/1,0.871762
   dense-2048,torch-cuda-eager,23.270290,22.834941–23.671836,23.488968,23.060504–24.277636,1.00940,0.9736–1.0646,inconclusive,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-none-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,3.920649,3.911244–3.965355,4.166663,3.904380–4.351502,1.06275,0.9719–1.1635,inconclusive,49/51/0,0.648887
   dense-64,cuda-cublas,5.674605,5.660838–5.701799,6.245970,5.852079–6.523479,1.10069,1.0085–1.1734,inconclusive,49/51/0,0.611009
   dense-64,torch-cuda-eager,11.829503,11.791700–11.978089,13.186150,11.773838–14.043598,1.11468,0.9614–1.2330,inconclusive,49/51/0,0.611009
   dense-256,cuda-native,6.525169,5.803021–7.099715,6.279880,6.253046–6.333868,0.96241,0.8706–1.1059,inconclusive,67/70/1,0.979365
   dense-256,cuda-cublas,8.098004,7.583664–9.648200,7.841983,7.740248–7.925288,0.96838,0.7812–1.0734,inconclusive,67/70/1,0.994421
   dense-256,torch-cuda-eager,17.773696,16.815669–20.249087,18.344075,17.836344–19.156613,1.03209,0.8700–1.1431,inconclusive,67/70/1,0.977359
   dense-1024,cuda-native,7.979628,7.909260–8.131789,8.191064,8.094971–8.249306,1.02650,0.9689–1.0766,inconclusive,70/73/1,0.928014
   dense-1024,cuda-cublas,9.142264,9.078347–9.228347,9.199725,9.109329–9.317029,1.00629,0.9763–1.0308,within 5%,70/73/1,0.928014
   dense-1024,torch-cuda-eager,17.415067,17.282888–17.498371,17.534715,17.528339–17.592651,1.00687,0.9868–1.0355,within 5%,70/73/1,0.928014
   dense-2048,cuda-native,14.939487,14.795201–15.196542,15.076313,14.841594–15.274166,1.00916,0.9735–1.0466,within 5%,70/73/1,0.871762
   dense-2048,cuda-cublas,15.813889,15.589552–16.143103,15.763234,15.628604–15.905750,0.99680,0.9599–1.0264,within 5%,70/73/1,0.871762
   dense-2048,torch-cuda-eager,23.698869,23.505097–24.276404,24.046171,23.547532–25.060578,1.01465,0.9568–1.0740,inconclusive,70/73/1,0.871762

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.044177,0.984684–1.269582,1.001512,0.998546–1.016874,0.95914,0.7683–1.0403,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,2.810224,2.679733–3.329852,2.731109,2.729404–2.735117,0.97185,0.7716–1.0456,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.243271,2.129258–2.894052,2.149955,2.124598–2.152917,0.95840,0.6754–1.0491,within 5%,5/7/0,0.0796191
   dense-256,cuda-native,0.947592,0.915763–1.011131,0.928508,0.915922–0.967511,0.97986,0.8537–1.0636,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,2.819311,2.646847–3.077898,2.880930,2.775056–3.044509,1.02186,0.8921–1.1588,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.056191,1.917993–2.409822,2.135820,1.984446–2.240715,1.03873,0.7911–1.2074,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.238280,1.224085–1.306517,1.356641,1.280091–1.497544,1.09558,0.8695–1.3717,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,3.030536,2.945783–3.167385,3.217680,2.965362–3.637247,1.06175,0.9229–1.2691,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.003415,1.928111–2.094355,2.242156,2.027974–2.600961,1.11917,0.9323–1.3541,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.535839,1.469008–1.567593,1.670551,1.554080–1.908949,1.08771,0.9872–1.3138,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,3.024857,2.944654–3.203375,3.448856,3.116245–4.023427,1.14017,0.9681–1.4108,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.142551,2.034492–2.332956,2.355818,2.082577–3.171981,1.09954,0.8813–1.5928,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.821586,0.732637–1.011873,0.708570,0.686554–0.726082,0.86244,0.6499–0.9917,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,2.712245,2.452931–3.611819,2.370928,2.287463–2.519506,0.87416,0.6199–1.0316,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.377959,2.095622–3.777839,2.138888,2.051811–2.201712,0.89946,0.5200–1.0589,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.757327,0.690966–0.951461,0.692977,0.691514–0.694163,0.91503,0.6389–1.0242,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,2.566451,2.364276–3.358207,2.400638,2.398173–2.404040,0.93539,0.7006–1.0218,within 5%,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.044604,1.859624–2.615505,1.874515,1.840013–1.885301,0.91681,0.6790–1.0388,within 5%,4/6/0,0.209854
   dense-1024,cuda-native,1.013063,0.996272–1.033354,0.999636,0.978330–1.018757,0.98675,0.9177–1.0270,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,2.675207,2.565730–2.996505,2.604455,2.583083–2.677782,0.97355,0.8512–1.0491,within 5%,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.119394,1.931060–2.582912,2.018583,2.008831–2.023041,0.95243,0.7439–1.0653,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.407590,1.336974–1.409442,1.357466,1.301961–1.396719,0.96439,0.9188–1.0468,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,2.855716,2.853080–2.911271,2.838779,2.749462–2.907357,0.99407,0.9382–1.0242,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.084807,2.053697–2.132242,2.068932,2.020733–2.106725,0.99239,0.9305–1.0378,within 5%,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.128999,1.093133–1.149412,1.009567,0.979797–1.032594,0.89421,0.8461–0.9498,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,1.163079,1.138310–1.185890,1.073600,1.040179–1.074733,0.92307,0.8654–0.9689,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.636712,2.584957–2.684674,2.492222,2.446653–2.498412,0.94520,0.8784–1.0020,within 5%,5/7/0,0.0796191
   dense-256,cuda-native,0.974251,0.959617–0.983511,0.942415,0.917953–0.943273,0.96732,0.9273–0.9974,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,0.980642,0.952458–0.998721,0.935273,0.918305–0.944341,0.95374,0.9130–0.9975,within 5%,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.105469,2.051251–2.180646,2.027782,1.976114–2.054682,0.96310,0.8979–1.0114,within 5%,4/6/0,0.209854
   dense-1024,cuda-native,1.283670,1.271408–1.295298,1.266212,1.254118–1.279381,0.98640,0.9612–1.0306,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,1.149104,1.130325–1.158005,1.124510,1.114162–1.152782,0.97860,0.9575–1.0401,within 5%,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.191905,2.163638–2.208797,2.154080,2.091395–2.265408,0.98274,0.9345–1.0723,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.595825,1.586174–1.611390,1.587031,1.582491–1.591034,0.99449,0.9782–1.0147,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,1.407275,1.380462–1.417172,1.398157,1.397832–1.398318,0.99352,0.9749–1.0222,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.225759,2.173619–2.275061,2.221999,2.211921–2.236676,0.99831,0.9462–1.0404,within 5%,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.822370,0.810353–0.832868,0.780085,0.774211–0.786675,0.94858,0.9129–0.9857,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,0.877126,0.867170–0.900608,0.855031,0.855031–0.867625,0.97481,0.9277–1.0266,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.597586,2.470286–2.885368,2.530194,2.453485–2.595603,0.97406,0.8345–1.0563,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.788716,0.756352–0.809067,0.718750,0.683180–0.754148,0.91129,0.8345–1.0031,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,0.817607,0.816785–0.820140,0.739571,0.715701–0.784063,0.90456,0.8437–0.9841,within 5%,4/6/0,0.209854
   dense-256,torch-cuda-eager,2.285092,2.222405–2.321416,2.074269,1.988693–2.170833,0.90774,0.8457–0.9888,within 5%,4/6/0,0.209854
   dense-1024,cuda-native,1.035756,1.024296–1.041694,1.019724,1.008364–1.043616,0.98452,0.9572–1.0301,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,0.891263,0.888287–0.897706,0.910766,0.883686–0.927351,1.02188,0.9728–1.0563,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,2.166253,2.094848–2.205652,2.246345,2.235781–2.301702,1.03697,1.0050–1.1122,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.455598,1.452973–1.460220,1.429555,1.417057–1.472713,0.98211,0.9573–1.0241,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,1.288269,1.288269–1.288385,1.265831,1.249911–1.284274,0.98258,0.9378–1.0143,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,2.344484,2.280553–2.389727,2.304294,2.215015–2.397918,0.98286,0.9106–1.0572,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,1.080323,0.998317–1.123514,0.964716,0.947835–0.997580,0.89299,0.8288–0.9993,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,1.072694,1.036861–1.096950,1.023499,1.000215–1.053875,0.95414,0.9011–1.0280,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.198707,2.189704–2.198707,2.059556,2.018206–2.084269,0.93671,0.8774–0.9853,within 5%,5/7/0,0.0796191
   dense-256,cuda-native,0.868017,0.868017–0.868276,0.899779,0.876784–0.987614,1.03659,0.9903–1.1528,inconclusive,4/6/0,0.209854
   dense-256,cuda-cublas,0.878098,0.869495–0.879842,0.926142,0.887748–0.952435,1.05471,0.9903–1.1234,inconclusive,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.735846,1.732201–1.741756,1.900549,1.801062–2.051969,1.09488,0.9985–1.2187,inconclusive,4/6/0,0.209854
   dense-1024,cuda-native,1.201768,1.158787–1.244525,1.263948,1.209654–1.350435,1.05174,0.9716–1.1715,inconclusive,4/6/0,0.013684
   dense-1024,cuda-cublas,1.054764,1.014149–1.105970,1.117646,1.053180–1.187466,1.05962,0.9497–1.2009,inconclusive,4/6/0,0.013684
   dense-1024,torch-cuda-eager,1.765022,1.615681–1.824547,1.947777,1.763957–2.064167,1.10354,0.9633–1.2766,inconclusive,4/6/0,0.013684
   dense-2048,cuda-native,1.568860,1.545627–1.592413,1.574775,1.532530–1.612615,1.00377,0.9535–1.0581,inconclusive,3/5/0,0.836412
   dense-2048,cuda-cublas,1.401909,1.372283–1.421055,1.372230,1.330302–1.412624,0.97883,0.9209–1.0374,within 5%,3/5/0,0.836412
   dense-2048,torch-cuda-eager,1.961140,1.959844–1.967150,1.932678,1.815764–2.043515,0.98549,0.8978–1.0761,inconclusive,3/5/0,0.836412

.. csv-table:: cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device
   :class: benchmark-results
   :header-rows: 1

   Case,Candidate,A pooled ms,A process range ms,B pooled ms,B process range ms,B/A,Saved CI95,Strict verdict,Observed I/M/R,Max TrueR/tol
   dense-64,cuda-native,0.731423,0.729778–0.732930,0.708711,0.706912–0.709467,0.96895,0.9378–1.0112,within 5%,5/7/0,0.0796191
   dense-64,cuda-cublas,0.793694,0.765811–0.812902,0.761303,0.758501–0.766990,0.95919,0.9152–1.0202,within 5%,5/7/0,0.0796191
   dense-64,torch-cuda-eager,2.161590,2.098235–2.249855,2.132350,2.090624–2.222311,0.98647,0.9238–1.0760,inconclusive,5/7/0,0.0796191
   dense-256,cuda-native,0.689377,0.682396–0.696560,0.640475,0.636805–0.642178,0.92906,0.8929–0.9734,within 5%,4/6/0,0.209854
   dense-256,cuda-cublas,0.722757,0.717894–0.732221,0.653315,0.641282–0.669409,0.90392,0.8577–0.9760,within 5%,4/6/0,0.209854
   dense-256,torch-cuda-eager,1.875619,1.833271–1.905965,1.684773,1.674403–1.698031,0.89825,0.8586–0.9488,within 5%,4/6/0,0.209854
   dense-1024,cuda-native,1.016820,1.015664–1.017135,0.966043,0.958612–0.976279,0.95006,0.9256–0.9783,within 5%,4/6/0,0.013684
   dense-1024,cuda-cublas,0.880308,0.858733–0.892875,0.832805,0.819931–0.847012,0.94604,0.9103–0.9945,within 5%,4/6/0,0.013684
   dense-1024,torch-cuda-eager,1.967815,1.830682–2.008050,1.809462,1.799338–1.812717,0.91953,0.8619–1.0124,within 5%,4/6/0,0.013684
   dense-2048,cuda-native,1.389899,1.370019–1.402781,1.397696,1.390467–1.422143,1.00561,0.9796–1.0468,within 5%,3/5/0,0.836412
   dense-2048,cuda-cublas,1.198602,1.180671–1.230419,1.220894,1.188011–1.258983,1.01860,0.9620–1.0822,inconclusive,3/5/0,0.836412
   dense-2048,torch-cuda-eager,1.902791,1.851777–1.947435,1.971213,1.971213–1.972118,1.03596,0.9634–1.1471,inconclusive,3/5/0,0.836412

Current-source framework comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are descriptive ratios of current pooled-round medians from the same
matched configuration, not strict comparisons across frameworks or scopes.
Every higher current-native median remains listed below.

.. csv-table:: All current matched scopes
   :class: benchmark-results
   :header-rows: 1

   Configuration,Native,Reference,Lower/higher/equal,Current ratio range,Same known I/M/R
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4449–0.7432,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.5689–0.7529,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3160–0.6270,True
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.4275–0.6555,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4347–0.7091,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,cuda-cublas,torch-cuda-eager,0/4/0,1.2703–1.4640,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3313–0.6561,True
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,cuda-cublas,torch-cuda-eager,0/4/0,1.1085–1.3721,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4051–0.7142,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4308–0.6292,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3083–0.6204,True
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3379–0.5493,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-native,torch-cuda-eager,4/0/0,0.4684–0.8148,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,cuda-cublas,torch-cuda-eager,4/0/0,0.4873–0.7100,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-native,torch-cuda-eager,4/0/0,0.3324–0.7091,True
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,cuda-cublas,torch-cuda-eager,4/0/0,0.3570–0.6194,True

.. csv-table:: Complete current higher-median inventory
   :class: benchmark-results
   :header-rows: 1

   Configuration,Case,Native/reference,Native pooled ms,Reference pooled ms,Ratio,Native I/M/R,Reference I/M/R
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-64,cuda-cublas/torch-cuda-eager,2.731109,2.149955,1.27031,5/7/0; 5/7/0,5/7/0; 5/7/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-256,cuda-cublas/torch-cuda-eager,2.880930,2.135820,1.34886,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-1024,cuda-cublas/torch-cuda-eager,3.217680,2.242156,1.43508,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,dense-2048,cuda-cublas/torch-cuda-eager,3.448856,2.355818,1.46397,3/5/0; 3/5/0,3/5/0; 3/5/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-64,cuda-cublas/torch-cuda-eager,2.370928,2.138888,1.10849,5/7/0; 5/7/0,5/7/0; 5/7/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-256,cuda-cublas/torch-cuda-eager,2.400638,1.874515,1.28067,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-1024,cuda-cublas/torch-cuda-eager,2.604455,2.018583,1.29024,4/6/0; 4/6/0,4/6/0; 4/6/0
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,dense-2048,cuda-cublas/torch-cuda-eager,2.838779,2.068932,1.37210,3/5/0; 3/5/0,3/5/0; 3/5/0

Source, controls and raw evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* baseline/cuda-release: compiled ``ce45ea9cc6d38cea05ef04cff8a321507e9684d5``, tree ``6cf00b82bac6f75e89093d1eb627f36d4fcc7d9b``, identity SHA256 ``e7232ead6d14f51f9f6081b982c63c3fc3c9c553876d2f4daa052ba5c8fc3d47``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``f5ae10070dd259e06e629113ac3c101d8c0e2248a2b5810239d3779cba11de25``.
* current/cuda-release: compiled ``14c37277de16182fdc7f4cfcf9193e0b352791d1``, tree ``26fcec1817f59f8dd680980fffcdf8fe0907e628``, identity SHA256 ``daf34907780c5b6fd366347ff81161904289228e9cf5960e85c0c669887e6466``.
  Loaded ``_core`` SHA256: ``84d7166a496b4190f373fbdf2ffd057af8c7ada67f53670aed9d30ba78ba764a``.
  Loaded ``_cuda`` SHA256: ``2b95332f5e7cfa622e5de9eeb109ed8d6b2798f7e9eb9f8ee7d686a9c1f71fbe``.

Manifest SHA256: ``7c5b216d188e49a1588d857478019fc3ab8a00f5b84ccb04772bff4a087554c5``.
Plan SHA256: ``3e9f579275d06bb4936f96e4759aeb120e2d7a96344b17750c1728b264887de8``.

.. csv-table:: Every full strict report and actual comparator
   :class: benchmark-results
   :header-rows: 1

   Configuration,Verdict,Report SHA256,Comparator SHA256,Comparator path
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host,inconclusive,65bc999b80e9e7ebe31af9c467d13f7ee53c70da1ecd4d126088617325df2f55,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device,inconclusive,5d3b20c295a8955639aec2a8326ac62e54efc0468192e3f3a3306eb6e048d3e1,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host,inconclusive,e5cd53aeb787ff91bbeea8425d9e86680cae2c2a95e52e05b45f65616ca0e8c2,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device,inconclusive,608799dd2f62eebfd162cb7b41949bacbe706c5e6393ac8e15e13e34eb9d8c1d,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host,inconclusive,bced81b819e8c64080f5845665813cb35551ecc640a4cf78288bdc501c048405,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device,inconclusive,77765492506166f1647f6fd1663c472712f5e6e7d055ca6f6fa76c2a9af98f5c,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host,inconclusive,c76edd07ec43f8ebf4ad7c76230d1e4b9cc5fa992c2025a42bb23a776fee816c,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device,inconclusive,39e60ef69ee649d994450c802cad2022d1799b8c4c9df244c46d07308bf8de5f,b057d071ebe47e8356a38cc2c43624ea15ff13a34abbcb02b249c83c52f23063,build/restart-optimization-20261017/launch-stages/cuda-release/tools/compare_performance.py

.. csv-table:: All original raw/import/log identities
   :class: benchmark-results
   :header-rows: 1

   Run,Raw path,Raw SHA256,Import SHA256,Log SHA256
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b01-baseline.json,3c3dbf178d7285ff9f2e5574beef9edab84a6509dd78b2998a4e13592daa3031,8ba912d0cd539dd09ad2b8aef0f0fa536bcea8b0f8b2c3396360060df0b24f9d,ff223851666e8c58c6ec634eb34525eb73a58e854dcfd688843342d19467e0bc
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b02-current.json,792c2bfe701064c217f9663d36f660b8b170b03fc59ada03f4e96c26bd389bca,95f34167b0e55426bdf18044816022cc07661186b6179e757bf712b9cd6899e2,f0123982bd310b9f251246bcf1ea1a0e22b782cf4437755f395a326767ac4382
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b03-current.json,e00ab6e4b9fb269712925e41a2ab079b2e5def5285b851fa1d3a17ef0237cbfd,b1920fc4d3aada21dcd7bd35059126365dfbd62c8bd4bd2ec943a8fa41e2bc20,fd4e51ff1593f697cda47f24454d71a516ed8114d299471d516ca70c17915e51
   cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-host-b04-baseline.json,eb177e9c7d62cde7194cdd32fd48d56467aa170be055bdae1623863642caf7c7,e8575a1ad5752af9fb65fb5a051d21dcd0d8c4e5128fa3177dcdc5eb7dfe7cf8,15a6dace9f5acbf482907686bb50cd04b75919c4913383b4e9fa2e57e45ed546
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b01-baseline.json,2985e4b99a3e87cac9c5ef97d16b24573708e41f286d0ca0c321c15c0a8b919e,420a7da2642b09b1a9d649f5ab95e73ba3a8fadbc4eda10677d4fba948c7db78,a869f9ff1d59afea852d7a050006635ea2506091b23ea243c6becb66523734a8
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b02-current.json,74709516dd6393235dd49570a3371d4e0ad17b06d5daa28be4997d1616a4b29f,e75ce35490941bd0e700ff0bb5bf4bfbc205a504c9caf0166bcf6798fe4b4002,54ba99b90217bd731714339b36eda801067ef0a99dae413be8a2862ddfc8e168
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b03-current.json,66263063c7f3a07f9c8d0be7b2ebfc9f1eecc13368a8bb28544fb6ff43bb27d0,f2cf08ddfde1bab0ce118bf50407a96d1f8486f8e0ec830a81d0ef077f0e6ec9,6b1bf3faf8e0c807918683dcc20d7719a03850a6c8aca060c10fd8c63b4bc174
   cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-none-api-fused1-device-b04-baseline.json,d2aaa200c0a193ec68f57251c734a76ed211c93dd9298e7e4ebc258a29d9a93d,4854e8d9fc93ed0f52f8caea9855efe2a0ea981e73030aaf62f95ab0346a1cd0,72a3d5968a3c55cf6f0b4756625bc2c9aeadc93fade7ad5b1eea86f9c97481d2
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b01-baseline.json,d63f0191a753ece8923c19275e20f6c0fa95ed1d20a2fe3947f67c792ccf9cb6,34391af7a81086416324e92e3e36fe7fcaaeecb49f9eaf0b70592780f4c94535,c6978225f706dcb8abb09a2a1be90fef91cd3012c88d2f872dadd8c70a119804
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b02-current.json,d41301a9ce308e24a1833e25e4a382720defa44686f01732229ed99c366df647,b577acdc9560afe5e1903564a12f7105cd66d47cf27b73f39be0c44a6f6b81a6,34b6c1ec29700b5778df54914b6eff2bfe238da5b35a85ed9dfa099088663570
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b03-current.json,fbb76498ea87c802d08fb8a57707328b97f83f734233a5a083ac86965b18d250,b439d75ca788a15b82ee1646c8d2162f6e34264399908bf6f883b4a99bd0f42c,82790a527c39956f508765062498739485dbaa19e32eccdd35523fa0f851039d
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-host-b04-baseline.json,c6e1a3111e29df24a10eeaa784cfe6c6d4ebdc26fee8b4a1ba3f16bc2996341d,bf5a23ac8d2697a20c4fa849ec4556b8d9cfca3a13d416c0314cec2585afea4a,61aecc2d6bd8ed97d73d5bd4a833e60e80f3ea8ff16435c2e13cc05a326813d4
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b01-baseline.json,92677ef560dd4f03512ea88400ea77822d1889d1bdee8818e473b23db7a2f47a,38ab51455441107d308f17b1c6669d86eb7d7c8f920267966a1b9f6a7457215c,9bf61133ebb679abdf71f4bf32cad68506ca48ddc2a4108c55e652e23bb8821a
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b02-current.json,c0630003267fbd12eeac313251fc9a81c7b96cda680612f8e8a78c9fd6a3bd70,9352bc1746a13997acc65a3b38e0c7b283f283d543fa1cd5fe8d8e049ac34cc2,b3bc2c3fa06e5d0f425bf379b3cae5369f3aba8c67ec099aecfcc4157ac3f856
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b03-current.json,33ca5eaaae4ca31bda9c632c7bea195b1a4a1998614300d0b5c441ce6f0cf861,2f0300722e0e6b42846e99e84226cb326d9dae36641ca4092bd5fc52d8f65b5a,c9b7eceda180b9d42b884f3d878e341e7de80f198ecf501dfa89b17531092a76
   cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-api-fused1-device-b04-baseline.json,1d402a18227fe597df79a28da04acd4e572944e92645d9f38d420bcf53d19d4b,4d76335599bb2839ae7bdbb2f2d9ac49aac978d8df7235bfa91da9e5f9462b70,eaed79d4ff4bcd4456bdbacd03cd66f6692e576edaaa69f17fa19a4d134e069e
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b01-baseline.json,fe20eb782832f6575e64e66ede05c7f697955cbb19cd78138d7ffe5b99fac5d5,5e750d8499a98c5810e302dfde1cde939b78df7fd86ca0205810243ec9da1dce,3e5cb9707da816e05fdf8d6034649556df81122f519b75bab9a5d2339355efed
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b02-current.json,1be1daf9f79c0006c46825ee21fc99c50cd5a54377bac0990fc4f94ce9b46f01,6f085c7357c29ca6ed1331f7b5b28add41ba163725ff6997dc01dc2f381b8118,914568b8a5bacd877242d237d6f4cd4c7995bc97f22b56592ee3f1bd1de6fbe1
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b03-current.json,576d8e855fd29ce5bea3c5e9bc603ef0693e43e2ff258ae33453d2be17bfcb0e,a4e06ebd371575d6f6f15febb1af3f589d58936e10d161d996193fed70799114,a43f4cb1beec6a3efaf3802d289944aa965f2715d317265e10675551e798ddd7
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-host-b04-baseline.json,d384ec4617bbae8fe0feb3bdc03692b121f5ce314d56225d9f4fcb8f7f6e8468,34609631ea27c535606cdcbb031874a753afe8dc9f2fde329fc1f377bef23776,9637c075f0dc923a71c0edb83773d50c86201dc5f1f3b2dcd29fb3ddeed033f8
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b01-baseline.json,bc2c2b29069d5ab7cf874d3a12b86d5a7cf0debf2d135ac461aa20979a953ff1,272be598bb7d5338a1c935e3ec2e6ff16e5fc44f5ea02d630c165896bb5c23b0,a16ba49b7753c5e29371d950d7682e631448ceb33319cbf98d178a1a26fdc457
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b02-current.json,8fac71825f6dc1b21a02672ed7c328548050d05eb511f4504de3e024a015df88,f2bd4dae06bb4de76a60ce2a3f9f2b27bcb6ad7e90d446922607e67ee20ab00a,e404edc96d4d7cc17224fb4f088e81374fdaac056f664f57ef21cb0a746ea1d5
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b03-current.json,67c3620878519c93958c997d9565f6682b68a960250296bd3dcc5f20cf2e3f54,619b6cf06ed3f39edb8c2628b8e2e111552f7f15bd6baa0bf04a4b60844b527f,e7f93d6563d06fa8382b3c2b2b4983dd637a206b4628a39527e92c4e8d22692f
   cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-session-fused1-device-b04-baseline.json,524b6ffe7f470b650d7d6514ae5bfbd3c6acca12d589e2f07f93a4bc6c35f073,0f192a74011f86604869be9c33c29fb2acd8cc62df891e9ec2f8ba4f46baa654,3592a01b52b1426ca39b8d7d5c16bbd915fb2dfd3aa9a0016e5965716451ca82
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b01-baseline.json,ed1094deb58a41b123acffce95f285dfea9180e957b0046a06d44778a51934a4,b3f4bab83f9d217c040c93f24091a6891e2690b957203765db5fcda40eea63d8,fdf43a87504f105c9d0af034034bbb176156b2bc7809ecc68c16257ede97c21a
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b02-current.json,f2189bb701d809739f75fdf70443d6f9aef8a82501950bd03fe77ff6efb2557b,8c4071d4c52acf78c6cc1ac56b3c04cce742bdc697c6f0d6e28a0634329ab91e,9f5fe106cd020842c9104cf7fdcabd8b02f5aaa4a469f78ff4304941b7963396
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b03-current.json,af1b23c346464203929a8ae275aad11c9a16785bb32248b271cfbaad28d33b1e,8313b124b3662139938030bbc9c05943a54bb39e56e05b53f03ecab8eb73d106,8398c1bb7a48ad048ed5845dfd2273527928af2ac578a56ec6c735356a281875
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-host-b04-baseline.json,1af287f0733a577b65bb3dcbae004479fc9df9e078ed335c64b97bd7b11e6cbf,5d773e7f35a79e76b4fb25a4c22685e873faa2b5731b9a224dfab5bcb679cf82,99d172ccf1ccae4c214d035a83741d1ff01d8348dd623f1b18b61b144f84a703
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b01-baseline.json,882630699b3a09739faa378e04195fa2b62b3e830f5fc2d19b4974b8129c36bb,d5fad1030e81b841b65bcad74a45919e6431190c59806a33b79d215678015969,f8a8623e144e42b79c67c05b6ca0c76e0950b6bf4fa83db65d0d0067b42666a8
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b02-current.json,14228cd467968932bacf91a97fd265f8a27a182cd589056918977c1c1471c104,628271910d851bc46a732a171c35ad0a8d978fc2a160cd03e79447edfdda1be5,d7d1d19af697150316e5d035495b2e8bfc5f4dfb25217ca3beb7e25f5bac4fba
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b03-current.json,0931d55e9d86e4d81ec06dc27be13f998cd24e641153f6fdae9cafbbfd7a7474,8ae39a79728fb5a00eb7de239ca0b6201cf71064d8c40ebd32dd0a3588aae275,a54f2ec25dbd7035b2af667adc8907fefa9d598c7be0cd877856ade8e9d870b4
   cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline,build/restart-optimization-20261017/launch-scope-abba/raw/cuda-release-cuda-dense-f64-v1-jacobi-prepared-fused1-device-b04-baseline.json,8dcdadaa3463521661d20584b2c8f66e753ecfe6ca8a3b12e09f6bdeacca7fd7,a99cb00cd91d0c27766b405021c5c36af4585f2e82c0d86c3a8006b7b40c7959,3b0fd904b82d2db3f2e4030e06d05fa42b6a35b78aa32abf7ef3abdf83873a97


Public reproduction
-------------------

Follow :doc:`../install`, :doc:`../cuda` and :doc:`methodology` with project-local
dependencies. Fetch each actual compiled source recorded above, build both
sides before quiet measurement with matching recorded compiler flags, and use
the public ``make pyprof`` configurations from their immutable plans. Preserve
all candidates, scope, M, fused flag, mode, threads, affinity and input suite.
Run A,B,B,A in fresh processes with seed20261017/warmups3/rounds36/repetitions1
and distinct outputs. Compare all blocks with the matching current-stage
``scripts/compare_performance.py --max-regression .05 --bootstrap-samples
10000 --seed 20261017``. The analysis JSON retains exact argv, source/build
controls and all original paths/hashes. Ignored helpers/raw/stages are local
evidence, not downloadable artifact links. Reproduction creates new timings.

Renderer SHA256: ``3c2d22577ae5e382eee57e1aec1e4938e12efa1eed9bf0a2d4cd57561eafa1d7``.


All 57 retained baseline/resource/launch diagnostic captures
------------------------------------------------------------

Each of the four actual sources retains every diagnostic capture separately.
No instrumented timing replaces an uninstrumented score or another capture.

Every phase retains its actual compiled source and all original captures; different source/scope timings are not pooled or subtracted.

profiles-baseline CUDA follow-up diagnostics
--------------------------------------------

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
construction is outside its solve capture; output and requested Jacobi setup
remain per-call.
No historical pinned-host-scalar allocation is claimed as a current cost.
Empty workspace kernel exports record zero instances only when the unchanged
Nsight stats log explicitly confirms no CUDA kernel data. An absent export
remains unreported; its absence is not converted into a fabricated zero.

.. csv-table:: All 17 full-capture diagnostics
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
   renderer,build/restart-optimization-20261017/render_followup_profiles.py,f74a687c7499ab92d6919368405045c780eb93974c217c57978e82e3f58bc488
   render_helpers,build/restart-optimization-20261017/render_baseline.py,af180cd5ecb8e916450024b0d5fabd0f6b7088f9f41e0e720b95ea5552154ab0
   baseline_inputs,build/restart-optimization-20261017/baseline-analysis.json,8d27f15818e58182a7c113e38024049849fc2ebc44048de4362cbd08e8385a9d

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


profiles-fresh CUDA follow-up diagnostics
-----------------------------------------

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
construction is outside its solve capture; output and requested Jacobi setup
remain per-call.
No historical pinned-host-scalar allocation is claimed as a current cost.
Empty workspace kernel exports record zero instances only when the unchanged
Nsight stats log explicitly confirms no CUDA kernel data. An absent export
remains unreported; its absence is not converted into a fabricated zero.

.. csv-table:: All 17 full-capture diagnostics
   :class: benchmark-results
   :header-rows: 1

   Capture,Instrumented wall median ms,10 outer NVTX total ms,Malloc/free,Launch APIs,GPU kernels,Kernel CSV,Memcpy/async,Stream/device sync,Gated I/M/R,Before TrueR,Before threshold
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,3.093505,33.890284,70/80,670,670,recorded,0/120,140/30,5/7/0,2.43445e-07,3.05762e-06
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,2.596909,28.483948,70/80,560,560,recorded,0/110,130/30,4/6/0,1.05477e-06,5.02623e-06
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,2.770620,31.182909,70/80,560,560,recorded,0/110,130/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,3.400825,39.899452,70/80,450,450,recorded,0/100,120/30,3/5/0,1.23878e-05,1.48107e-05
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,3.357289,36.867620,60/70,400,400,recorded,0/180,200/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,1.680262,17.124619,40/50,0,0,empty; no kernel events,0/0,20/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,1.739318,18.479692,50/60,0,0,empty; no kernel events,0/0,20/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,0.844008,11.860357,20/20,10,10,recorded,0/10,40/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-api-device-solve-jacobi-standard,1.632817,18.383212,50/50,500,500,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,3.170574,38.196746,0/0,720,720,recorded,0/210,190/20,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-session-device-solve-jacobi-standard,1.151059,13.300396,20/20,500,500,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,1.251014,14.144323,20/20,560,560,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,2.998172,35.489561,0/0,720,720,recorded,0/210,190/20,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,1.235602,16.460001,0/0,490,490,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,1.128499,14.080175,0/0,550,550,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,19.728952,206.913200,0/0,11990,11990,recorded,0/1320,1300/0,118/122/2,1.21108e-06,1.44144e-06
   cuda-native-dense-1024-prepared-device-solve-none-explicit,22.918593,223.527569,0/0,9030,9030,recorded,0/1410,1440/0,69/139/0,8.92006e-06,1.04058e-05

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
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaLaunchKernel,670
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMemcpyAsync,120
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaStreamSynchronize,140
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaLaunchKernel,450
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaFree,70
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaLaunchKernel,400
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMalloc,60
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMemcpyAsync,180
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaStreamSynchronize,200
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaFree,50
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaMalloc,40
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaStreamSynchronize,20
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaFree,60
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaMalloc,50
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
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,cuCtxSynchronize,1
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

Actual compiled source: ``e5c587889906511b0f9e74f42d88d8e857dea3d0``; tree: ``241772b9eb7a0ed25bfbf8e82bbe4118386991ea``.

Profile driver and campaign runner are different files and are independently
hashed below. Raw JSON, CSVs, logs, Nsight reports and frozen source/binaries
remain local ignored evidence; these paths are not public artifact downloads.

.. csv-table:: Source identities
   :class: benchmark-results
   :header-rows: 1

   Role,Path,SHA256
   profile_driver,build/restart-optimization-20261017/fresh-stages/cuda-release/tools/profile_solve.py,20b805f3bd95a9f0c1a5d23105c17888e4bd285b754575c1540661df87b7e0c4
   runner,build/restart-optimization-20261017/profile_current.py,eda0e6d1779698c6cc85d419c269cfb58150ef2c4cd8aad317ce066ea7f11d5e
   stage_identity,build/restart-optimization-20261017/fresh-stages/cuda-release/identity.json,aba137408ee38d8d66ed6e90c092ad5cbe0afe31193999fd19cd936fa0d23d77
   manifest,build/restart-optimization-20261017/profiles-fresh/manifest.json,74ad38680414aa28fbe09aa9f2cde6401c03b2461d98f85ab45b9a2912a01704
   plan,build/restart-optimization-20261017/profiles-fresh/plan.json,a706526d5bb51d09eb10121ab7b8b7c70183bec34a8a8e5f93f1631bf05ce3a7
   renderer,build/restart-optimization-20261017/render_followup_profiles.py,f74a687c7499ab92d6919368405045c780eb93974c217c57978e82e3f58bc488
   render_helpers,build/restart-optimization-20261017/render_baseline.py,af180cd5ecb8e916450024b0d5fabd0f6b7088f9f41e0e720b95ea5552154ab0
   baseline_inputs,build/restart-optimization-20261017/baseline-analysis.json,8d27f15818e58182a7c113e38024049849fc2ebc44048de4362cbd08e8385a9d

.. csv-table:: Every retained diagnostic artifact
   :class: benchmark-results
   :header-rows: 1

   Capture,Filename,SHA256
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard-stats.log,c7b513a79847793b1d797cc6968b0fee42984f442c65c1da17b39b9f2e26b811
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.json,e1353f93209f17d50d34f59081451646be0997219efb26ed876005ed53d170d3
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.log,90f142ce0269d4f5d9cdb3597cba088becb2b6b1c22c21534755f21b16c88e08
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.nsys-rep,2979ce9f09ef622b02d416b7a215009c83c7a45c3a1d21d32d91f5582e2d3fd6
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.sqlite,47f84a980cd781b3826ae5ccca08624113ea5f5b65f7be378a94ceef62eda0e9
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_cuda_api_sum.csv,c1d851b62123f227171e0a2daf0c6deb1c175e592ad94ba50a80852b1c635f6e
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,8d38c4ca7fae318529e1143cf14654c51467ccb137f42f195fc2549486d6d6e0
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,955272d77aeff3d7b7ab2e011d006ec269402e76b56d1f6c87a6393a0acabaf0
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard-stats.log,586bbff50cb8317bad4748d9539116fbab0256bb19efc3aebd21b93cc2d22cd7
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.json,fa44af7a1407067a01392c9f08bd7e26e33450892e03be38e233818a5a76c2d8
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.log,dacf3338ca206361623380101ebc00919476585e1a06a6e5d1ecb321dcb5cef5
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.nsys-rep,01bb9c325fa9602b822c72b87060d3bec5edd5a78b0e0bd4399f2ed0596d55c2
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.sqlite,e1e649910f6dffe176281506af41fefc3f543e63e965b0bdde07d8cfc6518317
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_cuda_api_sum.csv,9620ee59814a88e99203c3563726fb3e12b76f38662f88e10be95a42100e948c
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,ce8fddf8c2fe33e3d796d54473200d0b356b990397e3ce5ddeca8a1d7392ed2f
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,0b629918f09de93f3c47ddb38030480684cb3d72fe77a306658ba28b7ba5c345
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard-stats.log,51ad1141d1d7758155f6ae58cd041eac60bf488dcc912aebf8d07ddc7314ec87
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.json,b1b980aaddffcf243e7fb5ce4f3a088de2da7db4ccc27650a73c081848812544
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.log,62b40fc20c5cf2876e7b224895d3886f8f6bf376113528691b86b616b8882fb5
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.nsys-rep,db03b8d8c56affac0a21ff4103c67f7a18c2461d87e9420a5e58e37a9ce3b435
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.sqlite,a2af36e607925d26b7cb43019b4b27c272230c4e3ccb09aad7403f59ecad7138
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,94ef7eb020c95c1e2cb3f4731bd51909f90a6711907ea4198b6e933146078325
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,6f9180f87a795a8c627038c995cf0fa33ed21b16a0fc07042687e1f776490489
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,72df202e7ae289dd9bd941e2bec95359bd28bbc73cf7258b21fd865c38d801d9
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard-stats.log,007a80e01064fa41f83d773e18f1a428ec150d0f005763d20d9f4a99e6eea7fb
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.json,4351ccdfde96b7791301f9b9df22f235984c40f8570aee0cc9bb2fe2cf616fd6
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.log,830090ebfdcbce1402b91a653574e7dcb56823f0a0f5beebd683e9090cdc3694
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.nsys-rep,4ec7b256db4cbff4c6fc753cd3b3f0c7d9fbc9722ea27c3514d4ed2f8a31fa53
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.sqlite,e782109e19fc49837d8a08be76ab0e5df10f60ccbffa96238cf8306451129452
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_cuda_api_sum.csv,c303f7e67979c654618e6888636dac4b38acc20c21c9111153203126441237c4
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,36a2b599df06d5a3aa0c1db337e434b14327140e990a2470af42608e8a07cfe3
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,d179577c62963707261a9a1d0bb2c30e4246220e147a734b1ce58fd620ea626a
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard-stats.log,b7932c28076bcae8cc07df19b743f68d0799fc909ae227d41d65f7b15f73e4d0
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.json,b30dfaa223e553dda7a1c0ff0d2040c6b67f1840bbe5456fe3bef3d14b8b229f
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.log,63d73fca767e71591454e3499eac31e2c470507b98db4c752f380e8ae87b0072
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.nsys-rep,cbbc262bcfb590fe8833808e10c8ae881bd435bc760b0e4e3e6e2166813f5aa6
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.sqlite,4bb92d0b2e6483444903cde46fc577f738069069357e88f23b6ea46e8c083945
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_cuda_api_sum.csv,cae559c39be6ecb50fc1810aeebde3797a522237cf316b788ce2b6682857d1aa
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_cuda_gpu_kern_sum.csv,d6bdf7bb05b1c0316e51ab7a4740b315ed8323d3ec8f4eb6a5e04f7585895eb8
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_nvtx_pushpop_sum.csv,5446a582de06025ee2705fb9db2f0f43303e131e63c95ee8e15a7d16a70d6bcc
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard-stats.log,d8f6f1a0070cd2ff584b2c0894a260c64e47f6419c61f5cbe22c97473b30b89f
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.json,7889a42df83e8f3f74383bc3e8d5226e5566f27787ac29c74bb0f0fc16ca441a
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.log,c6734f036d3a3dc8dbe334875c82103ba4473efb3498f6b4360c30819f29ef29
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.nsys-rep,aa3cab5d96420c0d051e0eaa8adc6b404b9cf951730efb3fd1cbc938990959e7
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.sqlite,51decaaeeb35d7102a3b580dfeb8ff63719ee7e889598692510fee291d6eeb8b
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_cuda_api_sum.csv,7a50922cedd89c941f2a86e8f42cc59dc9aa30d518a26433901b5a7bcd86a61a
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_cuda_gpu_kern_sum.csv,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_nvtx_pushpop_sum.csv,70e6d335c243f7dc92623954bd8c666679542bc346fe644544f0cc2c2a5959eb
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard-stats.log,2babd65b794c2a71b3c753d47d480d614a379983958de1c79166f68b25f0947d
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.json,604104a76814d616ca61e4a9dd55f34b1dfd254842f26fe6f6dee16a7cfbe535
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.log,6721ec27984551316a0262bfb97a97d015138c65f0c5ad07d6e09ec4719e58f6
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.nsys-rep,8819aad0fcf6ed93124bf9f5529d55708305cfee71baa725b6f102f226c7a339
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.sqlite,92af46f73ff8106741f1832baa737768586cbbc14aac4dbb933cdced32113378
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_cuda_api_sum.csv,b9a3af5102ed361c4bb78d8664258ca1eb1747433e5f49e32ab04a50c9f2f2a6
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_cuda_gpu_kern_sum.csv,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_nvtx_pushpop_sum.csv,14b09c7ad9774e68f6bfed0220fca824be02d798b2e2ce3f3835d656a2313964
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard-stats.log,2d211975b8f0030431ad421f6331ea9a60c0038a471e807f617ebde8778e2fe2
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.json,0cd0625ff8a03dbdc2c10f37fd1eabe5b1283b924729559ce23a20f06c89c4f3
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.log,83a4d2a00b62477c03af9450d58e992dbd92589b03c38978ff6657c22d229959
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.nsys-rep,8b51778df50d0e57fc6399704cc7fdcd6e21ebfa6980090fbd2965a0132fe7d4
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.sqlite,763ecdf08aaaf27b297a517317da5c38e07f46f19568b7db8a006667378f0fbb
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_cuda_api_sum.csv,dbb12a5deb8adc64a9a7d9337d1b7ac48870f3a50ca20b0d2c638eb1e8367e3a
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_cuda_gpu_kern_sum.csv,972a61f520ecc5414490d595fc9eb306e4a753648967c8336be3df5cdd5fc5eb
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_nvtx_pushpop_sum.csv,b38f1c3d4b971accd4ef760dc4b64bce0d858da6e86bd5248ca3ec1e2e285574
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard-stats.log,c37f80ec205c0be2e0c82c7efa7f0d588812c30f9670fc5c1291f58764679b83
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.json,fd08e6ab98ce4478d6596ceca7c1c1997e392a54df364f0d1f9bfa280b0f5010
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.log,3b0ed3b1c033eb34cb0fccd6c2610334247dbd08a9c93972597435c2ac2693fa
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.nsys-rep,a6da292558c8331928cd5ed55c1457bca1b96e124dcc3610a6d981a8f465b3ef
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.sqlite,af1ffea242dc65ebe108ce01c20dd981afdbd40bc3805efc2405612db2128636
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,3a3fd08f48ae0993247c6f5b7bc9ad843a404e3743b9b54b791e6573b244110b
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,3d5bb1a2d7c75a5ca30315ba0c64e7b9566755d7cfbd610be0b963d888f734c2
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,511c012e638771b149acae2ccbbcc53aab03ede1d2e2db759cbe935770b1230a
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard-stats.log,920b3293c662c855f99583a978ce39dfc7e7e9a934721ae02cfed6fec1303be2
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.json,98dc5a4b496d3b93a2ecfc033718794a16c2dec4c6ebbcc28f16783dd66f16c0
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.log,9510ce599233e525390b3114c6882cc29b87bca239e6606a266db9923d7c82ac
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.nsys-rep,ad0d185c54956a5b873b4fbfa74fd78e36af4e45b90cbf2dbbc1cf0a9acf1a77
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.sqlite,f3b3066601854906aefc971396e44f5fd96279471bd5261aa2b05336b4be2744
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,68e282edbfaac2e09b70aa36e5fc927e2ffae71983051455ed57b7308f150f92
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,70174bb5df26af2ab02ba21d089f162c7da28bb9d2aaa839d0edaac7a96b3f1f
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,aef7896336f3066c6d5f65a49f661a3d8e65ec11e7604132e0b2cad590781ade
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard-stats.log,8c71fe02e1e0f57d0627687553e05c23b28e654fd94a44375f76a93af03c31d7
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.json,8f6a65813cf25685bed50c2efd21a8a5645982e71d0c954c89f1041541fbac32
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.log,d08c57695ca52126553162a888cc95865d9569ce9aef0bbc1ec410b13da1a314
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.nsys-rep,428638aa1c0b170bd7200d7742516c51bc0bd5a31cd69ea62566d8c16035d221
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.sqlite,17642be722aaad33b15e87656a56270689e2dcf0c3730f9c80d08bfc81bc4a01
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,4349a70b8690e641b89f29aae03aaf7110d08b6dcb5eedb48a0cd8f68cf7d5f8
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,4dbbc6db38d99fde533bbf6b771aa26542d3348f263319a557434d533b64edf1
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,8ff6007f0d9b1585b83d2c6f86b1b1e330e59f78e5ba77fcc06c35dd1554a1a7
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard-stats.log,5b1ed58e5479c792c81ed7fceedc98bfc983e01acb764566e7cbe33038358164
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.json,9e4ff34dbb38c11f74810f26c1db9c1d59c37f2efc2a2656870bcf105b636515
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.log,89a9d0269499d4ce2643339a9fcfdd80867b6f7c84040d491084b11b9dc9eb0d
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.nsys-rep,ee7a0ee30d5986812bd0becaeacaf2636fe9684fa4d246ddb32138f83d5f9e1b
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.sqlite,5aaf1d1bfcfae4acd70e4fed28be48538daeeff268a5d07cb8e96df418484fa6
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,67cb09c648f5378bd0d5484a0acf77b81110bdfc427bd7b2c4381d44a2db2c77
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,99c4a46f8e082d4727a761aabc0338dd53f22a44384c71a0a8b0d732e107ad87
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,254f7dc5cde5bfb87bcc50f6c2f980150d8c17055053e1de869dad520b272b48
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard-stats.log,478602accfb5e3acac0a21bdf44d5919a04c2830f92140a940022e8f72065538
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.json,a558c12d4a266b0e1df6475b5e90d0d1edde4d00af3429ecf8b4ccd31e21c862
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.log,47ebd31dafb0e3c23a660ea04d2621169d777030713fa956e4d47f8a9623fe32
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.nsys-rep,aca6e7b3c1dd0682427dbb04ac385eac2d1ff6185ac0cebf36e3827ea9ad67ea
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.sqlite,9eb95232204065d0cea24683c45e302e0ba628543ce117bfcc56ea66482c0b85
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,649d27422871e4e2fb30045b6395ea70862c2fe276b35603fa5213dd5b6b1f94
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,e12a76c459e5fe77205152ca96e81eb6f7ada5a6ae92f334108e7637728dda46
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,916115625a23a4b3c7e0edce5e715ec988c0c58652e7afe5ce16e0c7f1147d4d
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard-stats.log,4476c3e2544b2659188fd2cfe1fab1e9d2a2a7f1313cfd2674b8fda4dcd5885c
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.json,974ec7312921c284bbb25aa9d1107e0f9400ad0c8a93613eab4bb9437f74d22a
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.log,169b0d5e7a151e6bbd9f9b95500873b0feeab03b3ab92d6fd17815d0666e50b4
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,8faab530e73ec886fd960ff9575b7bc833290ef7103ccbb279a8629c66eab9a6
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.sqlite,cba80202817dd6415f6b053afc4eb29d15ab48dd0973fc32a68a55322d653c7e
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,f988bf95d97d874b33593b5ce4084c86e12563a83347213e02d187d28412f5c0
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,65351ffb7b5b5710b959b209c2cce91768026dd04312f2774ab88401586685a2
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,7d220bbb5d3dc5961a2ef2dd96d7a927ca0f05645501434e9491c8462c847c4e
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard-stats.log,06377512679ed27d927eb7fb96ade7b77b108ee33c99cdd6ec4037f0d2cf5965
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.json,e0782f94d6aa284e28459abebc8f63b1cbed64d31ca297a6b293426251b039f8
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.log,dc20e7947062cd5a6109ff5338403783c994c5db599d14f2e808c2a30550183b
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,e21536b816428f48b81a908b2e3657a7b6e45e2385793f8acb8a6b1a707be37d
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.sqlite,cfee4ae9528f3185d79051263671bf07f61f25ec39609f96c81d12baf05ffa9e
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,375b9c001992b9e3e9397e11b6c618245d893e0c8149b3e408e2338fcc573a2e
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,91cc2550486a1d8357bc5ace636583e21854791c94ea0b39a16d36429e06ea85
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,b0cef57a990f1f76cf620258f0d0723f90276e5a3d5315a516565d0e107e13c5
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard-stats.log,51672d78db947521d4b98dfa03f3ad9e7c1c984b1133eb7f8e5fd5449993a8a6
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.json,18c69f384560ff45f79b59917d3c6a73c360c3137bd601f677bcdb8cf1ce1a4a
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.log,e2028edf00d97f7911164c2c6b109b68c14c6c539ebf43bb289e35cc206eca39
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.nsys-rep,240ee404b72a594da95e135783d633dac20b6ffe405e888930bbb04e3209118f
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.sqlite,12af363875dd84cc3025c51662b42a5a9f9cc128e81c83da50c6d0dd03087c3d
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,0406e509d65fe966564df8efe011f3497a3f3e17b3530ca8cc9d1e968b6608e6
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,fb463cdafb87b1dbd1e6c6082912268ec6c770f15f7dacbece545001e209cf96
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,362e23b53d17ad9a421b21258c9990cc733c3beb8fef88f5dbae914abca2b96f
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit-stats.log,74d044de916356ff92b393263a91738c06eaaec26a2031394ba5fedc7c0ae652
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.json,701dfa7df7dd0ed652c353ec80b3dd9d555cb9cc0910bc66f2a098786ee09cb0
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.log,029335c72f8eae62ca0744265687074e3bbcf08a043482b83ef9c2734e34d4d8
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.nsys-rep,b1a8de18ff6d276d6bbb84a41944c4299754165ac7d1140a0cf45b7b9ea94e41
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.sqlite,4964be50aa90f8b90c40faa74d547e0702b70127391689e31ed7a3cca70f6d08
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_api_sum.csv,c3dc69f6914c2315687c503e932eb078435cb8815f116efe721ecc962d1900e0
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_gpu_kern_sum.csv,8c6efe794252bc207ea8feae008c7e46e0215617d535b909dfbf9b3d94f30c23
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_nvtx_pushpop_sum.csv,498d2e6bbc78e3eea8d517ab3e80603728d64c5f3cae3d0791db095a709cfafd

To reproduce a capture, use the public source-install and project-local CUDA
instructions in :doc:`../install` and :doc:`../cuda`, build the recorded source,
then run the matching public ``scripts/profile_solve.py`` inside Nsight Systems
with capture-range=cudaProfilerApi and cuda,cublas,nvtx tracing. The ignored
orchestrator is not needed. Nsight installation, WSL permissions and GPU
availability are external prerequisites. Exact frozen command argv and all
original summary CSV rows are retained in the analysis JSON.


profiles-fresh-span CUDA follow-up diagnostics
----------------------------------------------

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
construction is outside its solve capture; output and requested Jacobi setup
remain per-call.
No historical pinned-host-scalar allocation is claimed as a current cost.
Empty workspace kernel exports record zero instances only when the unchanged
Nsight stats log explicitly confirms no CUDA kernel data. An absent export
remains unreported; its absence is not converted into a fabricated zero.

This diagnostic variant was selected after the initial rejected resource
trial. Both original and follow-up captures remain separately retained.
Counts can verify a changed allocation/launch inventory under the same
capture protocol; selected instrumented timings do not prove that layout
caused an earlier regression or establish an independent speedup.

.. csv-table:: All 17 full-capture diagnostics
   :class: benchmark-results
   :header-rows: 1

   Capture,Instrumented wall median ms,10 outer NVTX total ms,Malloc/free,Launch APIs,GPU kernels,Kernel CSV,Memcpy/async,Stream/device sync,Gated I/M/R,Before TrueR,Before threshold
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,5.360959,61.858510,70/80,670,670,recorded,0/120,140/30,5/7/0,2.43445e-07,3.05762e-06
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,2.914120,31.078624,70/80,560,560,recorded,0/110,130/30,4/6/0,1.05477e-06,5.02623e-06
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,2.944275,36.146044,70/80,560,560,recorded,0/110,130/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,6.407979,65.298137,70/80,450,450,recorded,0/100,120/30,3/5/0,1.23878e-05,1.48107e-05
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,3.474406,38.588699,60/70,400,400,recorded,0/180,200/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,1.789818,18.873907,40/50,0,0,empty; no kernel events,0/0,20/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,1.668506,19.023418,50/60,0,0,empty; no kernel events,0/0,20/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,2.750163,30.154661,20/20,10,10,recorded,0/10,40/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-api-device-solve-jacobi-standard,1.210140,16.928656,50/50,500,500,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,3.483608,39.227683,0/0,720,720,recorded,0/210,190/20,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-session-device-solve-jacobi-standard,1.176751,16.560042,20/20,500,500,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,1.157695,15.934467,20/20,560,560,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,2.897893,38.092489,0/0,720,720,recorded,0/210,190/20,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,1.110623,15.968249,0/0,490,490,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,1.062854,14.052306,0/0,550,550,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,18.128611,174.960432,0/0,11990,11990,recorded,0/1320,1300/0,118/122/2,1.21108e-06,1.44144e-06
   cuda-native-dense-1024-prepared-device-solve-none-explicit,22.997159,229.690452,0/0,9030,9030,recorded,0/1410,1440/0,69/139/0,8.92006e-06,1.04058e-05

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
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaLaunchKernel,670
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMemcpyAsync,120
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cudaStreamSynchronize,140
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaLaunchKernel,560
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaLaunchKernel,450
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaFree,70
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaLaunchKernel,400
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMalloc,60
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMemcpyAsync,180
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cudaStreamSynchronize,200
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaFree,50
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaMalloc,40
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cudaStreamSynchronize,20
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaFree,60
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cudaMalloc,50
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
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuCtxSynchronize,1
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

Actual compiled source: ``ce45ea9cc6d38cea05ef04cff8a321507e9684d5``; tree: ``6cf00b82bac6f75e89093d1eb627f36d4fcc7d9b``.

Profile driver and campaign runner are different files and are independently
hashed below. Raw JSON, CSVs, logs, Nsight reports and frozen source/binaries
remain local ignored evidence; these paths are not public artifact downloads.

.. csv-table:: Source identities
   :class: benchmark-results
   :header-rows: 1

   Role,Path,SHA256
   profile_driver,build/restart-optimization-20261017/fresh-span-stages/cuda-release/tools/profile_solve.py,20b805f3bd95a9f0c1a5d23105c17888e4bd285b754575c1540661df87b7e0c4
   runner,build/restart-optimization-20261017/profile_current.py,eda0e6d1779698c6cc85d419c269cfb58150ef2c4cd8aad317ce066ea7f11d5e
   stage_identity,build/restart-optimization-20261017/fresh-span-stages/cuda-release/identity.json,e7232ead6d14f51f9f6081b982c63c3fc3c9c553876d2f4daa052ba5c8fc3d47
   manifest,build/restart-optimization-20261017/profiles-fresh-span/manifest.json,4f9a131d2fca093c80d105dbe3994c9acb3d283a1483f566f2d4c8aead3ceba0
   plan,build/restart-optimization-20261017/profiles-fresh-span/plan.json,7af94555376b7bccafc077f5402d67c82b683ddf186914cc712fcc5bb223d8a6
   renderer,build/restart-optimization-20261017/render_followup_profiles.py,f74a687c7499ab92d6919368405045c780eb93974c217c57978e82e3f58bc488
   render_helpers,build/restart-optimization-20261017/render_baseline.py,af180cd5ecb8e916450024b0d5fabd0f6b7088f9f41e0e720b95ea5552154ab0
   baseline_inputs,build/restart-optimization-20261017/baseline-analysis.json,8d27f15818e58182a7c113e38024049849fc2ebc44048de4362cbd08e8385a9d

.. csv-table:: Every retained diagnostic artifact
   :class: benchmark-results
   :header-rows: 1

   Capture,Filename,SHA256
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard-stats.log,758d3c3f59d3db8e3e1d5582814c0bc277cee35709faf670c566b8a0618b02b2
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.json,408174c01f7a68c44ee5fefd2aa212fff86a5a1929a1d9fb02facd2c57a33f55
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.log,1266bf83ba74614a68d74e54219a2882a69ccf4a342a55dbdc5314c42dee4ab0
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.nsys-rep,8447995f24eb39302d11910f0ec822479b48455e2d11f2853f279d721a231369
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard.sqlite,401e48c69dd125d21288c7b11af2420458a22b0b89b27532c9fcd54d92bf1cd4
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_cuda_api_sum.csv,beac281a113a112a04753c219dbe37b428ff59072442b3dcfb65d8cba05b25a0
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,6dddeec49e19d386bb1b7b81495e3fa793a3d7204184448ef14bd6707e928c14
   cuda-cublas-dense-64-api-device-solve-jacobi-standard,cuda-cublas-dense-64-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,0936df7a07fc0fb40db406a08093849642d33c56de9b1a89bf271e20eb615f0a
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard-stats.log,fa1bc7f2c2ccb618e45f0001ff407758df70a632963a15bde47e5c174a43d3cf
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.json,1a88750b26515f4878aa05132a93a915e6713a04321fedaba571228f20473fa5
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.log,5ef79c12456abc26f342165f545278d7ebc9bc64854ec2c44bd2f298eafce79b
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.nsys-rep,bf55f5648e4cd170ec01bb4a75b2e3038ff3987939aff5a9a9a4cea03a5b4ed6
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard.sqlite,d95013a249da1f05a3329aac00c710e9710fc01002464e6f01a8ca8cf592b0d5
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_cuda_api_sum.csv,5b2a2785a2bdd2f83ab1ef641fabf94f91a26a370f797889630d1c716d3efca3
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,80589f2d823af4f38cbd4f82e39173d1b6d77c917f948599a0d935a28aab2a8f
   cuda-cublas-dense-256-api-device-solve-jacobi-standard,cuda-cublas-dense-256-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,237541be06f1c720a3fb60429dda6090afd0193df176dd3c4f36d856cce87b96
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard-stats.log,f3ff255df13d03e11dd96b70ce6fa2521f2ba25ad7d8039164f9563e8ddd7cf9
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.json,527aedb31eb333dfd1942f2b3f55008f1b1419ee227ec386caee0e95db54f94e
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.log,12f1e5a3c3b201cc72bca18f50e8db0ded722c6c686404f267bf24f8d2929960
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.nsys-rep,ba4ebeae1314fb20d4e483c0101e4f006174862454adf42d1bc732a51c87e777
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.sqlite,4b61928d4fec91bc3c978b7dcb1d5a2b41f869c6dd0ef652195db41b5d72f10c
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,f0da5efc407332c53de1e115ae4a4cf9e0f90021487e8c1b4cab60a3032a1eb8
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,ed8e5fdef85217cbe2bf9c962bf2eeb55981cc32b9811c1647e4e085290eab33
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,564c0effd1789448241194ae205852bccac304b75613c5be773e5404e2b0f105
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard-stats.log,080ef485d8c24aaf2e312375a23b224fedf28738afc51dee31efda6fa5f270c2
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.json,44d96ea6d59d32f9ffc3050d7c97003c25cd92cbe6bedf1cfc5b2105033c44fd
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.log,989ac9c9b80c4546130e62a9401270afea5a3eadad1c832de48b7f47708b7f0b
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.nsys-rep,93a8cf0100612108d451e3d42078e56eab46fd4ed9aaad32de4312522dc24682
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard.sqlite,b0e3ff8126e19f60be7cc3fa0cebe5ace0e72920d1d2b4edfbfd31a54b1fcaec
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_cuda_api_sum.csv,b2df62c9c21e63bd93e9ffe57b3b37bed4e231b1a97590cd3d986b7ca76b5387
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,3fbc819100cf980dcbdc2c837b0c1c1f5aead95cb20ae11f0bc728fe32083616
   cuda-cublas-dense-2048-api-device-solve-jacobi-standard,cuda-cublas-dense-2048-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,d62b5292caffb4cd24139bebf2785526e3ea9eb33880f1834023956f67301a79
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard-stats.log,b9fc3d35ab6535486dbe69e3494ba18aa622929b06fb0aac8a84b5d6dccf82c4
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.json,fddd8e9e4e7ee3c241bcdf3caf13dfb5f02109fc31cc9a22e9cc98ecd56c8f81
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.log,d9efd1cba2c0c67f17fc0f6c3a595b4c4f2291b0c5d29add4873b2531171ecbf
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.nsys-rep,0115b102f5748771b45314153dfd69b11fbeeab7ee9a7d10ec4354415b8070f7
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard.sqlite,92af8e8df7195d7d29ef360545b490db6322954baf0808b6b58ab99b415833f0
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_cuda_api_sum.csv,5c7b67b293fde244524ce33af7719fdbba694446f7de5126b07e65411ed4ae77
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_cuda_gpu_kern_sum.csv,5b39440121440b99665817a19713ed6a64a96ef1579bf7528b5b94816bb1fc66
   cuda-cublas-dense-1024-api-host-solve-jacobi-standard,cuda-cublas-dense-1024-api-host-solve-jacobi-standard_nvtx_pushpop_sum.csv,4f9c56693f5cc9f33398e574c9f2083136aadeba3b2298855e4cfa3f03f8d61a
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard-stats.log,fab63d170518d364150d077f9a6dac3cbcc02d4de3bbeb9a26ead43795a51ae0
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.json,e6823d63cdacf5fd7e14eeda0c05106726651c42a9c2839cd63242362004f55c
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.log,d5cc871e98b06b3aa6ec81f22eecd4dbcfc94c0e9b24efd961cd80b0a4d5d45a
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.nsys-rep,3cf0c5ba47d0da4fe335c53311704c1f2cc132708fa71a35a62a211b37b198fe
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard.sqlite,f23b8a71e4e87e34aff0824af9b94bda25e3fcbae4a4d9544be9a265429d66ee
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_cuda_api_sum.csv,9af0fe5bf2556ecf0596e7c736865718892e8e5826405ec6201b9cee1637e6ee
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_cuda_gpu_kern_sum.csv,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   cuda-cublas-dense-1024-api-host-workspace-jacobi-standard,cuda-cublas-dense-1024-api-host-workspace-jacobi-standard_nvtx_pushpop_sum.csv,3b05a52e726fbe0e7144b1d224c8c7880d9629c05166ba095f009b143539a25a
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard-stats.log,67fc1619568208fd14b6ce494a149a217500d647672877a7a80e40fc6ba5ecd6
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.json,828f44ab8c0dcf43c369f796d6628f7dfac2f02fe910f273ccc4ad37deade42b
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.log,da0dd97d1232641d0fa3c0ac0c179ed78b7518a40a201fcd5590d24a99a0c46c
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.nsys-rep,7ec549f7345edf8a0dca3d1f139e5f68d5517cb91d1d9b4e12705562bf5a542e
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard.sqlite,00c50b8246940d17e404a828a99cdfde645e6ab97e3ec6399dd25e85aee33377
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_cuda_api_sum.csv,dc62e1dc3e53a7c196ad70ebfe4c85eb1aa2c4e0a457784b96e025d4797e8030
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_cuda_gpu_kern_sum.csv,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   cuda-cublas-dense-1024-api-device-workspace-jacobi-standard,cuda-cublas-dense-1024-api-device-workspace-jacobi-standard_nvtx_pushpop_sum.csv,f550fdfadc398e9a2c933e72306acb9a01ab1a77a579e8a0a1062d8562fe219c
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard-stats.log,2527a2cb54d8b15b68c9271990916d6f6f0d9f239b898efdd4ecf3d41315aa35
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.json,cf05c882fc7d33bd140f8fd9339ab16b9d7ef5aa76e4e957341a99f0903eee1b
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.log,0209876c599e6068d38b42b0ea730526b22a84c0369229fef3316ebe4f784f27
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.nsys-rep,b8f20d293b9aa1327e312b09cb969277b979c08b4139c721bf2464d54cb0317f
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard.sqlite,69efbbfe95d44cf59fdb0251ae8d932fe4f068e6099b37e6aa3cce4484e34ab2
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_cuda_api_sum.csv,6f08b3d8becb17bd6c42d160e69f0ab94dc0d3573020ca2e3f303c071cfe07d6
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_cuda_gpu_kern_sum.csv,ad0cef3934d7aa2621f290b963cc1934f8239dda189454ffee834dac6d5e8f76
   cuda-cublas-dense-1024-api-device-prepare-jacobi-standard,cuda-cublas-dense-1024-api-device-prepare-jacobi-standard_nvtx_pushpop_sum.csv,df2ee411e79a9495655c9643e65b3cdb80c0f3c1f5226bbeefd62a1f86f2117e
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard-stats.log,171485c190eee0372a94df0fd8ffd22f5c4e6b859ec56210f2efc1f166806985
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.json,0d822aecd29cb023bc0e7f002faf7549febb79cf782ed881904e9c71bfaa06bb
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.log,a3485b4777285cc98222a31bb53569b27fff49a3dd237f4594c4bc7bd40774ff
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.nsys-rep,9c96ac8fe2bf3a5c37de64a07e9308a331dc09be2050beb3a64c0520088b0b3b
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard.sqlite,61ab856778645ac8707c1e66fd4f6a48f01c7043363bae0a0908761aa9f5c6f7
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,a8b269499fb16a3098f62bb6ea621f02548114597e55aaeb063734d2bbf393e9
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,1ebaee2f1fccc43ee1ce4c4f3563268fd2fe20abf7de699c57f31a041c52fe84
   cuda-native-dense-1024-api-device-solve-jacobi-standard,cuda-native-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,eff0b7d19f70670f2e9161f205fab4c52051b18185731d391908d89333fe0702
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard-stats.log,972e7aa7294623d33615b83ed99f910970905ee63ce32fafea46ea2164ab9baa
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.json,7cb23b96c4c1f1b09dbfb66015e497b4c8f9341d88c3ea273568d4c31795c2c5
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.log,fe860cb8a0557933de6db3e7fc9879e8dc656e9b17c020bff0ad40f4a8f7e504
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.nsys-rep,cced1981dd2b5447a81437e47e5c0a7f09d950b7776ab71b0e3f99a151f1691c
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard.sqlite,8c82fc90afb52b2ed32daf60dc4fadb3ad1a78c4dc1521164d4217678ebdd6f2
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,c7de15649e47245a171b7235a13608f94d20b64b2bb21e8c4eeb8d6e79785646
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,bd85223965b7b69e12311bd486bf1286a31a02a38219ba242d5ba342fa2b87b2
   torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,8a153e8743f92c82809458399dcb1e0ed29a595eb64d85e72baf59caa76f709d
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard-stats.log,73dad88faf66da19e51bd5edff6b442a3909bc99e4e090e006338e641fa7feeb
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.json,34c1cd29e4e7a2b489382b7e843991628c7f5f9da74db3b6a3da95e182039866
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.log,986e9d81aa25e9907fd1847aebd97ed92ee67a74e1b75637ecccbe2d2ccfb79f
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.nsys-rep,62211563dea0f28c661680435c033f39787aff86dd634c60198fe2e961c71c96
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard.sqlite,3acf0b5c56da5a91f80a3fa8dedfbe7036b2201c08480c82bd5c4e26629a9210
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,e2cbd15b74fcc1e9f9faf9e3ff3f987c78d0044e03abf2c3ef8c9416cb8c619a
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,1f45588ed00eb5e9b9c522629cbd3528c155fd0807879d533a16592bf6888604
   cuda-native-dense-1024-session-device-solve-jacobi-standard,cuda-native-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,b71600c7cfcb87eb00879dce0f779174b444ff68694a24433a50d7067c86c246
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard-stats.log,c982ab592cf8a1dda9a0359dbe9403bd81bbe82e2aa39c02328829dee96b1cd5
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.json,77f7bb196cd11219478485b28049978e2bb0a1e000bd705028e724495ec28e4b
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.log,4c8489998c4a3cc2a2b78b45a3b000de8b1fe76b0ec53f2500f69ecece173120
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.nsys-rep,ed65502a17777261e9ecaecccb9afe35a99bdc9f6b9d94edf9a64868df72058e
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.sqlite,2bc68327b6e141b3f80da9311d198fc5f8f06f5a08473142c33d624f9ad4684c
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,b9e7b8b8bff1e81f352de3b37a132018666398a318cc9a26b0f253a817e3e386
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,198a43ad570caaa658a107dd0ceffb1ae1cd1990db4b21a79d3cc7efa88f0934
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,d6901925f7428a1c143438fc081d3c88bb4552ede40bdcfd8ad687ddc06948d6
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard-stats.log,17cb2aa0f8001047082280d84230c8b6f3c3d599f988f2abbd084ab0dcf88a95
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.json,7e9a5fb9e8a089a291860518af68cdc491eafebf33c6735bf8138892204cc1ff
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.log,b4ac1aac605d473f336ac8fb1eb9a03273eb4b4ae34e5345454c6c123c273b2b
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.nsys-rep,76c690110e6ba1fe7b40573e2344509ac2d3f4e1f70b05fce40380e032ad602b
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard.sqlite,261914f3e78a12e19938ad6149788dfeabfa202efd018cf4eeb7a8a8a5e73f1b
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,977d86364623b01ac0e7da263c09f905d7005f415d927709a207b48b30f1c4bb
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,89184d98107510d66800295591d864fa4396826457f09632cecd9bd9e6bb8589
   torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard,torch-cuda-eager-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,252232c20746b5accacf293d9bc73df89e284c8659ea1e432053393e45cf2506
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard-stats.log,ca63f06a237407de6b479594e6f37526351384b0eee9dddc4b57aa5ad27c90d9
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.json,4582a86dd48df5913461b1667db80176367188f427ec8fb95231e0d6dc11d3cc
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.log,7b6050cfffa3d131c6ce77230cc681c9a9dd9b7717faa9f8979df70253134e9e
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,419303509fc71ce45d9c9b2efef33e14b561052a619f658e8f968a18b8d21d76
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.sqlite,5fddf5116938999b57b863bbb6313d3bf3808072344ebaccbcfbb40fc2f0fdc2
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,a9fca7ba4119a88d5dc98461bd9bdc1d133ac0700547063979457e8a62a9e663
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,8d18f652cf477cf81149f33cd25d9f2ae320afc2650126f62a759673b584cec0
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,d7569ebc847193be2254a7d7f5c16fd847a18b0d64e6f2481a59a278b730520a
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard-stats.log,71528dec7b0667cf80b5d6cdf8ca08ce57bc1085b13451f262063cb9a33c4451
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.json,4b0d59cf86bf41b000e28c4f3578289be32889f4dd896c7e424001e65ef1fb69
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.log,ea95168401233c8ec19c5e428946ffe89f69552d205970509025aa8ac2d5363f
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,7fc5f33151d9245bba16c5f3f5c62549760489b4c861fa56a62a71447edf0d53
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.sqlite,3dda5b4c2fdd05d987d162980075c9a18547975f4629b76721119fa20cfe3172
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,3d4d25dcf11d1e680adf11380c9230476f47b4ab277c8ebcda3879248d095353
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,9d0bb611bf7c5f9ec6a7d9f2ad91516c6074463c42985bd4133c11ba47c3a987
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,dd361d753e6c313c1069ca53eb6e2f62d8b49c19f7598527a98feb512529030b
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard-stats.log,fb99010fc5199d5ad7c3114b99342c4be2fd2ddf783a94c16422ee70382df004
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.json,a548a8a1103d344a2825bc407f60f935dd658117f3164e2907bb1f893837db00
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.log,1ed2824beef87ff20991f0099aadc6fa0229c3390661237cb3a121c1722ce4a0
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.nsys-rep,969d3a7ce60746577b2f009a5d108aaeac6c82fc1151193031b41ee19e1ef08f
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.sqlite,3a85976c266b5f2b11612da668def7b2357d4ca8a735867c3a7fccb442c893ac
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,430660193e173b22f0b841779930f95ad2f95fcdcc5b7ce1557eca62acea769f
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,9ed58e1eef1e5664f13fff213ad8b819c4f3de254606171d4cc2bdbe5054690f
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,c3040979196dae82881acb124c2c36388d2feb71f3252370007e8133db425366
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit-stats.log,7ec1d2c97f40fadd5b6e4117d3eb7d8da11270e78433a23a0fecf540573e3f36
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.json,49b221369fd7947b4451e2df4f85c5a007fa53d80aff72c64233b21552cdb25b
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.log,1720ffdd2cec7c49ea084d06036081a82d418ac789b8b3c1269f77b644742b46
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.nsys-rep,43279645cc59d3085040bfde873610dcaa0acc3ccec6da1ac71875ad88054b18
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.sqlite,4dde39fb4459334500af43eb355d6013a21b362a30b0b2dbef3cae6f09116342
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_api_sum.csv,af064c0000457356e68e2b967aba44e7600efe35b44ff97844722f49d8a76bcd
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_gpu_kern_sum.csv,79cef5bd91e23b03b5f412669cd9b0042ee6ab467b1700a2244190f8a035a8be
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_nvtx_pushpop_sum.csv,a8b5a59f3fb362df90e724d7a9941d07a7e7aae58a148b97ffded5f71c57efef

To reproduce a capture, use the public source-install and project-local CUDA
instructions in :doc:`../install` and :doc:`../cuda`, build the recorded source,
then run the matching public ``scripts/profile_solve.py`` inside Nsight Systems
with capture-range=cudaProfilerApi and cuda,cublas,nvtx tracing. The ignored
orchestrator is not needed. Nsight installation, WSL permissions and GPU
availability are external prerequisites. Exact frozen command argv and all
original summary CSV rows are retained in the analysis JSON.


profiles-launch CUDA follow-up diagnostics
------------------------------------------

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
construction is outside its solve capture; output and requested Jacobi setup
remain per-call.
No historical pinned-host-scalar allocation is claimed as a current cost.
Empty workspace kernel exports record zero instances only when the unchanged
Nsight stats log explicitly confirms no CUDA kernel data. An absent export
remains unreported; its absence is not converted into a fabricated zero.

.. csv-table:: All 6 full-capture diagnostics
   :class: benchmark-results
   :header-rows: 1

   Capture,Instrumented wall median ms,10 outer NVTX total ms,Malloc/free,Launch APIs,GPU kernels,Kernel CSV,Memcpy/async,Stream/device sync,Gated I/M/R,Before TrueR,Before threshold
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,2.933460,34.083211,70/80,520,520,recorded,0/110,130/30,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,0.945598,12.621146,20/20,520,520,recorded,0/110,130/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,1.111044,12.685238,0/0,510,510,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,1.032236,12.163074,0/0,450,450,recorded,0/100,120/0,4/6/0,1.42392e-07,1.04058e-05
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,16.512261,161.154874,0/0,10810,10810,recorded,0/1320,1300/0,118/122/2,1.21108e-06,1.44144e-06
   cuda-native-dense-1024-prepared-device-solve-none-explicit,21.339047,208.854444,0/0,8340,8340,recorded,0/1410,1440/0,69/139/0,8.92006e-06,1.04058e-05

All CUDA API calls remain in the table below, including profiler and completion
fences outside the NVTX ranges. Raw original CSV rows and API durations remain
in the analysis JSON; durations are not combined into additive attribution.

.. csv-table:: Complete observed API count inventory
   :class: benchmark-results
   :header-rows: 1

   Capture,API,Calls per ten-call capture
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaDeviceSynchronize,30
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventCreateWithFlags,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaEventDestroy,160
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaFree,80
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaLaunchKernel,520
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMalloc,70
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,40
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaFree,20
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaLaunchKernel,520
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaMalloc,20
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaMemcpyAsync,110
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaMemsetAsync,10
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cudaStreamSynchronize,130
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaLaunchKernel,510
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaLaunchKernel,450
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaMemcpyAsync,100
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cudaStreamSynchronize,120
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuProfilerStart,1
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaLaunchKernel,10810
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaMemcpyAsync,1320
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaStreamGetDevice_v12080,30
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cudaStreamSynchronize,1300
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuProfilerStart,1
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaLaunchKernel,8340
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaMemcpyAsync,1410
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaStreamGetDevice_v12080,30
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cudaStreamSynchronize,1440

Compiled source and retained evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actual compiled source: ``14c37277de16182fdc7f4cfcf9193e0b352791d1``; tree: ``26fcec1817f59f8dd680980fffcdf8fe0907e628``.

Profile driver and campaign runner are different files and are independently
hashed below. Raw JSON, CSVs, logs, Nsight reports and frozen source/binaries
remain local ignored evidence; these paths are not public artifact downloads.

.. csv-table:: Source identities
   :class: benchmark-results
   :header-rows: 1

   Role,Path,SHA256
   profile_driver,build/restart-optimization-20261017/launch-stages/cuda-release/tools/profile_solve.py,20b805f3bd95a9f0c1a5d23105c17888e4bd285b754575c1540661df87b7e0c4
   runner,build/restart-optimization-20261017/profile_current.py,eda0e6d1779698c6cc85d419c269cfb58150ef2c4cd8aad317ce066ea7f11d5e
   stage_identity,build/restart-optimization-20261017/launch-stages/cuda-release/identity.json,daf34907780c5b6fd366347ff81161904289228e9cf5960e85c0c669887e6466
   manifest,build/restart-optimization-20261017/profiles-launch/manifest.json,8b2d0e4586843cf69e0992e03774a035095707df648812c686939cb38790bec6
   plan,build/restart-optimization-20261017/profiles-launch/plan.json,773336c15492930d13b476caebccbcd9dc1d861276e29cfe484505755754f1cb
   renderer,build/restart-optimization-20261017/render_followup_profiles.py,f74a687c7499ab92d6919368405045c780eb93974c217c57978e82e3f58bc488
   render_helpers,build/restart-optimization-20261017/render_baseline.py,af180cd5ecb8e916450024b0d5fabd0f6b7088f9f41e0e720b95ea5552154ab0
   baseline_inputs,build/restart-optimization-20261017/baseline-analysis.json,8d27f15818e58182a7c113e38024049849fc2ebc44048de4362cbd08e8385a9d

.. csv-table:: Every retained diagnostic artifact
   :class: benchmark-results
   :header-rows: 1

   Capture,Filename,SHA256
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard-stats.log,0237dfcedc8c0c33740f2770e874b8d5b9cd357a3ebc0e99072b50cac284fe5d
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.json,29bdb85f55c1d94d136b0628425fc17ad03a9af66b5f5167bc2caad773fce74c
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.log,06552195c8404d10f288953764f4b851b38d3da8f8e2b73f46ade227f3852fab
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.nsys-rep,a4b4d4905cf9191e8f039762383320b46c8558e35f0dbabb055816ad394ec60e
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard.sqlite,00a4cbc6b76fc86a3181412aea2616843e2771ed28a199d937fba32d58b3a6fb
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_api_sum.csv,f26a9d0a96ac7050e2e3d561baf25a9073e363ca6c8c70bcb24d1a054435f4d9
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,80174311ff3f1a8b5a53bab517d9718894d2dd5d359abf7d117c2c2d92b696be
   cuda-cublas-dense-1024-api-device-solve-jacobi-standard,cuda-cublas-dense-1024-api-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,bfccb2a4517748b517e51628807b597a10aa451500d3a9133d0f8d2eaf0203d0
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard-stats.log,b75a68e741804e40eb9e1d76cbac97cc4c255a305960431c2306e0114c69d457
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.json,97c32aeeb805022514dd46f00c5e6bb11c74477f7187f8a22f0da68be83f5ba1
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.log,fedc514536dc2315823f53114457a5aa985f99db36399d729388ba757c478bce
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.nsys-rep,9b4ea57b733667226f20182132074f86cd3cdc7ebd0843fb27a5978ec9a4fb73
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard.sqlite,96a07e8660bc336ef36f6cf8269a006b2a7357723bb30b53af43c5cc4987f7fb
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_api_sum.csv,99a7c4c67b716fdd84dfda320889a28bda1b661d2453afb9e9358dcf9ca78477
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,c3e9509300efe2838f7e38f5684ab77799579c168183d63d0636b864827953c5
   cuda-cublas-dense-1024-session-device-solve-jacobi-standard,cuda-cublas-dense-1024-session-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,22bcd966f70c5b0d2aa37ec6585be5632027e70aca17db196034e9e9ed2e3bfd
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard-stats.log,b89a1554610a0969f5f3181269612dae51ba7ba9d4d25c598365216bea82d9fd
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.json,38b7cff6f16494056ed02a8f631c1a76be46fe13f50b3af73a89ff42de05ef8b
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.log,746c98b3d46688071105d209be76c5921b389fe534341df2c531318f8685cce3
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,706b6a7336a6b69d46b3c9bc442343cd458c98d277ea07f552af6e832ba6b4c5
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard.sqlite,0e0ca6d39951561fe1853a78eeae61c8df35264469f8d0282ebdc670fbeb3772
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,c589369f2f4d8137d34fb208973508efa71e1af13465a425b06b9d49b7d956eb
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,2e5e27abe917795748d7eb4892ff01cc999ad24f7a6562498dd45eb9320868fa
   cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard,cuda-cublas-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,0296b6cf815ebb917cac6fa3c6f2d4db22b6027e0ac51c65716e57495024cea6
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard-stats.log,063db06752ce18ab3510b08d800d05cc7f3b77c7f8351097847c73cffde6be63
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.json,ce46e1d719ea00e4b0ddebc70ff3fd992ac484c3144e3cd85915236ceed0e73d
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.log,27045d881c1bc54e8f76577819ee3201600f4d8d0b43b409ad4e4b31447c17f2
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.nsys-rep,6b6af14a9f6313285500ac0fa3853eabdf425f7535183a14363d9a78926fc23a
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard.sqlite,66a59140fe160df62a0086d5205efa0ad730bd8621b8d523836ec1589740e3eb
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,2e3daf6332d694f0ac483c65b74b28acc9bf48383c7c5428b65484821c3127df
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,f47308f265751cd907c9b0ef8e64474c47bbfcee6ef0d641431dac80717932d8
   cuda-native-dense-1024-prepared-device-solve-jacobi-standard,cuda-native-dense-1024-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,00704490963ffadda5e312bb7b0d196dc7d41a599b6a4a10be0519fc653b64c7
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard-stats.log,b7d375c1db280f845f90292b7a5e0a3dbb7ef58ade50e8e95cb0a7ab3bf5f24f
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.json,5c39377c11f8724d42ecd7bce2512de8457b484b3658b98204f2d73ea085cbbf
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.log,66238007c17356bd533073b8a3c55e08a09cdff951ae70b7c103204e6208beef
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.nsys-rep,e0e7dbf5ec4c49ae42baa827fd78f427b204c75c2d2b12c5a2237bb3770662c0
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard.sqlite,a98bd6828772abc2625d6321ea501d891bd7a1dea17e6673f53fa41bccf5d84d
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_api_sum.csv,d47dd33af3e46c6589948e73a271308756b42fb619438c46d4867d9ab810ce40
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_cuda_gpu_kern_sum.csv,a3e04747ed12459a1b1565fb865a25dd1cc153327ebbe767b0523664ebcff809
   cuda-native-poisson-32-prepared-device-solve-jacobi-standard,cuda-native-poisson-32-prepared-device-solve-jacobi-standard_nvtx_pushpop_sum.csv,c770ece2aa428549c9ddd0cd0350e45aef5908e5ac35bd7b32bbf391075624a0
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit-stats.log,0a15081ad9838faa6ed99a41342a92857395c60cf02a11d9c35ff9657aeb4433
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.json,6a1f9c728f61f99657550d68ae7d239014177022a30a9ba07fec9621a2c63931
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.log,8e4ceab33dca51d54953cf1f0e130603076ff47a315008a09baa89e3d8fdc915
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.nsys-rep,0c1c8e74c9c424eed53d0568d313b3533f6fd2ca5ea0f97d99f1c5008855a2b8
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit.sqlite,e5a73ff12a00a266ed25a360367e00db261aaf210298813f9ec2dead805c7406
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_api_sum.csv,549aca636d878b643af0b54629ef2a40df6e26d52e7ae87255c1d14c01085396
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_cuda_gpu_kern_sum.csv,cdf675184ef359756a9dd3cbc2f4e824e817dc4975a4e65f80a92bd11aec7c59
   cuda-native-dense-1024-prepared-device-solve-none-explicit,cuda-native-dense-1024-prepared-device-solve-none-explicit_nvtx_pushpop_sum.csv,5b65754db66f6226801bfb481bb1d0517567280feea6b6dad7be666c21aab04a

To reproduce a capture, use the public source-install and project-local CUDA
instructions in :doc:`../install` and :doc:`../cuda`, build the recorded source,
then run the matching public ``scripts/profile_solve.py`` inside Nsight Systems
with capture-range=cudaProfilerApi and cuda,cublas,nvtx tracing. The ignored
orchestrator is not needed. Nsight installation, WSL permissions and GPU
availability are external prerequisites. Exact frozen command argv and all
original summary CSV rows are retained in the analysis JSON.

