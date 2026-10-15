Sparse solver results
=====================

Historical measurements on 2026-10-15 at revision
``ccc8ec7a76a569e4e2505251af3cb7d2469eb594``, using the host, packages and
one-thread Release build described in :doc:`cpu`.

The strategy suite compares native CG/Jacobi PCG, official SciPy CG/PCG and
SuperLU. SciPy uses its own stopping/recurrence policy. LU can produce much
tighter residuals, and LU reuse excludes factorization. These are times to
accepted solutions, not equal-error or same-algorithm speedup claims.
All 162 candidate/case results in the CPU runs, including implementation
comparisons, pass independent true-residual checks before and after timing.

Default suite
--------------

Scaled inputs are ``A = D P D`` with positive diagonal D spanning 1..20.
The neutral input is a constant-diagonal Poisson system.
Sampling uses three warmups, 30 Williams rounds and five solves per round.

.. csv-table:: Default strategy suite: median ms per solve
   :class: benchmark-results
   :header: "Scope","Case / n","Native CG","Native PCG","SciPy CG","SciPy PCG","LU total","LU reuse"

   "prepared","scaled-poisson-2d-16 / 256","0.3855","0.0489","2.4718","0.4837","0.3593","0.0264"
   "prepared","scaled-poisson-2d-32 / 1024","3.9723","0.3667","7.1950","1.2077","1.5682","0.0791"
   "prepared","scaled-poisson-3d-6 / 216","0.1567","0.0257","1.4619","0.2378","0.3693","0.0221"
   "prepared","scaled-poisson-3d-10 / 1000","1.3622","0.1411","3.3779","0.4779","3.2956","0.1145"
   "prepared","poisson-2d-16-neutral / 256","0.0369","0.0439","0.4214","0.4309","0.2523","0.0183"
   "api","scaled-poisson-2d-16 / 256","0.3671","0.0537","2.2860","0.4497","0.2951","0.0224"
   "api","scaled-poisson-2d-32 / 1024","3.7211","0.3553","6.7482","1.1249","1.4371","0.0727"
   "api","scaled-poisson-3d-6 / 216","0.1738","0.0363","1.5581","0.2825","0.4047","0.0228"
   "api","scaled-poisson-3d-10 / 1000","1.4899","0.1651","3.6362","0.5318","3.6259","0.1206"
   "api","poisson-2d-16-neutral / 256","0.0464","0.0563","0.4501","0.4929","0.2704","0.0211"

On the four scaled cases, Jacobi reduces native iterations
555→48, 1498→115, 252→25 and 444→37, yielding 6.10–10.83× lower prepared
medians than native CG. The neutral case remains at 50 iterations and PCG
adds overhead. LU reuse is faster than native PCG in all five default cases
in both scopes, after its prior factorization.

Large suite
------------

These unscaled, constant-diagonal cases give Jacobi no iteration benefit.
One warmup, six rounds and one solve per round provide less sampling than
the default suite.

.. csv-table:: Large unscaled suite: prepared median ms
   :class: benchmark-results
   :header: "case / n","Native CG","Native PCG","SciPy CG","SciPy PCG","LU total","LU reuse"

   "poisson-2d-64-large / 4096","3.0995","3.5854","4.0941","4.6194","6.6440","0.5216"
   "poisson-2d-128-large / 16384","16.6445","19.5010","24.4526","26.1848","34.0955","2.7351"
   "poisson-3d-20-large / 8000","2.1545","2.4136","2.6627","3.0328","163.1439","2.7112"
   "poisson-3d-30-large / 27000","10.2716","11.8192","12.2027","14.3893","2394.1336","17.3421"

LU reuse remains faster on the two-dimensional cases. Native CG is faster
than both LU candidates on these two three-dimensional cases.
At the largest grid, LU has a substantially tighter residual than CG,
as shown below.

Accuracy and setup
-------------------

The table uses the original prepared runs. All candidates meet
``max(1e-12, 1e-8 * ||b||2)``. The iterative range includes
native and official SciPy CG/PCG; their iteration counts can differ.
Direct solves have zero iterative updates and one final residual matvec.

.. csv-table:: Prepared accuracy
   :class: benchmark-results
   :header: "Case","Native CG iter/matvec","Native PCG iter/matvec","Iterative residual range","LU residual","Threshold"

   "scaled-poisson-2d-16","555/568","48/50","5.819e-05–6.260e-05","1.274e-12","6.386e-05"
   "scaled-poisson-2d-32","1498/1529","115/119","1.116e-04–1.456e-04","3.442e-12","1.473e-04"
   "scaled-poisson-3d-6","252/259","25/27","3.946e-05–8.780e-05","1.280e-12","8.968e-05"
   "scaled-poisson-3d-10","444/454","37/39","2.031e-04–2.094e-04","7.183e-12","2.149e-04"
   "poisson-2d-16-neutral","50/52","50/52","4.829e-07–4.829e-07","1.247e-14","6.319e-07"
   "poisson-2d-64-large","290/297","290/297","2.773e-06–2.775e-06","6.679e-14","2.810e-06"
   "poisson-2d-128-large","395/404","395/404","5.661e-06–5.704e-06","1.506e-13","5.749e-06"
   "poisson-3d-20-large","85/88","85/88","4.714e-06–5.157e-06","3.662e-13","5.817e-06"
   "poisson-3d-30-large","116/120","116/120","1.012e-05–1.038e-05","9.516e-13","1.063e-05"

Single untimed setup probes at the largest 3D grid measured 0.098942 ms
for Jacobi inverse construction and 2488.392033 ms for the reuse candidate's
factorization. The inverse contains 216,000 bytes; exported L/U arrays and
permutations contain 310,561,976 bytes. Shared inputs contain 2,743,204 bytes.
These array counts exclude workspace, owning snapshots and internal
allocator/SuperLU storage; they are not peak memory or setup distributions.

A separate fresh-process replay reproduced input hashes, candidates,
schedules and iteration/matvec counts. Its timings were kept separate,
rather than pooled with the original measurements.

Repeat the protocols
---------------------

Run after the CPU setup in :doc:`cpu`. These commands measure the current
checkout using the historical fixtures and sampling; source/environment
identity is recorded in each new local JSON.

.. code-block:: bash

   for scope in prepared api; do
       taskset -c 0 make pyprof PRESET=cpu-release \
           PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE="$scope" \
           PYPROF_ROUNDS=30 PYPROF_REPETITIONS=5 PYPROF_ARGS='--warmups 3'
   done
   taskset -c 0 make pyprof PRESET=cpu-release \
       PYPROF_SUITE=sparse-solvers-large PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
       PYPROF_ROUNDS=6 PYPROF_REPETITIONS=1 PYPROF_ARGS='--warmups 1'

:doc:`methodology` documents setup boundaries, solver defaults and how to
replay the newly generated files under ``build/``.
