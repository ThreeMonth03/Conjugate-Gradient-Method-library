Balanced process campaigns
==========================

``make pyprof`` and pytest use the same fixtures, solver adapters, correctness
checks and Williams scheduler. A single run describes round timings within
one process. Adoption decisions need repeated independent process starts as
well as the numerical checks; more rounds alone do not measure process
variance.

``scripts/run_performance_campaign.py`` orchestrates already built, frozen
packages through the public ``cgbench run`` command. It never builds or
installs dependencies. This runner requires Linux/WSL, ``taskset`` and POSIX
process groups. Prepare dependencies in the project's local virtual
environments and finish builds, tests and profilers before starting a quiet
campaign. Keep native, OpenMP, BLAS and CUDA builds in separate campaigns.

The input JSON declares ``schema_version=1``, ``mode`` (``calibration`` or
``comparison``), ``stages``, ``configurations`` and the comparator path.
Each ``baseline`` / ``current`` stage declares its absolute ``package_root``,
actual full ``compiled_revision`` and ``compiled_tree``, and ``files`` mapping
absolute paths to SHA256 digests. Include the complete staged Python/native
payload and retain its build cache, compiler settings and source manifest.
These revision fields are declarations: verify them against the checkout
used for compilation; hashing a binary cannot certify which commit built it.
Both sides must use identical Python wrappers and reference implementations.

Each configuration declares a unique safe ``name``, absolute ``python``
interpreter, ``preset``, logical CPU ``affinity`` and full ``spec`` from a
public benchmark artifact. Preserve its suite, ordered cases, implementations,
scope, threads, seed, solver and sampling fields. CPU controls include
``numpy-out`` for dense or ``scipy-out`` for CSR, plus ``torch-eager-out``;
CUDA retains ``torch-cuda-eager``.
There is no CPU Session scope. CUDA fresh API, Session and prepared remain
separate configurations with their original allocation and reset contracts.

Use ``mode="calibration"`` with identical frozen payloads on both sides to
test how often the procedure calls the same build different. Run calibration
before selecting an optimization. Retain failed and inconclusive calibrations
alongside comparisons; they are evidence about the measurement procedure,
not results to replace with more favorable retries.

After writing and reviewing an input plan outside the new output directory:

.. code-block:: bash

   .venv/bin/python scripts/run_performance_campaign.py \
       --plan build/campaign-input.json \
       --output-dir build/campaign-results --plan-only
   .venv/bin/python scripts/run_performance_campaign.py \
       --plan build/campaign-input.json \
       --output-dir build/campaign-results

The first command freezes the exact schedule and hashes. The second accepts
only that pristine plan. A started, failed or completed campaign cannot be
resumed or overwritten. Partial raw output and logs remain available on
failure. Changes to a frozen input, package, tool or interpreter invalidate
the campaign rather than silently changing its meaning.

Minimal A/A example
-------------------

With the local benchmark dependencies installed, the following creates a
calibration input for CPU Jacobi prepared solves. Start from a clean committed
checkout and keep that source fixed through the build and snapshot. The
example refuses existing snapshot/input paths. It generates the spec through
the same public parser used by ``make pyprof``; it does not inspect timings to
select cases.

.. code-block:: bash

   make build PRESET=cpu-release BUILD_DIR="$PWD/build/campaign-source"
   PYTHONPATH="$PWD/build/campaign-source/python" .venv/bin/python - <<'PY'
   import hashlib
   import json
   import pathlib
   import shutil
   import subprocess
   import sys
   from cgbench.cli import benchmark_spec, parser

   root = pathlib.Path.cwd()
   subprocess.run(["git", "diff", "--quiet", "HEAD"], check=True)
   stage = root / "build/campaign-stage"
   stage.mkdir()
   shutil.copytree(root / "build/campaign-source/python", stage / "python",
                   ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
   shutil.copy2(root / "build/campaign-source/CMakeCache.txt", stage)
   def git(ref):
       return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
   frozen = {
       "package_root": str(stage / "python"),
       "compiled_revision": git("HEAD"), "compiled_tree": git("HEAD^{tree}"),
       "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in stage.rglob("*") if p.is_file()},
   }
   options = parser().parse_args([
       "run", "--suite", "cpu-dense-f64-v1", "--scope", "prepared",
       "--preconditioner", "jacobi", "--seed", "20261017",
       "--warmups", "3", "--rounds", "36", "--repetitions", "1",
   ])
   spec = benchmark_spec(options)
   spec["sampling"] = {"warmups": 3, "rounds": 36, "repetitions": 1}
   plan = {
       "schema_version": 1, "mode": "calibration",
       "comparator": str(root / "scripts/compare_performance.py"),
       "stages": {"baseline": frozen, "current": frozen},
       "configurations": [{"name": "cpu-jacobi-prepared",
                           "python": sys.executable, "preset": "cpu-release",
                           "spec": spec, "affinity": [0]}],
       "window_pause_seconds": 60, "worker_timeout_seconds": 600,
   }
   with (root / "build/campaign-input.json").open("x") as output:
       json.dump(plan, output, indent=2)
   PY

Run the two campaign commands above after all preparation has finished. Choose
an available logical CPU if CPU0 is outside the process's permitted affinity.
For a comparison, build and snapshot each declared revision separately with
matching compiler flags and backend settings; use ``mode="comparison"`` and
the two actual source identities. Add each scope or fusion policy as its own
configuration before freezing the plan. A calibration pass applies to its
observations, rather than certifying future comparisons.

Scheduling and interpretation
-----------------------------

Two time windows each contain ABBA and BAAB quartets for every configuration.
A is baseline and B is current; every slot starts a fresh interpreter. The
second window reverses configuration order. This yields eight process blocks
per side, with explicit window, quartet, slot and launch timestamps in the
manifest. ``window_pause_seconds`` defaults to 60; a pause does not establish
thermal equilibrium or independence of the physical host.

Each child retains the declared warmups, repetitions and complete Williams
cycles. All six thread-pool environment controls, actual runtime pools,
affinity, loaded native payload and numerical work are checked. No framework
row is dropped to improve a verdict. Known iterations, matvecs and restarts
must match across blocks for the same candidate, and each result must pass
the independent true-residual gate.

Boundary snapshots record UTC and monotonic timestamps, Linux CPU/load
information, available frequency/temperature sensors and ``nvidia-smi``
status. Collection occurs before and after child execution, outside solve
timers; no continuous observer is enabled. Missing sensors, unsupported GPU
queries and command errors are explicit. Boundary snapshots cannot explain
every disturbance inside a process. WSL affinity does not isolate Windows
load, display activity, power management or physical cores.

The unchanged primary comparator uses 10,000 hierarchical artifact-then-round
bootstrap samples, a fixed seed of 20261018, 95% percentile intervals and a
5% regression budget. An upper bound at most 1.05 supports no material
regression; a lower bound above 1.05 indicates regression; overlap remains
inconclusive. Passing that gate does not establish a speedup. Reports retain
every native and unchanged framework control, including regressions.

Do not subtract or divide out control drift, mix scopes, normalize away
different numerical work, relax the threshold or discard started trials.
Eight process blocks across two windows provide more process evidence than
two blocks, but inner rounds can remain serially correlated and intervals
have per-case rather than simultaneous coverage. Report limitations and all
outcomes before deciding whether to adopt an optimization.
