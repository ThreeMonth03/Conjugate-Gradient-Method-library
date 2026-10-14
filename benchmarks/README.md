# CPU CG benchmarks

`make pyprof` runs the `cgbench` runner. `make pytest` exercises the same case
builder, solver adapters, correctness gate, Williams scheduler and artifact
writer. All dependencies belong in the project `.venv` / local build tree.

See [recorded CPU results](results.md) for complete measurements, raw artifacts,
hardware provenance, and reproduction commands.

The sparse solver strategy suite is the primary CG/PCG-versus-direct comparison:

```bash
make pyprof PYPROF_SUITE=sparse-solvers PYPROF_THREADS=1 PYPROF_SCOPE=prepared
make pyprof PYPROF_SUITE=smoke-sparse-solvers \
  PYPROF_ARGS="--warmups 1 --repetitions 1 --rounds 6"
```

Its six default candidates are `native-cg`, `native-pcg`, `scipy-cg`,
`scipy-pcg`, `scipy-lu-total` and `scipy-lu-reuse`. The SciPy CG candidates
call the actual public `scipy.sparse.linalg.cg`; PCG supplies a Jacobi
`LinearOperator`. Their residual recurrence follows SciPy's library policy,
without the native solver's configured periodic true-residual replacement.
This is a solver-strategy comparison, so its output reports verified solution
times and setup costs without a generic native implementation speedup ratio.
All candidates must satisfy the same requested true-residual threshold; direct
solves can achieve tighter residuals and are not claimed to have equal error.

The full suite has diagonally scaled 2D grids of sides 16/32 and 3D grids of
sides 6/10, plus a neutral unscaled 2D grid of side 16. Scaled matrices are
`A=D*P*D`, where P is the Dirichlet Poisson operator and D has a geometric
diagonal from 1 to 20. This preserves SPD and makes Jacobi useful. The parameter
specifies scaling, not an exact condition number. The neutral constant-diagonal
case keeps Jacobi's lack of useful scaling work visible. Smoke uses scaled 2D
side 4 and 3D side 3 (scale 10), plus unscaled 2D side 4. Inputs and known
solutions use the same case builder and SHA-256 contract in pytest and pyprof.

The opt-in `sparse-solvers-large` suite uses unscaled 2D grids of sides 64/128
and 3D grids of sides 20/30 (up to 27,000 unknowns). It exposes sparse factor
fill-in at a more useful scale and is excluded from default/CI smoke runs.
The larger direct factorizations require more time and stored factor memory.
Use an explicit short sampling plan when exploring it:

```bash
make pyprof PYPROF_SUITE=sparse-solvers-large PYPROF_THREADS=1 \
  PYPROF_SCOPE=prepared PYPROF_ARGS="--warmups 1 --repetitions 1 --rounds 6"
```

Native prepared PCG caches its Jacobi inverse; native API PCG constructs it in
each measured call. Official SciPy prepared PCG caches its Jacobi operator,
but public SciPy CG allocates iterative workspace in both scopes. API SciPy
PCG includes Jacobi construction each time. Setup samples are measured once
outside formal timing, and artifacts state whether setup repeats inside each
timed invocation. Native setup metadata comes from the plan's inverse-diagonal
construction timer when available. SciPy iteration/matvec instrumentation runs
only in the first untimed correctness probe; formal timings use direct CSR
input without a callback or counting wrapper.

SuperLU uses public `scipy.sparse.linalg.splu` on CSC storage. `scipy-lu-total`
factors and solves on every measured call. `scipy-lu-reuse` factors once before
measurement and times subsequent solves; its factorization setup time is
reported separately. CSR-to-CSC conversion is outside both variants and its
time/storage are disclosed. Every candidate is charged the output reset from
the same x0, although a direct factor solve does not use an initial guess.
Saved input, inverse-diagonal, CSC and L/U/permutation array bytes describe
stored numerical arrays, not peak RSS or all allocator/SuperLU overhead.
CSR input bytes count the shared SciPy arrays with their original index dtype.
They are not the native owning CSR size: native size_t indices may use more
bytes, and native snapshots/workspace are excluded from this shared-input count.

For implementation comparisons, use the dense/CSR suites below or optional
`scipy-reference-cg` / `scipy-reference-pcg` / `torch-cg` / `torch-pcg` candidates.
The reference candidates use this project's recurrence and replacement policy;
they are clearly separate from the official `scipy-cg` / `scipy-pcg` solvers.
Jacobi PCG supports `--algorithm standard`; the explicit legacy algorithm and
Jacobi combination is rejected. Normal implementation suites also accept
`--preconditioner jacobi` when selecting their common preconditioner.
`--no-fused` selects the native separate-operation kernel path; NumPy and
PyTorch retain their eager operations. The selected flag is saved for replay.

The underlying entry point is useful for selecting one comparison:

```bash
PYTHONPATH=python .venv/bin/python -m cgbench run \
  --suite cpu --implementations native,numpy-out,torch-eager-out \
  --scope prepared --threads 1 --algorithm standard \
  --seed 20261015 --warmups 3 --repetitions 3 --rounds 30 \
  --output build/benchmarks/cpu-t1.json
```

The smoke suite has one 16-by-16 system. The CPU suite has dense float64 SPD
systems of sizes 64, 256, 1024 and 2048. Inputs use a seeded generator, strict
symmetric diagonal dominance, a varying diagonal, a known solution, and zero
initial guesses. `condition_scale` specifies the diagonal range, not an exact
condition number. Input preparation and matrix generation are excluded from
both scopes. A SHA-256 hash includes the exact input bytes; a seed alone does
not guarantee identical inputs across library versions.

To compare both explicit dense kernel modes in a BLAS-enabled build:

```bash
make pyprof PRESET=cpu-blas PYPROF_SUITE=cpu-dense-f64-v1 \
  PYPROF_THREADS=1 PYPROF_SCOPE=prepared \
  PYPROF_ARGS='--implementations native,native-blas,numpy-out,torch-eager-out'
```

`native` always selects `dense_kernel="native"`; `native-blas` always selects
`dense_kernel="blas"`. There is no automatic selection or size threshold.
Both candidates retain the selected recurrence, preconditioner, residual policy
and `fused` flag. The private BLAS provider replaces dense matrix-vector
products; vector operations still use the native backend. `native-blas` requires
an enabled build, a dense suite and one thread; unsupported selections fail.
Use `PYPROF_SUITE=smoke` for the same adapters on a short correctness run.
Artifacts record the selected kernel and provider configuration, and replay
preserves the candidate names.

`smoke-csr` uses a 4-by-4 grid (16 unknowns). `cpu-csr-f64-v1` uses 8, 32, 64
and 128 points per grid side (64 through 16,384 unknowns). Each sparse system
is the five-point two-dimensional Dirichlet Poisson operator with diagonal 4
and neighbor entries -1, stored as canonical sorted CSR. The known solution
and initial guess follow the same seeded contract as dense cases. No large
dense matrix is constructed; hashes include CSR indptr, indices and data.

```bash
PYTHONPATH=python .venv/bin/python -m cgbench run \
  --suite cpu-csr-f64-v1 --implementations native,scipy-out,torch-eager-out \
  --scope prepared --threads 1 --algorithm standard \
  --warmups 3 --repetitions 3 --rounds 30 \
  --output build/benchmarks/cpu-csr-t1.json
```

Sparse defaults select `scipy-out` instead of `numpy-out`. The SciPy reference
uses public CSR `A @ x` and preallocated NumPy vector operations. Public SciPy
matvec allocates its product; that allocation and copying into workspace are
included in timing. This is not an allocation-free baseline or the independent
`scipy.sparse.linalg.cg` implementation. The Torch baseline uses CSR tensors
and eager `mv(..., out=...)` with the same vector operations as dense CG.
Converting canonical input arrays to native CSRMatrix / Torch CSR tensors is
outside both scopes, just like preparing compatible dense library inputs.
The `api` scope includes fresh x/workspace allocation and the public solve;
`prepared` keeps operator/workspace objects. CSR construction/canonical checks
are excluded and library-ready inputs are reused in both scopes.

`prepared` prepares the operator and vector workspace once. Each measured call
copies the same x0 into the output, then solves. Native, NumPy and PyTorch each
preallocate vector workspaces. The NumPy reference uses `out=` operations;
PyTorch uses eager CPU operators with inference mode. Reduction scalars still
have the costs of their respective libraries. Fixture finiteness is validated
once outside timing in reference adapters; native solve detects non-finite
arithmetic rather than rescanning all input entries.
`api` times a one-shot solve with a fresh output and vector workspace, using
already compatible NumPy arrays or Torch tensors as inputs. Native API input
validation and output/workspace allocation are included. It does not include
constructing Python lists, loading data, or importing a library. Both scopes
reset x0 on every invocation, including warmups and repeated timed calls.

`standard` uses r=b-Ax, the standard CG recurrence, true-residual verification
on candidate convergence, and residual replacement plus direction restart
every 50 iterations. `--residual-refresh 0` disables periodic replacement.
`explicit` preserves the original library's explicit r=Ax-b formulation, with
two matvecs per iteration. Matching-policy implementation candidates use the selected algorithm, tolerance
`max(atol, rtol*||b||2)`, maximum iterations and replacement policy. Recorded
matvec counts include initial and final true-residual evaluations. Official
SciPy and direct candidates follow the separate policies described above;
their independent accuracy gate uses the same requested tolerance.

Before timing, each candidate must report convergence and independently satisfy
the requested true residual. The norm comparison allows only a 64-machine-
epsilon relative margin for differing summation orders. Finite but incorrect
solutions are marked invalid and excluded from timing/speedup. A final check
after timing guards against accidentally retaining a solved initial guess.
Missing optional Torch is ineligible and visible in the artifact; missing
requested native code or a numerically invalid candidate fails the run.
Independent verification uses a scaled `math.hypot` norm to avoid admitting
false zero residuals when squaring tiny values underflows. Reference/native
iterative solvers retain their ordinary float64 fast path and report numerical
range failure when nonzero squared inner products cannot be represented;
this does not promise arbitrary-scale CG support.

Every run starts a fresh worker with BLAS/OpenMP thread environment variables
set before numerical imports. Torch intraop threads use the requested count;
interop threads use one. Artifacts record actual thread pools when the local
`threadpoolctl` dependency is present. Libraries can use fewer threads than
requested; inspect the recorded runtime information. Avoid competing compute
jobs and use the same CPU affinity/hardware for performance comparisons.

Warmups are untimed. Timed rounds use the Williams design also used by solvcon:
each candidate occupies each position equally over a complete cycle, and
within-row predecessor pairs are balanced. A cycle has k rows for even k
candidates and 2k for odd k. Requested rounds are rounded up to a complete
cycle for the candidates that passed correctness checks, and the resolved
count is saved. Each sample is one interval around `repetitions` calls.
Reported percentiles summarize per-round averages, not individual-call tail
latencies. The runner uses explicit fixed call counts; it does not silently
calibrate repetitions.

Artifacts save raw nanoseconds, actual schedules, input hashes, solver options,
revision/dirty state, CPU/OS/affinity, package versions, BLAS/thread information
and the loaded native module path/SHA-256 plus embedded build metadata and an
adjacent CMake cache when available. A cache from another build is never used.
Keep artifacts from separate
processes separate. Compare and replay with:

```bash
PYTHONPATH=python .venv/bin/python -m cgbench compare build/benchmarks/cpu-t1.json
PYTHONPATH=python .venv/bin/python -m cgbench replay \
  build/benchmarks/cpu-t1.json --output build/benchmarks/replay.json
```

Replay regenerates the recorded inputs/options and checks their hashes,
eligible candidates, resolved sampling counts and saved timing schedule; it
does not promise identical wall time. `compare` warns when recorded specs or
environments differ. CI should gate correctness and artifact collection in the
smoke suite. Speed thresholds belong on a stable machine, separate from
ordinary shared-runner PR checks.

The implementation suites measure the same CG recurrence and policy. Dense
`numpy.linalg.solve` / `torch.linalg.solve` use direct solvers and belong in a
separately labeled application comparison. The independent mature SciPy CG
solver is part of the sparse strategy suite with its stopping/replacement
policy disclosed. Converting sparse problems to dense solely for a
baseline would conflate storage and algorithm costs. Fixed-iteration
throughput and compiled Torch baselines can be added as
separately labeled metrics rather than replacing time to a verified solution.
