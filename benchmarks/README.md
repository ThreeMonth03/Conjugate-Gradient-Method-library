# CPU CG benchmarks

`make pyprof` runs the `cgbench` runner. `make pytest` exercises the same case
builder, solver adapters, correctness gate, Williams scheduler and artifact
writer. All dependencies belong in the project `.venv` / local build tree.

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
two matvecs per iteration. All candidates use the selected algorithm, tolerance
`max(atol, rtol*||b||2)`, maximum iterations and replacement policy. Recorded
matvec counts include initial and final true-residual evaluations.

Before timing, each candidate must report convergence and independently satisfy
the requested true residual. The norm comparison allows only a 64-machine-
epsilon relative margin for differing summation orders. Finite but incorrect
solutions are marked invalid and excluded from timing/speedup. A final check
after timing guards against accidentally retaining a solved initial guess.
Missing optional Torch is ineligible and visible in the artifact; missing
requested native code or a numerically invalid candidate fails the run.

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
and available CMake compiler/flag metadata. Keep artifacts from separate
processes separate. Compare and replay with:

```bash
PYTHONPATH=python .venv/bin/python -m cgbench compare build/benchmarks/cpu-t1.json
PYTHONPATH=python .venv/bin/python -m cgbench replay \
  build/benchmarks/cpu-t1.json --output build/benchmarks/replay.json
```

Replay regenerates the recorded inputs/options and checks their hashes; it
does not promise identical wall time. `compare` warns when recorded specs or
environments differ. CI should gate correctness and artifact collection in the
smoke suite. Speed thresholds belong on a stable machine, separate from
ordinary shared-runner PR checks.

These comparisons measure implementations of the same CG algorithm. Dense
`numpy.linalg.solve` / `torch.linalg.solve` use direct solvers and belong in a
separately labeled application comparison. The independent mature SciPy CG
solver can be added with a separate tag, since its stopping/replacement
implementation may differ. Converting sparse problems to dense solely for a
baseline would conflate storage and algorithm costs. Fixed-iteration
throughput and compiled Torch baselines can be added as
separately labeled metrics rather than replacing time to a verified solution.
