CUDA
====

CUDA is an explicit optional backend. CPU functions retain their existing
defaults; use ``cgpy.cuda`` for device operations. Importing this module works
with a CPU-only wheel and does not import PyTorch. ``cuda.enabled()`` checks
whether its extension and runtime libraries load; ``cuda.is_available()``
additionally checks the requested GPU. Neither query selects another backend.

Local build
-----------

The pinned bootstrap supports Linux x86-64, Python 3.12, CUDA Toolkit 12.8
and an existing compatible host compiler. The preset uses ``/usr/bin/g++-13``
and explicit architecture ``75`` for the RTX 2060. A compatible NVIDIA driver
must already be installed. A driver's advertised CUDA API version is separate
from the compiler toolkit version; this package requires Toolkit >=12.8 and
<13 to match its cu12 runtime dependencies.

Keep the GPU toolchain and comparison dependencies in a separate environment:

.. code-block:: bash

   python3.12 -m venv .venv-cuda
   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip \
       -r requirements/test.lock -r requirements/cuda.lock
   .venv-cuda/bin/python scripts/setup_cuda.py --prefix .deps/cuda-12.8
   CG_REQUIRE_CUDA=1 make test PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python"

The bootstrap verifies pinned NVIDIA compiler/cudart/CCCL archive checksums
and assembles them under ``.deps/cuda-12.8``. Public NVIDIA wheels supply the
runtime and cuBLAS headers/libraries. Downloads remain under ``.cache``; no
system toolkit, shell configuration or PyTorch-private library is changed.
Initial downloads require network access. Override ``CUDA_ROOT``,
``CUDA_HOST_COMPILER`` or a semicolon-separated ``CUDA_ARCHITECTURES`` list
explicitly when using another compatible local toolchain. CUDA compilation
uses C++20 without fast-math.

``CG_REQUIRE_CUDA=1`` makes local GPU validation fail if the device or extension
is unavailable. Hosted compile-only CI leaves it unset and reports GPU skips.

Optional Compute Sanitizer checks use the same local toolkit:

.. code-block:: bash

   .venv-cuda/bin/python scripts/setup_cuda.py --sanitizer
   .deps/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 1 build/cuda-release/cg_cuda_gtests
   .deps/cuda-12.8/bin/compute-sanitizer --tool racecheck --force-blocking-launches --error-exitcode 1 build/cuda-release/cg_cuda_gtests
   .deps/cuda-12.8/bin/compute-sanitizer --tool synccheck --error-exitcode 1 build/cuda-release/cg_cuda_gtests

The recorded WSL2 run needed serialized racecheck launches because its default
mode stalled on the concurrent-stream test. This setting checks kernel shared
memory hazards under serialization; it does not establish race freedom for
overlapping launches. Normal execution, memcheck and synccheck also exercise
the concurrent-stream test without this flag.

To build a relocatable local CUDA wheel:

.. code-block:: bash

   make wheel DIST_CUDA=ON DIST_DIR=dist/cuda PYTHON="$PWD/.venv-cuda/bin/python"
   make test-dist DIST_CUDA=ON DIST_DIR=dist/cuda PYTHON="$PWD/.venv-cuda/bin/python"
   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip \
       'dist/cuda/cgpy-0.4.0-cp312-cp312-linux_x86_64.whl[cuda]'

Select the actual wheel filename produced for your Python/platform. The
``[cuda]`` extra installs ``nvidia-cuda-runtime-cu12==12.8.90`` and
``nvidia-cublas-cu12==12.8.4.1``. The loader opens these public distributions
before importing ``cgpy._cuda``; installed wheels have no toolkit RPATH and need
no source checkout or compiler. The driver remains a host prerequisite. CUDA,
native and BLAS wheels share the package version/tag; keep their output
directories separate. These are local Linux platform wheels, with no
manylinux portability claim. Installing ``.[cuda]`` alone installs the extra
dependencies but retains the default CPU build; enable ``CG_ENABLE_CUDA``
explicitly when building the extension.

For a source install with the same local toolkit and public runtime extra:

.. code-block:: bash

   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip \
       -Ccmake.define.CG_ENABLE_CUDA=ON \
       -Ccmake.define.CMAKE_TOOLCHAIN_FILE=cmake/CGCudaToolchain.cmake \
       -Ccmake.define.CUDAToolkit_ROOT="$PWD/.deps/cuda-12.8" \
       '.[cuda]'

Replace ``.[cuda]`` with the selected local source archive followed by
``[cuda]`` to install from an sdist. Pip's isolated build environment supplies
the declared Python build tools; the CUDA toolkit remains the explicit local
path above.

Device data and prepared solves
-------------------------------

Host transfers use explicitly named methods. Dense input is copied to
row-major float64 device storage. CSR upload accepts the same owning canonical
host ``cgpy.CsrMatrix`` used by CPU functions, without dense conversion.

.. doctest:: cuda-device
   :skipif: not cuda.is_available()

   >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
   >>> b = cuda.Vector.from_host(np.array([1.0, 2.0]))
   >>> plan = cuda.prepare_host(A, preconditioner="jacobi", dense_kernel="native")
   >>> workspace = plan.create_workspace()
   >>> x = plan.create_vector()
   >>> plan.solve_into(b, x, workspace=workspace, rtol=1e-10).converged
   True
   >>> np.testing.assert_allclose(x.to_host(), [1 / 11, 7 / 11])
   >>> x.fill(0.0)
   >>> plan.solve_into(b, x, workspace=workspace, fused=False).converged
   True
   >>> device_A = cuda.DenseMatrix.from_host(A)
   >>> answer = cuda.solve(device_A, b, dense_kernel="cublas", algorithm="explicit")
   >>> answer.converged
   True
   >>> np.testing.assert_allclose(answer.x.to_host(), [1 / 11, 7 / 11])
   >>> host_csr = cgpy.CsrMatrix([0, 2, 4], [0, 1, 0, 1], [4., 1., 1., 3.], (2, 2))
   >>> device_csr = cuda.CsrMatrix.from_host(host_csr)
   >>> answer = cuda.solve(device_csr, b, preconditioner="jacobi")
   >>> answer.converged
   True
   >>> np.testing.assert_allclose(answer.x.to_host(), [1 / 11, 7 / 11])

``prepare`` accepts an already-device-resident matrix; ``prepare_host``
normalizes and uploads host input. The plan owns a matrix snapshot and caches
the selected Jacobi inverse. ``solve`` allocates its result and workspace;
``solve_into`` treats ``x`` as the initial guess/output and requires an
explicit workspace. Reinitialize ``x`` before repeated comparisons.

``dense_kernel="native"`` selects project CUDA kernels; ``"cublas"`` selects
double-precision cuBLAS GEMV for dense operators only. CSR rejects cuBLAS.
``fused`` independently selects supported vector/reduction fusion. Standard
CG supports Jacobi; the explicit algorithm requires no preconditioner. Both
modes retain the true-residual stopping rule and failure statuses described
in :doc:`api`. Invalid shapes, options, dtypes or layouts raise ``ValueError``;
CUDA allocation, stream and execution failures raise ``RuntimeError``.

Streams, lifetime and interoperability
--------------------------------------

Each operation uses an explicit ``device`` and borrowed integer ``stream``
handle; zero is the legacy default stream. Calls finish on that stream before
returning and restore the caller's current device. They do not change a
PyTorch current stream. A nondefault stream must belong to the selected
device and outlive every workspace using it.

The strict path accepts compatible CUDA Array Interface arrays, including
detached contiguous float64 CUDA tensors. Vector output must be writable and
disjoint from the right-hand side and operator. Native device ownership is
retained through a solve, and external input owners are retained across GIL
release. Keep external storage alive and free from conflicting mutation;
retaining an owner does not prevent writes from another stream. Version 3
producer stream dependencies are honored. Version 2 producers have no stream
metadata: make their writes ready on the selected stream or synchronize them
before passing their arrays. Explicit host/device-copy methods complete
before returning; exported native vectors have no pending producer work.

Prepared matrix and Jacobi snapshots can be shared, but each overlapping solve
needs its own output and workspace. The workspace owns device vectors,
reduction buffers and pinned scalar storage. A cuBLAS workspace additionally
owns its handle and 4 MiB scratch allocation; overlapping
reuse raises an error. It fixes the device/stream used for its solve. No API
reinterprets a NumPy host pointer as a device pointer.

Validation and benchmarks
-------------------------

Hosted CI compiles the pinned CUDA toolkit without a GPU and explicitly skips
device tests when unavailable. The optional ``cuda_gpu_tests`` workflow input
requires a configured self-hosted runner labeled ``linux``, ``x64``, ``gpu``;
that job requires a real device before executing tests. CPU builds exercise
disabled-extension behavior:

.. doctest:: cuda-disabled
   :skipif: cuda.enabled()

   >>> cuda.is_available()
   False
   >>> try:
   ...     cuda.Vector(2)
   ... except RuntimeError:
   ...     rejected = True
   ... else:
   ...     rejected = False
   >>> rejected
   True

To document the imported CUDA build with executable examples:

.. code-block:: bash

   .venv-cuda/bin/python -m pip install --cache-dir .cache/pip -r requirements/docs.lock
   make docs doctest PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python"

See :doc:`benchmarks` for explicit resident-GPU comparison commands. Prepared
and API scopes exclude host upload and correctness-gate download. Reported
upload/setup metadata is separate from solve latency; neither scope measures
host-to-host end-to-end performance. Timings include host dispatch,
iteration scalar synchronization and final completion, rather than only GPU
kernel events.

API reference
-------------

CUDA solves return the shared :class:`cgpy.SolveResult` documented in
:doc:`api`; its ``x`` owns a device ``Vector``.

.. automodule:: cgpy.cuda
   :members:
   :undoc-members:
   :exclude-members: SolveResult
