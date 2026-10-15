cuBLAS API lifecycle diagnostics — 2026-10-16
=================================================

This diagnostic campaign uses compiled main source
``6bfb7b4905e79381a3d09772016aab7a860a0dad``, before the next resource,
CPU GEMV or launch changes. All ten planned captures completed without retries;
each passed the independent true-residual gate before and after capture.
The frozen CPU/CUDA packages and raw traces remain under
``build/execution-optimization-20261016/``. These instrumented samples are
**not framework performance scores**.

An i7-13700K / WSL2 host and RTX 2060 run FP64, CPU affinity 0, one library
thread, CUDA 12.8.93 and Torch 2.8.0+cu128. Fixtures use seed 20261015,
rtol=1e-8, atol=1e-12, three warmups and ten captured repetitions.
No other project builds, tests or numerical measurements overlap capture.

Current fresh API evidence
--------------------------

The four cuBLAS device API captures keep the same numerical work as the
uninstrumented campaign: 5/7, 4/6, 4/6 and 3/5 iterations/matvecs at
n=64/256/1024/2048. API includes fresh output, workspace/backend resource
construction and destruction, and per-call Jacobi construction. Resident
matrix/RHS uploads, correctness downloads and warmups are excluded.

Workspace-only captures include construction **and destruction**, without
iterations. CUDA malloc/free calls account for 97.3% of recorded CUDA API
duration in the device workspace capture and 97.7% in host; those percentages
are not fractions of full solve wall time. Provider-internal allocations,
frees and synchronizations are included in the observed counts.

.. csv-table:: All captures; counts are totals across ten repetitions
   :class: benchmark-results
   :header: "capture","instrumented median ms","malloc/free","memcpy/stream sync/device sync","kernel launches","iterations/matvecs"

   "cuda-cublas-dense-64-jacobi-api-device-solve","5.198334","90/100","120/140/30","770","5/7 (gated solve)"
   "cuda-cublas-dense-256-jacobi-api-device-solve","3.917138","90/100","110/130/30","640","4/6 (gated solve)"
   "cuda-cublas-dense-1024-jacobi-api-device-solve","4.496294","90/100","110/130/30","640","4/6 (gated solve)"
   "cuda-cublas-dense-2048-jacobi-api-device-solve","4.661090","90/100","100/120/30","510","3/5 (gated solve)"
   "cuda-cublas-dense-1024-jacobi-api-host-solve","4.115286","80/90","180/200/30","400","4/6 (gated solve)"
   "cuda-cublas-dense-1024-jacobi-api-host-workspace","1.720404","50/60","0/20/30","0","4/6 (gated solve)"
   "cuda-cublas-dense-1024-jacobi-api-device-workspace","1.752028","60/70","0/20/30","0","4/6 (gated solve)"
   "cuda-native-dense-1024-jacobi-api-device-solve","1.568563","60/60","110/130/0","580","4/6 (gated solve)"
   "torch-cuda-eager-dense-1024-jacobi-api-device-solve","3.193666","0/0","210/190/20","720","4/6 (gated solve)"
   "cuda-cublas-dense-1024-jacobi-api-device-prepare","0.929155","30/30","10/40/0","10","4/6 (gated solve)"

The prepare probe includes the matrix snapshot and Jacobi setup; it does
not isolate Jacobi alone. Work counts beside resource probes describe the
before/after gated solves, not operations executed inside those probes.

The trace contains cuBLAS create ranges but does not expose complete
destroy/GEMV host ranges. At n=1024 device API it records ten
``cublasCreate_v2`` ranges, median 0.23134 ms, total 2.32150 ms.
The source still constructs a handle and 4 MiB scratch per fresh workspace.
NVIDIA documents internal resource allocation in create and an implicit
device synchronization in destroy; see the
`CUDA 12.8 cuBLAS reference <https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasdestroy>`_.

The device solve traces spend 50.6–58.5% of recorded CUDA API duration
inside ``cudaMemcpyAsync``. A host API duration can include waiting for
prior GPU work and pageable-memory handling; this does not establish a
transfer-bandwidth bottleneck. The native/Torch controls, workspace-only
captures and kernel activity should be read together. Do not sum independently
sampled probe medians, subtract API/prepared scores, or attribute a fixed
fraction of end-to-end time from these summaries.

These observations support testing explicit resource reuse and reducing
launch/control overhead. Reuse must remain a separately named scope:
fresh API continues charging its complete fresh resource lifecycle.
The earlier pinned-host-scalar allocation has already been removed.

Public reproduction
-------------------

Use a checkout of the compiled revision named above and follow :doc:`/build`
for project-local dependencies/toolkit. Build before starting the profiler;
keep builds/tests and other measurements outside the capture campaign.
With Nsight Systems already available on the host:

.. code-block:: bash

   make build PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python" \
       CUDA_HOST_COMPILER=/usr/bin/g++-13 CUDA_ARCHITECTURES=75
   mkdir -p build/profiles-api
   for n in 64 256 1024 2048; do
       PYTHONPATH="$PWD/build/cuda-release/python" taskset -c 0 \
           nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
           --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
           --output="build/profiles-api/cublas-device-$n" \
           .venv-cuda/bin/python scripts/profile_solve.py \
           --suite cuda-dense-f64-v1 --implementation cuda-cublas \
           --case "dense-$n" --device cuda --scope api --probe solve \
           --preconditioner jacobi --cuda-execution-mode device \
           --seed 20261015 --threads 1 --warmups 3 --repetitions 10 \
           --cuda-profiler --output="build/profiles-api/cublas-device-$n.json"
   done

Repeat n=1024 with ``--cuda-execution-mode host`` for host solve; use
``--probe workspace`` in each execution mode for lifecycle captures.
Use ``--implementation cuda-native`` and ``torch-cuda-eager`` with device
mode for solve controls, and ``--probe prepare`` for the snapshot/setup probe.
Extract ``cuda_api_sum,cuda_gpu_kern_sum,nvtx_pushpop_sum`` with ``nsys stats``.

Retained identities
-------------------

Raw artifacts are local under ``build/execution-optimization-20261016/``;
all samples and unsuccessful attempts would be retained rather than retried.
The manifest SHA256 is
``130cebc1bc7e53427408638054fbefb5fb955c92e998321ee4538ac596d73e5d``.
The frozen CUDA identity SHA256 is
``0c3d5f14418fc6a3e3532a2ab858f6882cede29307171165c8a78f91fefa5176``.

.. csv-table:: Capture JSON and CUDA API summary SHA256
   :class: benchmark-results
   :header: "capture","JSON SHA256","CUDA API CSV SHA256"

   "cuda-cublas-dense-64-jacobi-api-device-solve","9539f4e5f54c38a39c77783d5c148edf7ff120ccf41818d5317cdacfd08b7a59","b7891b3fa67808f1b657291f550eae347e8dca1c2129e8b04ba5f8f39b71f9a0"
   "cuda-cublas-dense-256-jacobi-api-device-solve","b3df9a92dd67aafe30b19f8e4a23a4125398980ed00f2d8ce34e7544d99d5881","f19e36007139b58441d50574e6701defd7c44537ccaf40a6040e2180fe56d5ba"
   "cuda-cublas-dense-1024-jacobi-api-device-solve","6229dd3527f944ce8a7c0737e14a0431797c54f314ebb71afc757149a8d3b853","89cb42497fbf931dfcd21db4cdcb200482afd29f0de12ea4515c2bd5d0df1620"
   "cuda-cublas-dense-2048-jacobi-api-device-solve","7c547b2f8fd1274a286bb2d61e00140889840c419dba2fbd9f7dfff4a866d2bb","f9fadf60c3f00232b43ef1657232b8efdce85cc7ac90bb7ac34adc720c3f6b7a"
   "cuda-cublas-dense-1024-jacobi-api-host-solve","487f3e53891556b63801f75c26d3afa73b6acd1a419835f3635ff49e9974f71f","a6416c42e529f7d526ff014d82eeebc04c561d9dc8fe9e41a03caea614896956"
   "cuda-cublas-dense-1024-jacobi-api-host-workspace","0ed8ffff099cf9896e413eba0ff85e33cf0ad79592346f32137d396f4d419775","0e7d0112c3b179a23a267df0751bc6be824979502f515b323022dafdde13f148"
   "cuda-cublas-dense-1024-jacobi-api-device-workspace","7bdfc50ffd9e21df4e33fec521cd650e48137bd611611b615a9c197a866955e3","9d479726a424d67dc9298a3512df84d3ba7511b5bd700c3d2b47ae1c9beb2fca"
   "cuda-native-dense-1024-jacobi-api-device-solve","d605644afdcb5bbde0e39ba80b0352a439fb0971ac1acc6bc2dcce0d8025549d","e450a062ef597c8e64fa673b5d620f5b7d30d791f81fed04fee66ae05b63965a"
   "torch-cuda-eager-dense-1024-jacobi-api-device-solve","e303c6d6c25faee39cfc8d7244070d8596f4b6f4450ef7f6862bd8a778746204","4d56979de51d8b4170cc1b532036f43f54d8d6101b5aedbbf00e66a3e18a782d"
   "cuda-cublas-dense-1024-jacobi-api-device-prepare","48d0297b228671ff5076671aada6efeceb7f5958da81548549d8bb65e1d58ccd","6f647bb485105e8d6ad1f4095fcf9639a560cf9a4483afb9857dee0c90d6754c"
