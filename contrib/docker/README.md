# CPU development container

Run these commands from the repository root. The build context is the whole checkout; the Dockerfile copies source files and local dependency locks explicitly, leaving host virtual environments, caches, build products, and research artifacts out of the image.

```bash
docker build -f contrib/docker/dockerfile -t cg-cpu .
docker run --rm cg-cpu
docker run --rm cg-cpu make test PRESET=cpu-openmp
docker run --rm cg-cpu make gtest PRESET=cpu-asan
docker run --rm -it cg-cpu bash
```

The default command is `make test`. Python tools are installed in `/workspace/cg/.venv`; downloads and build outputs stay under that container's project directory. Compiler, Git, and their package dependencies are installed inside the image. It uses no GPU runtime and mounts no host directories.

Include the optional CPU benchmark dependencies when building a benchmark image:

```bash
docker build -f contrib/docker/dockerfile --build-arg CG_WITH_BENCHMARKS=1 -t cg-cpu-bench .
docker run --rm cg-cpu-bench make pyprof
```

To retain a benchmark artifact, use a named container and copy the selected output before removing it:

```bash
docker run --name cg-profile cg-cpu-bench make pyprof PYPROF_OUTPUT=build/benchmarks/run.json
mkdir -p build/benchmarks
docker cp cg-profile:/workspace/cg/build/benchmarks/run.json build/benchmarks/container.json
docker rm cg-profile
```

Container timing has its own CPU, scheduler, compiler, and BLAS environment. Keep that provenance with the artifact and compare runs under the same resource limits.
