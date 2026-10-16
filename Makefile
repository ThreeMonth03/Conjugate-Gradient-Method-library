PYTHON ?= $(CURDIR)/.venv/bin/python
CMAKE ?= $(dir $(PYTHON))cmake
CTEST ?= $(dir $(CMAKE))ctest
PRESET ?= cpu-release
BUILD_DIR ?= $(CURDIR)/build/$(PRESET)
NATIVE_BUILD_DIR ?= $(CURDIR)/build/$(PRESET)-native
NATIVE_ARGS ?=
JOBS ?= 2
RUFF := $(dir $(PYTHON))ruff
CLANG_FORMAT := $(dir $(PYTHON))clang-format
recursive_files = $(foreach path,$(wildcard $1/*),$(call recursive_files,$(path),$2) $(filter $2,$(path)))
CPP_FILES := $(call recursive_files,cpp/cg,%.cpp %.hpp %.cu %.cuh) $(call recursive_files,tests/cpp,%.cpp %.hpp) $(call recursive_files,examples/cpp,%.cpp %.hpp)
VERSION = $(shell $(PYTHON) -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')
CMAKE_ARGS ?=
PYPROF_SUITE ?= sparse-solvers
PYPROF_THREADS ?= 1
PYPROF_SCOPE ?= prepared
PYPROF_ROUNDS ?= 30
PYPROF_REPETITIONS ?= 5
PYPROF_OUTPUT ?= $(BUILD_DIR)/benchmarks/$(PYPROF_SUITE)-t$(PYPROF_THREADS)-$(PYPROF_SCOPE).json
PYPROF_ARGS ?=
DIST_DIR ?= $(CURDIR)/dist
DIST_BLAS ?= OFF
DIST_CUDA ?= OFF
DIST_ARGS ?=
DOCS_DIR ?= $(CURDIR)/build/docs
CUDA_ROOT ?= $(CURDIR)/.deps/cuda-12.8
CUDA_HOST_COMPILER ?=
CUDA_ARCHITECTURES ?=
cuda_definitions = $(1)CUDAToolkit_ROOT="$(CUDA_ROOT)" $(if $(CUDA_HOST_COMPILER),$(1)CMAKE_CUDA_HOST_COMPILER="$(CUDA_HOST_COMPILER)") $(if $(CUDA_ARCHITECTURES),$(1)CMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)")
CUDA_CONFIGURE_ARGS = $(if $(filter cuda-release,$(PRESET)),$(call cuda_definitions,-D))
CUDA_DIST_ARGS = $(if $(filter ON,$(DIST_CUDA)),-Ccmake.define.CMAKE_TOOLCHAIN_FILE=cmake/CGCudaToolchain.cmake $(call cuda_definitions,-Ccmake.define.))

.DEFAULT_GOAL := all
.PHONY: all configure build test pytest gtest lint format typecheck test-native wheel sdist dist test-dist docs doctest clean pyprof demo

all: build

configure:
	$(CMAKE) --preset $(PRESET) -B $(BUILD_DIR) -DPython_EXECUTABLE=$(PYTHON) -DCMAKE_MAKE_PROGRAM=$(dir $(PYTHON))ninja $(CUDA_CONFIGURE_ARGS) $(CMAKE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

test: build
	$(CMAKE) -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

pytest: build
	$(CMAKE) -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(CTEST) --test-dir $(BUILD_DIR) --output-on-failure -L python

gtest: configure
	$(CMAKE) --build $(BUILD_DIR) --target cg_native_tests --parallel $(JOBS)
	$(CMAKE) -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(CTEST) --test-dir $(BUILD_DIR) --output-on-failure -L cpp

lint:
	$(RUFF) check python tests/python tests/typing examples scripts docs/conf.py
	$(RUFF) format --check python tests/python tests/typing examples scripts docs/conf.py
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)

format:
	$(RUFF) check --fix python tests/python tests/typing examples scripts docs/conf.py
	$(RUFF) format python tests/python tests/typing examples scripts docs/conf.py
	$(CLANG_FORMAT) -i $(CPP_FILES)

typecheck: build
	$(PYTHON) scripts/check_typing.py --package-root $(BUILD_DIR)/python

test-native:
	$(MAKE) build BUILD_DIR="$(NATIVE_BUILD_DIR)" CMAKE_ARGS="$(CMAKE_ARGS) -DCG_BUILD_PYTHON=OFF -DBUILD_TESTING=OFF"
	$(PYTHON) scripts/check_native_consumer.py --build-dir "$(NATIVE_BUILD_DIR)" --cmake "$(CMAKE)" --cuda-root "$(CUDA_ROOT)" --jobs $(JOBS) $(NATIVE_ARGS)

wheel:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) $(PYTHON) -m build --wheel --outdir $(DIST_DIR) -Ccmake.define.CG_ENABLE_BLAS=$(DIST_BLAS) -Ccmake.define.CG_ENABLE_CUDA=$(DIST_CUDA) $(CUDA_DIST_ARGS) $(DIST_ARGS)

sdist:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip $(PYTHON) -m build --sdist --outdir $(DIST_DIR)

# The default build frontend builds an sdist, then builds its wheel in isolation.
dist:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) $(PYTHON) -m build --outdir $(DIST_DIR) -Ccmake.define.CG_ENABLE_BLAS=$(DIST_BLAS) -Ccmake.define.CG_ENABLE_CUDA=$(DIST_CUDA) $(CUDA_DIST_ARGS) $(DIST_ARGS)

test-dist: sdist
	$(PYTHON) scripts/check_distribution.py --sdist $(DIST_DIR)/cgpy-$(VERSION).tar.gz --blas $(DIST_BLAS) --cuda $(DIST_CUDA) --cuda-root "$(CUDA_ROOT)" $(if $(CUDA_HOST_COMPILER),--cuda-host-compiler "$(CUDA_HOST_COMPILER)") $(if $(CUDA_ARCHITECTURES),--cuda-architectures "$(CUDA_ARCHITECTURES)") --jobs $(JOBS)

docs: build
	PYTHONPATH=$(BUILD_DIR)/python $(PYTHON) -m sphinx -W --keep-going -b html docs $(DOCS_DIR)/html

doctest: build
	PYTHONPATH=$(BUILD_DIR)/python $(PYTHON) -m sphinx -W --keep-going -b doctest docs $(DOCS_DIR)/doctest

pyprof: build
	PYTHONPATH=$(BUILD_DIR)/python $(PYTHON) -m cgbench run \
		--suite $(PYPROF_SUITE) --threads $(PYPROF_THREADS) --scope $(PYPROF_SCOPE) \
		--rounds $(PYPROF_ROUNDS) --repetitions $(PYPROF_REPETITIONS) \
		--output $(PYPROF_OUTPUT) $(PYPROF_ARGS)

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)

demo: build
	PYTHONPATH=$(BUILD_DIR)/python $(PYTHON) examples/dense.py
	PYTHONPATH=$(BUILD_DIR)/python $(PYTHON) examples/csr.py
	PYTHONPATH=$(BUILD_DIR)/python $(PYTHON) examples/prepared.py
