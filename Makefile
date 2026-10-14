PYTHON ?= $(CURDIR)/.venv/bin/python
CMAKE ?= $(dir $(PYTHON))cmake
CTEST ?= $(dir $(CMAKE))ctest
PRESET ?= cpu-release
BUILD_DIR ?= $(CURDIR)/build/$(PRESET)
JOBS ?= 2
RUFF := $(dir $(PYTHON))ruff
CLANG_FORMAT := $(dir $(PYTHON))clang-format
CPP_FILES := $(wildcard cpp/*/*.cpp cpp/*/*.hpp include/cg/*.hpp include/cg/detail/*.hpp src/*.cpp src/*.cu bindings/*.cpp tests/cpp/*.cpp)
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
CUDA_HOST_COMPILER ?= /usr/bin/g++-13
CUDA_ARCHITECTURES ?= 75
CUDA_CONFIGURE_ARGS = $(if $(filter cuda-release,$(PRESET)),-DCUDAToolkit_ROOT=$(CUDA_ROOT) -DCMAKE_CUDA_COMPILER=$(CUDA_ROOT)/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) -DCMAKE_CXX_COMPILER=$(CUDA_HOST_COMPILER) -DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)",)
CUDA_DIST_ARGS = $(if $(filter ON,$(DIST_CUDA)),-Ccmake.define.CUDAToolkit_ROOT=$(CUDA_ROOT) -Ccmake.define.CMAKE_CUDA_COMPILER=$(CUDA_ROOT)/bin/nvcc -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) -Ccmake.define.CMAKE_CXX_COMPILER=$(CUDA_HOST_COMPILER) -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)",)

.DEFAULT_GOAL := all
.PHONY: all configure build test pytest gtest lint format wheel sdist dist test-dist docs doctest clean pyprof demo

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
	$(RUFF) check python tests demo scripts
	$(RUFF) format --check python tests demo scripts
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)

format:
	$(RUFF) check --fix python tests demo scripts
	$(RUFF) format python tests demo scripts
	$(CLANG_FORMAT) -i $(CPP_FILES)

wheel:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) $(PYTHON) -m build --wheel --outdir $(DIST_DIR) -Ccmake.define.CG_ENABLE_BLAS=$(DIST_BLAS) -Ccmake.define.CG_ENABLE_CUDA=$(DIST_CUDA) $(CUDA_DIST_ARGS) $(DIST_ARGS)

sdist:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip $(PYTHON) -m build --sdist --outdir $(DIST_DIR)

# The default build frontend builds an sdist, then builds its wheel in isolation.
dist:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip CMAKE_BUILD_PARALLEL_LEVEL=$(JOBS) $(PYTHON) -m build --outdir $(DIST_DIR) -Ccmake.define.CG_ENABLE_BLAS=$(DIST_BLAS) -Ccmake.define.CG_ENABLE_CUDA=$(DIST_CUDA) $(CUDA_DIST_ARGS) $(DIST_ARGS)

test-dist: sdist
	$(PYTHON) scripts/check_distribution.py --sdist $(DIST_DIR)/cgpy-0.4.0.tar.gz --blas $(DIST_BLAS) --cuda $(DIST_CUDA) --cuda-root $(CUDA_ROOT) --cuda-host-compiler $(CUDA_HOST_COMPILER) --cuda-architectures "$(CUDA_ARCHITECTURES)" --jobs $(JOBS)

docs: build
	PYTHONPATH=$(BUILD_DIR)/python:$(CURDIR)/python $(PYTHON) -m sphinx -W --keep-going -b html docs $(DOCS_DIR)/html

doctest: build
	PYTHONPATH=$(BUILD_DIR)/python:$(CURDIR)/python $(PYTHON) -m sphinx -W --keep-going -b doctest docs $(DOCS_DIR)/doctest

pyprof: build
	PYTHONPATH=$(BUILD_DIR)/python:$(CURDIR)/python $(PYTHON) -m cgbench run \
		--suite $(PYPROF_SUITE) --threads $(PYPROF_THREADS) --scope $(PYPROF_SCOPE) \
		--rounds $(PYPROF_ROUNDS) --repetitions $(PYPROF_REPETITIONS) \
		--output $(PYPROF_OUTPUT) $(PYPROF_ARGS)

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)

demo: build
	PYTHONPATH=$(BUILD_DIR)/python:$(CURDIR)/python $(PYTHON) demo/demo_matrix.py
	PYTHONPATH=$(BUILD_DIR)/python:$(CURDIR)/python $(PYTHON) demo/demo_cg_method.py
