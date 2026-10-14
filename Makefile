PYTHON ?= $(CURDIR)/.venv/bin/python
CMAKE ?= $(CURDIR)/.venv/bin/cmake
PRESET ?= cpu-release
BUILD_DIR ?= $(CURDIR)/build/$(PRESET)
JOBS ?= 2
RUFF := $(CURDIR)/.venv/bin/ruff
CLANG_FORMAT := $(CURDIR)/.venv/bin/clang-format
CPP_FILES := $(wildcard cpp/*/*.cpp cpp/*/*.hpp include/cg/*.hpp src/*.cpp bindings/*.cpp tests/cpp/*.cpp)
CMAKE_ARGS ?=
PYPROF_SUITE ?= sparse-solvers
PYPROF_THREADS ?= 1
PYPROF_SCOPE ?= prepared
PYPROF_ROUNDS ?= 30
PYPROF_REPETITIONS ?= 5
PYPROF_OUTPUT ?= $(BUILD_DIR)/benchmarks/$(PYPROF_SUITE)-t$(PYPROF_THREADS)-$(PYPROF_SCOPE).json
PYPROF_ARGS ?=

.DEFAULT_GOAL := all
.PHONY: all configure build test pytest gtest lint format wheel clean pyprof demo

all: build

configure:
	$(CMAKE) --preset $(PRESET) -B $(BUILD_DIR) -DPython_EXECUTABLE=$(PYTHON) $(CMAKE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

test: build
	$(CMAKE) -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(CURDIR)/.venv/bin/ctest --test-dir $(BUILD_DIR) --output-on-failure

pytest: build
	$(CMAKE) -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(CURDIR)/.venv/bin/ctest --test-dir $(BUILD_DIR) --output-on-failure -L python

gtest: configure
	$(CMAKE) --build $(BUILD_DIR) --target cg_gtests --parallel $(JOBS)
	$(CMAKE) -E env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(CURDIR)/.venv/bin/ctest --test-dir $(BUILD_DIR) --output-on-failure -L cpp

lint:
	$(RUFF) check python tests demo
	$(RUFF) format --check python tests demo
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)

format:
	$(RUFF) check --fix python tests demo
	$(RUFF) format python tests demo
	$(CLANG_FORMAT) -i $(CPP_FILES)

wheel:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip $(PYTHON) -m build --wheel --no-isolation

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
