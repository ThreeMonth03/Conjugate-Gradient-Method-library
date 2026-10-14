PYTHON ?= $(CURDIR)/.venv/bin/python
CMAKE ?= $(CURDIR)/.venv/bin/cmake
PRESET ?= cpu-release
BUILD_DIR ?= $(CURDIR)/build/$(PRESET)
JOBS ?= 2
RUFF := $(CURDIR)/.venv/bin/ruff
CLANG_FORMAT := $(CURDIR)/.venv/bin/clang-format
CPP_FILES := $(wildcard cpp/*/*.cpp cpp/*/*.hpp tests/cpp/*.cpp)
CMAKE_ARGS ?=

.DEFAULT_GOAL := all
.PHONY: all configure build test pytest gtest lint format wheel clean

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
	$(CURDIR)/.venv/bin/ctest --test-dir $(BUILD_DIR) --output-on-failure -L cpp

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_FILES)

format:
	$(RUFF) check --fix .
	$(RUFF) format .
	$(CLANG_FORMAT) -i $(CPP_FILES)

wheel:
	PIP_CACHE_DIR=$(CURDIR)/.cache/pip $(PYTHON) -m build --wheel --no-isolation

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
