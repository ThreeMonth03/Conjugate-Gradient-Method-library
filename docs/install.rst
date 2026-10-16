Installation
============

Python installations use CPython 3.12.x. The package metadata accepts ``>=3.12,<3.13`` and
CMake enforces that range for Python builds. The default package uses the
native CPU kernel and requires NumPy at runtime. CUDA is disabled in the
default build and adds no NVIDIA runtime dependency to a CPU wheel. See :doc:`cuda` for explicit
CUDA builds and installation with the pinned public-runtime extra.
Development and documentation commands below
keep Python tools, downloads and build products inside the checkout.

Install a built wheel
---------------------

After building or obtaining a compatible wheel, install it into a virtual
environment. A wheel already contains the native extension; importing it does
not require the source checkout or a compiler.

.. code-block:: bash

   python3.12 -m venv .venv
   .venv/bin/python -m pip install --cache-dir .cache/pip dist/cgpy-*.whl
   .venv/bin/python -c 'import cgpy; print(cgpy.build_info())'

Build and install from source
-----------------------------

Use an existing C++20 compiler. From the repository root:

.. code-block:: bash

   python3.12 -m venv .venv
   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/dev.lock
   .venv/bin/python -m pip install --cache-dir .cache/pip --no-build-isolation .

Replace the final ``.`` with ``-e .`` for an editable development installation.
Changes to Python source are then visible immediately; rebuild after changing
native code. Source builds use scikit-build-core and CMake. The locked local
toolchain supplies CMake, Ninja and pybind11 without a system installation.

Alternatively, ``.venv/bin/python -m pip wheel --cache-dir .cache/pip .`` uses
pip's isolated build environment. Pip can download the declared build
dependencies; use the explicit local toolchain above when reproducing this
checkout's pinned development environment.

Native C++ consumers
--------------------

Build and install the C++ library with CMake 3.24 or later and a C++20 compiler.
This path does not locate Python or pybind11:

.. code-block:: bash

   cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
       -DCG_BUILD_PYTHON=OFF -DBUILD_TESTING=OFF
   cmake --build build/native --parallel 2
   cmake --install build/native --prefix "$PWD/build/native-install" --component development
   cmake -S examples/cpp -B build/consumer \
       -DCMAKE_PREFIX_PATH="$PWD/build/native-install"
   cmake --build build/consumer --parallel 2
   ctest --test-dir build/consumer --output-on-failure

The independent ``examples/cpp`` project consumes only the installation:

.. code-block:: cmake

   find_package(cgpy CONFIG REQUIRED)
   add_executable(app main.cpp)
   target_link_libraries(app PRIVATE cg::core)

The imported target supplies C++20, public include directories and link
dependencies. The example solves dense and CSR systems with CG and Jacobi
PCG, reuses a prepared solver/workspace, and checks the solution and true
residual independently. Its checks remain active in Release builds.

Set ``CG_ENABLE_OPENMP=ON`` or ``CG_ENABLE_BLAS=ON`` when building the library;
consumers inherit the necessary runtime link dependencies. Library-only
OpenMP compiler flags and the BLAS build macro stay private. The BLAS
installation retains the private provider archive and license, without
requiring a system BLAS or the original build directory. Installations are
relocatable: set ``CMAKE_PREFIX_PATH`` to the new prefix after moving it.
CMake metadata belongs
to the ``development`` component and is not included in Python wheels.

A CUDA-enabled installation additionally provides an explicit component:

.. code-block:: cmake

   find_package(cgpy CONFIG REQUIRED COMPONENTS CUDA)
   target_link_libraries(app PRIVATE cg::cuda)

Requesting CUDA locates Toolkit >=12.8 and <13 for its link dependencies;
the consumer itself can remain a CXX-only project without CUDA SDK include
directories in its compile settings. A core-only request does
not search for CUDA, even when the installation also contains that backend.
Set ``CG_CONSUMER_CUDA=ON`` when configuring ``examples/cpp`` to build its CUDA
example. Compilation and linking need no GPU; CTest explicitly skips CUDA
numerics without a usable device. ``CG_CONSUMER_REQUIRE_CUDA=ON`` makes that
absence a test failure.

Optional OpenMP and dense BLAS
------------------------------

Native OpenMP and BLAS are build-time choices. Selecting a CMake preset for a
Make command does not change an already installed wheel. Inspect the installed
extension rather than assuming its configuration:

.. doctest::

   >>> isinstance(cgpy.openmp_enabled(), bool)
   True
   >>> isinstance(cgpy.blas_enabled(), bool)
   True
   >>> "blas" in cgpy.build_info()
   True

To build an editable OpenMP package, add
``--config-settings=cmake.define.CG_ENABLE_OPENMP=ON`` to the source install
command. To select the optional BLAS package explicitly:

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip \
       --config-settings=cmake.define.CG_ENABLE_BLAS=ON .

This also works with the source-distribution archive in place of ``.``.
The pinned OpenBLAS build and its local toolchain requirements are described
in :doc:`build`. The first BLAS interface accepts dense arrays with
``dense_kernel="blas"`` and ``num_threads=1``. It rejects CSR operators and
unavailable BLAS support explicitly. Native remains the default kernel.
