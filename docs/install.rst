Installation
============

Use Python 3.10 or newer. The default package uses the native CPU kernel and
requires NumPy at runtime. Nonlinear legacy helpers additionally use the
optional ``autograd`` dependency. Development and documentation commands below
keep Python tools, downloads and build products inside the checkout.

Install a built wheel
---------------------

After building or obtaining a compatible wheel, install it into a virtual
environment. A wheel already contains the native extension; importing it does
not require the source checkout or a compiler.

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install --cache-dir .cache/pip dist/cgpy-*.whl
   .venv/bin/python -c 'import cgpy; print(cgpy.build_info())'

To build this checkout with the optional legacy derivatives, run:

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip '.[legacy]'

For a local wheel, append ``[legacy]`` to the selected wheel's full path in
the pip install argument. These are local source/wheel installs; a published
package release is not required.

Build and install from source
-----------------------------

Use an existing C++17 compiler. From the repository root:

.. code-block:: bash

   python3 -m venv .venv
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
