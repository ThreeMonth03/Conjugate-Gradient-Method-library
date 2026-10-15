Benchmarks
==========

Run the comparison suite with:

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
   make pyprof PYPROF_SUITE=sparse-solvers

Benchmarks measure time to an independently verified solution. Choose
``prepared`` to reuse setup or ``api`` to include per-call allocation.
:doc:`benchmarks/methodology` defines their exact boundaries, fixtures,
correctness checks and sampling.

.. toctree::
   :maxdepth: 1

   benchmarks/current
   benchmarks/methodology
   benchmarks/cpu
   benchmarks/sparse
   benchmarks/cuda

The tables summarize measurements made on 2026-10-15 at the source revisions
named on each page. They are historical results, not fresh measurements of
the current checkout.

Version control keeps the scripts and summary tables. New raw samples,
input hashes and environment metadata go under ``build/``; CI publishes its
smoke-run JSON as workflow artifacts. Documentation builds never require
benchmark execution.
