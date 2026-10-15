Coding conventions
==================

This project adopts selected conventions from the
`SOLVCON coding style <https://doc.solvcon.net/devguide/style.html>`_ and
`binding organization <https://doc.solvcon.net/devguide/binding.html>`_.
``make lint`` enforces formatter and Python lint settings; ownership and
dependency boundaries also require review.

Layout and dependencies
-----------------------

Keep declarations and implementations together under ``cpp/cg/<feature>/``.
Place bindings in that feature's ``pymod/``; ``cpp/cg/python/`` contains
extension entry points and shared Python conversion support. Python packages
follow functional responsibility. Create a directory for a coherent purpose.

Storage owns resources. Matrix types enforce representation invariants.
Backends execute operations. Preconditioners own prepared state. The solver
coordinates iteration and convergence. Bindings translate Python objects
and manage the language boundary. Core headers and targets do not depend on
Python or pybind11. Private binding headers are not public C++ interfaces.

Names and formatting
--------------------

* Use ``snake_case`` for files, functions and variables, and ``PascalCase``
  for classes. Treat abbreviations as words in new type names.
* Prefix C++ class data members with ``m_``. Plain data structs use
  descriptive unprefixed fields.
* Mathematical symbols such as A, b, x, r, p, Ap and z are appropriate in
  numerical recurrences. Use descriptive names for ownership, counters and
  configuration.
* Use qualified Python module imports and explicit relative imports for
  peers. Explicit public re-exports in package ``__init__.py`` are allowed;
  internal code imports the defining module. Never use wildcard imports.
* Keep ``#pragma once`` first in headers and use path-based angle-bracket
  includes. Public Python APIs use NumPy-style docstrings.

``.clang-format`` defines C++/CUDA formatting, including Allman braces and a
120-column limit. Ruff settings in ``pyproject.toml`` define Python formatting
with a 79-column limit. ``.editorconfig`` records encoding, line endings and
editor indentation, including Make tabs and two-space YAML indentation.

Design and verification
-----------------------

Use C++20 standard containers and RAII. CPU arrays use ``std::vector`` for
ownership and ``std::span`` for borrowing. Validate foreign pointers before
constructing a span. Device storage uses a standard owner with CUDA-aware
cleanup; host containers cannot own device allocations.

Preserve const correctness and state lifetime, aliasing and concurrency
requirements at public boundaries. Types with coupled representation
invariants must remain valid when copying or allocation fails. Prepare a
complete replacement before committing ownership changes.

Keep numerical policy separate from kernel selection. Preserve stopping
rules, failure statuses and true-residual verification when optimizing.
Tests should expose contract violations or numerical errors. Performance
comparisons require the same inputs, work, compiler and measurement policy;
see :doc:`benchmarks/methodology`.

Explain non-obvious invariants, numerical choices, lifetime constraints and
tradeoffs in comments. Split functions and introduce abstractions when doing
so clarifies responsibility; there is no arbitrary function-length limit.
