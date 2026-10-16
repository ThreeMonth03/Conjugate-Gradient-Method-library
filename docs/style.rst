Coding conventions
==================

This project adopts selected conventions from the
`SOLVCON coding style <https://doc.solvcon.net/devguide/style.html>`_ and
`binding organization <https://doc.solvcon.net/devguide/binding.html>`_.
``make lint`` enforces formatter and Python lint settings. Clang-tidy checks
native type/function naming; architecture tests enforce feature dependencies.
Ownership contracts also require review.

Layout and dependencies
-----------------------

Keep declarations and implementations together under ``cpp/cg/<feature>/``.
Place bindings in that feature's ``pymod/``; ``cpp/cg/python/`` contains
extension entry points and shared Python conversion support. Python packages
follow functional responsibility. Create a directory for a coherent purpose.

Storage owns resources. Matrix types enforce representation invariants.
Backends execute operations using their own execution resources.
Preconditioners own prepared state. The solver coordinates iteration,
convergence and prepared operator snapshots. Bindings translate Python objects
and manage the language boundary. Core headers and targets do not depend on
Python or pybind11. Keep private helpers under ``detail/``; export an explicit
public header list. Use ``cpu.cpp`` and ``cuda.cpp`` for feature registrations.

Names and formatting
--------------------

* Use ``snake_case`` for files, functions and variables, and ``PascalCase``
  for classes and enum types. Treat abbreviations as words: ``CsrMatrix``,
  ``CgOptions`` and ``SparseLuExecutor``. Enum values use ``snake_case``.
* Use ``normalize_*`` when conversion may change representation,
  ``validate_*`` for checks, ``parse_*`` for configuration translation and
  ``as_*_view`` for borrowed views. Name returned quantities explicitly:
  ``norm`` and ``squared_norm`` have different units; ``apply_and_dot``
  describes a fused operation. Prefer enum choices to ambiguous booleans.
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

Return borrowed contiguous ranges as spans by value, with ``const`` element
types for read-only access. Reject borrowing from temporary owners and document
invalidation. An ``auto`` copy of a span still borrows; use an owning container
for an independent snapshot.
Owner getters that return borrowed pointers, views or references use ``&`` or
``const &`` qualifiers and delete their rvalue overloads. Descriptor accessors
remain usable on temporary views because descriptors do not own storage.

Carry borrowed ranges through host kernels as spans and index the span
directly. Keep strided matrix and device views lightweight, with private
metadata and read-only accessors. Views do not share ownership; smart pointers
belong in resource owners. Raw pointers remain appropriate for native API
boundaries and device kernels. Keep casts and allocation calls localized to
those boundaries. CUDA view accessors must work on host and device without
pulling CUDA SDK headers into public interfaces.

Use value types for options/results and composition for owned state. Compose
storage from existing RAII owners; use Pimpl for a concrete dependency or ABI
boundary. A prepared solver holds a matrix variant and optional Jacobi state;
it does not need a subclass for each storage/preconditioner combination. The
shared numerical engine accepts the operations it needs through templates.
Benchmark references use a small Python protocol so array implementations can be
substituted without depending on concrete classes. Add a virtual interface or
plugin registry only when an actual use case needs it.

Derive configuration from owned state when it already expresses the choice.
Use ``std::optional`` for a directly held optional value when separate
allocation is unnecessary. Call shared type interfaces directly; keep an
adapter only when it translates a real difference. Share pure size checks
without introducing a generic arithmetic framework.

Select fixed execution settings at resource construction and borrow them at
use sites. Share small numerical decisions as private free functions while
keeping backend operations and device scheduling explicit.

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
