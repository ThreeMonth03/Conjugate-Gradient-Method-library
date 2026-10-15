# Coding conventions

This project adopts selected conventions from the [SOLVCON coding style](https://doc.solvcon.net/devguide/style.html)
and [binding organization](https://doc.solvcon.net/devguide/binding.html).
The rules below define this repository's contract. `make lint` enforces the
formatter and Python lint settings; ownership and dependency boundaries also
require review.

## Layout and dependencies

Keep declarations and implementations together under `cpp/cg/<feature>/`.
Place bindings in that feature's `pymod/`; `cpp/cg/python/` contains extension
entry points and shared Python conversion support. Python packages follow
functional responsibility. Create a directory when it represents a coherent
responsibility, rather than to meet a size or depth target.

Storage owns resources and defines views. Matrix types enforce representation
invariants. Backends execute operations. Preconditioners own prepared state.
The solver coordinates iteration and convergence. Bindings translate Python
objects and manage the language boundary. Core headers and targets must not
depend on Python or pybind11. Private binding headers are not installed as
public C++ interfaces.

## Names and formatting

- Use UTF-8, LF endings and four spaces. Make recipes require tabs; YAML uses
  two spaces. `.editorconfig` records these file conventions.
- Use `snake_case` for files, functions and variables; use `PascalCase` for
  classes. Treat abbreviations as words in new type names.
- Prefix C++ class data members with `m_`. Plain data structs use descriptive
  unprefixed fields. Keep public and private naming consistent.
- Use Allman braces and braces around every C++ control-flow body. Put
  `#pragma once` first in headers and use path-based angle-bracket includes.
  Aim for 120 columns in C++ and 79 in Python.
- Prefer qualified Python module imports and explicit relative imports for
  peers. Explicit public re-exports in package `__init__.py` are allowed;
  internal code imports the defining module. Never use wildcard imports.
- Mathematical symbols such as `A`, `b`, `x`, `r`, `p`, `Ap` and `z` are
  acceptable in equations and the numerical recurrence. Use descriptive names
  for application state, ownership, counters and configuration.

## Design and verification

Use standard containers and RAII. Distinguish owning objects from borrowed
views and preserve const correctness. State lifetime, aliasing and concurrency
requirements at public boundaries. Prefer the rule of zero for ordinary
storage; define move/cleanup behavior when managing device resources.

Keep numerical policy separate from kernel selection. Preserve the stopping
rule, failure statuses and true-residual verification when optimizing. Tests
should expose contract violations or numerical errors; avoid tests that merely
repeat implementation details. Performance claims require recorded comparable
measurements.

Explain non-obvious invariants, numerical choices, lifetime constraints and
tradeoffs in comments. Do not narrate statements that already read clearly.
Public Python APIs use NumPy-style docstrings, matching the Sphinx setup.
Split functions and introduce abstractions when doing so clarifies a real
responsibility; there is no arbitrary function-length limit or mandatory
framework.
