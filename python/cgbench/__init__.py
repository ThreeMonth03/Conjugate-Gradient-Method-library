"""Reproducible CPU CG comparisons used by pytest and ``make pyprof``.

Keep this module free of numerical imports: the CLI configures a fresh worker's
thread environment before importing NumPy, PyTorch, or the native extension.
"""

__version__ = "1"
